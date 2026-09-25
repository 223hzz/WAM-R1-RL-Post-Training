"""Small, dependency-free LoRA utilities for the Action DiT."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    r"(?:^|\.)attn1\.to_q$",
    r"(?:^|\.)attn1\.to_k$",
    r"(?:^|\.)attn1\.to_v$",
    r"(?:^|\.)attn1\.to_out\.0$",
)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return the original linear layer with the adapter merged into it."""
        delta = torch.matmul(self.lora_B.float(), self.lora_A.float()) * self.scaling
        self.base.weight.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


def _resolve_parent(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit():
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def inject_lora(
    root: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_patterns: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> List[str]:
    """Freeze ``root`` and attach LoRA adapters to matching linear layers."""
    compiled = [re.compile(pattern) for pattern in target_patterns]
    matches = [
        name
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear)
        and any(pattern.search(name) for pattern in compiled)
    ]
    if not matches:
        raise RuntimeError(
            "No Action DiT linear layers matched the LoRA target patterns: "
            + ", ".join(target_patterns)
        )

    for parameter in root.parameters():
        parameter.requires_grad_(False)
    for name in matches:
        parent, child_name = _resolve_parent(root, name)
        base = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        _set_child(parent, child_name, LoRALinear(base, rank, alpha, dropout))
    return matches


def merge_lora(root: nn.Module) -> List[str]:
    """Merge every LoRA adapter in ``root`` and restore plain linear modules."""
    matches = [name for name, module in root.named_modules() if isinstance(module, LoRALinear)]
    for name in matches:
        parent, child_name = _resolve_parent(root, name)
        adapter = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        _set_child(parent, child_name, adapter.merged_linear())
    return matches


@torch.no_grad()
def merged_lora_state_dict(
    full_model: nn.Module, adapted_root: nn.Module
) -> Mapping[str, torch.Tensor]:
    """Build a serving-compatible state dict without mutating the live model."""
    root_names = [
        name for name, module in full_model.named_modules() if module is adapted_root
    ]
    if len(root_names) != 1:
        raise RuntimeError(
            f"Expected one adapted root inside the full model, found {root_names}"
        )
    root_prefix = root_names[0]
    state = OrderedDict(full_model.state_dict())
    for name, adapter in adapted_root.named_modules():
        if not isinstance(adapter, LoRALinear):
            continue
        prefix = ".".join(part for part in (root_prefix, name) if part)
        delta = torch.matmul(adapter.lora_B.float(), adapter.lora_A.float())
        delta = (delta * adapter.scaling).to(
            device=adapter.base.weight.device, dtype=adapter.base.weight.dtype
        )
        state[f"{prefix}.weight"] = adapter.base.weight.detach() + delta
        if adapter.base.bias is not None:
            state[f"{prefix}.bias"] = adapter.base.bias.detach()
        for suffix in ("base.weight", "base.bias", "lora_A", "lora_B"):
            state.pop(f"{prefix}.{suffix}", None)
    return state


def enable_output_head_training(action_model: nn.Module) -> List[str]:
    """Train only the velocity output projections and action decoder."""
    for parameter in action_model.parameters():
        parameter.requires_grad_(False)

    enabled: List[str] = []
    for name, module in action_model.named_modules():
        if name in {"model.proj_out_1", "model.proj_out_2", "action_decoder"}:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            enabled.append(name)
    if not enabled:
        raise RuntimeError("Could not find Action DiT output modules to train")
    return enabled
