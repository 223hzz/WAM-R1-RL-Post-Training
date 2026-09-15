# -*- coding: utf-8 -*-
"""
Step 1: Network Freezing and Decoupling Engine (Trainer Initialization)
=====================================================

First step of the Action-RL post-training framework: load the SFT-trained
DiT4DiT model as the RL training engine and perform the following decoupling:

1. Load the DiT4DiT model according to the checkpoint config
2. Fully freeze the front-end perception networks:
     - shared VAE encoder      (backbone_interface.extractor.vae)
     - Video DiT backbone      (backbone_interface.extractor.transformer)
     - Text Encoder            (backbone_interface.extractor.text_encoder)
   Only the Action DiT (action_model) stays trainable.
3. Recursively replace every nn.Dropout in the Action DiT with nn.Identity:
   Online RL exploration needs a deterministic forward pass apart from the SDE-injected
   noise; extra Dropout randomness would break the discrete SDE transition probabilities.
4. Create a reference Action DiT (frozen deep copy) for the KL divergence:
     D_KL(π_φ || π_ref) ∝ ||v_φ^action - v_ref^action||²
5. Create the optimizer over action_model parameters with requires_grad=True only.
6. Verify the freezing state and the causal-mask integrity.

References:
  - public GRPO implementations (reference-model creation and KL computation)
  - public GRPO implementations (KL constraint)
"""

from __future__ import annotations

import copy
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dit4dit_rl_trainer.lora import (
    DEFAULT_LORA_TARGETS,
    enable_output_head_training,
    inject_lora,
    merged_lora_state_dict,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ============================================================================

@dataclass
class RLEngineConfig:
    """Configuration for the RL post-training engine (Step 1 parameters).

    Attributes:
        pretrained_checkpoint: path to the full DiT4DiT checkpoint produced by SFT.
        learning_rate: RL fine-tuning LR for the Action DiT (usually below the SFT LR, e.g. 1e-5).
        weight_decay: AdamW weight decay.
        adam_beta1 / adam_beta2: AdamW momentum coefficients.
        beta_kl: KL divergence penalty coefficient beta.
        clip_range: PPO/GRPO policy-ratio clipping range epsilon.
        group_size: GRPO group size G (action trajectories sampled in parallel per prompt).
        device: primary training device.
        ref_model_offload: whether to CPU-offload the reference model (enable when VRAM is tight).
        dtype: training precision (bf16 recommended, matching the original DiT4DiT).
    """
    pretrained_checkpoint: str = ""
    model_variant: str = "auto"  # auto | original | multistream
    trainable_scope: str = "lora"  # action_head/full | output | lora
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # GRPO hyperparameters (used in Steps 3/4; declared here only)
    beta_kl: float = 0.04
    clip_range: float = 1e-3
    group_size: int = 8

    device: str = "cuda"
    ref_model_offload: bool = False
    dtype: str = "bfloat16"
    create_reference_model: bool = True


# ============================================================================
# ============================================================================

class RLEngine:
    """Action-RL post-training engine.

    Step 1 responsibility: network freezing and decoupling.

    Usage::

        cfg = RLEngineConfig(pretrained_checkpoint="/path/to/sft.pt")
        engine = RLEngine(cfg)
        # engine.action_model     -> trainable Action DiT
        # engine.ref_action_model -> frozen reference Action DiT
        # engine.optimizer        -> optimizes the Action DiT only
        # engine.model            -> the full DiT4DiT model (backbone frozen)
    """

    def __init__(self, cfg: RLEngineConfig):
        self.cfg = cfg
        self.loaded_action_checkpoint: Optional[str] = None
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, cfg.dtype, torch.bfloat16)

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Loading SFT-pretrained DiT4DiT model ═══")
        self.model = self._load_pretrained_model(cfg.pretrained_checkpoint)
        self.model_variant = self._detect_model_variant()
        if cfg.model_variant != "auto" and cfg.model_variant != self.model_variant:
            raise ValueError(
                f"Requested model_variant={cfg.model_variant!r}, but checkpoint "
                f"contains {self.model_variant!r}."
            )
        logger.info(f"  Model variant: {self.model_variant}")

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Freezing non-action modules ═══")
        freeze_stats = self._freeze_non_action_modules()

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Replacing Dropout → Identity in Action DiT ═══")
        dropout_count = self._replace_dropout_with_identity(self.action_model)
        logger.info(f"  Replaced {dropout_count} Dropout layers with Identity")

        # Restrict the update surface before cloning the frozen reference.
        self.adapter_module_names = self._configure_trainable_scope()

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        if cfg.create_reference_model:
            logger.info("[Step1] ═══ Creating reference Action DiT ═══")
        self.ref_action_model = (
            self._create_reference_model() if cfg.create_reference_model else None
        )
        if self.ref_action_model is None:
            logger.info("[Step1] Reference Action DiT disabled for non-GRPO training")

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Creating optimizer ═══")
        self.optimizer = self._create_optimizer()

        # ----------------------------------------------------------
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Verifying freezing & causal mask ═══")
        self._verify_freezing(freeze_stats, dropout_count)

        self._set_train_eval_mode()

        logger.info("[Step1] ═══ Engine initialization complete ═══")

    # ========================================================================
    # ========================================================================

    def _configure_trainable_scope(self) -> List[str]:
        scope = self.cfg.trainable_scope.lower()
        if scope in ("full", "action_head"):
            for parameter in self.action_model.parameters():
                parameter.requires_grad_(True)
            logger.info("  Trainable scope: full Action DiT (no LoRA)")
            return []
        if scope == "output":
            names = enable_output_head_training(self.action_model)
            logger.info("  Trainable scope: output modules (%s)", ", ".join(names))
            return names
        if scope == "lora":
            if self.cfg.lora_dropout != 0.0:
                raise ValueError(
                    "LoRA dropout must be 0 for on-policy SDE log-probability "
                    "recomputation"
                )
            names = inject_lora(
                self.action_model,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
                target_patterns=self.cfg.lora_targets,
            )
            logger.info(
                "  Trainable scope: LoRA rank=%d alpha=%g on %d linear layers",
                self.cfg.lora_rank,
                self.cfg.lora_alpha,
                len(names),
            )
            return names
        raise ValueError(
            f"Unknown trainable_scope={self.cfg.trainable_scope!r}; "
            "choose action_head/full, output, or lora"
        )

    @property
    def backbone_interface(self):
        """Multimodal backbone (MultiStreamInterface): frozen VAE + Video DiT."""
        return self.model.backbone_interface

    @property
    def action_model(self) -> nn.Module:
        """Trainable Action DiT (MaskedFlowmatchingActionHead)."""
        return self.model.action_model

    @property
    def vae(self):
        """Frozen shared VAE encoder."""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "vae", None)

    @property
    def video_dit(self):
        """Frozen Video DiT backbone (Cosmos Transformer)."""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "transformer", None)

    @property
    def text_encoder(self):
        """Frozen text encoder."""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "text_encoder", None)

    # ========================================================================
    # ========================================================================

    def _load_pretrained_model(self, checkpoint_path: str) -> nn.Module:
        """Load the DiT4DiT model from an SFT checkpoint.

        Uses baseframework.from_pretrained, which:
          1. reads config + norm_stats from the checkpoint
          2. calls build_framework(cfg) to build the model from framework.name
          3. loads the state_dict (strict=True)

        Args:
            checkpoint_path: path to the .pt file produced by the SFT stage.

        Returns:
            The DiT4DiT model instance (weights loaded, moved to self.device).
        """
        if not checkpoint_path:
            raise ValueError(
                "[Step1] pretrained_checkpoint is required. "
                "Please provide the path to the SFT-trained checkpoint."
            )

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[Step1] Checkpoint not found: {ckpt_path}")

        # Register the optional multistream framework before checkpoint config
        # dispatch. Importing the original framework is handled by DiT4DiT.
        try:
            import dit4dit_multistream.DiT4DiT_Multistream  # noqa: F401
        except ImportError:
            logger.debug("Multistream framework is unavailable; loading original only")

        from DiT4DiT.model.framework.base_framework import baseframework
        model = baseframework.from_pretrained(str(ckpt_path))

        model = model.to(self.device)
        logger.info(f"  Model loaded from {ckpt_path}")
        logger.info(f"  Device: {self.device}")

        return model

    def _detect_model_variant(self) -> str:
        builder = getattr(self.model.backbone_interface, "build_multistream_inputs", None)
        return "multistream" if callable(builder) else "original"

    def _freeze_non_action_modules(self) -> Dict[str, int]:
        """Freeze every module except action_model.

        Freezing policy:
          - backbone_interface (Video DiT + VAE + text_encoder) -> frozen
          - action_model -> stays trainable
          - other top-level submodules -> frozen

        Returns:
            Dict[str, int]: parameter count per frozen module.
        """
        stats: Dict[str, int] = {}

        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            n = self._freeze_module(self.model.backbone_interface, "backbone_interface")
            stats["backbone_interface"] = n

        for name, module in self.model.named_children():
            if name in ("backbone_interface", "action_model"):
                continue
            n = self._freeze_module(module, name)
            if n > 0:
                stats[name] = n

        total_frozen = sum(stats.values())
        logger.info(f"  Total frozen parameters: {total_frozen:,}")
        return stats

    @staticmethod
    def _freeze_module(module: nn.Module, name: str = "") -> int:
        """Freeze a single module: requires_grad=False + eval().

        Args:
            module: the nn.Module to freeze.
            name: module name, used for logging.

        Returns:
            int: total number of frozen parameters.
        """
        if module is None:
            return 0
        n_params = 0
        for p in module.parameters():
            n_params += p.numel()
            p.requires_grad_(False)
        module.eval()
        if name and n_params > 0:
            logger.info(f"  [Frozen] {name}: {n_params:,} params")
        return n_params

    def _replace_dropout_with_identity(self, module: nn.Module) -> int:
        """Recursively replace every nn.Dropout in the Action DiT with nn.Identity.

        Rationale (Flow-SDE):
          During online RL exploration the SDE already injects controlled randomness
          via the Gaussian noise eps_tau; extra Dropout inside the Action DiT would
          add an untrackable randomness source and break the discrete Gaussian
          transition probabilities (eps_k must stay the only randomness source).

        Replacement policy:
          Walk all submodules and map nn.Dropout -> nn.Identity.
          nn.Identity is a no-op that preserves tensor shapes, so dimensions stay compatible.

        Args:
            module: root module to walk recursively (usually action_model).

        Returns:
            int: number of Dropout layers replaced.
        """
        count = 0
        # Collect all (parent, attr_name) pairs first: mutating the submodule dict
        # while iterating over it raises a RuntimeError.
        replacements: List[Tuple[nn.Module, str]] = []

        for name, child in module.named_children():
            if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                replacements.append((module, name))
            elif isinstance(child, nn.Module):
                count += self._replace_dropout_with_identity(child)

        for parent, name in replacements:
            old_dropout = getattr(parent, name)
            dropout_p = getattr(old_dropout, "p", 0.0)
            setattr(parent, name, nn.Identity())
            logger.debug(f"    Dropout(p={dropout_p}) → Identity at {_full_name(parent, name)}")
            count += 1

        return count

    def _create_reference_model(self) -> nn.Module:
        """Create a reference Action DiT (frozen deep copy) for the KL divergence.

        KL divergence::

            D_KL(π_φ ‖ π_ref) = (Δτ / (2 σ_τ (1-τ)^{2τ+1} / σ_τ²))
                                 × ‖v_φ^action - v_ref^action‖² / 2

        Equivalent implementation::

            kl_loss = ((prev_sample_mean - prev_sample_mean_ref)²).mean() / (2 * std_dev_t²)

        Here prev_sample_mean_ref is computed from the reference model's velocity field,
        which is why a frozen copy of the original SFT Action DiT is needed.

        Returns:
            nn.Module: the frozen reference Action DiT.
        """
        # Deep-copy the Action DiT (action_model already has Dropout replaced)
        ref = copy.deepcopy(self.action_model)

        self._freeze_module(ref, "ref_action_model")

        # Apply the same Dropout replacement so the architecture matches the live model
        self._replace_dropout_with_identity(ref)

        if self.cfg.ref_model_offload:
            # CPU-offload mode: the ref model stays on CPU and is moved to GPU only when needed
            ref = ref.cpu()
            logger.info("  Reference model offloaded to CPU (save VRAM)")
        else:
            ref = ref.to(self.device)

        n_ref = sum(p.numel() for p in ref.parameters())
        logger.info(f"  Reference Action DiT: {n_ref:,} params (frozen)")
        return ref

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer over action_model parameters with requires_grad=True only.

        Returns:
            torch.optim.AdamW: optimizer covering Action DiT parameters only.
        """
        trainable_param_groups: List[Dict[str, Any]] = []
        n_trainable = 0

        for name, p in self.action_model.named_parameters():
            if p.requires_grad:
                trainable_param_groups.append({"params": [p], "name": name})
                n_trainable += p.numel()

        if n_trainable == 0:
            raise RuntimeError(
                "[Step1] Action DiT has no trainable parameters! "
                "Check if _freeze_non_action_modules accidentally froze action_model."
            )

        logger.info(f"  [Trainable] action_model: {n_trainable:,} params "
                    f"({len(trainable_param_groups)} param groups)")

        optimizer = torch.optim.AdamW(
            [p for group in trainable_param_groups for p in group["params"]],
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.weight_decay,
        )
        return optimizer

    # ========================================================================
    # ========================================================================

    def _verify_freezing(self, freeze_stats: Dict[str, int], dropout_count: int):
        """Verify the freezing state and that freezing is complete.

        Checks:
          #1 causal-mask leakage -> _verify_causal_mask()
          #2 missed freezing in the VAE / front end -> count trainable backbone_interface params
          #3 (optical-flow / heatmap live updates -> checked in Step 4)
        """
        n_action_trainable = sum(
            p.numel() for p in self.action_model.parameters() if p.requires_grad
        )
        assert n_action_trainable > 0, \
            "[Step1] FAIL: Action DiT has no trainable parameters!"

        n_backbone_trainable = 0
        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            n_backbone_trainable = sum(
                p.numel() for p in self.backbone_interface.parameters() if p.requires_grad
            )
        assert n_backbone_trainable == 0, \
            f"[Step1] FAIL: backbone_interface still has {n_backbone_trainable:,} trainable params! " \
            f"(VAE/Video DiT must be fully frozen)"

        n_vae_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad) \
            if self.vae is not None else 0
        assert n_vae_trainable == 0, \
            f"[Step1] FAIL: VAE still has {n_vae_trainable:,} trainable params!"

        # Check 4: the causal mask is only required for the multistream variant; the
        # original variant uses standard cross-attention.
        if self.model_variant == "multistream":
            self._verify_causal_mask()
        else:
            logger.info("  [Causal Mask] Original DiT4DiT: not applicable")

        logger.info(f"  [Verify] Frozen modules: {list(freeze_stats.keys())}")
        logger.info(f"  [Verify] Dropout replaced: {dropout_count}")
        logger.info(f"  [Verify] Trainable (Action DiT): {n_action_trainable:,} params")
        logger.info(f"  [Verify] backbone trainable: {n_backbone_trainable} (must be 0)")
        logger.info(f"  [Verify] VAE trainable: {n_vae_trainable} (must be 0)")

    def _verify_causal_mask(self):
        """Verify causal-mask integrity.

        Confirms the causal mask M in the Action DiT cross-attention:
          - first N positions (semantic Z_x) = -1e9 or -inf (fully masked)
          - last 2N positions (physical Z_f + intent Z_v) = 0 (fully open)

        How it is checked:
          1. if the mask is a statically precomputed Tensor, inspect the values directly
          2. if the mask is built dynamically at forward time, inspect the parameters of that logic
        """
        action_model = self.action_model

        # Look for the causal mask in several possible locations
        mask_candidates = []

        for attr in ("causal_mask", "mask", "causal_key_mask", "_causal_mask", "intent_physical_mask"):
            if hasattr(action_model, attr):
                m = getattr(action_model, attr)
                if isinstance(m, torch.Tensor):
                    mask_candidates.append((f"action_model.{attr}", m))

        dit = getattr(action_model, "dit", None) or getattr(action_model, "model", None)
        if dit is not None:
            for attr in ("causal_mask", "mask", "causal_key_mask"):
                if hasattr(dit, attr):
                    m = getattr(dit, attr)
                    if isinstance(m, torch.Tensor):
                        mask_candidates.append((f"action_model.dit.{attr}", m))

        if dit is not None and hasattr(dit, "transformer_blocks"):
            for i, block in enumerate(dit.transformer_blocks):
                if hasattr(block, "masked_cross_attn"):
                    mca = block.masked_cross_attn
                    for attr in ("causal_mask", "mask", "causal_key_mask"):
                        if hasattr(mca, attr):
                            m = getattr(mca, attr)
                            if isinstance(m, torch.Tensor):
                                mask_candidates.append((f"block[{i}].masked_cross_attn.{attr}", m))

        if mask_candidates:
            for name, mask in mask_candidates:
                self._check_mask_values(name, mask)
        else:
            # Mask is built dynamically at forward time from N_per_stream
            n_per_stream = getattr(action_model, "N_per_stream", None) or \
                           getattr(dit, "N_per_stream", None) if dit is not None else None
            if n_per_stream is not None:
                logger.info(f"  [Causal Mask] Dynamic mode: N_per_stream={n_per_stream}, "
                            f"mask built at forward time (first {n_per_stream} cols=-1e9, "
                            f"last {2*n_per_stream} cols=0)")
            else:
                logger.info("  [Causal Mask] Dynamic mode: mask built at forward time "
                            "(will verify during rollout in Step 2)")

    @staticmethod
    def _check_mask_values(name: str, mask: torch.Tensor):
        """Inspect the concrete values of the causal mask."""
        # Mask shape is usually (L_action, 3N), (B, L_action, 3N), or carries attention heads
        if mask.dim() < 2:
            logger.info(f"  [Causal Mask] {name}: dim={mask.dim()}, skipping value check")
            return

        last_dim = mask.shape[-1]
        # The first N positions should be a large negative number (-1e9 or -inf)
        front = mask[..., : last_dim // 3]
        # The last 2N positions should be 0
        back = mask[..., last_dim // 3:]

        front_max = front.abs().max().item() if front.numel() > 0 else 0
        back_max = back.abs().max().item() if back.numel() > 0 else 0

        if front_max > 1e3 and back_max < 1e-6:
            logger.info(f"  [Causal Mask] {name}: OK ✓ "
                        f"(front |max|={front_max:.0f} → masked, "
                        f"back |max|={back_max:.6f} → open)")
        else:
            logger.warning(f"  [Causal Mask] {name}: ⚠ CHECK NEEDED "
                           f"(front |max|={front_max:.6f}, back |max|={back_max:.6f})")

    def _set_train_eval_mode(self):
        """Set train/eval mode for each module.

        - backbone_interface (VAE + Video DiT) -> eval() (frozen)
        - ref_action_model -> eval() (frozen)
        - action_model -> train() (trainable)
        """
        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            self.backbone_interface.eval()
        if self.ref_action_model is not None:
            self.ref_action_model.eval()

        self.action_model.train()

    # ========================================================================
    # Utility methods used by Steps 2/3/4
    # ========================================================================

    def compute_ref_velocity(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the action velocity field v_ref with the reference model, for the KL divergence.

        Runs under no_grad and builds no computation graph.
        With ref_model_offload=True the ref model is moved to GPU for the call and moved back to CPU after.

        Returns:
            torch.Tensor: the reference model's velocity-field output.
        """
        if self.ref_action_model is None:
            raise RuntimeError("Reference model was disabled for this engine")
        with torch.no_grad():
            if self.cfg.ref_model_offload:
                self.ref_action_model.to(self.device)
            v_ref = self.ref_action_model(*args, **kwargs)
            if self.cfg.ref_model_offload:
                self.ref_action_model.cpu()
                torch.cuda.empty_cache()
        return v_ref

    def compute_ref_velocities(
        self,
        rollout_result,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        N_per_stream: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Evaluate the frozen reference policy on recorded SDE states."""
        from dit4dit_rl_trainer.sde_rollout import compute_action_velocity

        if self.ref_action_model is None:
            raise RuntimeError("Reference model was disabled for this engine")
        velocities: List[torch.Tensor] = []
        with torch.no_grad():
            if self.cfg.ref_model_offload:
                self.ref_action_model.to(self.device)
            for record in rollout_result.sde_records:
                velocities.append(
                    compute_action_velocity(
                        action_model=self.ref_action_model,
                        a_tau=record.a_tau,
                        tau=record.timestep,
                        vl_embs=vl_embs,
                        state=state,
                        N_per_stream=N_per_stream,
                    ).detach()
                )
            if self.cfg.ref_model_offload:
                self.ref_action_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return velocities

    def get_trainable_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """Yield all trainable parameters as (name, tensor) pairs."""
        return [
            (name, p) for name, p in self.action_model.named_parameters()
            if p.requires_grad
        ]

    def save_checkpoint(
        self, path: str, trainer_state: Optional[Dict[str, Any]] = None
    ):
        """Save the RL training state of the Action DiT.

        Only action_model and optimizer state are saved; the backbone is frozen and needs no saving.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "action_model": self.action_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "engine_cfg": self.cfg.__dict__,
        }
        if trainer_state is not None:
            ckpt["trainer_state"] = trainer_state
        torch.save(ckpt, path)
        logger.info(f"[Step1] Checkpoint saved to {path}")

    def export_pretrained_checkpoint(self, path: str):
        """Export a full checkpoint loadable by ``baseframework.from_pretrained``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.trainable_scope.lower() == "lora":
            state_dict = merged_lora_state_dict(self.model, self.action_model)
            logger.info("[Step1] Materialized merged LoRA weights for serving export")
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)

        checkpoint = Path(self.cfg.pretrained_checkpoint).resolve()
        source_run = checkpoint.parent
        if not all(
            (source_run / name).exists()
            for name in ("config.yaml", "dataset_statistics.json")
        ):
            source_run = checkpoint.parents[1]
        target_run = path.resolve().parents[1]
        for metadata_name in ("config.yaml", "dataset_statistics.json"):
            source = source_run / metadata_name
            if not source.exists():
                raise FileNotFoundError(
                    f"Cannot export serving checkpoint: missing {source}"
                )
            target = target_run / metadata_name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        logger.info(f"[Step1] Full pretrained checkpoint exported to {path}")

    def load_action_checkpoint(
        self, path: str, load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """Load the RL training state of the Action DiT."""
        ckpt = torch.load(path, map_location="cpu")
        self.action_model.load_state_dict(ckpt["action_model"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"[Step1] Checkpoint loaded from {path}")
        self.loaded_action_checkpoint = str(path)
        return dict(ckpt.get("trainer_state", {}))

    def sync_ref_to_current(self):
        """Update the reference model to the current action_model state.

        Used by GRPO variants that re-sync the reference model periodically instead of freezing it permanently.
        """
        if self.ref_action_model is None:
            raise RuntimeError("Reference model was disabled for this engine")
        ref_sd = {k: v.detach().clone() for k, v in self.action_model.state_dict().items()}
        self.ref_action_model.load_state_dict(ref_sd)
        self._freeze_module(self.ref_action_model, "ref_action_model")
        logger.info("[Step1] Reference model synced to current Action DiT")


# ============================================================================
# ============================================================================

def _full_name(parent: nn.Module, attr: str) -> str:
    """Return a readable module name, for logging."""
    parent_name = parent.__class__.__name__
    return f"{parent_name}.{attr}"
