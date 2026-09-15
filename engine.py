# -*- coding: utf-8 -*-
"""
Step 1: 网络冻结与解耦引擎 (Trainer Initialization)
=====================================================

Action-RL 后训练框架的第一步：将 SFT 后的原版或 Multistream DiT4DiT
加载为 RL 训练引擎，完成以下解耦操作：

1. 根据 checkpoint config 加载原版或 Multistream DiT4DiT
2. 彻底冻结前端感知网络：
     - 共享 VAE Encoder        (backbone_interface.extractor.vae)
     - Video DiT 主干          (backbone_interface.extractor.transformer)
     - Text Encoder            (backbone_interface.extractor.text_encoder)
   仅保留 Action DiT (action_model) 为唯一可训练模块。
3. 递归将 Action DiT 中所有 nn.Dropout 替换为 nn.Identity：
     RL 在线探索需要确定性前向（除 SDE 注入噪声外），Dropout 的额外随机性
     会破坏离散 SDE 转移概率的一致性。
4. 创建 reference Action DiT（冻结深拷贝），用于 KL 散度计算：
     D_KL(π_φ || π_ref) ∝ ||v_φ^action - v_ref^action||²
5. 创建优化器，仅收集 action_model 中 requires_grad=True 的参数。
6. 验证冻结状态与因果掩码完整性。

参考：
  - 公开 GRPO 实现 (ref model 创建与 KL 计算)
  - 公开 GRPO 实现 (KL 约束)
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
# 引擎配置
# ============================================================================

@dataclass
class RLEngineConfig:
    """RL 后训练引擎配置（Step 1 相关参数）。

    Attributes:
        pretrained_checkpoint: SFT 阶段产出的完整 DiT4DiT checkpoint 路径。
        learning_rate: Action DiT 的 RL 微调学习率（通常比 SFT 小，如 1e-5）。
        weight_decay: AdamW 权重衰减。
        adam_beta1 / adam_beta2: AdamW 动量参数。
        beta_kl: KL 散度惩罚系数 β（公式中的 β）。
        clip_range: PPO/GRPO 策略比值裁剪范围 ε。
        group_size: GRPO 组大小 G（同一 prompt 并行采样的动作轨迹数）。
        device: 主训练设备。
        ref_model_offload: reference model 是否 CPU offload（显存不足时开启）。
        dtype: 训练精度（bf16 推荐，与 DiT4DiT 原版一致）。
    """
    # 模型加载
    pretrained_checkpoint: str = ""
    model_variant: str = "auto"  # auto | original | multistream
    trainable_scope: str = "lora"  # action_head/full | output | lora
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: Tuple[str, ...] = DEFAULT_LORA_TARGETS

    # 优化器
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # GRPO 超参（Step 3/4 使用，此处仅声明）
    beta_kl: float = 0.04
    clip_range: float = 1e-3
    group_size: int = 8

    # 设备与显存
    device: str = "cuda"
    ref_model_offload: bool = False
    dtype: str = "bfloat16"
    create_reference_model: bool = True


# ============================================================================
# 核心引擎
# ============================================================================

class RLEngine:
    """Action-RL 后训练引擎。

    Step 1 职责：网络冻结与解耦。

    使用方式::

        cfg = RLEngineConfig(pretrained_checkpoint="/path/to/sft.pt")
        engine = RLEngine(cfg)
        # engine.action_model  → 可训练的 Action DiT
        # engine.ref_action_model → 冻结的 reference Action DiT
        # engine.optimizer     → 仅优化 Action DiT
        # engine.model         → 完整的原版或 Multistream DiT4DiT (backbone 冻结)
    """

    def __init__(self, cfg: RLEngineConfig):
        self.cfg = cfg
        self.loaded_action_checkpoint: Optional[str] = None
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, cfg.dtype, torch.bfloat16)

        # ----------------------------------------------------------
        # 1. 加载 SFT 预训练模型
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
        # 2. 冻结除 Action DiT 外的所有模块
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Freezing non-action modules ═══")
        freeze_stats = self._freeze_non_action_modules()

        # ----------------------------------------------------------
        # 3. 递归替换 Action DiT 中的 Dropout → Identity
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Replacing Dropout → Identity in Action DiT ═══")
        dropout_count = self._replace_dropout_with_identity(self.action_model)
        logger.info(f"  Replaced {dropout_count} Dropout layers with Identity")

        # Restrict the update surface before cloning the frozen reference.
        self.adapter_module_names = self._configure_trainable_scope()

        # ----------------------------------------------------------
        # 4. 创建 reference Action DiT (用于 KL 散度计算)
        # ----------------------------------------------------------
        if cfg.create_reference_model:
            logger.info("[Step1] ═══ Creating reference Action DiT ═══")
        self.ref_action_model = (
            self._create_reference_model() if cfg.create_reference_model else None
        )
        if self.ref_action_model is None:
            logger.info("[Step1] Reference Action DiT disabled for non-GRPO training")

        # ----------------------------------------------------------
        # 5. 创建优化器 (仅优化 Action DiT)
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Creating optimizer ═══")
        self.optimizer = self._create_optimizer()

        # ----------------------------------------------------------
        # 6. 验证冻结状态 & 因果掩码完整性
        # ----------------------------------------------------------
        logger.info("[Step1] ═══ Verifying freezing & causal mask ═══")
        self._verify_freezing(freeze_stats, dropout_count)

        # 设置训练/评估模式
        self._set_train_eval_mode()

        logger.info("[Step1] ═══ Engine initialization complete ═══")

    # ========================================================================
    # 属性访问 —— 快捷访问模型内部组件
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
        """多模态主干 (MultiStreamInterface)：含冻结的 VAE + Video DiT。"""
        return self.model.backbone_interface

    @property
    def action_model(self) -> nn.Module:
        """可训练的 Action DiT (MaskedFlowmatchingActionHead)。"""
        return self.model.action_model

    @property
    def vae(self):
        """冻结的共享 VAE 编码器。"""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "vae", None)

    @property
    def video_dit(self):
        """冻结的 Video DiT 主干 (Cosmos Transformer)。"""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "transformer", None)

    @property
    def text_encoder(self):
        """冻结的文本编码器。"""
        extractor = getattr(self.backbone_interface, "extractor", None)
        return getattr(extractor, "text_encoder", None)

    # ========================================================================
    # 核心方法
    # ========================================================================

    def _load_pretrained_model(self, checkpoint_path: str) -> nn.Module:
        """从 SFT checkpoint 加载原版或 Multistream DiT4DiT 模型。

        利用 baseframework.from_pretrained，该方法会：
          1. 从 checkpoint 读取 config + norm_stats
          2. 调用 build_framework(cfg) 根据 framework.name 构建模型
          3. 加载 state_dict (strict=True)

        Args:
            checkpoint_path: SFT 阶段产出的 .pt 文件路径。

        Returns:
            DiT4DiT 模型实例（已加载权重，放置于 self.device）。
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

        # 通过 baseframework.from_pretrained 加载（自动根据 config 选择框架）
        from DiT4DiT.model.framework.base_framework import baseframework
        model = baseframework.from_pretrained(str(ckpt_path))

        # 设备放置
        model = model.to(self.device)
        logger.info(f"  Model loaded from {ckpt_path}")
        logger.info(f"  Device: {self.device}")

        return model

    def _detect_model_variant(self) -> str:
        builder = getattr(self.model.backbone_interface, "build_multistream_inputs", None)
        return "multistream" if callable(builder) else "original"

    def _freeze_non_action_modules(self) -> Dict[str, int]:
        """冻结除 action_model 外的所有模块。

        冻结策略：
          - backbone_interface (Video DiT + VAE + text_encoder) → 冻结
          - action_model → 保持可训练
          - 其他顶层子模块 → 冻结

        Returns:
            Dict[str, int]: 各冻结模块的参数数量。
        """
        stats: Dict[str, int] = {}

        # 冻结 backbone_interface (Video DiT + VAE + text_encoder)
        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            n = self._freeze_module(self.model.backbone_interface, "backbone_interface")
            stats["backbone_interface"] = n

        # 冻结除 action_model 和 backbone_interface 外的所有顶层子模块
        for name, module in self.model.named_children():
            if name in ("backbone_interface", "action_model"):
                continue
            n = self._freeze_module(module, name)
            if n > 0:
                stats[name] = n

        # 汇总
        total_frozen = sum(stats.values())
        logger.info(f"  Total frozen parameters: {total_frozen:,}")
        return stats

    @staticmethod
    def _freeze_module(module: nn.Module, name: str = "") -> int:
        """冻结单个模块：requires_grad=False + eval()。

        Args:
            module: 要冻结的 nn.Module。
            name: 模块名称（用于日志）。

        Returns:
            int: 冻结的参数总数。
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
        """递归将 Action DiT 中所有 nn.Dropout 替换为 nn.Identity。

        原因（Flow-SDE）：
          RL 在线探索阶段，SDE 已通过高斯噪声 ε_τ 注入受控随机性。
          如果 Action DiT 内部还有 Dropout，会引入额外的、不可追踪的随机源，
          导致离散高斯转移概率与实际 rollout 不一致
          （公式中的 ε_k 必须是唯一随机源）。

        替换策略：
          遍历所有子模块，将 nn.Dropout → nn.Identity。
          nn.Identity 是无操作模块，不改变张量形状，保证维度兼容。

        Args:
            module: 递归遍历的根模块（通常是 action_model）。

        Returns:
            int: 替换的 Dropout 层数量。
        """
        count = 0
        # 先收集所有需要替换的 (parent, attr_name) 对
        # 避免在遍历时修改子模块字典（会触发 RuntimeError）
        replacements: List[Tuple[nn.Module, str]] = []

        for name, child in module.named_children():
            if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                replacements.append((module, name))
            elif isinstance(child, nn.Module):
                count += self._replace_dropout_with_identity(child)

        # 执行替换
        for parent, name in replacements:
            old_dropout = getattr(parent, name)
            dropout_p = getattr(old_dropout, "p", 0.0)
            setattr(parent, name, nn.Identity())
            logger.debug(f"    Dropout(p={dropout_p}) → Identity at {_full_name(parent, name)}")
            count += 1

        return count

    def _create_reference_model(self) -> nn.Module:
        """创建 reference Action DiT（冻结深拷贝，用于 KL 散度计算）。

        KL 散度公式::

            D_KL(π_φ ‖ π_ref) = (Δτ / (2 σ_τ (1-τ)^{2τ+1} / σ_τ²))
                                 × ‖v_φ^action - v_ref^action‖² / 2

        等价实现::

            kl_loss = ((prev_sample_mean - prev_sample_mean_ref)²).mean() / (2 * std_dev_t²)

        其中 prev_sample_mean_ref 由 reference model 的速度场计算。
        因此需要一个冻结的原始 SFT Action DiT 副本。

        Returns:
            nn.Module: 冻结的 reference Action DiT。
        """
        # 深拷贝 Action DiT（此时 action_model 已完成 Dropout 替换）
        ref = copy.deepcopy(self.action_model)

        # 冻结 reference model
        self._freeze_module(ref, "ref_action_model")

        # 同样替换 Dropout（保持与新模型结构一致）
        self._replace_dropout_with_identity(ref)

        # 设备放置
        if self.cfg.ref_model_offload:
            # CPU offload 模式：ref model 常驻 CPU，需要时临时移到 GPU
            ref = ref.cpu()
            logger.info("  Reference model offloaded to CPU (save VRAM)")
        else:
            ref = ref.to(self.device)

        n_ref = sum(p.numel() for p in ref.parameters())
        logger.info(f"  Reference Action DiT: {n_ref:,} params (frozen)")
        return ref

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器，仅收集 action_model 中 requires_grad=True 的参数。

        Returns:
            torch.optim.AdamW: 仅优化 Action DiT 参数的优化器。
        """
        # 收集可训练参数
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
    # 验证方法
    # ========================================================================

    def _verify_freezing(self, freeze_stats: Dict[str, int], dropout_count: int):
        """验证冻结状态与冻结完整性。

        检查项：
          #1 因果掩码泄露 → _verify_causal_mask()
          #2 VAE/前端漏冻结 → 检查 backbone_interface 可训练参数数
          #3 (光流/热力图实时更新 → Step 4 检查)
        """
        # 检查1: Action DiT 仍有可训练参数
        n_action_trainable = sum(
            p.numel() for p in self.action_model.parameters() if p.requires_grad
        )
        assert n_action_trainable > 0, \
            "[Step1] FAIL: Action DiT has no trainable parameters!"

        # 检查2: backbone_interface 完全冻结
        n_backbone_trainable = 0
        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            n_backbone_trainable = sum(
                p.numel() for p in self.backbone_interface.parameters() if p.requires_grad
            )
        assert n_backbone_trainable == 0, \
            f"[Step1] FAIL: backbone_interface still has {n_backbone_trainable:,} trainable params! " \
            f"(VAE/Video DiT 必须完全冻结)"

        # 检查3: VAE 单独确认冻结
        n_vae_trainable = sum(p.numel() for p in self.vae.parameters() if p.requires_grad) \
            if self.vae is not None else 0
        assert n_vae_trainable == 0, \
            f"[Step1] FAIL: VAE still has {n_vae_trainable:,} trainable params!"

        # 检查4: Multistream 才要求因果掩码；原版使用标准 cross-attention。
        if self.model_variant == "multistream":
            self._verify_causal_mask()
        else:
            logger.info("  [Causal Mask] Original DiT4DiT: not applicable")

        # 汇总日志
        logger.info(f"  [Verify] Frozen modules: {list(freeze_stats.keys())}")
        logger.info(f"  [Verify] Dropout replaced: {dropout_count}")
        logger.info(f"  [Verify] Trainable (Action DiT): {n_action_trainable:,} params")
        logger.info(f"  [Verify] backbone trainable: {n_backbone_trainable} (must be 0)")
        logger.info(f"  [Verify] VAE trainable: {n_vae_trainable} (must be 0)")

    def _verify_causal_mask(self):
        """验证因果掩码完整性。

        确认 Action DiT 的 Cross-Attention 中因果掩码 M：
          - 前 N 个位置 (语义 Z_x) = -1e9 或 -inf（彻底屏蔽）
          - 后 2N 个位置 (物理 Z_f + 意图 Z_v) = 0（完全开放）

        检查方式：
          1. 如果掩码是静态预计算的 Tensor，直接检查数值
          2. 如果掩码是动态构建的（forward 时生成），检查构建逻辑的参数
        """
        action_model = self.action_model

        # 尝试从多个位置查找因果掩码
        mask_candidates = []

        # 位置1: action_model 顶层属性
        for attr in ("causal_mask", "mask", "causal_key_mask", "_causal_mask", "intent_physical_mask"):
            if hasattr(action_model, attr):
                m = getattr(action_model, attr)
                if isinstance(m, torch.Tensor):
                    mask_candidates.append((f"action_model.{attr}", m))

        # 位置2: action_model.dit (MaskedDiT) 内部
        dit = getattr(action_model, "dit", None) or getattr(action_model, "model", None)
        if dit is not None:
            for attr in ("causal_mask", "mask", "causal_key_mask"):
                if hasattr(dit, attr):
                    m = getattr(dit, attr)
                    if isinstance(m, torch.Tensor):
                        mask_candidates.append((f"action_model.dit.{attr}", m))

        # 位置3: transformer_blocks 内的 masked_cross_attn
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
            # 掩码是动态构建的（在 forward 时根据 N_per_stream 生成）
            # 检查 N_per_stream 参数是否存在
            n_per_stream = getattr(action_model, "N_per_stream", None) or \
                           getattr(dit, "N_per_stream", None) if dit is not None else None
            if n_per_stream is not None:
                logger.info(f"  [Causal Mask] Dynamic mode: N_per_stream={n_per_stream}, "
                            f"mask built at forward time (前{n_per_stream}列=-1e9, "
                            f"后{2*n_per_stream}列=0)")
            else:
                logger.info("  [Causal Mask] Dynamic mode: mask built at forward time "
                            "(will verify during rollout in Step 2)")

    @staticmethod
    def _check_mask_values(name: str, mask: torch.Tensor):
        """检查因果掩码的具体数值。"""
        # mask shape 通常是 (L_action, 3N) 或 (B, L_action, 3N) 或带 attention heads
        if mask.dim() < 2:
            logger.info(f"  [Causal Mask] {name}: dim={mask.dim()}, skipping value check")
            return

        # 取最后一个维度作为 key 维度 (3N)
        last_dim = mask.shape[-1]
        # 前 N 个位置应该是很大的负数 (-1e9 或 -inf)
        front = mask[..., : last_dim // 3]
        # 后 2N 个位置应该是 0
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
        """设置各模块的训练/评估模式。

        - backbone_interface (VAE + Video DiT) → eval() (冻结)
        - ref_action_model → eval() (冻结)
        - action_model → train() (可训练)
        """
        # 冻结模块设为 eval
        if hasattr(self.model, "backbone_interface") and self.model.backbone_interface is not None:
            self.backbone_interface.eval()
        if self.ref_action_model is not None:
            self.ref_action_model.eval()

        # 可训练模块设为 train
        self.action_model.train()

    # ========================================================================
    # 工具方法 —— 供 Step 2/3/4 调用
    # ========================================================================

    def compute_ref_velocity(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """用 reference model 计算动作速度场 v_ref（用于 KL 散度）。

        在 no_grad 上下文中执行，不构建计算图。
        如果 ref_model_offload=True，临时将 ref model 移到 GPU 计算后移回 CPU。

        Returns:
            torch.Tensor: reference model 的速度场输出。
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
        """获取所有可训练参数（名称 + 张量）。"""
        return [
            (name, p) for name, p in self.action_model.named_parameters()
            if p.requires_grad
        ]

    def save_checkpoint(
        self, path: str, trainer_state: Optional[Dict[str, Any]] = None
    ):
        """保存 Action DiT 的 RL 训练状态。

        仅保存 action_model 和 optimizer 状态（backbone 已冻结，无需保存）。
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
        """加载 Action DiT 的 RL 训练状态。"""
        ckpt = torch.load(path, map_location="cpu")
        self.action_model.load_state_dict(ckpt["action_model"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"[Step1] Checkpoint loaded from {path}")
        self.loaded_action_checkpoint = str(path)
        return dict(ckpt.get("trainer_state", {}))

    def sync_ref_to_current(self):
        """将 reference model 更新为当前 action_model 的状态。

        用途：在某些 GRPO 变体中，ref model 会定期同步而非永久冻结。
        """
        if self.ref_action_model is None:
            raise RuntimeError("Reference model was disabled for this engine")
        ref_sd = {k: v.detach().clone() for k, v in self.action_model.state_dict().items()}
        self.ref_action_model.load_state_dict(ref_sd)
        self._freeze_module(self.ref_action_model, "ref_action_model")
        logger.info("[Step1] Reference model synced to current Action DiT")


# ============================================================================
# 辅助函数
# ============================================================================

def _full_name(parent: nn.Module, attr: str) -> str:
    """获取模块的可读名称（用于日志）。"""
    parent_name = parent.__class__.__name__
    return f"{parent_name}.{attr}"
