from dataclasses import dataclass
from typing import Optional


ABLATION_CHOICES = ("none", "wo_edl", "wo_gating", "wo_macbert", "wo_kl")
TEXT_BACKBONE_CHOICES = ("macbert", "bert")


@dataclass
class AblationConfig:
    ablation_mode: str = "none"
    use_edl: bool = True
    use_gating: bool = True
    text_backbone: str = "macbert"
    use_kl_loss: bool = True


def resolve_ablation_config(
    ablation_mode: str,
    text_backbone: str = "macbert",
    use_kl_loss_override: Optional[bool] = None,
) -> AblationConfig:
    if ablation_mode not in ABLATION_CHOICES:
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
    if text_backbone not in TEXT_BACKBONE_CHOICES:
        raise ValueError(f"Unsupported text_backbone: {text_backbone}")

    cfg = AblationConfig(ablation_mode=ablation_mode, text_backbone=text_backbone)

    if ablation_mode == "wo_edl":
        cfg.use_edl = False
        cfg.use_gating = False
        cfg.use_kl_loss = False
    elif ablation_mode == "wo_gating":
        cfg.use_gating = False
    elif ablation_mode == "wo_macbert":
        cfg.text_backbone = "bert"
    elif ablation_mode == "wo_kl":
        cfg.use_kl_loss = False

    if use_kl_loss_override is not None:
        cfg.use_kl_loss = bool(use_kl_loss_override)

    if not cfg.use_edl:
        cfg.use_gating = False
        cfg.use_kl_loss = False

    return cfg


def exp_name_from_ablation(ablation_mode: str) -> str:
    return "exp_full" if ablation_mode == "none" else f"exp_{ablation_mode}"
