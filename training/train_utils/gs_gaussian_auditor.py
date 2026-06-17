"""Gaussian physical-state audit for GS training (scale / opacity / anisotropy)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from panovggt.heads.gaussian_adapter import GS_SCALE_MAX
from panovggt.models.loss_gs import GSLoss
from panovggt.render.odgs_bridge import TensorGaussianCloud
from train_utils.general import copy_data_to_device

logger = logging.getLogger(__name__)


def resolve_gs_scale_max(trainer) -> float:
    """Read active scale cap from the model adapter (falls back to module default)."""
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    adapter = getattr(model, "gaussian_adapter", None)
    if adapter is not None and hasattr(adapter, "gs_scale_max"):
        return float(adapter.gs_scale_max)
    if hasattr(model, "gs_scale_max"):
        return float(model.gs_scale_max)
    return GS_SCALE_MAX


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_epoch_run_dir(trainer) -> str:
    """Same layout as ``export_epoch_train_inference`` output folder."""
    exp_name = str(getattr(trainer.logging_conf, "log_exp", "exp"))
    ve = getattr(trainer.logging_conf, "visual_export", None)
    output_root = str(_cfg_get(ve, "output_root", f"./output/{exp_name}"))
    epoch_display = int(trainer.epoch) + 1
    global_step = int(trainer.steps.get("train", 0))
    return os.path.join(
        output_root,
        f"Epoch_{epoch_display:06d}_step_{global_step:06d}",
    )


@dataclass
class GaussianAuditStats:
    num_gaussians: int = 0
    scale_mean: float = 0.0
    scale_max: float = 0.0
    scale_std: float = 0.0
    scale_at_cap_pct: float = 0.0
    opacity_mean: float = 0.0
    opacity_min: float = 0.0
    opacity_saturated_pct: float = 0.0
    aspect_ratio_mean: float = 0.0


def audit_gaussian_tensors(
    scales: torch.Tensor,
    opacities: torch.Tensor,
    *,
    scale_max: float = GS_SCALE_MAX,
) -> GaussianAuditStats:
    """Compute audit stats from activated scales (N,3) and opacities (N,) / (N,1)."""
    opacities = opacities.float().reshape(-1)
    n = int(scales.shape[0])
    if n == 0:
        return GaussianAuditStats()

    scales = scales.float()
    scales_flat = scales.reshape(-1)
    cap = scale_max - 1e-3
    ratios = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp(min=1e-8)

    return GaussianAuditStats(
        num_gaussians=n,
        scale_mean=float(scales_flat.mean().item()),
        scale_max=float(scales_flat.max().item()),
        scale_std=float(scales_flat.std(unbiased=False).item()),
        scale_at_cap_pct=float((scales >= cap).float().mean().item() * 100.0),
        opacity_mean=float(opacities.mean().item()),
        opacity_min=float(opacities.min().item()),
        opacity_saturated_pct=float((opacities > 0.99).float().mean().item() * 100.0),
        aspect_ratio_mean=float(ratios.mean().item()),
    )


def audit_gaussian_cloud(
    cloud: TensorGaussianCloud,
    *,
    scale_max: float = GS_SCALE_MAX,
) -> GaussianAuditStats:
    """Audit activated scales / opacities / anisotropy for one Gaussian cloud."""
    return audit_gaussian_tensors(
        cloud.get_scaling, cloud.get_opacity, scale_max=scale_max
    )


def merge_gaussian_audit_stats(items: List[GaussianAuditStats]) -> GaussianAuditStats:
    """Backward-compatible merge (prefer ``audit_gaussian_tensors`` on concatenated data)."""
    if not items:
        return GaussianAuditStats()

    total_n = sum(s.num_gaussians for s in items)
    if total_n == 0:
        return GaussianAuditStats()

    def wavg(attr: str) -> float:
        return sum(getattr(s, attr) * s.num_gaussians for s in items) / total_n

    return GaussianAuditStats(
        num_gaussians=total_n,
        scale_mean=wavg("scale_mean"),
        scale_max=max(getattr(s, "scale_max") for s in items),
        scale_std=wavg("scale_std"),
        scale_at_cap_pct=wavg("scale_at_cap_pct"),
        opacity_mean=wavg("opacity_mean"),
        opacity_min=min(getattr(s, "opacity_min") for s in items),
        opacity_saturated_pct=wavg("opacity_saturated_pct"),
        aspect_ratio_mean=wavg("aspect_ratio_mean"),
    )


def format_gaussian_audit_report(
    stats: GaussianAuditStats,
    *,
    epoch: int,
    global_step: int,
    phase: str,
    num_batches: int,
    scale_max: float = GS_SCALE_MAX,
) -> str:
    cap = scale_max - 1e-3
    lines = [
        f"epoch: {epoch}",
        f"global_step: {global_step}",
        f"phase: {phase}",
        f"num_batches: {num_batches}",
        f"num_gaussians: {stats.num_gaussians}",
        "",
        "[Scale Audit]",
        f"  mean: {stats.scale_mean:.6f}",
        f"  max: {stats.scale_max:.6f}",
        f"  std: {stats.scale_std:.6f}",
        f"  at_hard_cap_pct (scale >= {cap:.6f}): {stats.scale_at_cap_pct:.4f}%",
        "",
        "[Opacity & Saturation Audit]",
        f"  mean: {stats.opacity_mean:.6f}",
        f"  min: {stats.opacity_min:.6f}",
        f"  saturated_pct (opacity > 0.99): {stats.opacity_saturated_pct:.4f}%",
        "",
        "[Shape Aspect Ratio Audit]",
        f"  mean (max_axis / min_axis): {stats.aspect_ratio_mean:.6f}",
        "",
    ]
    return "\n".join(lines)


@torch.no_grad()
def export_epoch_gaussian_audit(trainer, phase: str = "val") -> Optional[str]:
    """
    Run Gaussian physical-state audit after an epoch val pass.

    Writes ``gaussian_info.txt`` under the epoch output folder, e.g.
    ``output/<exp>/Epoch_000004_step_001200/gaussian_info.txt``.
    """
    if trainer.rank != 0:
        return None
    if not isinstance(trainer.loss, GSLoss):
        return None
    if trainer.val_dataset is None:
        logger.warning("gaussian_audit: no val dataset, skipping.")
        return None

    ve = getattr(trainer.logging_conf, "visual_export", None)
    num_batches_cfg = _cfg_get(ve, "num_batches", 1) if ve is not None else 1

    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    was_training = model.training
    model.eval()

    amp_enabled = bool(trainer.optim_conf.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if trainer.optim_conf.amp.amp_dtype == "bfloat16"
        else torch.float16
    )

    scale_chunks: List[torch.Tensor] = []
    opacity_chunks: List[torch.Tensor] = []
    batches_seen = 0

    try:
        dataloader = trainer.val_dataset.get_loader(epoch=trainer.epoch)
        if num_batches_cfg is None or num_batches_cfg == "all" or (
            isinstance(num_batches_cfg, int) and num_batches_cfg < 0
        ):
            num_batches = len(dataloader)
        else:
            num_batches = int(num_batches_cfg)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch = trainer._process_batch(batch)
            batch = copy_data_to_device(batch, trainer.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                pred = model(images=batch["images"])

            adapter_out = pred.get("gaussian_adapter_out")
            if not adapter_out:
                logger.warning(
                    "gaussian_audit: no gaussian_adapter_out at batch %s.", batch_idx
                )
                continue

            for item in adapter_out:
                cloud = item.cloud
                if cloud.get_xyz.numel() == 0:
                    continue
                scale_chunks.append(cloud.get_scaling.detach().float().cpu())
                opacity_chunks.append(cloud.get_opacity.detach().float().cpu().reshape(-1))

            batches_seen += 1
    finally:
        model.train(was_training)

    if not scale_chunks:
        logger.warning("gaussian_audit: no Gaussians collected, skipping write.")
        return None

    all_scales = torch.cat(scale_chunks, dim=0)
    all_opacities = torch.cat(opacity_chunks, dim=0)
    scale_max = resolve_gs_scale_max(trainer)
    stats = audit_gaussian_tensors(all_scales, all_opacities, scale_max=scale_max)
    run_dir = resolve_epoch_run_dir(trainer)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "gaussian_info.txt")

    epoch_display = int(trainer.epoch) + 1
    global_step = int(trainer.steps.get("train", 0))
    report = format_gaussian_audit_report(
        stats,
        epoch=epoch_display,
        global_step=global_step,
        phase=phase,
        num_batches=batches_seen,
        scale_max=scale_max,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(
        "gaussian_audit epoch=%s step=%s gaussians=%s scale_max=%.4f "
        "cap_pct=%.2f%% opacity_sat=%.2f%% aspect=%.3f → %s",
        epoch_display,
        global_step,
        stats.num_gaussians,
        stats.scale_max,
        stats.scale_at_cap_pct,
        stats.opacity_saturated_pct,
        stats.aspect_ratio_mean,
        out_path,
    )
    return out_path
