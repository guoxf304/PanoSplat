"""Shared render-quality metrics and export naming for train / inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from panovggt.utils.erp_loss import l1_erp, psnr_erp, ssim_erp, ws_psnr
from panovggt.utils.export_meta import get_batch_frame_ids, get_batch_frame_stems
from panovggt.utils.lpips_metric import compute_lpips


@dataclass
class ViewRenderMetrics:
    l1: float = 0.0
    ws_psnr: float = 0.0
    erp_psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = float("nan")
    scene_name: str = ""
    image_stem: str = ""
    image_name: str = ""
    batch_idx: int = 0
    view_idx: int = 0
    num_batch_samples: int = 1

    @property
    def compare_filename(self) -> str:
        return compare_image_filename(
            self.scene_name,
            self.image_stem,
            batch_idx=self.batch_idx,
            num_batch_samples=self.num_batch_samples,
        )


def compare_image_filename(
    scene_name: str,
    image_stem: str,
    *,
    batch_idx: int = 0,
    num_batch_samples: int = 1,
) -> str:
    """Export name: ``{scene}_{stem}_compare.png`` or ``{scene}_b{i}_{stem}_compare.png``."""
    scene = str(scene_name).strip() or "scene"
    stem = str(image_stem).strip() or "view"
    if int(num_batch_samples) > 1:
        return f"{scene}_b{int(batch_idx)}_{stem}_compare.png"
    return f"{scene}_{stem}_compare.png"


def parse_scene_name(seq_name: Any) -> str:
    seq = str(seq_name).strip()
    if seq.startswith("SynPano_"):
        return seq[len("SynPano_") :]
    return seq or "scene"


def resolve_scene_name(batch: Mapping[str, Any], batch_idx: int) -> str:
    seq = batch.get("seq_name")
    if seq is None:
        scene = batch.get("scene")
        if scene is None:
            return "scene"
        if isinstance(scene, (list, tuple)):
            return str(scene[batch_idx] if batch_idx < len(scene) else scene[0])
        return str(scene)
    if isinstance(seq, (list, tuple)):
        return parse_scene_name(seq[batch_idx] if batch_idx < len(seq) else seq[0])
    return parse_scene_name(seq)


def resolve_image_stem(
    batch: Mapping[str, Any],
    batch_idx: int,
    view_idx: int,
    *,
    image_paths: Optional[List[str]] = None,
) -> str:
    if image_paths is not None and view_idx < len(image_paths):
        return Path(image_paths[view_idx]).stem

    if batch.get("frame_stems") is not None:
        stems = get_batch_frame_stems(batch, batch_idx)
        if view_idx < len(stems):
            return str(stems[view_idx])

    if batch.get("ids") is not None:
        ids = get_batch_frame_ids(batch, batch_idx)
        if view_idx < len(ids):
            return f"{int(ids[view_idx]):04d}"

    return f"view{view_idx:02d}"


def compute_view_render_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    scene_name: str = "",
    image_stem: str = "",
    image_name: str = "",
    batch_idx: int = 0,
    view_idx: int = 0,
    num_batch_samples: int = 1,
) -> ViewRenderMetrics:
    """Compute L1 / WS-PSNR / ERP-PSNR / SSIM / LPIPS for one CHW pair."""
    rendered = rendered.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    return ViewRenderMetrics(
        l1=float(l1_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item()),
        ws_psnr=ws_psnr(rendered, target),
        erp_psnr=psnr_erp(rendered, target),
        ssim=float(ssim_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item()),
        lpips=compute_lpips(rendered, target),
        scene_name=scene_name,
        image_stem=image_stem,
        image_name=image_name or f"{image_stem}.png",
        batch_idx=int(batch_idx),
        view_idx=int(view_idx),
        num_batch_samples=int(num_batch_samples),
    )


def mean_metric(values: List[float]) -> float:
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def format_result_metrics_block(metrics: List[ViewRenderMetrics]) -> str:
    """Shared ``result.txt`` metrics section for train export and inference."""
    lines = [
        "[render_quality_avg]",
        f"  l1: {mean_metric([m.l1 for m in metrics]):.6f}",
        f"  ws_psnr: {mean_metric([m.ws_psnr for m in metrics]):.4f}",
        f"  erp_psnr: {mean_metric([m.erp_psnr for m in metrics]):.4f}",
        f"  ssim: {mean_metric([m.ssim for m in metrics]):.6f}",
    ]
    mean_lp = mean_metric([m.lpips for m in metrics])
    lpips_str = f"{mean_lp:.6f}" if mean_lp == mean_lp else "nan"
    lines.append(f"  lpips: {lpips_str}")
    lines.append("")
    lines.append("[per_view_metrics]")
    multi = any(int(m.num_batch_samples) > 1 for m in metrics)
    for i, m in enumerate(metrics):
        lp = f"{m.lpips:.6f}" if m.lpips == m.lpips else "nan"
        if multi:
            label = f"b{m.batch_idx}/v{m.view_idx} {m.image_name}"
        else:
            label = m.image_name or m.compare_filename
        lines.append(
            f"  view_{i:03d} ({label}): l1={m.l1:.6f} "
            f"ws_psnr={m.ws_psnr:.4f} erp_psnr={m.erp_psnr:.4f} "
            f"ssim={m.ssim:.6f} lpips={lp}"
        )
    return "\n".join(lines)
