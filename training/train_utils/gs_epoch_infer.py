"""Per-epoch GS inference on train batches with GT/render/error export and metrics."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torchvision

from panovggt.models.loss_gs import GSLoss
from panovggt.render.camera import build_erp_camera, build_erp_camera_from_w2c
from panovggt.render.coord_frame import (
    erp_ray_convention_note,
    prepare_render_gaussian_xyz,
    resolve_scene_scale,
)
from panovggt.render.odgs_bridge import check_odgs_available, render_erp
from panovggt.utils.erp_loss import est_wsmap, l1_erp, ssim_erp
from panovggt.utils.gs_ply_export import save_gaussian_splat_ply
from train_utils.general import AverageMeter, copy_data_to_device

logger = logging.getLogger(__name__)

_lpips_model = None


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_export_num_batches(trainer, dataloader, ve) -> int:
    """Resolve how many dataloader batches to export (null / -1 / all = entire loader)."""
    raw = _cfg_get(ve, "num_batches", 1)
    export_all = raw is None or raw == "all" or (isinstance(raw, str) and raw.lower() == "all")
    if export_all or (isinstance(raw, int) and raw < 0):
        n = len(dataloader)
    else:
        n = int(raw)

    limit = getattr(trainer, "limit_train_batches", None)
    if limit is not None:
        # Match train_epoch: batches 0..limit_train_batches inclusive.
        n = min(n, int(limit) + 1)
    return max(n, 0)


@dataclass
class RenderMetrics:
    l1: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = float("nan")


@dataclass
class EpochInferSummary:
    epoch: int
    step: int
    output_dir: str
    batch_losses: Dict[str, float] = field(default_factory=dict)
    render_metrics: List[RenderMetrics] = field(default_factory=list)

    @property
    def mean_psnr(self) -> float:
        vals = [m.psnr for m in self.render_metrics]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_ssim(self) -> float:
        vals = [m.ssim for m in self.render_metrics]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_lpips(self) -> float:
        vals = [m.lpips for m in self.render_metrics if m.lpips == m.lpips]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def mean_l1(self) -> float:
        vals = [m.l1 for m in self.render_metrics]
        return sum(vals) / len(vals) if vals else 0.0


def _get_lpips_model(device: torch.device):
    global _lpips_model
    if _lpips_model is None:
        try:
            from kornia.losses import LPIPS

            _lpips_model = LPIPS().to(device).eval()
        except Exception:
            try:
                import lpips as lpips_pkg

                _lpips_model = lpips_pkg.LPIPS(net="alex").to(device).eval()
            except Exception as exc:
                logger.warning("LPIPS unavailable (%s); metrics will omit LPIPS.", exc)
                _lpips_model = False
    return _lpips_model


def _compute_lpips(render: torch.Tensor, gt: torch.Tensor) -> float:
    model = _get_lpips_model(render.device)
    if not model:
        return float("nan")
    r = render.unsqueeze(0)
    g = gt.unsqueeze(0)
    with torch.no_grad():
        if model.__class__.__module__.startswith("lpips"):
            r = r * 2.0 - 1.0
            g = g * 2.0 - 1.0
        val = model(r, g)
    return float(val.mean().item())


def psnr_erp(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    ws = est_wsmap(pred)
    if ws.dim() == 2:
        ws = ws.view(1, 1, *ws.shape)
    mse = ((pred - gt) ** 2 * ws).mean()
    if mse <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_val**2) / mse).item())


def _meter_avg(meters: Dict[str, AverageMeter], phase: str, key: str) -> Optional[float]:
    meter = meters.get(f"Loss/{phase}_{key}")
    if meter is None or meter.count == 0:
        return None
    return float(meter.avg)


def _stack_gt_render_error(
    gt: torch.Tensor, render: torch.Tensor, error_map: torch.Tensor
) -> torch.Tensor:
    """Vertical stack: GT | Render | Error (CHW)."""
    return torch.cat([gt, render, error_map], dim=1)


def _set_color_augmentation_enabled(trainer, enabled: bool) -> bool:
    """Toggle ComposedDataset color jitter for export (returns previous flag)."""
    ds = getattr(trainer, "train_dataset", None)
    if ds is None:
        return True
    composed = getattr(ds, "dataset", ds)
    prev = bool(getattr(composed, "training", True))
    if hasattr(composed, "training"):
        composed.training = enabled
    return prev


@torch.no_grad()
def export_epoch_train_inference(
    trainer,
    final_train_loss_meters: Optional[Dict[str, AverageMeter]] = None,
) -> Optional[EpochInferSummary]:
    """
    Run GS inference on train batches at epoch end; save comparisons and metrics.

    Output layout::

        output/<exp_name>/Epoch_{epoch:06d}_step_{step:06d}/
            render_gt_images/view_{vi:02d}_compare.png
            result.txt
    """
    if trainer.rank != 0:
        return None

    ve = getattr(trainer.logging_conf, "visual_export", None)
    if ve is None or not _cfg_get(ve, "enabled", False):
        return None
    if not _cfg_get(ve, "every_epoch", True):
        return None
    if not isinstance(trainer.loss, GSLoss):
        logger.warning("epoch_infer: loss is not GSLoss, skipping.")
        return None
    if not check_odgs_available():
        logger.warning("epoch_infer: ODGS not available, skipping.")
        return None
    if trainer.train_dataset is None:
        logger.warning("epoch_infer: no train dataset, skipping.")
        return None

    exp_name = str(getattr(trainer.logging_conf, "log_exp", "exp"))
    output_root = str(_cfg_get(ve, "output_root", f"./output/{exp_name}"))
    render_all_views = bool(_cfg_get(ve, "render_all_views", True))

    epoch_display = int(trainer.epoch) + 1
    global_step = int(trainer.steps.get("train", 0))
    run_dir = os.path.join(
        output_root,
        f"Epoch_{epoch_display:06d}_step_{global_step:06d}",
    )
    image_dir = os.path.join(run_dir, "render_gt_images")
    os.makedirs(image_dir, exist_ok=True)

    loss_fn: GSLoss = trainer.loss
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    was_training = model.training
    model.eval()

    bg = loss_fn._bg.to(device=trainer.device, dtype=torch.float32)
    pipe = loss_fn.render_pipe
    amp_enabled = bool(trainer.optim_conf.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if trainer.optim_conf.amp.amp_dtype == "bfloat16"
        else torch.float16
    )

    dataloader = trainer.train_dataset.get_loader(epoch=trainer.epoch)
    num_batches = resolve_export_num_batches(trainer, dataloader, ve)
    logger.info(
        "epoch_infer epoch=%s exporting %s/%s batches (render_all_views=%s)",
        epoch_display,
        num_batches,
        len(dataloader),
        render_all_views,
    )

    summary = EpochInferSummary(
        epoch=epoch_display,
        step=global_step,
        output_dir=run_dir,
    )
    if final_train_loss_meters:
        for key in (
            "loss_objective",
            "loss_rgb",
            "loss_depth_consis",
            "loss_camera",
            "loss_global_point",
        ):
            val = _meter_avg(final_train_loss_meters, "train", key)
            if val is not None:
                summary.batch_losses[key] = val

    saved_views = 0
    gaussian_ply_saved = False
    batch_loss_accum: Dict[str, List[float]] = {}
    anchor_records: List = []
    prev_training_flag = _set_color_augmentation_enabled(trainer, False)

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch = trainer._process_batch(batch)
            batch = copy_data_to_device(batch, trainer.device, non_blocking=True)
            batch["_dataloader_batch_idx"] = batch_idx
            batch["gs_anchor_audit_dir"] = run_dir

            with torch.amp.autocast(
                "cuda", enabled=amp_enabled, dtype=amp_dtype
            ):
                pred = model(images=batch["images"])

            with torch.amp.autocast("cuda", enabled=False):
                loss_dict = loss_fn(pred, batch)
            for lk, lv in loss_dict.items():
                if not isinstance(lv, torch.Tensor):
                    continue
                if lk.startswith("loss_"):
                    batch_loss_accum.setdefault(lk, []).append(float(lv.detach().float().item()))

            gt = loss_fn.geo_loss.prepare_gt(batch)
            norm_factor = resolve_scene_scale(pred, gt, batch)
            imgs = loss_fn._denorm_images(batch["images"])
            if not loss_fn.use_gt_pose and loss_fn.lambda_geo == 0:
                pred_render = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in pred.items()
                }
                loss_fn.geo_loss.normalize_pred(pred_render, gt)
            else:
                pred_render = pred
            if loss_fn.use_gt_pose:
                poses = gt["camera_poses"]
            else:
                poses = pred_render.get("camera_poses")
                if poses is None:
                    logger.warning("epoch_infer: missing camera_poses, skip batch %s.", batch_idx)
                    continue
            poses = poses.float()

            adapter_out = pred_render.get("gaussian_adapter_out")
            if not adapter_out:
                logger.warning("epoch_infer: no gaussian_adapter_out, skip batch %s.", batch_idx)
                continue

            b = imgs.shape[0]
            for bi in range(b):
                cloud_raw = adapter_out[bi].cloud
                if cloud_raw.get_xyz.numel() == 0:
                    logger.warning("epoch_infer: empty cloud batch=%s bi=%s", batch_idx, bi)
                    continue
                anchor_idx = (
                    int(pred["gs_anchor_idx"][bi].item())
                    if pred.get("gs_anchor_idx") is not None
                    else 0
                )
                base_aligned_xyz = loss_fn._build_aligned_cloud_xyz(
                    pred_render, gt, adapter_out, bi
                )

                view_range = range(poses.shape[1]) if render_all_views else [0]
                anchor = pred.get("gs_anchor_idx")
                if not render_all_views and anchor is not None:
                    view_range = [int(anchor[bi].item())]

                for vi in view_range:
                    if vi >= poses.shape[1]:
                        continue
                    aligned_xyz = base_aligned_xyz
                    if aligned_xyz is None or aligned_xyz.numel() == 0:
                        aligned_xyz = cloud_raw.get_xyz.float()
                    aligned_xyz, aligned_rot = prepare_render_gaussian_xyz(
                        aligned_xyz,
                        cloud_raw._rotation,
                        anchor_idx=anchor_idx,
                        view_idx=int(vi),
                        axis_align_mode=loss_fn.gs_cross_view_axis_align,
                        cross_view_only=True,
                        align_rotation=True,
                    )
                    cloud = loss_fn._prepare_render_cloud(
                        cloud_raw,
                        norm_factor,
                        bi,
                        aligned_xyz=aligned_xyz,
                        xyz_already_normalized=(
                            base_aligned_xyz is not None and base_aligned_xyz.numel() > 0
                        ),
                        aligned_rotation=aligned_rot,
                    )
                    if cloud.get_xyz.numel() == 0:
                        continue
                    if not gaussian_ply_saved and batch_idx == 0 and bi == 0:
                        save_gaussian_splat_ply(
                            os.path.join(run_dir, "gaussians.ply"), cloud
                        )
                        gaussian_ply_saved = True
                    audit = loss_fn.build_anchor_view_record(
                        pred_render,
                        batch,
                        batch_idx=batch_idx,
                        tensor_bi=bi,
                        cur_view_idx=vi,
                        global_step=global_step,
                    )
                    anchor_records.append(audit)
                    if loss_fn.use_gt_pose and batch.get("extrinsics") is not None:
                        w2c = batch["extrinsics"][bi, vi].float()
                        cam = build_erp_camera_from_w2c(
                            w2c, imgs.shape[-2], imgs.shape[-1], trainer.device
                        )
                    else:
                        cam = build_erp_camera(
                            poses[bi, vi], imgs.shape[-2], imgs.shape[-1], trainer.device
                        )
                    try:
                        pkg = render_erp(cloud, cam, bg, pipe=pipe)
                    except RuntimeError as exc:
                        logger.warning(
                            "epoch_infer: render failed batch=%s view=%s: %s",
                            batch_idx,
                            vi,
                            exc,
                        )
                        continue

                    rendered = pkg["render"].float().clamp(0.0, 1.0)
                    target = imgs[bi, vi].float().clamp(0.0, 1.0)
                    error_map = torch.abs(rendered - target)

                    stack = _stack_gt_render_error(target, rendered, error_map)
                    out_name = f"batch{batch_idx:02d}_view{vi:02d}_compare.png"
                    torchvision.utils.save_image(
                        stack, os.path.join(image_dir, out_name)
                    )

                    metrics = RenderMetrics(
                        l1=float(l1_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item()),
                        psnr=psnr_erp(rendered, target),
                        ssim=float(
                            ssim_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item()
                        ),
                        lpips=_compute_lpips(rendered, target),
                    )
                    summary.render_metrics.append(metrics)
                    saved_views += 1
    finally:
        _set_color_augmentation_enabled(trainer, prev_training_flag)
        model.train(was_training)

    if saved_views == 0:
        logger.warning("epoch_infer: no views rendered for epoch %s.", epoch_display)
        return summary

    if anchor_records:
        loss_fn.write_anchor_audit_file(
            run_dir,
            anchor_records,
            epoch=epoch_display,
            global_step=global_step,
            use_gt_pose=loss_fn.use_gt_pose,
        )
        axis_path = os.path.join(run_dir, "anchor.txt")
        with open(axis_path, "a", encoding="utf-8") as f:
            f.write("\n[axis_alignment]\n")
            f.write(f"  gs_cross_view_axis_align: {loss_fn.gs_cross_view_axis_align}\n")
            f.write(f"  env PANOSPLAT_GS_AXIS_ALIGN: {os.environ.get('PANOSPLAT_GS_AXIS_ALIGN', '')}\n")
            f.write(f"  {erp_ray_convention_note()}\n")
        logger.info(
            "epoch_infer anchor audit: %s records → %s",
            len(anchor_records),
            os.path.join(run_dir, "anchor.txt"),
        )

    infer_losses = {
        k: sum(v) / len(v) for k, v in batch_loss_accum.items() if v
    }

    result_path = os.path.join(run_dir, "result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"epoch: {epoch_display}\n")
        f.write(f"global_step: {global_step}\n")
        f.write(f"num_export_batches: {num_batches}\n")
        f.write(f"num_rendered_views: {saved_views}\n")
        f.write("\n[epoch_train_loss_avg]\n")
        for k, v in summary.batch_losses.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n[infer_batch_loss_avg]\n")
        for k, v in infer_losses.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n[render_quality_avg]\n")
        f.write(f"  l1: {summary.mean_l1:.6f}\n")
        f.write(f"  psnr: {summary.mean_psnr:.4f}\n")
        f.write(f"  ssim: {summary.mean_ssim:.6f}\n")
        lpips_str = (
            f"{summary.mean_lpips:.6f}"
            if summary.mean_lpips == summary.mean_lpips
            else "nan"
        )
        f.write(f"  lpips: {lpips_str}\n")
        f.write("\n[per_view_metrics]\n")
        for i, m in enumerate(summary.render_metrics):
            lp = f"{m.lpips:.6f}" if m.lpips == m.lpips else "nan"
            f.write(
                f"  view_{i:03d}: l1={m.l1:.6f} psnr={m.psnr:.4f} "
                f"ssim={m.ssim:.6f} lpips={lp}\n"
            )

    tb_payload: Dict[str, float] = {
        "epoch_infer/l1": summary.mean_l1,
        "epoch_infer/psnr": summary.mean_psnr,
        "epoch_infer/ssim": summary.mean_ssim,
    }
    if summary.mean_lpips == summary.mean_lpips:
        tb_payload["epoch_infer/lpips"] = summary.mean_lpips
    for k, v in summary.batch_losses.items():
        tb_payload[f"epoch_infer/train_avg/{k}"] = v
    for k, v in infer_losses.items():
        tb_payload[f"epoch_infer/infer/{k}"] = v

    if trainer.tb_writer is not None:
        trainer.tb_writer.log_dict(tb_payload, global_step, flush=True)

    logger.info(
        "epoch_infer epoch=%s step=%s views=%s psnr=%.4f ssim=%.4f l1=%.6f lpips=%s → %s",
        epoch_display,
        global_step,
        saved_views,
        summary.mean_psnr,
        summary.mean_ssim,
        summary.mean_l1,
        lpips_str,
        run_dir,
    )
    for k, v in summary.batch_losses.items():
        logger.info("epoch_infer train_avg/%s=%.6f", k, v)
    for k, v in infer_losses.items():
        logger.info("epoch_infer infer/%s=%.6f", k, v)

    return summary
