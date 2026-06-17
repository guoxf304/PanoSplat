"""Per-epoch GS inference on train batches with GT/render/error export and metrics."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torchvision

from panovggt.models.loss_gs import GSLoss
from panovggt.render.odgs_bridge import check_odgs_available, render_erp
from panovggt.utils.gs_ply_export import save_gaussian_splat_ply
from panovggt.utils.export_meta import (
    EXPORT_META_FILENAME,
    ExportMeta,
    extract_export_batch_meta,
    format_export_meta_block,
    get_batch_frame_stems,
    save_export_meta,
)
from panovggt.utils.render_metrics import (
    ViewRenderMetrics,
    compute_view_render_metrics,
    format_result_metrics_block,
    mean_metric,
    resolve_image_stem,
    resolve_scene_name,
)
from train_utils.general import AverageMeter, copy_data_to_device

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_export_steps(ve) -> List[int]:
    """Optional mid-training export steps (e.g. ``[20]`` for early Gaussian checks)."""
    if ve is None:
        return []
    raw = _cfg_get(ve, "export_steps", None)
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    return sorted({int(s) for s in raw})


def resolve_export_num_batches(trainer, dataloader, ve) -> int:
    """Resolve how many dataloader batches to export (default: 1)."""
    raw = _cfg_get(ve, "num_batches", 1)
    export_all = raw is None or raw == "all" or (isinstance(raw, str) and raw.lower() == "all")
    if export_all or (isinstance(raw, int) and raw < 0):
        n = len(dataloader)
    else:
        n = int(raw)

    limit = getattr(trainer, "limit_train_batches", None)
    if limit is not None:
        n = min(n, int(limit) + 1)
    return max(n, 0)


@dataclass
class EpochInferSummary:
    epoch: int
    step: int
    output_dir: str
    batch_losses: Dict[str, float] = field(default_factory=dict)
    render_metrics: List[ViewRenderMetrics] = field(default_factory=list)

    @property
    def mean_ws_psnr(self) -> float:
        return mean_metric([m.ws_psnr for m in self.render_metrics])

    @property
    def mean_erp_psnr(self) -> float:
        return mean_metric([m.erp_psnr for m in self.render_metrics])

    @property
    def mean_ssim(self) -> float:
        return mean_metric([m.ssim for m in self.render_metrics])

    @property
    def mean_lpips(self) -> float:
        return mean_metric([m.lpips for m in self.render_metrics])

    @property
    def mean_l1(self) -> float:
        return mean_metric([m.l1 for m in self.render_metrics])


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


def _resolve_synpano_dir(trainer) -> str:
    try:
        ds_cfgs = trainer.data_conf.train.dataset.dataset_configs
        for dc in ds_cfgs:
            path = getattr(dc, "SynPano_DIR", None)
            if path:
                return str(path)
    except Exception:
        pass
    return ""


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
    *,
    step_override: Optional[int] = None,
) -> Optional[EpochInferSummary]:
    """
    Run GS inference on train batches; save comparisons, gaussians.ply, metrics.

    Output layout::

        output/<exp_name>/Epoch_{epoch:06d}_step_{step:06d}/
            gaussians.ply / gaussians_b{i}.ply
            render_gt_images/{scene}_b{i}_{stem}_compare.png  (B>1)
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
    export_sample_only = bool(_cfg_get(ve, "export_sample_only", False))

    epoch_display = int(trainer.epoch) + 1
    global_step = (
        int(step_override)
        if step_override is not None
        else int(trainer.steps.get("train", 0))
    )
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
        "epoch_infer epoch=%s exporting %s/%s batches (render_all_views=%s, "
        "export_sample_only=%s)",
        epoch_display,
        num_batches,
        len(dataloader),
        render_all_views,
        export_sample_only,
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
    gaussian_ply_saved: Dict[int, bool] = {}
    batch_loss_accum: Dict[str, List[float]] = {}
    export_batches_meta = []
    forward_gs_step = int(
        getattr(model, "global_step", max(0, global_step - 1))
    )
    prev_training_flag = _set_color_augmentation_enabled(trainer, False)
    synpano_dir = _resolve_synpano_dir(trainer)
    trainer_seed = int(getattr(trainer, "seed_value", 42))

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch = trainer._process_batch(batch)
            batch = copy_data_to_device(batch, trainer.device, non_blocking=True)
            batch["_dataloader_batch_idx"] = batch_idx

            num_collate = int(batch["images"].shape[0])
            sample_indices = (
                [0]
                if export_sample_only
                else list(range(num_collate))
            )
            for bi in sample_indices:
                export_batches_meta.append(
                    extract_export_batch_meta(
                        batch,
                        dataloader_batch_idx=batch_idx,
                        epoch=epoch_display,
                        global_step=global_step,
                        trainer_seed=trainer_seed,
                        synpano_dir=synpano_dir,
                        bi=bi,
                    )
                )

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

            pred_render, gt, norm_factor, imgs = loss_fn.prepare_render_bundle(
                pred, batch
            )
            poses = (
                gt["camera_poses"]
                if loss_fn.use_gt_pose
                else pred_render.get("camera_poses")
            )
            if poses is None:
                logger.warning("epoch_infer: missing camera_poses, skip batch %s.", batch_idx)
                continue

            adapter_out = pred_render.get("gaussian_adapter_out")
            if not adapter_out:
                logger.warning("epoch_infer: no gaussian_adapter_out, skip batch %s.", batch_idx)
                continue

            multi_sample = num_collate > 1

            for bi in sample_indices:
                cloud_raw = adapter_out[bi].cloud
                if cloud_raw.get_xyz.numel() == 0:
                    logger.warning(
                        "epoch_infer: empty cloud batch=%s bi=%s", batch_idx, bi
                    )
                    continue
                cloud = loss_fn.prepare_render_cloud_for_batch_item(
                    pred_render, gt, adapter_out, bi, norm_factor
                )
                if cloud is None or cloud.get_xyz.numel() == 0:
                    continue

                view_range = range(poses.shape[1]) if render_all_views else [0]
                anchor = pred.get("gs_anchor_idx")
                if not render_all_views and anchor is not None:
                    view_range = [int(anchor[bi].item())]

                scene_name = resolve_scene_name(batch, bi)
                frame_stems = get_batch_frame_stems(batch, bi)

                if bi not in gaussian_ply_saved:
                    if multi_sample:
                        save_gaussian_splat_ply(
                            os.path.join(run_dir, f"gaussians_b{bi}.ply"), cloud
                        )
                    if bi == 0:
                        save_gaussian_splat_ply(
                            os.path.join(run_dir, "gaussians.ply"), cloud
                        )
                    gaussian_ply_saved[bi] = True

                for vi in view_range:
                    if vi >= poses.shape[1]:
                        continue

                    view_cloud = loss_fn.cloud_for_view_render(
                        cloud, pred_render, bi, vi
                    )
                    cam = loss_fn.build_render_camera(
                        batch,
                        gt,
                        pred_render,
                        bi,
                        vi,
                        imgs.shape[-2],
                        imgs.shape[-1],
                        trainer.device,
                    )
                    try:
                        pkg = render_erp(view_cloud, cam, bg, pipe=pipe)
                    except RuntimeError as exc:
                        logger.warning(
                            "epoch_infer: render failed batch=%s bi=%s view=%s: %s",
                            batch_idx,
                            bi,
                            vi,
                            exc,
                        )
                        continue

                    rendered = pkg["render"].float().clamp(0.0, 1.0)
                    target = imgs[bi, vi].float().clamp(0.0, 1.0)
                    error_map = torch.abs(rendered - target)

                    image_stem = resolve_image_stem(batch, bi, vi)
                    image_name = (
                        f"{frame_stems[vi]}.png"
                        if vi < len(frame_stems)
                        else f"{image_stem}.png"
                    )
                    metrics = compute_view_render_metrics(
                        rendered,
                        target,
                        scene_name=scene_name,
                        image_stem=image_stem,
                        image_name=image_name,
                        batch_idx=bi,
                        view_idx=vi,
                        num_batch_samples=num_collate,
                    )
                    stack = _stack_gt_render_error(target, rendered, error_map)
                    torchvision.utils.save_image(
                        stack, os.path.join(image_dir, metrics.compare_filename)
                    )
                    summary.render_metrics.append(metrics)
                    saved_views += 1
    finally:
        _set_color_augmentation_enabled(trainer, prev_training_flag)
        model.train(was_training)

    if saved_views == 0:
        logger.warning("epoch_infer: no views rendered for epoch %s.", epoch_display)
        return summary

    infer_losses = {
        k: sum(v) / len(v) for k, v in batch_loss_accum.items() if v
    }

    export_meta = ExportMeta(
        epoch=epoch_display,
        global_step=global_step,
        trainer_seed=trainer_seed,
        model_forward_global_step=forward_gs_step,
        batches=export_batches_meta,
    )
    save_export_meta(os.path.join(run_dir, EXPORT_META_FILENAME), export_meta)

    result_path = os.path.join(run_dir, "result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"epoch: {epoch_display}\n")
        f.write(f"global_step: {global_step}\n")
        f.write(f"num_export_batches: {num_batches}\n")
        f.write(f"num_rendered_views: {saved_views}\n")
        if not export_sample_only:
            f.write("export_all_collate_samples: true\n")
        f.write("\n[epoch_train_loss_avg]\n")
        for k, v in summary.batch_losses.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n[infer_batch_loss_avg]\n")
        for k, v in infer_losses.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")
        f.write(format_result_metrics_block(summary.render_metrics))
        f.write("\n")
        f.write(format_export_meta_block(export_meta))
        f.write("\n")

    tb_payload: Dict[str, float] = {
        "epoch_infer/l1": summary.mean_l1,
        "epoch_infer/ws_psnr": summary.mean_ws_psnr,
        "epoch_infer/erp_psnr": summary.mean_erp_psnr,
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

    lpips_str = (
        f"{summary.mean_lpips:.6f}"
        if summary.mean_lpips == summary.mean_lpips
        else "nan"
    )
    logger.info(
        "epoch_infer epoch=%s step=%s views=%s ws_psnr=%.4f erp_psnr=%.4f "
        "ssim=%.4f l1=%.6f lpips=%s → %s",
        epoch_display,
        global_step,
        saved_views,
        summary.mean_ws_psnr,
        summary.mean_erp_psnr,
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
