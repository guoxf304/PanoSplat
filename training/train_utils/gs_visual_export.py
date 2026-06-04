"""Export ERP renders after GS training for qualitative comparison."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torchvision

from panovggt.models.loss_gs import GSLoss
from panovggt.render.camera import build_erp_camera, build_erp_camera_from_w2c
from panovggt.render.coord_frame import resolve_scene_scale
from panovggt.render.odgs_bridge import check_odgs_available, render_erp
from train_utils.general import copy_data_to_device

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@torch.no_grad()
def export_erp_visual_comparison(
    trainer,
    output_dir: str = "./output/visual_comparison",
    num_samples: int = 5,
    view_index: Optional[int] = None,
) -> int:
    """
    Render val batches with the trained GS model and save ERP RGB PNGs.

    Filenames: ``epoch{epoch}_sample_{i}.png`` under ``output_dir``.
    """
    if trainer.rank != 0:
        return 0
    if not trainer.val_dataset:
        logger.info("visual_export: no val dataset, skipping.")
        return 0
    if not isinstance(trainer.loss, GSLoss):
        logger.warning("visual_export: loss is not GSLoss, skipping.")
        return 0
    if not check_odgs_available():
        logger.warning("visual_export: ODGS not available, skipping.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    loss_fn: GSLoss = trainer.loss
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    was_training = model.training
    model.eval()

    dataloader = trainer.val_dataset.get_loader(epoch=trainer.epoch)
    epoch_tag = int(trainer.epoch)
    bg = loss_fn._bg.to(device=trainer.device, dtype=torch.float32)
    pipe = loss_fn.render_pipe

    amp_enabled = bool(trainer.optim_conf.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if trainer.optim_conf.amp.amp_dtype == "bfloat16"
        else torch.float16
    )

    saved = 0
    try:
        for batch in dataloader:
            if saved >= num_samples:
                break

            batch = trainer._process_batch(batch)
            batch = copy_data_to_device(batch, trainer.device, non_blocking=True)

            with torch.amp.autocast(
                "cuda", enabled=amp_enabled, dtype=amp_dtype
            ):
                pred = model(images=batch["images"])

            gt = loss_fn.geo_loss.prepare_gt(batch)
            norm_factor = resolve_scene_scale(pred, gt, batch)
            imgs = loss_fn._denorm_images(batch["images"])
            if not loss_fn.use_gt_pose:
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
                    logger.warning("visual_export: missing camera_poses, skip batch.")
                    continue
            poses = poses.float()

            b = imgs.shape[0]
            for bi in range(b):
                if saved >= num_samples:
                    break
                adapter_out = pred_render.get("gaussian_adapter_out")
                if not adapter_out or adapter_out[bi].cloud.get_xyz.numel() == 0:
                    logger.warning("visual_export: empty cloud at batch %s, skip.", bi)
                    continue

                aligned_xyz = loss_fn._build_aligned_cloud_xyz(
                    pred_render, gt, adapter_out, bi
                )
                cloud = loss_fn._prepare_render_cloud(
                    adapter_out[bi].cloud,
                    norm_factor,
                    bi,
                    aligned_xyz=aligned_xyz,
                    xyz_already_normalized=aligned_xyz is not None,
                )
                if cloud.get_xyz.numel() == 0:
                    continue

                vi = view_index
                if vi is None:
                    anchor = pred_render.get("gs_anchor_idx")
                    vi = int(anchor[bi].item()) if anchor is not None else 0
                vi = max(0, min(vi, poses.shape[1] - 1))

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
                    logger.warning("visual_export: render failed: %s", exc)
                    continue

                rendered = pkg["render"].float().clamp(0.0, 1.0)
                out_path = os.path.join(
                    output_dir, f"epoch{epoch_tag}_sample_{saved}.png"
                )
                torchvision.utils.save_image(rendered, out_path)
                saved += 1
    finally:
        model.train(was_training)

    logger.info("visual_export: saved %s ERP images to %s", saved, output_dir)
    return saved
