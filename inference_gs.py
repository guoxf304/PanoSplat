#!/usr/bin/env python3
"""
PanoVGGT + 3DGS inference aligned with training export (``gs_epoch_infer.py``).

Uses the same SynPano preprocessing, ``Trainer._process_batch`` normalization,
``GSLoss.prepare_render_bundle`` + render path, metrics, and compare layout as
epoch-end training inference. Training code is not modified.

Outputs:
  render_gt_images/     {scene}_{stem}_compare.png (GT | Render | Error)
  gaussians/            Gaussian PLY
  pointclouds/          PanoVGGT geometry PLYs
  result.txt            metrics summary

With ``--compare_batch_modes`` (requires ``--reference_export``)::

  <output_dir>/B1_two_images/   B=1 forward, 2 rendered views
  <output_dir>/B2_four_images/  B=2 forward, 4 rendered views
  <output_dir>/comparison.txt   side-by-side metrics
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Repo root + training/ on path (same as launch.py / diagnose_checkpoint.py).
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_TRAIN_ROOT = os.path.join(_PROJECT_ROOT, "training")
for _p in (_PROJECT_ROOT, _TRAIN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torchvision
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from panovggt.models.loss_gs import GSLoss
from panovggt.render.odgs_bridge import check_odgs_available, render_erp
from panovggt.utils.gs_ply_export import save_gaussian_splat_ply
from panovggt.utils.lpips_metric import get_lpips_model, lpips_backend_name
from panovggt.utils.export_meta import (
    ExportBatchMeta,
    ExportMeta,
    find_export_meta,
    get_batch_frame_stems,
    is_identity_R_delta,
    load_export_meta,
    order_image_paths_by_meta,
    resolve_image_paths_from_meta,
    resolve_model_forward_global_step,
    r_delta_from_nested,
    training_dataloader_epoch_from_export,
)
from panovggt.utils.render_metrics import (
    ViewRenderMetrics,
    compute_view_render_metrics,
    format_result_metrics_block,
    mean_metric,
    resolve_image_stem,
    resolve_scene_name,
)
from data.datasets.synpano import SynPanoDataset, _c2w_to_w2c
from data.dataset_util import erp_target_resolution
from train_utils.general import copy_data_to_device
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class InferenceRunSummary:
    """Collected metrics from one ``run_gs_inference`` pass."""

    mode: str
    output_dir: str
    collate_batch_size: int
    num_rendered_views: int
    infer_losses: Dict[str, float]
    render_metrics: List[ViewRenderMetrics]


def _stack_gt_render_error(
    gt: torch.Tensor, render: torch.Tensor, error_map: torch.Tensor
) -> torch.Tensor:
    """Vertical stack: GT | Render | Error (CHW), same as ``gs_epoch_infer``."""
    return torch.cat([gt, render, error_map], dim=1)


def _fmt_psnr(val: float) -> str:
    return "inf" if val == float("inf") else f"{val:.4f}"


def _fmt_ssim(val: float) -> str:
    return f"{val:.4f}"


def _fmt_lpips(val: float) -> str:
    return "nan" if val != val else f"{val:.4f}"


def _mean_metric(values: List[float]) -> float:
    return mean_metric(values)


def _print_view_metrics(records: List[ViewRenderMetrics]) -> None:
    """Print per-image WS-PSNR / ERP-PSNR / SSIM / LPIPS and mean on the next line."""
    multi = any(int(r.num_batch_samples) > 1 for r in records)
    for r in records:
        tag = (
            f"b{r.batch_idx}/v{r.view_idx} {r.image_name}"
            if multi
            else r.image_name
        )
        print(
            f"[infer] {tag}: "
            f"WS-PSNR={_fmt_psnr(r.ws_psnr)} dB, "
            f"ERP-PSNR={_fmt_psnr(r.erp_psnr)} dB, "
            f"SSIM={_fmt_ssim(r.ssim)}, "
            f"LPIPS={_fmt_lpips(r.lpips)}"
        )
    if not records:
        return
    print(
        f"[infer] mean: "
        f"WS-PSNR={_fmt_psnr(_mean_metric([r.ws_psnr for r in records]))} dB, "
        f"ERP-PSNR={_fmt_psnr(_mean_metric([r.erp_psnr for r in records]))} dB, "
        f"SSIM={_fmt_ssim(_mean_metric([r.ssim for r in records]))}, "
        f"LPIPS={_fmt_lpips(_mean_metric([r.lpips for r in records]))}"
    )


def collect_images(image_dir: str) -> List[str]:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMG_EXTS
    )
    if not paths:
        raise ValueError(f"No images in {image_dir}")
    return paths


def _resolve_checkpoint_global_step(ckpt: dict) -> Optional[int]:
    for key in ("global_step", "step", "iteration"):
        if key in ckpt and ckpt[key] is not None:
            return int(ckpt[key])
    steps = ckpt.get("steps")
    if isinstance(steps, dict) and steps.get("train") is not None:
        return int(steps["train"])
    return None


def load_model_and_loss_from_config(
    config_name: str,
    checkpoint_path: str,
    device: str,
    *,
    img_size: int,
    global_step: Optional[int] = None,
    model_forward_step: Optional[int] = None,
) -> Tuple[Any, GSLoss, Any]:
    config_dir = os.path.join(os.path.dirname(__file__), "training", "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.img_size = img_size
        if cfg.get("model") is not None:
            cfg.model.img_size = img_size
    model = instantiate(cfg.model, _recursive_=True)
    loss_fn = instantiate(cfg.loss, _recursive_=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] checkpoint: {checkpoint_path}")
    if missing:
        print(f"[load] missing ({len(missing)}): {missing[:8]}...")
    if unexpected:
        print(f"[load] unexpected ({len(unexpected)}): {unexpected[:8]}...")

    step = model_forward_step
    if step is None:
        step = global_step
    if step is None:
        step = _resolve_checkpoint_global_step(ckpt)
    if step is not None and hasattr(model, "set_global_step"):
        model.set_global_step(int(step))
        label = "model_forward_step" if model_forward_step is not None else "global_step"
        print(f"[load] {label}={int(step)}")

    model = model.to(device).eval()
    loss_fn = loss_fn.to(device)
    loss_fn.eval()
    return model, loss_fn, cfg


def create_synpano_preprocessor(cfg) -> SynPanoDataset:
    """SynPano loader used for the same ``process_one_image`` path as training export."""
    common_conf = cfg.data.train.common_config
    ds_cfg = cfg.data.train.dataset.dataset_configs[0]
    return instantiate(ds_cfg, common_conf=common_conf, _recursive_=False)


def ensure_distributed_single_process() -> None:
    """Init a 1-rank process group so ``DynamicTorchDataset`` can build a loader."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def build_batch_from_training_dataloader(
    cfg,
    export_meta: ExportMeta,
    *,
    dataloader_batch_idx: int = 0,
) -> dict:
    """
    Rebuild the exact collated train batch used by ``export_epoch_train_inference``.

    Matches training export batch size (e.g. B=2 when ``max_img_per_gpu=4`` and
    ``fix_img_num=2``), which can change the model forward relative to a B=1 replay.
    """
    from train_utils.general import set_seeds

    ensure_distributed_single_process()
    dl_epoch = training_dataloader_epoch_from_export(export_meta)
    set_seeds(int(export_meta.trainer_seed) + dl_epoch * 100, 50, 0)

    train_ds = instantiate(cfg.data.train, _recursive_=False)
    composed = getattr(train_ds, "dataset", train_ds)
    prev_training = bool(getattr(composed, "training", True))
    if hasattr(composed, "training"):
        composed.training = False

    loader = train_ds.get_loader(epoch=dl_epoch)
    batch = None
    for bi, candidate in enumerate(loader):
        if bi == dataloader_batch_idx:
            batch = candidate
            break

    if hasattr(composed, "training"):
        composed.training = prev_training

    if batch is None:
        raise ValueError(
            f"dataloader batch {dataloader_batch_idx} not found "
            f"(epoch={dl_epoch}, display_epoch={export_meta.epoch})"
        )
    return batch


def _set_aug_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_epoch_infer_batch_from_paths(
    image_paths: List[str],
    synpano: SynPanoDataset,
    *,
    aug_seed: Optional[int] = None,
    geom_aug_R_delta: Optional[torch.Tensor] = None,
    scene_name: Optional[str] = None,
) -> dict:
    """
    Build one train-export-style batch (B=1) from explicit image paths.

    Matches ``gs_epoch_infer`` dataloader tensors:
    - SynPano ``process_one_image`` (ERP pitch/yaw/roll aug when ``training=True``)
    - **No** ComposedDataset color jitter (export sets ``ComposedDataset.training=False``)
    - images-only placeholders for depth / pose / masks

  When ``geom_aug_R_delta`` is provided (from ``export_meta.json``), it takes
  precedence over ``aug_seed``.
    """
    if geom_aug_R_delta is None:
        _set_aug_seed(aug_seed)
        R_delta = synpano._prepare_augmentation_params()
    else:
        R_delta = (
            None
            if is_identity_R_delta(geom_aug_R_delta)
            else geom_aug_R_delta.to(dtype=torch.float32)
        )

    target_resolution = erp_target_resolution(synpano.img_size, synpano.patch_size)
    equi_rotate = synpano._get_equi_rotate(target_resolution[0])

    if scene_name:
        seq_index = _synpano_seq_index(synpano, scene_name)
        orig_resolution = np.array(
            synpano.trajectories[seq_index]["resolution"], dtype=np.int32
        )
    else:
        orig_resolution = np.array(target_resolution, dtype=np.int32)

    identity_c2w = np.eye(4, dtype=np.float32)
    pose_w2c = _c2w_to_w2c(identity_c2w)

    lists: Dict[str, list] = {
        "images": [],
        "depths": [],
        "extrinsics": [],
        "cam_points": [],
        "world_points": [],
        "point_masks": [],
        "original_sizes": [],
        "frame_stems": [],
    }

    for rgb_path in image_paths:
        image = synpano._read_and_resize_image(rgb_path, target_resolution)
        depth_map = synpano._placeholder_depth_map(target_resolution)
        frame_data = synpano.process_one_image(
            image=image,
            depth_map=depth_map,
            extrinsic_w2c=pose_w2c,
            shape=target_resolution,
            equi_rotate=equi_rotate,
            R_delta=R_delta,
            depth_max=synpano.depth_max,
        )
        if not synpano.require_depth:
            frame_data["valid_mask"] = torch.ones_like(
                frame_data["valid_mask"], dtype=torch.bool
            )

        lists["images"].append(frame_data["rgb"])
        lists["depths"].append(frame_data["depth_tensor"])
        lists["extrinsics"].append(frame_data["extrinsic"])
        lists["cam_points"].append(frame_data["cam_coords"])
        lists["world_points"].append(frame_data["world_coords"])
        lists["point_masks"].append(frame_data["valid_mask"])
        lists["original_sizes"].append(orig_resolution)
        lists["frame_stems"].append(Path(rgb_path).stem)

    scene_tag = scene_name or "scene"
    sample: Dict[str, Any] = {
        "seq_name": f"SynPano_{scene_tag}",
        "ids": list(range(len(image_paths))),
        "frame_num": len(image_paths),
        "frame_stems": lists["frame_stems"],
    }
    for key, values in lists.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            sample[key] = torch.stack(values, dim=0)
        elif all(isinstance(v, np.ndarray) for v in values):
            sample[key] = torch.from_numpy(np.stack(values))

    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    if R_delta is not None:
        batch["geom_aug_R_delta"] = R_delta.unsqueeze(0)
    else:
        batch["geom_aug_R_delta"] = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    return batch


def _synpano_seq_index(synpano: SynPanoDataset, scene_name: str) -> int:
    for i, traj in enumerate(synpano.trajectories):
        if traj["scene"] == scene_name:
            return i
    raise ValueError(f"Scene {scene_name!r} not found in SynPanoDataset")


def _synpano_sample_to_batch(sample: Mapping[str, Any]) -> dict:
    """Wrap one ``get_data`` sample as a B=1 training batch."""
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, list):
            if value and all(isinstance(v, torch.Tensor) for v in value):
                batch[key] = torch.stack(value, dim=0).unsqueeze(0)
            elif value and all(isinstance(v, np.ndarray) for v in value):
                batch[key] = torch.from_numpy(np.stack(value)).unsqueeze(0)
            else:
                batch[key] = [value]
        elif torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    if "geom_aug_R_delta" in sample and torch.is_tensor(sample["geom_aug_R_delta"]):
        g = sample["geom_aug_R_delta"]
        batch["geom_aug_R_delta"] = g.unsqueeze(0) if g.dim() == 2 else g
    return batch


def build_batch_from_export_meta(
    synpano: SynPanoDataset,
    batch_meta: ExportBatchMeta,
    *,
    geom_aug_R_delta: Optional[torch.Tensor] = None,
) -> dict:
    """
    Rebuild an export batch through SynPano ``get_data`` (training dataloader path).

    Prefer this over ``build_epoch_infer_batch_from_paths`` when replaying
    ``export_meta.json`` so resize, poses, ``original_sizes``, and geom aug match.
    """
    seq_index = _synpano_seq_index(synpano, batch_meta.scene_name)
    frame_ids = [int(i) for i in batch_meta.frame_ids]
    if len(frame_ids) < 2:
        raise ValueError("export meta must list at least two frame_ids")

    R_delta = geom_aug_R_delta
    if R_delta is None and batch_meta.geom_aug_R_delta is not None:
        R_delta = r_delta_from_nested(batch_meta.geom_aug_R_delta)

    sample = synpano.get_data(
        seq_index=seq_index,
        img_per_seq=len(frame_ids),
        ids=frame_ids,
        geom_aug_R_delta=R_delta,
    )
    return _synpano_sample_to_batch(sample)


def resolve_inference_inputs_from_export(
    reference_export: str,
    *,
    image_dir: Optional[str] = None,
    export_batch_idx: int = 0,
    aug_seed_override: Optional[int] = None,
    global_step_override: Optional[int] = None,
) -> Tuple[
    List[str],
    ExportMeta,
    ExportBatchMeta,
    int,
    Optional[int],
    Optional[torch.Tensor],
    str,
]:
    """
    Load ``export_meta.json`` from an epoch export directory.

    Returns:
        image_paths, export_meta, batch_meta, global_step, aug_seed,
        geom_aug_R_delta, reference_meta_path
    """
    meta_path = find_export_meta(reference_export)
    export_meta = load_export_meta(meta_path)
    batch_meta = export_meta.primary_batch(export_batch_idx)

    if image_dir:
        image_paths = order_image_paths_by_meta(
            collect_images(image_dir), batch_meta
        )
    else:
        image_paths = resolve_image_paths_from_meta(batch_meta)

    global_step = (
        global_step_override
        if global_step_override is not None
        else export_meta.global_step
    )
    aug_seed = (
        aug_seed_override
        if aug_seed_override is not None
        else batch_meta.aug_seed
    )
    geom_aug_R_delta = r_delta_from_nested(batch_meta.geom_aug_R_delta)

    return (
        image_paths,
        export_meta,
        batch_meta,
        int(global_step),
        int(aug_seed) if aug_seed is not None else None,
        geom_aug_R_delta,
        meta_path,
    )


def process_batch_like_trainer(batch: Mapping, cfg) -> dict:
    """Same normalization as ``Trainer._process_batch`` (no distributed repeat)."""
    batch = dict(batch)
    repeat_batch = bool(cfg.data.train.common_config.get("repeat_batch", False))
    if repeat_batch:
        tensor_keys = [
            "images",
            "depths",
            "extrinsics",
            "cam_points",
            "world_points",
            "point_masks",
        ]
        for key in tensor_keys:
            if key in batch:
                t = batch[key]
                batch[key] = torch.cat([t, torch.flip(t, dims=[1])], dim=0)

    norm_ext, norm_cam, norm_world, norm_depth, avg_scale = (
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )
    )
    batch["extrinsics"] = norm_ext
    batch["cam_points"] = norm_cam
    batch["world_points"] = norm_world
    batch["depths"] = norm_depth
    batch["norm_factors"] = avg_scale
    return batch


def pseudo_gt_from_pred(pred: dict) -> dict:
    local_points = pred["local_points"]
    masks = torch.isfinite(local_points).all(dim=-1) & (
        local_points.norm(dim=-1) > 1e-3
    )
    return {"valid_masks": masks}


def save_ply(
    path: str, xyz: np.ndarray, rgb: np.ndarray, label: str = "points"
) -> None:
    assert xyz.shape[0] == rgb.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = xyz.shape[0]
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        arr = np.empty(
            n,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("r", np.uint8),
                ("g", np.uint8),
                ("b", np.uint8),
            ],
        )
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(arr.tobytes())
    print(f"[ply] {n:,} {label} → {path}")


def points_and_colors_from_frame(
    points_hw3: np.ndarray,
    image_hw3: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (
        valid_mask
        if valid_mask is not None
        else np.ones(points_hw3.shape[:2], dtype=bool)
    )
    depth = np.linalg.norm(points_hw3, axis=-1)
    mask = mask & (depth > 1e-3) & np.isfinite(depth)
    xyz = points_hw3[mask]
    if image_hw3.max() <= 1.0:
        rgb = (image_hw3[mask] * 255).astype(np.uint8)
    else:
        rgb = image_hw3[mask].astype(np.uint8)
    return xyz, rgb


def _tensor_to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def export_panovggt_geometry_pointclouds(
    pred: dict,
    images: torch.Tensor,
    image_paths: List[str],
    output_dir: str,
    valid_masks: Optional[torch.Tensor] = None,
    *,
    batch_idx: int = 0,
) -> Tuple[int, int]:
    local_pts = pred.get("local_points")
    world_pts = pred.get("world_points")
    if world_pts is None:
        world_pts = pred.get("points")

    if local_pts is None and world_pts is None:
        print("[warn] No local_points/world_points in model output; skip geometry PLY.")
        return 0, 0

    per_frame_dir = os.path.join(output_dir, "pointclouds", "per_frame")
    merged_dir = os.path.join(output_dir, "pointclouds")
    os.makedirs(per_frame_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    if torch.is_tensor(local_pts) and local_pts.dim() == 5:
        local_pts = _tensor_to_numpy_f32(local_pts[batch_idx])
    elif torch.is_tensor(local_pts):
        local_pts = _tensor_to_numpy_f32(local_pts)

    if torch.is_tensor(world_pts) and world_pts.dim() == 5:
        world_pts = _tensor_to_numpy_f32(world_pts[batch_idx])
    elif torch.is_tensor(world_pts):
        world_pts = _tensor_to_numpy_f32(world_pts)

    s = images.shape[1]
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    per_frame_count = 0

    for vi in range(s):
        stem = Path(image_paths[vi]).stem
        img_rgb = (
            images[batch_idx, vi].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        )
        img_rgb = np.clip(img_rgb, 0.0, 1.0)

        mask_np: Optional[np.ndarray] = None
        if valid_masks is not None:
            mask_np = valid_masks[batch_idx, vi].detach().cpu().numpy().astype(bool)

        if local_pts is not None:
            xyz, rgb = points_and_colors_from_frame(
                local_pts[vi], img_rgb, valid_mask=mask_np
            )
            if xyz.shape[0] > 0:
                save_ply(
                    os.path.join(per_frame_dir, f"{stem}.ply"),
                    xyz,
                    rgb,
                    label="geometry (local frame)",
                )
                per_frame_count += 1

        if world_pts is not None:
            xyz_w, rgb_w = points_and_colors_from_frame(
                world_pts[vi], img_rgb, valid_mask=mask_np
            )
            if xyz_w.shape[0] > 0:
                all_xyz.append(xyz_w)
                all_rgb.append(rgb_w)

    merged_count = 0
    if all_xyz:
        merged_xyz = np.concatenate(all_xyz, axis=0)
        merged_rgb = np.concatenate(all_rgb, axis=0)
        save_ply(
            os.path.join(merged_dir, "merged.ply"),
            merged_xyz,
            merged_rgb,
            label="geometry (world frame, merged)",
        )
        merged_count = int(merged_xyz.shape[0])
        print(f"[geometry] merged cloud: {merged_count:,} points")

    return per_frame_count, merged_count


def _write_result_txt(
    path: str,
    *,
    global_step: int,
    num_views: int,
    metrics: List[ViewRenderMetrics],
    infer_losses: Dict[str, float],
    use_gt_pose: bool,
    run_label: str = "",
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("mode: standalone_inference (gs_epoch_infer aligned)\n")
        if run_label:
            f.write(f"run_label: {run_label}\n")
        f.write(f"global_step: {global_step}\n")
        f.write(f"num_export_batches: 1\n")
        f.write(f"num_rendered_views: {num_views}\n")
        f.write(f"use_gt_pose: {use_gt_pose}\n")
        f.write("\n[infer_batch_loss_avg]\n")
        for k, v in infer_losses.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")
        f.write(format_result_metrics_block(metrics))
        f.write("\n")


@torch.no_grad()
def run_gs_inference(
    model,
    loss_fn: GSLoss,
    batch: dict,
    image_paths: List[str],
    output_dir: str,
    device: torch.device,
    cfg,
    *,
    global_step: int = 0,
    batch_idx: int = 0,
    render_all_views: bool = True,
    reference_meta_path: Optional[str] = None,
    export_batch_meta=None,
    checkpoint_path: Optional[str] = None,
    render_sample_indices: Optional[List[int]] = None,
    geometry_sample_bi: int = 0,
    run_label: str = "",
) -> InferenceRunSummary:
    """Mirror ``export_epoch_train_inference`` for a pre-built batch."""
    if not check_odgs_available():
        raise RuntimeError("odgs_gaussian_rasterization is required for ERP rendering.")

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "render_gt_images")
    gs_dir = os.path.join(output_dir, "gaussians")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gs_dir, exist_ok=True)

    batch = process_batch_like_trainer(batch, cfg)
    batch = copy_data_to_device(batch, device, non_blocking=True)
    batch["_dataloader_batch_idx"] = batch_idx

    amp_enabled = bool(cfg.optim.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if cfg.optim.amp.amp_dtype == "bfloat16"
        else torch.float16
    )

    bg = loss_fn._bg.to(device=device, dtype=torch.float32)
    pipe = loss_fn.render_pipe

    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
        pred = model(images=batch["images"])

    with torch.amp.autocast("cuda", enabled=False):
        loss_dict = loss_fn(pred, batch)

    infer_losses = {
        k: float(v.detach().float().item())
        for k, v in loss_dict.items()
        if isinstance(v, torch.Tensor) and k.startswith("loss_")
    }

    gt_geom = pseudo_gt_from_pred(pred)
    n_per_frame, n_merged = export_panovggt_geometry_pointclouds(
        pred,
        batch["images"],
        image_paths,
        output_dir,
        valid_masks=gt_geom.get("valid_masks"),
        batch_idx=geometry_sample_bi,
    )

    pred_render, gt, norm_factor, imgs = loss_fn.prepare_render_bundle(pred, batch)
    poses = (
        gt["camera_poses"]
        if loss_fn.use_gt_pose
        else pred_render.get("camera_poses")
    )
    if poses is None:
        raise RuntimeError("Missing camera_poses for rendering.")
    poses = poses.float()

    adapter_out = pred_render.get("gaussian_adapter_out")
    if not adapter_out:
        raise RuntimeError("No gaussian_adapter_out after prepare_render_bundle.")

    b = imgs.shape[0]
    render_metrics: List[ViewRenderMetrics] = []
    gaussian_ply_saved: Dict[int, bool] = {}
    saved_views = 0
    sample_indices = (
        render_sample_indices
        if render_sample_indices is not None
        else list(range(b))
    )
    multi_sample = b > 1

    get_lpips_model(device)
    backend = lpips_backend_name()
    if backend:
        print(f"[infer] LPIPS backend: {backend}")

    for bi in sample_indices:
        scene_name = resolve_scene_name(batch, bi)
        frame_stems = get_batch_frame_stems(batch, bi)

        cloud_raw = adapter_out[bi].cloud
        if cloud_raw.get_xyz.numel() == 0:
            print(f"[warn] empty cloud bi={bi}, skip")
            continue
        cloud = loss_fn.prepare_render_cloud_for_batch_item(
            pred_render, gt, adapter_out, bi, norm_factor
        )
        if cloud is None or cloud.get_xyz.numel() == 0:
            print(f"[warn] empty render cloud bi={bi}, skip")
            continue

        if bi not in gaussian_ply_saved:
            ply_name = (
                "gaussians.ply"
                if not multi_sample and bi == 0
                else f"gaussians_b{bi}.ply"
            )
            save_gaussian_splat_ply(os.path.join(gs_dir, ply_name), cloud)
            gaussian_ply_saved[bi] = True
            if bi == 0 and multi_sample:
                save_gaussian_splat_ply(os.path.join(gs_dir, "gaussians.ply"), cloud)

        view_range = range(poses.shape[1]) if render_all_views else [0]
        anchor = pred.get("gs_anchor_idx")
        if not render_all_views and anchor is not None:
            view_range = [int(anchor[bi].item())]

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
                device,
            )
            try:
                pkg = render_erp(view_cloud, cam, bg, pipe=pipe)
            except RuntimeError as exc:
                print(f"[warn] render failed bi={bi} view={vi}: {exc}")
                continue

            rendered = pkg["render"].float().clamp(0.0, 1.0)
            target = imgs[bi, vi].float().clamp(0.0, 1.0)
            error_map = torch.abs(rendered - target)

            stack = _stack_gt_render_error(target, rendered, error_map)
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
                num_batch_samples=b,
            )
            torchvision.utils.save_image(
                stack, os.path.join(image_dir, metrics.compare_filename)
            )

            render_subdir = os.path.join(output_dir, "renders")
            os.makedirs(render_subdir, exist_ok=True)
            render_fname = (
                f"b{bi}_{image_stem}_render.png"
                if multi_sample
                else f"{image_stem}_render.png"
            )
            torchvision.utils.save_image(
                rendered.cpu(), os.path.join(render_subdir, render_fname)
            )

            render_metrics.append(metrics)
            saved_views += 1

    if saved_views == 0:
        raise RuntimeError("No views rendered.")

    _write_result_txt(
        os.path.join(output_dir, "result.txt"),
        global_step=global_step,
        num_views=saved_views,
        metrics=render_metrics,
        infer_losses=infer_losses,
        use_gt_pose=loss_fn.use_gt_pose,
        run_label=run_label,
    )

    info_path = os.path.join(output_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("pipeline: gs_epoch_infer aligned\n")
        if run_label:
            f.write(f"run_label: {run_label}\n")
        if checkpoint_path:
            f.write(f"checkpoint: {os.path.abspath(checkpoint_path)}\n")
        f.write(f"images: {len(image_paths)}\n")
        f.write(f"global_step: {global_step}\n")
        f.write(f"model_forward_step: {pred.get('gs_global_step', global_step)}\n")
        f.write(f"inference_batch_size: {batch['images'].shape[0]}\n")
        if reference_meta_path:
            f.write(f"reference_export_meta: {reference_meta_path}\n")
        if export_batch_meta is not None:
            f.write(f"export_frame_stems: {export_batch_meta.frame_stems}\n")
            f.write(f"export_frame_ids: {export_batch_meta.frame_ids}\n")
            f.write(f"export_aug_seed: {export_batch_meta.aug_seed}\n")
            f.write(
                "export_geom_aug_R_delta: "
                f"{'present' if export_batch_meta.geom_aug_R_delta else 'none'}\n"
            )
        f.write(f"gs_means_mode: {pred.get('gs_means_mode', 'anchor')}\n")
        f.write(f"use_gt_pose: {loss_fn.use_gt_pose}\n")
        f.write(f"gs_cross_view_axis_align: {loss_fn.gs_cross_view_axis_align}\n")
        f.write(f"num_rendered_views: {saved_views}\n")
        if multi_sample:
            f.write(
                f"render_batch_samples: {sample_indices} "
                f"({len(sample_indices)} sample(s) x {poses.shape[1]} view(s))\n"
            )
        f.write(f"geometry_per_frame_ply: {n_per_frame}\n")
        f.write(f"geometry_merged_points: {n_merged}\n")
        f.write("\n[image_paths]\n")
        for i, p in enumerate(image_paths):
            f.write(f"  [{i}] {p}\n")
        if render_metrics:
            f.write("\n")
            f.write(format_result_metrics_block(render_metrics))
            f.write("\n")

    _print_view_metrics(render_metrics)
    print(f"[done] {saved_views} views → {output_dir}")
    return InferenceRunSummary(
        mode=run_label or "inference",
        output_dir=output_dir,
        collate_batch_size=int(batch["images"].shape[0]),
        num_rendered_views=saved_views,
        infer_losses=infer_losses,
        render_metrics=render_metrics,
    )


def _metric_for_view(
    metrics: List[ViewRenderMetrics], batch_idx: int, view_idx: int
) -> Optional[ViewRenderMetrics]:
    for m in metrics:
        if m.batch_idx == batch_idx and m.view_idx == view_idx:
            return m
    return None


def write_compare_batch_modes_report(
    path: str,
    b1_summary: InferenceRunSummary,
    b2_summary: InferenceRunSummary,
) -> None:
    """Write side-by-side metrics for B=1 (2 images) vs B=2 (4 images) runs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _loss_block(title: str, losses: Dict[str, float]) -> List[str]:
        lines = [title]
        for k in sorted(losses):
            if k.startswith("loss_"):
                lines.append(f"  {k}: {losses[k]:.6f}")
        return lines

    def _avg_metrics(metrics: List[ViewRenderMetrics]) -> Dict[str, float]:
        return {
            "l1": _mean_metric([m.l1 for m in metrics]),
            "ws_psnr": _mean_metric([m.ws_psnr for m in metrics]),
            "erp_psnr": _mean_metric([m.erp_psnr for m in metrics]),
            "ssim": _mean_metric([m.ssim for m in metrics]),
            "lpips": _mean_metric([m.lpips for m in metrics]),
        }

    lines = [
        "batch_mode_comparison",
        "=" * 72,
        "",
        "Run A — B=1, 2 images (single collate sample, direct 2-view input)",
        f"  output: {b1_summary.output_dir}",
        f"  collate_batch_size: {b1_summary.collate_batch_size}",
        f"  num_rendered_views: {b1_summary.num_rendered_views}",
        "",
        "Run B — B=2, 4 images (training dataloader collated batch)",
        f"  output: {b2_summary.output_dir}",
        f"  collate_batch_size: {b2_summary.collate_batch_size}",
        f"  num_rendered_views: {b2_summary.num_rendered_views}",
        "",
        "[infer_batch_loss_avg]",
    ]
    loss_keys = sorted(
        {k for k in b1_summary.infer_losses if k.startswith("loss_")}
        | {k for k in b2_summary.infer_losses if k.startswith("loss_")}
    )
    for k in loss_keys:
        v1 = b1_summary.infer_losses.get(k, float("nan"))
        v2 = b2_summary.infer_losses.get(k, float("nan"))
        delta = v2 - v1 if v1 == v1 and v2 == v2 else float("nan")
        lines.append(f"  {k}:")
        lines.append(f"    B1_two_images: {v1:.6f}")
        lines.append(f"    B2_four_images: {v2:.6f}")
        lines.append(f"    delta (B2-B1): {delta:.6f}")

    avg1 = _avg_metrics(b1_summary.render_metrics)
    avg2 = _avg_metrics(b2_summary.render_metrics)
    lines.extend(["", "[render_quality_avg]"])
    for key, label in [
        ("l1", "l1"),
        ("ws_psnr", "ws_psnr"),
        ("erp_psnr", "erp_psnr"),
        ("ssim", "ssim"),
        ("lpips", "lpips"),
    ]:
        v1, v2 = avg1[key], avg2[key]
        if key in ("ws_psnr", "erp_psnr"):
            lines.append(
                f"  {label}: B1={v1:.4f}  B2_all_views_avg={v2:.4f}  "
                f"delta={v2 - v1:.4f}"
            )
        else:
            lines.append(
                f"  {label}: B1={v1:.6f}  B2_all_views_avg={v2:.6f}  "
                f"delta={v2 - v1:.6f}"
            )

    lines.extend(
        [
            "",
            "[per_view — same pixels b0, different forward batch size]",
            "  Compare B1 views vs B2 collate_b0 (should match if batch size "
            "were irrelevant):",
        ]
    )
    for m1 in b1_summary.render_metrics:
        m2 = _metric_for_view(b2_summary.render_metrics, 0, m1.view_idx)
        tag = f"v{m1.view_idx} ({m1.image_name})"
        if m2 is None:
            lines.append(f"  {tag}: B1 ws_psnr={m1.ws_psnr:.4f}  B2_b0: missing")
            continue
        lines.append(f"  {tag}:")
        lines.append(
            f"    B1: ws_psnr={m1.ws_psnr:.4f} erp_psnr={m1.erp_psnr:.4f} "
            f"ssim={m1.ssim:.4f}"
        )
        lines.append(
            f"    B2_b0: ws_psnr={m2.ws_psnr:.4f} erp_psnr={m2.erp_psnr:.4f} "
            f"ssim={m2.ssim:.4f}"
        )
        lines.append(
            f"    delta_b0: ws_psnr={m2.ws_psnr - m1.ws_psnr:+.4f} dB  "
            f"erp_psnr={m2.erp_psnr - m1.erp_psnr:+.4f} dB"
        )

    b1_only = [m for m in b2_summary.render_metrics if m.batch_idx == 1]
    if b1_only:
        lines.extend(["", "[per_view — B2 collate_b1 only (extra 2 images)]"])
        for m in b1_only:
            lines.append(
                f"  b1/v{m.view_idx} ({m.image_name}): "
                f"ws_psnr={m.ws_psnr:.4f} erp_psnr={m.erp_psnr:.4f} "
                f"ssim={m.ssim:.4f}"
            )

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[compare] wrote {path}")


def run_compare_batch_modes(
    model,
    loss_fn,
    cfg,
    synpano: SynPanoDataset,
    *,
    export_meta: ExportMeta,
    export_batch_meta: ExportBatchMeta,
    image_paths: List[str],
    geom_aug_R_delta: Optional[torch.Tensor],
    output_root: str,
    device: torch.device,
    global_step: int,
    export_batch_idx: int,
    reference_meta_path: Optional[str],
    checkpoint_path: str,
) -> None:
    """Run B=1 (2 images) and B=2 (4 images) inference; write comparison report."""
    b0_meta = export_meta.primary_batch(
        export_batch_idx, collate_sample_idx=0
    )
    out_b1 = os.path.join(output_root, "B1_two_images")
    out_b2 = os.path.join(output_root, "B2_four_images")

    print("\n" + "=" * 72)
    print("[compare] Run A: B=1 — single sample, 2 images")
    print("=" * 72)
    batch_b1 = build_batch_from_export_meta(
        synpano,
        b0_meta,
        geom_aug_R_delta=geom_aug_R_delta,
    )
    summary_b1 = run_gs_inference(
        model,
        loss_fn,
        batch_b1,
        image_paths,
        out_b1,
        device,
        cfg,
        global_step=global_step,
        batch_idx=export_batch_idx,
        reference_meta_path=reference_meta_path,
        export_batch_meta=b0_meta,
        checkpoint_path=checkpoint_path,
        render_sample_indices=[0],
        geometry_sample_bi=0,
        run_label="B1_two_images",
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("[compare] Run B: B=2 — training dataloader collated batch, 4 images")
    print("=" * 72)
    batch_b2 = build_batch_from_training_dataloader(
        cfg,
        export_meta,
        dataloader_batch_idx=export_batch_idx,
    )
    summary_b2 = run_gs_inference(
        model,
        loss_fn,
        batch_b2,
        image_paths,
        out_b2,
        device,
        cfg,
        global_step=global_step,
        batch_idx=export_batch_idx,
        reference_meta_path=reference_meta_path,
        export_batch_meta=export_batch_meta,
        checkpoint_path=checkpoint_path,
        render_sample_indices=None,
        geometry_sample_bi=0,
        run_label="B2_four_images",
    )

    report_path = os.path.join(output_root, "comparison.txt")
    write_compare_batch_modes_report(report_path, summary_b1, summary_b2)
    print(f"\n[compare] done → {output_root}")
    print(f"  B1 (2 images): {out_b1}")
    print(f"  B2 (4 images): {out_b2}")
    print(f"  report:        {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PanoVGGT-GS inference (aligned with gs_epoch_infer)"
    )
    parser.add_argument("--config", default="synpano_gs", help="Hydra config name")
    parser.add_argument(
        "--checkpoint",
        default="logs/synpano_gs_0527_1252/ckpts/checkpoint_10.pt",
        help="Training checkpoint (.pt).",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Directory with input panoramas. Optional when --reference_export is set "
        "(paths resolved from export meta); if set, images are reordered to match meta.",
    )
    parser.add_argument("--output_dir", default="output/apartment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--global_step",
        type=int,
        default=None,
        help="Override model global_step (default: read from checkpoint).",
    )
    parser.add_argument(
        "--aug_seed",
        type=int,
        default=None,
        help="Seed ERP geom aug (pitch/yaw/roll). Ignored when --reference_export "
        "provides geom_aug_R_delta.",
    )
    parser.add_argument(
        "--reference_export",
        type=str,
        default=None,
        help="Epoch export directory (or export_meta.json) to reproduce frame order, "
        "geom aug, and global_step.",
    )
    parser.add_argument(
        "--export_batch_idx",
        type=int,
        default=0,
        help="Which dataloader batch inside export_meta.json to replay.",
    )
    parser.add_argument(
        "--single_sample_replay",
        action="store_true",
        help="With --reference_export, rebuild B=1 via get_data instead of the full "
        "training dataloader batch (not aligned with epoch export forward).",
    )
    parser.add_argument(
        "--export_sample_only",
        action="store_true",
        help="With --reference_export, render only dataloader batch item 0 "
        "(2 views). Default: render every sample in the collated batch.",
    )
    parser.add_argument(
        "--compare_batch_modes",
        action="store_true",
        help="With --reference_export, run inference twice: B=1 (2 images) vs "
        "B=2 (4 images, training collated batch). Writes comparison.txt under "
        "--output_dir.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        print("[warn] CUDA unavailable, using CPU (slow).")

    reference_meta_path: Optional[str] = None
    export_meta: Optional[ExportMeta] = None
    export_batch_meta = None
    geom_aug_R_delta: Optional[torch.Tensor] = None
    aug_seed = args.aug_seed
    model_forward_step: Optional[int] = None
    render_sample_indices: Optional[List[int]] = None

    if args.reference_export:
        (
            image_paths,
            export_meta,
            export_batch_meta,
            ref_global_step,
            ref_aug_seed,
            geom_aug_R_delta,
            reference_meta_path,
        ) = resolve_inference_inputs_from_export(
            args.reference_export,
            image_dir=args.image_dir if args.image_dir else None,
            export_batch_idx=args.export_batch_idx,
            aug_seed_override=args.aug_seed,
            global_step_override=args.global_step,
        )
        if args.global_step is None:
            args.global_step = ref_global_step
        model_forward_step = resolve_model_forward_global_step(export_meta)
        if args.export_sample_only:
            render_sample_indices = [0]
        if args.aug_seed is None:
            aug_seed = ref_aug_seed
        print(f"[infer] loaded export meta: {reference_meta_path}")
        print(
            f"[infer] reproduce frame order {export_batch_meta.frame_stems}, "
            f"aug_seed={aug_seed}, "
            f"geom_aug_R_delta={'yes' if geom_aug_R_delta is not None else 'no'}, "
            f"model_forward_step={model_forward_step}"
        )
    else:
        if not args.image_dir:
            parser.error("Provide --image_dir or --reference_export.")
        image_paths = collect_images(args.image_dir)

    config_dir = os.path.join(os.path.dirname(__file__), "training", "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg_preview = compose(config_name=args.config)
    OmegaConf.resolve(cfg_preview)
    img_size = int(cfg_preview.get("img_size", 518))
    patch = 14
    h, w = erp_target_resolution(img_size, patch)
    if h % patch != 0 or w % patch != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch}")

    model, loss_fn, cfg = load_model_and_loss_from_config(
        args.config,
        args.checkpoint,
        str(device),
        img_size=img_size,
        global_step=args.global_step,
        model_forward_step=model_forward_step,
    )

    gstep = args.global_step
    if gstep is None:
        gstep = getattr(model, "global_step", 0)

    synpano = create_synpano_preprocessor(cfg)

    if args.compare_batch_modes:
        if export_meta is None:
            parser.error("--compare_batch_modes requires --reference_export.")
        run_compare_batch_modes(
            model,
            loss_fn,
            cfg,
            synpano,
            export_meta=export_meta,
            export_batch_meta=export_batch_meta,
            image_paths=image_paths,
            geom_aug_R_delta=geom_aug_R_delta,
            output_root=args.output_dir,
            device=device,
            global_step=int(gstep),
            export_batch_idx=args.export_batch_idx,
            reference_meta_path=reference_meta_path,
            checkpoint_path=args.checkpoint,
        )
        return

    if export_meta is not None:
        if args.single_sample_replay:
            scene_name = export_batch_meta.scene_name
            print(
                f"[infer] single-sample replay via SynPano.get_data "
                f"(scene={scene_name}, frame_ids={export_batch_meta.frame_ids})"
            )
            batch = build_batch_from_export_meta(
                synpano,
                export_batch_meta,
                geom_aug_R_delta=geom_aug_R_delta,
            )
        else:
            print(
                f"[infer] dataloader replay epoch={export_meta.epoch} "
                f"batch_idx={args.export_batch_idx} "
                f"(full collated batch, matches gs_epoch_infer)"
            )
            batch = build_batch_from_training_dataloader(
                cfg,
                export_meta,
                dataloader_batch_idx=args.export_batch_idx,
            )
    else:
        scene_name = Path(args.image_dir).resolve().parent.name
        print(
            f"[infer] {len(image_paths)} images → {h}x{w} via SynPano process_one_image "
            f"(color_aug=off, geom_aug={synpano.training}, scene={scene_name})"
        )
        batch = build_epoch_infer_batch_from_paths(
            image_paths,
            synpano,
            aug_seed=aug_seed,
            geom_aug_R_delta=geom_aug_R_delta,
            scene_name=scene_name,
        )
    run_gs_inference(
        model,
        loss_fn,
        batch,
        image_paths,
        args.output_dir,
        device,
        cfg,
        global_step=int(gstep),
        batch_idx=args.export_batch_idx,
        reference_meta_path=reference_meta_path,
        export_batch_meta=export_batch_meta,
        checkpoint_path=args.checkpoint,
        render_sample_indices=render_sample_indices,
        geometry_sample_bi=0,
    )


if __name__ == "__main__":
    main()
