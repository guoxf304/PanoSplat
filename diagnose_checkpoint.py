#!/usr/bin/env python3
"""
Diagnose GS checkpoint: Gaussian attributes, cameras, and render settings.

Usage:
  export PYTHONPATH=/mnt/gxf/PanoSplat:/mnt/gxf/PanoSplat/training
  python diagnose_checkpoint.py \\
    --checkpoint logs/synpano_gs_0527_1252/ckpts/checkpoint_10.pt \\
    --config synpano_gs \\
    --val_batch_index 0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from hydra import compose
from hydra.utils import instantiate
from hydra.initialize import initialize_config_dir
from omegaconf import OmegaConf

# training helpers
_TRAIN_ROOT = os.path.join(os.path.dirname(__file__), "training")
if _TRAIN_ROOT not in sys.path:
    sys.path.insert(0, _TRAIN_ROOT)

from train_utils.general import copy_data_to_device
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

from panovggt.models.loss import Loss
from panovggt.models.loss_gs import GSLoss
from panovggt.render.camera import build_erp_camera
from panovggt.render.odgs_bridge import (
    TensorGaussianCloud,
    check_odgs_available,
    render_erp,
)
from panovggt.utils.sh_utils import sh_to_rgb


def _stat(name: str, t: torch.Tensor, indent: str = "  ") -> None:
    t = t.detach().float().reshape(-1)
    finite = torch.isfinite(t)
    n = t.numel()
    n_bad = int((~finite).sum().item())
    if finite.any():
        tf = t[finite]
        print(
            f"{indent}{name}: shape={tuple(t.shape)} "
            f"min={tf.min().item():.6g} max={tf.max().item():.6g} "
            f"mean={tf.mean().item():.6g} std={tf.std().item():.6g} "
            f"finite={n - n_bad}/{n}"
        )
    else:
        print(f"{indent}{name}: shape={tuple(t.shape)} ALL non-finite ({n_bad}/{n})")


def _yellow_fraction(rgb: torch.Tensor) -> float:
    """Fraction of Gaussians with R,G high and B relatively low (yellow-ish)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mask = (r > 0.75) & (g > 0.55) & (b < 0.45)
    return float(mask.float().mean().item())


def _init_single_process_distributed() -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)


def _load_model_and_cfg(config_name: str, checkpoint_path: str, device: torch.device):
    config_dir = os.path.join(os.path.dirname(__file__), "training", "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] {checkpoint_path}")
    print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing and len(missing) <= 12:
        print(f"  missing: {missing}")
    return model.to(device).eval(), cfg


def _process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    extrinsics, cam_points, world_points, depths, avg_scale = (
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )
    )
    batch["extrinsics"] = extrinsics
    batch["cam_points"] = cam_points
    batch["world_points"] = world_points
    batch["depths"] = depths
    batch["norm_factors"] = avg_scale
    return batch


def _pseudo_gt(pred: Dict[str, Any]) -> Dict[str, Any]:
    lp = pred["local_points"]
    masks = torch.isfinite(lp).all(dim=-1) & (lp.norm(dim=-1) > 1e-3)
    return {"valid_masks": masks}


def _cloud_rgb(cloud: TensorGaussianCloud) -> torch.Tensor:
    if cloud.active_sh_degree == 0:
        sh_dc = cloud._features_dc.squeeze(1).float()
        return sh_to_rgb(sh_dc).clamp(0.0, 1.0)
    feats = cloud.get_features.float()
    if feats.shape[1] >= 1:
        return sh_to_rgb(feats[:, 0, :]).clamp(0.0, 1.0)
    return sh_to_rgb(feats.squeeze(1)).clamp(0.0, 1.0)


def _diagnose_cloud(label: str, cloud: TensorGaussianCloud) -> None:
    print(f"\n{'=' * 72}\n  Gaussian cloud: {label}\n{'=' * 72}")
    n = cloud.get_xyz.shape[0]
    print(f"  num_gaussians: {n:,}  sh_degree={cloud.max_sh_degree}")

    _stat("xyz (world / anchor frame)", cloud.get_xyz)
    _stat("scaling_log (_scaling)", cloud._scaling)
    scales = cloud.get_scaling
    _stat("scaling radius [m] (exp)", scales)
    if scales.numel() > 0:
        s_max = scales.max(dim=-1).values
        for thr in (0.05, 0.1, 0.3, 0.5):
            cnt = int((s_max >= thr).sum().item())
            print(f"  Gaussians with max_axis_radius >= {thr} m: {cnt:,} ({100 * cnt / n:.2f}%)")

    _stat("opacity_logit (_opacity)", cloud._opacity.reshape(-1))
    op = cloud.get_opacity.reshape(-1)
    _stat("opacity (sigmoid)", op)
    print(
        f"  opacity > 0.99: {int((op > 0.99).sum().item()):,} "
        f"({100 * (op > 0.99).float().mean().item():.1f}%)"
    )

    _stat("rotation (normalized)", cloud.get_rotation)
    _stat("features_dc raw", cloud._features_dc.reshape(-1))

    rgb = _cloud_rgb(cloud)
    print("  --- color (RGB from SH DC) ---")
    for c, ch in enumerate("RGB"):
        _stat(f"  {ch}", rgb[:, c])
    yf = _yellow_fraction(rgb)
    print(f"  yellow-ish fraction (R>0.75,G>0.55,B<0.45): {100 * yf:.2f}%")
    if yf > 0.5:
        print("  >>> WARNING: majority of Gaussians are yellow-ish (check features_dc / SH).")


def _diagnose_camera_and_render(
    cloud: TensorGaussianCloud,
    poses_c2w: torch.Tensor,
    view_idx: int,
    h: int,
    w: int,
    device: torch.device,
    bg: torch.Tensor,
) -> None:
    print(f"\n{'=' * 72}\n  Render / camera (view {view_idx}, {h}x{w})\n{'=' * 72}")

    print(f"  bg_color (render_erp): {bg.detach().cpu().tolist()}  (RGB, expect [0,0,0] for black)")
    if bg.numel() == 3 and bg.min() > 0.4 and bg[0] > 0.8:
        print("  >>> WARNING: background is not black; may tint the image.")

    c2w = poses_c2w[view_idx].float()
    print("  c2w (camera-to-world):")
    print(c2w.cpu().numpy())

    cam = build_erp_camera(c2w, h, w, device)
    _stat("camera_center", cam.camera_center)
    _stat("world_view_transform", cam.world_view_transform.reshape(-1))

    if not check_odgs_available():
        print("  [skip] ODGS not installed — cannot run test render.")
        return

    try:
        pkg = render_erp(cloud, cam, bg.float())
        img = pkg["render"].float().clamp(0, 1)
        print("  --- rendered ERP image stats ---")
        for c, ch in enumerate("RGB"):
            _stat(f"  pixel {ch}", img[c])
        print(f"  rendered mean RGB: {img.mean(dim=(1, 2)).cpu().tolist()}")
        if img[0].mean() > 0.7 and img[1].mean() > 0.5 and img[2].mean() < 0.4:
            print("  >>> WARNING: rendered image is predominantly yellow.")
    except RuntimeError as exc:
        print(f"  [render FAILED] {exc}")


def _diagnose_gs_raw(gs_out: torch.Tensor) -> None:
    print(f"\n{'=' * 72}\n  Raw head output gs_out\n{'=' * 72}")
    names = [
        "density",
        "scale_raw",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "sh0",
        "sh1",
        "sh2",
        "conf",
    ]
    for i, nm in enumerate(names[: gs_out.shape[1]]):
        _stat(nm, gs_out[0, i])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose GS checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="logs/synpano_gs_0527_1252/ckpts/checkpoint_10.pt",
        help="Path to .pt checkpoint (avoid checkpoint.pt while training writes)",
    )
    parser.add_argument("--config", default="synpano_gs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--val_batch_index",
        type=int,
        default=0,
        help="Which val dataloader batch to use",
    )
    parser.add_argument(
        "--view_index",
        type=int,
        default=None,
        help="Camera view for render test (default: model anchor)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    _init_single_process_distributed()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[device] {device}")

    model, cfg = _load_model_and_cfg(args.config, args.checkpoint, device)
    gs_loss = GSLoss(
        lambda_rgb=float(OmegaConf.select(cfg, "loss.lambda_rgb", default=1.0)),
        lambda_geo=float(OmegaConf.select(cfg, "loss.lambda_geo", default=0.0)),
        use_gt_pose=bool(OmegaConf.select(cfg, "loss.use_gt_pose", default=True)),
        background_color=OmegaConf.to_container(
            OmegaConf.select(cfg, "loss.background_color", default=[0.0, 0.0, 0.0])
        ),
    ).to(device)
    geo = Loss(train_conf=False)

    val_cfg = cfg.data.val
    val_dataset = instantiate(val_cfg, _recursive_=False)
    if hasattr(val_dataset, "seed") and hasattr(cfg, "seed_value"):
        val_dataset.seed = int(cfg.seed_value)
    loader = val_dataset.get_loader(epoch=0)

    batch = None
    for i, b in enumerate(loader):
        if i == args.val_batch_index:
            batch = b
            break
    if batch is None:
        raise IndexError(
            f"val_batch_index={args.val_batch_index} out of range "
            f"(loader has {i + 1} batches)"
        )

    batch = _process_batch(batch)
    batch = copy_data_to_device(batch, device, non_blocking=True)

    print(f"\n[batch] images {tuple(batch['images'].shape)} extrinsics {tuple(batch['extrinsics'].shape)}")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
        pred = model(images=batch["images"])

    if "gs_raw" in pred:
        _diagnose_gs_raw(pred["gs_raw"].float())

    adapter_out = pred.get("gaussian_adapter_out")
    if not adapter_out:
        raise RuntimeError("No gaussian_adapter_out in predictions.")

    cloud_raw = adapter_out[0].cloud
    _diagnose_cloud("adapter output (pre scene-scale)", cloud_raw)

    gt = _pseudo_gt(pred)
    norm_factor = gs_loss._compute_norm_factor(pred, gt)
    print(f"\n  norm_factor (scene scale): {norm_factor.detach().cpu().tolist()}")

    pred_norm = geo.normalize_pred(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in pred.items()}, gt
    )
    cloud_render = gs_loss._prepare_render_cloud(cloud_raw, norm_factor, 0)
    _diagnose_cloud("after scale_gaussian + sanitize (used in render_erp)", cloud_render)

    if gs_loss.use_gt_pose:
        gt_full = gs_loss.geo_loss.prepare_gt(batch)
        poses = gt_full["camera_poses"]
    else:
        poses = pred_norm["camera_poses"]
    poses = gs_loss._scale_camera_poses(poses.float(), norm_factor.to(device))

    b, s, _, h, w = batch["images"].shape
    vi = args.view_index
    if vi is None:
        anchor = pred.get("gs_anchor_idx")
        vi = int(anchor[0].item()) if anchor is not None else s // 2

    bg = gs_loss._bg.to(device=device)
    _diagnose_camera_and_render(
        cloud_render, poses[0], vi, h, w, device, bg
    )

    print(f"\n{'=' * 72}\n  Done.\n{'=' * 72}\n")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
