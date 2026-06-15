#!/usr/bin/env python3
"""
PanoVGGT + 3DGS inference: ERP renders, geometry point clouds, and Gaussian export.

Default resolution: 518×1036 (same as inference.py).

Outputs:
  renders/              ERP RGB renders + inputs
  pointclouds/          PanoVGGT geometry (local per-frame + merged world), same as inference.py
  gaussians/            Gaussian center PLY from GS head
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TextIO, Tuple

import cv2
import numpy as np
import torch
import torchvision
from hydra import compose
from hydra.initialize import initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from panovggt.models.loss_gs import GSLoss
from panovggt.render.odgs_bridge import check_odgs_available, render_erp
from panovggt.utils.erp_loss import l1_erp, psnr_erp, ssim_erp
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from panovggt.utils.gs_ply_export import save_gaussian_splat_ply
from panovggt.utils.lpips_metric import compute_lpips, get_lpips_model, lpips_backend_name

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class ViewMetrics:
    name: str
    l1: float
    ws_psnr: float
    psnr: float
    ssim: float
    lpips: float


def _fmt_psnr(val: float) -> str:
    return "inf" if val == float("inf") else f"{val:.4f}"


def _fmt_ssim(val: float) -> str:
    return f"{val:.4f}"


def _fmt_lpips(val: float) -> str:
    return "nan" if val != val else f"{val:.4f}"


def _mean_metric(values: List[float]) -> float:
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def _write_metrics_table(f: TextIO, records: List[ViewMetrics]) -> None:
    f.write("Per-image metrics:\n")
    if not records:
        f.write("(no images)\n")
        return

    name_w = max(len("image"), *(len(r.name) for r in records))
    f.write(
        f"{'image':<{name_w}}  {'L1':>10}  {'WS-PSNR':>10}  {'PSNR-ERP':>10}  "
        f"{'SSIM':>10}  {'LPIPS':>10}\n"
    )
    for r in records:
        f.write(
            f"{r.name:<{name_w}}  {r.l1:>10.6f}  {_fmt_psnr(r.ws_psnr):>10}  "
            f"{_fmt_psnr(r.psnr):>10}  {_fmt_ssim(r.ssim):>10}  "
            f"{_fmt_lpips(r.lpips):>10}\n"
        )


def _write_mean_metrics(f: TextIO, records: List[ViewMetrics]) -> None:
    f.write("Mean:\n")
    if not records:
        f.write("WS-PSNR: nan dB\nPSNR: nan dB\nSSIM: nan\nLPIPS: nan\n")
        return
    mean_ws = _mean_metric([r.ws_psnr for r in records])
    mean_psnr = _mean_metric([r.psnr for r in records])
    mean_ssim = _mean_metric([r.ssim for r in records])
    mean_lpips = _mean_metric([r.lpips for r in records])
    f.write(f"WS-PSNR: {_fmt_psnr(mean_ws)} dB\n")
    f.write(f"PSNR: {_fmt_psnr(mean_psnr)} dB\n")
    f.write(f"SSIM: {_fmt_ssim(mean_ssim)}\n")
    f.write(f"LPIPS: {_fmt_lpips(mean_lpips)}\n")

# Match inference.py: H=518, W=1036 (2:1 ERP, divisible by patch_size=14)
_INPUT_H = 518
_INPUT_W = 1036


def collect_images(image_dir: str) -> List[str]:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMG_EXTS
    )
    if not paths:
        raise ValueError(f"No images in {image_dir}")
    return paths


def load_images(
    image_paths: List[str], height: int, width: int, device: torch.device
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """Return (1, S, 3, H, W) in [0, 1] and list of RGB uint8 previews."""
    frames = []
    previews = []
    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            raise IOError(f"Cannot read: {p}")
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        previews.append(rgb)
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        frames.append(t)
    batch = torch.stack(frames, dim=0).unsqueeze(0).to(device)
    return batch, previews


def _stack_gt_render_error(
    gt: torch.Tensor, render: torch.Tensor, error_map: torch.Tensor
) -> torch.Tensor:
    """Vertical stack: GT | Render | Error (CHW), same as gs_epoch_infer."""
    return torch.cat([gt, render, error_map], dim=1)


def build_export_style_batch(images: torch.Tensor) -> dict:
    """
    Build a dataloader-style batch for ``GSLoss.prepare_render_bundle``.

    Mirrors SynPano images-only placeholders + ``Trainer._process_batch`` normalization
    used by training export (``gs_epoch_infer``).
    """
    if images.dim() == 4:
        images = images.unsqueeze(0)
    b, s, _, h, w = images.shape
    target_device = images.device

    extrinsics = torch.zeros(b, s, 3, 4, dtype=torch.float32)
    extrinsics[..., :3, :3] = torch.eye(3)

    point_masks = torch.ones(b, s, h, w, dtype=torch.bool)
    depths = torch.ones(b, s, h, w, dtype=torch.float32)
    cam_points = torch.zeros(b, s, h, w, 3, dtype=torch.float32)
    world_points = torch.zeros(b, s, h, w, 3, dtype=torch.float32)

    batch = {
        "images": images.detach().cpu().float(),
        "extrinsics": extrinsics,
        "cam_points": cam_points,
        "world_points": world_points,
        "depths": depths,
        "point_masks": point_masks,
    }

    norm_ext, norm_cam, norm_world, norm_depth, norm_factors = (
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
    batch["norm_factors"] = norm_factors

    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(target_device)
    return batch


def _resolve_checkpoint_global_step(ckpt: dict) -> Optional[int]:
    for key in ("global_step", "step", "iteration"):
        if key in ckpt and ckpt[key] is not None:
            return int(ckpt[key])
    return None


def load_model_and_loss_from_config(
    config_name: str,
    checkpoint_path: str,
    device: str,
    *,
    img_size: int,
    global_step: Optional[int] = None,
):
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

    step = global_step
    if step is None:
        step = _resolve_checkpoint_global_step(ckpt)
    if step is not None and hasattr(model, "set_global_step"):
        model.set_global_step(int(step))
        print(f"[load] model global_step={int(step)}")

    model = model.to(device).eval()
    loss_fn = loss_fn.to(device)
    loss_fn.eval()
    return model, loss_fn, cfg


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
    """Same as inference.py: flatten (H,W,3) geometry + RGB with validity mask."""
    mask = valid_mask if valid_mask is not None else np.ones(points_hw3.shape[:2], dtype=bool)
    depth = np.linalg.norm(points_hw3, axis=-1)
    mask = mask & (depth > 1e-3) & np.isfinite(depth)
    xyz = points_hw3[mask]
    if image_hw3.max() <= 1.0:
        rgb = (image_hw3[mask] * 255).astype(np.uint8)
    else:
        rgb = image_hw3[mask].astype(np.uint8)
    return xyz, rgb


def _tensor_to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    """NumPy does not support bfloat16; cast before export."""
    return t.detach().float().cpu().numpy()


def export_panovggt_geometry_pointclouds(
    pred: dict,
    images: torch.Tensor,
    image_paths: List[str],
    output_dir: str,
    valid_masks: Optional[torch.Tensor] = None,
) -> Tuple[int, int]:
    """
    Export PanoVGGT geometry point clouds (local + merged world), not Gaussians.

    Layout mirrors inference.py:
      pointclouds/per_frame/{stem}.ply  — camera-frame local_points
      pointclouds/merged.ply            — world_points merged over views
    """
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
        local_pts = _tensor_to_numpy_f32(local_pts[0])
    elif torch.is_tensor(local_pts):
        local_pts = _tensor_to_numpy_f32(local_pts)

    if torch.is_tensor(world_pts) and world_pts.dim() == 5:
        world_pts = _tensor_to_numpy_f32(world_pts[0])
    elif torch.is_tensor(world_pts):
        world_pts = _tensor_to_numpy_f32(world_pts)

    s = images.shape[1]
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    per_frame_count = 0

    for vi in range(s):
        stem = Path(image_paths[vi]).stem
        img_rgb = (
            images[0, vi].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        )
        img_rgb = np.clip(img_rgb, 0.0, 1.0)

        mask_np: Optional[np.ndarray] = None
        if valid_masks is not None:
            mask_np = valid_masks[0, vi].detach().cpu().numpy().astype(bool)

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


@torch.no_grad()
def run_gs_inference(
    model,
    loss_fn: GSLoss,
    images: torch.Tensor,
    image_paths: List[str],
    output_dir: str,
    device: torch.device,
) -> None:
    if not check_odgs_available():
        raise RuntimeError("odgs_gaussian_rasterization is required for ERP rendering.")

    render_dir = os.path.join(output_dir, "renders")
    compare_dir = os.path.join(output_dir, "render_gt_images")
    gs_dir = os.path.join(output_dir, "gaussians")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(gs_dir, exist_ok=True)

    if images.dim() == 4:
        images = images.unsqueeze(0)
    batch = build_export_style_batch(images)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with amp_ctx:
        pred = model(images=batch["images"])

    if "gaussian_adapter_out" not in pred:
        raise RuntimeError("Model did not return gaussian_adapter_out; is enable_gaussian=True?")

    gt_geom = pseudo_gt_from_pred(pred)
    n_per_frame, n_merged = export_panovggt_geometry_pointclouds(
        pred,
        batch["images"],
        image_paths,
        output_dir,
        valid_masks=gt_geom.get("valid_masks"),
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

    cloud = loss_fn.prepare_render_cloud_for_batch_item(
        pred_render, gt, adapter_out, 0, norm_factor
    )
    if cloud is None or cloud.get_xyz.numel() == 0:
        raise RuntimeError("Empty Gaussian cloud after masking.")

    save_gaussian_splat_ply(os.path.join(gs_dir, "gaussians.ply"), cloud)
    xyz = cloud.get_xyz.detach().cpu().numpy()

    b, s = imgs.shape[0], imgs.shape[1]
    h, w = imgs.shape[-2], imgs.shape[-1]
    bg = loss_fn._bg.to(device=device, dtype=torch.float32)
    pipe = loss_fn.render_pipe

    anchor = int(pred["gs_anchor_idx"][0].item()) if "gs_anchor_idx" in pred else s // 2
    gs_mode = pred.get("gs_means_mode", "anchor")
    print(
        f"[infer] sequence length S={s}, anchor view={anchor}, "
        f"gs_means_mode={gs_mode}, Gaussians={xyz.shape[0]:,}, "
        f"use_gt_pose={loss_fn.use_gt_pose}, "
        f"axis_align={loss_fn.gs_cross_view_axis_align}"
    )

    get_lpips_model(device)
    backend = lpips_backend_name()
    if backend:
        print(f"[infer] LPIPS backend: {backend}")
    else:
        print("[warn] LPIPS unavailable; install with: pip install lpips")

    metric_records: List[ViewMetrics] = []

    for vi in range(s):
        view_cloud = loss_fn.cloud_for_view_render(cloud, pred_render, 0, vi)
        cam = loss_fn.build_render_camera(
            batch,
            gt,
            pred_render,
            0,
            vi,
            h,
            w,
            device,
        )
        pkg = render_erp(view_cloud, cam, bg, pipe=pipe)
        rendered = pkg["render"].float().clamp(0.0, 1.0)
        target = imgs[0, vi].float().clamp(0.0, 1.0)
        error_map = torch.abs(rendered - target)

        stem = Path(image_paths[vi]).stem
        image_name = Path(image_paths[vi]).name
        out_render = os.path.join(render_dir, f"{stem}_render.png")
        torchvision.utils.save_image(rendered.cpu(), out_render)
        out_gt = os.path.join(render_dir, f"{stem}_input.png")
        torchvision.utils.save_image(target.cpu(), out_gt)

        stack = _stack_gt_render_error(target, rendered, error_map)
        out_compare = os.path.join(compare_dir, f"view{vi:02d}_compare.png")
        torchvision.utils.save_image(stack.cpu(), out_compare)

        l1 = float(l1_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item())
        ws_psnr = psnr_erp(rendered.cpu(), target.cpu())
        psnr = psnr_erp(rendered, target)
        ssim = float(ssim_erp(rendered.unsqueeze(0), target.unsqueeze(0)).item())
        lpips = compute_lpips(rendered, target)
        metric_records.append(
            ViewMetrics(
                name=image_name,
                l1=l1,
                ws_psnr=ws_psnr,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips,
            )
        )
        print(
            f"[infer] {image_name}: L1={l1:.6f}, WS-PSNR={_fmt_psnr(ws_psnr)} dB, "
            f"PSNR-ERP={_fmt_psnr(psnr)} dB, SSIM={_fmt_ssim(ssim)}, LPIPS={_fmt_lpips(lpips)}"
        )

        if vi == anchor:
            torchvision.utils.save_image(
                rendered.cpu(), os.path.join(render_dir, "anchor_render.png")
            )

    meta_path = os.path.join(output_dir, "info.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"images: {len(image_paths)}\n")
        f.write(f"anchor_view: {anchor}\n")
        f.write(f"gs_means_mode: {pred.get('gs_means_mode', 'anchor')}\n")
        f.write(f"use_gt_pose: {loss_fn.use_gt_pose}\n")
        f.write(f"gs_cross_view_axis_align: {loss_fn.gs_cross_view_axis_align}\n")
        f.write(f"num_gaussians: {xyz.shape[0]}\n")
        f.write(f"geometry_per_frame_ply: {n_per_frame}\n")
        f.write(f"geometry_merged_points: {n_merged}\n")
        f.write(f"geometry_ply_dir: {os.path.join(output_dir, 'pointclouds')}\n")
        f.write(f"gaussian_ply: {os.path.join(gs_dir, 'gaussians.ply')}\n")
        f.write(f"compare_dir: {compare_dir}\n")
        f.write(f"resolution: {h}x{w}\n")
        f.write("\n[image_paths]\n")
        for i, p in enumerate(image_paths):
            f.write(f"  [{i}] {p}\n")
        f.write("\n")
        _write_metrics_table(f, metric_records)
        f.write("\n")
        _write_mean_metrics(f, metric_records)
        if metric_records:
            f.write("\n[render_quality_avg]\n")
            f.write(f"  l1: {_mean_metric([r.l1 for r in metric_records]):.6f}\n")
            f.write(f"  psnr: {_mean_metric([r.psnr for r in metric_records]):.4f}\n")
            f.write(f"  ssim: {_mean_metric([r.ssim for r in metric_records]):.6f}\n")
            mean_lpips = _mean_metric([r.lpips for r in metric_records])
            f.write(
                f"  lpips: {_fmt_lpips(mean_lpips)}\n"
                if mean_lpips == mean_lpips
                else "  lpips: nan\n"
            )

    if metric_records:
        mean_ws = _mean_metric([r.ws_psnr for r in metric_records])
        mean_psnr = _mean_metric([r.psnr for r in metric_records])
        mean_ssim = _mean_metric([r.ssim for r in metric_records])
        mean_lpips = _mean_metric([r.lpips for r in metric_records])
        print(
            f"[infer] mean WS-PSNR={_fmt_psnr(mean_ws)} dB, PSNR-ERP={_fmt_psnr(mean_psnr)} dB, "
            f"SSIM={_fmt_ssim(mean_ssim)}, LPIPS={_fmt_lpips(mean_lpips)}"
        )
    print(f"[done] outputs → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PanoVGGT-GS inference")
    parser.add_argument("--config", default="synpano_gs", help="Hydra config name")
    parser.add_argument(
        "--checkpoint",
        default="logs/synpano_gs_0527_1252/ckpts/checkpoint_10.pt",
        help="Training checkpoint (.pt). Avoid checkpoint.pt while training is saving.",
    )
    parser.add_argument("--image_dir", default="examples/apartment")
    parser.add_argument("--output_dir", default="output/apartment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--global_step",
        type=int,
        default=None,
        help="Override model global_step (default: read from checkpoint if present).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        print("[warn] CUDA unavailable, using CPU (slow).")

    image_paths = collect_images(args.image_dir)
    height, width = _INPUT_H, _INPUT_W
    patch = 14
    if height % patch != 0 or width % patch != 0:
        raise ValueError(f"H,W must be divisible by patch_size={patch}")

    model, loss_fn, _ = load_model_and_loss_from_config(
        args.config,
        args.checkpoint,
        str(device),
        img_size=height,
        global_step=args.global_step,
    )
    print(f"[infer] {len(image_paths)} images → {height}x{width} (fixed, same as inference.py)")
    images, _ = load_images(image_paths, height, width, device)
    run_gs_inference(model, loss_fn, images, image_paths, args.output_dir, device)


if __name__ == "__main__":
    main()
