"""Coordinate-frame utilities for ERP GS rendering (cam0 / ODGS alignment)."""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cross-view axis alignment test (before render_erp)
# Override order: env PANOSPLAT_GS_AXIS_ALIGN > loss.gs_cross_view_axis_align > default
#
#   "none"     - no change
#   "flip_yz"  - try A: flip Y and Z (OpenCV +Z forward vs OpenGL -Z forward)
#   "flip_xz"  - try B: flip X and Z (left-right / forward-back swap test)
#
# Applied only when anchor_idx != render_view_idx unless cross_view_only=False.
# ---------------------------------------------------------------------------
AxisAlignMode = Literal["none", "flip_yz", "flip_xz"]
_DEFAULT_AXIS_ALIGN: AxisAlignMode = "none"


def resolve_axis_align_mode(config_value: Optional[str] = None) -> AxisAlignMode:
    """Resolve axis-align mode from env, yaml, or module default."""
    raw = os.environ.get("PANOSPLAT_GS_AXIS_ALIGN", config_value or _DEFAULT_AXIS_ALIGN)
    raw = str(raw).strip().lower()
    if raw in ("a", "flip_yz", "yz", "opencv_opengl"):
        return "flip_yz"
    if raw in ("b", "flip_xz", "xz"):
        return "flip_xz"
    if raw in ("none", "off", "0", ""):
        return "none"
    return "none"


def erp_ray_convention_note() -> str:
    """
    ERP ray convention shared by dataset, PanoVGGT, and ODGS rasterizer.

    Spherical mapping (center pixel looks at +Z):
      theta_h = (u - 0.5) * 2pi   # azimuth, matches dataset_util ``theta``
      phi_v   = -(v - 0.5) * pi   # elevation, matches dataset_util ``phi``

    Cartesian (OpenCV panorama: X-right, Y-down, Z-forward):
      x = cos(phi_v) * sin(theta_h)
      y = -sin(phi_v)
      z = cos(phi_v) * cos(theta_h)

    ODGS ``forward.cu`` uses ``lon = atan2(x, z)``, ``lat = atan2(-y, dist_xz)``,
    i.e. the same +Z forward / Y-down frame as ``unproject_pano_depth_to_camera_coords``.
    """
    return (
        "ERP rays: OpenCV (+X right, +Y down, +Z forward); "
        "ODGS lon=atan2(x,z), lat=atan2(-y,sqrt(x^2+z^2))"
    )


def apply_axis_alignment_test(
    xyz: torch.Tensor,
    mode: Union[AxisAlignMode, str],
) -> torch.Tensor:
    """
    Hard-coded axis reflection tests on Gaussian centers (world xyz).

    try A (flip_yz):
        xyz[..., 1] *= -1; xyz[..., 2] *= -1
    try B (flip_xz):
        xyz[..., 0] *= -1; xyz[..., 2] *= -1
    """
    if mode is None or mode == "none":
        return xyz
    out = xyz.clone()
    if mode == "flip_yz":
        out[..., 1] = -out[..., 1]
        out[..., 2] = -out[..., 2]
    elif mode == "flip_xz":
        out[..., 0] = -out[..., 0]
        out[..., 2] = -out[..., 2]
    return out


def apply_cross_view_xyz_alignment(
    xyz: torch.Tensor,
    *,
    anchor_idx: int,
    view_idx: int,
    mode: Union[AxisAlignMode, str] = "none",
    cross_view_only: bool = True,
) -> torch.Tensor:
    """Apply axis-align test when rendering a non-anchor camera view."""
    if cross_view_only and int(anchor_idx) == int(view_idx):
        return xyz
    return apply_axis_alignment_test(xyz, mode)


def apply_axis_alignment_to_quaternion(
    rotation: torch.Tensor,
    mode: Union[AxisAlignMode, str],
) -> torch.Tensor:
    """
    Conjugate Gaussian rotation under axis reflection S (optional, best-effort).

    R' = S @ R @ S  for reflection matrices diag(±1).
    """
    if mode is None or mode == "none" or rotation.numel() == 0:
        return rotation
    if mode == "flip_yz":
        s = rotation.new_tensor([1.0, -1.0, -1.0])
    elif mode == "flip_xz":
        s = rotation.new_tensor([-1.0, 1.0, -1.0])
    else:
        return rotation

    q = F.normalize(rotation.float(), dim=-1, eps=1e-6)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    r = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    sd = s.view(1, 3)
    r2 = sd.unsqueeze(-1) * r * sd.unsqueeze(-2)
    tr = r2[..., 0, 0] + r2[..., 1, 1] + r2[..., 2, 2]
    q2 = torch.zeros_like(q)
    q2[..., 0] = torch.sqrt((1.0 + tr).clamp(min=1e-8)) * 0.5
    denom = (4.0 * q2[..., 0]).clamp(min=1e-8)
    q2[..., 1] = (r2[..., 2, 1] - r2[..., 1, 2]) / denom
    q2[..., 2] = (r2[..., 0, 2] - r2[..., 2, 0]) / denom
    q2[..., 3] = (r2[..., 1, 0] - r2[..., 0, 1]) / denom
    return F.normalize(q2, dim=-1, eps=1e-6).to(dtype=rotation.dtype)


def prepare_render_gaussian_xyz(
    xyz: torch.Tensor,
    rotation: Optional[torch.Tensor],
    *,
    anchor_idx: int,
    view_idx: int,
    axis_align_mode: Union[AxisAlignMode, str] = "none",
    cross_view_only: bool = True,
    align_rotation: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Final xyz (+ optional quat) adjustment immediately before render_erp."""
    mode = resolve_axis_align_mode(axis_align_mode)
    xyz_out = apply_cross_view_xyz_alignment(
        xyz,
        anchor_idx=anchor_idx,
        view_idx=view_idx,
        mode=mode,
        cross_view_only=cross_view_only,
    )
    rot_out = rotation
    if align_rotation and rotation is not None and mode != "none":
        if not cross_view_only or int(anchor_idx) != int(view_idx):
            rot_out = apply_axis_alignment_to_quaternion(rotation, mode)
    return xyz_out, rot_out


def resolve_scene_scale(
    pred: dict,
    gt: dict,
    gt_raw: Optional[dict] = None,
) -> torch.Tensor:
    """Scene scale for render normalization (prefer dataloader avg_scale)."""
    if gt_raw is not None and gt_raw.get("norm_factors") is not None:
        return gt_raw["norm_factors"].float().clamp(min=1e-6)
    local_points = pred["local_points"]
    masks = gt["valid_masks"]
    b = local_points.shape[0]
    all_pts = local_points.clone()
    all_pts[~masks] = 0
    all_pts = all_pts.reshape(b, -1, 3)
    all_dis = all_pts.norm(dim=-1)
    denom = masks.float().sum(dim=[-1, -2, -3]).clamp(min=1e-8)
    return (all_dis.sum(dim=[-1, -2]) / denom).clamp(min=1e-3, max=1e3)


def local_points_to_world(
    local_points: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    """
    Map per-pixel camera-frame points to world / cam0-normalized frame.

    Args:
        local_points: (H, W, 3) or (B, H, W, 3)
        c2w: (4, 4) or (B, 4, 4)
    """
    if local_points.dim() == 3:
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        return torch.einsum("ij,hwj->hwi", R, local_points) + t
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    return torch.einsum("bij,bhwj->bhwi", R, local_points) + t.unsqueeze(1).unsqueeze(1)


def align_global_points_to_cam0(
    global_points: torch.Tensor,
    c2w0: torch.Tensor,
) -> torch.Tensor:
    """
    Same cam0 re-rooting as ``Loss.normalize_pred`` (world -> camera-0 frame).

    global_points: (B, S, H, W, 3)
    c2w0: (B, 4, 4)
    """
    R0 = c2w0[:, :3, :3]
    t0 = c2w0[:, :3, 3]
    t_w2c0 = -torch.einsum("bij,bj->bi", R0, t0)
    return (
        torch.einsum("bshwc,bc->bshw", global_points, R0)
        + t_w2c0.view(-1, 1, 1, 1, 3)
    )


def subsample_hw(points_hw: torch.Tensor, stride: int) -> torch.Tensor:
    if stride <= 1:
        return points_hw
    return points_hw[::stride, ::stride]


def w2c_to_world_view_transform(w2c: torch.Tensor) -> torch.Tensor:
    """Build ODGS ``world_view_transform`` from w2c (SynPano / batch convention)."""
    if w2c.shape[-2:] == (3, 4):
        bottom = w2c.new_zeros(*w2c.shape[:-2], 1, 4)
        bottom[..., 0, 3] = 1.0
        w2c = torch.cat([w2c, bottom], dim=-2)
    return w2c.transpose(-1, -2).contiguous()


def camera_center_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    if w2c.shape[-2:] == (3, 4):
        bottom = w2c.new_zeros(*w2c.shape[:-2], 1, 4)
        bottom[..., 0, 3] = 1.0
        w2c = torch.cat([w2c, bottom], dim=-2)
    c2w = torch.linalg.inv(w2c)
    return c2w[..., :3, 3]


def prepare_view_world_means(
    pred: dict,
    gt: dict,
    batch_idx: int,
    view_idx: int,
    *,
    use_gt_pose: bool,
) -> torch.Tensor:
    """
    World-frame Gaussian centers for one view, aligned with render cameras.

    When ``use_gt_pose``, prefer batch-normalized ``world_points`` (same frame as
    normalized ``extrinsics``).  Otherwise use ``local_points @ c2w`` or cam0-aligned
    ``global_points``.
    """
    if use_gt_pose and gt.get("global_points") is not None:
        return gt["global_points"][batch_idx, view_idx].float()

    if use_gt_pose:
        c2w = gt["camera_poses"][batch_idx, view_idx]
    else:
        c2w = pred["camera_poses"][batch_idx, view_idx]

    if pred.get("local_points") is not None:
        lp = pred["local_points"][batch_idx, view_idx].float()
        return local_points_to_world(lp, c2w.float())

    gp = pred["global_points"].float()
    c2w0 = gt["camera_poses"][batch_idx, 0] if use_gt_pose else pred["camera_poses"][batch_idx, 0]
    gp_cam0 = align_global_points_to_cam0(
        gp.unsqueeze(0), c2w0.unsqueeze(0)
    ).squeeze(0)
    return gp_cam0[view_idx]


def prepare_anchor_world_means(
    pred: dict,
    gt: dict,
    batch_idx: int,
    *,
    use_gt_pose: bool,
) -> torch.Tensor:
    """World-frame Gaussian centers at the GS anchor view."""
    anchor = int(pred["gs_anchor_idx"][batch_idx].item())
    return prepare_view_world_means(
        pred, gt, batch_idx, anchor, use_gt_pose=use_gt_pose
    )


def prepare_merged_world_means(
    pred: dict,
    gt: dict,
    batch_idx: int,
    *,
    use_gt_pose: bool,
    stride: int = 1,
) -> Optional[torch.Tensor]:
    """Stack all views' world means (same layout as merged geometry PLY)."""
    s = pred["global_points"].shape[1]
    masks = gt.get("valid_masks")
    parts: List[torch.Tensor] = []
    for vi in range(s):
        means_hw = prepare_view_world_means(
            pred, gt, batch_idx, vi, use_gt_pose=use_gt_pose
        )
        means_hw = subsample_hw(means_hw, stride)
        valid = torch.isfinite(means_hw).all(dim=-1)
        if masks is not None:
            vm = masks[batch_idx, vi].bool()
            if stride > 1:
                vm = vm[::stride, ::stride]
            valid = valid & vm
        if valid.any():
            parts.append(means_hw[valid])
    if not parts:
        return None
    return torch.cat(parts, dim=0)
