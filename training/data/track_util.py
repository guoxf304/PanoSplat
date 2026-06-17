# Copyright (c) Meta Platforms, Inc. and affiliates.
# Panoramic track utilities (adapted from VGGT training/data/track_util.py).

from __future__ import annotations

import logging
import math

import torch

def get_depth_inside_flag(depths, batch_indices, uv_int, proj_depths, rel_thres):
    sampled_depths = depths[batch_indices, uv_int[..., 1], uv_int[..., 0]]
    depth_diff = (proj_depths - sampled_depths).abs()
    return torch.logical_and(
        depth_diff < (proj_depths * rel_thres),
        depth_diff < (sampled_depths * rel_thres),
    )


def sample_positive_tracks(tracks, tracks_mask, track_num, half_top=True, seq_name=None):
    tracks_mask = tracks_mask.clone()
    tracks_mask[:, tracks_mask[0] == False] = False
    track_frame_num = tracks_mask.sum(dim=0)
    tracks_mask[:, track_frame_num <= 1] = False
    track_frame_num = tracks_mask.sum(dim=0)
    _, track_num_sort_idx = track_frame_num.sort(descending=True)
    if half_top and len(track_num_sort_idx) // 2 > track_num:
        track_num_sort_idx = track_num_sort_idx[: len(track_num_sort_idx) // 2]
    pick_idx = torch.randperm(len(track_num_sort_idx), device=tracks.device)[:track_num]
    track_num_sort_idx = track_num_sort_idx[pick_idx]
    tracks = tracks[:, track_num_sort_idx].clone()
    tracks_mask = tracks_mask[:, track_num_sort_idx].clone()
    return tracks, tracks_mask.bool()


def _stack_sequence(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors
    return torch.stack([t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors])


def project_world_points_to_erp(
    world_points: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    image_shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points to ERP pixel coordinates.

    Args:
        world_points: (P, 3)
        extrinsic_w2c: (3, 4) world-to-camera
        image_shape: (H, W)

    Returns:
        uv: (P, 2) pixel coordinates (u, v)
        cam_points: (P, 3) points in camera frame
    """
    h, w = image_shape
    device = world_points.device
    R = extrinsic_w2c[:3, :3].to(device=device, dtype=world_points.dtype)
    t = extrinsic_w2c[:3, 3].to(device=device, dtype=world_points.dtype)
    cam_points = (world_points @ R.T) + t
    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    theta = torch.atan2(x, z)
    norm = cam_points.norm(dim=-1).clamp(min=1e-8)
    phi = -torch.asin((y / norm).clamp(-1.0, 1.0))
    u = (theta / (2 * math.pi) + 0.5) * (w - 1)
    v = (-phi / math.pi + 0.5) * (h - 1)
    uv = torch.stack([u, v], dim=-1)
    return uv, cam_points


def build_tracks_by_depth_pano(
    extrinsics,
    world_points,
    depths,
    point_masks,
    images,
    pos_rel_thres=0.05,
    boundary_thres=4,
    target_track_num=512,
    neg_ratio=0.0,
    neg_sample_size_ratio=0.5,
    seq_name=None,
):
    """Build ERP point tracks by reprojecting query-frame 3D points across views."""
    extrinsics = _stack_sequence(extrinsics).float()
    world_points = _stack_sequence(world_points).float()
    depths = _stack_sequence(depths).float()
    if depths.dim() == 4:
        depths = depths.squeeze(1)
    point_masks = _stack_sequence(point_masks).bool()

    b, h, w, _ = world_points.shape
    device = world_points.device

    query_world_points = world_points[0]
    query_point_masks = point_masks[0]

    if query_point_masks.sum() == 0:
        logging.warning(f"No valid query points in {seq_name}")
        empty_tracks = torch.zeros(b, target_track_num, 2, device=device)
        empty_mask = torch.zeros(b, target_track_num, device=device, dtype=torch.bool)
        pos_mask = torch.zeros(target_track_num, device=device, dtype=torch.bool)
        return empty_tracks, empty_mask, pos_mask

    valid_query_points = query_world_points[query_point_masks]
    image_points_list = []
    proj_depths_list = []
    for frame_idx in range(b):
        uv, cam_pts = project_world_points_to_erp(
            valid_query_points, extrinsics[frame_idx], (h, w)
        )
        image_points_list.append(uv)
        proj_depths_list.append(cam_pts[:, 2])
    image_points = torch.stack(image_points_list, dim=0)
    proj_depths = torch.stack(proj_depths_list, dim=0)

    uv_int = image_points.floor().long().clone()
    uv_inside_flag = (
        (uv_int[..., 0] >= boundary_thres)
        & (uv_int[..., 0] < (w - boundary_thres))
        & (uv_int[..., 1] >= boundary_thres)
        & (uv_int[..., 1] < (h - boundary_thres))
    )
    uv_int[~uv_inside_flag] = 0
    batch_indices = torch.arange(b, device=device).view(b, 1).expand(-1, uv_int.shape[1])

    depth_inside_flag = None
    for shift in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        cur_uv_int = uv_int + torch.tensor(shift, device=device)
        cur_flag = get_depth_inside_flag(
            depths, batch_indices, cur_uv_int, proj_depths, pos_rel_thres
        )
        depth_inside_flag = cur_flag if depth_inside_flag is None else torch.logical_or(
            depth_inside_flag, cur_flag
        )

    positive_tracks = image_points
    positive_vis_masks = torch.logical_and(uv_inside_flag, depth_inside_flag)

    sampled_neg_track_num = target_track_num * 4
    perb_range = [int(w * neg_sample_size_ratio), int(h * neg_sample_size_ratio)]
    us = torch.randint(0, w, (1, sampled_neg_track_num), device=device)
    vs = torch.randint(0, h, (1, sampled_neg_track_num), device=device)
    neg_query_uvs = torch.stack([us, vs], dim=-1).expand(b, -1, -1)
    delta_us = torch.rand(b, sampled_neg_track_num, device=device) * perb_range[0]
    delta_vs = torch.rand(b, sampled_neg_track_num, device=device) * perb_range[1]
    delta_us[0] = 0
    delta_vs[0] = 0
    negative_tracks = neg_query_uvs + torch.stack([delta_us, delta_vs], dim=-1)

    final_tracks = torch.zeros(b, target_track_num, 2, device=device)
    final_vis_masks = torch.zeros(b, target_track_num, device=device, dtype=torch.bool)
    final_pos_masks = torch.zeros(target_track_num, device=device, dtype=torch.bool)

    target_pos_track_num = target_track_num - int(target_track_num * neg_ratio)
    sampled_positive_tracks, sampled_positive_vis_masks = sample_positive_tracks(
        positive_tracks, positive_vis_masks, target_pos_track_num, seq_name=seq_name
    )
    sampled_pos_track_num = sampled_positive_tracks.shape[1]
    final_tracks[:, :sampled_pos_track_num] = sampled_positive_tracks
    final_vis_masks[:, :sampled_pos_track_num] = sampled_positive_vis_masks
    final_pos_masks[:sampled_pos_track_num] = True

    target_neg_track_num = target_track_num - sampled_pos_track_num
    if negative_tracks.shape[1] > 0 and target_neg_track_num > 0:
        rand_indices = torch.randperm(negative_tracks.shape[1], device=device)
        sampled_neg_tracks = negative_tracks[:, rand_indices[:target_neg_track_num]]
        sampled_neg_track_num = sampled_neg_tracks.shape[1]
        final_tracks[
            :, sampled_pos_track_num : sampled_pos_track_num + sampled_neg_track_num
        ] = sampled_neg_tracks

    return final_tracks, final_vis_masks, final_pos_masks
