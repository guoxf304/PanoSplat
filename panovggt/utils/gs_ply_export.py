"""Export TensorGaussianCloud to standard 3DGS binary PLY (Supersplat / Inria compatible)."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch

from panovggt.render.odgs_bridge import TensorGaussianCloud


def _cloud_arrays(cloud: TensorGaussianCloud) -> Tuple[np.ndarray, ...]:
    """Extract PLY columns in Inria 3DGS layout (all float32)."""
    xyz = cloud._xyz.detach().float().cpu().numpy()
    n = int(xyz.shape[0])
    if n == 0:
        raise ValueError("Cannot export empty Gaussian cloud.")

    normals = np.zeros((n, 3), dtype=np.float32)

    f_dc_t = cloud._features_dc.detach().float().cpu().numpy()
    # (N, 1, 3) -> (N, 3) same as Inria transpose(1,2).flatten(start_dim=1)
    f_dc = np.transpose(f_dc_t, (0, 2, 1)).reshape(n, -1).astype(np.float32)

    f_rest_t = cloud._features_rest.detach().float().cpu().numpy()
    if f_rest_t.size > 0:
        f_rest = np.transpose(f_rest_t, (0, 2, 1)).reshape(n, -1).astype(np.float32)
    else:
        f_rest = np.empty((n, 0), dtype=np.float32)

    # Standard 3DGS PLY stores logit opacity and log-scale (renderers apply sigmoid/exp).
    opacity = cloud._opacity.detach().float().cpu().numpy().reshape(n, -1).astype(np.float32)
    scale = cloud._scaling.detach().float().cpu().numpy().reshape(n, -1).astype(np.float32)
    rotation = cloud._rotation.detach().float().cpu().numpy().reshape(n, -1).astype(np.float32)

    if scale.shape[1] != 3:
        raise ValueError(f"Expected 3 scale channels, got shape {scale.shape}")
    if rotation.shape[1] != 4:
        raise ValueError(f"Expected 4 rotation channels (WXYZ), got shape {rotation.shape}")
    if opacity.shape[1] != 1:
        opacity = opacity.reshape(n, 1)

    return xyz.astype(np.float32), normals, f_dc, f_rest, opacity, scale, rotation


def _property_names(f_dc_cols: int, f_rest_cols: int) -> List[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{i}" for i in range(f_dc_cols))
    names.extend(f"f_rest_{i}" for i in range(f_rest_cols))
    names.append("opacity")
    names.extend(f"scale_{i}" for i in range(3))
    names.extend(f"rot_{i}" for i in range(4))
    return names


def save_gaussian_splat_ply(path: str, cloud: TensorGaussianCloud) -> int:
    """
    Write a binary little-endian 3DGS PLY readable by Supersplat / Inria viewers.

    Property order: x,y,z, nx,ny,nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*.
    opacity = logit; scale = log; rotation = quaternion WXYZ (rot_0..3).
    """
    xyz, normals, f_dc, f_rest, opacity, scale, rotation = _cloud_arrays(cloud)
    n = xyz.shape[0]
    props = _property_names(f_dc.shape[1], f_rest.shape[1])

    columns = [xyz, normals, f_dc, f_rest, opacity, scale, rotation]
    data = np.concatenate(columns, axis=1).astype(np.float32, copy=False)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    header_lines.extend(f"property float {name}" for name in props)
    header_lines.append("end_header\n")
    header = "\n".join(header_lines)

    dtype = [(name, "<f4") for name in props]
    structured = np.empty(n, dtype=dtype)
    for i, name in enumerate(props):
        structured[name] = data[:, i]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(structured.tobytes())

    print(f"[ply] {n:,} Gaussians (3DGS) → {path}")
    return n


def save_gaussian_splat_ply_from_tensors(
    path: str,
    xyz: torch.Tensor,
    opacity_logit: torch.Tensor,
    scaling_log: torch.Tensor,
    rotation_wxyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    sh_degree: int = 0,
) -> int:
    """Convenience wrapper when a TensorGaussianCloud is not available."""
    cloud = TensorGaussianCloud(
        xyz=xyz,
        scaling=scaling_log,
        rotation=rotation_wxyz,
        opacity=opacity_logit,
        features_dc=features_dc,
        features_rest=features_rest,
        sh_degree=sh_degree,
    )
    return save_gaussian_splat_ply(path, cloud)
