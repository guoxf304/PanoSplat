#!/usr/bin/env python3
"""
Gaussian-aware geometry diagnosis for 3DGS PLY comparison.

Loads xyz + opacity + scale, hard-filters invisible Gaussians (opacity > threshold),
then reports physics-aligned metrics: Raw Chamfer, mean max scale, anisotropy,
and optional local geometry on visible centers only.

Example:
  python X_debug/compare_pure_point_clouds.py \\
      --ours     output/inference/apartment/gaussians/gaussians.ply \\
      --baseline /path/to/panosplatt3r/apartment/gaussians.ply \\
      --output   X_debug/info.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

try:
    import open3d as o3d

    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

_PLY_TYPE_MAP = {
    "char": ("i1", 1),
    "int8": ("i1", 1),
    "uchar": ("u1", 1),
    "uint8": ("u1", 1),
    "short": ("i2", 2),
    "int16": ("i2", 2),
    "ushort": ("u2", 2),
    "uint16": ("u2", 2),
    "int": ("i4", 4),
    "int32": ("i4", 4),
    "uint": ("u4", 4),
    "uint32": ("u4", 4),
    "float": ("f4", 4),
    "float32": ("f4", 4),
    "double": ("f8", 8),
    "float64": ("f8", 8),
}

_OPACITY_CANDIDATES = ("opacity", "opacity_raw", "density", "opac", "alpha")
_SCALE_NAME_SETS: Tuple[Tuple[str, ...], ...] = (
    ("scale_0", "scale_1", "scale_2"),
    ("scale_x", "scale_y", "scale_z"),
    ("scaling_0", "scaling_1", "scaling_2"),
    ("log_scale_0", "log_scale_1", "log_scale_2"),
)


def _parse_ply_header(path: Path) -> Tuple[str, int, List[Tuple[str, str]], int]:
    with path.open("rb") as f:
        raw = f.read(131072)

    text_lines: List[str] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        text_lines.append(line.decode("ascii", errors="replace").strip())
        offset += len(line)
        if text_lines[-1] == "end_header":
            break
    else:
        raise ValueError(f"Invalid PLY (no end_header): {path}")

    if not text_lines or text_lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    fmt = None
    n_vert = None
    props: List[Tuple[str, str]] = []
    in_vertex = False
    for line in text_lines[1:]:
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex "):
            n_vert = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex = False
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))
        elif line == "end_header":
            break

    if fmt is None or n_vert is None or not props:
        raise ValueError(f"Malformed PLY header: {path}")
    return fmt, n_vert, props, offset


def _header_lines(path: Path) -> int:
    with path.open("rb") as f:
        n = 0
        while True:
            line = f.readline()
            n += 1
            if line.strip() == b"end_header":
                break
    return n


def _pick_property(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _pick_scale_names(names: Sequence[str]) -> Optional[Tuple[str, str, str]]:
    lower = {n.lower(): n for n in names}
    for group in _SCALE_NAME_SETS:
        if all(g in lower for g in group):
            return lower[group[0]], lower[group[1]], lower[group[2]]
    return None


def _activate_opacity(raw: np.ndarray, prop_name: str) -> Tuple[np.ndarray, str]:
    """Map stored opacity to [0, 1] render alpha."""
    t = torch.from_numpy(raw.astype(np.float32))
    name = prop_name.lower()

    if "logit" in name or name.endswith("_raw"):
        out = torch.sigmoid(t)
        return out.numpy(), "sigmoid(logit)"

    if "density" in name:
        out = torch.sigmoid(t)
        return out.numpy(), "sigmoid(density)"

    # Standard 3DGS PLY: logit unless clearly already activated.
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        raise ValueError("Opacity field has no finite values.")

    in_unit = (finite.min() >= 0.0) and (finite.max() <= 1.0)
    looks_logit = (finite.min() < 0.0) or (finite.max() > 1.0) or (
        in_unit and float(finite.median()) < 0.15 and float(finite.std()) > 0.08
    )
    if looks_logit or name == "opacity":
        out = torch.sigmoid(t)
        return out.numpy(), "sigmoid(3dgs_logit)"

    return t.clamp(0.0, 1.0).numpy(), "linear_clamped"


def _activate_scale(raw: np.ndarray, prop_names: Tuple[str, str, str]) -> Tuple[np.ndarray, str]:
    """Map stored scales to linear axis half-lengths."""
    t = torch.from_numpy(raw.astype(np.float32))
    names = [n.lower() for n in prop_names]

    if any("log" in n for n in names):
        out = torch.exp(t)
        return out.clamp(min=1e-8).numpy(), "exp(log_name)"

    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        raise ValueError("Scale field has no finite values.")

    # Inria / ODGS export stores log-scale when median is negative.
    if float(finite.median()) < 0.0:
        out = torch.exp(t)
        return out.clamp(min=1e-8).numpy(), "exp(3dgs_log_scale)"

    if float(finite.max()) <= 1.0 and float(finite.min()) >= 0.0:
        return t.clamp(min=1e-8).numpy(), "linear"

    out = F.softplus(t)
    return out.clamp(min=1e-8).numpy(), "softplus"


@dataclass
class LoadedGaussianCloud:
    xyz: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    n_ply_vertices: int
    n_valid: int
    n_filtered_nonfinite: int
    opacity_prop: str
    scale_props: Tuple[str, str, str]
    opacity_activation: str
    scale_activation: str
    meta: Dict[str, str] = field(default_factory=dict)


def load_ply_gaussian(path: str) -> LoadedGaussianCloud:
    """Load Gaussian PLY with xyz, activated opacity, and activated scale."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    fmt, n_vert, props, header_end = _parse_ply_header(p)
    names = [name for _, name in props]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PLY missing '{axis}': {path}")

    opacity_name = _pick_property(names, _OPACITY_CANDIDATES)
    scale_names = _pick_scale_names(names)
    if opacity_name is None:
        raise ValueError(
            f"PLY missing opacity/density field: {path}\n"
            f"  found properties: {names[:12]}{'…' if len(names) > 12 else ''}"
        )
    if scale_names is None:
        raise ValueError(
            f"PLY missing scale_0/1/2 (or scale_x/y/z): {path}"
        )

    if fmt == "ascii":
        data = np.loadtxt(p, skiprows=_header_lines(p), max_rows=n_vert, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        name_to_col = {n: i for i, n in enumerate(names)}
        xyz = data[:, [name_to_col["x"], name_to_col["y"], name_to_col["z"]]].astype(np.float32)
        op_raw = data[:, name_to_col[opacity_name]].astype(np.float32)
        sc_raw = data[:, [name_to_col[n] for n in scale_names]].astype(np.float32)
    elif fmt == "binary_little_endian":
        dtype_fields = []
        for ptype, pname in props:
            if ptype not in _PLY_TYPE_MAP:
                raise ValueError(f"Unsupported PLY type '{ptype}' in {path}")
            np_code, _ = _PLY_TYPE_MAP[ptype]
            dtype_fields.append((pname, np_code))
        vertex_dtype = np.dtype(dtype_fields)
        with p.open("rb") as f:
            f.seek(header_end)
            blob = f.read(n_vert * vertex_dtype.itemsize)
        arr = np.frombuffer(blob, dtype=vertex_dtype, count=n_vert)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
        op_raw = arr[opacity_name].astype(np.float32)
        sc_raw = np.stack([arr[n] for n in scale_names], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PLY format '{fmt}': {path}")

    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(op_raw)
        & np.isfinite(sc_raw).all(axis=1)
    )
    n_valid = int(finite.sum())
    if n_valid == 0:
        raise ValueError(f"No finite Gaussian vertices in {path}")

    xyz = xyz[finite]
    op_act, op_mode = _activate_opacity(op_raw[finite], opacity_name)
    sc_act, sc_mode = _activate_scale(sc_raw[finite], scale_names)

    return LoadedGaussianCloud(
        xyz=xyz,
        opacity=op_act.astype(np.float32),
        scale=sc_act.astype(np.float32),
        n_ply_vertices=n_vert,
        n_valid=n_valid,
        n_filtered_nonfinite=n_vert - n_valid,
        opacity_prop=opacity_name,
        scale_props=scale_names,
        opacity_activation=op_mode,
        scale_activation=sc_mode,
    )


def filter_visible_gaussians(
    cloud: LoadedGaussianCloud,
    min_opacity: float,
) -> Tuple[LoadedGaussianCloud, Dict[str, int]]:
    """Hard prune Gaussians with opacity <= min_opacity."""
    mask = cloud.opacity > min_opacity
    n_invisible = int((~mask).sum())
    n_visible = int(mask.sum())
    if n_visible == 0:
        raise ValueError(
            f"No visible Gaussians after opacity > {min_opacity} filter "
            f"(invisible={n_invisible:,})."
        )

    filtered = LoadedGaussianCloud(
        xyz=cloud.xyz[mask],
        opacity=cloud.opacity[mask],
        scale=cloud.scale[mask],
        n_ply_vertices=cloud.n_ply_vertices,
        n_valid=cloud.n_valid,
        n_filtered_nonfinite=cloud.n_filtered_nonfinite,
        opacity_prop=cloud.opacity_prop,
        scale_props=cloud.scale_props,
        opacity_activation=cloud.opacity_activation,
        scale_activation=cloud.scale_activation,
        meta=dict(cloud.meta),
    )
    stats = {
        "n_ply_vertices": cloud.n_ply_vertices,
        "n_finite": cloud.n_valid,
        "n_invisible": n_invisible,
        "n_visible": n_visible,
        "min_opacity": min_opacity,
    }
    return filtered, stats


# ---------------------------------------------------------------------------
# Alignment (optional, for aligned local geometry only — NOT for Raw Chamfer)
# ---------------------------------------------------------------------------

def _umeyama_sim3(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    X = source.T
    Y = target.T
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    s = float(np.trace(np.diag(D) @ S) / max(var_x, 1e-12))
    R = U @ S @ Vt
    t = (mu_y - s * R @ mu_x).reshape(3)
    return s, R.astype(np.float64), t.astype(np.float64)


def _apply_sim3(xyz: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (xyz @ R.T) + t).astype(np.float32)


def _voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    vox = np.floor(xyz / max(voxel_size, 1e-8)).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def _estimate_sim3_nn(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float = 0.05,
    max_source_pts: int = 10000,
    seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src = _voxel_downsample(source, voxel_size)
    tgt = _voxel_downsample(target, voxel_size)
    if src.shape[0] > max_source_pts:
        rng = np.random.default_rng(seed)
        src = src[rng.choice(src.shape[0], max_source_pts, replace=False)]
    if tgt.shape[0] == 0 or src.shape[0] == 0:
        raise ValueError("对齐失败：降采样后点云为空。")
    tree = cKDTree(tgt)
    _, nn_idx = tree.query(src, k=1, workers=-1)
    return _umeyama_sim3(src, tgt[nn_idx])


def align_xyz(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "icp",
    icp_voxel: float = 0.05,
    icp_max_corr: float = 0.25,
    icp_iters: int = 50,
    match_points: int = 10000,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if method == "none":
        return source.copy(), {"method": "none", "fitness": 1.0, "rmse": 0.0}

    if method == "umeyama":
        s, R, t = _estimate_sim3_nn(source, target, icp_voxel, match_points, seed)
        return _apply_sim3(source, s, R, t), {"method": "umeyama_sim3_nn", "scale": s}

    if method == "icp" and _HAS_O3D:
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source.astype(np.float64)))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target.astype(np.float64)))
        src_ds = src.voxel_down_sample(max(icp_voxel, 1e-4))
        tgt_ds = tgt.voxel_down_sample(max(icp_voxel, 1e-4))
        src_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 4, max_nn=30)
        )
        tgt_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 4, max_nn=30)
        )
        reg = o3d.pipelines.registration.registration_icp(
            src_ds,
            tgt_ds,
            icp_max_corr,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iters),
        )
        T = reg.transformation
        ones = np.ones((source.shape[0], 1), dtype=np.float64)
        aligned = (np.hstack([source.astype(np.float64), ones]) @ T.T)[:, :3].astype(np.float32)
        return aligned, {"method": "icp", "fitness": float(reg.fitness), "rmse": float(reg.inlier_rmse)}

    s, R, t = _estimate_sim3_nn(source, target, icp_voxel, match_points, seed)
    return _apply_sim3(source, s, R, t), {"method": "umeyama_sim3_nn", "scale": s}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class GaussianMetrics:
    filter_stats: Dict[str, int]
    opacity_activation: str
    scale_activation: str
    active_effective_points: int
    invisible_zombie_points: int
    mean_opacity_visible: float
    raw_chamfer_l1: float
    raw_chamfer_ours_to_base: float
    raw_chamfer_base_to_ours: float
    mean_max_scale: float
    median_max_scale: float
    anisotropy_ratio_gt_threshold: float
    anisotropy_threshold: float
    mean_roughness: float
    sor_std: float
    sparse_voxel_ratio: float


def raw_chamfer_l1(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Symmetric mean L1 Chamfer without alignment. Returns (l1, a→b, b→a)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("Chamfer input empty.")
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_ab, _ = tree_b.query(a, k=1, workers=-1)
    d_ba, _ = tree_a.query(b, k=1, workers=-1)
    m_ab = float(np.mean(d_ab))
    m_ba = float(np.mean(d_ba))
    return 0.5 * (m_ab + m_ba), m_ab, m_ba


def _gaussian_physics_metrics(
    cloud: LoadedGaussianCloud,
    anisotropy_threshold: float,
) -> Tuple[float, float, float, float]:
    max_scale = cloud.scale.max(axis=1)
    min_scale = cloud.scale.min(axis=1).clip(min=1e-8)
    ratio = max_scale / min_scale
    aniso_frac = float(np.mean(ratio > anisotropy_threshold))
    return float(np.mean(max_scale)), float(np.median(max_scale)), aniso_frac, float(np.mean(cloud.opacity))


def _knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)
    if k == 1:
        idx = idx[:, None]
    return idx.astype(np.int64)


def _chunked_pca_roughness(
    xyz: np.ndarray,
    neighbor_idx: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> float:
    pts = torch.from_numpy(xyz).to(device)
    nbr = torch.from_numpy(neighbor_idx).to(device)
    n = pts.shape[0]
    residuals: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        p = pts[start:end]
        nb = pts[nbr[start:end]][:, 1:, :]
        centroid = nb.mean(dim=1, keepdim=True)
        centered = nb - centroid
        cov = centered.transpose(1, 2) @ centered / max(centered.shape[1] - 1, 1)
        _, evecs = torch.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        diff = p - centroid.squeeze(1)
        dist = torch.abs(torch.sum(diff * normal, dim=1))
        residuals.append(dist.detach().cpu())
    return float(torch.cat(residuals).mean())


def _sor_std(
    xyz: np.ndarray,
    neighbor_idx: np.ndarray,
    sor_k: int,
    device: torch.device,
    chunk_size: int,
) -> float:
    pts = torch.from_numpy(xyz).to(device)
    nbr = torch.from_numpy(neighbor_idx[:, 1 : sor_k + 1]).to(device)
    n = pts.shape[0]
    dists_all: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        p = pts[start:end].unsqueeze(1)
        nb = pts[nbr[start:end]]
        d = torch.linalg.norm(nb - p, dim=2).mean(dim=1)
        dists_all.append(d.detach().cpu())
    return float(torch.cat(dists_all).std())


def _sparse_voxel_ratio(xyz: np.ndarray, voxel_size: float) -> float:
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    _, inv = np.unique(vox, axis=0, return_inverse=True)
    active = int(inv.max()) + 1 if inv.size else 0
    if active == 0:
        return 0.0
    cnt = np.bincount(inv)
    sparse = int(np.sum((cnt == 1) | (cnt == 2)))
    return sparse / active


def compute_gaussian_metrics(
    visible: LoadedGaussianCloud,
    filter_stats: Dict[str, int],
    *,
    anisotropy_threshold: float = 4.0,
    roughness_k: int = 24,
    sor_k: int = 12,
    voxel_size: float = 0.1,
    device: torch.device,
    chunk_size: int = 65536,
) -> GaussianMetrics:
    mean_max, med_max, aniso_frac, mean_op = _gaussian_physics_metrics(
        visible, anisotropy_threshold
    )
    xyz = visible.xyz
    knn_k = max(roughness_k, sor_k + 1)
    nbr_idx = _knn_indices(xyz, knn_k)
    rough = _chunked_pca_roughness(xyz, nbr_idx[:, :roughness_k], device, chunk_size)
    sor_s = _sor_std(xyz, nbr_idx, sor_k, device, chunk_size)
    sparse = _sparse_voxel_ratio(xyz, voxel_size)

    return GaussianMetrics(
        filter_stats=filter_stats,
        opacity_activation=visible.opacity_activation,
        scale_activation=visible.scale_activation,
        active_effective_points=int(visible.xyz.shape[0]),
        invisible_zombie_points=int(filter_stats["n_invisible"]),
        mean_opacity_visible=mean_op,
        raw_chamfer_l1=0.0,
        raw_chamfer_ours_to_base=0.0,
        raw_chamfer_base_to_ours=0.0,
        mean_max_scale=mean_max,
        median_max_scale=med_max,
        anisotropy_ratio_gt_threshold=aniso_frac,
        anisotropy_threshold=anisotropy_threshold,
        mean_roughness=rough,
        sor_std=sor_s,
        sparse_voxel_ratio=sparse,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def validate_inputs(ours_path: str, baseline_path: str) -> Tuple[Path, Path]:
    ours_p = Path(ours_path).resolve()
    base_p = Path(baseline_path).resolve()
    if not ours_p.is_file():
        raise FileNotFoundError(f"--ours 不存在: {ours_p}")
    if not base_p.is_file():
        raise FileNotFoundError(f"--baseline 不存在: {base_p}")
    if ours_p == base_p:
        raise ValueError(f"--ours 与 --baseline 指向同一文件: {ours_p}")
    if _file_fingerprint(ours_p) == _file_fingerprint(base_p):
        raise ValueError("两个 PLY 内容完全相同（hash 一致）。")
    return ours_p, base_p


def _winner(a: float, b: float, lower_is_better: bool = True) -> str:
    denom = max(abs(a), abs(b), 1e-12)
    if abs(a - b) / denom < 1e-6:
        return "tie"
    if lower_is_better:
        return "ours" if a < b else "baseline"
    return "ours" if a > b else "baseline"


def _fmt(v: float, digits: int = 6) -> str:
    if abs(v) >= 1000 or (0 < abs(v) < 1e-3):
        return f"{v:.4e}"
    return f"{v:.{digits}f}"


def _diagnosis(
    ours: GaussianMetrics,
    base: GaussianMetrics,
    chamfer_o: float,
    chamfer_b: float,
) -> List[str]:
    lines: List[str] = []
    if chamfer_o > 0.05 or chamfer_b > 0.05:
        lines.append(
            f"**全局几何漂移**：Raw Chamfer L1 较大（Ours={_fmt(chamfer_o)}, "
            f"Baseline={_fmt(chamfer_b)}），未经对齐的空间偏差会直接拉低 2D PSNR。"
        )
    if ours.invisible_zombie_points > base.invisible_zombie_points * 1.5:
        lines.append(
            f"**隐形僵尸点偏多**：Ours 有 {ours.invisible_zombie_points:,} 个 opacity≤阈值点 "
            f"（Baseline {base.invisible_zombie_points:,}），旧版纯 XYZ 指标会被这些点稀释。"
        )
    if _winner(ours.anisotropy_ratio_gt_threshold, base.anisotropy_ratio_gt_threshold) == "baseline":
        lines.append(
            "**拉丝/飞线退化更严重**：Anisotropy Ratio>4 占比高于 Baseline，"
            "高长宽比高斯会在全景渲染中产生毛刺悬浮物。"
        )
    if _winner(ours.mean_roughness, base.mean_roughness) == "baseline":
        lines.append(
            "**显形高斯中心局部粗糙度偏高**（opacity 过滤后 PCA 残差更大），表面不够平滑。"
        )
    if _winner(ours.sor_std, base.sor_std) == "baseline" or _winner(
        ours.sparse_voxel_ratio, base.sparse_voxel_ratio
    ) == "baseline":
        lines.append(
            "**高不透明度悬浮噪点/稀疏杂质**：显形点中 SOR 离散度或稀疏体素占比较高。"
        )
    if not lines:
        lines.append(
            "显形高斯物理指标整体不劣于 Baseline；若 PSNR 仍低，优先排查 Raw Chamfer 全局漂移与外观/评测协议。"
        )
    return lines


def render_markdown_report(
    label_ours: str,
    label_base: str,
    ours: GaussianMetrics,
    base: GaussianMetrics,
    chamfer: Tuple[float, float, float],
    align_info: Dict[str, float],
    elapsed_s: float,
    *,
    ours_path: str,
    baseline_path: str,
    ours_fp: str,
    baseline_fp: str,
    min_opacity: float,
    align_requested: str,
    align_warnings: List[str],
) -> str:
    chamfer_l1, chamfer_o2b, chamfer_b2o = chamfer
    ours.raw_chamfer_l1 = chamfer_l1
    ours.raw_chamfer_ours_to_base = chamfer_o2b
    ours.raw_chamfer_base_to_ours = chamfer_b2o

    primary_rows = [
        (
            "PLY 总顶点数",
            f"{ours.filter_stats['n_ply_vertices']:,}",
            f"{base.filter_stats['n_ply_vertices']:,}",
            "—",
            "PLY header 原始 Gaussian 数量",
        ),
        (
            "Invisible Zombie Points (opacity≤阈值)",
            f"{ours.invisible_zombie_points:,}",
            f"{base.invisible_zombie_points:,}",
            _winner(ours.invisible_zombie_points, base.invisible_zombie_points),
            f"不参与渲染的隐形点；阈值={min_opacity}",
        ),
        (
            "Active Effective Points",
            f"{ours.active_effective_points:,}",
            f"{base.active_effective_points:,}",
            "—",
            f"opacity > {min_opacity}，参与全部指标",
        ),
        (
            "Mean Visible Opacity",
            _fmt(ours.mean_opacity_visible, 4),
            _fmt(base.mean_opacity_visible, 4),
            "—",
            "显形点平均不透明度（已激活）",
        ),
        (
            "Raw Chamfer L1 ↓",
            _fmt(chamfer_o2b),
            _fmt(chamfer_b2o),
            _winner(chamfer_o2b, chamfer_b2o),
            "无 ICP 对齐：列分别为 Ours→Baseline / Baseline→Ours 单向均值；"
            f"对称 L1={_fmt(chamfer_l1)}",
        ),
        (
            "Mean Max Scale",
            _fmt(ours.mean_max_scale),
            _fmt(base.mean_max_scale),
            "—",
            "显形高斯最大半轴均值；过大=臃肿，过小=漏光",
        ),
        (
            f"Anisotropy Ratio (>{ours.anisotropy_threshold:g}) ↓",
            f"{100.0 * ours.anisotropy_ratio_gt_threshold:.2f}%",
            f"{100.0 * base.anisotropy_ratio_gt_threshold:.2f}%",
            _winner(ours.anisotropy_ratio_gt_threshold, base.anisotropy_ratio_gt_threshold),
            "最大轴/最小轴 > 阈值 的占比；量化拉丝飞线",
        ),
    ]

    secondary_rows = [
        (
            "Mean Roughness (visible) ↓",
            _fmt(ours.mean_roughness),
            _fmt(base.mean_roughness),
            _winner(ours.mean_roughness, base.mean_roughness),
            "显形中心 PCA 平面残差（Ours 对齐后 / Baseline 原坐标）",
        ),
        (
            "SOR Std (visible) ↓",
            _fmt(ours.sor_std),
            _fmt(base.sor_std),
            _winner(ours.sor_std, base.sor_std),
            "显形点邻居距离标准差",
        ),
        (
            "Sparse Voxel Ratio (visible) ↓",
            f"{100.0 * ours.sparse_voxel_ratio:.2f}%",
            f"{100.0 * base.sparse_voxel_ratio:.2f}%",
            _winner(ours.sparse_voxel_ratio, base.sparse_voxel_ratio),
            "0.1 m 网格中 1–2 点体素占比",
        ),
    ]

    lines: List[str] = [
        "## Gaussian Physics-Aligned Geometry Diagnosis",
        "",
        f"- **Ours PLY**: `{ours_path}`",
        f"- **Baseline PLY**: `{baseline_path}`",
        f"- **Ours SHA256**: `{ours_fp}` | **Baseline SHA256**: `{baseline_fp}`",
        f"- **Opacity 激活**: Ours=`{ours.opacity_activation}` | Baseline=`{base.opacity_activation}`",
        f"- **Scale 激活**: Ours=`{ours.scale_activation}` | Baseline=`{base.scale_activation}`",
        f"- **硬剪枝阈值**: opacity > {min_opacity}",
        "",
        "### 核心物理指标（显形高斯）",
        "",
        f"| 指标 | {label_ours} | {label_base} | 更优 | 说明 |",
        "| --- | ---: | ---: | :---: | --- |",
    ]
    for name, v_o, v_b, win, note in primary_rows:
        win_disp = {"ours": "**ours**", "baseline": "**baseline**", "tie": "tie"}.get(win, win)
        lines.append(f"| {name} | {v_o} | {v_b} | {win_disp} | {note} |")

    lines.extend([
        "",
        "### 显形高斯局部几何（辅助）",
        "",
        f"| 指标 | {label_ours} | {label_base} | 更优 | 说明 |",
        "| --- | ---: | ---: | :---: | --- |",
    ])
    for name, v_o, v_b, win, note in secondary_rows:
        win_disp = {"ours": "**ours**", "baseline": "**baseline**", "tie": "tie"}.get(win, win)
        lines.append(f"| {name} | {v_o} | {v_b} | {win_disp} | {note} |")

    lines.extend(["", "### 对齐信息（仅辅助局部几何行）", ""])
    lines.append(f"- **requested**: {align_requested}")
    for k, v in align_info.items():
        lines.append(f"- **{k}**: {v}")
    if align_warnings:
        for w in align_warnings:
            lines.append(f"- ⚠ {w}")

    lines.extend(["", "### 诊断结论", ""])
    for line in _diagnosis(ours, base, chamfer_o2b, chamfer_b2o):
        lines.append(f"- {line}")

    lines.extend(["", f"_分析耗时: {elapsed_s:.1f}s_", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gaussian physics-aligned comparison of two 3DGS PLY files."
    )
    p.add_argument("--ours", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--label-ours", default="Ours")
    p.add_argument("--label-baseline", default="PanoSplatt3R")
    p.add_argument("--output", default="X_debug/info.md")
    p.add_argument("--min-opacity", type=float, default=0.05,
                   help="Hard prune Gaussians with opacity <= this value (default 0.05).")
    p.add_argument("--anisotropy-threshold", type=float, default=4.0)
    p.add_argument("--align", choices=("icp", "umeyama", "none"), default="icp",
                   help="Alignment for auxiliary local geometry on Ours only.")
    p.add_argument("--roughness-k", type=int, default=24)
    p.add_argument("--sor-k", type=int, default=12)
    p.add_argument("--voxel-size", type=float, default=0.1)
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--icp-voxel", type=float, default=0.05)
    p.add_argument("--icp-max-corr", type=float, default=0.25)
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    device = _resolve_device(args.device)

    ours_p, base_p = validate_inputs(args.ours, args.baseline)
    ours_raw = load_ply_gaussian(str(ours_p))
    base_raw = load_ply_gaussian(str(base_p))

    ours_vis, ours_fstats = filter_visible_gaussians(ours_raw, args.min_opacity)
    base_vis, base_fstats = filter_visible_gaussians(base_raw, args.min_opacity)

    # Raw Chamfer: NO alignment, visible centers only.
    chamfer = raw_chamfer_l1(ours_vis.xyz, base_vis.xyz)

    align_method = args.align
    align_warnings: List[str] = []
    if align_method == "icp" and not _HAS_O3D:
        align_warnings.append("open3d 未安装，局部几何对齐回退 umeyama。")
        align_method = "umeyama"

    ours_xyz_aligned, align_info = align_xyz(
        ours_vis.xyz,
        base_vis.xyz,
        method=align_method,
        icp_voxel=args.icp_voxel,
        icp_max_corr=args.icp_max_corr,
        seed=args.seed,
    )
    ours_for_local = LoadedGaussianCloud(
        xyz=ours_xyz_aligned,
        opacity=ours_vis.opacity,
        scale=ours_vis.scale,
        n_ply_vertices=ours_vis.n_ply_vertices,
        n_valid=ours_vis.n_valid,
        n_filtered_nonfinite=ours_vis.n_filtered_nonfinite,
        opacity_prop=ours_vis.opacity_prop,
        scale_props=ours_vis.scale_props,
        opacity_activation=ours_vis.opacity_activation,
        scale_activation=ours_vis.scale_activation,
    )

    m_ours = compute_gaussian_metrics(
        ours_for_local,
        ours_fstats,
        anisotropy_threshold=args.anisotropy_threshold,
        roughness_k=args.roughness_k,
        sor_k=args.sor_k,
        voxel_size=args.voxel_size,
        device=device,
        chunk_size=args.chunk_size,
    )
    m_base = compute_gaussian_metrics(
        base_vis,
        base_fstats,
        anisotropy_threshold=args.anisotropy_threshold,
        roughness_k=args.roughness_k,
        sor_k=args.sor_k,
        voxel_size=args.voxel_size,
        device=device,
        chunk_size=args.chunk_size,
    )

    report = render_markdown_report(
        args.label_ours,
        args.label_baseline,
        m_ours,
        m_base,
        chamfer,
        align_info,
        time.time() - t0,
        ours_path=str(ours_p),
        baseline_path=str(base_p),
        ours_fp=_file_fingerprint(ours_p),
        baseline_fp=_file_fingerprint(base_p),
        min_opacity=args.min_opacity,
        align_requested=args.align,
        align_warnings=align_warnings,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
