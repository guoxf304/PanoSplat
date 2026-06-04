"""ODGS omnidirectional rasterization bridge for feed-forward Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.render.camera import ERPPanoCamera
from panovggt.utils.sh_utils import sh_to_rgb


def check_odgs_available() -> bool:
    try:
        import odgs_gaussian_rasterization  # noqa: F401

        return True
    except ImportError:
        return False


def _require_odgs():
    if not check_odgs_available():
        raise ImportError(
            "odgs_gaussian_rasterization is not installed. Install with:\n"
            "  pip install -e /path/to/ODGS/submodules/odgs-gaussian-rasterization"
        )


@dataclass
class ODGSRenderPipe:
    """Pipeline flags mirroring ODGS gaussian_renderer."""

    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False
    debug: bool = False


class TensorGaussianCloud:
    """
    Tensor-backed Gaussian cloud with the ODGS GaussianModel read API.
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        sh_degree: int = 0,
        active_sh_degree: Optional[int] = None,
    ):
        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        self._features_dc = features_dc
        self._features_rest = features_rest
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = (
            int(active_sh_degree) if active_sh_degree is not None else int(sh_degree)
        )

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        s = torch.exp(self._scaling)
        if s.dim() == 1:
            s = s.unsqueeze(-1).expand(-1, 3)
        elif s.shape[-1] == 1:
            s = s.expand(-1, 3)
        return s

    @property
    def get_rotation(self) -> torch.Tensor:
        # Adapter stores unit quaternions; re-normalize for legacy / sanitize safety.
        return F.normalize(self._rotation, p=2, dim=-1, eps=1e-6)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)


def pack_sh_features(
    sh: torch.Tensor,
    sh_degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack SH coefficients into ODGS feature layout.

    ODGS stores ``_features_dc`` as (N, 1, 3) and ``_features_rest`` as (N, K-1, 3);
    ``get_features`` concatenates along dim=1 -> (N, K, 3).

    Args:
        sh: (N, K, 3), (N, 3, K), or (N, 1, 3) when K=1.
        sh_degree: Maximum SH degree.

    Returns:
        features_dc: (N, 1, 3)
        features_rest: (N, (d+1)^2 - 1, 3)
    """
    n_coeff = (sh_degree + 1) ** 2
    if sh.dim() != 3:
        raise ValueError(f"SH must be 3D, got shape {sh.shape}")

    if sh.shape[1] == 1 and sh.shape[2] == 3:
        sh_nk3 = sh
    elif sh.shape[1] == n_coeff and sh.shape[2] == 3:
        sh_nk3 = sh
    elif sh.shape[1] == 3 and sh.shape[2] == n_coeff:
        sh_nk3 = sh.permute(0, 2, 1).contiguous()
    else:
        raise ValueError(f"Unexpected SH shape {sh.shape} for degree {sh_degree}")

    features_dc = sh_nk3[:, :1, :]
    if n_coeff > 1:
        features_rest = sh_nk3[:, 1:, :]
    else:
        features_rest = sh.new_zeros((sh.shape[0], 0, 3), device=sh.device, dtype=sh.dtype)
    return features_dc, features_rest


def build_tensor_cloud(
    means: torch.Tensor,
    scales_log: torch.Tensor,
    rotations: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh: torch.Tensor,
    sh_degree: int,
    active_sh_degree: Optional[int] = None,
) -> TensorGaussianCloud:
    """Build TensorGaussianCloud from activated tensors (scales already log-space)."""
    features_dc, features_rest = pack_sh_features(sh, sh_degree)
    return TensorGaussianCloud(
        xyz=means,
        scaling=scales_log,
        rotation=rotations,
        opacity=opacity_logits,
        features_dc=features_dc,
        features_rest=features_rest,
        sh_degree=sh_degree,
        active_sh_degree=active_sh_degree,
    )


def sanitize_gaussian_cloud(
    cloud: TensorGaussianCloud,
    max_abs_xyz: float = 50.0,
) -> TensorGaussianCloud:
    """Clamp / de-NaN Gaussian parameters before ODGS rasterization."""
    xyz = torch.nan_to_num(cloud._xyz.float(), nan=0.0, posinf=max_abs_xyz, neginf=-max_abs_xyz)
    xyz = xyz.clamp(-max_abs_xyz, max_abs_xyz)
    scaling = torch.nan_to_num(cloud._scaling.float(), nan=-6.0, posinf=2.0, neginf=-12.0)
    rotation = F.normalize(torch.nan_to_num(cloud._rotation.float(), nan=0.0), dim=-1, eps=1e-6)
    opacity = torch.nan_to_num(cloud._opacity.float(), nan=0.0, posinf=10.0, neginf=-10.0)
    features_dc = torch.nan_to_num(cloud._features_dc.float(), nan=0.0)
    features_rest = torch.nan_to_num(cloud._features_rest.float(), nan=0.0)
    return TensorGaussianCloud(
        xyz=xyz,
        scaling=scaling,
        rotation=rotation,
        opacity=opacity,
        features_dc=features_dc,
        features_rest=features_rest,
        sh_degree=cloud.max_sh_degree,
        active_sh_degree=cloud.active_sh_degree,
    )


def scale_gaussian_cloud_for_render(
    cloud: TensorGaussianCloud,
    scale: torch.Tensor,
    *,
    scale_xyz: bool = True,
) -> TensorGaussianCloud:
    """Match ``normalize_pred`` scene scale: shrink xyz and Gaussian sizes together.

    When ``scale_xyz=False``, only adjust log-scales (xyz already in render-normalized
    coordinates, e.g. from batch ``world_points``).
    """
    s = scale.reshape(()).clamp(min=1e-6).to(device=cloud._xyz.device, dtype=cloud._xyz.dtype)
    log_inv = -torch.log(s)
    scaling = cloud._scaling + log_inv
    xyz = cloud._xyz / s if scale_xyz else cloud._xyz
    return TensorGaussianCloud(
        xyz=xyz,
        scaling=scaling,
        rotation=cloud._rotation,
        opacity=cloud._opacity,
        features_dc=cloud._features_dc,
        features_rest=cloud._features_rest,
        sh_degree=cloud.max_sh_degree,
        active_sh_degree=cloud.active_sh_degree,
    )


def scale_gaussian_cloud_xyz(cloud: TensorGaussianCloud, scale: torch.Tensor) -> TensorGaussianCloud:
    """Backward-compatible alias."""
    return scale_gaussian_cloud_for_render(cloud, scale)


def render_erp(
    cloud: TensorGaussianCloud,
    camera: ERPPanoCamera,
    bg_color: torch.Tensor,
    pipe: Optional[ODGSRenderPipe] = None,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> dict:
    """
    Differentiable ERP render via ODGS rasterizer.

    Returns dict with keys: render, depth, accuracy, radii, ...
    """
    _require_odgs()
    from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if pipe is None:
        pipe = ODGSRenderPipe()

    device = cloud.get_xyz.device
    screenspace_points = torch.zeros_like(cloud.get_xyz, requires_grad=True, device=device)
    if screenspace_points.requires_grad and torch.is_grad_enabled():
        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        sh_degree=cloud.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = cloud.get_xyz.float()
    opacity = cloud.get_opacity.float()
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        raise NotImplementedError("compute_cov3D_python is not implemented for TensorGaussianCloud")
    else:
        scales = cloud.get_scaling.float()
        rotations = cloud.get_rotation.float()

    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    elif cloud.active_sh_degree == 0:
        # SH=0: precompute RGB to avoid unstable SH backward in the CUDA rasterizer.
        colors_precomp = sh_to_rgb(cloud._features_dc.squeeze(1).float()).clamp(0.0, 1.0)
    else:
        shs = cloud.get_features.float()

    rendered_image, depth, acc, radii, psi, lat, lon = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "depth": depth,
        "accuracy": acc,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "psi": psi,
        "lat": lat,
        "lon": lon,
        "radii": radii,
    }


def render_pinhole(*args, **kwargs):
    """Pinhole rendering stub; not used in ERP training v1."""
    raise NotImplementedError(
        "Pinhole ODGS rendering is not enabled in v1. Use view_type='erp'."
    )
