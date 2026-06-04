"""Gaussian adapters: ERP (ODGS isotropic) vs pinhole (interface only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from panovggt.render.odgs_bridge import TensorGaussianCloud, build_tensor_cloud, pack_sh_features
from panovggt.utils.gs_debug import (
    gs_grad_print,
    gs_grad_verbose_should_run,
    is_gs_grad_debug_enabled,
)
from panovggt.utils.opacity import map_pdf_to_opacity
from panovggt.utils.sh_utils import make_sh_mask, rgb_to_sh


# Base multiplier for 3D Gaussian radius (world units). 0.001 collapses below rasterizer
# screen-space floor (~0.3 px variance), killing scale/rotation gradients.
GS_SCALE_BASE = 0.02
GS_SCALE_MIN = 1e-5
GS_SCALE_MAX = 0.05
# Max axis / min axis; prevents needle-like "streak" Gaussians on flat regions.
GS_MAX_ASPECT_RATIO = 4.0
GS_OPACITY_FLOOR = 0.05


# Default 3D voxel grid size for Winner-Take-All voxelization (world units).
GS_VOXEL_SIZE = 0.02
GS_RAW_FEAT_DIM = 11
GS_CONF_CHANNEL = 11


def _scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-voxel max without torch_scatter (PyTorch scatter_reduce)."""
    out = torch.full(
        (dim_size,),
        float("-inf"),
        device=src.device,
        dtype=src.dtype,
    )
    out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    return out


def _scatter_min(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-voxel min without torch_scatter (PyTorch scatter_reduce)."""
    fill = float("inf") if src.is_floating_point() else torch.iinfo(src.dtype).max
    out = torch.full((dim_size,), fill, device=src.device, dtype=src.dtype)
    out.scatter_reduce_(0, index, src, reduce="amin", include_self=True)
    return out


@dataclass
class GaussianAdapterOutput:
    cloud: TensorGaussianCloud
    means: torch.Tensor
    opacities: torch.Tensor
    mask: torch.Tensor


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit quaternion with stable autograd (prefer over manual q / ||q||)."""
    return F.normalize(q, p=2, dim=-1, eps=eps)


def _sample_rgb(images: torch.Tensor, stride: int) -> torch.Tensor:
    """images: (B, 3, H, W) -> (B, H', W', 3) RGB in [0,1]."""
    if stride > 1:
        images = images[:, :, ::stride, ::stride]
    return images.permute(0, 2, 3, 1).contiguous()


class PanoGaussianAdapterERP(torch.nn.Module):
    """ERP / ODGS anisotropic Gaussian adapter."""

    def __init__(
        self,
        sh_degree: int = 0,
        opacity_init: float = 0.0,
        opacity_final: float = 0.0,
        gs_scale_max: float = GS_SCALE_MAX,
        gs_voxelize: bool = True,
        gs_voxel_size: float = GS_VOXEL_SIZE,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.opacity_init = opacity_init
        self.opacity_final = opacity_final
        self.gs_scale_max = float(gs_scale_max)
        self.gs_voxelize = bool(gs_voxelize)
        self.gs_voxel_size = float(gs_voxel_size)
        self.register_buffer(
            "sh_mask",
            make_sh_mask(sh_degree, torch.device("cpu"), torch.float32),
            persistent=False,
        )

    def voxelization_with_fusion(
        self,
        raw_feats: torch.Tensor,
        pts3d: torch.Tensor,
        voxel_size: float,
        conf: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        3D voxelization with Winner-Take-All (WTA) selection.

        Within each voxel, the pixel with the highest raw confidence fully
        determines position, raw GS features, and RGB — no weighted averaging.

        Args:
            raw_feats: (N, 11) pre-activation GS head output (density, scale, rot, sh).
            pts3d: (N, 3) world coordinates.
            voxel_size: voxel edge length in world units.
            conf: (N,) raw linear confidence (channel 11, pre-sigmoid).
            rgb: optional (N, 3) image colors in [0, 1].

        Returns:
            fused_pts: (M, 3)
            fused_feats: (M, 11)
            fused_rgb: (M, 3) or None
        """
        if pts3d.numel() == 0:
            empty_feats = raw_feats.new_zeros((0, raw_feats.shape[-1]))
            empty_pts = pts3d.new_zeros((0, 3))
            empty_rgb = None if rgb is None else rgb.new_zeros((0, 3))
            return empty_pts, empty_feats, empty_rgb

        pts3d = pts3d.float()
        raw_feats = raw_feats.float()
        conf = conf.float()
        if rgb is not None:
            rgb = rgb.float()

        voxel_indices = (pts3d / voxel_size).round().to(torch.int64)
        _, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
        num_voxels = int(inverse_indices.max().item()) + 1
        num_points = pts3d.shape[0]

        conf_voxel_max = _scatter_max(conf, inverse_indices, num_voxels)
        is_max_conf = conf >= conf_voxel_max[inverse_indices] - 1e-6

        point_idx = torch.arange(num_points, device=pts3d.device, dtype=torch.int64)
        sentinel = torch.full_like(point_idx, num_points)
        candidate_idx = torch.where(is_max_conf, point_idx, sentinel)
        winner_idx = _scatter_min(candidate_idx, inverse_indices, num_voxels)

        fused_pts = pts3d[winner_idx]
        fused_feats = raw_feats[winner_idx]
        fused_rgb = rgb[winner_idx] if rgb is not None else None
        return fused_pts, fused_feats, fused_rgb

    def _build_cloud_from_raw(
        self,
        mu: torch.Tensor,
        raw_feats: torch.Tensor,
        rgb_base: torch.Tensor,
        global_step: int,
    ) -> tuple[TensorGaussianCloud, torch.Tensor, torch.Tensor]:
        """Apply physical activations and safety clamps to fused or pixel-wise raw GS."""
        n = mu.shape[0]
        if n == 0:
            empty = mu.new_zeros((0, 3))
            cloud = TensorGaussianCloud(
                xyz=empty,
                scaling=empty,
                rotation=empty.new_zeros((0, 4)),
                opacity=empty.new_zeros((0, 1)),
                features_dc=empty.new_zeros((0, 1, 3)),
                features_rest=empty.new_zeros((0, 0, 3)),
                sh_degree=self.sh_degree,
            )
            return cloud, empty, empty.new_zeros(0)

        density_raw = raw_feats[:, 0]
        scale_sel = raw_feats[:, 1:4]
        quat_raw = raw_feats[:, 4:8]
        sh_res_raw = raw_feats[:, 8:11]

        pdf = torch.sigmoid(density_raw)
        opacity = map_pdf_to_opacity(
            pdf, global_step, self.opacity_init, self.opacity_final
        )
        opacity = (opacity + GS_OPACITY_FLOOR).clamp(0.0, 1.0)

        scales = (
            GS_SCALE_BASE * F.softplus(scale_sel)
        ).clamp(min=GS_SCALE_MIN, max=self.gs_scale_max)
        max_scale = torch.max(scales, dim=-1, keepdim=True)[0]
        min_allowed_scale = max_scale / GS_MAX_ASPECT_RATIO
        scales = torch.max(scales, min_allowed_scale)
        scales_log = torch.log(scales.clamp(min=1e-6))

        rot = normalize_quaternion(quat_raw)
        rgb = (rgb_base + torch.sigmoid(sh_res_raw)).clamp(0.0, 1.0)
        sh_dc = rgb_to_sh(rgb)
        n_coeff = (self.sh_degree + 1) ** 2
        sh_b = sh_dc.unsqueeze(-1)
        if n_coeff > 1:
            sh_full = torch.zeros(
                n, 3, n_coeff, device=mu.device, dtype=mu.dtype
            )
            sh_full[..., 0] = sh_dc
            sh_full[..., 1] = torch.sigmoid(sh_res_raw[..., 0:1]).expand(-1, 3)
            sh_b = sh_full
        sh_mask = self.sh_mask.to(device=mu.device, dtype=mu.dtype)
        if self.sh_degree > 0:
            sh_b = sh_b * sh_mask.view(1, 1, n_coeff)
        if sh_b.dim() == 2:
            sh_b = sh_b.unsqueeze(1)
        if sh_b.shape[-1] == 1 and sh_b.shape[-2] == 3:
            sh_b = sh_b.permute(0, 2, 1)
        features_dc, features_rest = pack_sh_features(sh_b, self.sh_degree)

        opacity_logit = torch.logit(opacity.clamp(1e-4, 1 - 1e-4))
        cloud = TensorGaussianCloud(
            xyz=mu,
            scaling=scales_log,
            rotation=rot,
            opacity=opacity_logit.unsqueeze(-1),
            features_dc=features_dc,
            features_rest=features_rest,
            sh_degree=self.sh_degree,
        )
        return cloud, mu, opacity

    def _preprocess_view_gs(
        self,
        means: torch.Tensor,
        gs_out: torch.Tensor,
        images: torch.Tensor,
        point_mask: Optional[torch.Tensor],
        stride: int,
        detach_means: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared prep before flatten / voxelize (no activations applied)."""
        gs_out = torch.nan_to_num(gs_out.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        while images.dim() > 4:
            images = images.squeeze(1)
        if detach_means:
            means = means.detach()

        b, _, h, w = gs_out.shape
        if means.shape[1:3] != (h, w):
            means = (
                F.interpolate(
                    means.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        valid_geom = torch.isfinite(means).all(dim=-1)
        if stride > 1:
            gs_out = gs_out[:, :, ::stride, ::stride]
            means = means[:, ::stride, ::stride]
            if point_mask is not None:
                point_mask = point_mask[:, ::stride, ::stride]
            valid_geom = valid_geom[:, ::stride, ::stride]

        rgb_grid = _sample_rgb(images, 1)
        conf_raw = torch.nan_to_num(
            gs_out[:, GS_CONF_CHANNEL].float(), nan=0.0, posinf=0.0, neginf=0.0
        )

        mask = valid_geom.clone()
        if point_mask is not None:
            pm = point_mask.bool()
            if pm.shape != mask.shape:
                raise ValueError(
                    f"point_mask shape {tuple(pm.shape)} != expected {tuple(mask.shape)}"
                )
            mask = mask & pm
        return gs_out, means, rgb_grid, conf_raw, mask

    @staticmethod
    def _resolve_batch_mask(
        mask: torch.Tensor,
        valid_geom: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        m = mask[batch_idx]
        if m.sum() == 0:
            m = valid_geom[batch_idx]
        if m.sum() == 0:
            m = torch.ones_like(valid_geom[batch_idx], dtype=torch.bool)
        return m

    def flat_masked_gs_batch_item(
        self,
        gs_out: torch.Tensor,
        means: torch.Tensor,
        rgb_grid: torch.Tensor,
        conf_raw: torch.Tensor,
        mask: torch.Tensor,
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten one batch item to raw (N,11), pts, conf, rgb — pre-voxelization."""
        valid_geom = torch.isfinite(means).all(dim=-1)
        m = self._resolve_batch_mask(mask, valid_geom, batch_idx)
        raw_feats = gs_out[batch_idx, :GS_RAW_FEAT_DIM].permute(1, 2, 0)[m]
        pts = means[batch_idx][m]
        conf = conf_raw[batch_idx][m]
        rgb = rgb_grid[batch_idx][m]
        return raw_feats, pts, conf, rgb

    def build_gaussian_output_from_flat(
        self,
        raw_feats: torch.Tensor,
        pts: torch.Tensor,
        conf: torch.Tensor,
        rgb: torch.Tensor,
        global_step: int,
    ) -> GaussianAdapterOutput:
        """Voxel-fuse (optional) flat GS buffers and apply physical activations."""
        n_in = int(pts.shape[0])
        if self.gs_voxelize and n_in > 0:
            mu, fused_feats, fused_rgb = self.voxelization_with_fusion(
                raw_feats,
                pts,
                self.gs_voxel_size,
                conf,
                rgb=rgb,
            )
            rgb_base = fused_rgb if fused_rgb is not None else rgb.new_zeros(
                (mu.shape[0], 3)
            )
            out_mask = torch.ones(mu.shape[0], device=pts.device, dtype=torch.bool)
        else:
            mu, fused_feats, rgb_base = pts, raw_feats, rgb
            out_mask = torch.ones(n_in, device=pts.device, dtype=torch.bool)

        cloud, means_out, opacity = self._build_cloud_from_raw(
            mu, fused_feats, rgb_base, global_step
        )
        return GaussianAdapterOutput(
            cloud=cloud,
            means=means_out,
            opacities=opacity,
            mask=out_mask,
        )

    def forward(
        self,
        means: torch.Tensor,
        gs_out: torch.Tensor,
        images: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        global_step: int = 0,
        stride: int = 1,
        detach_means: bool = False,
    ) -> List[GaussianAdapterOutput]:
        """
        Args:
            means: (B, H, W, 3) world coordinates (anchor frame).
            gs_out: (B, 12, H, W) raw head output.
            images: (B, 3, H, W) in [0, 1].
            point_mask: (B, H, W) valid geometry mask.
        """
        b, _, h, w = gs_out.shape
        gs_out, means, rgb_grid, conf_raw, mask = self._preprocess_view_gs(
            means,
            gs_out,
            images,
            point_mask,
            stride,
            detach_means,
        )
        valid_geom = torch.isfinite(means).all(dim=-1)

        outputs: List[GaussianAdapterOutput] = []

        for bi in range(b):
            raw_feats, pts, conf, rgb = self.flat_masked_gs_batch_item(
                gs_out, means, rgb_grid, conf_raw, mask, bi
            )
            n_in = int(pts.shape[0])
            out = self.build_gaussian_output_from_flat(
                raw_feats, pts, conf, rgb, global_step
            )
            outputs.append(out)
            if is_gs_grad_debug_enabled() and bi == 0:
                step = int(global_step)
                if gs_grad_verbose_should_run(step):
                    n_out = int(out.means.shape[0])
                    gs_grad_print(
                        "adapter_batch0",
                        step=step,
                        verbose=True,
                        gs_voxelize=self.gs_voxelize,
                        gs_voxel_size=self.gs_voxel_size,
                        num_input_points=n_in,
                        num_gaussians=n_out,
                        voxelize_ratio=float(n_out / max(n_in, 1)),
                        means=out.means,
                        gs_out_requires_grad=bool(gs_out.requires_grad),
                        means_requires_grad=bool(means.requires_grad),
                    )
        return outputs


class PanoGaussianAdapterPinhole(torch.nn.Module):
    """Pinhole adapter stub for future perspective training."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Pinhole Gaussian adapter is not trained in v1. Use view_type='erp'."
        )


def merge_gaussian_adapter_outputs(
    outputs: List[GaussianAdapterOutput],
) -> GaussianAdapterOutput:
    """Concatenate per-view adapter outputs into one Gaussian cloud."""
    if not outputs:
        raise ValueError("Cannot merge empty GaussianAdapterOutput list")
    if len(outputs) == 1:
        return outputs[0]

    cloud0 = outputs[0].cloud
    cloud = TensorGaussianCloud(
        xyz=torch.cat([o.cloud._xyz for o in outputs], dim=0),
        scaling=torch.cat([o.cloud._scaling for o in outputs], dim=0),
        rotation=torch.cat([o.cloud._rotation for o in outputs], dim=0),
        opacity=torch.cat([o.cloud._opacity for o in outputs], dim=0),
        features_dc=torch.cat([o.cloud._features_dc for o in outputs], dim=0),
        features_rest=torch.cat([o.cloud._features_rest for o in outputs], dim=0),
        sh_degree=cloud0.max_sh_degree,
        active_sh_degree=cloud0.active_sh_degree,
    )
    return GaussianAdapterOutput(
        cloud=cloud,
        means=torch.cat([o.means for o in outputs], dim=0),
        opacities=torch.cat([o.opacities for o in outputs], dim=0),
        mask=torch.cat([o.mask.reshape(-1) for o in outputs], dim=0),
    )


def build_gaussian_cloud_batch(
    adapter: torch.nn.Module,
    view_type: str,
    means: torch.Tensor,
    gs_out: torch.Tensor,
    images: torch.Tensor,
    **kwargs,
) -> List[GaussianAdapterOutput]:
    if view_type == "erp":
        if not isinstance(adapter, PanoGaussianAdapterERP):
            raise TypeError("ERP view requires PanoGaussianAdapterERP")
        return adapter(means, gs_out, images, **kwargs)
    if view_type == "pinhole":
        raise NotImplementedError("pinhole training is not enabled in v1")
    raise ValueError(f"Unknown view_type: {view_type}")
