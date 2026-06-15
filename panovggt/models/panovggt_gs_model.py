# PanoVGGT with feed-forward Gaussian splatting head (ODGS ERP).

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from panovggt.heads.dpt_gs_head import PanoDPT_GS_Head
from panovggt.heads.gaussian_adapter import (
    GS_RAW_FEAT_DIM,
    GaussianAdapterOutput,
    PanoGaussianAdapterERP,
    PanoGaussianAdapterPinhole,
    build_gaussian_cloud_batch,
    merge_gaussian_adapter_outputs,
)
from panovggt.models.panovggt_model import PanoVGGTModel
from panovggt.render.coord_frame import (
    batched_view_local_means,
    transform_gaussian_adapter_output_c2w,
)
from panovggt.render.odgs_bridge import check_odgs_available
from panovggt.utils.gs_debug import (
    gs_debug_hook_list,
    gs_debug_print,
    gs_grad_print,
    gs_grad_verbose_should_run,
    is_gs_debug_enabled,
    is_gs_grad_debug_enabled,
    register_grad_hook,
    report_cloud,
    safe_retain_grad,
    set_gs_debug,
)

logger = logging.getLogger(__name__)

# GS raw channels: density, scale_xyz(3), quat(4), sh0(3), conf
GS_RAW_DIM = 11
GS_OUT_DIM = GS_RAW_DIM + 1


class PanoVGGTGSModel(PanoVGGTModel):
    """
    PanoVGGT + AnySplat-style DPT Gaussian head + ODGS ERP adapter.

    Geometry (mu) from global_points (anchor) or per-view local geometry merged
    to world via ``c2w`` (same rule as PanoVGGT ``merged.ply``).
    Appearance / scale / rotation / opacity from PanoDPT_GS_Head (per view in merged mode).
    """

    def __init__(
        self,
        enable_gaussian: bool = True,
        gs_sh_degree: int = 0,
        gs_stride: int = 2,
        gs_means_mode: str = "anchor",
        gs_view_type: str = "erp",
        gs_detach_means: bool = False,
        gs_dpt_features: int = 256,
        gs_scale_max: float = 0.05,
        gs_voxelize: bool = True,
        gs_voxel_size: float = 0.02,
        gs_debug_forward: bool = False,
        gs_debug_every: int = 1,
        **kwargs,
    ):
        if not kwargs.get("enable_global_points", True):
            raise ValueError("PanoVGGTGSModel requires enable_global_points=True")

        super().__init__(**kwargs)

        self.enable_gaussian = enable_gaussian
        self.gs_sh_degree = gs_sh_degree
        self.gs_stride = gs_stride
        self.gs_means_mode = str(gs_means_mode).strip().lower()
        if self.gs_means_mode not in ("anchor", "merged"):
            raise ValueError(
                f"gs_means_mode must be 'anchor' or 'merged', got {gs_means_mode!r}"
            )
        self.gs_view_type = gs_view_type
        self.gs_detach_means = gs_detach_means
        self.gs_scale_max = float(gs_scale_max)
        self.gs_voxelize = bool(gs_voxelize)
        self.gs_voxel_size = float(gs_voxel_size)
        self.global_step = 0
        self.gs_debug_forward = bool(gs_debug_forward)
        self.gs_debug_every = max(1, int(gs_debug_every))
        self._gs_debug_counter = 0
        if self.gs_debug_forward:
            set_gs_debug(True)
        elif is_gs_debug_enabled():
            logger.info(
                "PanoSplat GS debug enabled via env PANOSPLAT_GS_DEBUG=1"
            )

        if enable_gaussian:
            if not check_odgs_available():
                logger.warning(
                    "odgs_gaussian_rasterization not found; install ODGS rasterizer before training GS."
                )
            self.gaussian_param_head = PanoDPT_GS_Head(
                dim_in=self.aggregator.dec_embed_dim,
                patch_size=self.patch_size,
                output_dim=GS_OUT_DIM,
                features=gs_dpt_features,
            )
            if gs_view_type == "erp":
                self.gaussian_adapter = PanoGaussianAdapterERP(
                    sh_degree=gs_sh_degree,
                    gs_scale_max=self.gs_scale_max,
                    gs_voxelize=self.gs_voxelize,
                    gs_voxel_size=self.gs_voxel_size,
                )
            else:
                self.gaussian_adapter = PanoGaussianAdapterPinhole()
            self._init_gaussian_head()

    def _init_gaussian_head(self) -> None:
        """Small random init for the new GS head (checkpoint has no GS weights)."""
        head = self.gaussian_param_head
        for m in head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        out_conv = head.scratch.output_conv2[-1]
        if isinstance(out_conv, nn.Conv2d):
            nn.init.normal_(out_conv.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(out_conv.bias)
            if out_conv.bias is not None and out_conv.bias.numel() >= GS_OUT_DIM:
                with torch.no_grad():
                    # ch0 density, ch1-3 scale_xyz, ch4-7 quat (w,x,y,z), ch8-10 sh0, ch11 conf
                    out_conv.bias[0] = -2.0  # low initial opacity (sigmoid density)
                    out_conv.bias[1:4] = 1.0  # anisotropic scale init via softplus
                    out_conv.bias[4] = 1.0  # identity quaternion w
                    out_conv.bias[5:8] = 0.0  # identity quaternion x,y,z

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def _should_log_gs_debug(self) -> bool:
        if not (self.gs_debug_forward or is_gs_debug_enabled()):
            return False
        self._gs_debug_counter += 1
        return (self._gs_debug_counter - 1) % self.gs_debug_every == 0

    def _denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return linear RGB in [0, 1] for the GS head / adapter.

        Batch images are already linear RGB before aggregator ImageNet normalization.
        """
        return images.float().clamp(0.0, 1.0)

    def _resolve_anchor_idx(self, b: int, s: int, device: torch.device) -> torch.Tensor:
        if self.training:
            return torch.randint(0, s, (b,), device=device)
        mid = s // 2
        return torch.full((b,), mid, device=device, dtype=torch.long)

    def _slice_hooks_anchor(
        self,
        hook_list: List[torch.Tensor],
        anchor_idx: torch.Tensor,
    ) -> List[torch.Tensor]:
        """hook_list entries: (B, S, P, C) or already-sliced (B, P, C)."""
        out = []
        b = int(anchor_idx.shape[0])
        for h in hook_list:
            if h.dim() == 4:
                if h.shape[0] != b:
                    raise RuntimeError(
                        f"Hook batch {h.shape[0]} != anchor batch {b}"
                    )
                out.append(
                    torch.stack(
                        [h[i, int(anchor_idx[i].item())] for i in range(b)], dim=0
                    )
                )
            elif h.dim() == 3:
                if h.shape[0] != b:
                    raise RuntimeError(
                        f"Hook batch {h.shape[0]} != anchor batch {b}"
                    )
                out.append(h)
            else:
                raise RuntimeError(
                    f"Unexpected hook tensor shape {tuple(h.shape)}"
                )
        return out

    def _prepare_dpt_encoder_tokens(
        self,
        hook_list: List[torch.Tensor],
        anchor_idx: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Build 4-scale token lists for PanoDPT_GS_Head (each scale expects dim_in channels).

        Aggregator hooks are (B, S, P, dec_embed_dim). Some code paths may pass the
        fused decoder token (2 * dec_embed_dim) as a single hook; split it into 4
        chunks of dec_embed_dim for the DPT head.
        """
        dec_dim = self.aggregator.dec_embed_dim
        n_scales = len(self.gaussian_param_head.intermediate_layer_idx)
        hooks = self._slice_hooks_anchor(hook_list, anchor_idx)
        c = hooks[0].shape[-1]

        if is_gs_debug_enabled():
            gs_debug_print(
                "prepare_dpt_tokens",
                num_in=len(hook_list),
                in0=hook_list[0] if hook_list else None,
                anchor_idx=anchor_idx,
                after_slice0=hooks[0] if hooks else None,
                dec_dim=dec_dim,
                c=c,
            )

        if c == 2 * dec_dim:
            # Fused decoder output = concat(last, second-last) along channel (2 * dec_dim).
            half_a, half_b = torch.chunk(hooks[0], 2, dim=-1)
            if n_scales == 4:
                return [half_a, half_b, half_a, half_b]
            if n_scales == 2:
                return [half_a, half_b]
            return [half_a] * n_scales

        if c == dec_dim:
            if len(hooks) >= n_scales:
                return hooks[:n_scales]
            if len(hooks) == 1:
                return [hooks[0]] * n_scales
            out = list(hooks)
            while len(out) < n_scales:
                out.append(out[-1])
            return out[:n_scales]

        raise RuntimeError(
            f"Unexpected hook feature dim {c} (expected {dec_dim} or {2 * dec_dim}), "
            f"num_hooks={len(hook_list)}"
        )

    def _geometry_point_mask(
        self,
        local_points: Optional[torch.Tensor],
        global_points: torch.Tensor,
        view_idx: int,
    ) -> torch.Tensor:
        """Same validity rule as merged geometry PLY export (finite + depth > 1e-3).

        Returns (B, H, W) when all batch items share the same ``view_idx``.
        """
        if local_points is not None:
            pts = local_points[:, view_idx]
        else:
            pts = global_points[:, view_idx]
        return torch.isfinite(pts).all(dim=-1) & (pts.norm(dim=-1) > 1e-3)

    def _geometry_point_mask_for_anchor_batch(
        self,
        local_points: Optional[torch.Tensor],
        global_points: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """(B, H, W) mask with a possibly different anchor view per batch item."""
        masks: List[torch.Tensor] = []
        for i in range(int(anchor_idx.shape[0])):
            vi = int(anchor_idx[i].item())
            if local_points is not None:
                pts = local_points[i, vi]
            else:
                pts = global_points[i, vi]
            masks.append(
                torch.isfinite(pts).all(dim=-1) & (pts.norm(dim=-1) > 1e-3)
            )
        return torch.stack(masks, dim=0)

    def _run_gs_head_for_view(
        self,
        *,
        view_idx: int,
        b: int,
        global_points: torch.Tensor,
        hook_list: List[torch.Tensor],
        images: torch.Tensor,
        local_points: Optional[torch.Tensor],
        patch_start_idx: int,
        means: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DPT GS head for one view (raw gs_out, before adapter)."""
        device = images.device
        view_idx_tensor = torch.full((b,), view_idx, device=device, dtype=torch.long)
        if means is None:
            means = global_points[:, view_idx]
        hooks = self._prepare_dpt_encoder_tokens(hook_list, view_idx_tensor)
        images_view = images[:, view_idx]
        images_rgb = self._denormalize_images(images_view)
        point_mask = self._geometry_point_mask(local_points, global_points, view_idx)

        hooks_fp32 = [h.float() for h in hooks]
        gs_out = self.gaussian_param_head(
            hooks_fp32,
            images_rgb.float(),
            patch_start_idx=patch_start_idx,
        )
        return gs_out, means.float(), images_rgb, point_mask

    def _run_gaussian_adapter_for_view(
        self,
        *,
        view_idx: int,
        b: int,
        global_points: torch.Tensor,
        hook_list: List[torch.Tensor],
        images: torch.Tensor,
        local_points: Optional[torch.Tensor],
        patch_start_idx: int,
    ) -> tuple[torch.Tensor, List[GaussianAdapterOutput]]:
        gs_out, means, images_rgb, point_mask = self._run_gs_head_for_view(
            view_idx=view_idx,
            b=b,
            global_points=global_points,
            hook_list=hook_list,
            images=images,
            local_points=local_points,
            patch_start_idx=patch_start_idx,
        )
        adapter_out = build_gaussian_cloud_batch(
            self.gaussian_adapter,
            self.gs_view_type,
            means=means,
            gs_out=gs_out,
            images=images_rgb,
            global_step=self.global_step,
            stride=self.gs_stride,
            detach_means=self.gs_detach_means,
            point_mask=point_mask,
        )
        return gs_out, adapter_out

    def _build_gaussian_outputs(
        self,
        *,
        b: int,
        s: int,
        global_points: torch.Tensor,
        hook_list: List[torch.Tensor],
        images: torch.Tensor,
        local_points: Optional[torch.Tensor],
        patch_start_idx: int,
        anchor_idx: torch.Tensor,
        log_dbg: bool,
        camera_poses: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, List[GaussianAdapterOutput]]:
        if self.gs_means_mode == "anchor":
            batch_idx = torch.arange(b, device=images.device)
            means_anchor = global_points[batch_idx, anchor_idx]
            hooks_anchor = self._prepare_dpt_encoder_tokens(hook_list, anchor_idx)
            images_anchor = torch.stack(
                [images[i, int(anchor_idx[i].item())] for i in range(b)], dim=0
            )
            images_rgb = self._denormalize_images(images_anchor)
            point_mask = self._geometry_point_mask_for_anchor_batch(
                local_points, global_points, anchor_idx
            )
            if log_dbg:
                gs_debug_hook_list("hooks_anchor(dpt_in)", hooks_anchor)
                gs_debug_print(
                    "gs.images",
                    images_anchor=images_anchor,
                    images_rgb=images_rgb,
                )
            hooks_fp32 = [h.float() for h in hooks_anchor]
            gs_out = self.gaussian_param_head(
                hooks_fp32,
                images_rgb.float(),
                patch_start_idx=patch_start_idx,
            )
            adapter_out = build_gaussian_cloud_batch(
                self.gaussian_adapter,
                self.gs_view_type,
                means=means_anchor.float(),
                gs_out=gs_out,
                images=images_rgb,
                global_step=self.global_step,
                stride=self.gs_stride,
                detach_means=self.gs_detach_means,
                point_mask=point_mask,
            )
            if log_dbg:
                gs_debug_print("gs.forward.out", gs_raw=gs_out, gs_means_mode="anchor")
            return gs_out, adapter_out, None

        per_batch_view_outs: List[List[GaussianAdapterOutput]] = [
            [] for _ in range(b)
        ]
        gs_out_views: List[torch.Tensor] = []
        adapter = self.gaussian_adapter
        if camera_poses is None:
            raise RuntimeError(
                "merged gs_means_mode requires camera_poses for local→world merge"
            )
        for vi in range(s):
            means_v = batched_view_local_means(
                vi,
                local_points=local_points,
                global_points=global_points,
            )
            gs_out_v, means_v, images_rgb_v, point_mask_v = self._run_gs_head_for_view(
                view_idx=vi,
                b=b,
                global_points=global_points,
                hook_list=hook_list,
                images=images,
                local_points=local_points,
                patch_start_idx=patch_start_idx,
                means=means_v,
            )
            gs_out_views.append(gs_out_v)
            gs_out_p, means_p, rgb_grid, mask = adapter._preprocess_view_gs(
                means_v,
                gs_out_v,
                images_rgb_v,
                point_mask_v,
                self.gs_stride,
                detach_means=self.gs_detach_means,
            )
            for bi in range(b):
                flat = adapter.flat_masked_gs_batch_item(
                    gs_out_p, means_p, rgb_grid, mask, bi
                )
                view_out = adapter.build_gaussian_output_from_flat(
                    flat[0], flat[1], flat[2], self.global_step
                )
                c2w = camera_poses[bi, vi].float()
                per_batch_view_outs[bi].append(
                    transform_gaussian_adapter_output_c2w(view_out, c2w)
                )
            del gs_out_v, means_v, images_rgb_v, point_mask_v, gs_out_p, means_p
            if vi + 1 < s and torch.cuda.is_available():
                torch.cuda.empty_cache()

        merged_out: List[GaussianAdapterOutput] = []
        for bi in range(b):
            views = per_batch_view_outs[bi]
            if not views:
                empty = global_points.new_zeros((0, 3))
                merged_out.append(
                    adapter.build_gaussian_output_from_flat(
                        global_points.new_zeros((0, GS_RAW_FEAT_DIM)),
                        empty,
                        empty,
                        self.global_step,
                    )
                )
            elif len(views) == 1:
                merged_out.append(views[0])
            else:
                merged_out.append(merge_gaussian_adapter_outputs(views))
        gs_out = gs_out_views[int(anchor_idx[0].item())] if gs_out_views else gs_out_views[0]
        if log_dbg:
            gs_debug_print(
                "gs.forward.out",
                gs_means_mode="merged",
                num_views=s,
                total_gaussians=sum(int(o.means.shape[0]) for o in merged_out),
            )
        return gs_out, merged_out, gs_out_views

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        # Normalize to (B, S, 3, H, W) before super() so aggregator and GS share batch size.
        if images.dim() == 4:
            if images.shape[1] != 3:
                raise ValueError(
                    f"Expected (S, 3, H, W) for a single sequence, got {tuple(images.shape)}"
                )
            images = images.unsqueeze(0)

        predictions = super().forward(images, query_points=query_points)

        if not self.enable_gaussian:
            return predictions

        b, s, _, h, w = images.shape
        log_dbg = self._should_log_gs_debug()

        dec_dim = self.aggregator.dec_embed_dim
        hook_list = predictions.get("aggregated_hooks")
        if hook_list is None:
            raise RuntimeError(
                "aggregated_hooks missing; ensure Aggregator returns hook tokens."
            )
        refreshed_hooks = False
        if hook_list[0].shape[-1] == 2 * dec_dim:
            if log_dbg:
                gs_debug_print(
                    "gs.refresh_hooks",
                    reason="aggregated_hooks look like fused tokens (2*dec_dim)",
                    bad_hook0=hook_list[0],
                    dec_dim=dec_dim,
                )
            hook_list, _, _, _ = self.aggregator(images)
            refreshed_hooks = True

        anchor_idx = self._resolve_anchor_idx(b, s, device=images.device)
        predictions["gs_anchor_idx"] = anchor_idx
        predictions["gs_means_mode"] = self.gs_means_mode

        global_points = predictions.get("global_points")
        if global_points is None:
            raise RuntimeError("global_points required for Gaussian branch")

        local_points = predictions.get("local_points")
        camera_poses = predictions.get("camera_poses")
        patch_start_idx = getattr(self.aggregator, "patch_start_idx", 0)

        if log_dbg:
            batch_idx = torch.arange(b, device=images.device)
            means_anchor = global_points[batch_idx, anchor_idx]
            gs_debug_print(
                "gs.forward",
                step=self.global_step,
                images=images,
                b=b,
                s=s,
                h=h,
                w=w,
                dec_dim=dec_dim,
                refreshed_hooks=refreshed_hooks,
                anchor_idx=anchor_idx,
                gs_means_mode=self.gs_means_mode,
                global_points=global_points,
                means_anchor=means_anchor,
                patch_start_idx=patch_start_idx,
            )
            gs_debug_hook_list("aggregated_hooks(raw)", hook_list)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            gs_out, adapter_out, gs_raw_views = self._build_gaussian_outputs(
                b=b,
                s=s,
                global_points=global_points,
                hook_list=hook_list,
                images=images,
                local_points=local_points,
                patch_start_idx=patch_start_idx,
                anchor_idx=anchor_idx,
                log_dbg=log_dbg,
                camera_poses=camera_poses,
            )
            if is_gs_grad_debug_enabled() and torch.is_grad_enabled():
                c = adapter_out[0].cloud if adapter_out else None
                if c is not None:
                    register_grad_hook(
                        c._scaling, "cloud.scaling_log", step=self.global_step
                    )
                    register_grad_hook(
                        c._opacity, "cloud.opacity_logit", step=self.global_step
                    )
                    register_grad_hook(
                        c._rotation, "cloud.rotation", step=self.global_step
                    )
                    register_grad_hook(
                        c._features_dc, "cloud.features_dc", step=self.global_step
                    )
                if gs_grad_verbose_should_run(self.global_step):
                    safe_retain_grad(gs_out)
                    register_grad_hook(
                        gs_out, "gs_out", step=self.global_step, verbose=True
                    )
                    gs_grad_print(
                        "gs_forward",
                        gs_out=gs_out,
                        step=self.global_step,
                        verbose=True,
                    )
                    if c is not None:
                        report_cloud(
                            "adapter_cloud", c, step=self.global_step
                        )
        predictions["gs_raw"] = gs_out
        predictions["gs_conf"] = gs_out[:, 11:12]
        if gs_raw_views is not None:
            predictions["gs_raw_views"] = gs_raw_views
        predictions["gaussian_adapter_out"] = adapter_out
        predictions["gaussians"] = [o.cloud for o in adapter_out]
        predictions["gs_global_step"] = self.global_step
        predictions["gs_stride"] = self.gs_stride
        predictions["_gs_adapter_ref"] = self.gaussian_adapter

        return predictions
