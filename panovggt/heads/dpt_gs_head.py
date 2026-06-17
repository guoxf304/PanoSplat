"""DPT-style dense Gaussian parameter head for panoramic feed-forward GS."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.heads.dpt_blocks import (
    _make_fusion_block,
    _make_scratch,
    custom_interpolate,
)
from panovggt.utils.gs_debug import gs_debug_print, is_gs_debug_enabled


def _create_uv_grid(patch_w: int, patch_h: int, aspect_ratio: float, device, dtype):
    u = torch.linspace(0, 1, patch_w, device=device, dtype=dtype)
    v = torch.linspace(0, 1, patch_h, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    grid_u = (grid_u - 0.5) * 2.0
    grid_v = (grid_v - 0.5) * 2.0 * aspect_ratio
    return torch.stack([grid_u, grid_v], dim=-1)


def _position_grid_to_embed(
    grid: torch.Tensor, dim: int, max_bands: int = 16
) -> torch.Tensor:
    """Fourier features from 2D grid -> dim channels.

    Use a fixed small number of frequency bands.  The old ``dim // 4`` bands made
    ``2**(dim//4 - 1)`` blow up for large ``dim`` (e.g. 512ch -> 2^127 -> inf in fp32).
    """
    num_bands = min(max(dim // 4, 1), max_bands)
    freq = torch.arange(num_bands, device=grid.device, dtype=grid.dtype)
    freq = (2.0**freq) * torch.pi
    emb = []
    for f in freq:
        emb.append(torch.sin(f * grid[..., 0:1]))
        emb.append(torch.cos(f * grid[..., 0:1]))
        emb.append(torch.sin(f * grid[..., 1:2]))
        emb.append(torch.cos(f * grid[..., 1:2]))
    out = torch.cat(emb, dim=-1)
    if out.shape[-1] < dim:
        out = F.pad(out, (0, dim - out.shape[-1]))
    return out[..., :dim]


class PanoDPT_GS_Head(nn.Module):
    """
    Multi-scale DPT head predicting per-pixel Gaussian parameters + confidence.

    Output channels (default 12):
        0: density, 1-3: scale_xyz, 4-7: quaternion, 8-10: sh0, 11: conf
    """

    def __init__(
        self,
        dim_in: int = 1024,
        patch_size: int = 14,
        output_dim: int = 12,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        intermediate_layer_idx: Optional[List[int]] = None,
        pos_embed: bool = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx or [8, 18, 27, 35]
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
                for oc in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        merge_dim = head_features_1 // 2
        head_features_2 = 128 if output_dim > 50 else 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, merge_dim, 3, 1, 1
        )
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, merge_dim, 7, 1, 3),
            nn.ReLU(inplace=True),
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(merge_dim, head_features_2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, 1, 1, 0),
        )

    def _apply_pos_embed(self, x: torch.Tensor, w: int, h: int, ratio: float = 0.1):
        ph, pw = x.shape[-2], x.shape[-1]
        grid = _create_uv_grid(pw, ph, w / h, x.device, x.dtype)
        pe = _position_grid_to_embed(grid, x.shape[1])
        pe = pe.permute(2, 0, 1).unsqueeze(0) * ratio
        return x + pe.expand(x.shape[0], -1, -1, -1)

    def _reduce_token_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Map fused decoder tokens (2 * dim_in) to hook dim for 1x1 project convs."""
        c = x.shape[-1]
        if c == self.dim_in:
            return x
        if c == 2 * self.dim_in:
            return x[..., : self.dim_in]
        raise RuntimeError(
            f"PanoDPT_GS_Head expects token dim {self.dim_in} or {2 * self.dim_in}, got {c}"
        )

    def _reduce_spatial_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Align NCHW features to dim_in (handles missed token-level reduction)."""
        c = x.shape[1]
        if c == self.dim_in:
            return x
        if c == 2 * self.dim_in:
            return x[:, : self.dim_in]
        raise RuntimeError(
            f"PanoDPT_GS_Head spatial channels {c}, expected {self.dim_in} or {2 * self.dim_in}"
        )

    def scratch_forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = feats
        l1 = self.scratch.layer1_rn(l1)
        l2 = self.scratch.layer2_rn(l2)
        l3 = self.scratch.layer3_rn(l3)
        l4 = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1)
        return self.scratch.output_conv1(out)

    def _forward_impl(
        self,
        encoder_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        b_img, _, h, w = images.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        if is_gs_debug_enabled():
            gs_debug_print(
                "dpt_gs_head.enter",
                images=images,
                dim_in=self.dim_in,
                patch_start_idx=patch_start_idx,
                patch_grid=(patch_h, patch_w),
                num_scales=len(encoder_tokens_list),
                enc0=encoder_tokens_list[0] if encoder_tokens_list else None,
            )
        feats = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            list_idx = min(dpt_idx, len(encoder_tokens_list) - 1)
            x = encoder_tokens_list[list_idx]
            if x.dim() == 4:
                if x.shape[1] == 1:
                    x = x[:, 0]
                else:
                    raise ValueError(
                        f"Expected anchor-sliced tokens (B, P, C), got {tuple(x.shape)}"
                    )
            x = x[:, patch_start_idx:]
            x = self._reduce_token_channels(x)
            if is_gs_debug_enabled() and dpt_idx == 0:
                gs_debug_print(
                    "dpt_gs_head.scale0_tokens",
                    list_idx=list_idx,
                    tokens=x,
                    b_img=b_img,
                )
            bt = x.shape[0]
            if bt != b_img:
                raise RuntimeError(
                    f"Token batch {bt} != anchor image batch {b_img}; "
                    "check aggregated_hooks and anchor indexing."
                )
            n_patch = patch_h * patch_w
            if x.shape[1] != n_patch:
                raise RuntimeError(
                    f"Token count {x.shape[1]} != patch grid {n_patch} "
                    f"({patch_h}x{patch_w})"
                )
            x = x.reshape(bt, n_patch, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(bt, -1, patch_h, patch_w)
            x = self._reduce_spatial_channels(x)
            if is_gs_debug_enabled() and dpt_idx == 0:
                gs_debug_print(
                    "dpt_gs_head.scale0_spatial",
                    spatial=x,
                    project_in=self.dim_in,
                    project_out=self.projects[dpt_idx].out_channels,
                )
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, w, h)
            x = self.resize_layers[dpt_idx](x)
            feats.append(x)

        out = self.scratch_forward(feats)
        out = custom_interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        direct = self.input_merger(images)
        out = out + direct
        if self.pos_embed:
            out = self._apply_pos_embed(out, w, h)
        return self.scratch.output_conv2(out)

    def forward(
        self,
        encoder_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int = 5,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_tokens_list: list of (B, P, C) or (B, 1, P, C) after anchor slice.
            images: (B, 3, H, W) anchor frame, values in [0, 1] or normalized — caller denorms.
            patch_start_idx: register token count.

        Returns:
            (B, output_dim, H, W)
        """
        if images.dim() == 5:
            images = images[:, 0]
        return self._forward_impl(encoder_tokens_list, images, patch_start_idx)
