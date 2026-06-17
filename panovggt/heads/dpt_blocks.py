"""DPT fusion blocks (adapted from AnySplat / DPT)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > int_max:
        chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
        return torch.cat(
            [
                F.interpolate(c, size=size, mode=mode, align_corners=align_corners)
                for c in chunks
            ],
            dim=0,
        ).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation, groups: int = 1):
        super().__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation,
        has_residual: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.res_conf_unit1 = ResidualConvUnit(features, activation, groups=groups)
        self.res_conf_unit2 = ResidualConvUnit(features, activation, groups=groups)
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True, groups=groups)

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            output = output + self.res_conf_unit1(xs[1])
        output = self.res_conf_unit2(output)
        if size is None:
            output = custom_interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            output = custom_interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), has_residual=has_residual)


def _make_scratch(in_shape: List[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, 3, 1, 1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, 3, 1, 1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, 3, 1, 1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, 3, 1, 1, bias=False)
    return scratch
