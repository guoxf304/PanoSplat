"""ERP camera utilities for ODGS omnidirectional rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


from panovggt.render.coord_frame import (
    camera_center_from_w2c,
    w2c_to_world_view_transform,
)


def _c2w_to_world_view_transform(c2w: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world (4x4) to ODGS world_view_transform (w2c, glm layout)."""
    w2c = torch.linalg.inv(c2w)
    return w2c.transpose(-1, -2).contiguous()


@dataclass
class ERPPanoCamera:
    """Minimal camera bundle consumed by ODGS GaussianRasterizer."""

    image_height: int
    image_width: int
    world_view_transform: torch.Tensor
    camera_center: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.world_view_transform.device


def build_erp_camera_from_w2c(
    w2c: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | None = None,
) -> ERPPanoCamera:
    """
    Build ERP camera from w2c extrinsics (SynPano / training batch convention).

    Avoids c2w -> inv round-trip and matches ODGS ``getWorld2View2`` layout.
    """
    if w2c.dim() == 4:
        w2c = w2c[0]
    if device is not None:
        w2c = w2c.to(device=device, dtype=torch.float32)
    else:
        w2c = w2c.float()

    world_view_transform = w2c_to_world_view_transform(w2c)
    cam_center = camera_center_from_w2c(w2c)
    return ERPPanoCamera(
        image_height=int(height),
        image_width=int(width),
        world_view_transform=world_view_transform,
        camera_center=cam_center,
    )


def build_erp_camera(
    c2w: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | None = None,
) -> ERPPanoCamera:
    """
    Build an ERP panorama camera from a c2w pose.

    Args:
        c2w: (4, 4) or (B, 4, 4) camera-to-world matrix.
        height: ERP image height.
        width: ERP image width.
        device: Target device.

    Returns:
        ERPPanoCamera (uses first batch item if batched).
    """
    if c2w.dim() == 3:
        c2w = c2w[0]
    if device is not None:
        c2w = c2w.to(device=device, dtype=torch.float32)
    else:
        c2w = c2w.float()

    w2c = torch.linalg.inv(c2w)
    return build_erp_camera_from_w2c(w2c, height, width, device=c2w.device)


def colmap_w2c_from_c2w(c2w: Union[torch.Tensor, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP-style R (transposed) and T from c2w, for numpy pipelines."""
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    w2c = np.linalg.inv(c2w)
    r = w2c[:3, :3].T
    t = w2c[:3, 3]
    return r.astype(np.float32), t.astype(np.float32)
