"""Spherical harmonics helpers for Gaussian color."""

import torch

C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Map RGB in [0,1] to SH DC coefficients."""
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5


def make_sh_mask(sh_degree: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Per-coefficient mask; higher degrees start small."""
    n = (sh_degree + 1) ** 2
    mask = torch.ones(n, device=device, dtype=dtype)
    for degree in range(1, sh_degree + 1):
        mask[degree**2 : (degree + 1) ** 2] = 0.1 * (0.25**degree)
    return mask
