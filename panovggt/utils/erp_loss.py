"""ERP-aware photometric losses (latitude-weighted SSIM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def est_wsmap(img: torch.Tensor) -> torch.Tensor:
    """Latitude cosine weights for equirectangular images."""
    h, w = img.shape[-2:]
    col = torch.arange(h, device=img.device, dtype=img.dtype)
    ws_map = torch.cos((col + 0.5 - h / 2) * torch.pi / h).reshape(h, 1).expand(h, w)
    return ws_map


def _gaussian(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    ws_map: torch.Tensor | None,
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1_sq = sigma1_sq.clamp(min=0.0)
    sigma2_sq = sigma2_sq.clamp(min=0.0)
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denom = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = (num / denom.clamp(min=1e-8)).clamp(0.0, 1.0)
    if ws_map is None:
        return ssim_map.mean()
    # Average RGB before latitude weighting (same as unweighted mean over C,H,W).
    ssim_spatial = ssim_map.mean(dim=1)
    ws = ws_map.to(device=ssim_map.device, dtype=ssim_map.dtype)
    if ws.dim() == 2:
        ws = ws.unsqueeze(0)
    weighted = (ssim_spatial * ws).sum() / ws.sum().clamp(min=1e-8)
    return weighted


def ssim_erp(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM with ERP latitude weighting. Inputs: (B, 3, H, W). Returns value in [0, 1]."""
    channel = img1.size(-3)
    device, dtype = img1.device, img1.dtype
    g1d = _gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    window = (g1d @ g1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    ws_map = est_wsmap(img1[0])
    # Conv SSIM can exceed 1.0 slightly; clamp so (1 - SSIM) loss stays non-negative.
    return _ssim(img1, img2, window, window_size, channel, ws_map).clamp(0.0, 1.0)


def dssim_erp(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural dissimilarity: 1 - SSIM, always in [0, 1] when SSIM is clamped."""
    return 1.0 - ssim_erp(img1, img2, window_size=window_size)


def l1_erp(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 with ERP latitude weights."""
    ws = est_wsmap(pred)
    ws = ws / ws.mean().clamp(min=1e-8)
    if ws.dim() == 2:
        ws = ws.view(1, 1, *ws.shape)
    return (torch.abs(pred - gt) * ws).mean()


def _psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
    mse_f = float(mse.detach().float().item())
    if mse_f <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_val**2) / mse_f).item())


def _erp_sin_v_weights(pred: torch.Tensor) -> torch.Tensor:
    """Per-pixel sin(v) weights (v = colatitude); equals cos(latitude) on ERP rows."""
    ws = est_wsmap(pred)
    if pred.dim() == 3:
        return ws
    if ws.dim() == 2:
        return ws.unsqueeze(0)
    return ws


def psnr_uniform(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Standard (uniform pixel) PSNR in dB."""
    return _psnr_from_mse(((pred - gt) ** 2).mean(), max_val=max_val)


def psnr_erp(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """ERP sin(v) / cos(latitude) weighted PSNR in dB."""
    ws = _erp_sin_v_weights(pred).to(device=pred.device, dtype=pred.dtype)
    err2 = (pred - gt) ** 2
    if err2.dim() == 3:
        err2 = err2.mean(dim=0)
    mse = (err2 * ws).sum() / ws.sum().clamp(min=1e-8)
    return _psnr_from_mse(mse, max_val=max_val)
