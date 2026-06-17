"""LPIPS helpers for inference / evaluation (AlexNet, inputs in [0, 1])."""

from __future__ import annotations

from typing import Optional, Union

import torch

_lpips_model: Optional[Union[torch.nn.Module, bool]] = None
_lpips_backend: Optional[str] = None


def _load_lpips_model(device: torch.device) -> tuple[Optional[torch.nn.Module], Optional[str]]:
    try:
        import lpips as lpips_pkg

        model = lpips_pkg.LPIPS(net="alex").to(device).eval()
        return model, "lpips"
    except Exception:
        pass

    for import_path in (
        "kornia.losses",
        "kornia.metrics",
    ):
        try:
            mod = __import__(import_path, fromlist=["LPIPS"])
            if hasattr(mod, "LPIPS"):
                model = mod.LPIPS().to(device).eval()
                return model, "kornia"
        except Exception:
            continue

    return None, None


def get_lpips_model(device: torch.device) -> Optional[torch.nn.Module]:
    global _lpips_model, _lpips_backend
    if _lpips_model is None:
        model, backend = _load_lpips_model(device)
        if model is None:
            _lpips_model = False
            _lpips_backend = None
        else:
            _lpips_model = model
            _lpips_backend = backend
    if _lpips_model is False:
        return None
    return _lpips_model  # type: ignore[return-value]


def lpips_backend_name() -> Optional[str]:
    return _lpips_backend


def _prepare_lpips_inputs(
    render: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r = render.unsqueeze(0) if render.dim() == 3 else render
    g = gt.unsqueeze(0) if gt.dim() == 3 else gt
    if _lpips_backend == "lpips":
        r = r * 2.0 - 1.0
        g = g * 2.0 - 1.0
    return r, g.detach()


def lpips_loss(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Differentiable LPIPS between two RGB images in [0, 1]. Returns scalar tensor.
    """
    model = get_lpips_model(render.device)
    if model is None:
        return render.new_zeros(())

    r, g = _prepare_lpips_inputs(render, gt)
    val = model(r, g)
    return val.mean()


def compute_lpips(render: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute LPIPS between two RGB images in [0, 1].

    Args:
        render: (3, H, W) or (B, 3, H, W)
        gt: same shape as render
    """
    model = get_lpips_model(render.device)
    if model is None:
        return float("nan")

    with torch.no_grad():
        val = lpips_loss(render, gt)
    return float(val.item())
