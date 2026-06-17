"""Optional debug prints for PanoVGGT Gaussian-splatting forward and gradient checks."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

_GS_DEBUG: Optional[bool] = None
_GS_GRAD_DEBUG: Optional[bool] = None
_GS_GRAD_DEBUG_COUNTER = 0

# Backward hooks always logged at core frequency (geometry grad health).
_CORE_BACKWARD_HOOK_NAMES = frozenset(
    {
        "cloud.rotation",
        "cloud.scaling_log",
        "cloud.opacity_logit",
        "cloud.features_dc",
    }
)


def set_gs_debug(enabled: bool) -> None:
    """Enable or disable GS debug prints globally."""
    global _GS_DEBUG
    _GS_DEBUG = bool(enabled)


def is_gs_debug_enabled() -> bool:
    """True if env PANOSPLAT_GS_DEBUG=1 or set_gs_debug(True) was called."""
    global _GS_DEBUG
    if _GS_DEBUG is not None:
        return _GS_DEBUG
    return os.environ.get("PANOSPLAT_GS_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_dist_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    if torch.is_tensor(value):
        return (
            f"Tensor{tuple(value.shape)}"
            f" dtype={value.dtype} device={value.device}"
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        if all(torch.is_tensor(x) for x in value):
            head = ", ".join(f"{i}:{tuple(x.shape)}" for i, x in enumerate(value[:6]))
            suffix = "..." if len(value) > 6 else ""
            return f"[{head}{suffix}]"
        return repr(value)
    if isinstance(value, torch.Size):
        return f"Size{tuple(value)}"
    return repr(value)


def gs_debug_print(tag: str, rank0_only: bool = True, **fields: Any) -> None:
    """
    Print one line of GS debug info when debug is enabled.

    Example:
        gs_debug_print("gs.forward", images=images, hooks=hook_list[0])
    """
    if not is_gs_debug_enabled():
        return
    if rank0_only and get_dist_rank() != 0:
        return
    parts = [f"[PanoSplat-GS] {tag}"]
    for key, val in fields.items():
        parts.append(f"{key}={_fmt(val)}")
    print(" | ".join(parts), flush=True)


def set_gs_grad_debug(enabled: bool) -> None:
    """Enable gradient debug prints globally."""
    global _GS_GRAD_DEBUG
    _GS_GRAD_DEBUG = bool(enabled)


def is_gs_grad_debug_enabled() -> bool:
    """True if env PANOSPLAT_GS_GRAD_DEBUG=1 or set_gs_grad_debug(True) was called."""
    global _GS_GRAD_DEBUG
    if _GS_GRAD_DEBUG is not None:
        return _GS_GRAD_DEBUG
    return os.environ.get("PANOSPLAT_GS_GRAD_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gs_grad_debug_every() -> int:
    """Core summary frequency (default: every step)."""
    return max(1, int(os.environ.get("PANOSPLAT_GS_GRAD_DEBUG_EVERY", "1")))


def gs_grad_verbose_every() -> int:
    """Verbose tensor stats frequency (default: every 50 steps)."""
    return max(1, int(os.environ.get("PANOSPLAT_GS_GRAD_VERBOSE_EVERY", "50")))


def _grad_debug_rank0() -> bool:
    return is_gs_grad_debug_enabled() and get_dist_rank() == 0


def gs_grad_core_should_run(step: Optional[int] = None) -> bool:
    """Every-step (or PANOSPLAT_GS_GRAD_DEBUG_EVERY) core grad logs on rank 0."""
    if not _grad_debug_rank0():
        return False
    every = gs_grad_debug_every()
    if step is not None:
        return step % every == 0
    global _GS_GRAD_DEBUG_COUNTER
    _GS_GRAD_DEBUG_COUNTER += 1
    return (_GS_GRAD_DEBUG_COUNTER - 1) % every == 0


def gs_grad_verbose_should_run(step: Optional[int] = None) -> bool:
    """Detailed tensor stats (PANOSPLAT_GS_GRAD_VERBOSE_EVERY, default 50)."""
    if not _grad_debug_rank0() or step is None:
        return False
    return step % gs_grad_verbose_every() == 0


def gs_grad_debug_should_run(step: Optional[int] = None) -> bool:
    """Backward-compatible alias: True when core OR verbose would print."""
    return gs_grad_core_should_run(step) or gs_grad_verbose_should_run(step)


def tensor_stats(t: Optional[torch.Tensor]) -> Dict[str, Any]:
    if t is None:
        return {"present": False}
    if not torch.is_tensor(t):
        return {"present": True, "type": type(t).__name__}
    with torch.no_grad():
        finite = torch.isfinite(t)
        n = t.numel()
        n_finite = int(finite.sum().item()) if n > 0 else 0
        out: Dict[str, Any] = {
            "present": True,
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "requires_grad": bool(t.requires_grad),
            "finite_ratio": (n_finite / n) if n > 0 else 1.0,
            "numel": n,
        }
        if n_finite > 0:
            tf = t.detach()[finite]
            out["min"] = float(tf.min().item())
            out["max"] = float(tf.max().item())
            out["mean"] = float(tf.mean().item())
        else:
            out["min"] = out["max"] = out["mean"] = None
        if n > n_finite:
            out["non_finite"] = n - n_finite
    return out


def _stats_line(tag: str, stats: Dict[str, Any]) -> str:
    if not stats.get("present", False):
        return f"{tag}: <missing>"
    if stats.get("finite_ratio", 1.0) < 1.0:
        fin = f"finite={stats['finite_ratio']:.4f} NON_FINITE={stats.get('non_finite', '?')}"
    else:
        fin = "finite=1.0"
    rg = "grad" if stats.get("requires_grad") else "no_grad"
    rng = ""
    if stats.get("min") is not None:
        rng = f" min={stats['min']:.4g} max={stats['max']:.4g} mean={stats['mean']:.4g}"
    return f"{tag}: {stats.get('shape')} {stats.get('dtype')} {rg} {fin}{rng}"


def _grad_head(step: Optional[int]) -> str:
    return (
        f"[PanoSplat-GS-GRAD] step={step}"
        if step is not None
        else "[PanoSplat-GS-GRAD]"
    )


def gs_grad_print(
    tag: str, step: Optional[int] = None, *, verbose: bool = False, **tensors: Any
) -> None:
    """Print grad debug block. verbose=True uses PANOSPLAT_GS_GRAD_VERBOSE_EVERY."""
    if verbose:
        if not gs_grad_verbose_should_run(step):
            return
    elif not gs_grad_core_should_run(step):
        return
    print(f"{_grad_head(step)} | {tag}", flush=True)
    for name, val in tensors.items():
        if torch.is_tensor(val):
            print(f"  {_stats_line(name, tensor_stats(val))}", flush=True)
        elif isinstance(val, dict):
            print(f"  {name}: {val}", flush=True)
        else:
            print(f"  {name}: {val}", flush=True)


def report_cloud(tag: str, cloud, step: Optional[int] = None) -> None:
    """Full cloud tensor stats (verbose cadence only)."""
    if not gs_grad_verbose_should_run(step):
        return
    gs_grad_print(
        tag,
        step=step,
        verbose=True,
        xyz=cloud.get_xyz,
        scaling_log=cloud._scaling,
        scaling=cloud.get_scaling,
        rotation=cloud.get_rotation,
        opacity_logit=cloud._opacity,
        opacity=cloud.get_opacity,
        features_dc=cloud._features_dc,
    )


def safe_retain_grad(t: Optional[torch.Tensor]) -> None:
    """retain_grad only when the tensor participates in autograd (train, grad enabled)."""
    if torch.is_tensor(t) and t.requires_grad and torch.is_grad_enabled():
        t.retain_grad()


def _print_backward_hook(name: str, grad: Optional[torch.Tensor]) -> None:
    st = tensor_stats(grad)
    if st.get("finite_ratio", 1.0) < 1.0 or grad is None:
        print(
            f"  [backward] {name}: grad {st} "
            f"({'None' if grad is None else 'BAD'})",
            flush=True,
        )
    else:
        print(
            f"  [backward] {name}: ok max_abs={st.get('max', 0):.4g}",
            flush=True,
        )


def register_grad_hook(
    t: torch.Tensor,
    name: str,
    step: Optional[int] = None,
    *,
    verbose: Optional[bool] = None,
) -> None:
    """
    Register a backward hook that prints grad health.

    verbose=None (default): core cadence for cloud.* geometry tensors, else verbose cadence.
  """
    if not is_gs_grad_debug_enabled() or not t.requires_grad:
        return
    if verbose is None:
        verbose = name not in _CORE_BACKWARD_HOOK_NAMES

    def _hook(grad: torch.Tensor) -> None:
        if verbose:
            if not gs_grad_verbose_should_run(step):
                return
        elif not gs_grad_core_should_run(step):
            return
        _print_backward_hook(name, grad)

    t.register_hook(_hook)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def report_gaussian_head_gradients(
    model: torch.nn.Module,
    module_substr: str = "gaussian_param_head",
    step: Optional[int] = None,
    max_lines: int = 12,
    *,
    detail: bool = False,
) -> Dict[str, int]:
    """Summarize grads for trainable GS head params. Returns counts."""
    if not gs_grad_core_should_run(step):
        return {}

    model = _unwrap_model(model)
    no_grad: List[str] = []
    zero_grad: List[str] = []
    nan_grad: List[Tuple[str, int]] = []
    ok_grad: List[Tuple[str, float]] = []

    for name, p in model.named_parameters():
        if module_substr not in name or not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            nan_grad.append((name, int((~torch.isfinite(p.grad)).sum().item())))
            continue
        gmax = float(p.grad.abs().max().item())
        if gmax == 0.0:
            zero_grad.append(name)
        else:
            ok_grad.append((name, gmax))

    counts = {
        "ok": len(ok_grad),
        "zero": len(zero_grad),
        "nan": len(nan_grad),
        "none": len(no_grad),
    }
    gs_grad_print(
        f"param_grads[{module_substr}]",
        step=step,
        summary=counts,
    )
    if not detail:
        return counts

    for name, cnt in nan_grad[:max_lines]:
        print(f"  NAN: {name} ({cnt} bad elements)", flush=True)
    if len(nan_grad) > max_lines:
        print(f"  ... and {len(nan_grad) - max_lines} more NAN params", flush=True)
    for name in no_grad[:max_lines]:
        print(f"  NO_GRAD: {name}", flush=True)
    for name in zero_grad[:max_lines]:
        print(f"  ZERO: {name}", flush=True)
    for name, gmax in sorted(ok_grad, key=lambda x: -x[1])[:5]:
        print(f"  OK: {name} max_abs={gmax:.4g}", flush=True)
    return counts


def report_after_backward(
    model: torch.nn.Module,
    pred: Dict[str, Any],
    loss_dict: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Post-backward: core head summary every step; tensor dumps on verbose steps."""
    if not is_gs_grad_debug_enabled() or get_dist_rank() != 0:
        return

    verbose = gs_grad_verbose_should_run(step)
    report_gaussian_head_gradients(model, step=step, detail=verbose)

    if not verbose:
        return

    loss_obj = loss_dict.get("loss_objective")
    loss_rgb = loss_dict.get("loss_rgb")
    gs_grad_print(
        "after_backward",
        step=step,
        verbose=True,
        loss_objective=loss_obj if torch.is_tensor(loss_obj) else None,
        loss_rgb_detached=(
            loss_rgb.detach() if torch.is_tensor(loss_rgb) else loss_rgb
        ),
        note="loss_rgb in loss_dict is detached; use loss_objective for grad check",
    )

    gs_raw = pred.get("gs_raw")
    if torch.is_tensor(gs_raw):
        gs_grad_print("gs_raw", step=step, verbose=True, forward=gs_raw)
        if gs_raw.grad is not None:
            gs_grad_print("gs_raw.grad", step=step, verbose=True, grad=gs_raw.grad)
        else:
            print("  gs_raw.grad: <None> (not a leaf or no path from loss)", flush=True)

    dbg = pred.get("_gs_grad_trace") or {}
    for key in ("rendered", "screenspace_points", "loss_view"):
        t = dbg.get(key)
        if torch.is_tensor(t):
            gs_grad_print(f"trace.{key}", step=step, verbose=True, tensor=t)
            if t.grad is not None:
                gs_grad_print(f"trace.{key}.grad", step=step, verbose=True, grad=t.grad)


def gs_debug_hook_list(name: str, hook_list: Sequence[torch.Tensor]) -> None:
    """Summarize a list of multi-scale hook tensors."""
    if not is_gs_debug_enabled():
        return
    if get_dist_rank() != 0:
        return
    dec = hook_list[0].shape[-1] if hook_list else "?"
    gs_debug_print(
        name,
        rank0_only=False,
        num_hooks=len(hook_list),
        dec_dim_last=dec,
        shapes=[tuple(h.shape) for h in hook_list],
    )
