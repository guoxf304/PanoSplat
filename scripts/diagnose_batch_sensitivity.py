#!/usr/bin/env python3
"""Compare B=1 vs B=2 forward activations at key model stages."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_ROOT = os.path.join(_PROJECT_ROOT, "training")
for _p in (_PROJECT_ROOT, _TRAIN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_gs import (  # noqa: E402
    build_batch_from_export_meta,
    build_batch_from_training_dataloader,
    create_synpano_preprocessor,
    load_model_and_loss_from_config,
    process_batch_like_trainer,
)
from panovggt.utils.export_meta import (  # noqa: E402
    ExportMeta,
    load_export_meta,
    resolve_model_forward_global_step,
    r_delta_from_nested,
)
from train_utils.general import copy_data_to_device


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        return {
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
        }
    d = (a - b).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rel_mean": float(d.mean() / (a.abs().mean() + 1e-8)),
    }


def _slice_b0_from_b2(t: torch.Tensor, s: int) -> torch.Tensor:
    """Extract batch-0 from tensors shaped (B, ...) or (B*S, ...)."""
    if t.dim() >= 1 and t.shape[0] >= 2:
        # Common prediction layout: (B, S, ...) or (B, C, H, W)
        if t.dim() >= 2 and t.shape[1] == s:
            return t[0:1]
        # gs_raw etc.: (B, C, H, W)
        if t.dim() == 4 and t.shape[1] <= 32:
            return t[0:1]
    if t.dim() >= 1 and t.shape[0] % s == 0 and t.shape[0] > s:
        b = t.shape[0] // s
        if b >= 2:
            return t[0:s]
    return t


def _compare_dict(
    name: str,
    out_b1: Dict[str, Any],
    out_b2: Dict[str, Any],
    s: int,
) -> List[str]:
    lines: List[str] = []
    for k in sorted(set(out_b1) | set(out_b2)):
        a = out_b1.get(k)
        b = out_b2.get(k)
        if a is None or b is None:
            lines.append(f"  {name}.{k}: missing in one run")
            continue
        if not isinstance(a, torch.Tensor):
            continue
        b0 = _slice_b0_from_b2(b, s)
        if a.shape == b0.shape:
            diff = _tensor_diff(a, b0)
            lines.append(
                f"  {name}.{k} shape={tuple(a.shape)} "
                f"max={diff['max_abs']:.6e} mean={diff['mean_abs']:.6e} "
                f"rel={diff.get('rel_mean', 0):.6e}"
            )
        else:
            lines.append(
                f"  {name}.{k} shape mismatch B1={tuple(a.shape)} B2_b0={tuple(b0.shape)}"
            )
    return lines


@torch.no_grad()
def run_stage_hooks(
    model,
    images_b1: torch.Tensor,
    images_b2: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    captures_b1: Dict[str, Any] = {}
    captures_b2: Dict[str, Any] = {}
    handles: List[Any] = []

    def make_hook(name: str, store: Dict[str, Any]):
        def _hook(_mod, _inp, out):
            if isinstance(out, (list, tuple)):
                store[name] = out[-1] if out else None
            elif isinstance(out, dict):
                store[name] = out.get("x_norm_patchtokens", out)
            else:
                store[name] = out

        return _hook

    agg = model.aggregator
    handles.append(agg.patch_embed.register_forward_hook(make_hook("patch_embed", captures_b1)))
    for i, blk in enumerate(agg.decoder):
        handles.append(
            blk.register_forward_hook(make_hook(f"decoder_{i}", captures_b1))
        )

    def run_once(images, store):
        nonlocal handles
        # rebind hooks to correct store
        for h in handles:
            h.remove()
        handles = []
        handles.append(agg.patch_embed.register_forward_hook(make_hook("patch_embed", store)))
        for i, blk in enumerate(agg.decoder):
            handles.append(
                blk.register_forward_hook(make_hook(f"decoder_{i}", store))
            )
        pred = model(images=images)
        store["pred_local_points"] = pred.get("local_points")
        store["pred_camera_poses"] = pred.get("camera_poses")
        store["pred_global_points"] = pred.get("global_points")
        store["pred_gs_raw"] = pred.get("gs_raw")
        adapter = pred.get("gaussian_adapter_out")
        if adapter is not None:
            if isinstance(adapter, list) and len(adapter) > 0:
                store["pred_gauss_means_b0"] = adapter[0].means
                store["pred_gauss_count_b0"] = torch.tensor(
                    [adapter[0].means.shape[0]], device=images.device
                )
            elif hasattr(adapter, "means"):
                store["pred_gauss_means_b0"] = adapter.means
                store["pred_gauss_count_b0"] = torch.tensor(
                    [adapter.means.shape[0]], device=images.device
                )
        hooks = pred.get("aggregated_hooks")
        if hooks:
            store["pred_hook0"] = hooks[0]
        return pred

    run_once(images_b1, captures_b1)
    run_once(images_b2, captures_b2)

    for h in handles:
        h.remove()
    return captures_b1, captures_b2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="synpano_gs")
    parser.add_argument(
        "--checkpoint",
        default="logs/synpano_gs_0616_1802/ckpts/checkpoint.pt",
    )
    parser.add_argument(
        "--reference_export",
        default="output/synpano_gs_0616_1802/Epoch_000002_step_001200",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    meta_path = os.path.join(args.reference_export, "export_meta.json")
    export_meta = load_export_meta(meta_path)
    fwd_step = resolve_model_forward_global_step(export_meta)
    batch_meta = export_meta.primary_batch(0, collate_sample_idx=0)

    model, _, cfg = load_model_and_loss_from_config(
        args.config,
        args.checkpoint,
        args.device,
        img_size=518,
        model_forward_step=fwd_step,
    )
    synpano = create_synpano_preprocessor(cfg)
    geom_aug = r_delta_from_nested(batch_meta.geom_aug_R_delta)

    batch_b1 = build_batch_from_export_meta(
        synpano, batch_meta, geom_aug_R_delta=geom_aug
    )
    batch_b2 = build_batch_from_training_dataloader(cfg, export_meta, dataloader_batch_idx=0)

    batch_b1 = process_batch_like_trainer(batch_b1, cfg)
    batch_b2 = process_batch_like_trainer(batch_b2, cfg)
    batch_b1 = copy_data_to_device(batch_b1, args.device)
    batch_b2 = copy_data_to_device(batch_b2, args.device)

    img_b1 = batch_b1["images"]
    img_b2 = batch_b2["images"]
    s = img_b1.shape[1]

    img_diff = _tensor_diff(img_b1, img_b2[0:1])
    print(f"[input] B1 images {tuple(img_b1.shape)} vs B2[0:1] {tuple(img_b2[0:1].shape)}")
    print(f"[input] max_abs={img_diff['max_abs']:.6e} mean_abs={img_diff['mean_abs']:.6e}")

    cap_b1, cap_b2 = run_stage_hooks(model, img_b1, img_b2)

    print("\n[stage diffs B1 vs B2 collate_b0]")
    lines = _compare_dict("root", cap_b1, cap_b2, s)
    for ln in lines:
        print(ln)

    # Find first decoder layer where diff becomes non-trivial
    threshold = 1e-4
    first_layer = None
    for i in range(len(model.aggregator.decoder)):
        k = f"decoder_{i}"
        if k not in cap_b1 or k not in cap_b2:
            continue
        a = cap_b1[k]
        b0 = _slice_b0_from_b2(cap_b2[k], s)
        if a.shape != b0.shape:
            first_layer = k
            break
        d = _tensor_diff(a, b0)
        if d["mean_abs"] > threshold:
            first_layer = k
            print(f"\n[first layer mean_abs > {threshold}] {k}: mean={d['mean_abs']:.6e}")
            break

    if first_layer is None:
        print("\n[note] No decoder layer exceeded threshold; check pred heads.")


if __name__ == "__main__":
    main()
