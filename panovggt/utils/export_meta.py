"""Export metadata for reproducing epoch_infer renders in inference_gs.py."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

EXPORT_META_VERSION = 1
EXPORT_META_FILENAME = "export_meta.json"


@dataclass
class ExportBatchMeta:
  dataloader_batch_idx: int = 0
  collate_sample_idx: int = 0
  seq_name: str = ""
  scene_name: str = ""
  synpano_dir: str = ""
  frame_stems: List[str] = field(default_factory=list)
  frame_ids: List[int] = field(default_factory=list)
  image_names: List[str] = field(default_factory=list)
  aug_seed: int = 0
  geom_aug_R_delta: Optional[List[List[float]]] = None


@dataclass
class ExportMeta:
  version: int = EXPORT_META_VERSION
  epoch: int = 0
  global_step: int = 0
  trainer_seed: int = 42
  # ``model.global_step`` at export forward (usually ``global_step - 1`` after training).
  model_forward_global_step: int = -1
  batches: List[ExportBatchMeta] = field(default_factory=list)

  def primary_batch(
    self,
    dataloader_batch_idx: int = 0,
    collate_sample_idx: int = 0,
  ) -> ExportBatchMeta:
    for batch in self.batches:
      if (
        batch.dataloader_batch_idx == dataloader_batch_idx
        and batch.collate_sample_idx == collate_sample_idx
      ):
        return batch
    for batch in self.batches:
      if batch.dataloader_batch_idx == dataloader_batch_idx:
        return batch
    if self.batches:
      return self.batches[0]
    raise ValueError("export meta has no batches")


def derive_export_aug_seed(
  epoch: int,
  global_step: int,
  batch_idx: int,
  base_seed: int = 42,
  *,
  collate_sample_idx: int = 0,
) -> int:
  """Deterministic portable seed recorded alongside actual geom aug."""
  mixed = (
    int(base_seed)
    + int(epoch) * 1_000_003
    + int(global_step) * 1_009
    + int(batch_idx) * 9_176
    + int(collate_sample_idx) * 7_919
  )
  return int(mixed % (2**31 - 1))


def parse_scene_name(seq_name: Any) -> str:
  seq = str(seq_name).strip()
  if seq.startswith("SynPano_"):
    return seq[len("SynPano_") :]
  return seq or "scene"


def _tensor_to_nested_list(t: torch.Tensor) -> List[List[float]]:
  return t.detach().cpu().float().tolist()


def r_delta_from_nested(data: Optional[List[List[float]]]) -> Optional[torch.Tensor]:
  if data is None:
    return None
  t = torch.tensor(data, dtype=torch.float32)
  if is_identity_R_delta(t):
    return None
  return t


def is_identity_R_delta(
  mat: Optional[torch.Tensor], atol: float = 1e-5
) -> bool:
  if mat is None:
    return True
  eye = torch.eye(3, dtype=mat.dtype, device=mat.device)
  return bool(torch.allclose(mat, eye, atol=atol, rtol=0.0))


def _coerce_str_list(value: Any) -> List[str]:
  if value is None:
    return []
  if isinstance(value, torch.Tensor):
    flat = value.detach().cpu().reshape(-1)
    return [str(v.item()) for v in flat]
  if isinstance(value, (list, tuple)):
    out: List[str] = []
    for item in value:
      out.extend(_coerce_str_list(item))
    return out
  return [str(value)]


def _coerce_int_list(value: Any) -> List[int]:
  if value is None:
    return []
  if isinstance(value, torch.Tensor):
    flat = value.detach().cpu().reshape(-1)
    return [int(v.item()) for v in flat]
  if isinstance(value, (list, tuple)):
    out: List[int] = []
    for item in value:
      out.extend(_coerce_int_list(item))
    return out
  return [int(value)]


def _collate_index(value: Any, bi: int) -> Any:
  if isinstance(value, torch.Tensor):
    if value.dim() == 0:
      return value.item()
    flat = value.reshape(-1)
    if flat.numel() == 1:
      return flat[0].item()
    return flat[bi].item() if bi < flat.shape[0] else flat[0].item()
  if isinstance(value, (list, tuple)):
    if len(value) == 0:
      return None
    if len(value) == 1:
      return value[0]
    return value[bi] if bi < len(value) else value[0]
  return value


def _batch_frame_num(batch: Mapping[str, Any], bi: int = 0) -> Optional[int]:
  frame_num = batch.get("frame_num")
  if frame_num is None:
    return None
  if isinstance(frame_num, torch.Tensor):
    if frame_num.dim() == 0:
      return int(frame_num.item())
    return int(frame_num[bi].item()) if bi < frame_num.shape[0] else int(frame_num[0].item())
  if isinstance(frame_num, (list, tuple)):
    return int(frame_num[bi]) if bi < len(frame_num) else int(frame_num[0])
  return int(frame_num)


def _is_view_transposed_collate(field: Sequence[Any], frame_num: Optional[int]) -> bool:
  if frame_num is None or frame_num <= 0:
    return False
  if not isinstance(field, (list, tuple)) or len(field) != frame_num:
    return False
  first = field[0]
  if isinstance(first, torch.Tensor):
    return first.dim() >= 1
  if isinstance(first, (list, tuple)):
    # Batch-major lists hold S frame ids/stems per batch item.
    if len(first) == frame_num and first and isinstance(first[0], (int, float)):
      return False
    return True
  return False


def _batch_field_per_sample(
  field: Any,
  *,
  bi: int,
  frame_num: Optional[int],
  coerce,
) -> List[Any]:
  if field is None:
    return []
  if isinstance(field, torch.Tensor):
    if field.dim() >= 2:
      return [coerce(field[bi, vi].item()) for vi in range(field.shape[1])]
    return [coerce(v.item()) for v in field.reshape(-1)]

  if not isinstance(field, (list, tuple)) or len(field) == 0:
    return [coerce(field)]

  if len(field) == 1:
    inner = field[0]
    return _coerce_int_list(inner) if coerce is int else _coerce_str_list(inner)

  if _is_view_transposed_collate(field, frame_num):
    return [coerce(_collate_index(field[vi], bi)) for vi in range(len(field))]

  elem = field[bi] if bi < len(field) else field[0]
  if isinstance(elem, (list, tuple)) and len(elem) > 1:
    return [coerce(v) for v in elem]
  if isinstance(elem, torch.Tensor) and elem.reshape(-1).numel() > 1:
    flat = elem.reshape(-1)
    return [coerce(v.item()) for v in flat]

  return [coerce(v) for v in (_coerce_int_list(elem) if coerce is int else _coerce_str_list(elem))]


def get_batch_frame_stems(batch: Mapping[str, Any], bi: int = 0) -> List[str]:
  """Per-view frame stems for one batch item (handles default_collate layout)."""
  return _batch_frame_stems(batch, bi)


def get_batch_frame_ids(batch: Mapping[str, Any], bi: int = 0) -> List[int]:
  """Per-view frame indices for one batch item (handles default_collate layout)."""
  return _batch_frame_ids(batch, bi)


def _batch_frame_stems(batch: Mapping[str, Any], bi: int = 0) -> List[str]:
  frame_num = _batch_frame_num(batch, bi)
  stems = _batch_field_per_sample(
    batch.get("frame_stems"),
    bi=bi,
    frame_num=frame_num,
    coerce=str,
  )
  return [str(s) for s in stems]


def _batch_frame_ids(batch: Mapping[str, Any], bi: int = 0) -> List[int]:
  frame_num = _batch_frame_num(batch, bi)
  ids = _batch_field_per_sample(
    batch.get("ids"),
    bi=bi,
    frame_num=frame_num,
    coerce=int,
  )
  return [int(i) for i in ids]


def _batch_geom_aug_R_delta(
  batch: Mapping[str, Any], bi: int = 0
) -> Optional[torch.Tensor]:
  raw = batch.get("geom_aug_R_delta")
  if raw is None:
    return None
  if isinstance(raw, (list, tuple)):
    item = raw[bi] if bi < len(raw) else raw[0]
    if item is None:
      return None
    if torch.is_tensor(item):
      return item.detach().cpu().float()
    return None
  if torch.is_tensor(raw):
    if raw.dim() == 3:
      t = raw[bi].detach().cpu().float()
    elif raw.dim() == 2:
      t = raw.detach().cpu().float()
    else:
      return None
    return None if is_identity_R_delta(t) else t
  return None


def extract_export_batch_meta(
  batch: Mapping[str, Any],
  *,
  dataloader_batch_idx: int,
  epoch: int,
  global_step: int,
  trainer_seed: int,
  synpano_dir: str = "",
  bi: int = 0,
) -> ExportBatchMeta:
  seq_name = batch.get("seq_name")
  if isinstance(seq_name, (list, tuple)):
    seq_name = seq_name[bi] if bi < len(seq_name) else seq_name[0]
  seq_name = str(seq_name or "")

  frame_stems = _batch_frame_stems(batch, bi)
  frame_ids = _batch_frame_ids(batch, bi)
  image_names = [f"{stem}.png" for stem in frame_stems]

  r_delta = _batch_geom_aug_R_delta(batch, bi)
  geom_aug_R_delta = (
    _tensor_to_nested_list(r_delta) if r_delta is not None else None
  )

  return ExportBatchMeta(
    dataloader_batch_idx=int(dataloader_batch_idx),
    collate_sample_idx=int(bi),
    seq_name=seq_name,
    scene_name=parse_scene_name(seq_name),
    synpano_dir=str(synpano_dir or ""),
    frame_stems=frame_stems,
    frame_ids=frame_ids,
    image_names=image_names,
    aug_seed=derive_export_aug_seed(
      epoch,
      global_step,
      dataloader_batch_idx,
      trainer_seed,
      collate_sample_idx=int(bi),
    ),
    geom_aug_R_delta=geom_aug_R_delta,
  )


def save_export_meta(path: str, meta: ExportMeta) -> str:
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(asdict(meta), f, indent=2)
  return path


def load_export_meta(path: str) -> ExportMeta:
  with open(path, encoding="utf-8") as f:
    raw = json.load(f)
  batches = []
  for item in raw.get("batches", []):
    entry = dict(item)
    entry.setdefault("collate_sample_idx", 0)
    batches.append(ExportBatchMeta(**entry))
  return ExportMeta(
    version=int(raw.get("version", EXPORT_META_VERSION)),
    epoch=int(raw.get("epoch", 0)),
    global_step=int(raw.get("global_step", 0)),
    trainer_seed=int(raw.get("trainer_seed", 42)),
    model_forward_global_step=int(raw.get("model_forward_global_step", -1)),
    batches=batches,
  )


def resolve_model_forward_global_step(meta: ExportMeta) -> int:
  """Step passed to ``model.set_global_step`` to match epoch export forward."""
  fwd = int(meta.model_forward_global_step)
  if fwd >= 0:
    return fwd
  gstep = int(meta.global_step)
  return max(0, gstep - 1) if gstep > 0 else 0


def training_dataloader_epoch_from_export(meta: ExportMeta) -> int:
  """0-indexed epoch for ``DynamicTorchDataset.get_loader`` (display epoch is ``meta.epoch``)."""
  return max(0, int(meta.epoch) - 1)


def find_export_meta(reference: str) -> str:
  """Resolve ``export_meta.json`` from a run directory or direct file path."""
  ref = os.path.abspath(reference)
  if os.path.isfile(ref):
    return ref
  candidate = os.path.join(ref, EXPORT_META_FILENAME)
  if os.path.isfile(candidate):
    return candidate
  raise FileNotFoundError(
    f"export meta not found at {ref} or {candidate}"
  )


def resolve_image_paths_from_meta(
  batch_meta: ExportBatchMeta,
  *,
  image_dir: Optional[str] = None,
) -> List[str]:
  if image_dir:
    base_dir = image_dir
  elif batch_meta.synpano_dir and batch_meta.scene_name:
    base_dir = os.path.join(
      batch_meta.synpano_dir, batch_meta.scene_name, "images"
    )
  else:
    raise ValueError(
      "Cannot resolve image paths: provide --image_dir or synpano_dir+scene in export meta."
    )

  paths: List[str] = []
  for name in batch_meta.image_names:
    path = os.path.join(base_dir, name)
    if not os.path.isfile(path):
      raise FileNotFoundError(f"Missing image for export meta: {path}")
    paths.append(path)
  return paths


def order_image_paths_by_meta(
  image_paths: Sequence[str],
  batch_meta: ExportBatchMeta,
) -> List[str]:
  """Reorder user-provided images to match epoch export frame order."""
  by_stem = {Path(p).stem: p for p in image_paths}
  ordered: List[str] = []
  for stem in batch_meta.frame_stems:
    if stem not in by_stem:
      raise ValueError(
        f"image_dir missing frame stem '{stem}' required by export meta "
        f"(expected order: {batch_meta.frame_stems})."
      )
    ordered.append(by_stem[stem])
  return ordered


def format_export_meta_block(meta: ExportMeta, batch_idx: int = 0) -> str:
  matched = [
    b for b in meta.batches if b.dataloader_batch_idx == batch_idx
  ]
  if not matched:
    matched = list(meta.batches)

  lines = [
    "[export_meta]",
    f"  epoch: {meta.epoch}",
    f"  global_step: {meta.global_step}",
    f"  model_forward_global_step: {meta.model_forward_global_step}",
    f"  trainer_seed: {meta.trainer_seed}",
    f"  dataloader_batch_idx: {batch_idx}",
    f"  num_collate_samples: {len(matched)}",
  ]
  for batch in matched:
    lines.append(f"  [collate_b{batch.collate_sample_idx}]")
    lines.append(f"    seq_name: {batch.seq_name}")
    lines.append(f"    scene_name: {batch.scene_name}")
    lines.append(f"    aug_seed: {batch.aug_seed}")
    lines.append(f"    frame_stems: {batch.frame_stems}")
    lines.append(f"    frame_ids: {batch.frame_ids}")
    lines.append(f"    image_names: {batch.image_names}")
    lines.append(
      "    geom_aug_R_delta: "
      f"{'present' if batch.geom_aug_R_delta else 'none'}"
    )
  return "\n".join(lines)
