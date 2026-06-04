"""Opacity mapping schedule (from AnySplat)."""

from __future__ import annotations

import torch
from torch import Tensor


def map_pdf_to_opacity(
    pdf: Tensor,
    global_step: int,
    initial: float = 0.0,
    final: float = 0.0,
    warm_up: int = 1,
) -> Tensor:
    """
    Map predicted density in [0, 1] to opacity with optional step schedule.

    When initial == final, exponent is 1 (identity-style blend).
    """
    x = initial + min(global_step / max(warm_up, 1), 1.0) * (final - initial)
    exponent = 2**x
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
