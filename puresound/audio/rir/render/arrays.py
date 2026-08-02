"""Shared RIR array shaping for the render layer.

Every backend and the crossover agree on ``[channels, samples]`` float64;
these two helpers are what enforce it.  Moved out of
``puresound.audio.rir.render.hybrid`` in R2 of ``RIR_MODULARIZATION_PLAN.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def pad_or_trim(rirs: list[np.ndarray], num_samples: int) -> np.ndarray:
    out = np.zeros((len(rirs), num_samples), dtype=np.float64)
    for idx, rir in enumerate(rirs):
        flat = np.asarray(rir, dtype=np.float64).reshape(-1)
        n = min(flat.shape[0], num_samples)
        out[idx, :n] = flat[:n]
    return out


def coerce_rir_array(rir: Any, num_sources: int, num_samples: int) -> np.ndarray:
    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 2 and arr.shape[0] != num_sources and arr.shape[1] == num_sources:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[0] != num_sources:
        raise ValueError(
            f"Expected RIR array [{num_sources}, samples], got shape {arr.shape}"
        )
    return pad_or_trim([arr[idx] for idx in range(arr.shape[0])], num_samples)


__all__ = ["coerce_rir_array", "pad_or_trim"]
