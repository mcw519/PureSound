"""Render layer: backends, crossover, coupling, and spatial assembly.

Layer 3 of the RIR package.  May compose ``scene``, ``path_events``,
``metrics`` and ``physics``; must not write bank manifests or touch training
data.
"""

from puresound.audio.rir.render.arrays import coerce_rir_array, pad_or_trim
from puresound.audio.rir.render.backend import RIRBackend
from puresound.audio.rir.render.crossover import (
    align_high_band_direct,
    clip_rir_before_physical_arrival,
    hybrid_crossover,
    hybrid_crossover_with_metadata,
)

__all__ = [
    "RIRBackend",
    "align_high_band_direct",
    "clip_rir_before_physical_arrival",
    "coerce_rir_array",
    "hybrid_crossover",
    "hybrid_crossover_with_metadata",
    "pad_or_trim",
]
