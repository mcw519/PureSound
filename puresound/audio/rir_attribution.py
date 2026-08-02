"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.metrics.attribution` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.metrics.attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    complementary_early_late_masks,
    decompose_direct_early_later,
    reconstruction_error,
)

__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "complementary_early_late_masks",
    "decompose_direct_early_later",
    "reconstruction_error",
]
