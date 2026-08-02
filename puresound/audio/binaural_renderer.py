"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.render.binaural` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.render.binaural import (
    AMBISONIC_BINAURAL_POLICY,
    AmbisonicBinauralDecoder,
    BinauralBRIRRender,
    analytic_first_order_binaural_decoder,
    render_ambisonic_brir,
)

__all__ = [
    "AMBISONIC_BINAURAL_POLICY",
    "AmbisonicBinauralDecoder",
    "BinauralBRIRRender",
    "analytic_first_order_binaural_decoder",
    "render_ambisonic_brir",
]
