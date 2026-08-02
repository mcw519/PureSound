"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.render.multiband_fdn` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.render.multiband_fdn import (
    MULTIBAND_FDN_POLICY,
    MultibandFDNDesign,
    MultibandFDNRender,
    analyze_fdn_coloration,
    delay_proportional_loop_gains,
    design_multiband_fdn,
    fdn_filterbank_power_response,
    randomized_hadamard_matrix,
    render_multiband_fdn,
    render_multiband_fdn_impulse,
    select_prime_delay_lengths,
)

__all__ = [
    "MULTIBAND_FDN_POLICY",
    "MultibandFDNDesign",
    "MultibandFDNRender",
    "analyze_fdn_coloration",
    "delay_proportional_loop_gains",
    "design_multiband_fdn",
    "fdn_filterbank_power_response",
    "randomized_hadamard_matrix",
    "render_multiband_fdn",
    "render_multiband_fdn_impulse",
    "select_prime_delay_lengths",
]
