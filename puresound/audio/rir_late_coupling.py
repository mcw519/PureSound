"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.render.coupling` during R4 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.render.coupling import (
    PATH_EVENT_FDN_COUPLING_POLICY,
    PathEventFDNCouplingResult,
    couple_path_event_rir_with_fdn,
    energy_preserving_diffuse_gain,
    equal_power_transition_weights,
    extrapolated_path_tail_energy_target,
    transition_samples,
)

__all__ = [
    "PATH_EVENT_FDN_COUPLING_POLICY",
    "PathEventFDNCouplingResult",
    "couple_path_event_rir_with_fdn",
    "energy_preserving_diffuse_gain",
    "equal_power_transition_weights",
    "extrapolated_path_tail_energy_target",
    "transition_samples",
]
