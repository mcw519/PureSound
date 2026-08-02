"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.inverse_m4` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.inverse_m4 import (
    M4InverseObservation,
    M4InverseParameters,
    M4MixingProfilePoint,
    M4ProfileBounds,
    M4ProfileFit,
    M4ProfileObjectiveConfig,
    M4_PARAMETER_PROFILE_INVERSE_POLICY,
    fit_m4_parameter_profile,
    render_m4_inverse_observation,
)

__all__ = [
    "M4InverseObservation",
    "M4InverseParameters",
    "M4MixingProfilePoint",
    "M4ProfileBounds",
    "M4ProfileFit",
    "M4ProfileObjectiveConfig",
    "M4_PARAMETER_PROFILE_INVERSE_POLICY",
    "fit_m4_parameter_profile",
    "render_m4_inverse_observation",
]
