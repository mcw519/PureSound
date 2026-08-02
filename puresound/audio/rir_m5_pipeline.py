"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.inverse_m5` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.inverse_m5 import (
    GroupedPathGainFit,
    GroupedPathObservation,
    GroupedPathParameters,
    LocalIdentifiabilityReport,
    M5_GROUPED_PATH_INVERSE_POLICY,
    M5_LOCAL_IDENTIFIABILITY_POLICY,
    M5_SPATIAL_CANDIDATE_PROFILE_POLICY,
    SpatialCalibrationSelection,
    analyze_local_identifiability,
    fit_grouped_path_gains,
    render_grouped_path_observation,
    select_spatial_calibration_candidate,
)

__all__ = [
    "GroupedPathGainFit",
    "GroupedPathObservation",
    "GroupedPathParameters",
    "LocalIdentifiabilityReport",
    "M5_GROUPED_PATH_INVERSE_POLICY",
    "M5_LOCAL_IDENTIFIABILITY_POLICY",
    "M5_SPATIAL_CANDIDATE_PROFILE_POLICY",
    "SpatialCalibrationSelection",
    "analyze_local_identifiability",
    "fit_grouped_path_gains",
    "render_grouped_path_observation",
    "select_spatial_calibration_candidate",
]
