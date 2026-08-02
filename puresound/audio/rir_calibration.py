"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.loss` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.loss import (
    CalibrationLossReport,
    CalibrationLossWeights,
    RIR_CALIBRATION_LOSS_POLICY,
    analyze_rir_calibration_loss,
    arrival_timing_distance,
    causality_penalty,
    decay_growth_penalty,
    energy_decay_curve_distance,
    multiresolution_stft_distance,
    octave_acoustic_distance,
    spatial_coherence_distance,
)

__all__ = [
    "CalibrationLossReport",
    "CalibrationLossWeights",
    "RIR_CALIBRATION_LOSS_POLICY",
    "analyze_rir_calibration_loss",
    "arrival_timing_distance",
    "causality_penalty",
    "decay_growth_penalty",
    "energy_decay_curve_distance",
    "multiresolution_stft_distance",
    "octave_acoustic_distance",
    "spatial_coherence_distance",
]
