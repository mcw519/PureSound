"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.synthetic_recovery` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.synthetic_recovery import (
    PerturbedSyntheticMeasurement,
    RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY,
    RIR_SYNTHETIC_RECOVERY_POLICY,
    SyntheticMeasurementPerturbation,
    SyntheticRecoveryBounds,
    SyntheticRecoveryFit,
    SyntheticRecoveryObjectiveConfig,
    SyntheticRecoveryObservation,
    SyntheticRecoveryParameters,
    build_synthetic_recovery_observation,
    evaluate_synthetic_recovery,
    fit_synthetic_recovery_parameters,
    perturb_synthetic_recovery_measurement,
    render_synthetic_recovery_rir,
)

__all__ = [
    "PerturbedSyntheticMeasurement",
    "RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY",
    "RIR_SYNTHETIC_RECOVERY_POLICY",
    "SyntheticMeasurementPerturbation",
    "SyntheticRecoveryBounds",
    "SyntheticRecoveryFit",
    "SyntheticRecoveryObjectiveConfig",
    "SyntheticRecoveryObservation",
    "SyntheticRecoveryParameters",
    "build_synthetic_recovery_observation",
    "evaluate_synthetic_recovery",
    "fit_synthetic_recovery_parameters",
    "perturb_synthetic_recovery_measurement",
    "render_synthetic_recovery_rir",
]
