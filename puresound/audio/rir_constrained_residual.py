"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.residual` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.residual import (
    CausalDecayResidualModel,
    ConstrainedResidualFit,
    M5_CONSTRAINED_RESIDUAL_POLICY,
    evaluate_residual_ablation,
    fit_causal_decay_residual,
)

__all__ = [
    "CausalDecayResidualModel",
    "ConstrainedResidualFit",
    "M5_CONSTRAINED_RESIDUAL_POLICY",
    "evaluate_residual_ablation",
    "fit_causal_decay_residual",
]
