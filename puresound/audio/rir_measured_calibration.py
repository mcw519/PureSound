"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.measured_runner` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.measured_runner import (
    CampaignNotReadyError,
    M5_MEASURED_ROOM_FIT_POLICY,
    M5_POSITION_SPLIT_POLICY,
    MeasuredRoomFitResult,
    deterministic_position_assignments,
    run_measured_campaign_fit,
)

__all__ = [
    "CampaignNotReadyError",
    "M5_MEASURED_ROOM_FIT_POLICY",
    "M5_POSITION_SPLIT_POLICY",
    "MeasuredRoomFitResult",
    "deterministic_position_assignments",
    "run_measured_campaign_fit",
]
