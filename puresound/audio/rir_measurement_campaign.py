"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.calibration.measured_campaign` during R5 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.calibration.measured_campaign import (
    CalibratedTransducer,
    MeasuredRIRRecord,
    MeasuredRoom,
    MeasurementAsset,
    MeasurementPose,
    RIRMeasurementCampaign,
    RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION,
    ROOM_SPLITS,
    SweepCapture,
    audit_measurement_campaign,
    sha256_file,
)

__all__ = [
    "CalibratedTransducer",
    "MeasuredRIRRecord",
    "MeasuredRoom",
    "MeasurementAsset",
    "MeasurementPose",
    "RIRMeasurementCampaign",
    "RIR_MEASUREMENT_CAMPAIGN_SCHEMA_VERSION",
    "ROOM_SPLITS",
    "SweepCapture",
    "audit_measurement_campaign",
    "sha256_file",
]
