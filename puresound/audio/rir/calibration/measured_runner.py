"""Fail-closed measured-room calibration runner for M5.3.

The runner consumes only a ready :mod:`rir_measurement_campaign`, fits physical
room parameters on deterministic train positions, and reports position and
physical-room holdouts separately. It never treats a schema template or a
legacy final-RIR bank as controlled evidence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from puresound.audio.rir.calibration.loss import analyze_rir_calibration_loss
from puresound.audio.rir.calibration.inverse_m4 import (
    M4InverseObservation,
    M4InverseParameters,
    M4ProfileObjectiveConfig,
    fit_m4_parameter_profile,
)
from puresound.audio.rir.calibration.inverse_m5 import (
    GroupedPathObservation,
    GroupedPathParameters,
    fit_grouped_path_gains,
    render_grouped_path_observation,
)
from puresound.audio.rir.calibration.measured_campaign import (
    MeasuredRIRRecord,
    MeasuredRoom,
    RIRMeasurementCampaign,
    audit_measurement_campaign,
)
from puresound.audio.rir.metrics import valid_octave_centers
from puresound.audio.rir.path_events import (
    PathEventSet,
    generate_shoebox_path_events,
    render_path_events,
)
from puresound.audio.rir.scene.schema import SHOEBOX_BOUNDARIES


M5_MEASURED_ROOM_FIT_POLICY = "puresound.m5_measured_room_fit.v1"
M5_POSITION_SPLIT_POLICY = "puresound.m5_position_split.sha256.v1"


class CampaignNotReadyError(RuntimeError):
    """Raised before fitting when retained campaign evidence is incomplete."""

    def __init__(self, audit: Mapping[str, Any]):
        super().__init__(
            "controlled campaign is not ready for M5 measured-room fitting"
        )
        self.audit = dict(audit)


def deterministic_position_assignments(
    campaign: RIRMeasurementCampaign,
    *,
    holdout_fraction: float = 0.25,
) -> dict[str, str]:
    """Assign train-room records to fit/holdout without room leakage."""

    if not math.isfinite(holdout_fraction) or not 0.0 < holdout_fraction < 0.5:
        raise ValueError("position holdout fraction must lie in (0, 0.5)")
    assignments: dict[str, str] = {}
    for room in campaign.rooms:
        records = sorted(
            (record for record in campaign.records if record.room_id == room.room_id),
            key=lambda record: record.measurement_id,
        )
        room_split = campaign.room_splits[room.room_id]
        if room_split != "train":
            label = f"room_{room_split}"
            assignments.update({record.measurement_id: label for record in records})
            continue
        if len(records) < 2:
            raise ValueError(
                "every train room needs at least two positions for holdout"
            )
        holdout_count = min(
            len(records) - 1,
            max(1, int(round(len(records) * holdout_fraction))),
        )
        scored = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{campaign.campaign_id}:{room.room_id}:"
                f"{record.measurement_id}".encode()
            ).hexdigest(),
        )
        heldout = {record.measurement_id for record in scored[:holdout_count]}
        assignments.update(
            {
                record.measurement_id: (
                    "position_holdout"
                    if record.measurement_id in heldout
                    else "position_fit"
                )
                for record in records
            }
        )
    return assignments


def _sound_speed_m_s(record: MeasuredRIRRecord) -> float:
    # Cramer-style first-order atmospheric approximation. The environment is
    # retained per capture; pressure has negligible first-order effect here.
    return float(
        331.3 + 0.606 * record.temperature_c + 0.0124 * record.relative_humidity_percent
    )


def _read_deconvolved_rir(
    record: MeasuredRIRRecord,
    root: Path,
) -> np.ndarray:
    asset = record.capture.deconvolved_rir_asset
    values, sample_rate = sf.read(
        root / asset.path,
        dtype="float64",
        always_2d=True,
    )
    rir = np.asarray(values.T, dtype=np.float64)
    if (
        sample_rate != record.capture.sample_rate
        or rir.shape[0] != len(record.receiver_ids)
        or rir.shape[1] != asset.frame_count
        or not np.all(np.isfinite(rir))
    ):
        raise ValueError(f"deconvolved RIR shape mismatch for {record.measurement_id}")
    return rir


@dataclass(frozen=True)
class _ChannelModel:
    record: MeasuredRIRRecord
    receiver_index: int
    target: np.ndarray
    event_set: PathEventSet
    m4_observation: M4InverseObservation
    grouped_observation: GroupedPathObservation
    physical_first_sample: int


def _channel_models(
    room: MeasuredRoom,
    records: Sequence[MeasuredRIRRecord],
    root: Path,
    *,
    max_order: int,
    fdn_seed: int,
) -> tuple[_ChannelModel, ...]:
    if room.dimensions_m is None:
        raise NotImplementedError(
            "M5.3 reference runner currently requires shoebox dimensions_m; "
            "mesh campaigns need a registered mesh PathEvent backend"
        )
    models = []
    surface_ids = {boundary: boundary for boundary in SHOEBOX_BOUNDARIES}
    for record in records:
        target_channels = _read_deconvolved_rir(record, root)
        sound_speed = _sound_speed_m_s(record)
        for receiver_index, receiver_pose in enumerate(record.receiver_poses):
            event_set = generate_shoebox_path_events(
                dimensions_m=room.dimensions_m,
                source_position_m=record.source_pose.position_m,
                receiver_position_m=receiver_pose.position_m,
                sound_speed_m_s=sound_speed,
                scene_id=f"{room.room_id}:{record.measurement_id}",
                source_id=record.source_id,
                receiver_id=record.receiver_ids[receiver_index],
                surface_ids=surface_ids,
                max_order=int(max_order),
                edge_corner_policy="exclude",
            )
            direct = next(
                event for event in event_set.events if event.path_type == "direct"
            )
            sample_rate = record.capture.sample_rate
            sample_count = target_channels.shape[1]
            direct_sample = int(round(direct.delay_s * sample_rate))
            seed_digest = hashlib.sha256(
                f"{record.measurement_id}:{receiver_index}:{fdn_seed}".encode()
            ).digest()
            channel_seed = int.from_bytes(seed_digest[:4], "little")
            coherent = render_path_events(
                event_set,
                sample_rate_hz=sample_rate,
                num_samples=sample_count,
            )
            m4 = M4InverseObservation(
                observation_id=(f"{record.measurement_id}:{receiver_index}"),
                path_event_rir=coherent,
                sample_rate=sample_rate,
                direct_sample=direct_sample,
                fdn_seed=channel_seed,
                delay_line_count=4,
            )
            grouped = GroupedPathObservation(
                observation_id=m4.observation_id,
                event_set=event_set,
                surface_group_by_id={
                    boundary: boundary for boundary in SHOEBOX_BOUNDARIES
                },
                sample_rate=sample_rate,
                sample_count=sample_count,
                direct_sample=direct_sample,
                fdn_seed=channel_seed,
                delay_line_count=4,
            )
            models.append(
                _ChannelModel(
                    record=record,
                    receiver_index=receiver_index,
                    target=target_channels[receiver_index],
                    event_set=event_set,
                    m4_observation=m4,
                    grouped_observation=grouped,
                    physical_first_sample=int(math.floor(direct.delay_s * sample_rate)),
                )
            )
    return tuple(models)


def _centers_for_sample_rate(sample_rate: int) -> tuple[float, ...]:
    centers = tuple(valid_octave_centers(sample_rate, (500.0, 1000.0, 2000.0, 4000.0)))
    if not centers:
        raise ValueError("campaign sample rate has no supported M5 octave centers")
    return centers


def _initial_m4(sample_rate: int) -> M4InverseParameters:
    centers = _centers_for_sample_rate(sample_rate)
    return M4InverseParameters(
        mixing_time_s=0.024,
        coherent_reflection_gain_db=-2.0,
        target_rt60_s_by_hz={center: 0.5 for center in centers},
    )


def _grouped_from_m4(parameters: M4InverseParameters) -> GroupedPathParameters:
    # The M4 parameter is an aggregate coherent-path adjustment whereas the
    # grouped model applies the value once per boundary hit.  It is only a
    # bounded initializer; the grouped fit determines the six effective gains.
    per_hit = float(np.clip(parameters.coherent_reflection_gain_db, -7.5, 2.5))
    return GroupedPathParameters(
        mixing_time_s=parameters.mixing_time_s,
        reflection_adjustment_db_by_group={
            boundary: per_hit for boundary in SHOEBOX_BOUNDARIES
        },
        target_rt60_s_by_hz=parameters.target_rt60_s_by_hz,
    )


def _evaluate_record_models(
    models: Sequence[_ChannelModel],
    parameters: GroupedPathParameters,
) -> list[dict[str, Any]]:
    reports = []
    record_ids = sorted({model.record.measurement_id for model in models})
    for measurement_id in record_ids:
        channels = [
            model for model in models if model.record.measurement_id == measurement_id
        ]
        targets = np.vstack([model.target for model in channels])
        candidates = np.vstack(
            [
                render_grouped_path_observation(
                    model.grouped_observation,
                    parameters,
                ).rir
                for model in channels
            ]
        )
        direct = [model.m4_observation.direct_sample for model in channels]
        report = analyze_rir_calibration_loss(
            targets,
            candidates,
            channels[0].record.capture.sample_rate,
            measured_direct_samples=direct,
            synthetic_direct_samples=direct,
            physical_first_samples=[model.physical_first_sample for model in channels],
            octave_centers_hz=parameters.target_rt60_s_by_hz,
        )
        reports.append(
            {
                "measurement_id": measurement_id,
                "receiver_count": len(channels),
                **report.to_dict(),
            }
        )
    return reports


def _mean_total(reports: Sequence[Mapping[str, Any]]) -> float | None:
    return (
        float(np.mean([float(report["total"]) for report in reports]))
        if reports
        else None
    )


@dataclass(frozen=True)
class MeasuredRoomFitResult:
    room_id: str
    m4_profile: Mapping[str, Any]
    grouped_fit: Mapping[str, Any]
    parameters: GroupedPathParameters
    fit_position_reports: tuple[Mapping[str, Any], ...]
    heldout_position_reports: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "m4_profile": dict(self.m4_profile),
            "grouped_fit": dict(self.grouped_fit),
            "parameters": self.parameters.to_dict(),
            "fit_position_reports": [
                dict(report) for report in self.fit_position_reports
            ],
            "heldout_position_reports": [
                dict(report) for report in self.heldout_position_reports
            ],
            "fit_position_mean_total": _mean_total(self.fit_position_reports),
            "heldout_position_mean_total": _mean_total(self.heldout_position_reports),
        }


def _median_parameters(
    values: Sequence[GroupedPathParameters],
) -> GroupedPathParameters:
    if not values:
        raise ValueError("at least one fitted train room is required")
    groups = values[0].group_names
    centers = tuple(values[0].target_rt60_s_by_hz)
    if any(
        value.group_names != groups or tuple(value.target_rt60_s_by_hz) != centers
        for value in values
    ):
        raise ValueError("train-room parameter schemas differ")
    mixing_values = np.asarray([value.mixing_time_s for value in values])
    return GroupedPathParameters(
        mixing_time_s=float(np.median(mixing_values)),
        reflection_adjustment_db_by_group={
            group: float(
                np.median(
                    [value.reflection_adjustment_db_by_group[group] for value in values]
                )
            )
            for group in groups
        },
        target_rt60_s_by_hz={
            center: float(
                np.median([value.target_rt60_s_by_hz[center] for value in values])
            )
            for center in centers
        },
    )


def run_measured_campaign_fit(
    campaign: RIRMeasurementCampaign,
    root: str | Path,
    *,
    minimum_records_per_room: int = 12,
    position_holdout_fraction: float = 0.25,
    mixing_time_candidates_s: Sequence[float] = (0.020, 0.024, 0.032),
    max_order: int = 4,
    fdn_seed: int = 20260802,
    maximum_evaluations: int = 60,
) -> dict[str, Any]:
    """Fit train rooms and separately evaluate position/room holdouts."""

    root_path = Path(root)
    audit = audit_measurement_campaign(
        campaign,
        root_path,
        minimum_records_per_room=int(minimum_records_per_room),
    )
    if not audit["ready_for_m5_inverse_calibration"]:
        raise CampaignNotReadyError(audit)
    assignments = deterministic_position_assignments(
        campaign,
        holdout_fraction=float(position_holdout_fraction),
    )
    room_by_id = {room.room_id: room for room in campaign.rooms}
    train_results = []
    room_models: dict[str, tuple[_ChannelModel, ...]] = {}
    for room in campaign.rooms:
        records = tuple(
            record for record in campaign.records if record.room_id == room.room_id
        )
        room_models[room.room_id] = _channel_models(
            room,
            records,
            root_path,
            max_order=int(max_order),
            fdn_seed=int(fdn_seed),
        )
        if campaign.room_splits[room.room_id] != "train":
            continue
        fit_models = tuple(
            model
            for model in room_models[room.room_id]
            if assignments[model.record.measurement_id] == "position_fit"
        )
        heldout_models = tuple(
            model
            for model in room_models[room.room_id]
            if assignments[model.record.measurement_id] == "position_holdout"
        )
        sample_rates = {model.record.capture.sample_rate for model in fit_models}
        if len(sample_rates) != 1:
            raise ValueError("one fitted room must use one sample rate")
        initial_m4 = _initial_m4(sample_rates.pop())
        m4_fit = fit_m4_parameter_profile(
            [model.m4_observation for model in fit_models],
            [model.target for model in fit_models],
            initial_m4,
            mixing_time_candidates_s,
            objective=M4ProfileObjectiveConfig(),
            maximum_evaluations=int(maximum_evaluations),
        )
        grouped_initial = _grouped_from_m4(m4_fit.best.parameters)
        grouped_fit = fit_grouped_path_gains(
            [model.grouped_observation for model in fit_models],
            [model.target for model in fit_models],
            grouped_initial,
            maximum_evaluations=int(maximum_evaluations),
        )
        train_results.append(
            MeasuredRoomFitResult(
                room_id=room.room_id,
                m4_profile=m4_fit.to_dict(),
                grouped_fit=grouped_fit.to_dict(),
                parameters=grouped_fit.parameters,
                fit_position_reports=tuple(
                    _evaluate_record_models(fit_models, grouped_fit.parameters)
                ),
                heldout_position_reports=tuple(
                    _evaluate_record_models(
                        heldout_models,
                        grouped_fit.parameters,
                    )
                ),
            )
        )
    population = _median_parameters([result.parameters for result in train_results])
    heldout_rooms = {}
    for split in ("validation", "test"):
        reports = []
        for room_id, models in room_models.items():
            if campaign.room_splits[room_id] == split:
                reports.extend(_evaluate_record_models(models, population))
        heldout_rooms[split] = {
            "room_ids": sorted(
                room_id
                for room_id in room_by_id
                if campaign.room_splits[room_id] == split
            ),
            "reports": reports,
            "mean_total": _mean_total(reports),
        }
    train_fit_reports = [
        report for result in train_results for report in result.fit_position_reports
    ]
    position_holdout_reports = [
        report for result in train_results for report in result.heldout_position_reports
    ]
    checks = {
        "controlled_campaign_audit_passed": bool(
            audit["ready_for_m5_inverse_calibration"]
        ),
        "room_splits_remain_disjoint": all(
            assignment.startswith("room_")
            if campaign.room_splits[
                next(
                    record.room_id
                    for record in campaign.records
                    if record.measurement_id == measurement_id
                )
            ]
            != "train"
            else assignment in {"position_fit", "position_holdout"}
            for measurement_id, assignment in assignments.items()
        ),
        "every_train_room_has_fit_and_position_holdout": all(
            any(
                assignments[record.measurement_id] == "position_fit"
                for record in campaign.records
                if record.room_id == room.room_id
            )
            and any(
                assignments[record.measurement_id] == "position_holdout"
                for record in campaign.records
                if record.room_id == room.room_id
            )
            for room in campaign.rooms
            if campaign.room_splits[room.room_id] == "train"
        ),
        # "converged" means the fit reached a stable minimum, not that
        # ``least_squares`` tripped one of its own tolerances.  See
        # M4_PROFILE_CONVERGENCE_POLICY for why the two came apart.
        "all_train_room_m4_profiles_converged": all(
            result.m4_profile["best"]["converged"] for result in train_results
        ),
        "all_train_room_grouped_fits_converged": all(
            result.grouped_fit["success"] for result in train_results
        ),
        "position_holdout_evaluated": bool(position_holdout_reports),
        "validation_rooms_evaluated": bool(heldout_rooms["validation"]["reports"]),
        "test_rooms_evaluated": bool(heldout_rooms["test"]["reports"]),
        "all_reported_losses_finite": all(
            math.isfinite(float(report["total"]))
            for report in (
                *train_fit_reports,
                *position_holdout_reports,
                *heldout_rooms["validation"]["reports"],
                *heldout_rooms["test"]["reports"],
            )
        ),
    }
    report = {
        "schema_version": "puresound.m5_measured_room_fit_report.v1",
        "milestone": "M5.3",
        "policy": M5_MEASURED_ROOM_FIT_POLICY,
        "campaign_id": campaign.campaign_id,
        "campaign_root": str(root_path),
        "campaign_audit": audit,
        "position_split": {
            "policy": M5_POSITION_SPLIT_POLICY,
            "holdout_fraction": float(position_holdout_fraction),
            "assignments": assignments,
        },
        "configuration": {
            "mixing_time_candidates_s": list(mixing_time_candidates_s),
            "max_path_event_order": int(max_order),
            "maximum_evaluations": int(maximum_evaluations),
            "reference_scope": (
                "shoebox dimensions; mesh campaigns require registered backend"
            ),
            "deconvolved_asset_semantics": (
                "calibrated and latency-corrected final linear RIR; raw/inverse/"
                "noise assets remain retained for audit"
            ),
        },
        "train_room_fits": [result.to_dict() for result in train_results],
        "population_parameters_for_unseen_rooms": population.to_dict(),
        "aggregates": {
            "train_fit_mean_total": _mean_total(train_fit_reports),
            "position_holdout_mean_total": _mean_total(position_holdout_reports),
            "heldout_rooms": heldout_rooms,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_3_measured_fit_executed": bool(all(checks.values())),
            "production_enabled": False,
        },
    }
    return report


__all__ = [
    "M5_MEASURED_ROOM_FIT_POLICY",
    "M5_POSITION_SPLIT_POLICY",
    "CampaignNotReadyError",
    "MeasuredRoomFitResult",
    "deterministic_position_assignments",
    "run_measured_campaign_fit",
]
