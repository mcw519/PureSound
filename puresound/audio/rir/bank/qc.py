"""Deterministic per-item acoustic QC and quarantine for M6 RIR banks.

The checks in this module deliberately separate three outcomes:

``pass``
    A measured quantity satisfies a physically motivated bound.
``fail``
    A hard invariant is violated and the item must be quarantined.
``not_evaluable`` / ``not_applicable``
    The asset does not contain enough evidence, or the metric's physical
    channel semantics do not apply.  Missing evidence is never fabricated.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from puresound.audio.rir.bank.schema import (
    BANK_SPLITS,
    RIRBankItem,
    RIRBankManifest,
    audit_rir_bank_manifest,
    canonical_json_sha256,
    sha256_file,
    split_index_rows,
)
from puresound.audio.rir.metrics import analyze_echo_density, analyze_rir


RIR_BANK_QC_POLICY_ID = "puresound.rir_bank_qc.physical.v1"
RIR_BANK_ITEM_QC_SCHEMA_VERSION = "puresound.rir_bank_item_qc.v1"
RIR_BANK_QC_RELEASE_SCHEMA_VERSION = "puresound.rir_bank_qc_release.v1"
DEFAULT_MANIFEST_NAME = "rir_bank_manifest.json"
DEFAULT_QC_SUMMARY_NAME = "rir_bank_qc_summary.json"


@dataclass(frozen=True)
class RIRBankQCPolicy:
    """Versioned bounds for structural, causal, and acoustic item QC."""

    maximum_peak_abs: float = 1.0
    minimum_total_energy: float = 1e-12
    maximum_prearrival_relative_peak: float = 1e-7
    maximum_arrival_error_ms: float = 1.0
    direct_search_after_expected_ms: float = 6.0
    tail_start_ms: float = 50.0
    minimum_tail_energy_fraction: float = 1e-7
    minimum_decay_r_squared: float = 0.70
    minimum_decay_coverage_fraction: float = 0.50
    minimum_octave_decay_coverage_fraction: float = 0.25
    decay_required_duration_s: float = 0.12
    minimum_rt60_s: float = 0.02
    maximum_rt60_s: float = 20.0
    minimum_late_echo_density: float = 0.40
    maximum_abs_spectral_tilt_db_per_octave: float = 18.0
    maximum_abs_level_metric_db: float = 80.0
    direct_window_ms: float = 2.5
    octave_centers_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0)
    policy_id: str = RIR_BANK_QC_POLICY_ID

    def __post_init__(self) -> None:
        positive = (
            "maximum_peak_abs",
            "minimum_total_energy",
            "maximum_prearrival_relative_peak",
            "maximum_arrival_error_ms",
            "direct_search_after_expected_ms",
            "tail_start_ms",
            "minimum_decay_r_squared",
            "decay_required_duration_s",
            "minimum_rt60_s",
            "maximum_rt60_s",
            "minimum_late_echo_density",
            "maximum_abs_spectral_tilt_db_per_octave",
            "maximum_abs_level_metric_db",
            "direct_window_ms",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_tail_energy_fraction",
            "minimum_decay_coverage_fraction",
            "minimum_octave_decay_coverage_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        if self.minimum_rt60_s >= self.maximum_rt60_s:
            raise ValueError("minimum_rt60_s must be less than maximum_rt60_s")
        if self.direct_search_after_expected_ms <= self.maximum_arrival_error_ms:
            raise ValueError(
                "direct search window must exceed the maximum arrival error"
            )
        centers = tuple(float(value) for value in self.octave_centers_hz)
        if not centers or any(not math.isfinite(value) or value <= 0 for value in centers):
            raise ValueError("octave centers must be finite and positive")
        object.__setattr__(self, "octave_centers_hz", centers)
        if self.policy_id != RIR_BANK_QC_POLICY_ID:
            raise ValueError("unsupported RIR bank QC policy")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["octave_centers_hz"] = list(self.octave_centers_hz)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RIRBankQCPolicy":
        fields = dict(value)
        if "octave_centers_hz" in fields:
            fields["octave_centers_hz"] = tuple(fields["octave_centers_hz"])
        return cls(**fields)

    def content_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _gate(
    checks: dict[str, dict[str, Any]],
    name: str,
    status: str,
    *,
    measured: Any = None,
    criterion: str,
    reason: str | None = None,
    severity: str = "quarantine",
) -> None:
    if status not in {"pass", "fail", "not_evaluable", "not_applicable"}:
        raise ValueError("invalid QC check status")
    checks[name] = {
        "status": status,
        "severity": severity,
        "measured": measured,
        "criterion": criterion,
        "reason": reason,
    }


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _completed_report(
    item: RIRBankItem,
    policy: RIRBankQCPolicy,
    checks: Mapping[str, Mapping[str, Any]],
    channels: Sequence[Mapping[str, Any]],
    *,
    spatial: Mapping[str, Any],
) -> dict[str, Any]:
    failure_reasons = sorted(
        name
        for name, check in checks.items()
        if check["status"] == "fail" and check["severity"] == "quarantine"
    )
    channel_check = checks.get("per_channel_physical_checks", {})
    channel_measurement = channel_check.get("measured")
    if isinstance(channel_measurement, Mapping):
        channel_reasons = channel_measurement.get("failed_reasons", [])
        if isinstance(channel_reasons, list):
            failure_reasons = sorted(
                {*failure_reasons, *(str(reason) for reason in channel_reasons)}
            )
    unevaluable_reasons = sorted(
        name
        for name, check in checks.items()
        if check["status"] in {"not_evaluable", "not_applicable"}
    )
    return {
        "schema_version": RIR_BANK_ITEM_QC_SCHEMA_VERSION,
        "item_id": item.item_id,
        "bank_identity": {
            "room_id": item.room_id,
            "acoustic_space_id": item.acoustic_space_id,
            "scene_id": item.scene_id,
            "split": item.split,
        },
        "asset_identity": {
            "rir_path": item.rir_path,
            "rir_sha256": item.rir_sha256,
            "metadata_path": item.metadata_path,
            "metadata_sha256": item.metadata_sha256,
            "scene_sha256": item.scene_sha256,
        },
        "policy": policy.to_dict(),
        "policy_sha256": policy.content_sha256(),
        "status": "fail" if failure_reasons else "pass",
        "failure_reasons": failure_reasons,
        "unevaluable_reasons": unevaluable_reasons,
        "checks": dict(checks),
        "channels": list(channels),
        "spatial": dict(spatial),
    }


def evaluate_rir_bank_item(
    root: str | Path,
    item: RIRBankItem,
    policy: RIRBankQCPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate one M6 item without mutating it or its source assets."""

    bank_root = Path(root)
    policy = policy or RIRBankQCPolicy()
    checks: dict[str, dict[str, Any]] = {}
    channels: list[dict[str, Any]] = []
    rir_path = bank_root / item.rir_path
    metadata_path = bank_root / item.metadata_path

    rir_exists = rir_path.is_file()
    metadata_exists = metadata_path.is_file()
    _gate(
        checks,
        "assets_exist",
        "pass" if rir_exists and metadata_exists else "fail",
        measured={"rir": rir_exists, "metadata": metadata_exists},
        criterion="both same-item RIR and metadata assets exist",
    )
    if not rir_exists or not metadata_exists:
        return _completed_report(
            item,
            policy,
            checks,
            channels,
            spatial={"status": "not_evaluable", "reason": "missing_assets"},
        )

    rir_hash_matches = sha256_file(rir_path) == item.rir_sha256
    metadata_hash_matches = sha256_file(metadata_path) == item.metadata_sha256
    _gate(
        checks,
        "asset_hashes_match_manifest",
        "pass" if rir_hash_matches and metadata_hash_matches else "fail",
        measured={"rir": rir_hash_matches, "metadata": metadata_hash_matches},
        criterion="SHA-256 digests equal the M6 manifest",
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("metadata root is not an object")
    except (OSError, json.JSONDecodeError, TypeError) as error:
        _gate(
            checks,
            "metadata_is_readable",
            "fail",
            measured=None,
            criterion="metadata is strict JSON with an object root",
            reason=str(error),
        )
        return _completed_report(
            item,
            policy,
            checks,
            channels,
            spatial={"status": "not_evaluable", "reason": "invalid_metadata"},
        )
    _gate(
        checks,
        "metadata_is_readable",
        "pass",
        measured=True,
        criterion="metadata is strict JSON with an object root",
    )
    scene = metadata.get("scene")
    scene_valid = isinstance(scene, Mapping)
    scene_identity_matches = bool(
        scene_valid
        and str(scene.get("scene_id", item.scene_id)) == item.scene_id
        and str(metadata.get("sample_id", item.item_id)) == item.item_id
        and str(metadata.get("room_id", item.room_id)) == item.room_id
        and canonical_json_sha256(scene) == item.scene_sha256
    )
    _gate(
        checks,
        "metadata_identity_and_scene_hash_match",
        "pass" if scene_identity_matches else "fail",
        measured=scene_identity_matches,
        criterion="item/room/scene identities and canonical scene hash match manifest",
    )
    if not scene_valid:
        return _completed_report(
            item,
            policy,
            checks,
            channels,
            spatial={"status": "not_evaluable", "reason": "missing_scene"},
        )

    try:
        rir, sample_rate = sf.read(rir_path, always_2d=True, dtype="float64")
    except (OSError, RuntimeError) as error:
        _gate(
            checks,
            "audio_is_readable",
            "fail",
            measured=None,
            criterion="RIR is readable as an audio matrix",
            reason=str(error),
        )
        return _completed_report(
            item,
            policy,
            checks,
            channels,
            spatial={"status": "not_evaluable", "reason": "unreadable_audio"},
        )
    _gate(
        checks,
        "audio_is_readable",
        "pass",
        measured=True,
        criterion="RIR is readable as an audio matrix",
    )
    shape_matches = bool(
        int(sample_rate) == item.sample_rate
        and rir.shape == (item.frame_count, item.channel_count)
    )
    _gate(
        checks,
        "audio_shape_matches_manifest",
        "pass" if shape_matches else "fail",
        measured={
            "sample_rate": int(sample_rate),
            "frame_count": int(rir.shape[0]),
            "channel_count": int(rir.shape[1]),
        },
        criterion=(
            f"sample_rate={item.sample_rate}, frame_count={item.frame_count}, "
            f"channel_count={item.channel_count}"
        ),
    )
    finite_audio = bool(np.isfinite(rir).all())
    _gate(
        checks,
        "audio_is_finite",
        "pass" if finite_audio else "fail",
        measured=finite_audio,
        criterion="every sample is finite",
    )
    safe_rir = np.nan_to_num(rir, nan=0.0, posinf=0.0, neginf=0.0)
    peak_abs = float(np.max(np.abs(safe_rir))) if safe_rir.size else 0.0
    total_energy = float(np.square(safe_rir, dtype=np.float64).sum())
    _gate(
        checks,
        "audio_is_non_silent",
        "pass" if total_energy >= policy.minimum_total_energy else "fail",
        measured=total_energy,
        criterion=f"total energy >= {policy.minimum_total_energy:g}",
    )
    peak_is_valid = bool(
        item.level_policy == "calibrated"
        or peak_abs <= policy.maximum_peak_abs + 1e-7
    )
    _gate(
        checks,
        "peak_is_within_level_policy",
        "pass" if peak_is_valid else "fail",
        measured={
            "peak_abs": peak_abs,
            "level_policy": item.level_policy,
        },
        criterion=(
            "calibrated float RIRs require only a finite peak; "
            f"peak_normalized RIRs require absolute peak <= "
            f"{policy.maximum_peak_abs:g}"
        ),
    )

    channel_map = scene.get("channel_map")
    channel_map_valid = bool(
        isinstance(channel_map, list)
        and len(channel_map) == item.channel_count
        and all(isinstance(entry, Mapping) for entry in channel_map)
    )
    channel_indices: list[int] = []
    if channel_map_valid:
        try:
            channel_indices = [int(entry["channel"]) for entry in channel_map]
        except (KeyError, TypeError, ValueError):
            channel_map_valid = False
    channel_map_valid = bool(
        channel_map_valid
        and sorted(channel_indices) == list(range(item.channel_count))
        and len(set(channel_indices)) == item.channel_count
    )
    _gate(
        checks,
        "channel_map_matches_audio",
        "pass" if channel_map_valid else "fail",
        measured={"entries": len(channel_map) if isinstance(channel_map, list) else None},
        criterion="channel_map has one unique indexed entry per audio channel",
    )
    if not shape_matches or not channel_map_valid:
        return _completed_report(
            item,
            policy,
            checks,
            channels,
            spatial={"status": "not_evaluable", "reason": "invalid_channel_layout"},
        )

    environment = scene.get("environment")
    sound_speed = _finite(
        environment.get("sound_speed_m_s")
        if isinstance(environment, Mapping)
        else None
    )
    if sound_speed is None:
        sound_speed = 343.0
        sound_speed_status = "not_evaluable"
        sound_speed_reason = "scene_sound_speed_missing; used 343 m/s fallback"
    elif sound_speed <= 0.0:
        sound_speed_status = "fail"
        sound_speed_reason = "scene sound speed is not positive"
        sound_speed = 343.0
    else:
        sound_speed_status = "pass"
        sound_speed_reason = None
    _gate(
        checks,
        "sound_speed_is_physical",
        sound_speed_status,
        measured=sound_speed,
        criterion="finite positive sound speed, or explicit 343 m/s compatibility fallback",
        reason=sound_speed_reason,
    )

    per_channel_failures: list[str] = []
    valid_decay_channels = 0
    near_drr: list[float] = []
    far_drr: list[float] = []
    for entry in sorted(channel_map, key=lambda value: int(value["channel"])):
        channel = int(entry["channel"])
        label = str(entry.get("label", ""))
        signal = safe_rir[:, channel]
        magnitude = np.abs(signal)
        channel_peak = float(np.max(magnitude)) if magnitude.size else 0.0
        distance_m = _finite(entry.get("distance_m"))
        channel_failures: list[str] = []
        if distance_m is None or distance_m < 0.0:
            channel_failures.append("invalid_distance")
            channels.append(
                {
                    "channel": channel,
                    "label": label,
                    "distance_m": distance_m,
                    "status": "fail",
                    "failure_reasons": channel_failures,
                }
            )
            per_channel_failures.append(f"channel_{channel}_invalid_distance")
            continue

        expected_direct = distance_m / sound_speed * float(sample_rate)
        first_physical = max(0, int(math.floor(expected_direct)))
        prearrival_peak = (
            float(np.max(magnitude[:first_physical])) if first_physical else 0.0
        )
        prearrival_relative = prearrival_peak / max(channel_peak, 1e-30)
        if prearrival_relative > policy.maximum_prearrival_relative_peak:
            channel_failures.append("prearrival_energy")

        search_end = min(
            signal.size,
            int(math.ceil(expected_direct))
            + max(
                1,
                int(
                    round(
                        policy.direct_search_after_expected_ms
                        * sample_rate
                        / 1000.0
                    )
                ),
            ),
        )
        if search_end <= first_physical:
            direct_index = min(first_physical, max(0, signal.size - 1))
        else:
            direct_index = first_physical + int(
                np.argmax(magnitude[first_physical:search_end])
            )

        # Do not scan from sample zero: a finite modal reconstruction may have
        # tiny numerical residue before the geometric arrival.  The direct
        # onset is only meaningful in the physically allowed search window.
        onset_threshold = max(1e-12, channel_peak * 1e-8)
        onset_candidates = np.flatnonzero(
            magnitude[first_physical:search_end] >= onset_threshold
        )
        onset_sample = (
            first_physical + int(onset_candidates[0])
            if onset_candidates.size
            else None
        )
        arrival_error_ms = (
            abs(float(onset_sample) - expected_direct)
            / float(sample_rate)
            * 1000.0
            if onset_sample is not None
            else None
        )
        if arrival_error_ms is None or arrival_error_ms > policy.maximum_arrival_error_ms:
            channel_failures.append("direct_arrival_timing")

        tail_start = min(
            signal.size,
            first_physical + int(round(policy.tail_start_ms * sample_rate / 1000.0)),
        )
        channel_energy = float(np.square(signal, dtype=np.float64).sum())
        tail_fraction = float(
            np.square(signal[tail_start:], dtype=np.float64).sum()
            / max(channel_energy, 1e-30)
        )
        if tail_fraction < policy.minimum_tail_energy_fraction:
            channel_failures.append("insufficient_tail_energy")

        try:
            metrics = analyze_rir(
                signal,
                int(sample_rate),
                direct_window_ms=policy.direct_window_ms,
                direct_index=direct_index,
                octave_centers_hz=policy.octave_centers_hz,
                echo_density=True,
            )
            active_threshold = max(1e-12, channel_peak * 1e-8)
            active_samples = np.flatnonzero(magnitude >= active_threshold)
            echo_half_window = max(
                1,
                int(round(0.5 * 20.0e-3 * sample_rate)),
            )
            if active_samples.size:
                echo_analysis_end = min(
                    signal.size,
                    int(active_samples[-1]) + 2 * echo_half_window + 2,
                )
            else:
                echo_analysis_end = signal.size
            echo_analysis_end = max(
                min(signal.size, echo_analysis_end),
                min(
                    signal.size,
                    direct_index + 2 * echo_half_window + 2,
                ),
            )
            metrics["echo_density"] = analyze_echo_density(
                signal[:echo_analysis_end],
                int(sample_rate),
                direct_index=direct_index,
                window_ms=20.0,
                hop_ms=1.0,
                threshold=0.9,
                minimum_sustain_ms=10.0,
            )
            metrics["echo_density_analysis"] = {
                "policy": "active_tail_plus_one_full_analysis_window",
                "active_threshold_abs": active_threshold,
                "analysis_end_sample": int(echo_analysis_end),
                "full_record_end_sample": int(signal.size),
            }
        except (TypeError, ValueError, FloatingPointError) as error:
            metrics = {"analysis_error": str(error)}
            channel_failures.append("metric_analysis_error")

        for metric_name in ("drr_db", "c50_db", "c80_db"):
            value = _finite(metrics.get(metric_name))
            if value is None or abs(value) > policy.maximum_abs_level_metric_db:
                channel_failures.append(f"implausible_{metric_name}")
        drr = _finite(metrics.get("drr_db"))
        if drr is not None:
            (near_drr if label.startswith("near") else far_drr).append(drr)
        spectral_tilt = _finite(metrics.get("spectral_tilt_db_per_octave"))
        if (
            spectral_tilt is None
            or abs(spectral_tilt) > policy.maximum_abs_spectral_tilt_db_per_octave
        ):
            channel_failures.append("implausible_spectral_tilt")

        t20 = metrics.get("t20")
        decay_status = "not_evaluable"
        decay_reason = "T20 interval is not available above the estimated noise floor"
        if isinstance(t20, Mapping):
            rt60_s = _finite(t20.get("rt60_s"))
            r_squared = _finite(t20.get("r_squared"))
            if r_squared is not None and r_squared >= policy.minimum_decay_r_squared:
                valid_decay_channels += 1
                if (
                    rt60_s is not None
                    and policy.minimum_rt60_s <= rt60_s <= policy.maximum_rt60_s
                ):
                    decay_status = "pass"
                    decay_reason = None
                else:
                    decay_status = "fail"
                    decay_reason = "T20 RT60 extrapolation lies outside policy bounds"
                    channel_failures.append("implausible_t20")
            else:
                decay_reason = "T20 fit quality is below the policy threshold"

        octave_bands = metrics.get("octave_bands", {})
        reliable_octave_bands = 0
        octave_band_count = 0
        if isinstance(octave_bands, Mapping):
            for octave in octave_bands.values():
                if not isinstance(octave, Mapping):
                    continue
                octave_band_count += 1
                octave_rt60 = _finite(octave.get("t20_s"))
                raw_fit = octave.get("decay_fit_r2", {})
                octave_r_squared = (
                    _finite(raw_fit.get("t20"))
                    if isinstance(raw_fit, Mapping)
                    else None
                )
                if (
                    octave_rt60 is not None
                    and policy.minimum_rt60_s
                    <= octave_rt60
                    <= policy.maximum_rt60_s
                    and octave_r_squared is not None
                    and octave_r_squared >= policy.minimum_decay_r_squared
                ):
                    reliable_octave_bands += 1
        octave_coverage = reliable_octave_bands / max(1, octave_band_count)
        octave_required = bool(
            item.frame_count / float(item.sample_rate)
            >= policy.decay_required_duration_s
        )
        if (
            octave_required
            and octave_coverage
            < policy.minimum_octave_decay_coverage_fraction
        ):
            channel_failures.append("octave_decay_coverage")

        echo_density = metrics.get("echo_density", {})
        late_density = (
            _finite(echo_density.get("late_median_normalized_density"))
            if isinstance(echo_density, Mapping)
            else None
        )
        if late_density is None:
            echo_status = "fail"
            echo_reason = "echo-density profile is unavailable in the active tail"
            channel_failures.append("insufficient_late_echo_density")
        elif late_density < policy.minimum_late_echo_density:
            echo_status = "fail"
            echo_reason = "late echo density is below the minimum"
            channel_failures.append("insufficient_late_echo_density")
        else:
            echo_status = "pass"
            echo_reason = None

        channel_failures = sorted(set(channel_failures))
        per_channel_failures.extend(
            f"channel_{channel}_{reason}" for reason in channel_failures
        )
        channels.append(
            {
                "channel": channel,
                "label": label,
                "distance_m": distance_m,
                "status": "fail" if channel_failures else "pass",
                "failure_reasons": channel_failures,
                "geometry": {
                    "sound_speed_m_s": sound_speed,
                    "expected_direct_sample": expected_direct,
                    "first_physical_sample": first_physical,
                    "observed_onset_sample": onset_sample,
                    "direct_analysis_sample": direct_index,
                    "arrival_error_ms": arrival_error_ms,
                    "prearrival_peak_abs": prearrival_peak,
                    "prearrival_relative_peak": prearrival_relative,
                },
                "energy": {
                    "peak_abs": channel_peak,
                    "total": channel_energy,
                    "tail_start_sample": tail_start,
                    "tail_fraction": tail_fraction,
                },
                "decay_assessment": {
                    "status": decay_status,
                    "reason": decay_reason,
                },
                "octave_decay_assessment": {
                    "status": (
                        "pass"
                        if not octave_required
                        or octave_coverage
                        >= policy.minimum_octave_decay_coverage_fraction
                        else "fail"
                    ),
                    "reliable_band_count": reliable_octave_bands,
                    "band_count": octave_band_count,
                    "coverage_fraction": octave_coverage,
                    "minimum_required_fraction": (
                        policy.minimum_octave_decay_coverage_fraction
                    ),
                },
                "echo_density_assessment": {
                    "status": echo_status,
                    "reason": echo_reason,
                },
                "metrics": metrics,
            }
        )

    _gate(
        checks,
        "per_channel_physical_checks",
        "pass" if not per_channel_failures else "fail",
        measured={"failed_reasons": sorted(per_channel_failures)},
        criterion=(
            "each channel is causal, arrives at geometric time, has late energy, "
            "and has finite bounded acoustic metrics"
        ),
    )
    decay_coverage = valid_decay_channels / max(1, item.channel_count)
    requires_decay = item.frame_count / float(item.sample_rate) >= (
        policy.decay_required_duration_s
    )
    if not requires_decay:
        decay_gate_status = "not_evaluable"
        decay_gate_reason = "RIR duration is shorter than decay_required_duration_s"
    elif decay_coverage < policy.minimum_decay_coverage_fraction:
        decay_gate_status = "fail"
        decay_gate_reason = "too few channels contain a reliable T20 fit"
    else:
        decay_gate_status = "pass"
        decay_gate_reason = None
    _gate(
        checks,
        "decay_fit_coverage",
        decay_gate_status,
        measured={
            "valid_t20_channels": valid_decay_channels,
            "channel_count": item.channel_count,
            "fraction": decay_coverage,
        },
        criterion=(
            f"T20 fit R² >= {policy.minimum_decay_r_squared:g} for at least "
            f"{policy.minimum_decay_coverage_fraction:g} of channels when duration "
            f">= {policy.decay_required_duration_s:g} s"
        ),
        reason=decay_gate_reason,
    )
    near_median = float(np.median(near_drr)) if near_drr else None
    far_median = float(np.median(far_drr)) if far_drr else None
    _gate(
        checks,
        "near_far_drr_relationship",
        "pass" if near_median is not None and far_median is not None else "not_evaluable",
        measured={
            "near_median_drr_db": near_median,
            "far_median_drr_db": far_median,
            "gap_db": (
                near_median - far_median
                if near_median is not None and far_median is not None
                else None
            ),
        },
        criterion="record near/far DRR relationship for bank-level distribution QC",
        reason=(
            None
            if near_median is not None and far_median is not None
            else "near and far labelled channels are both required"
        ),
        severity="informational",
    )
    spatial = {
        "status": "not_applicable",
        "reason": (
            "channels represent different source-to-mono-receiver transfer paths; "
            "they are not synchronized receivers for one source"
        ),
        "required_evidence": (
            "one source observed by a synchronized receiver pair/array with geometry"
        ),
    }
    _gate(
        checks,
        "spatial_pair_metrics",
        "not_applicable",
        measured=None,
        criterion="IACC/coherence only on synchronized receiver channels for one source",
        reason=spatial["reason"],
        severity="informational",
    )
    return _completed_report(item, policy, checks, channels, spatial=spatial)


def _candidate_row(item: RIRBankItem) -> dict[str, Any]:
    row = split_index_rows((item,), item.split)[0]
    row["qc_report_path"] = item.qc_report_path
    row["qc_report_sha256"] = item.qc_report_sha256
    return row


def _evaluate_rir_bank_item_worker(
    payload: tuple[str, RIRBankItem, RIRBankQCPolicy],
) -> tuple[str, dict[str, Any]]:
    """Evaluate one item in a worker and return its stable item id/report."""

    root, item, policy = payload
    return item.item_id, evaluate_rir_bank_item(root, item, policy)


def _quarantine_row(item: RIRBankItem, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "room_id": item.room_id,
        "acoustic_space_id": item.acoustic_space_id,
        "split": item.split,
        "rir_path": item.rir_path,
        "metadata_path": item.metadata_path,
        "qc_report_path": item.qc_report_path,
        "qc_report_sha256": item.qc_report_sha256,
        "failure_reasons": list(report["failure_reasons"]),
    }


def _write_index(root: Path, relative_path: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = root / relative_path
    _atomic_write_jsonl(path, rows)
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "item_count": len(rows),
    }


def _safe_audit_path(root: Path, value: Any) -> Path | None:
    relative = PurePosixPath(str(value or ""))
    if not value or relative.is_absolute() or ".." in relative.parts:
        return None
    return root / relative


def run_rir_bank_qc(
    root: str | Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    summary_name: str = DEFAULT_QC_SUMMARY_NAME,
    policy: RIRBankQCPolicy | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate every manifest item and atomically publish QC admission indexes."""

    bank_root = Path(root)
    manifest_path = bank_root / manifest_name
    manifest = RIRBankManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    source_audit = audit_rir_bank_manifest(manifest, bank_root)
    if not source_audit["ready_for_m6_bank_generation"]:
        raise ValueError("source M6 manifest or assets failed integrity audit")
    policy = policy or RIRBankQCPolicy()
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    source_manifest_sha256 = manifest.manifest_sha256
    previous_summary_path = bank_root / summary_name
    if previous_summary_path.is_file():
        try:
            previous_summary = json.loads(
                previous_summary_path.read_text(encoding="utf-8")
            )
            if (
                previous_summary.get("schema_version")
                == RIR_BANK_QC_RELEASE_SCHEMA_VERSION
                and previous_summary.get("qc_manifest_sha256")
                == manifest.manifest_sha256
            ):
                source_manifest_sha256 = previous_summary.get(
                    "source_manifest_sha256", source_manifest_sha256
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    reports: dict[str, dict[str, Any]] = {}
    completed_items: list[RIRBankItem] = []
    ordered_items = tuple(sorted(manifest.items, key=lambda value: value.item_id))
    item_by_id = {item.item_id: item for item in ordered_items}
    payloads = ((str(bank_root), item, policy) for item in ordered_items)
    if int(workers) == 1:
        evaluations = (
            (item.item_id, evaluate_rir_bank_item(bank_root, item, policy))
            for item in ordered_items
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=int(workers))
        evaluations = executor.map(
            _evaluate_rir_bank_item_worker,
            payloads,
            chunksize=1,
        )
    try:
        for item_id, report in evaluations:
            item = item_by_id[item_id]
            report_path = bank_root / "qc" / "items" / f"{item.item_id}.json"
            _atomic_write_json(report_path, report)
            completed = replace(
                item,
                qc_status=str(report["status"]),
                qc_report_path=report_path.relative_to(bank_root).as_posix(),
                qc_report_sha256=sha256_file(report_path),
            )
            reports[item.item_id] = report
            completed_items.append(completed)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    completed_items_tuple = tuple(
        sorted(completed_items, key=lambda value: value.item_id)
    )
    candidate_indexes = {
        split: _write_index(
            bank_root,
            f"qc/candidate_indexes/{split}.jsonl",
            [
                _candidate_row(item)
                for item in completed_items_tuple
                if item.split == split and item.qc_status == "pass"
            ],
        )
        for split in BANK_SPLITS
    }
    quarantined = [
        item for item in completed_items_tuple if item.qc_status == "fail"
    ]
    quarantine_index = _write_index(
        bank_root,
        "qc/quarantine/index.jsonl",
        [_quarantine_row(item, reports[item.item_id]) for item in quarantined],
    )
    candidate_ids = {
        item.item_id for item in completed_items_tuple if item.qc_status == "pass"
    }
    quarantined_ids = {item.item_id for item in quarantined}
    candidate_splits_nonempty = all(
        candidate_indexes[split]["item_count"] > 0 for split in BANK_SPLITS
    )
    completed_manifest = replace(
        manifest,
        release_status="candidate" if candidate_splits_nonempty else "draft",
        items=completed_items_tuple,
        manifest_sha256=None,
    ).with_content_sha256()
    _atomic_write_json(manifest_path, completed_manifest.to_dict())
    completed_audit = audit_rir_bank_manifest(completed_manifest, bank_root)

    failure_counts = Counter(
        reason
        for report in reports.values()
        for reason in report["failure_reasons"]
    )
    summary: dict[str, Any] = {
        "schema_version": RIR_BANK_QC_RELEASE_SCHEMA_VERSION,
        "bank_id": completed_manifest.bank_id,
        "source_manifest_sha256": source_manifest_sha256,
        "qc_manifest_path": manifest_name,
        "qc_manifest_sha256": completed_manifest.manifest_sha256,
        "policy": policy.to_dict(),
        "policy_sha256": policy.content_sha256(),
        "candidate_indexes": candidate_indexes,
        "quarantine_index": quarantine_index,
        "counts": {
            "total": len(completed_items_tuple),
            "passed": sum(item.qc_status == "pass" for item in completed_items_tuple),
            "quarantined": len(quarantined),
            "by_split": {
                split: {
                    "total": sum(item.split == split for item in completed_items_tuple),
                    "passed": candidate_indexes[split]["item_count"],
                    "quarantined": sum(
                        item.split == split and item.qc_status == "fail"
                        for item in completed_items_tuple
                    ),
                }
                for split in BANK_SPLITS
            },
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "checks": {
            "source_manifest_integrity_passed": True,
            "every_item_has_a_content_addressed_report": all(
                item.qc_report_path is not None and item.qc_report_sha256 is not None
                for item in completed_items_tuple
            ),
            "failed_items_are_excluded_from_candidate_indexes": all(
                item_id not in candidate_ids for item_id in quarantined_ids
            ),
            "all_failed_items_are_quarantined": (
                quarantine_index["item_count"] == len(quarantined)
            ),
            "candidate_has_train_validation_test_items": candidate_splits_nonempty,
            "completed_manifest_integrity_passed": completed_audit[
                "ready_for_m6_bank_generation"
            ],
        },
        "release": {
            "status": completed_manifest.release_status,
            "ready_for_candidate": bool(
                candidate_splits_nonempty
                and completed_audit["ready_for_m6_bank_generation"]
            ),
            "ready_for_production": False,
            "production_blockers": [
                "renderer profiles still require empirical production approval",
                "M6.4 distribution release is incomplete",
                "M6.5 downstream and listening evaluation is incomplete",
            ],
        },
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_json(bank_root / summary_name, summary)
    return summary


def audit_rir_bank_qc_release(
    root: str | Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    summary_name: str = DEFAULT_QC_SUMMARY_NAME,
) -> dict[str, Any]:
    """Recompute hashes and membership for an already-published QC release."""

    bank_root = Path(root)
    summary = json.loads((bank_root / summary_name).read_text(encoding="utf-8"))
    manifest = RIRBankManifest.from_json(
        (bank_root / manifest_name).read_text(encoding="utf-8")
    )
    stored_summary_hash = summary.get("summary_sha256")
    unhashed_summary = dict(summary)
    unhashed_summary.pop("summary_sha256", None)
    report_hashes_match = True
    reports: dict[str, dict[str, Any]] = {}
    for item in manifest.items:
        if item.qc_report_path is None or item.qc_report_sha256 is None:
            report_hashes_match = False
            continue
        path = bank_root / item.qc_report_path
        if not path.is_file() or sha256_file(path) != item.qc_report_sha256:
            report_hashes_match = False
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            report_hashes_match = False
            continue
        reports[item.item_id] = report
        asset_identity = report.get("asset_identity")
        if (
            report.get("schema_version") != RIR_BANK_ITEM_QC_SCHEMA_VERSION
            or report.get("item_id") != item.item_id
            or report.get("status") != item.qc_status
            or report.get("policy_sha256") != summary.get("policy_sha256")
            or not isinstance(asset_identity, Mapping)
            or asset_identity.get("rir_path") != item.rir_path
            or asset_identity.get("rir_sha256") != item.rir_sha256
            or asset_identity.get("metadata_path") != item.metadata_path
            or asset_identity.get("metadata_sha256") != item.metadata_sha256
            or asset_identity.get("scene_sha256") != item.scene_sha256
        ):
            report_hashes_match = False

    candidate_indexes_match = True
    candidate_ids: set[str] = set()
    for split in BANK_SPLITS:
        reference = summary["candidate_indexes"][split]
        path = _safe_audit_path(bank_root, reference.get("path"))
        expected = [
            _candidate_row(item)
            for item in sorted(manifest.items, key=lambda value: value.item_id)
            if item.split == split and item.qc_status == "pass"
        ]
        try:
            if path is None:
                raise OSError("unsafe candidate index path")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            rows = []
            candidate_indexes_match = False
        candidate_ids.update(str(row.get("item_id")) for row in rows)
        candidate_indexes_match = bool(
            candidate_indexes_match
            and rows == expected
            and reference["item_count"] == len(expected)
            and path is not None
            and path.is_file()
            and sha256_file(path) == reference["sha256"]
        )

    quarantine_reference = summary["quarantine_index"]
    quarantine_path = _safe_audit_path(
        bank_root, quarantine_reference.get("path")
    )
    failed_items = [
        item
        for item in sorted(manifest.items, key=lambda value: value.item_id)
        if item.qc_status == "fail" and item.item_id in reports
    ]
    expected_quarantine = [
        _quarantine_row(item, reports[item.item_id]) for item in failed_items
    ]
    try:
        if quarantine_path is None:
            raise OSError("unsafe quarantine index path")
        quarantine_rows = [
            json.loads(line)
            for line in quarantine_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        quarantine_rows = []
    quarantine_matches = bool(
        quarantine_rows == expected_quarantine
        and quarantine_reference["item_count"] == len(expected_quarantine)
        and quarantine_path is not None
        and quarantine_path.is_file()
        and sha256_file(quarantine_path) == quarantine_reference["sha256"]
    )
    manifest_audit = audit_rir_bank_manifest(manifest, bank_root)
    pass_ids = {item.item_id for item in manifest.items if item.qc_status == "pass"}
    fail_ids = {item.item_id for item in manifest.items if item.qc_status == "fail"}
    quarantine_ids = {str(row.get("item_id")) for row in quarantine_rows}
    expected_counts = {
        "total": len(manifest.items),
        "passed": len(pass_ids),
        "quarantined": len(fail_ids),
        "by_split": {
            split: {
                "total": sum(item.split == split for item in manifest.items),
                "passed": sum(
                    item.split == split and item.qc_status == "pass"
                    for item in manifest.items
                ),
                "quarantined": sum(
                    item.split == split and item.qc_status == "fail"
                    for item in manifest.items
                ),
            }
            for split in BANK_SPLITS
        },
    }
    candidate_splits_nonempty = all(
        expected_counts["by_split"][split]["passed"] > 0
        for split in BANK_SPLITS
    )
    expected_release_status = "candidate" if candidate_splits_nonempty else "draft"
    release = summary.get("release", {})
    checks = {
        "summary_schema_is_supported": (
            summary.get("schema_version") == RIR_BANK_QC_RELEASE_SCHEMA_VERSION
        ),
        "summary_content_hash_matches": (
            stored_summary_hash == canonical_json_sha256(unhashed_summary)
        ),
        "manifest_hash_matches_summary": (
            manifest.manifest_sha256 == summary.get("qc_manifest_sha256")
        ),
        "policy_hash_matches_summary": (
            canonical_json_sha256(summary["policy"]) == summary.get("policy_sha256")
        ),
        "manifest_integrity_passed": manifest_audit["ready_for_m6_bank_generation"],
        "item_report_hashes_and_statuses_match": report_hashes_match,
        "candidate_indexes_match_passed_items": (
            candidate_indexes_match and candidate_ids == pass_ids
        ),
        "quarantine_index_matches_failed_items": (
            quarantine_matches and quarantine_ids == fail_ids
        ),
        "candidate_and_quarantine_are_disjoint": not (
            candidate_ids & quarantine_ids
        ),
        "summary_counts_match_manifest": summary.get("counts") == expected_counts,
        "release_decision_matches_candidate_coverage": (
            isinstance(release, Mapping)
            and release.get("status") == expected_release_status
            and bool(release.get("ready_for_candidate")) == candidate_splits_nonempty
            and release.get("ready_for_production") is False
            and manifest.release_status == expected_release_status
        ),
    }
    return {
        "schema_version": "puresound.rir_bank_qc_audit.v1",
        "bank_id": manifest.bank_id,
        "checks": checks,
        "valid": bool(all(checks.values())),
        "candidate_item_count": len(candidate_ids),
        "quarantine_item_count": len(quarantine_ids),
    }


__all__ = [
    "DEFAULT_QC_SUMMARY_NAME",
    "RIR_BANK_ITEM_QC_SCHEMA_VERSION",
    "RIR_BANK_QC_POLICY_ID",
    "RIR_BANK_QC_RELEASE_SCHEMA_VERSION",
    "RIRBankQCPolicy",
    "audit_rir_bank_qc_release",
    "evaluate_rir_bank_item",
    "run_rir_bank_qc",
]
