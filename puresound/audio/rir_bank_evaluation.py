"""M6.5 bank-level acoustic, listening, and downstream evidence contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from puresound.audio.rir_bank_manifest import canonical_json_sha256
from puresound.audio.rir_bank_release import (
    RIRBankReleaseManifest,
    audit_m6_variant_release,
)


M6_THROUGHPUT_SCHEMA_VERSION = "puresound.m6_generation_throughput.v1"
M6_LISTENING_SCHEMA_VERSION = "puresound.m6_controlled_listening.v1"
M6_DOWNSTREAM_SCHEMA_VERSION = "puresound.m6_downstream_evaluation.v1"
M6_EVALUATION_SCHEMA_VERSION = "puresound.m6_bank_evaluation.v1"
EVIDENCE_TIERS = ("contract_fixture", "empirical")


def _paired_t_confidence_interval(
    values: Sequence[Any],
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float] | None:
    samples = np.asarray(tuple(values), dtype=np.float64)
    if (
        samples.ndim != 1
        or samples.size < 2
        or not np.isfinite(samples).all()
        or not 0.0 < float(confidence_level) < 1.0
    ):
        return None
    mean = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
    if standard_error == 0.0:
        return mean, mean
    critical = float(
        student_t.ppf(
            0.5 + 0.5 * float(confidence_level),
            int(samples.size - 1),
        )
    )
    return mean - critical * standard_error, mean + critical * standard_error


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric) or numeric != float(result):
        return default
    return result


def _sha256(value: Any) -> bool:
    digest = str(value or "").lower()
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _load_json(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("evidence JSON root must be an object")
    return parsed


def _release_and_distributions(
    release_root: Path,
) -> tuple[RIRBankReleaseManifest, dict[str, dict[str, Any]]]:
    release = RIRBankReleaseManifest.from_json(
        (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    distributions = {
        variant.variant_id: json.loads(
            (release_root / variant.distribution_path).read_text(encoding="utf-8")
        )
        for variant in release.variants
    }
    return release, distributions


def _channel_map(distribution: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["item_id"]), int(row["channel"])): dict(row)
        for row in distribution["channels"]
    }


def _variant_invariance(
    calibrated: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    first = _channel_map(calibrated)
    second = {
        (str(row["item_id"]).removesuffix("__peak_normalized"), int(row["channel"])): row
        for row in normalized["channels"]
    }
    metrics = ("drr_db", "c50_db", "c80_db", "t20_s", "mixing_time_s")
    errors: dict[str, float | None] = {}
    for metric in metrics:
        values = [
            abs(float(row[metric]) - float(second[key][metric]))
            for key, row in first.items()
            if key in second and row.get(metric) is not None and second[key].get(metric) is not None
        ]
        errors[metric] = max(values) if values else None
    matched = set(first) == set(second)
    return {
        "status": "pass" if matched else "fail",
        "matched_channel_identities": matched,
        "maximum_absolute_error": errors,
        "level_dependent_peak_change_expected": True,
    }


def _normalized_wasserstein(
    first: Sequence[Any],
    reference: Sequence[Any],
) -> float | None:
    a = np.asarray(
        [float(value) for value in first if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    b = np.asarray(
        [
            float(value)
            for value in reference
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if not a.size or not b.size:
        return None
    quantiles = np.linspace(0.0, 1.0, 101)
    distance = float(np.mean(np.abs(np.quantile(a, quantiles) - np.quantile(b, quantiles))))
    scale = float(np.quantile(b, 0.95) - np.quantile(b, 0.05))
    return distance / max(scale, 1e-12)


def compare_release_distributions(release_root: str | Path) -> dict[str, Any]:
    """Compare level variants and, when present, synthetic against measured."""

    root = Path(release_root)
    release, distributions = _release_and_distributions(root)
    calibrated = distributions.get("synthetic_calibrated")
    normalized = distributions.get("synthetic_peak_normalized")
    invariance = (
        _variant_invariance(calibrated, normalized)
        if calibrated is not None and normalized is not None
        else {
            "status": "not_evaluable",
            "reason": "calibrated and normalized variants are both required",
        }
    )
    real_variants = [
        variant for variant in release.variants if variant.origin == "real"
    ]
    if calibrated is None or not real_variants:
        measured = {
            "status": "not_evaluable",
            "reason": "no QC-passed measured variant is present in this release",
            "distances": None,
        }
    else:
        real = distributions[real_variants[0].variant_id]
        metrics = (
            "distance_m",
            "drr_db",
            "c50_db",
            "c80_db",
            "t20_s",
            "spectral_tilt_db_per_octave",
            "mixing_time_s",
            "late_median_normalized_density",
        )
        measured = {
            "status": "evaluated",
            "reference_variant_id": real_variants[0].variant_id,
            "distance_policy": "quantile_wasserstein_divided_by_reference_p05_p95_span",
            "distances": {
                metric: _normalized_wasserstein(
                    [row.get(metric) for row in calibrated["channels"]],
                    [row.get(metric) for row in real["channels"]],
                )
                for metric in metrics
            },
        }
    result: dict[str, Any] = {
        "schema_version": "puresound.m6_acoustic_distribution_comparison.v1",
        "release_sha256": release.release_sha256,
        "variant_scale_invariance": invariance,
        "synthetic_to_measured": measured,
    }
    result["comparison_sha256"] = canonical_json_sha256(result)
    return result


def validate_throughput_report(
    report: Mapping[str, Any] | None,
    *,
    release_sha256: str,
) -> dict[str, Any]:
    if report is None:
        return {
            "valid": False,
            "empirical": False,
            "checks": {"report_present": False},
            "reason": "throughput report is missing",
        }
    elapsed = _finite(report.get("elapsed_seconds"))
    generated = _integer(report.get("items_generated"), -1)
    skipped = _integer(report.get("items_skipped"), -1)
    failed = _integer(report.get("items_failed"), -1)
    task_count = _integer(report.get("task_count"), -1)
    worker_count = _integer(report.get("num_workers"), 0)
    assert generated is not None
    assert skipped is not None
    assert failed is not None
    assert task_count is not None
    assert worker_count is not None
    rate = _finite(report.get("items_per_second"))
    checks = {
        "schema_matches": report.get("schema_version") == M6_THROUGHPUT_SCHEMA_VERSION,
        "release_matches": report.get("release_sha256") == release_sha256,
        "counts_are_consistent": (
            min(generated, skipped, failed, task_count) >= 0
            and generated + skipped + failed == task_count
        ),
        "elapsed_time_is_positive": elapsed is not None and elapsed > 0.0,
        "reported_rate_is_consistent": bool(
            elapsed is not None
            and elapsed > 0.0
            and rate is not None
            and math.isclose(rate, generated / elapsed, rel_tol=1e-9, abs_tol=1e-12)
        ),
        "worker_count_is_positive": worker_count > 0,
        "failure_count_is_explicit": failed >= 0,
    }
    return {
        "valid": bool(all(checks.values())),
        "empirical": bool(all(checks.values())),
        "checks": checks,
        "summary": {
            "task_count": task_count,
            "items_generated": generated,
            "items_skipped": skipped,
            "items_failed": failed,
            "elapsed_seconds": elapsed,
            "items_per_second": rate,
        },
    }


def validate_listening_report(
    report: Mapping[str, Any] | None,
    *,
    release_sha256: str,
) -> dict[str, Any]:
    if report is None:
        return {
            "valid": False,
            "empirical": False,
            "checks": {"report_present": False},
            "reason": "controlled listening report is missing",
        }
    raw_protocol = report.get("protocol", {})
    raw_results = report.get("results", {})
    raw_artifacts = report.get("artifacts", {})
    protocol = raw_protocol if isinstance(raw_protocol, Mapping) else {}
    results = raw_results if isinstance(raw_results, Mapping) else {}
    artifacts = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    tier = report.get("evidence_tier")
    participants = _integer(protocol.get("participant_count"), 0)
    assert participants is not None
    margin = _finite(results.get("noninferiority_margin"))
    ci_low = _finite(results.get("confidence_interval_low"))
    explicitly_nonhuman = bool(
        protocol.get("explicitly_not_human_responses")
    )
    empirical_records_valid = True
    empirical_statistics_valid = True
    if tier == "empirical":
        assignments = artifacts.get("assignment_records")
        responses = artifacts.get("response_records")
        analysis_record = artifacts.get("analysis_record")
        empirical_records_valid = bool(
            isinstance(assignments, list)
            and assignments
            and isinstance(responses, list)
            and responses
            and isinstance(analysis_record, Mapping)
            and canonical_json_sha256(assignments)
            == artifacts.get("assignment_sha256")
            and canonical_json_sha256(responses)
            == artifacts.get("responses_sha256")
            and canonical_json_sha256(analysis_record)
            == artifacts.get("analysis_sha256")
        )
        participant_values: dict[str, list[float]] = {}
        if empirical_records_valid:
            for response in responses:
                if not isinstance(response, Mapping):
                    empirical_records_valid = False
                    break
                participant_id = str(response.get("participant_id", "")).strip()
                value = _finite(response.get("primary_endpoint_difference"))
                if not participant_id or value is None:
                    empirical_records_valid = False
                    break
                participant_values.setdefault(participant_id, []).append(value)
        participant_means = [
            float(np.mean(values))
            for _participant_id, values in sorted(participant_values.items())
        ]
        interval = _paired_t_confidence_interval(participant_means)
        estimate = _finite(results.get("estimate"))
        analysis_estimate = (
            _finite(analysis_record.get("estimate"))
            if isinstance(analysis_record, Mapping)
            else None
        )
        analysis_ci_low = (
            _finite(analysis_record.get("confidence_interval_low"))
            if isinstance(analysis_record, Mapping)
            else None
        )
        empirical_statistics_valid = bool(
            empirical_records_valid
            and len(participant_values) == participants
            and interval is not None
            and estimate is not None
            and ci_low is not None
            and analysis_estimate is not None
            and analysis_ci_low is not None
            and math.isclose(
                estimate,
                float(np.mean(participant_means)),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                ci_low,
                interval[0],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                analysis_estimate,
                estimate,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                analysis_ci_low,
                ci_low,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
    checks = {
        "schema_matches": report.get("schema_version") == M6_LISTENING_SCHEMA_VERSION,
        "release_matches": report.get("release_sha256") == release_sha256,
        "evidence_tier_is_declared": tier in EVIDENCE_TIERS,
        "randomized_and_double_blind": bool(
            protocol.get("randomized") and protocol.get("double_blind")
        ),
        "common_loudness_policy": bool(protocol.get("common_loudness_master_gain")),
        "hidden_reference_and_anchor_present": bool(
            protocol.get("hidden_reference") and protocol.get("degraded_anchor")
        ),
        "room_disjoint_stimuli": bool(protocol.get("room_disjoint_stimuli")),
        "participant_count_is_declared": participants > 0,
        "empirical_claim_is_not_explicitly_nonhuman": bool(
            tier != "empirical" or not explicitly_nonhuman
        ),
        "empirical_artifacts_are_parsed_and_content_addressed": bool(
            tier != "empirical" or empirical_records_valid
        ),
        "empirical_participant_statistics_are_recomputed": bool(
            tier != "empirical" or empirical_statistics_valid
        ),
        "analysis_is_complete": bool(results.get("completed")),
        "noninferiority_is_recomputed": bool(
            margin is not None
            and margin >= 0.0
            and ci_low is not None
            and ci_low >= -margin
        ),
        "artifacts_are_content_addressed": bool(
            all(
                _sha256(artifacts.get(name))
                for name in ("assignment_sha256", "responses_sha256", "analysis_sha256")
            )
        ),
    }
    valid = bool(all(checks.values()))
    empirical = bool(valid and tier == "empirical" and participants >= 20)
    return {
        "valid": valid,
        "empirical": empirical,
        "checks": checks,
        "participant_count": participants,
        "evidence_tier": tier,
        "artifact_sha256": {
            name: artifacts.get(name)
            for name in (
                "assignment_sha256",
                "responses_sha256",
                "analysis_sha256",
            )
        },
    }


def _recipe_space_hash(
    release_root: Path,
    release: RIRBankReleaseManifest,
    recipe_id: str,
    split: str,
) -> str | None:
    recipe = next((item for item in release.recipes if item.recipe_id == recipe_id), None)
    if recipe is None or recipe.status != "ready" or recipe.split_indexes is None:
        return None
    path = release_root / recipe.split_indexes[split].path
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return canonical_json_sha256(
        sorted({str(row["acoustic_space_id"]) for row in rows})
    )


def validate_downstream_report(
    report: Mapping[str, Any] | None,
    *,
    release_root: str | Path,
    release: RIRBankReleaseManifest,
) -> dict[str, Any]:
    if report is None:
        return {
            "valid": False,
            "empirical": False,
            "checks": {"report_present": False},
            "reason": "room-disjoint downstream report is missing",
        }
    raw_protocol = report.get("protocol", {})
    raw_results = report.get("results", [])
    raw_artifacts = report.get("artifacts", {})
    protocol = raw_protocol if isinstance(raw_protocol, Mapping) else {}
    results = raw_results if isinstance(raw_results, list) else []
    artifacts = raw_artifacts if isinstance(raw_artifacts, Mapping) else {}
    tier = report.get("evidence_tier")
    recipe_id = str(protocol.get("recipe_id", ""))
    raw_seeds = protocol.get("seeds", ())
    parsed_seeds = (
        tuple(_integer(value) for value in raw_seeds)
        if isinstance(raw_seeds, Sequence) and not isinstance(raw_seeds, str)
        else ()
    )
    seeds = tuple(value for value in parsed_seeds if value is not None)
    seeds_well_formed = len(seeds) == len(parsed_seeds) and bool(parsed_seeds)
    expected_train_hash = _recipe_space_hash(
        Path(release_root), release, recipe_id, "train"
    )
    expected_test_hash = _recipe_space_hash(
        Path(release_root), release, recipe_id, "test"
    )
    result_consistency = True
    all_lower_bounds_positive = True
    recomputed_intervals: list[dict[str, float]] = []
    for result in results:
        if not isinstance(result, Mapping):
            result_consistency = False
            all_lower_bounds_positive = False
            continue
        try:
            baseline = np.asarray(
                result.get("baseline_by_seed", ()), dtype=np.float64
            )
            candidate = np.asarray(
                result.get("candidate_by_seed", ()), dtype=np.float64
            )
        except (TypeError, ValueError):
            result_consistency = False
            all_lower_bounds_positive = False
            continue
        if (
            baseline.size != len(seeds)
            or candidate.size != len(seeds)
            or not np.isfinite(baseline).all()
            or not np.isfinite(candidate).all()
        ):
            result_consistency = False
            continue
        direction = 1.0 if bool(result.get("higher_is_better")) else -1.0
        improvement = direction * (candidate - baseline)
        reported_mean = _finite(result.get("mean_improvement"))
        ci_low = _finite(result.get("confidence_interval_low"))
        interval = _paired_t_confidence_interval(improvement)
        result_consistency = bool(
            result_consistency
            and reported_mean is not None
            and math.isclose(
                reported_mean,
                float(np.mean(improvement)),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and ci_low is not None
            and interval is not None
            and math.isclose(
                ci_low,
                interval[0],
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        all_lower_bounds_positive = bool(
            all_lower_bounds_positive
            and interval is not None
            and interval[0] > 0.0
        )
        if interval is not None:
            recomputed_intervals.append(
                {
                    "confidence_interval_low": float(interval[0]),
                    "confidence_interval_high": float(interval[1]),
                }
            )
    checks = {
        "schema_matches": report.get("schema_version") == M6_DOWNSTREAM_SCHEMA_VERSION,
        "release_matches": report.get("release_sha256") == release.release_sha256,
        "evidence_tier_is_declared": tier in EVIDENCE_TIERS,
        "ready_recipe_is_named": expected_train_hash is not None and expected_test_hash is not None,
        "at_least_three_unique_seeds": seeds_well_formed and len(set(seeds)) >= 3,
        "train_and_test_space_hashes_match_release": bool(
            protocol.get("train_acoustic_spaces_sha256") == expected_train_hash
            and protocol.get("test_acoustic_spaces_sha256") == expected_test_hash
        ),
        "room_disjoint_evaluation_is_declared": bool(
            protocol.get("room_disjoint") and protocol.get("evaluation_split") == "test"
        ),
        "frozen_training_and_model_recipe": bool(
            protocol.get("frozen_training_recipe")
            and protocol.get("frozen_model_recipe")
        ),
        "at_least_one_task_result": len(results) > 0,
        "per_seed_results_are_consistent": result_consistency,
        "all_primary_confidence_bounds_improve": all_lower_bounds_positive,
        "artifacts_are_content_addressed": bool(
            all(
                _sha256(artifacts.get(name))
                for name in ("training_recipe_sha256", "checkpoints_sha256", "analysis_sha256")
            )
        ),
    }
    valid = bool(all(checks.values()))
    empirical = bool(valid and tier == "empirical")
    return {
        "valid": valid,
        "empirical": empirical,
        "checks": checks,
        "seed_count": len(set(seeds)),
        "task_count": len(results),
        "evidence_tier": tier,
        "artifact_sha256": {
            name: artifacts.get(name)
            for name in (
                "training_recipe_sha256",
                "checkpoints_sha256",
                "analysis_sha256",
            )
        },
        "recomputed_confidence_intervals": recomputed_intervals,
    }


def evaluate_m6_release(
    release_root: str | Path,
    *,
    throughput_report: str | Path | Mapping[str, Any] | None = None,
    listening_report: str | Path | Mapping[str, Any] | None = None,
    downstream_report: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an M6.4 release while keeping external evidence fail closed."""

    root = Path(release_root)
    release, _distributions = _release_and_distributions(root)
    release_audit = audit_m6_variant_release(root)
    acoustics = compare_release_distributions(root)
    throughput = validate_throughput_report(
        _load_json(throughput_report), release_sha256=str(release.release_sha256)
    )
    listening = validate_listening_report(
        _load_json(listening_report), release_sha256=str(release.release_sha256)
    )
    downstream = validate_downstream_report(
        _load_json(downstream_report), release_root=root, release=release
    )
    invariant = acoustics["variant_scale_invariance"]
    invariant_errors = invariant.get("maximum_absolute_error", {})
    scale_invariance_passed = bool(
        invariant.get("status") == "pass"
        and all(
            value is None or float(value) <= 1e-4
            for value in invariant_errors.values()
        )
    )
    implementation_checks = {
        "m6_4_release_audit_passed": release_audit["valid"],
        "distribution_comparison_is_content_addressed": bool(
            acoustics.get("comparison_sha256")
            == canonical_json_sha256(
                {
                    key: value
                    for key, value in acoustics.items()
                    if key != "comparison_sha256"
                }
            )
        ),
        "level_variant_acoustic_invariance_passed": scale_invariance_passed,
        "generation_throughput_and_failures_reported": throughput["valid"],
        "listening_contract_is_fail_closed": bool(
            listening["valid"] or listening.get("reason")
        ),
        "downstream_contract_is_fail_closed": bool(
            downstream["valid"] or downstream.get("reason")
        ),
    }
    measured_status = acoustics["synthetic_to_measured"]["status"]
    empirical_checks = {
        "measured_acoustic_distribution_compared": measured_status == "evaluated",
        "controlled_listening_empirical_passed": listening["empirical"],
        "room_disjoint_downstream_empirical_passed": downstream["empirical"],
    }
    implementation_passed = bool(all(implementation_checks.values()))
    empirical_passed = bool(all(empirical_checks.values()))
    report: dict[str, Any] = {
        "schema_version": M6_EVALUATION_SCHEMA_VERSION,
        "milestone": "M6.5",
        "release_id": release.release_id,
        "release_sha256": release.release_sha256,
        "release_audit": release_audit,
        "acoustic_distribution_evaluation": acoustics,
        "throughput_evaluation": throughput,
        "listening_evaluation": listening,
        "downstream_evaluation": downstream,
        "implementation_checks": implementation_checks,
        "empirical_checks": empirical_checks,
        "implementation_exit": {
            "passed": implementation_passed,
            "m6_5_evaluation_pipeline_complete": implementation_passed,
        },
        "empirical_exit": {
            "passed": empirical_passed,
            "missing_evidence": [
                name for name, passed in empirical_checks.items() if not passed
            ],
        },
        "production_enablement": {
            "ready": bool(implementation_passed and empirical_passed),
            "reason": (
                "production requires measured acoustic, controlled listening, "
                "and room-disjoint downstream empirical evidence"
            ),
        },
    }
    report["evaluation_sha256"] = canonical_json_sha256(report)
    return report


__all__ = [
    "M6_DOWNSTREAM_SCHEMA_VERSION",
    "M6_EVALUATION_SCHEMA_VERSION",
    "M6_LISTENING_SCHEMA_VERSION",
    "M6_THROUGHPUT_SCHEMA_VERSION",
    "compare_release_distributions",
    "evaluate_m6_release",
    "validate_downstream_report",
    "validate_listening_report",
    "validate_throughput_report",
]
