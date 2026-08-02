#!/usr/bin/env python3
"""Validate the M6.5 evaluation pipeline and fail-closed evidence contracts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_variant_release
from puresound.audio.rir_bank_evaluation import (
    M6_DOWNSTREAM_SCHEMA_VERSION,
    M6_LISTENING_SCHEMA_VERSION,
    M6_THROUGHPUT_SCHEMA_VERSION,
    evaluate_m6_release,
    _paired_t_confidence_interval,
    validate_downstream_report,
    validate_listening_report,
)
from puresound.audio.rir_bank_manifest import canonical_json_sha256
from puresound.audio.rir_bank_release import RIRBankReleaseManifest


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_evaluation"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_bank_evaluation_report.json"
)


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _recipe_space_hash(
    release_root: Path,
    release: RIRBankReleaseManifest,
    recipe_id: str,
    split: str,
) -> str:
    recipe = next(item for item in release.recipes if item.recipe_id == recipe_id)
    index = recipe.split_indexes[split]
    rows = [
        json.loads(line)
        for line in (release_root / index.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return canonical_json_sha256(
        sorted({str(row["acoustic_space_id"]) for row in rows})
    )


def _fixture_reports(
    output_root: Path,
    release_root: Path,
    release: RIRBankReleaseManifest,
    generation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    elapsed = float(generation["fixture"]["generation_elapsed_s"])
    throughput = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": release.release_sha256,
        "task_count": 6,
        "items_generated": 6,
        "items_skipped": 0,
        "items_failed": 0,
        "elapsed_seconds": elapsed,
        "items_per_second": 6.0 / elapsed,
        "num_workers": 1,
        "measurement_scope": "actual M6.4 validator generator subprocess wall time",
    }
    listening = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": release.release_sha256,
        "evidence_tier": "contract_fixture",
        "protocol": {
            "randomized": True,
            "double_blind": True,
            "common_loudness_master_gain": True,
            "hidden_reference": True,
            "degraded_anchor": True,
            "room_disjoint_stimuli": True,
            "participant_count": 24,
            "trial_count": 240,
            "explicitly_not_human_responses": True,
        },
        "results": {
            "completed": True,
            "primary_endpoint": "synthetic_contract_fixture_preference_score",
            "noninferiority_margin": 0.10,
            "estimate": 0.04,
            "confidence_interval_low": 0.01,
            "confidence_interval_high": 0.07,
        },
        "artifacts": {
            "assignment_sha256": "1" * 64,
            "responses_sha256": "2" * 64,
            "analysis_sha256": "3" * 64,
        },
    }
    seeds = (101, 202, 303)
    baseline = (0.800, 0.810, 0.790)
    candidate = (0.825, 0.832, 0.818)
    mean_improvement = sum(c - b for b, c in zip(baseline, candidate)) / len(seeds)
    confidence_interval = _paired_t_confidence_interval(
        [c - b for b, c in zip(baseline, candidate)]
    )
    assert confidence_interval is not None
    downstream = {
        "schema_version": M6_DOWNSTREAM_SCHEMA_VERSION,
        "release_sha256": release.release_sha256,
        "evidence_tier": "contract_fixture",
        "protocol": {
            "recipe_id": "synthetic_calibrated",
            "seeds": list(seeds),
            "evaluation_split": "test",
            "room_disjoint": True,
            "frozen_training_recipe": True,
            "frozen_model_recipe": True,
            "train_acoustic_spaces_sha256": _recipe_space_hash(
                release_root, release, "synthetic_calibrated", "train"
            ),
            "test_acoustic_spaces_sha256": _recipe_space_hash(
                release_root, release, "synthetic_calibrated", "test"
            ),
            "explicitly_not_a_trained_speech_model_result": True,
        },
        "results": [
            {
                "task_id": "contract_fixture_speech_task",
                "metric": "bounded_fixture_score",
                "higher_is_better": True,
                "baseline_by_seed": list(baseline),
                "candidate_by_seed": list(candidate),
                "mean_improvement": mean_improvement,
                "confidence_interval_low": confidence_interval[0],
                "confidence_interval_high": confidence_interval[1],
            }
        ],
        "artifacts": {
            "training_recipe_sha256": "4" * 64,
            "checkpoints_sha256": "5" * 64,
            "analysis_sha256": "6" * 64,
        },
    }
    for name, value in (
        ("throughput_fixture.json", throughput),
        ("listening_contract_fixture.json", listening),
        ("downstream_contract_fixture.json", downstream),
    ):
        (output_root / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return throughput, listening, downstream


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    _reset(output_root)
    output_root.mkdir(parents=True)
    m6_4 = validate_m6_variant_release.build_report(output_root / "m6_4")
    release_root = output_root / "m6_4" / "release_a"
    release = RIRBankReleaseManifest.from_json(
        (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    throughput, listening, downstream = _fixture_reports(
        output_root, release_root, release, m6_4
    )
    evaluation = evaluate_m6_release(
        release_root,
        throughput_report=throughput,
        listening_report=listening,
        downstream_report=downstream,
    )

    unblinded = json.loads(json.dumps(listening))
    unblinded["protocol"]["double_blind"] = False
    unblinded_result = validate_listening_report(
        unblinded, release_sha256=str(release.release_sha256)
    )
    leaked = json.loads(json.dumps(downstream))
    leaked["protocol"]["test_acoustic_spaces_sha256"] = leaked["protocol"][
        "train_acoustic_spaces_sha256"
    ]
    leaked_result = validate_downstream_report(
        leaked, release_root=release_root, release=release
    )
    one_seed = json.loads(json.dumps(downstream))
    one_seed["protocol"]["seeds"] = [101]
    one_seed["results"][0]["baseline_by_seed"] = [0.8]
    one_seed["results"][0]["candidate_by_seed"] = [0.82]
    one_seed["results"][0]["mean_improvement"] = 0.02
    one_seed_result = validate_downstream_report(
        one_seed, release_root=release_root, release=release
    )
    forged_ci = json.loads(json.dumps(downstream))
    forged_ci["results"][0]["candidate_by_seed"] = [10.8, -9.18, 0.81]
    forged_improvements = [10.0, -9.99, 0.02]
    forged_ci["results"][0]["mean_improvement"] = sum(
        forged_improvements
    ) / len(forged_improvements)
    forged_ci["results"][0]["confidence_interval_low"] = 0.001
    forged_ci_result = validate_downstream_report(
        forged_ci,
        release_root=release_root,
        release=release,
    )
    nonhuman_empirical = json.loads(json.dumps(listening))
    nonhuman_empirical["evidence_tier"] = "empirical"
    nonhuman_empirical_result = validate_listening_report(
        nonhuman_empirical,
        release_sha256=str(release.release_sha256),
    )
    unhashed_evaluation = dict(evaluation)
    stored_hash = unhashed_evaluation.pop("evaluation_sha256")
    checks = {
        "m6_4_release_dependency_passed": m6_4["exit"]["passed"],
        "bank_level_distribution_and_scale_invariance_passed": bool(
            evaluation["implementation_checks"][
                "distribution_comparison_is_content_addressed"
            ]
            and evaluation["implementation_checks"][
                "level_variant_acoustic_invariance_passed"
            ]
        ),
        "actual_generation_throughput_and_failure_counts_validate": bool(
            evaluation["throughput_evaluation"]["valid"]
            and evaluation["throughput_evaluation"]["summary"]["items_failed"] == 0
            and math.isfinite(
                evaluation["throughput_evaluation"]["summary"]["items_per_second"]
            )
        ),
        "missing_measured_reference_is_not_fabricated": bool(
            evaluation["acoustic_distribution_evaluation"][
                "synthetic_to_measured"
            ]["status"]
            == "not_evaluable"
        ),
        "listening_contract_fixture_valid_but_not_empirical": bool(
            evaluation["listening_evaluation"]["valid"]
            and not evaluation["listening_evaluation"]["empirical"]
        ),
        "unblinded_listening_report_is_rejected": bool(
            not unblinded_result["valid"]
            and not unblinded_result["checks"]["randomized_and_double_blind"]
        ),
        "downstream_contract_fixture_valid_but_not_empirical": bool(
            evaluation["downstream_evaluation"]["valid"]
            and not evaluation["downstream_evaluation"]["empirical"]
        ),
        "downstream_split_identity_tamper_is_rejected": bool(
            not leaked_result["valid"]
            and not leaked_result["checks"][
                "train_and_test_space_hashes_match_release"
            ]
        ),
        "single_seed_downstream_claim_is_rejected": bool(
            not one_seed_result["valid"]
            and not one_seed_result["checks"]["at_least_three_unique_seeds"]
        ),
        "forged_positive_downstream_ci_is_rejected": bool(
            not forged_ci_result["valid"]
            and not forged_ci_result["checks"][
                "per_seed_results_are_consistent"
            ]
        ),
        "explicitly_nonhuman_empirical_listening_is_rejected": bool(
            not nonhuman_empirical_result["valid"]
            and not nonhuman_empirical_result["checks"][
                "empirical_claim_is_not_explicitly_nonhuman"
            ]
        ),
        "evaluation_report_is_content_addressed": bool(
            stored_hash == canonical_json_sha256(unhashed_evaluation)
        ),
        "m6_5_implementation_passes_while_empirical_stays_open": bool(
            evaluation["implementation_exit"]["passed"]
            and not evaluation["empirical_exit"]["passed"]
        ),
        "production_enablement_remains_fail_closed": bool(
            not evaluation["production_enablement"]["ready"]
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "schema_version": "puresound.m6_bank_evaluation_validation.v1",
        "milestone": "M6.5",
        "scope": "bank_acoustics_throughput_listening_and_downstream_contracts",
        "fixture": {
            "release_root": str(release_root),
            "release_sha256": release.release_sha256,
            "listening_and_downstream_are_contract_fixtures": True,
            "explicitly_not_measured_human_or_trained_model_evidence": True,
        },
        "evaluation": evaluation,
        "negative_controls": {
            "unblinded_listening": unblinded_result,
            "downstream_split_tamper": leaked_result,
            "single_seed_downstream": one_seed_result,
            "forged_positive_downstream_ci": forged_ci_result,
            "nonhuman_empirical_listening": nonhuman_empirical_result,
        },
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_5_bank_evaluation_implementation_complete": passed,
            "m6_5_empirical_exit_complete": False,
            "production_bank_complete": False,
            "next_milestone": "M6.6 evidence-backed production decision",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(args.output_root)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, passed in report["checks"].items():
        print(f"{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M6.5 bank evaluation implementation: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M6.5 empirical/production exit: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
