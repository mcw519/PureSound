#!/usr/bin/env python3
"""Validate M6.6 immutable promotion decisions and false-production controls."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_bank_evaluation
from puresound.audio.rir.bank.loader import PreGeneratedReleaseBank
from puresound.audio.rir.bank.schema import canonical_json_sha256
from puresound.audio.rir.bank.production import (
    DEFAULT_PRODUCTION_DECISION_NAME,
    M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
    audit_m6_production_evidence,
    build_m6_production_decision,
    validate_m6_production_certificate,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_production_decision"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_production_decision_report.json"
)


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    _reset(output_root)
    output_root.mkdir(parents=True)
    m6_5 = validate_m6_bank_evaluation.build_report(output_root / "m6_5")
    release_root = output_root / "m6_5" / "m6_4" / "release_a"
    evaluation = m6_5["evaluation"]
    evaluation_path = output_root / "m6_bank_evaluation.json"
    _write_json(evaluation_path, evaluation)

    first = build_m6_production_decision(release_root, evaluation_path)
    second = build_m6_production_decision(release_root, evaluation_path)
    certificate_path = release_root / DEFAULT_PRODUCTION_DECISION_NAME
    _write_json(certificate_path, first)
    certificate = validate_m6_production_certificate(
        certificate_path,
        release_root=release_root,
    )
    required_certificate = validate_m6_production_certificate(
        certificate_path,
        release_root=release_root,
        require_approved=True,
    )

    ordinary_bank = PreGeneratedReleaseBank(
        str(release_root),
        recipe_id="synthetic_calibrated",
        split="train",
    )
    ordinary_scene = ordinary_bank.sample_scene()
    ordinary_rir, ordinary_metadata, ordinary_sr = ordinary_bank.select_channel(
        ordinary_scene,
        source_role="foreground",
    )
    production_reader_rejected = False
    try:
        PreGeneratedReleaseBank(
            str(release_root),
            recipe_id="synthetic_calibrated",
            split="train",
            require_production=True,
        )
    except ValueError:
        production_reader_rejected = True

    forged = json.loads(json.dumps(first))
    forged["production_ready"] = True
    forged["decision"] = "approved"
    forged_certificate = validate_m6_production_certificate(
        forged,
        release_root=release_root,
    )
    recomputed_hash_forgery = json.loads(json.dumps(first))
    recomputed_hash_forgery["checks"] = {
        name: True for name in recomputed_hash_forgery["checks"]
    }
    recomputed_hash_forgery["blockers"] = []
    recomputed_hash_forgery["production_ready"] = True
    recomputed_hash_forgery["decision"] = "approved"
    recomputed_hash_forgery.pop("decision_sha256")
    recomputed_hash_forgery["decision_sha256"] = canonical_json_sha256(
        recomputed_hash_forgery
    )
    recomputed_hash_forgery_result = validate_m6_production_certificate(
        recomputed_hash_forgery,
        release_root=release_root,
    )

    tampered_evaluation = json.loads(json.dumps(evaluation))
    tampered_evaluation["empirical_exit"]["passed"] = True
    tampered_evaluation["production_enablement"]["ready"] = True
    tampered_decision = build_m6_production_decision(
        release_root,
        tampered_evaluation,
    )
    malformed_decision = build_m6_production_decision(
        release_root,
        {
            "schema_version": "wrong",
            "implementation_exit": [],
            "empirical_exit": "invalid",
            "production_enablement": None,
        },
    )

    unsafe_bundle_payload = {
        "schema_version": M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
        "release_sha256": first["release_sha256"],
        "evaluation_sha256": first["evaluation_sha256"],
        "artifacts": [
            {"kind": "listening_responses", "path": "../escape.json", "sha256": "0" * 64}
        ],
    }
    unsafe_bundle = {
        **unsafe_bundle_payload,
        "bundle_sha256": canonical_json_sha256(unsafe_bundle_payload),
    }
    unsafe_audit = audit_m6_production_evidence(
        unsafe_bundle,
        evidence_root=output_root,
        release_sha256=str(first["release_sha256"]),
        evaluation=evaluation,
        renderer_approval_sha256=(),
    )

    false_production_root = output_root / "false_production_release"
    shutil.copytree(release_root, false_production_root)
    release_path = false_production_root / "rir_bank_release.json"
    false_release = json.loads(release_path.read_text(encoding="utf-8"))
    false_release["release_status"] = "production"
    _write_json(release_path, false_release)
    false_production = build_m6_production_decision(
        false_production_root,
        evaluation_path,
    )

    checks = {
        "m6_5_implementation_dependency_passed": bool(m6_5["exit"]["passed"]),
        "decision_is_deterministic_and_content_addressed": bool(
            first == second
            and first["decision_sha256"]
            == canonical_json_sha256(
                {
                    key: value
                    for key, value in first.items()
                    if key != "decision_sha256"
                }
            )
        ),
        "candidate_release_and_qc_integrity_pass": bool(
            first["checks"]["candidate_release_audit_passed"]
            and first["checks"]["source_release_is_immutable_candidate"]
            and first["checks"]["all_variant_items_are_qc_passed"]
            and first["checks"]["all_generator_revisions_are_pinned"]
        ),
        "missing_real_and_mixed_recipes_block_promotion": bool(
            not first["checks"]["all_required_recipes_are_ready"]
            and not first["checks"]["real_and_mixed_recipe_semantics_are_valid"]
        ),
        "development_renderer_profiles_block_promotion": bool(
            not first["checks"]["all_renderer_profiles_are_production_approved"]
        ),
        "open_m6_5_empirical_exit_blocks_promotion": bool(
            not first["checks"]["m6_5_empirical_exit_passed"]
            and not first["checks"]["m6_5_production_enablement_passed"]
        ),
        "missing_external_evidence_bundle_blocks_promotion": bool(
            not first["checks"]["external_evidence_bundle_audits"]
            and not first["evidence_audit"]["checks"]["bundle_present"]
        ),
        "unsafe_or_unverifiable_evidence_assets_are_rejected": bool(
            not unsafe_audit["valid"]
            and not unsafe_audit["checks"]["artifact_paths_and_hashes_match"]
        ),
        "evaluation_tamper_is_rejected_even_if_pass_flags_are_flipped": bool(
            not tampered_decision["checks"]["evaluation_schema_and_content_hash_match"]
            and not tampered_decision["production_ready"]
            and not malformed_decision["checks"]["m6_5_implementation_exit_passed"]
            and not malformed_decision["production_ready"]
        ),
        "forged_approved_certificate_is_rejected": bool(
            not forged_certificate["valid"] and not forged_certificate["approved"]
        ),
        "rehashed_all_true_certificate_is_rejected": bool(
            not recomputed_hash_forgery_result["valid"]
            and not recomputed_hash_forgery_result["approved"]
        ),
        "blocked_certificate_is_valid_but_not_approved": bool(
            certificate["valid"]
            and not certificate["approved"]
            and not required_certificate["valid"]
        ),
        "candidate_reader_works_but_production_reader_fails_closed": bool(
            len(ordinary_bank) > 0
            and ordinary_rir.ndim == 2
            and ordinary_sr > 0
            and ordinary_metadata["production_certificate_sha256"] is None
            and production_reader_rejected
        ),
        "editing_candidate_status_cannot_create_production": bool(
            not false_production["checks"]["candidate_release_audit_passed"]
            and not false_production["checks"]["source_release_is_immutable_candidate"]
            and not false_production["production_ready"]
        ),
        "current_release_remains_blocked_and_unmodified": bool(
            first["decision"] == "blocked"
            and not first["production_ready"]
            and first["promotion_model"]
            == "immutable_candidate_plus_content_addressed_certificate"
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "schema_version": "puresound.m6_production_decision_validation.v1",
        "milestone": "M6.6",
        "scope": "immutable_fail_closed_production_promotion_certificate",
        "fixture": {
            "release_root": str(release_root),
            "evaluation_path": str(evaluation_path),
            "explicitly_not_production_evidence": True,
        },
        "decision": first,
        "negative_controls": {
            "unsafe_evidence_bundle": unsafe_audit,
            "evaluation_hash_tamper": tampered_decision,
            "malformed_evaluation": malformed_decision,
            "forged_certificate": forged_certificate,
            "rehashed_all_true_certificate": recomputed_hash_forgery_result,
            "false_production_release": false_production,
        },
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_6_production_decision_implementation_complete": passed,
            "production_promotion_complete": False,
            "next_action": (
                "supply measured/mixed variants, production renderer approval, "
                "empirical M6.5 evidence, hashed artifacts, and three role signoffs"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(args.output_root)
    _write_json(args.output_report, report)
    for name, passed in report["checks"].items():
        print(f"{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M6.6 production decision implementation: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M6 production promotion: BLOCKED")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
