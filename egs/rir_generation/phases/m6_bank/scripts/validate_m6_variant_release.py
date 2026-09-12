#!/usr/bin/env python3
"""Validate M6.4 variant lineage, distributions, and release recipes."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_reproducible_generation import _run_generator
from puresound.audio.rir.bank.loader import PreGeneratedReleaseBank
from puresound.audio.rir.bank.schema import RIRBankManifest
from puresound.audio.rir.bank.qc import run_rir_bank_qc
from puresound.audio.rir.bank.release import (
    RIRBankReleaseManifest,
    audit_m6_variant_release,
    build_m6_variant_release,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m6/rir_m6_variant_release"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/reports/m6_variant_release_report.json"
)


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_distribution(release_root: Path, variant_id: str) -> dict[str, Any]:
    release = RIRBankReleaseManifest.from_json(
        (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    variant = next(item for item in release.variants if item.variant_id == variant_id)
    return json.loads(
        (release_root / variant.distribution_path).read_text(encoding="utf-8")
    )


def _scale_invariance_error(release_root: Path) -> dict[str, float]:
    calibrated = _load_distribution(release_root, "synthetic_calibrated")
    normalized = _load_distribution(release_root, "synthetic_peak_normalized")
    calibrated_rows = {
        (row["item_id"], row["channel"]): row for row in calibrated["channels"]
    }
    normalized_rows = {
        (
            str(row["item_id"]).removesuffix("__peak_normalized"),
            row["channel"],
        ): row
        for row in normalized["channels"]
    }
    result: dict[str, float] = {}
    for metric in ("drr_db", "c50_db", "c80_db", "t20_s"):
        errors = [
            abs(float(row[metric]) - float(normalized_rows[key][metric]))
            for key, row in calibrated_rows.items()
            if row[metric] is not None and normalized_rows[key][metric] is not None
        ]
        result[metric] = max(errors, default=0.0)
    return result


def _normalized_peaks(release_root: Path) -> list[float]:
    release = RIRBankReleaseManifest.from_json(
        (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    variant = next(
        item for item in release.variants if item.variant_id == "synthetic_peak_normalized"
    )
    bank_root = release_root / variant.bank_path
    manifest = RIRBankManifest.from_json(
        (release_root / variant.manifest_path).read_text(encoding="utf-8")
    )
    peaks = []
    for item in manifest.items:
        rir, _ = sf.read(bank_root / item.rir_path, always_2d=True, dtype="float64")
        peaks.append(float(np.max(np.abs(rir))))
    return peaks


def build_report(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    raw_root = output_root / "source_bank"
    release_a_root = output_root / "release_a"
    release_b_root = output_root / "release_b"
    tampered_root = output_root / "tampered_recipe"
    _reset(output_root)
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    generated = _run_generator(raw_root, workers=1)
    generation_elapsed_s = time.perf_counter() - started
    run_rir_bank_qc(raw_root)
    release_a = build_m6_variant_release(
        raw_root,
        release_a_root,
        release_id="puresound-m6.4-validation-candidate",
    )
    release_b = build_m6_variant_release(
        raw_root,
        release_b_root,
        release_id="puresound-m6.4-validation-candidate",
    )
    audit_a = audit_m6_variant_release(release_a_root)
    audit_b = audit_m6_variant_release(release_b_root)
    errors = _scale_invariance_error(release_a_root)
    peaks = _normalized_peaks(release_a_root)
    recipes = {recipe.recipe_id: recipe for recipe in release_a.recipes}
    variants = {variant.variant_id: variant for variant in release_a.variants}
    recipe_bank = PreGeneratedReleaseBank(
        str(release_a_root),
        recipe_id="synthetic_calibrated",
        split="train",
    )
    recipe_scene = recipe_bank.sample_scene()
    recipe_rir, recipe_metadata, recipe_sample_rate = recipe_bank.select_channel(
        recipe_scene,
        source_role="foreground",
    )

    shutil.copytree(release_a_root, tampered_root)
    recipe_index = recipes["synthetic_calibrated"].split_indexes["train"]
    with (tampered_root / recipe_index.path).open("ab") as handle:
        handle.write(b"m6.4-recipe-index-tamper")
    tampered_audit = audit_m6_variant_release(tampered_root)

    checks = {
        "source_is_actual_qc_passed_m6_generator_bank": bool(
            generated["audit"]["ready_for_m6_bank_generation"]
            and all(
                item.qc_status == "pass"
                for item in RIRBankManifest.from_json(
                    (raw_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
                ).items
            )
        ),
        "calibrated_and_peak_normalized_variants_audit": bool(
            audit_a["valid"] and audit_b["valid"]
        ),
        "variant_release_is_byte_deterministic": bool(
            release_a.release_sha256 == release_b.release_sha256
            and {
                variant.variant_id: variant.distribution_file_sha256
                for variant in release_a.variants
            }
            == {
                variant.variant_id: variant.distribution_file_sha256
                for variant in release_b.variants
            }
        ),
        "peak_normalized_variant_hits_one_common_gain_target": bool(
            peaks and max(abs(value - 0.98) for value in peaks) <= 2e-7
        ),
        "level_invariant_acoustic_metrics_are_preserved": bool(
            errors["drr_db"] <= 1e-4
            and errors["c50_db"] <= 1e-4
            and errors["c80_db"] <= 1e-4
            and errors["t20_s"] <= 1e-4
        ),
        "variants_share_acoustic_spaces_and_splits": bool(
            audit_a["checks"][
                "variant_lineage_preserves_identity_parent_hash_and_transform"
            ]
            and audit_a["checks"][
                "acoustic_spaces_remain_split_disjoint_across_variants"
            ]
        ),
        "synthetic_recipes_have_content_addressed_three_split_indexes": bool(
            all(
                recipes[recipe_id].status == "ready"
                and all(
                    recipes[recipe_id].split_indexes[split].item_count > 0
                    for split in ("train", "validation", "test")
                )
                for recipe_id in (
                    "synthetic_calibrated",
                    "synthetic_peak_normalized",
                )
            )
        ),
        "ready_recipe_is_consumable_with_release_provenance": bool(
            len(recipe_bank)
            == recipes["synthetic_calibrated"].split_indexes["train"].item_count
            and recipe_rir.ndim == 2
            and recipe_rir.shape[0] == 1
            and recipe_sample_rate > 0
            and recipe_metadata["split"] == "train"
            and recipe_metadata["release_sha256"] == release_a.release_sha256
            and recipe_metadata["release_recipe_id"] == "synthetic_calibrated"
            and recipe_metadata["release_variant_id"] == "synthetic_calibrated"
            and recipe_metadata["release_origin"] == "synthetic"
        ),
        "missing_measured_data_blocks_real_and_mixed_recipes": bool(
            recipes["real_native"].status == "blocked"
            and recipes["mixed_calibrated_real"].status == "blocked"
            and not recipes["real_native"].split_indexes
            and not recipes["mixed_calibrated_real"].split_indexes
        ),
        "distribution_snapshots_are_content_addressed": bool(
            all(variant.distribution_file_sha256 for variant in variants.values())
            and audit_a["checks"]["all_variant_assets_and_distributions_audit"]
        ),
        "recipe_index_tamper_fails_release_audit": bool(
            not tampered_audit["valid"]
            and not tampered_audit["recipe_checks"]["synthetic_calibrated"]
        ),
        "candidate_does_not_claim_production": bool(
            release_a.release_status == "candidate"
            and not audit_a["production_ready"]
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "schema_version": "puresound.m6_variant_release_validation.v1",
        "milestone": "M6.4",
        "scope": "distribution_variant_lineage_and_recipe_release",
        "fixture": {
            "source": "actual M6.2 generator plus M6.3 QC",
            "explicitly_not_measured_or_production_evidence": True,
            "generation_elapsed_s": generation_elapsed_s,
            "generated_items": 6,
            "generation_items_per_second": 6.0 / generation_elapsed_s,
        },
        "release": {
            "root": str(release_a_root),
            "release_sha256": release_a.release_sha256,
            "variant_ids": sorted(variants),
            "recipe_statuses": {
                recipe_id: recipe.status for recipe_id, recipe in recipes.items()
            },
            "audit": audit_a,
        },
        "scale_invariance_max_abs_error": errors,
        "normalized_peak_abs": {
            "minimum": min(peaks),
            "maximum": max(peaks),
        },
        "checks": checks,
        "exit": {
            "passed": passed,
            "m6_4_distribution_and_variant_release_complete": passed,
            "measured_and_mixed_recipes_complete": False,
            "production_bank_complete": False,
            "next_milestone": "M6.5 bank-level and downstream evaluation",
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
    print(f"# M6.4 variant release: {'PASS' if report['exit']['passed'] else 'FAIL'}")
    print("# measured/mixed/production release: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
