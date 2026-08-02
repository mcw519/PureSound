"""M6.6 fail-closed production promotion certificates for immutable releases."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from puresound.audio.rir_bank_evaluation import M6_EVALUATION_SCHEMA_VERSION
from puresound.audio.rir_bank_manifest import (
    RIRBankManifest,
    canonical_json_sha256,
    sha256_file,
)
from puresound.audio.rir_bank_release import (
    DEFAULT_RELEASE_MANIFEST_NAME,
    RIRBankReleaseManifest,
    audit_m6_variant_release,
)


M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION = "puresound.m6_production_evidence.v1"
M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION = "puresound.m6_production_signoff.v1"
M6_PRODUCTION_DECISION_SCHEMA_VERSION = "puresound.m6_production_decision.v1"
DEFAULT_PRODUCTION_DECISION_NAME = "rir_bank_production_decision.json"

REQUIRED_RECIPE_IDS = (
    "synthetic_calibrated",
    "synthetic_peak_normalized",
    "real_native",
    "mixed_calibrated_real",
)
REQUIRED_EVIDENCE_KINDS = (
    "listening_assignment",
    "listening_responses",
    "listening_analysis",
    "downstream_training_recipe",
    "downstream_checkpoints",
    "downstream_analysis",
    "acoustics_signoff",
    "ml_signoff",
    "release_signoff",
)
SIGNOFF_KIND_TO_ROLE = {
    "acoustics_signoff": "acoustics",
    "ml_signoff": "ml",
    "release_signoff": "release",
}
EVALUATION_ARTIFACT_LINKS = {
    "listening_assignment": ("listening_evaluation", "assignment_sha256"),
    "listening_responses": ("listening_evaluation", "responses_sha256"),
    "listening_analysis": ("listening_evaluation", "analysis_sha256"),
    "downstream_training_recipe": (
        "downstream_evaluation",
        "training_recipe_sha256",
    ),
    "downstream_checkpoints": (
        "downstream_evaluation",
        "checkpoints_sha256",
    ),
    "downstream_analysis": ("downstream_evaluation", "analysis_sha256"),
}
PRODUCTION_DECISION_CHECK_NAMES = (
    "candidate_release_audit_passed",
    "source_release_is_immutable_candidate",
    "all_required_recipes_are_ready",
    "real_and_mixed_recipe_semantics_are_valid",
    "all_variant_items_are_qc_passed",
    "all_generator_revisions_are_pinned",
    "all_renderer_profiles_are_production_approved",
    "evaluation_schema_and_content_hash_match",
    "evaluation_targets_this_release",
    "m6_5_implementation_exit_passed",
    "m6_5_empirical_exit_passed",
    "m6_5_production_enablement_passed",
    "external_evidence_bundle_audits",
)


def _load_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def _safe_relative_path(value: Any) -> str | None:
    path = PurePosixPath(str(value or ""))
    if not value or path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _evaluation_payload(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    report = _load_mapping(value)
    nested = report.get("evaluation")
    if isinstance(nested, Mapping):
        report = dict(nested)
    return report


def _evaluation_hash_valid(evaluation: Mapping[str, Any]) -> bool:
    claimed = evaluation.get("evaluation_sha256")
    payload = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    return bool(claimed and claimed == canonical_json_sha256(payload))


def _nested_bool(value: Mapping[str, Any], section: str, name: str) -> bool:
    nested = value.get(section)
    return bool(isinstance(nested, Mapping) and nested.get(name) is True)


def _variant_manifests(
    release_root: Path,
    release: RIRBankReleaseManifest,
) -> list[RIRBankManifest]:
    return [
        RIRBankManifest.from_json(
            (release_root / variant.manifest_path).read_text(encoding="utf-8")
        )
        for variant in release.variants
    ]


def _recipe_semantics_valid(release: RIRBankReleaseManifest) -> bool:
    recipes = {recipe.recipe_id: recipe for recipe in release.recipes}
    variants = {variant.variant_id: variant for variant in release.variants}
    try:
        real = recipes["real_native"]
        mixed = recipes["mixed_calibrated_real"]
        real_origins = {variants[value].origin for value in real.variant_ids}
        mixed_origins = {variants[value].origin for value in mixed.variant_ids}
    except KeyError:
        return False
    return bool(
        real.status == "ready"
        and real_origins == {"real"}
        and real.origin_weights == {"real": 1.0}
        and mixed.status == "ready"
        and mixed_origins == {"synthetic", "real"}
        and set(mixed.origin_weights) == {"synthetic", "real"}
        and all(value > 0.0 for value in mixed.origin_weights.values())
    )


def _artifact_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = bundle.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return []
    return [dict(value) for value in artifacts if isinstance(value, Mapping)]


def _signoff_valid(
    path: Path,
    *,
    role: str,
    release_sha256: str,
    evaluation_sha256: str,
) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(record, Mapping)
        and record.get("schema_version") == M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION
        and record.get("role") == role
        and str(record.get("reviewer_id", "")).strip()
        and record.get("decision") == "approve"
        and record.get("release_sha256") == release_sha256
        and record.get("evaluation_sha256") == evaluation_sha256
        and str(record.get("reviewed_scope", "")).strip()
    )


def audit_m6_production_evidence(
    bundle: str | Path | Mapping[str, Any] | None,
    *,
    evidence_root: str | Path,
    release_sha256: str,
    evaluation: Mapping[str, Any],
    renderer_approval_sha256: Sequence[str],
) -> dict[str, Any]:
    """Audit external evidence files and role sign-offs against M6.5 claims."""

    if bundle is None:
        checks = {
            "bundle_present": False,
            "bundle_schema_matches": False,
            "bundle_content_hash_matches": False,
            "bundle_release_and_evaluation_match": False,
            "artifact_paths_and_hashes_match": False,
            "required_evidence_kinds_present": False,
            "evaluation_artifact_hashes_match": False,
            "renderer_approval_files_match_profiles": False,
            "all_three_role_signoffs_approve": False,
        }
        return {"valid": False, "checks": checks, "artifact_count": 0}

    try:
        parsed = _load_mapping(bundle)
    except (OSError, ValueError, json.JSONDecodeError):
        parsed = {}
    root = Path(evidence_root)
    rows = _artifact_rows(parsed)
    evaluation_sha256 = str(evaluation.get("evaluation_sha256", ""))
    payload = {
        key: value for key, value in parsed.items() if key != "bundle_sha256"
    }
    paths: list[str] = []
    hashes_by_kind: dict[str, list[str]] = {}
    assets_valid = len(rows) == len(parsed.get("artifacts", ())) if isinstance(
        parsed.get("artifacts"), list
    ) else False
    for row in rows:
        kind = str(row.get("kind", ""))
        relative = _safe_relative_path(row.get("path"))
        claimed_hash = str(row.get("sha256", ""))
        if not kind or relative is None:
            assets_valid = False
            continue
        path = root / relative
        try:
            matches = sha256_file(path) == claimed_hash
        except OSError:
            matches = False
        assets_valid = bool(assets_valid and matches)
        paths.append(relative)
        hashes_by_kind.setdefault(kind, []).append(claimed_hash)
    assets_valid = bool(assets_valid and len(paths) == len(set(paths)))

    required_kinds = all(kind in hashes_by_kind for kind in REQUIRED_EVIDENCE_KINDS)
    evaluation_links = True
    for kind, (section, name) in EVALUATION_ARTIFACT_LINKS.items():
        expected = evaluation.get(section, {})
        artifacts = expected.get("artifact_sha256", {}) if isinstance(expected, Mapping) else {}
        expected_hash = artifacts.get(name) if isinstance(artifacts, Mapping) else None
        evaluation_links = bool(
            evaluation_links and expected_hash in hashes_by_kind.get(kind, ())
        )

    expected_renderer_hashes = set(renderer_approval_sha256)
    actual_renderer_hashes = set(hashes_by_kind.get("renderer_approval", ()))
    renderer_links = bool(
        expected_renderer_hashes
        and expected_renderer_hashes.issubset(actual_renderer_hashes)
    )

    signoffs_valid = True
    for kind, role in SIGNOFF_KIND_TO_ROLE.items():
        candidates = [row for row in rows if row.get("kind") == kind]
        signoffs_valid = bool(signoffs_valid and len(candidates) == 1)
        if len(candidates) == 1:
            relative = _safe_relative_path(candidates[0].get("path"))
            signoffs_valid = bool(
                signoffs_valid
                and relative is not None
                and _signoff_valid(
                    root / relative,
                    role=role,
                    release_sha256=release_sha256,
                    evaluation_sha256=evaluation_sha256,
                )
            )

    checks = {
        "bundle_present": True,
        "bundle_schema_matches": (
            parsed.get("schema_version") == M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION
        ),
        "bundle_content_hash_matches": bool(
            parsed.get("bundle_sha256")
            and parsed.get("bundle_sha256") == canonical_json_sha256(payload)
        ),
        "bundle_release_and_evaluation_match": bool(
            parsed.get("release_sha256") == release_sha256
            and parsed.get("evaluation_sha256") == evaluation_sha256
        ),
        "artifact_paths_and_hashes_match": assets_valid,
        "required_evidence_kinds_present": required_kinds,
        "evaluation_artifact_hashes_match": evaluation_links,
        "renderer_approval_files_match_profiles": renderer_links,
        "all_three_role_signoffs_approve": signoffs_valid,
    }
    return {
        "valid": bool(all(checks.values())),
        "checks": checks,
        "artifact_count": len(rows),
        "artifact_kinds": sorted(hashes_by_kind),
        "bundle_sha256": parsed.get("bundle_sha256"),
    }


def _production_decision_components(
    root: Path,
    release: RIRBankReleaseManifest,
    evaluation: Mapping[str, Any],
    *,
    evidence_bundle: Mapping[str, Any] | None,
    evidence_root: Path,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    release_audit = audit_m6_variant_release(root)
    manifests = _variant_manifests(root, release)
    profiles = [
        profile
        for manifest in manifests
        for profile in manifest.renderer_profiles
    ]
    renderer_hashes = sorted(
        {
            str(profile.approval_report_sha256)
            for profile in profiles
            if profile.approval_report_sha256 is not None
        }
    )
    evidence_audit = audit_m6_production_evidence(
        evidence_bundle,
        evidence_root=evidence_root,
        release_sha256=str(release.release_sha256),
        evaluation=evaluation,
        renderer_approval_sha256=renderer_hashes,
    )
    recipes = {recipe.recipe_id: recipe for recipe in release.recipes}
    revisions = [
        manifest.generator.code_revision.lower()
        for manifest in manifests
    ]
    checks = {
        "candidate_release_audit_passed": bool(release_audit["valid"]),
        "source_release_is_immutable_candidate": (
            release.release_status == "candidate"
        ),
        "all_required_recipes_are_ready": bool(
            all(
                recipe_id in recipes
                and recipes[recipe_id].status == "ready"
                for recipe_id in REQUIRED_RECIPE_IDS
            )
        ),
        "real_and_mixed_recipe_semantics_are_valid": (
            _recipe_semantics_valid(release)
        ),
        "all_variant_items_are_qc_passed": bool(
            manifests
            and all(
                item.qc_status == "pass"
                and item.qc_report_path is not None
                and item.qc_report_sha256 is not None
                for manifest in manifests
                for item in manifest.items
            )
        ),
        "all_generator_revisions_are_pinned": bool(
            revisions
            and all(
                value not in {"dirty", "unknown"}
                and not value.endswith("-dirty")
                for value in revisions
            )
        ),
        "all_renderer_profiles_are_production_approved": bool(
            profiles
            and all(
                profile.evidence_tier == "production_approved"
                and profile.approval_report_sha256 is not None
                for profile in profiles
            )
        ),
        "evaluation_schema_and_content_hash_match": bool(
            evaluation.get("schema_version")
            == M6_EVALUATION_SCHEMA_VERSION
            and _evaluation_hash_valid(evaluation)
        ),
        "evaluation_targets_this_release": bool(
            evaluation.get("release_sha256") == release.release_sha256
        ),
        "m6_5_implementation_exit_passed": bool(
            _nested_bool(evaluation, "implementation_exit", "passed")
        ),
        "m6_5_empirical_exit_passed": bool(
            _nested_bool(evaluation, "empirical_exit", "passed")
        ),
        "m6_5_production_enablement_passed": bool(
            _nested_bool(evaluation, "production_enablement", "ready")
        ),
        "external_evidence_bundle_audits": bool(evidence_audit["valid"]),
    }
    if tuple(checks) != PRODUCTION_DECISION_CHECK_NAMES:
        raise RuntimeError("production decision check order is not canonical")
    return checks, release_audit, evidence_audit


def _evidence_root_relative_to_release(
    release_root: Path,
    evidence_root: Path,
) -> str | None:
    try:
        relative = evidence_root.resolve().relative_to(release_root.resolve())
    except ValueError:
        return None
    return relative.as_posix() or "."


def build_m6_production_decision(
    release_root: str | Path,
    evaluation_report: str | Path | Mapping[str, Any],
    *,
    evidence_bundle: str | Path | Mapping[str, Any] | None = None,
    evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic M6.6 decision without mutating the candidate bank."""

    root = Path(release_root)
    release = RIRBankReleaseManifest.from_json(
        (root / DEFAULT_RELEASE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    try:
        evaluation = _evaluation_payload(evaluation_report)
    except (OSError, ValueError, json.JSONDecodeError):
        evaluation = {}
    try:
        evidence_snapshot = (
            _load_mapping(evidence_bundle)
            if evidence_bundle is not None
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError):
        evidence_snapshot = None
    requested_evidence_root = Path(evidence_root or root)
    evidence_root_relative = _evidence_root_relative_to_release(
        root,
        requested_evidence_root,
    )
    auditable_bundle = (
        evidence_snapshot if evidence_root_relative is not None else None
    )
    auditable_root = (
        root / evidence_root_relative
        if evidence_root_relative is not None
        else root
    )
    checks, release_audit, evidence_audit = _production_decision_components(
        root,
        release,
        evaluation,
        evidence_bundle=auditable_bundle,
        evidence_root=auditable_root,
    )
    blockers = [name for name, passed in checks.items() if not passed]
    approved = not blockers
    decision: dict[str, Any] = {
        "schema_version": M6_PRODUCTION_DECISION_SCHEMA_VERSION,
        "milestone": "M6.6",
        "release_id": release.release_id,
        "release_sha256": release.release_sha256,
        "evaluation_sha256": evaluation.get("evaluation_sha256"),
        "evidence_bundle_sha256": evidence_audit.get("bundle_sha256"),
        "evaluation_snapshot": evaluation,
        "evidence_bundle_snapshot": auditable_bundle,
        "evidence_root_relative": evidence_root_relative,
        "release_audit": release_audit,
        "evidence_audit": evidence_audit,
        "checks": checks,
        "blockers": blockers,
        "decision": "approved" if approved else "blocked",
        "production_ready": approved,
        "promotion_model": "immutable_candidate_plus_content_addressed_certificate",
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)
    return decision


def validate_m6_production_certificate(
    certificate: str | Path | Mapping[str, Any],
    *,
    release_root: str | Path,
    require_approved: bool = False,
) -> dict[str, Any]:
    """Validate a stored decision certificate against the immutable release."""

    try:
        parsed = _load_mapping(certificate)
        release = RIRBankReleaseManifest.from_json(
            (Path(release_root) / DEFAULT_RELEASE_MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return {"valid": False, "approved": False}
    root = Path(release_root)
    claimed_hash = parsed.get("decision_sha256")
    payload = {key: value for key, value in parsed.items() if key != "decision_sha256"}
    checks = parsed.get("checks", {})
    blockers = parsed.get("blockers", ())
    evaluation = parsed.get("evaluation_snapshot")
    evidence_bundle = parsed.get("evidence_bundle_snapshot")
    evidence_root_relative = _safe_relative_path(
        parsed.get("evidence_root_relative")
    )
    checks_are_boolean = bool(
        isinstance(checks, Mapping)
        and tuple(checks) == PRODUCTION_DECISION_CHECK_NAMES
        and all(isinstance(value, bool) for value in checks.values())
    )
    snapshots_are_well_formed = bool(
        isinstance(evaluation, Mapping)
        and (evidence_bundle is None or isinstance(evidence_bundle, Mapping))
        and evidence_root_relative is not None
    )
    try:
        actual_checks, actual_release_audit, actual_evidence_audit = (
            _production_decision_components(
                root,
                release,
                dict(evaluation) if isinstance(evaluation, Mapping) else {},
                evidence_bundle=(
                    dict(evidence_bundle)
                    if isinstance(evidence_bundle, Mapping)
                    else None
                ),
                evidence_root=(
                    root / evidence_root_relative
                    if evidence_root_relative is not None
                    else root
                ),
            )
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
        actual_checks = {}
        actual_release_audit = {}
        actual_evidence_audit = {}
    checks_match_recomputed = bool(
        checks_are_boolean and dict(checks) == actual_checks
    )
    computed_approved = bool(
        checks_match_recomputed and all(actual_checks.values())
    )
    valid = bool(
        parsed.get("schema_version") == M6_PRODUCTION_DECISION_SCHEMA_VERSION
        and claimed_hash == canonical_json_sha256(payload)
        and parsed.get("release_sha256") == release.release_sha256
        and checks_are_boolean
        and snapshots_are_well_formed
        and checks_match_recomputed
        and parsed.get("release_audit") == actual_release_audit
        and parsed.get("evidence_audit") == actual_evidence_audit
        and parsed.get("evaluation_sha256")
        == evaluation.get("evaluation_sha256")
        and parsed.get("evidence_bundle_sha256")
        == actual_evidence_audit.get("bundle_sha256")
        and parsed.get("production_ready") is computed_approved
        and parsed.get("decision") == ("approved" if computed_approved else "blocked")
        and isinstance(blockers, list)
        and blockers == [name for name, passed in checks.items() if not passed]
    )
    approved = bool(valid and computed_approved)
    return {
        "valid": valid and (approved if require_approved else True),
        "approved": approved,
        "decision_sha256": claimed_hash,
        "release_sha256": parsed.get("release_sha256"),
        "blockers": list(blockers) if isinstance(blockers, list) else [],
    }


__all__ = [
    "DEFAULT_PRODUCTION_DECISION_NAME",
    "M6_PRODUCTION_DECISION_SCHEMA_VERSION",
    "M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION",
    "M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION",
    "PRODUCTION_DECISION_CHECK_NAMES",
    "REQUIRED_EVIDENCE_KINDS",
    "REQUIRED_RECIPE_IDS",
    "audit_m6_production_evidence",
    "build_m6_production_decision",
    "validate_m6_production_certificate",
]
