"""Produce the external evidence M6.6's production decision audits.

M6 has validators for every piece of evidence a production decision rests on, and
producers for almost none of it. ``audit_m6_production_evidence`` will check a
bundle, three role sign-offs and a renderer approval record per profile against
the release and evaluation hashes — but nothing in the repository writes any of
those, so the decision is permanently blocked on artifacts that cannot be made.
This module writes them.

Everything here records a decision somebody made; none of it decides anything.
That distinction is the whole design:

*Approvals and sign-offs need a person.* ``write_renderer_approval_record`` and
``write_production_signoff`` refuse to emit an approval without a named approver
and a stated scope. They copy the evidence hashes they are handed into the record
and never compute, guess, or default them — a record that cites evidence nobody
supplied would be worse than no record.

*Throughput is measured, not asserted.* ``build_throughput_report`` reads the
generator's own audit. It cannot be called without one, and the elapsed time and
failure count come from the run rather than from an argument.

*The bundle hashes real files.* ``build_evidence_bundle`` fails when a declared
artifact is missing, so a bundle either describes files that exist or does not
exist itself.

Approval has to be stamped before QC, which makes the release a two-pass process.
``manifest_hash_matches_summary`` binds a bank's QC summary to its manifest hash,
and stamping a renderer profile changes that hash, so an approval applied after
QC would invalidate the release it was meant to bless. The order is therefore:
build a candidate release and evaluate it, approve the renderer *on the strength
of that evaluation*, then re-run QC and rebuild so the approval is inside the hash
chain, and finally sign off against the rebuilt release. The approval record cites
the first pass's ``evaluation_sha256``, which is exactly what it was decided on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

# Imported rather than restated: this module's whole purpose is to produce what
# these validators check, so a drifting copy of a schema version or a required-kind
# list would produce artifacts that fail for no visible reason.
from .evaluation import M6_THROUGHPUT_SCHEMA_VERSION
from .production import (
    M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
    M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION,
    REQUIRED_EVIDENCE_KINDS,
    SIGNOFF_KIND_TO_ROLE,
)
from .qc import DEFAULT_QC_SUMMARY_NAME, RIRBankQCPolicy, run_rir_bank_qc
from .schema import (
    RIRBankManifest,
    canonical_json_sha256,
    sha256_file,
)

M6_RENDERER_APPROVAL_SCHEMA_VERSION = "puresound.m6_renderer_approval.v1"

#: Every artifact kind the bundle audit will look for, including the per-profile
#: renderer approvals that ``REQUIRED_EVIDENCE_KINDS`` does not list because their
#: count depends on the release.
RENDERER_APPROVAL_KIND = "renderer_approval"

APPROVAL_DECISIONS = ("approve", "reject")


def _required_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _relative_path(name: str, value: str | Path) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path inside the evidence root")
    return path.as_posix()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def build_throughput_report(
    generation_audit: str | Path | Mapping[str, Any],
    *,
    release_sha256: str,
) -> dict[str, Any]:
    """Turn a generator audit's ``generation_run`` block into an M6.5 report.

    The generator is the only thing that knows how long it ran and how many items
    it produced, so the report is derived from its audit rather than assembled by
    hand. A run whose audit carries no elapsed time cannot produce a report: the
    throughput contract requires a positive duration, and inventing one here would
    turn a missing measurement into a passing check.
    """

    if isinstance(generation_audit, Mapping):
        audit = dict(generation_audit)
    else:
        audit = json.loads(Path(generation_audit).read_text(encoding="utf-8"))
    run = audit.get("generation_run")
    if not isinstance(run, Mapping):
        raise ValueError("generation audit has no generation_run block")
    elapsed = run.get("elapsed_seconds")
    if elapsed is None or float(elapsed) <= 0.0:
        raise ValueError(
            "generation audit records no positive elapsed_seconds; regenerate the "
            "bank with a generator that times its run"
        )
    if run.get("items_failed") is None:
        raise ValueError("generation audit does not state items_failed")
    elapsed = float(elapsed)
    generated = int(run["items_generated"])
    skipped = int(run["items_skipped_as_complete"])
    failed = int(run["items_failed"])
    task_count = int(run["task_count"])
    report = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": _required_text("release_sha256", release_sha256),
        "task_count": task_count,
        "items_generated": generated,
        "items_skipped": skipped,
        "items_failed": failed,
        "elapsed_seconds": elapsed,
        "items_per_second": generated / elapsed,
        "num_workers": int(run["num_workers"]),
        "source": {
            "bank_id": audit.get("bank_id"),
            "bank_root": audit.get("root"),
            "resume_requested": bool(run.get("resume_requested")),
            "items_failed_policy": run.get("items_failed_policy"),
        },
    }
    if generated + skipped + failed != task_count:
        raise ValueError(
            "generation audit counts are inconsistent: "
            f"{generated} + {skipped} + {failed} != {task_count}"
        )
    return report


@dataclass(frozen=True)
class RendererApproval:
    """One person's recorded decision about one renderer profile.

    ``evidence_sha256`` is whatever the approver actually looked at, keyed by a
    name they choose — the M6.5 evaluation, validator reports, a QC summary. It is
    copied into the record verbatim so the decision stays attached to its basis.
    """

    profile_id: str
    approver_id: str
    reviewed_scope: str
    evidence_sha256: Mapping[str, str]
    decision: str = "approve"
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", _required_text("profile_id", self.profile_id)
        )
        object.__setattr__(
            self, "approver_id", _required_text("approver_id", self.approver_id)
        )
        object.__setattr__(
            self, "reviewed_scope", _required_text("reviewed_scope", self.reviewed_scope)
        )
        if self.decision not in APPROVAL_DECISIONS:
            raise ValueError(f"decision must be one of {APPROVAL_DECISIONS}")
        if not self.evidence_sha256:
            raise ValueError(
                "an approval must cite at least one evidence hash; approving a "
                "renderer on no stated evidence is what this record exists to prevent"
            )
        object.__setattr__(
            self,
            "evidence_sha256",
            {
                _required_text("evidence name", key): _required_text(
                    f"evidence_sha256[{key}]", value
                )
                for key, value in sorted(dict(self.evidence_sha256).items())
            },
        )


def write_renderer_approval_record(
    path: str | Path,
    approval: RendererApproval,
    *,
    profile: Mapping[str, Any],
) -> tuple[Path, str]:
    """Write one approval record and return its path and *file* hash.

    The file hash is what matters: the bundle audit verifies
    ``sha256_file(path) == row["sha256"]`` and then requires that same value to
    appear as the profile's ``approval_report_sha256``. A content hash of the
    payload would not match the file on disk once it is indented and newline
    terminated, so callers must use the value returned here.
    """

    record = {
        "schema_version": M6_RENDERER_APPROVAL_SCHEMA_VERSION,
        "profile_id": approval.profile_id,
        "decision": approval.decision,
        "approver_id": approval.approver_id,
        "reviewed_scope": approval.reviewed_scope,
        "notes": approval.notes,
        "renderer": {
            "renderer_id": profile.get("renderer_id"),
            "renderer_version": profile.get("renderer_version"),
            "low_backend": profile.get("low_backend"),
            "high_backend": profile.get("high_backend"),
            "renderer_config_sha256": profile.get("renderer_config_sha256"),
            "scene_schema_version": profile.get("scene_schema_version"),
        },
        "evidence_sha256": dict(approval.evidence_sha256),
    }
    record["approval_sha256"] = canonical_json_sha256(record)
    written = _atomic_json(Path(path), record)
    return written, sha256_file(written)


def approve_bank_renderer_profiles(
    bank_root: str | Path,
    approvals: Sequence[RendererApproval],
    *,
    approval_dir: str | Path,
    qc_policy: RIRBankQCPolicy | None = None,
    qc_workers: int = 1,
    summary_name: str = DEFAULT_QC_SUMMARY_NAME,
) -> dict[str, Any]:
    """Stamp a bank's renderer profiles as production approved, then re-run QC.

    Every profile in the bank needs an approval: the production decision requires
    *all* of them to be approved, so a partial pass would only move the failure to
    a later, more confusing place. Rejections are written as records too but leave
    the profile's tier alone — a rejected renderer is evidence, not an approval.

    QC is re-run because stamping changes ``manifest_sha256`` and the QC summary is
    bound to it. That is the price of putting the approval inside the hash chain
    rather than beside it.
    """

    root = Path(bank_root)
    manifest_path = root / "rir_bank_manifest.json"
    manifest = RIRBankManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    by_profile = {item.profile_id: item for item in manifest.renderer_profiles}
    approvals_by_profile = {approval.profile_id: approval for approval in approvals}

    unknown = sorted(set(approvals_by_profile) - set(by_profile))
    if unknown:
        raise ValueError(f"approval references unknown renderer profiles: {unknown}")
    approving = {
        profile_id
        for profile_id, approval in approvals_by_profile.items()
        if approval.decision == "approve"
    }
    missing = sorted(set(by_profile) - approving)
    if missing:
        raise ValueError(
            "every renderer profile in the bank needs an approval before the "
            f"production decision can pass; missing or rejected: {missing}"
        )

    approval_root = Path(approval_dir)
    records: list[dict[str, Any]] = []
    stamped: list[Any] = []
    for profile in manifest.renderer_profiles:
        approval = approvals_by_profile[profile.profile_id]
        path, file_sha256 = write_renderer_approval_record(
            approval_root / f"renderer_approval_{profile.profile_id}.json",
            approval,
            profile=profile.to_dict(),
        )
        records.append(
            {
                "profile_id": profile.profile_id,
                "path": str(path),
                "sha256": file_sha256,
                "approver_id": approval.approver_id,
                "decision": approval.decision,
            }
        )
        stamped.append(
            replace(
                profile,
                evidence_tier="production_approved",
                approval_report_sha256=file_sha256,
            )
        )

    approved = RIRBankManifest(
        bank_id=manifest.bank_id,
        release_status=manifest.release_status,
        split_policy=manifest.split_policy,
        generator=manifest.generator,
        renderer_profiles=tuple(stamped),
        items=manifest.items,
        split_indexes=manifest.split_indexes,
    ).with_content_sha256()
    _atomic_json(manifest_path, approved.to_dict())
    summary = run_rir_bank_qc(
        root, summary_name=summary_name, policy=qc_policy, workers=int(qc_workers)
    )
    return {
        "schema_version": M6_RENDERER_APPROVAL_SCHEMA_VERSION,
        "bank_root": str(root),
        "manifest_sha256": approved.manifest_sha256,
        "approval_records": records,
        "qc_counts": summary["counts"],
    }


def write_production_signoff(
    path: str | Path,
    *,
    role: str,
    reviewer_id: str,
    reviewed_scope: str,
    release_sha256: str,
    evaluation_sha256: str,
    decision: str = "approve",
    notes: str = "",
) -> tuple[Path, str]:
    """Write one role sign-off bound to a specific release and evaluation.

    Binding both hashes is what makes a sign-off non-transferable: it approves one
    release evaluated one way, and re-running either invalidates it.
    """

    if role not in set(SIGNOFF_KIND_TO_ROLE.values()):
        raise ValueError(
            f"role must be one of {sorted(set(SIGNOFF_KIND_TO_ROLE.values()))}"
        )
    if decision not in APPROVAL_DECISIONS:
        raise ValueError(f"decision must be one of {APPROVAL_DECISIONS}")
    record = {
        "schema_version": M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION,
        "role": role,
        "reviewer_id": _required_text("reviewer_id", reviewer_id),
        "reviewed_scope": _required_text("reviewed_scope", reviewed_scope),
        "decision": decision,
        "release_sha256": _required_text("release_sha256", release_sha256),
        "evaluation_sha256": _required_text("evaluation_sha256", evaluation_sha256),
        "notes": notes,
    }
    written = _atomic_json(Path(path), record)
    return written, sha256_file(written)


def build_evidence_bundle(
    evidence_root: str | Path,
    artifacts: Mapping[str, Sequence[str | Path]],
    *,
    release_sha256: str,
    evaluation_sha256: str,
    bundle_name: str = "rir_bank_production_evidence.json",
) -> dict[str, Any]:
    """Hash the declared evidence files and write the M6.6 bundle.

    ``artifacts`` maps an evidence kind to paths relative to ``evidence_root``. Each
    file is hashed from disk, so a bundle cannot describe an artifact that is not
    there. Missing required kinds are reported rather than silently omitted,
    because a bundle that quietly lacks a kind fails the audit with no indication
    of which one.
    """

    root = Path(evidence_root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, paths in sorted(dict(artifacts).items()):
        kind_text = _required_text("evidence kind", kind)
        for value in paths:
            relative = _relative_path(f"{kind_text} path", value)
            if relative in seen:
                raise ValueError(
                    f"evidence path {relative} is declared twice; the audit "
                    "requires unique paths"
                )
            absolute = root / relative
            if not absolute.is_file():
                raise FileNotFoundError(
                    f"declared {kind_text} evidence is missing: {absolute}"
                )
            seen.add(relative)
            rows.append(
                {
                    "kind": kind_text,
                    "path": relative,
                    "sha256": sha256_file(absolute),
                }
            )
    missing = [kind for kind in REQUIRED_EVIDENCE_KINDS if kind not in artifacts]
    if missing:
        raise ValueError(
            f"evidence bundle is missing required kinds: {missing}. Every one is "
            "an external artifact that has to exist before production is decided."
        )
    if RENDERER_APPROVAL_KIND not in artifacts:
        raise ValueError(
            f"evidence bundle needs at least one {RENDERER_APPROVAL_KIND} artifact, "
            "one per renderer profile in the release"
        )
    bundle: dict[str, Any] = {
        "schema_version": M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
        "release_sha256": _required_text("release_sha256", release_sha256),
        "evaluation_sha256": _required_text("evaluation_sha256", evaluation_sha256),
        "artifacts": rows,
    }
    bundle["bundle_sha256"] = canonical_json_sha256(bundle)
    _atomic_json(root / bundle_name, bundle)
    return bundle


__all__ = [
    "APPROVAL_DECISIONS",
    "M6_RENDERER_APPROVAL_SCHEMA_VERSION",
    "RENDERER_APPROVAL_KIND",
    "RendererApproval",
    "approve_bank_renderer_profiles",
    "build_evidence_bundle",
    "build_throughput_report",
    "write_production_signoff",
    "write_renderer_approval_record",
]
