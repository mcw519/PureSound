#!/usr/bin/env python3
"""Produce the M6.6 evidence a production decision audits, and report what is left.

M6 could validate every piece of external evidence and produce none of it, so the
decision was blocked on artifacts nothing could make. This drives the chain:

  throughput      from the generator's own audit
  approval        one record per renderer profile, stamped into the bank before QC
  listening       a randomized room-disjoint assignment, and the response ingest
  sign-offs       three role records bound to this release and evaluation
  bundle          the above, hashed from disk

Approval has to land before QC — a bank's QC summary is bound to its manifest hash,
and stamping a profile changes it — so this is a two-pass flow. Pass one builds a
candidate release and evaluates it; pass two approves the renderer on the strength
of that evaluation, re-runs QC, rebuilds, and signs off against the rebuilt
release. Use --pass to choose.

Without human responses the listening report goes out at contract_fixture tier and
controlled_listening_empirical_passed stays False. That is the correct outcome, not
a failure of this script: supply real responses with --listening-responses to make
it empirical.

Exit code 0 when everything requested was produced; 3 when the decision is still
blocked, which is reported rather than treated as an error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from puresound.audio.rir.bank.evaluation import evaluate_m6_release
from puresound.audio.rir.bank.evidence import (
    RendererApproval,
    approve_bank_renderer_profiles,
    build_evidence_bundle,
    build_throughput_report,
    write_production_signoff,
)
from puresound.audio.rir.bank.listening import (
    DRY_RUN_EVIDENCE_TIER,
    EMPIRICAL_EVIDENCE_TIER,
    ListeningProtocol,
    build_listening_assignment,
    ingest_listening_responses,
)
from puresound.audio.rir.bank.production import (
    PRODUCTION_DECISION_CHECK_NAMES,
    build_m6_production_decision,
)
from puresound.audio.rir.bank.release import (
    RIRBankReleaseManifest,
    build_m6_variant_release,
    prune_bank_to_qc_passed,
)
from puresound.audio.rir.bank.schema import RIRBankManifest

SIGNOFF_ROLES = (
    ("acoustics", "acoustics_signoff"),
    ("ml", "ml_signoff"),
    ("release", "release_signoff"),
)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _release_sha256(release_root: Path) -> str:
    return str(
        RIRBankReleaseManifest.from_json(
            (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
        ).release_sha256
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--generation-audit",
        type=Path,
        help="rir_bank_generation_audit.json; required to build the throughput report",
    )
    parser.add_argument(
        "--pass",
        dest="stage",
        choices=("evaluate", "approve", "attest"),
        default="evaluate",
        help=(
            "evaluate: throughput + evaluation only. approve: stamp renderer "
            "approvals into the banks and rebuild. attest: listening, sign-offs, "
            "bundle and the production decision."
        ),
    )
    parser.add_argument("--synthetic-bank", type=Path)
    parser.add_argument("--measured-bank", type=Path)
    parser.add_argument("--rebuild-release", type=Path)
    parser.add_argument("--approver-id")
    parser.add_argument(
        "--approval-scope",
        default=(
            "M6.5 implementation exit, item QC yield, and the acoustic distribution "
            "comparison against the measured variant"
        ),
    )
    parser.add_argument("--reviewer-id", help="used for all three role sign-offs")
    parser.add_argument("--participants", type=int, default=24)
    parser.add_argument("--trials-per-participant", type=int, default=10)
    parser.add_argument("--noninferiority-margin", type=float, default=0.5)
    parser.add_argument("--assignment-seed", type=int, default=20260804)
    parser.add_argument(
        "--listening-responses",
        type=Path,
        help=(
            "collected responses (.json or .jsonl). Omit for a contract_fixture dry "
            "run, which cannot satisfy the empirical listening check."
        ),
    )
    parser.add_argument("--qc-workers", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    evidence_root = args.evidence_root
    evidence_root.mkdir(parents=True, exist_ok=True)
    release_sha256 = _release_sha256(args.release)

    throughput = None
    if args.generation_audit is not None:
        throughput = build_throughput_report(
            args.generation_audit, release_sha256=release_sha256
        )
        _write_json(evidence_root / "throughput.json", throughput)
        print(
            f"[m6_evidence] throughput {throughput['items_per_second']:.3f} items/s, "
            f"generated={throughput['items_generated']} "
            f"failed={throughput['items_failed']}"
        )

    if args.stage == "evaluate":
        evaluation = evaluate_m6_release(args.release, throughput_report=throughput)
        _write_json(evidence_root / "evaluation_pass1.json", evaluation)
        print(
            f"[m6_evidence] evaluation={evaluation['evaluation_sha256'][:16]} "
            f"implementation_exit={evaluation['implementation_exit']['passed']}"
        )
        for name, ok in evaluation["implementation_checks"].items():
            if not ok:
                print(f"[m6_evidence]   failing: {name}")
        print(f"[m6_evidence] wrote {evidence_root / 'evaluation_pass1.json'}")
        return 0

    if args.stage == "approve":
        for name, value in (
            ("--synthetic-bank", args.synthetic_bank),
            ("--rebuild-release", args.rebuild_release),
            ("--approver-id", args.approver_id),
        ):
            if not value:
                parser.error(f"{name} is required for --pass approve")
        evaluation = json.loads(
            (evidence_root / "evaluation_pass1.json").read_text(encoding="utf-8")
        )
        approval_dir = evidence_root / "approvals"
        records = []
        # Prune before stamping. A release variant must be all-pass, because the
        # production decision checks every item in every variant manifest, and a
        # real 100-room pilot arrives with quarantine to shed just as a measured
        # bank does — one item in 400 failed decay_fit_coverage. Pruning also
        # copies, which keeps the source banks untouched by the stamping below.
        synthetic_target = args.rebuild_release.parent / "synthetic_approved"
        if synthetic_target.exists():
            shutil.rmtree(synthetic_target)
        pruned_synthetic = prune_bank_to_qc_passed(
            args.synthetic_bank, synthetic_target, qc_workers=args.qc_workers
        )
        print(
            f"[m6_evidence] pruned synthetic kept={pruned_synthetic['kept_item_count']} "
            f"dropped={pruned_synthetic['dropped_item_count']}"
        )
        banks = [synthetic_target]
        measured_target = None
        if args.measured_bank is not None:
            measured_target = args.rebuild_release.parent / "measured_approved"
            if measured_target.exists():
                shutil.rmtree(measured_target)
            pruned_measured = prune_bank_to_qc_passed(
                args.measured_bank, measured_target, qc_workers=args.qc_workers
            )
            print(
                f"[m6_evidence] pruned measured kept={pruned_measured['kept_item_count']} "
                f"dropped={pruned_measured['dropped_item_count']}"
            )
            banks.append(measured_target)
        for bank in banks:
            manifest = RIRBankManifest.from_json(
                (bank / "rir_bank_manifest.json").read_text(encoding="utf-8")
            )
            result = approve_bank_renderer_profiles(
                bank,
                [
                    RendererApproval(
                        profile_id=profile.profile_id,
                        approver_id=args.approver_id,
                        reviewed_scope=args.approval_scope,
                        evidence_sha256={
                            "m6_5_evaluation": evaluation["evaluation_sha256"],
                            "candidate_release": evaluation["release_sha256"],
                        },
                    )
                    for profile in manifest.renderer_profiles
                ],
                approval_dir=approval_dir,
                qc_workers=args.qc_workers,
            )
            records.extend(result["approval_records"])
            print(
                f"[m6_evidence] approved {len(result['approval_records'])} profiles "
                f"in {bank}"
            )
        release = build_m6_variant_release(
            synthetic_target,
            args.rebuild_release,
            release_id="puresound-m6-candidate",
            measured_bank_root=measured_target,
            qc_workers=args.qc_workers,
        )
        _write_json(
            evidence_root / "approval_records.json",
            {"records": records, "release_sha256": release.release_sha256},
        )
        print(
            f"[m6_evidence] rebuilt release={release.release_sha256[:16]} "
            f"at {args.rebuild_release}"
        )
        return 0

    # attest
    if not args.reviewer_id:
        parser.error("--reviewer-id is required for --pass attest")
    protocol = ListeningProtocol(
        participant_count=args.participants,
        trials_per_participant=args.trials_per_participant,
        noninferiority_margin=args.noninferiority_margin,
    )
    assignment = build_listening_assignment(
        args.release, protocol, assignment_seed=args.assignment_seed
    )
    _write_json(evidence_root / "listening/assignment_plan.json", assignment)
    responses = args.listening_responses
    tier = EMPIRICAL_EVIDENCE_TIER if responses else DRY_RUN_EVIDENCE_TIER
    if responses is None:
        print(
            "[m6_evidence] no --listening-responses given; emitting a "
            f"{DRY_RUN_EVIDENCE_TIER} dry run, which cannot satisfy "
            "controlled_listening_empirical_passed"
        )
        import numpy as np

        rng = np.random.default_rng(args.assignment_seed)
        payload = [
            {
                "participant_id": row["participant_id"],
                "stimulus_label": row["stimulus_label"],
                "primary_endpoint_difference": float(rng.normal(0.0, 0.3)),
            }
            for row in assignment["assignment_records"]
        ]
        listening = ingest_listening_responses(
            assignment,
            payload,
            evidence_tier=DRY_RUN_EVIDENCE_TIER,
            explicitly_not_human_responses=True,
            analysis_notes="pipeline dry run; responses are not from human listeners",
        )
    else:
        listening = ingest_listening_responses(
            assignment, responses, evidence_tier=EMPIRICAL_EVIDENCE_TIER
        )
    _write_json(evidence_root / "listening/report.json", listening)
    print(
        f"[m6_evidence] listening tier={listening['evidence_tier']} "
        f"estimate={listening['results']['estimate']:.4f} "
        f"ci_low={listening['results']['confidence_interval_low']:.4f} "
        f"noninferior={listening['results']['noninferior']} "
        f"coverage_complete={listening['coverage']['complete']}"
    )

    evaluation = evaluate_m6_release(
        args.release, throughput_report=throughput, listening_report=listening
    )
    _write_json(evidence_root / "evaluation.json", evaluation)

    artifacts: dict[str, list[str]] = {}
    for kind, name, payload in (
        (
            "listening_assignment",
            "listening/assignment.json",
            listening["artifacts"]["assignment_records"],
        ),
        (
            "listening_responses",
            "listening/responses.json",
            listening["artifacts"]["response_records"],
        ),
        (
            "listening_analysis",
            "listening/analysis.json",
            listening["artifacts"]["analysis_record"],
        ),
    ):
        path = _write_json(evidence_root / name, payload)
        artifacts[kind] = [str(path.relative_to(evidence_root))]

    approval_path = evidence_root / "approval_records.json"
    if approval_path.is_file():
        stored = json.loads(approval_path.read_text(encoding="utf-8"))
        artifacts["renderer_approval"] = [
            str(Path(row["path"]).relative_to(evidence_root))
            for row in stored["records"]
        ]

    for role, kind in SIGNOFF_ROLES:
        path, _sha = write_production_signoff(
            evidence_root / f"signoffs/{role}.json",
            role=role,
            reviewer_id=args.reviewer_id,
            reviewed_scope=f"{role} review of release {release_sha256[:16]}",
            release_sha256=release_sha256,
            evaluation_sha256=evaluation["evaluation_sha256"],
        )
        artifacts[kind] = [str(path.relative_to(evidence_root))]

    bundle_path = evidence_root / "rir_bank_production_evidence.json"
    bundle = None
    try:
        bundle = build_evidence_bundle(
            evidence_root,
            artifacts,
            release_sha256=release_sha256,
            evaluation_sha256=evaluation["evaluation_sha256"],
        )
        print(f"[m6_evidence] bundle {len(bundle['artifacts'])} artifacts")
    except (ValueError, FileNotFoundError) as error:
        print(f"[m6_evidence] bundle not built: {error}")

    decision = build_m6_production_decision(
        args.release,
        evaluation,
        evidence_bundle=bundle_path if bundle else None,
        evidence_root=evidence_root,
    )
    _write_json(evidence_root / "production_decision.json", decision)
    for name in PRODUCTION_DECISION_CHECK_NAMES:
        flag = "PASS" if decision["checks"][name] else "----"
        print(f"[m6_evidence]   {flag} {name}")
    print(f"[m6_evidence] production_ready={decision['production_ready']}")
    for blocker in decision["blockers"]:
        print(f"[m6_evidence] blocker={blocker}")

    if args.report is not None:
        _write_json(
            args.report,
            {
                "release_sha256": release_sha256,
                "listening_evidence_tier": tier,
                "throughput": throughput,
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "bundle_sha256": bundle["bundle_sha256"] if bundle else None,
                "decision": decision,
            },
        )
        print(f"[m6_evidence] wrote {args.report}")
    return 0 if decision["production_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
