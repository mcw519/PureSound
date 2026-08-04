#!/usr/bin/env python3
"""Ingest published measured RIRs into a QC-passed M6 ``measured`` variant.

Unblocks M6's ``real_native`` recipe. Measured corpora fail M6 item QC as
published — every channel fails ``prearrival_energy`` — because their time origin
is their own direct arrival rather than the source emission instant M6 assumes.
This aligns each channel's ISO 3382-1 impulse-response start onto the geometric
arrival implied by its published distance, then runs the same item QC that
synthetic banks run.

Writes two banks: the full ingest (with quarantine intact, as the record) and,
with ``--pruned-bank``, a copy holding only QC-passed items, which is what
``build_m6_variant_release --measured-bank`` consumes.

Exit code 0 when the QC release audits and at least one item passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from puresound.audio.rir.bank.measured_ingest import (
    MeasuredAlignmentPolicy,
    build_measured_m6_bank,
)
from puresound.audio.rir.bank.qc import RIRBankQCPolicy
from puresound.audio.rir.bank.release import prune_bank_to_qc_passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="corpus view directory of <corpus>_<room>_<index>.{wav,json} pairs",
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument(
        "--pruned-bank",
        type=Path,
        help="also write a quarantine-free copy for use as a release variant",
    )
    parser.add_argument(
        "--corpora",
        nargs="*",
        help="restrict to these corpus prefixes (default: every corpus found)",
    )
    parser.add_argument(
        "--limit-per-corpus",
        type=int,
        help="sample this many items per corpus, spread across its rooms",
    )
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--bank-id", default="puresound-m6-measured")
    parser.add_argument("--code-revision", default="unknown")
    parser.add_argument(
        "--alignment-policy-json",
        type=Path,
        help="JSON object overriding the versioned measured alignment policy",
    )
    parser.add_argument(
        "--qc-policy-json",
        type=Path,
        help="JSON object overriding the complete versioned item QC policy",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    alignment_policy = None
    if args.alignment_policy_json is not None:
        alignment_policy = MeasuredAlignmentPolicy(
            **json.loads(args.alignment_policy_json.read_text(encoding="utf-8"))
        )
    qc_policy = None
    if args.qc_policy_json is not None:
        qc_policy = RIRBankQCPolicy.from_dict(
            json.loads(args.qc_policy_json.read_text(encoding="utf-8"))
        )

    report = build_measured_m6_bank(
        args.source,
        args.bank,
        corpora=args.corpora or None,
        limit_per_corpus=args.limit_per_corpus,
        split_seed=args.split_seed,
        bank_id=args.bank_id,
        code_revision=args.code_revision,
        alignment_policy=alignment_policy,
        qc_policy=qc_policy,
        qc_workers=args.workers,
    )
    counts = report.counts
    print(
        "[measured_ingest] "
        f"source={counts['source_items']} aligned={counts['aligned_items']} "
        f"rejected={counts['rejected_items']}"
    )
    for corpus, row in sorted(report.per_corpus.items()):
        alignment = row.get("alignment", {})
        print(
            f"[measured_ingest]   {corpus:<10} "
            f"aligned={row.get('aligned', 0):>4} rejected={row.get('rejected', 0):>4} "
            f"shift_p50={alignment.get('shift_samples_p50')} "
            f"reasons={row.get('rejection_reasons', {})}"
        )
    if report.qc_summary is not None:
        qc = report.qc_summary["counts"]
        print(
            f"[measured_ingest] qc passed={qc['passed']} "
            f"quarantined={qc['quarantined']} "
            f"audit={'PASS' if report.qc_audit_valid else 'FAIL'}"
        )

    pruned = None
    if args.pruned_bank is not None:
        pruned = prune_bank_to_qc_passed(
            args.bank,
            args.pruned_bank,
            qc_policy=qc_policy,
            qc_workers=args.workers,
        )
        print(
            f"[measured_ingest] pruned kept={pruned['kept_item_count']} "
            f"dropped={pruned['dropped_item_count']} "
            f"audit={'PASS' if pruned['qc_audit_valid'] else 'FAIL'}"
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"ingest": report.to_dict(), "pruned": pruned},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[measured_ingest] wrote {args.report}")

    ok = bool(report.qc_audit_valid) and bool(
        report.qc_summary and report.qc_summary["counts"]["passed"]
    )
    if pruned is not None:
        ok = ok and bool(pruned["qc_audit_valid"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
