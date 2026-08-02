#!/usr/bin/env python3
"""Run deterministic M6.3 per-item RIR QC and publish quarantine indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from puresound.audio.rir_bank_qc import (
    DEFAULT_QC_SUMMARY_NAME,
    RIRBankQCPolicy,
    audit_rir_bank_qc_release,
    run_rir_bank_qc,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest-name", default="rir_bank_manifest.json")
    parser.add_argument("--summary-name", default=DEFAULT_QC_SUMMARY_NAME)
    parser.add_argument(
        "--policy-json",
        type=Path,
        help="Optional JSON object overriding the complete versioned QC policy.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel item evaluators; report publication remains deterministic.",
    )
    args = parser.parse_args(argv)
    policy = None
    if args.policy_json is not None:
        policy = RIRBankQCPolicy.from_dict(
            json.loads(args.policy_json.read_text(encoding="utf-8"))
        )
    summary = run_rir_bank_qc(
        args.bank,
        manifest_name=args.manifest_name,
        summary_name=args.summary_name,
        policy=policy,
        workers=args.workers,
    )
    audit = audit_rir_bank_qc_release(
        args.bank,
        manifest_name=args.manifest_name,
        summary_name=args.summary_name,
    )
    counts = summary["counts"]
    print(
        "[m6_item_qc] "
        f"passed={counts['passed']} quarantined={counts['quarantined']} "
        f"release={summary['release']['status']}"
    )
    print(f"[m6_item_qc] audit={'PASS' if audit['valid'] else 'FAIL'}")
    print(f"[m6_item_qc] wrote {args.bank / args.summary_name}")
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
