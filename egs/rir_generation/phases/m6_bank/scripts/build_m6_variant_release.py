#!/usr/bin/env python3
"""Build and audit an M6.4 calibrated/normalized candidate RIR release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from puresound.audio.rir.bank.release import (
    audit_m6_variant_release,
    build_m6_variant_release,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", default="puresound-m6-candidate")
    parser.add_argument("--normalized-peak", type=float, default=0.98)
    parser.add_argument(
        "--qc-workers",
        type=int,
        default=1,
        help="Parallel item evaluators for the normalized variant QC.",
    )
    args = parser.parse_args(argv)
    release = build_m6_variant_release(
        args.source_bank,
        args.output_dir,
        release_id=args.release_id,
        normalized_peak=args.normalized_peak,
        qc_workers=args.qc_workers,
    )
    audit = audit_m6_variant_release(args.output_dir)
    audit_path = args.output_dir / "rir_bank_release_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[m6_variant_release] release={release.release_sha256} "
        f"audit={'PASS' if audit['valid'] else 'FAIL'}"
    )
    print(f"[m6_variant_release] wrote {args.output_dir}")
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
