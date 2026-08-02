#!/usr/bin/env python3
"""Run M6.5 bank-level evaluation with optional external evidence reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from puresound.audio.rir_bank_evaluation import evaluate_m6_release


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--throughput-report", type=Path, required=True)
    parser.add_argument("--listening-report", type=Path)
    parser.add_argument("--downstream-report", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_m6_release(
        args.release,
        throughput_report=args.throughput_report,
        listening_report=args.listening_report,
        downstream_report=args.downstream_report,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "[m6_evaluation] implementation="
        f"{'PASS' if report['implementation_exit']['passed'] else 'FAIL'} "
        "empirical="
        f"{'PASS' if report['empirical_exit']['passed'] else 'OPEN'}"
    )
    print(f"[m6_evaluation] wrote {args.output_report}")
    return 0 if report["implementation_exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
