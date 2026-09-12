#!/usr/bin/env python3
"""Build an M6.6 immutable production decision certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from puresound.audio.rir.bank.production import (
    DEFAULT_PRODUCTION_DECISION_NAME,
    build_m6_production_decision,
    validate_m6_production_certificate,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="return a non-zero status when the valid decision is blocked",
    )
    args = parser.parse_args(argv)
    output = args.output_report or args.release / DEFAULT_PRODUCTION_DECISION_NAME
    decision = build_m6_production_decision(
        args.release,
        args.evaluation_report,
        evidence_bundle=args.evidence_bundle,
        evidence_root=args.evidence_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    certificate = validate_m6_production_certificate(
        output,
        release_root=args.release,
    )
    status = "APPROVED" if decision["production_ready"] else "BLOCKED"
    print(f"[m6_production] decision={status} certificate={certificate['valid']}")
    for blocker in decision["blockers"]:
        print(f"[m6_production] blocker={blocker}")
    print(f"[m6_production] wrote {output}")
    if not certificate["valid"]:
        return 1
    if args.require_approved and not certificate["approved"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
