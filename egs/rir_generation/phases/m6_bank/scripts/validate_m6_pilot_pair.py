#!/usr/bin/env python
"""Check that a finished A/B pilot is actually an A/B, then compare the arms.

A matched pilot is only interpretable if the two arms differ in exactly one
thing: the high-frequency renderer.  Everything else — the sampled scenes, the
split assignment, the seeds, the low band, the output mode — has to be
identical, or an acoustic difference between the banks cannot be attributed to
the backend.

So this checks the pairing contract first and refuses to report a comparison it
cannot attribute:

    same acoustic-space set              both arms rendered the same rooms
    per-item scene/split/seed/shape      matched item by item, not just in bulk
    identical low-band configuration     the shared half really is shared
    differing high-band backend          the intended difference is present
    QC yield and release audit per arm   a yield gap is itself a result

Then it summarizes the realized acoustics each arm recorded, bucketed by source
distance.  Those are the generator's own measurements; for an independent read
of the WAVs run ``compare_bank_acoustics.py`` over the two bank roots.

Exit code 0 if the pairing holds, 1 if it does not.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.bank.release import audit_m6_variant_release
from puresound.audio.rir.bank.schema import RIRBankManifest

#: Item-level identity that must agree across arms for the pair to be matched.
MATCHED_KEYS = (
    "scene_sha256",
    "split",
    "generation_seed",
    "sample_rate",
    "channel_count",
    "frame_count",
)

#: Low-band metadata that must agree; the shared half of the renderer.
SHARED_LOW_BAND_KEYS = (
    "backend",
    "boundary_model",
    "global_rt60_envelope_applied",
    "mode_excitation_model",
    "frequency_hz",
)

DISTANCE_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 6.0), (6.0, math.inf))


def _bucket(distance_m: float) -> str:
    for low, high in DISTANCE_BUCKETS:
        if low <= distance_m < high:
            return f"{low:g}-{high:g}m" if math.isfinite(high) else f"{low:g}m+"
    return "unknown"


def _load_arm(bank_root: Path) -> dict[str, Any]:
    manifest = RIRBankManifest.from_json(
        (bank_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    items: dict[str, dict[str, Any]] = {}
    for item in manifest.items:
        metadata = json.loads(
            (bank_root / item.metadata_path).read_text(encoding="utf-8")
        )
        items[metadata["m6"]["acoustic_space_id"] + "|" + item.item_id] = {
            "item": item,
            "metadata": metadata,
        }
    return {"root": bank_root, "manifest": manifest, "items": items}


def _realized_rows(arm: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in arm["items"].values():
        realized = record["metadata"].get("realized_acoustics")
        if not isinstance(realized, dict):
            continue
        for channel in realized.get("channels", ()):
            metrics = channel.get("metrics") or {}
            rows.append(
                {
                    "bucket": _bucket(float(channel.get("distance_m", float("nan")))),
                    "drr_db": metrics.get("drr_db"),
                    "c50_db": metrics.get("c50_db"),
                    "t30_s": (metrics.get("t30_s") or metrics.get("t20_s")),
                }
            )
    return rows


def _median(values) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(finite) if finite else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)
    summary = {}
    for bucket, bucket_rows in grouped.items():
        summary[bucket] = {
            "channels": len(bucket_rows),
            "drr_db": _median(r["drr_db"] for r in bucket_rows),
            "c50_db": _median(r["c50_db"] for r in bucket_rows),
            "t30_s": _median(r["t30_s"] for r in bucket_rows),
        }
    return summary


def _qc_counts(arm: dict[str, Any]) -> dict[str, Any]:
    per_split: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "pass": 0, "fail": 0, "pending": 0}
    )
    for record in arm["items"].values():
        item = record["item"]
        bucket = per_split[item.split]
        bucket["total"] += 1
        bucket[str(item.qc_status)] = bucket.get(str(item.qc_status), 0) + 1
    return {split: dict(counts) for split, counts in sorted(per_split.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
        help="Directory holding <backend>_bank and <backend>_release for both arms.",
    )
    parser.add_argument("--arm-a", default="pyroomacoustics")
    parser.add_argument("--arm-b", default="path-events-m4")
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Where to write the JSON summary (default: <pilot-root>/pilot_pair_report.json)",
    )
    parser.add_argument(
        "--skip-release-audit",
        action="store_true",
        help="Skip re-auditing both releases (the audit re-hashes every asset).",
    )
    args = parser.parse_args()

    report_path = args.output_report or (args.pilot_root / "pilot_pair_report.json")
    arms = {}
    for name in (args.arm_a, args.arm_b):
        bank_root = args.pilot_root / f"{name}_bank"
        if not (bank_root / "rir_bank_manifest.json").is_file():
            print(f"missing bank manifest for arm '{name}': {bank_root}", file=sys.stderr)
            return 1
        arms[name] = _load_arm(bank_root)

    a, b = arms[args.arm_a], arms[args.arm_b]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["both_arms_have_items"] = bool(a["items"]) and bool(b["items"])
    checks["same_item_count"] = len(a["items"]) == len(b["items"])
    checks["same_acoustic_space_set"] = set(a["items"]) == set(b["items"])
    details["item_counts"] = {args.arm_a: len(a["items"]), args.arm_b: len(b["items"])}

    shared = sorted(set(a["items"]) & set(b["items"]))
    mismatches = [
        {
            "key": key,
            "identity": identity,
            args.arm_a: a["items"][identity]["metadata"]["m6"][key],
            args.arm_b: b["items"][identity]["metadata"]["m6"][key],
        }
        for identity in shared
        for key in MATCHED_KEYS
        if a["items"][identity]["metadata"]["m6"][key]
        != b["items"][identity]["metadata"]["m6"][key]
    ]
    checks["per_item_identity_matches"] = not mismatches
    details["identity_mismatches"] = mismatches[:10]

    if shared:
        sample = shared[0]
        low_a = a["items"][sample]["metadata"]["bands"]["low"]
        low_b = b["items"][sample]["metadata"]["bands"]["low"]
        low_diff = {
            key: {args.arm_a: low_a.get(key), args.arm_b: low_b.get(key)}
            for key in SHARED_LOW_BAND_KEYS
            if low_a.get(key) != low_b.get(key)
        }
        checks["low_band_is_shared"] = not low_diff
        details["low_band"] = {
            "shared_configuration": {
                key: low_a.get(key) for key in SHARED_LOW_BAND_KEYS
            },
            "differences": low_diff,
        }
        high_a = a["items"][sample]["metadata"]["bands"]["high"]["backend"]
        high_b = b["items"][sample]["metadata"]["bands"]["high"]["backend"]
        checks["high_band_differs_as_intended"] = high_a != high_b
        details["high_band_backends"] = {args.arm_a: high_a, args.arm_b: high_b}

    details["qc"] = {name: _qc_counts(arm) for name, arm in arms.items()}
    checks["every_split_populated_in_both_arms"] = all(
        {"train", "validation", "test"} <= set(_qc_counts(arm)) for arm in arms.values()
    )

    # QC yield is an outcome, not part of the pairing contract.  If one backend
    # quarantines items the other does not, the arms are still matched and the
    # difference is precisely what the pilot is measuring — so it is reported
    # as a finding and must not suppress the comparison.
    quarantined = {
        name: {
            split: counts["fail"]
            for split, counts in _qc_counts(arm).items()
            if counts.get("fail", 0)
        }
        for name, arm in arms.items()
    }
    details["quarantined_by_split"] = quarantined
    details["quarantined_items"] = {
        name: sorted(
            record["item"].item_id
            for record in arm["items"].values()
            if record["item"].qc_status != "pass"
        )
        for name, arm in arms.items()
    }

    if not args.skip_release_audit:
        audits = {}
        for name in arms:
            release_root = args.pilot_root / f"{name}_release"
            audits[name] = (
                audit_m6_variant_release(release_root)
                if release_root.is_dir()
                else {"valid": False, "reason": "release directory missing"}
            )
        details["release_audit"] = {
            name: {"valid": bool(result.get("valid"))} for name, result in audits.items()
        }
        checks["both_releases_pass_audit"] = all(
            bool(result.get("valid")) for result in audits.values()
        )

    details["realized_acoustics_by_distance"] = {
        name: _summarize(_realized_rows(arm)) for name, arm in arms.items()
    }

    passed = all(checks.values())
    report = {
        "schema_version": "puresound.m6_pilot_pair_report.v1",
        "pilot_root": str(args.pilot_root),
        "arms": [args.arm_a, args.arm_b],
        "checks": checks,
        "details": details,
        "pairing_holds": passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for name, ok in checks.items():
        print(f"check\t{name}\t{'PASS' if ok else 'FAIL'}")

    print("\nQC yield (an outcome, not a pairing check):")
    for name in arms:
        counts = details["qc"][name]
        total = sum(row["total"] for row in counts.values())
        passed = sum(row["pass"] for row in counts.values())
        gap = details["quarantined_by_split"][name]
        print(
            f"  {name:<17} {passed}/{total} pass"
            + (f"   quarantined {gap} -> {details['quarantined_items'][name]}" if gap else "")
        )
    print()
    for name, summary in details["realized_acoustics_by_distance"].items():
        print(f"{name}:")
        print(f"  {'bucket':>10} {'ch':>6} {'DRR dB':>9} {'C50 dB':>9} {'T30 s':>9}")
        for bucket in sorted(summary, key=lambda key: float(key.split("-")[0].rstrip("m+"))):
            row = summary[bucket]
            def _fmt(value):
                return "     n/a" if value is None else f"{value:9.2f}"
            print(
                f"  {bucket:>10} {row['channels']:>6}"
                f" {_fmt(row['drr_db'])} {_fmt(row['c50_db'])} {_fmt(row['t30_s'])}"
            )
    print(f"\n{'PAIRING HOLDS' if passed else 'PAIRING BROKEN'} — report: {report_path}")
    if not passed:
        print("A comparison between these arms is not attributable to the backend.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
