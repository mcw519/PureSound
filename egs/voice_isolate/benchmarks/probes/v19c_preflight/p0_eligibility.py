"""P0 -- does the hinge fire on the rows the training set actually contains?

v19c_round_design.md 3.3, Decision point B. On >= 600 rows of the REAL v19c
train loader (seed 7), with v16 ep19 loaded, report:

  * the eligible share at the >= 1 s interferer-before-onset rule and at the
    pre-registered >= 0.5 s fallback,
  * the MEDIAN HINGE over eligible rows at init, for both arms (M = 10 and 6),
  * both split by length bucket and by row type.

Kill: eligible share < 3% at the >= 0.5 s rule, OR the median hinge over
eligible rows is 0 (the term would be training a tail, not a behaviour).

Nothing is tuned here. The hinge for any margin M is `relu(M - contrast)` with
`contrast = Pbar_on - Qbar_pre`, so both arms come out of one GPU pass.

    cd egs/voice_isolate && uv run python \
      benchmarks/probes/v19c_preflight/p0_eligibility.py \
      --n-rows 600 --seed 7 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_common as pc  # noqa: E402

from puresound.nnet.loss import AnchorInheritanceLoss  # noqa: E402

ARMS = (10.0, 6.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=pc.CONFIG)
    ap.add_argument("--ckpt", default=pc.V16_EP19)
    ap.add_argument("--n-rows", type=int, default=600)
    ap.add_argument("--max-batches", type=int, default=400)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.seed_everything(args.seed)

    model, _, device = pc.load_model(args.config, args.ckpt, args.device)
    train, _valid, _recipe = pc.build_loaders(args.config, args.num_workers)
    # margin 0 -> `hinge` is reported per arm from `contrast`, not from this
    # instance, so the instance's own margin never enters the numbers.
    loss = AnchorInheritanceLoss(margin_db=0.0)

    rows: list[dict] = []
    fh = (out_dir / "rows.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for bi, batch in enumerate(train):
        if bi >= args.max_batches or len(rows) >= args.n_rows:
            break
        noisy = batch["noisy_speech"].to(device)
        clean = batch["clean_speech"].to(device)
        with torch.no_grad():
            enhanced = model(noisy)
        on_device = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        with torch.no_grad():
            scores = loss.row_scores(
                enhanced, clean, on_device, batch["vad_target"].to(device)
            )
        for r in range(clean.shape[0]):
            record = {
                "batch": bi,
                "row": r,
                "batch_rows": int(clean.shape[0]),
                "row_seconds": pc.bucket(batch, r),
                "row_type": pc.row_type(batch, r),
                "turn_taking": pc.is_turn_taking(batch, r),
                "mix_mode": pc.MIX_MODE_NAMES.get(pc.scalar(batch, "mix_mode", r)),
                "n_interferers": pc.scalar(batch, "n_interferers", r),
                "target_absent": pc.scalar(batch, "target_absent", r),
                "has_background_speech": pc.scalar(batch, "has_background_speech", r),
                "realized_sir": pc.scalar(batch, "realized_speech_sir", r),
                "noise_snr": pc.scalar(batch, "noise_snr", r),
                "eligible": bool(scores["eligible"][r]),
                "eligible_fallback": bool(scores["eligible_fallback"][r]),
                "onset_frame": int(scores["onset_frame"][r]),
                "pre_interferer_frames": float(scores["pre_interferer_frames"][r]),
                "n_onset_frames": int(scores["n_onset_frames"][r]),
                "n_pre_frames": int(scores["n_pre_frames"][r]),
                "n_active_after_onset": float(scores["n_active_after_onset"][r]),
                "n_frames": int(scores["n_frames"][r]),
                "pbar_on": float(scores["pbar_on"][r]),
                "qbar_pre": float(scores["qbar_pre"][r]),
                "contrast": float(scores["contrast"][r]),
            }
            for margin in ARMS:
                record[f"hinge_M{margin:g}"] = max(0.0, margin - record["contrast"])
            # Each eligibility clause on its own, so a low share says WHICH
            # clause the distribution fails rather than just that it fails.
            record["clause_has_interferer"] = (record["n_interferers"] or 0.0) >= 1.0
            record["clause_onset_ge_1s"] = record["onset_frame"] >= 100
            record["clause_50_active_after"] = record["n_active_after_onset"] >= 50
            record["clause_pre_itf_ge_1s"] = record["pre_interferer_frames"] >= 100
            record["clause_pre_itf_ge_0.5s"] = record["pre_interferer_frames"] >= 50
            record["clause_pre_itf_gt_0"] = record["pre_interferer_frames"] > 0
            record["clause_B_nonempty"] = record["n_pre_frames"] > 0
            rows.append(record)
            fh.write(json.dumps(record) + "\n")
        fh.flush()
        if bi % 5 == 0:
            print(f"batch {bi} rows={len(rows)} elapsed={time.time() - t0:.0f}s",
                  flush=True)
    fh.close()
    rows = rows[: args.n_rows]
    print(f"DONE rows={len(rows)} elapsed={time.time() - t0:.0f}s", flush=True)

    report = summarize(rows)
    (out_dir / "p0.json").write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"\nreport -> {out_dir / 'p0.json'}")
    return 0


def _share(rows, key) -> tuple[int, int, float]:
    hit = sum(1 for r in rows if r[key])
    return hit, len(rows), (100.0 * hit / max(len(rows), 1))


def wilson(hit: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI on a share. Printed beside every share because n is the whole
    story here: the eligible subset is a few percent of the rows."""
    if total == 0:
        return float("nan"), float("nan")
    p = hit / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def boot_median(values, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI on a median. The round's metric policy reports medians;
    a median over 30-odd rows needs its interval stated."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    medians = np.median(arr[draws], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _cell(rows, gate="eligible") -> dict:
    eligible = [r for r in rows if r[gate]]
    out = {"n": len(rows)}
    for name in ("eligible", "eligible_fallback"):
        hit, total, pct = _share(rows, name)
        out[name] = f"{hit}/{total} {pct:.1f}%"
    if eligible:
        out["contrast"] = pc.describe([r["contrast"] for r in eligible])
        out["pbar_on"] = pc.describe([r["pbar_on"] for r in eligible])
        out["qbar_pre"] = pc.describe([r["qbar_pre"] for r in eligible])
        for margin in ARMS:
            values = [r[f"hinge_M{margin:g}"] for r in eligible]
            out[f"hinge_M{margin:g}"] = pc.describe(values)
            out[f"nonzero_M{margin:g}"] = (
                f"{sum(1 for v in values if v > 0)}/{len(values)} "
                f"{100.0 * sum(1 for v in values if v > 0) / len(values):.1f}%"
            )
    return out


def summarize(rows: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print(f"P0  eligibility + median hinge at init   (n={len(rows)} train rows)")
    print("=" * 78)

    for name, label in (("eligible", ">=1.0 s interferer-before-onset (the rule)"),
                        ("eligible_fallback", ">=0.5 s (the pre-registered fallback)")):
        hit, total, pct = _share(rows, name)
        lo, hi = wilson(hit, total)
        print(f"{label:52s}: {hit:5d}/{total:<5d} {pct:5.2f}%  "
              f"95% CI [{lo:.2f}, {hi:.2f}]%")

    overall = _cell(rows)
    fb = _cell(rows, gate="eligible_fallback")
    print()
    for gate, cell in (("eligible @>=1s", overall), ("eligible @>=0.5s", fb)):
        if "contrast" not in cell:
            print(f"{gate:18s}  (no rows)")
            continue
        print(f"{gate:18s}  contrast  {pc.fmt(cell['contrast'])}")
        print(f"{'':18s}  Pbar_on   {pc.fmt(cell['pbar_on'])}")
        print(f"{'':18s}  Qbar_pre  {pc.fmt(cell['qbar_pre'])}")
        for margin in ARMS:
            print(f"{'':18s}  hinge M={margin:<4g}{pc.fmt(cell[f'hinge_M{margin:g}'])}"
                  f"   >0 in {cell[f'nonzero_M{margin:g}']}")

    by_bucket, by_type, by_cross = {}, {}, {}
    print("\nby length bucket:")
    header = (f"{'sec':>7} {'n':>5} {'elig>=1s':>16} {'elig>=0.5s':>16} "
              f"{'med contrast':>13} {'med hinge M10':>14} {'med hinge M6':>13}")
    print(header)
    for seconds in sorted({r["row_seconds"] for r in rows}):
        subset = [r for r in rows if r["row_seconds"] == seconds]
        by_bucket[f"{seconds:g}"] = _cell(subset)
        _print_line(f"{seconds:7.2f}", subset)
    print("\nby row type (turn-taking reported as its own bucket, as pre-registered):")
    print(header.replace("    sec", "    typ"))
    kinds = sorted({r["row_type"] for r in rows}) + ["turn-taking"]
    for kind in kinds:
        subset = ([r for r in rows if r["turn_taking"]] if kind == "turn-taking"
                  else [r for r in rows if r["row_type"] == kind])
        by_type[kind] = _cell(subset)
        _print_line(f"{kind:>7s}", subset)
    print("\nrow type x turn-taking (the same rows, cross-tabulated):")
    print(header.replace("    sec", "    typ"))
    for kind in sorted({r["row_type"] for r in rows}):
        for flag in (False, True):
            subset = [r for r in rows
                      if r["row_type"] == kind and r["turn_taking"] == flag]
            if not subset:
                continue
            tag = f"{kind[:9]}{'+tt' if flag else ''}"
            by_cross[tag] = _cell(subset)
            _print_line(f"{tag:>7s}", subset)

    # ---- which clause the distribution fails -----------------------------
    clauses = [k for k in rows[0] if k.startswith("clause_")] if rows else []
    print("\neligibility clause by clause (each on its own, over all rows):")
    clause_shares = {}
    for name in clauses:
        hit, total, pct = _share(rows, name)
        clause_shares[name] = f"{hit}/{total} {pct:.2f}%"
        print(f"  {name:26s}: {hit:5d}/{total:<5d} {pct:6.2f}%")

    # ---- the kill criteria, evaluated ------------------------------------
    fb_hit, fb_total, fallback_pct = _share(rows, "eligible_fallback")
    fb_lo, fb_hi = wilson(fb_hit, fb_total)
    eligible = [r for r in rows if r["eligible"]]
    medians, median_ci, nonzero = {}, {}, {}
    for margin in ARMS:
        key = f"M{margin:g}"
        values = [r[f"hinge_M{margin:g}"] for r in eligible]
        medians[key] = float(np.median(values)) if values else float("nan")
        median_ci[key] = boot_median(values)
        hit = sum(1 for v in values if v > 0)
        nonzero[key] = {"hit": hit, "n": len(values),
                        "pct": 100.0 * hit / max(len(values), 1),
                        "ci": wilson(hit, len(values))}
    kill_share = fallback_pct < 3.0
    kill_median = bool(eligible) and all(m == 0.0 for m in medians.values())
    print("\n" + "-" * 78)
    print(f"KILL 1  eligible share at >=0.5 s < 3%      : {fallback_pct:.2f}% "
          f"CI [{fb_lo:.2f}, {fb_hi:.2f}]%  -> {'KILL' if kill_share else 'pass'}")
    print("KILL 2  median hinge over eligible rows == 0: "
          + ", ".join(f"{k}={v:.3f} CI [{median_ci[k][0]:.3f}, {median_ci[k][1]:.3f}]"
                      for k, v in medians.items())
          + f"  -> {'KILL' if kill_median else 'pass'}")
    for key, cell in nonzero.items():
        print(f"        rows where the hinge is >0 ({key:>3s})   : "
              f"{cell['hit']}/{cell['n']} {cell['pct']:.1f}% "
              f"CI [{cell['ci'][0]:.1f}, {cell['ci'][1]:.1f}]%")
    verdict = "KILL" if (kill_share or kill_median or not eligible) else "PASS"
    print(f"P0 VERDICT: {verdict}")
    print("-" * 78)

    return {
        "n_rows": len(rows),
        "overall_eligible": overall,
        "overall_eligible_fallback": fb,
        "by_length_bucket": by_bucket,
        "by_row_type": by_type,
        "by_row_type_x_turn_taking": by_cross,
        "eligibility_clauses": clause_shares,
        "kill_eligible_share_lt_3pct": kill_share,
        "eligible_fallback_pct": fallback_pct,
        "eligible_fallback_ci_pct": [fb_lo, fb_hi],
        "median_hinge": medians,
        "median_hinge_ci": median_ci,
        "hinge_nonzero_share": nonzero,
        "kill_median_hinge_zero": kill_median,
        "verdict": verdict,
    }


def _print_line(tag: str, subset: list[dict]) -> None:
    if not subset:
        print(f"{tag} {0:5d}")
        return
    e1 = _share(subset, "eligible")
    e5 = _share(subset, "eligible_fallback")
    eligible = [r for r in subset if r["eligible"]]

    def med(key):
        return (f"{np.median([r[key] for r in eligible]):+.2f}" if eligible else "--")

    print(f"{tag} {len(subset):5d} {f'{e1[0]}/{e1[1]} {e1[2]:.1f}%':>16} "
          f"{f'{e5[0]}/{e5[1]} {e5[2]:.1f}%':>16} {med('contrast'):>13} "
          f"{med('hinge_M10'):>14} {med('hinge_M6'):>13}")


if __name__ == "__main__":
    raise SystemExit(main())
