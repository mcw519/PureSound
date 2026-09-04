"""P1 (amended) -- does the hinge measure the failure the shipped guard fixes?

v19c_round_design.md 3.3 asked for this on 450 Dawn utterances. The AMENDMENT
(2026-09-05) withdraws every Dawn waveform number: `speech` is not
waveform-consistent with `mix` (median |corr| 0.30 at the best lag, polarity
flips, lags into the thousands of samples), so a foreground projection on Dawn --
which is exactly what P and Q are -- is meaningless there. Dawn is ASR-only.

So the score is exercised on the construction the failure was *measured* on
instead: synthetic wrong-anchor rows built the way
`v19c_diagnostics/anchor_synthetic/anchor_synthetic_probe.py` builds its `utt`
arm -- 2 or 3 s of a DIFFERENT speaker's ref-active mix, then the utterance --
over the 200 utterances of `data_report/wer_set_moderate_test`, i.e. 400 rows.
Its own helpers are imported rather than re-implemented.

Both arms are scored on the SAME model output:
  (a) ungated v16 ep19,
  (b) that output through `puresound.system.onset_guard.OnsetGuard().apply(...)`
      at its shipped defaults.

Pass: median L_inherit drops >= 30% under (b). If it does not, the score does not
measure the failure -- that is the answer, and the score gets redesigned rather
than retuned.

    cd egs/voice_isolate && uv run python \
      benchmarks/probes/v19c_preflight/p1_guard_sensitivity.py --out <dir>
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
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "v19c_diagnostics/anchor_synthetic"),
)
import preflight_common as pc  # noqa: E402

import anchor_synthetic_probe as asp  # noqa: E402
from puresound.audio.vad import EnergyVADLabeler  # noqa: E402
from puresound.nnet.loss import AnchorInheritanceLoss  # noqa: E402
from puresound.system.onset_guard import OnsetGuard  # noqa: E402

ARMS = (10.0, 6.0)
PREFIX_SECONDS = (2.0, 3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=pc.CONFIG)
    ap.add_argument("--ckpt", default=pc.V16_EP19)
    ap.add_argument("--set-dir",
                    default=str(pc.RECIPE_DIR / "data_report/wer_set_moderate_test"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = asp.load_set(Path(args.set_dir), args.limit)
    material = [asp.build_material(rec) for rec in items]
    partner = asp.partner_index(items)
    print(f"{len(items)} utterances; {len(items) * len(PREFIX_SECONDS)} rows",
          flush=True)

    model, _, device = pc.load_model(args.config, args.ckpt, args.device)
    labeler = EnergyVADLabeler(frame_length=pc.FRAME, hop_length=pc.HOP)
    guard = OnsetGuard()
    print(f"[guard] defaults = {guard.as_manifest()}", flush=True)
    loss = AnchorInheritanceLoss(margin_db=0.0)

    rows: list[dict] = []
    fh = (out_dir / "rows.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for i, rec in enumerate(items):
        m = material[i]
        n_frames_20ms = m["nf"]
        mix = rec["mix"][: n_frames_20ms * asp.FRAME]
        ref = rec["ref"][: n_frames_20ms * asp.FRAME]
        speech = material[partner[i]]["speech"]      # the other talker's own frames

        for seconds in PREFIX_SECONDS:
            prefix = asp.tile_to(speech, seconds)
            x = np.concatenate([prefix, mix])
            ref_row = np.concatenate([np.zeros_like(prefix), ref])
            background = x - ref_row                  # prefix talker, then itf+noise

            dry = torch.from_numpy(x.astype(np.float32)).view(1, -1).to(device)
            with torch.no_grad():
                enhanced = model(dry)
            guarded = guard.apply(enhanced, dry, hop=pc.HOP)

            ref_t = torch.from_numpy(ref_row.astype(np.float32)).view(1, -1).to(device)
            noise_t = torch.from_numpy(
                background.astype(np.float32)
            ).view(1, -1).to(device)
            vad = labeler(ref_t.cpu()).view(1, -1).to(device)
            background_vad = labeler(noise_t.cpu()).view(1, -1).to(device)
            batch = {
                "consistency_noise": noise_t,
                "background_vad_target": background_vad,
                "n_interferers": torch.tensor(
                    [float(rec["n_interferers"])], device=device
                ),
            }

            record = {
                "id": rec["id"],
                "partner": items[partner[i]]["id"],
                "prefix_s": seconds,
                "rt60": rec["rt60"],
                "near_dist": rec["near_dist"],
                "n_interferers": rec["n_interferers"],
            }
            gain = guard.frame_gain(x, hop=pc.HOP)
            for arm, signal in (("ungated", enhanced), ("guarded", guarded)):
                with torch.no_grad():
                    scores = loss.row_scores(signal, ref_t, batch, vad)
                record[f"{arm}_eligible"] = bool(scores["eligible"][0])
                for key in ("pbar_on", "qbar_pre", "contrast"):
                    record[f"{arm}_{key}"] = float(scores[key][0])
                for margin in ARMS:
                    record[f"{arm}_hinge_M{margin:g}"] = max(
                        0.0, margin - float(scores["contrast"][0])
                    )
                record[f"{arm}_onset_frame"] = int(scores["onset_frame"][0])
                record[f"{arm}_n_pre_frames"] = int(scores["n_pre_frames"][0])
                record["onset_frame"] = int(scores["onset_frame"][0])
            # what the guard actually did, on the two frame sets the hinge reads
            onset = record["onset_frame"]
            n = min(len(gain), int(scores["n_frames"][0]))
            record["guard_gain_mean_pre"] = float(gain[:min(onset, n)].mean()) \
                if min(onset, n) > 0 else None
            record["guard_gain_mean_onset"] = float(
                gain[onset:min(onset + 50, n)].mean()
            ) if onset < n else None
            record["guard_dry_frac_pre"] = float(
                (gain[:min(onset, n)] > 0.5).mean()
            ) if min(onset, n) > 0 else None
            rows.append(record)
            fh.write(json.dumps(record) + "\n")
        if (i + 1) % 20 == 0:
            fh.flush()
            print(f"  {i + 1}/{len(items)} elapsed={time.time() - t0:.0f}s", flush=True)
    fh.close()
    print(f"DONE rows={len(rows)} elapsed={time.time() - t0:.0f}s", flush=True)

    report = summarize(rows)
    (out_dir / "p1.json").write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"\nreport -> {out_dir / 'p1.json'}")
    return 0


def _wilcoxon(deltas: np.ndarray):
    from scipy.stats import wilcoxon

    d = deltas[np.isfinite(deltas)]
    d = d[d != 0]
    if len(d) < 6:
        return float("nan"), 0, 0
    return float(wilcoxon(d).pvalue), int((d < 0).sum()), int((d > 0).sum())


def summarize(rows: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print(f"P1  does the score measure the guard's fix?   (n={len(rows)} rows)")
    print("=" * 78)
    eligible_u = sum(1 for r in rows if r["ungated_eligible"])
    eligible_g = sum(1 for r in rows if r["guarded_eligible"])
    print(f"eligible by construction: ungated {eligible_u}/{len(rows)}, "
          f"guarded {eligible_g}/{len(rows)}")
    keep = [r for r in rows if r["ungated_eligible"] and r["guarded_eligible"]]
    print(f"paired on {len(keep)} rows\n")

    report: dict = {"n_rows": len(rows), "n_paired": len(keep), "arms": {}}
    for key in ("pbar_on", "qbar_pre", "contrast"):
        u = pc.describe([r[f"ungated_{key}"] for r in keep])
        g = pc.describe([r[f"guarded_{key}"] for r in keep])
        delta = np.array([r[f"guarded_{key}"] - r[f"ungated_{key}"] for r in keep])
        p, down, up = _wilcoxon(delta)
        report["arms"][key] = {"ungated": u, "guarded": g,
                              "paired_delta": pc.describe(list(delta)),
                              "wilcoxon_p": p, "down": down, "up": up}
        print(f"{key:10s} ungated {pc.fmt(u)}")
        print(f"{'':10s} guarded {pc.fmt(g)}")
        print(f"{'':10s} paired delta med={np.median(delta):+8.3f}  p={p:.3g}  "
              f"down/up={down}/{up}")

    print()
    for margin in ARMS:
        name = f"hinge_M{margin:g}"
        u = np.array([r[f"ungated_{name}"] for r in keep])
        g = np.array([r[f"guarded_{name}"] for r in keep])
        med_u, med_g = float(np.median(u)), float(np.median(g))
        drop = (100.0 * (med_u - med_g) / med_u) if med_u > 0 else float("nan")
        p, down, up = _wilcoxon(g - u)
        passed = bool(med_u > 0 and drop >= 30.0)
        report[name] = {
            "median_ungated": med_u, "median_guarded": med_g,
            "median_drop_pct": drop, "wilcoxon_p": p, "down": down, "up": up,
            "pass_30pct_drop": passed,
            "ungated": pc.describe(list(u)), "guarded": pc.describe(list(g)),
        }
        print(f"L_inherit M={margin:<4g} median ungated={med_u:8.3f}  "
              f"guarded={med_g:8.3f}  drop={drop:+7.2f}%  p={p:.3g}  "
              f"-> {'PASS' if passed else 'FAIL'}")

    gains = {
        "guard_gain_mean_pre": pc.describe([r["guard_gain_mean_pre"] for r in rows]),
        "guard_gain_mean_onset": pc.describe(
            [r["guard_gain_mean_onset"] for r in rows]),
        "guard_dry_frac_pre": pc.describe([r["guard_dry_frac_pre"] for r in rows]),
    }
    print("\nwhat the guard did (gain 1.0 = dry / model not consulted):")
    for name, stats in gains.items():
        print(f"  {name:24s} {pc.fmt(stats)}")
    report["guard_behaviour"] = gains

    by_prefix = {}
    print("\nby prefix length:")
    for seconds in sorted({r["prefix_s"] for r in rows}):
        subset = [r for r in keep if r["prefix_s"] == seconds]
        if not subset:
            continue
        u = np.array([r["ungated_hinge_M10"] for r in subset])
        g = np.array([r["guarded_hinge_M10"] for r in subset])
        by_prefix[f"{seconds:g}"] = {
            "n": len(subset),
            "median_ungated_M10": float(np.median(u)),
            "median_guarded_M10": float(np.median(g)),
            "median_ungated_contrast": float(
                np.median([r["ungated_contrast"] for r in subset])),
            "median_guarded_contrast": float(
                np.median([r["guarded_contrast"] for r in subset])),
        }
        print(f"  {seconds:g}s n={len(subset):4d} M10 ungated={np.median(u):8.3f} "
              f"guarded={np.median(g):8.3f}")
    report["by_prefix"] = by_prefix

    verdict = ("PASS" if all(report[f"hinge_M{m:g}"]["pass_30pct_drop"] for m in ARMS)
               else "FAIL")
    print("\n" + "-" * 78)
    print(f"P1 VERDICT: {verdict}")
    print("-" * 78)
    report["verdict"] = verdict
    return report


if __name__ == "__main__":
    raise SystemExit(main())
