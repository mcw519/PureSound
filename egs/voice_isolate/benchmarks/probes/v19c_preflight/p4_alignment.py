"""P4 -- alignment on speed-perturbed 30 s rows.

v19c_round_design.md 3.3: the loss's onset index must coincide with
`vad_target`'s within 2 frames on a speed-perturbed 30 s row.

Three numbers per row, all on the 400/160 label grid:

  loss_onset      the index `AnchorInheritanceLoss` frames on
  label_onset     `argmax(batch["vad_target"])`
  waveform_onset  the first active frame of an EnergyVADLabeler run on
                  `clean_speech` -- the post-device-chain target itself

The first pair is the pre-registered gate. The second pair is the property that
gate exists to protect: `vad_reference` is cloned AFTER the speed block in
`task/ns.py`, so the label is on the post-speed grid. Had it been a pre-speed
snapshot, a +-5% speed draw would put the label up to ~150 frames from the
waveform on a 30 s row, and the loss's 0.5 s onset window would be measuring the
wrong half second.

Speed perturbation is not emitted as a scalar, so the provably-perturbed subset
is identified by length: a row shorter than its bucket was SPED UP (the row is
cropped to the bucket, so a slowed row lands back at the cap). Reported
separately, with the implied speed factor.

No model is loaded: the onset index depends only on `vad_target`, so
`clean_speech` stands in for `enhanced` and this runs on CPU.

    cd egs/voice_isolate && uv run python \
      benchmarks/probes/v19c_preflight/p4_alignment.py --out <dir>
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

from puresound.audio.vad import EnergyVADLabeler  # noqa: E402
from puresound.nnet.loss import AnchorInheritanceLoss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=pc.CONFIG)
    ap.add_argument("--n-rows", type=int, default=40)
    ap.add_argument("--min-seconds", type=float, default=20.0,
                    help="only the long bucket: a +-5% timing error is invisible "
                         "on a 3 s row and unmissable on a 30 s one")
    ap.add_argument("--max-batches", type=int, default=400)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.seed_everything(args.seed)

    train, _valid, _recipe = pc.build_loaders(args.config, args.num_workers)
    labeler = EnergyVADLabeler(frame_length=pc.FRAME, hop_length=pc.HOP)
    loss = AnchorInheritanceLoss(margin_db=0.0)

    rows: list[dict] = []
    t0 = time.time()
    for bi, batch in enumerate(train):
        if bi >= args.max_batches or len(rows) >= args.n_rows:
            break
        seconds = pc.bucket(batch, 0)
        if seconds < args.min_seconds:
            continue
        clean = batch["clean_speech"]
        vad = batch["vad_target"]
        with torch.no_grad():
            scores = loss.row_scores(clean, clean, batch, vad)
        for r in range(clean.shape[0]):
            n_samples = int(batch["length"].reshape(-1)[r].item())
            row_seconds = n_samples / pc.SR
            if clean[r, :n_samples].abs().amax().item() == 0.0:
                continue                       # target-absent row: no onset to align
            waveform = labeler(clean[r, :n_samples]).view(-1)
            label = vad[r].view(-1)
            n = min(len(waveform), len(label), int(scores["n_frames"][r]))
            if label[:n].sum() == 0 or waveform[:n].sum() == 0:
                continue
            record = {
                "batch": bi,
                "row": r,
                "bucket_seconds": seconds,
                "row_seconds": round(row_seconds, 4),
                # 30 s / 28.57 s = 1.05: the row was sped up by this factor.
                "implied_speed": round(seconds / row_seconds, 4),
                "speed_perturbed": bool(row_seconds < seconds - 0.05),
                "row_type": pc.row_type(batch, r),
                "loss_onset": int(scores["onset_frame"][r]),
                "label_onset": int(label[:n].argmax()),
                "waveform_onset": int(waveform[:n].argmax()),
                "n_frames": int(n),
            }
            record["d_loss_label"] = abs(record["loss_onset"]
                                         - record["label_onset"])
            record["d_label_waveform"] = abs(record["label_onset"]
                                             - record["waveform_onset"])
            rows.append(record)
        print(f"batch {bi} sec={seconds:g} rows={len(rows)} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    rows = rows[: args.n_rows]

    report = summarize(rows)
    (out_dir / "p4.json").write_text(
        json.dumps({"rows": rows, **report}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nreport -> {out_dir / 'p4.json'}")
    return 0


def summarize(rows: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print(f"P4  onset alignment on long rows   (n={len(rows)})")
    print("=" * 78)
    if not rows:
        print("  (no rows)")
        return {"verdict": "NO DATA"}

    perturbed = [r for r in rows if r["speed_perturbed"]]
    print(f"provably speed-perturbed (row shorter than its bucket): "
          f"{len(perturbed)}/{len(rows)}")
    if perturbed:
        speeds = sorted({r["implied_speed"] for r in perturbed})
        print(f"  implied speed factors seen: {speeds}")

    out: dict = {"n_rows": len(rows), "n_speed_perturbed": len(perturbed)}
    for name, subset in (("all long rows", rows), ("speed-perturbed", perturbed)):
        if not subset:
            continue
        d1 = np.array([r["d_loss_label"] for r in subset])
        d2 = np.array([r["d_label_waveform"] for r in subset])
        print(f"\n{name} (n={len(subset)}):")
        print(f"  |loss_onset - label_onset|    max={d1.max():4d}  "
              f"median={np.median(d1):5.1f}  >2 in {(d1 > 2).sum()}/{len(d1)}")
        print(f"  |label_onset - waveform_onset| max={d2.max():4d}  "
              f"median={np.median(d2):5.1f}  >2 in {(d2 > 2).sum()}/{len(d2)}")
        out[name.replace(" ", "_").replace("-", "_")] = {
            "n": len(subset),
            "d_loss_label_max": int(d1.max()),
            "d_loss_label_median": float(np.median(d1)),
            "d_loss_label_over_2": int((d1 > 2).sum()),
            "d_label_waveform_max": int(d2.max()),
            "d_label_waveform_median": float(np.median(d2)),
            "d_label_waveform_over_2": int((d2 > 2).sum()),
        }

    gate = perturbed or rows
    worst = max(r["d_loss_label"] for r in gate)
    passed = worst <= 2
    print("\n" + "-" * 78)
    print(f"P4  worst |loss_onset - label_onset| on the gated subset: {worst} frames "
          f"(<= 2 required) -> {'PASS' if passed else 'FAIL'}")
    print(f"P4 VERDICT: {'PASS' if passed else 'FAIL'}")
    print("-" * 78)
    out["worst_d_loss_label"] = int(worst)
    out["verdict"] = "PASS" if passed else "FAIL"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
