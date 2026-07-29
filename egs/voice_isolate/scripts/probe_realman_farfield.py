"""Suppression of real recorded speech vs annotated distance (RealMAN corpus).

Companion to ``probe_retransmitted_farfield.py`` (VOiCES). RealMAN val static
recordings are per-utterance, per-channel mono flacs (48 kHz) with an annotated
speaker->array distance in ``val_static_source_location.csv`` (sentinel -10000 for
missing). The direct-path reference (dp_speech, aligned sample-for-sample with CH0)
gives clean speech-active spans, so scene noise does not pollute span detection.

Reads reduction on the model output over dp-active spans, bucketed by distance.
Target behaviour: <1m keep (~0 dB), >1m suppress (very negative).

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/probe_realman_farfield.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
        --realman-val /work/any_exp_link/puresound_exp/real_e2e_corpora/realman/val \
        --device cuda --per-bucket 40
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_retransmitted_farfield import (  # noqa: E402
    active_spans,
    load_model,
    quantiles,
    reduction_db,
)

from puresound.audio.io import AudioIO  # noqa: E402

BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99.0)]


def bucket_of(dist: float):
    for lo, hi in BUCKETS:
        if lo <= dist < hi:
            return (lo, hi)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--realman-val", required=True)
    parser.add_argument("--channel", default="CH0")
    parser.add_argument("--per-bucket", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    val_root = Path(args.realman_val)
    rows_by_bucket: dict[tuple, list[dict]] = defaultdict(list)
    with open(val_root / "val_static_source_location.csv") as f:
        for row in csv.DictReader(f):
            dist = float(row["distance"])
            if not 0.0 < dist < 50.0:  # -10000 sentinel / bad rows
                continue
            b = bucket_of(dist)
            if b is not None:
                rows_by_bucket[b].append({"filename": row["filename"], "distance": dist})

    rng = random.Random(args.seed)
    print("# static val rows with valid distance per bucket:")
    for b in BUCKETS:
        print(f"#   {b[0]:.0f}-{b[1]:.0f}m: {len(rows_by_bucket[b])}")

    device = torch.device(args.device)
    model = load_model(args.config_path, args.ckpt, device)

    results = []
    for b in BUCKETS:
        pool = rows_by_bucket[b]
        picks = pool if len(pool) <= args.per_bucket else rng.sample(pool, args.per_bucket)
        for item in picks:
            rel = item["filename"]
            rel = rel[4:] if rel.startswith("val/") else rel  # paths in csv start with val/
            ma_path = val_root / rel.replace(".flac", f"_{args.channel}.flac")
            dp_path = val_root / rel.replace("ma_noisy_speech", "dp_speech")
            if not (ma_path.is_file() and dp_path.is_file()):
                continue
            ma, sr = AudioIO.open(f_path=str(ma_path), target_lvl=None, resample_to=16000)
            dp, _ = AudioIO.open(f_path=str(dp_path), target_lvl=None, resample_to=16000)
            ma, dp = ma.view(1, -1), dp.view(1, -1)
            spans = active_spans(dp, sr, min_span_sec=0.2)
            if sum(e - s for s, e in spans) < 0.5 * sr:
                continue
            with torch.no_grad():
                enh = model(ma.to(device)).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
            scene = rel.split("/")[1]
            results.append(
                {
                    "file": ma_path.name,
                    "scene": scene,
                    "distance_m": item["distance"],
                    "bucket": f"{b[0]:.0f}-{b[1]:.0f}m",
                    "active_reduction_db": reduction_db(enh, ma, spans),
                    "active_sec": sum(e - s for s, e in spans) / sr,
                }
            )
        done = [r for r in results if r["bucket"] == f"{b[0]:.0f}-{b[1]:.0f}m"]
        if done:
            _, med, _ = quantiles([r["active_reduction_db"] for r in done])
            print(f"# bucket {b[0]:.0f}-{b[1]:.0f}m done: n={len(done)} median={med:.2f} dB", flush=True)

    if args.out_json:
        Path(args.out_json).write_text("\n".join(json.dumps(r) for r in results) + "\n")

    print()
    print("bucket\tn\tp25_dB\tmedian_dB\tp75_dB\tscenes")
    for b in BUCKETS:
        key = f"{b[0]:.0f}-{b[1]:.0f}m"
        rs = [r for r in results if r["bucket"] == key]
        if not rs:
            continue
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in rs])
        n_scenes = len({r["scene"] for r in rs})
        print(f"{key}\t{len(rs)}\t{p25:7.2f}\t{med:7.2f}\t{p75:7.2f}\t{n_scenes}")
    far = [r for r in results if r["distance_m"] > 1.0]
    near = [r for r in results if r["distance_m"] <= 1.0]
    if far:
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in far])
        print(f"\n# ALL >1m (suppress targets): n={len(far)} median={med:.2f} dB")
    if near:
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in near])
        print(f"# ALL <=1m (keep targets):   n={len(near)} median={med:.2f} dB (should be ~0)")


if __name__ == "__main__":
    main()
