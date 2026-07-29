#!/usr/bin/env python
"""Measure and compare the acoustics of RIR banks, side by side.

`inspect_bank.py` reports what the metadata CLAIMS; this measures what the WAVs
actually contain, so simulated and measured banks can be compared on the same
footing before a bank goes into training. Per sampled channel:

    DRR    direct(2.5 ms)-to-reverberant ratio (the training pipeline's cue)
    C50    early(50 ms)/late energy ratio (the 'early' target boundary)
    T30    RT60 from a Schroeder-integration linear fit over [-5, -35] dB
    tilt   RIR magnitude slope 200 Hz - 4 kHz, dB/octave (coloration proxy)

Buckets by distance and prints one block per bank, so a distribution shift
between banks (or between a bank and the measured-RIR bank built from real
rooms) is visible as a table instead of an opinion.

Usage:
  uv run python egs/rir_generation/compare_bank_acoustics.py \\
      synthetic=exp/hybrid_rir_16k_levels/wide \\
      measured=exp/real_rir_16k_train_view/all --per-bank 300
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir_bank import PreGeneratedRoomBank  # noqa: E402

BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 6.0), (6.0, 99.0)]


def c50_db(rir: torch.Tensor, sr: int) -> float:
    x = rir.reshape(-1)
    pk = int(x.abs().argmax())
    cut = pk + int(0.05 * sr)
    early = float(x[pk:cut].square().sum())
    late = float(x[cut:].square().sum())
    if late <= 0:
        return float("inf")
    return 10.0 * math.log10(max(early, 1e-12) / late)


def t30_s(rir: torch.Tensor, sr: int) -> float:
    """RT60 via Schroeder backward integration, linear fit on [-5, -35] dB."""
    x = rir.reshape(-1).double()
    pk = int(x.abs().argmax())
    x = x[pk:]
    edc = torch.flip(torch.cumsum(torch.flip(x.square(), [0]), 0), [0])
    edc_db = 10.0 * torch.log10(edc / edc[0].clamp_min(1e-20) + 1e-20)
    idx = torch.where((edc_db <= -5.0) & (edc_db >= -35.0))[0]
    if idx.numel() < sr // 100:
        return float("nan")
    t = idx.double() / sr
    y = edc_db[idx]
    slope = float(((t - t.mean()) * (y - y.mean())).sum() / (t - t.mean()).square().sum())
    if slope >= 0:
        return float("nan")
    return -60.0 / slope


def tilt_db_per_oct(rir: torch.Tensor, sr: int) -> float:
    """Log-magnitude slope 200 Hz..4 kHz in dB/octave."""
    x = rir.reshape(-1)
    mag = torch.fft.rfft(x, 2 ** int(math.ceil(math.log2(x.numel())))).abs()
    freqs = torch.linspace(0, sr / 2, mag.numel())
    m = (freqs >= 200) & (freqs <= 4000)
    lf = torch.log2(freqs[m])
    y = 20.0 * torch.log10(mag[m].clamp_min(1e-12))
    slope = float(((lf - lf.mean()) * (y - y.mean())).sum() / (lf - lf.mean()).square().sum())
    return slope


def bucket_label(d: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}m" if hi < 99 else f"{lo:g}m+"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banks", nargs="+", metavar="TAG=PATH",
                    help="banks/views to compare, as tag=path")
    ap.add_argument("--per-bank", type=int, default=300, help="channels sampled per bank")
    ap.add_argument("--drr-window-ms", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    stats: dict[str, dict[str, list[dict]]] = {}
    for spec in args.banks:
        tag, _, path = spec.partition("=")
        if not tag or not path:
            raise SystemExit(f"bad bank spec (want TAG=PATH): {spec}")
        bank = PreGeneratedRoomBank(folder=path, drr_window_ms=args.drr_window_ms)
        rows: list[dict] = []
        for _ in range(args.per_bank):
            scene = bank.sample_scene()
            rir, meta, sr = bank.select_channel(scene, source_role="any")
            d = float(meta["source_receiver_distance"])
            rows.append({
                "d": d,
                "drr": float(meta["drr_db"]),
                "c50": c50_db(rir, sr),
                "t30": t30_s(rir, sr),
                "tilt": tilt_db_per_oct(rir, sr),
                "rt60_meta": meta.get("rt60"),
            })
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_bucket[bucket_label(r["d"])].append(r)
        stats[tag] = by_bucket
        print(f"# {tag}: {len(rows)} channels from {path} ({len(bank)} rooms)")

    def med(vals):
        v = [x for x in vals if x == x and not math.isinf(x)]
        return float(np.median(v)) if v else float("nan")

    print()
    header = "bank\tbucket\tn\tDRR_dB\tC50_dB\tT30_s\ttilt_dB/oct\trt60_meta"
    print(header)
    for tag, by_bucket in stats.items():
        for lo, hi in BUCKETS:
            b = f"{lo:g}-{hi:g}m" if hi < 99 else f"{lo:g}m+"
            rs = by_bucket.get(b, [])
            if not rs:
                continue
            rt = med([r["rt60_meta"] for r in rs if r["rt60_meta"] is not None])
            print(f"{tag}\t{b}\t{len(rs)}\t{med([r['drr'] for r in rs]):6.2f}"
                  f"\t{med([r['c50'] for r in rs]):6.2f}\t{med([r['t30'] for r in rs]):5.2f}"
                  f"\t{med([r['tilt'] for r in rs]):6.2f}\t{rt if rt == rt else float('nan'):5.2f}")
    print("\n# T30 far above rt60_meta = the metadata undersells the tail (or the fit failed);")
    print("# a DRR/C50/tilt shift between banks at the same distance is a domain shift the")
    print("# training pipeline will inherit.")


if __name__ == "__main__":
    main()
