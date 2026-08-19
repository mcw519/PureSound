"""Can DRR alone separate presence in the training bank?

The v11 heads must learn "is a near user talking". If the bank's near-channel and
far-channel DRR distributions barely overlap, DRR is a sufficient statistic for
that label and the heads will learn DRR -- inheriting the reverberation confound
by construction (DRR falls with distance AND with RT60). Pre-registered rule:

    best single-DRR-threshold balanced accuracy on (near vs far channel)
      <= 0.90  ->  enough ambiguous mass; v11 is a config-only change
      >= 0.97  ->  DRR is sufficient; mix in the high bank (rt60 0.85-1.5)

DRR computed from the RIR wavs with the repo's own compute_drr_db at the recipe's
2.5 ms window. Uniform sample over items, so origins are represented at their
sampling weight (~22% measured).
"""
import json, pathlib, random, sys
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
from puresound.audio.impulse_response import compute_drr_db

N_SAMPLE = 6000
NEAR = {"near_0", "near_1"}
FAR = {"far_0", "far_1", "far_2"}


def collect(folder, n_sample, seed=0):
    """Items live either flat under items/ (merged views) or per-room (raw banks)."""
    root = pathlib.Path(folder)
    items = sorted((root / "items").glob("*.json")) or sorted(root.rglob("*.json"))
    random.Random(seed).shuffle(items)
    rows = []
    for p in items[:n_sample]:
        meta = json.loads(p.read_text())
        if "scene" not in meta:
            continue
        scene = meta["scene"]
        rt60 = float(scene.get("rt60", float("nan")))
        origin = "real" if p.name.startswith("real_") else "sim"
        try:
            wav, sr = sf.read(p.with_suffix(".wav"))
        except Exception:
            continue
        if wav.ndim == 1:
            wav = wav[:, None]
        for ch in scene["channel_map"]:
            lab = ch["label"]
            if lab not in NEAR | FAR:
                continue
            rir = torch.from_numpy(np.ascontiguousarray(wav[:, ch["channel"]])).float()
            drr = compute_drr_db(rir, sr, direct_window_ms=2.5)
            rows.append((origin, rt60, lab in NEAR, float(ch.get("distance_m", np.nan)),
                         float(drr)))
    return rows


def best_threshold(near_drr, far_drr):
    """Best single threshold: near above, far below."""
    lo = min(near_drr.min(), far_drr.min()); hi = max(near_drr.max(), far_drr.max())
    ts = np.linspace(lo, hi, 2000)
    accs = [(0.5 * ((near_drr >= t).mean() + (far_drr < t).mean()), t) for t in ts]
    return max(accs)


def report(tag, rows):
    origin = np.array([r[0] for r in rows]); rt60 = np.array([r[1] for r in rows])
    is_near = np.array([r[2] for r in rows]); dist = np.array([r[3] for r in rows])
    drr = np.array([r[4] for r in rows])
    ok = np.isfinite(drr)
    origin, rt60, is_near, dist, drr = origin[ok], rt60[ok], is_near[ok], dist[ok], drr[ok]
    nd, fd = drr[is_near], drr[~is_near]
    print(f"\n===== {tag}:  {is_near.sum()} near / {(~is_near).sum()} far channels =====")
    for name, v in (("near", nd), ("far", fd)):
        print(f"  {name:4s} DRR  p5 {np.percentile(v,5):7.2f}  p25 {np.percentile(v,25):7.2f}"
              f"  median {np.median(v):7.2f}  p75 {np.percentile(v,75):7.2f}"
              f"  p95 {np.percentile(v,95):7.2f} dB")
    acc, thr = best_threshold(nd, fd)
    amb = ((drr >= np.percentile(fd, 5)) & (drr <= np.percentile(nd, 95)))
    # mass in the band where the two distributions genuinely interleave:
    band_lo, band_hi = np.percentile(nd, 5), np.percentile(fd, 95)
    if band_lo < band_hi:
        in_band = (drr >= band_lo) & (drr <= band_hi)
        band_txt = f"[{band_lo:.1f}, {band_hi:.1f}] dB holds {100*in_band.mean():.1f}% of channels"
    else:
        band_txt = f"none (near p5 {band_lo:.1f} > far p95 {band_hi:.1f} -- gap {band_lo-band_hi:.1f} dB)"
    print(f"  best single threshold: {thr:+.2f} dB  ->  balanced acc {acc:.3f}")
    print(f"  interleave band {band_txt}")

    for og in ("sim", "real"):
        m = origin == og
        if m.sum() < 50:
            continue
        a, t = best_threshold(drr[m & is_near], drr[m & ~is_near])
        print(f"    {og:4s}: n={m.sum():6d}  near med {np.median(drr[m & is_near]):7.2f}"
              f"  far med {np.median(drr[m & ~is_near]):7.2f}"
              f"  best-thr acc {a:.3f}")

    print(f"  DRR by rt60 band (median near / median far):")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 2.0)):
        m = (rt60 >= lo) & (rt60 < hi)
        if m.sum() < 40:
            continue
        print(f"    rt60 {lo:.1f}-{hi:.1f}  n={m.sum():6d}   near {np.median(drr[m & is_near]):7.2f}"
              f"   far {np.median(drr[m & ~is_near]):7.2f}"
              f"   (near p5 {np.percentile(drr[m & is_near], 5) if (m & is_near).sum() >= 20 else float('nan'):7.2f})")
    return dict(acc=acc, thr=thr)


if __name__ == "__main__":
    live = collect("exp/hybrid_rir_16k_realfar", N_SAMPLE)
    report("LIVE BANK hybrid_rir_16k_realfar (recipe folder)", live)

    high = collect("exp/hybrid_rir_16k_high", 1500)
    report("HIGH BANK rt60 0.85-1.5 (on disk, never trained; reference only)", high)

    k = min(len(high), int(len(live) * 0.10 / 0.90))
    report("MIX live + high at ~10% of channels (reference only)", live + high[:k])
