"""Check the data a recipe will actually train on, before spending a run on it.

Three checks, all driven by the same config the training run uses:

0. Manifest integrity. Train/valid speaker sets must be disjoint (overlap makes
   validation blind to speaker generalization) and a random subset of audio paths
   must exist on disk.

1. Near/far separability, measured on real sampled items instead of re-derived from a
   room simulator, so it works for pre-generated RIR banks and real-recording rows as
   well as on-the-fly simulation. The headline number is the median DRR gap between
   the foreground and the strongest interferer: a single-channel model can only learn
   "near = foreground" when that gap is comfortably positive (>= 6 dB in the recipes
   here). Distances, RT60 and the realized target-to-residual ratio are reported the
   same way, because a range configured is not a range delivered.

2. Sample dump for listening. Saves multi-channel wavs [noisy | clean | noise | vad],
   where the last channel renders the frame-level VAD ground truth as a 0.9/0.0 step
   signal that can be eyeballed against the clean waveform, plus a per-item table.

3. Temporal shape, on the frame grid the labels live on (400 / 160 = 100 fps). Check 1
   measures no temporal property at all, which is why the distribution the onset and
   wrong-anchor rounds argue about was never on the record: when does the target
   start, how long has an interferer been talking before it does, how long are the
   target-free gaps, does the target ever re-enter after one, and is a silent target
   labelled as absent. Method ported from
   benchmarks/probes/v19c_diagnostics/training_data_audit/sample_rows.py.

`--split train` samples the TRAINING loader, which is what every temporal claim about
"the rows the model trains on" needs -- the default stays `valid`, so an existing
invocation reads exactly as it did before.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/check_training_data.py \
        egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
    uv run python egs/voice_isolate/scripts/check_training_data.py \
        egs/voice_isolate/config/exp/train_dpcrn_v16_lengthmix.yaml \
        --split train --n 600 --dump 0 --num-workers 8 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.config import load_recipe, with_overrides
from puresound.audio.io import AudioIO  # noqa: E402
from puresound.audio.vad import BatchedSileroVADLabeler  # noqa: E402


def text_histogram(values: list[float], bins: int = 12, width: int = 40) -> str:
    if not values:
        return "  (no values)"
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    peak = max(int(counts.max()), 1)
    return "\n".join(
        f"  [{edges[i]:7.2f}, {edges[i + 1]:7.2f})  {counts[i]:5d} |"
        + "#" * int(round(width * counts[i] / peak))
        for i in range(bins)
    )


def summary(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def resolve_metafile(path_str: str, config_path: str) -> Path:
    """Metafile paths in the config are CWD-relative; fall back to the recipe dir."""
    p = Path(path_str)
    if p.exists():
        return p
    fallback = Path(config_path).resolve().parent.parent / path_str
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"metafile not found: {path_str} (also tried {fallback})")


def read_manifest(f_path: Path) -> list[tuple[str, str]]:
    """Returns [(spkid, audio_path), ...]."""
    rows = []
    for line in f_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("uttid"):
            continue
        cols = [c.strip() for c in line.split(",")]
        rows.append((cols[1], cols[3]))
    return rows


def check_manifests(corpus, config_path: str, n_path_checks: int) -> dict:
    train_path = resolve_metafile(corpus.train_metafile, config_path)
    valid_path = resolve_metafile(corpus.valid_metafile, config_path)
    train_rows = read_manifest(train_path)
    valid_rows = read_manifest(valid_path)
    train_spk = {s for s, _ in train_rows}
    valid_spk = {s for s, _ in valid_rows}
    overlap = sorted(train_spk & valid_spk)

    rng = np.random.RandomState(0)
    missing = []
    for rows in (train_rows, valid_rows):
        idx = rng.choice(len(rows), size=min(n_path_checks, len(rows)), replace=False)
        for i in idx:
            if not Path(rows[i][1]).is_file():
                missing.append(rows[i][1])

    print("=" * 72)
    print("MANIFEST INTEGRITY")
    print("=" * 72)
    print(f"train: {len(train_rows)} utts / {len(train_spk)} speakers ({train_path})")
    print(f"valid: {len(valid_rows)} utts / {len(valid_spk)} speakers ({valid_path})")
    print(f"speaker overlap : {len(overlap)}   "
          f"({'PASS' if not overlap else 'FAIL -- validation cannot measure speaker generalization'})")
    print(f"missing paths   : {len(missing)}/{2 * min(n_path_checks, len(train_rows))} sampled   "
          f"({'PASS' if not missing else 'FAIL'})")
    for m in missing[:5]:
        print(f"  missing: {m}")

    return {
        "train_utts": len(train_rows),
        "train_speakers": len(train_spk),
        "valid_utts": len(valid_rows),
        "valid_speakers": len(valid_spk),
        "speaker_overlap": len(overlap),
        "speaker_overlap_pass": not overlap,
        "missing_paths_sampled": len(missing),
        "path_check_pass": not missing,
    }


def rms_db(x: torch.Tensor) -> float:
    return 20.0 * torch.log10(torch.tensor(x.pow(2).mean().sqrt().item() + 1e-12)).item()


def scalar(batch: dict, key: str, row: int) -> float | None:
    """Per-item scalar metadata, or None when absent / NaN (labels are optional)."""
    v = batch.get(key)
    if v is None:
        return None
    x = v.reshape(-1)[row]
    return float(x) if torch.isfinite(x) else None


# --------------------------------------------------------------------------- #
# Check 3: temporal shape. Ported from
# benchmarks/probes/v19c_diagnostics/training_data_audit/sample_rows.py so the
# numbers a round quotes and the numbers this script prints are one method.
# --------------------------------------------------------------------------- #

SR = 16000
FRAME = 400
HOP = 160
FPS = SR / HOP  # 100


def frames_energy_db(wav: torch.Tensor) -> np.ndarray:
    """Per-frame power in dB relative to the row's own peak frame, on the label
    grid (frame 400, hop 160 -- the pipeline's EnergyVADLabeler)."""
    x = wav.reshape(1, -1)
    if x.shape[-1] < FRAME:
        x = torch.nn.functional.pad(x, (0, FRAME - x.shape[-1]))
    frames = x.unfold(-1, FRAME, HOP)
    power = frames.square().mean(dim=-1).squeeze(0)
    reference = power.max().clamp_min(1e-20)
    return (10.0 * torch.log10(power.clamp_min(1e-30) / reference)).numpy()


def runs_of_zero(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] for every run of zeros in a 0/1 mask."""
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i] == 0:
            j = i
            while j < n and mask[j] == 0:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def describe_activity(mask: np.ndarray) -> dict:
    """Onset / offset / gaps / re-entry of one 0/1 frame-activity mask.

    ``longest_gap_s`` counts the leading and trailing silence too (that is the
    "target-free gap" a suppression term sees); ``longest_interior_gap_s`` is the
    one a re-entry has to cross, so re-entry is read off that.
    """
    active = np.flatnonzero(mask > 0)
    out = {"active_frac": float(mask.mean()) if len(mask) else float("nan")}
    if active.size == 0:
        out.update(onset_s=None, offset_s=None, longest_gap_s=float(len(mask) / FPS),
                   longest_interior_gap_s=0.0, reentry_after_5s=False, n_spans=0)
        return out
    zero_runs = runs_of_zero(mask)
    longest = max((e - s for s, e in zero_runs), default=0)
    interior = max((e - s for s, e in zero_runs if s > 0 and e < len(mask)), default=0)
    edges = np.diff(np.concatenate(([0], (mask > 0).astype(int), [0])))
    out.update(
        onset_s=float(active[0] / FPS),
        offset_s=float(active[-1] / FPS),
        longest_gap_s=float(longest / FPS),
        longest_interior_gap_s=float(interior / FPS),
        reentry_after_5s=bool(interior >= 5 * FPS),
        n_spans=int((edges == 1).sum()),
    )
    return out


def temporal_rows(batch: dict) -> list[dict]:
    """One temporal record per row of one batch, on the label frame grid."""
    clean = batch["clean_speech"]
    far = batch.get("far_target")
    vad = batch.get("vad_target")
    background_vad = batch.get("background_vad_target")
    lengths = batch["length"].reshape(-1)
    out = []
    for r in range(clean.shape[0]):
        n_samples = int(lengths[r].item())
        n_frames = max(1, n_samples // HOP)
        silent = bool(clean[r, :n_samples].abs().amax().item() == 0.0)

        # target activity: the pipeline's own labels, plus a stricter -25 dB
        # waveform detector as the control the audit reports beside them.
        if silent:
            target = np.zeros(n_frames)
            target_db25 = target
        else:
            db = frames_energy_db(clean[r, :n_samples])
            target = (db > -40.0).astype(float)
            target_db25 = (db > -25.0).astype(float)
        if vad is not None:
            target = (vad[r].numpy()[:n_frames] > 0.5).astype(float)

        # interferer activity: pipeline labels when present, else far_target energy
        if far is not None and far[r].abs().amax().item() > 0.0:
            interferer = (frames_energy_db(far[r, : min(n_samples, far.shape[-1])])
                          > -40.0).astype(float)
        else:
            interferer = np.zeros(n_frames)
        if background_vad is not None:
            interferer = (background_vad[r].numpy()[:n_frames] > 0.5).astype(float)

        m = min(len(target), len(target_db25), len(interferer))
        target, target_db25, interferer = target[:m], target_db25[:m], interferer[:m]

        record = {
            "row_seconds": round(n_samples / SR, 3),
            "n_frames": int(m),
            "target_silent_waveform": silent,
            "scalar_target_absent": scalar(batch, "target_absent", r),
            "scalar_target_present": scalar(batch, "target_present", r),
            "turn_taking": scalar(batch, "turn_taking", r),
            "far_count": scalar(batch, "far_count", r),
            "n_interferers": scalar(batch, "n_interferers", r),
            "has_background_speech": scalar(batch, "has_background_speech", r),
            "target": describe_activity(target),
            "target_db25": describe_activity(target_db25),
            "interferer": describe_activity(interferer),
        }
        onset = record["target"]["onset_s"]
        if onset is None:
            record["itf_before_onset_s"] = float(interferer.sum() / FPS)
        else:
            k = int(round(onset * FPS))
            record["itf_before_onset_s"] = float(interferer[:k].sum() / FPS)
        out.append(record)
    return out


def report_temporal(rows: list[dict], split: str) -> dict:
    """The five temporal columns, overall and per length bucket."""

    def share(predicate, subset=None) -> tuple[int, int]:
        subset = rows if subset is None else subset
        hit = sum(1 for r in subset if predicate(r))
        return hit, len(subset)

    def med(key, subset=None):
        subset = rows if subset is None else subset
        vals = [key(r) for r in subset if key(r) is not None]
        return float(np.median(vals)) if vals else float("nan")

    onset = lambda r: r["target"]["onset_s"]                       # noqa: E731
    gap = lambda r: r["target"]["longest_gap_s"]                    # noqa: E731
    interior = lambda r: r["target"]["longest_interior_gap_s"]      # noqa: E731
    before = lambda r: r["itf_before_onset_s"]                      # noqa: E731

    print("\n" + "=" * 72)
    print(f"TEMPORAL SHAPE  (n={len(rows)} rows, split={split}, 100 fps label grid)")
    print("=" * 72)
    if not rows:
        print("  (no rows)")
        return {}

    def pct(hit_total):
        hit, total = hit_total
        return f"{hit:5d}/{total:<5d} {100.0 * hit / max(total, 1):5.1f}%"

    print(f"target onset median            : {med(onset):6.2f} s")
    print(f"  onset within 0.5 s          : {pct(share(lambda r: (onset(r) or 0.0) <= 0.5))}")
    print(f"  onset >= 1.0 s (>=100 frames): {pct(share(lambda r: (onset(r) or 0.0) >= 1.0))}")
    print(f"interferer-before-onset median: {med(before):6.2f} s")
    for threshold in (0.001, 0.5, 1.0, 2.0):
        label = ">0 s" if threshold < 0.01 else f">={threshold:g} s"
        print(f"  any interferer {label:>7s}       : {pct(share(lambda r, t=threshold: before(r) >= t))}")
    print(f"longest target-free gap median: {med(gap):6.2f} s")
    print(f"  gap >= 5 s                  : {pct(share(lambda r: gap(r) >= 5.0))}")
    print(f"longest INTERIOR gap median   : {med(interior):6.2f} s")
    print(f"  re-entry after >= 5 s       : {pct(share(lambda r: r['target']['reentry_after_5s']))}")

    # target-absent provenance. The label bug this exposes: a row whose gated
    # target is exactly zero can still carry target_present=1, so the waveform
    # losses treat it as lone-far while ResidualReferenceLoss(target_present_only)
    # counts it present.
    prov = {
        "silent_and_labelled_absent": share(
            lambda r: r["target_silent_waveform"] and (r["scalar_target_absent"] or 0) > 0.5),
        "silent_but_labelled_present": share(
            lambda r: r["target_silent_waveform"] and (r["scalar_target_absent"] or 0) <= 0.5),
        "not_silent_but_labelled_absent": share(
            lambda r: not r["target_silent_waveform"] and (r["scalar_target_absent"] or 0) > 0.5),
    }
    print("\ntarget-absent provenance (waveform vs label):")
    for name, hit_total in prov.items():
        print(f"  {name:32s}: {pct(hit_total)}")

    buckets = sorted({r["row_seconds"] for r in rows})
    print(f"\nper length bucket ({len(buckets)} seen):")
    print(f"{'sec':>7} {'n':>5} {'onset<=0.5s':>12} {'itf>=1s':>9} {'itf>=0.5s':>11} "
          f"{'gap>=5s':>9} {'reentry':>8} {'silent':>7}")
    per_bucket = {}
    for seconds in buckets:
        subset = [r for r in rows if r["row_seconds"] == seconds]
        cells = {
            "n": len(subset),
            "onset_le_0.5s": share(lambda r: (onset(r) or 0.0) <= 0.5, subset)[0],
            "itf_ge_1s": share(lambda r: before(r) >= 1.0, subset)[0],
            "itf_ge_0.5s": share(lambda r: before(r) >= 0.5, subset)[0],
            "gap_ge_5s": share(lambda r: gap(r) >= 5.0, subset)[0],
            "reentry_5s": share(lambda r: r["target"]["reentry_after_5s"], subset)[0],
            "target_silent": share(lambda r: r["target_silent_waveform"], subset)[0],
        }
        per_bucket[f"{seconds:g}"] = cells
        print(f"{seconds:7.2f} {cells['n']:5d} {cells['onset_le_0.5s']:12d} "
              f"{cells['itf_ge_1s']:9d} {cells['itf_ge_0.5s']:11d} "
              f"{cells['gap_ge_5s']:9d} {cells['reentry_5s']:8d} "
              f"{cells['target_silent']:7d}")

    return {
        "split": split,
        "n_rows": len(rows),
        "target_onset_s": summary([onset(r) for r in rows if onset(r) is not None]),
        "itf_before_onset_s": summary([before(r) for r in rows]),
        "longest_target_free_gap_s": summary([gap(r) for r in rows]),
        "longest_interior_gap_s": summary([interior(r) for r in rows]),
        "shares": {
            "onset_le_0.5s": share(lambda r: (onset(r) or 0.0) <= 0.5),
            "onset_ge_1.0s": share(lambda r: (onset(r) or 0.0) >= 1.0),
            "itf_before_onset_gt_0s": share(lambda r: before(r) > 0.0),
            "itf_before_onset_ge_0.5s": share(lambda r: before(r) >= 0.5),
            "itf_before_onset_ge_1.0s": share(lambda r: before(r) >= 1.0),
            "itf_before_onset_ge_2.0s": share(lambda r: before(r) >= 2.0),
            "gap_ge_5s": share(lambda r: gap(r) >= 5.0),
            "reentry_after_5s": share(lambda r: r["target"]["reentry_after_5s"]),
        },
        "target_absent_provenance": prov,
        "per_length_bucket": per_bucket,
    }


def collect(loader, n_items: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Pull batches until n_items rows are seen.

    Returns ``(separability rows, temporal rows, [first batch])``. Only the first
    batch is retained: it is the one `dump_samples` writes, and holding every
    batch of a 600-row train sample costs hundreds of MB of waveform for nothing.
    """
    rows: list[dict] = []
    temporal: list[dict] = []
    batches: list[dict] = []
    for batch in loader:
        if not batches:
            batches.append(batch)
        temporal.extend(temporal_rows(batch))
        for r in range(batch["clean_speech"].shape[0]):
            absent = bool(batch["clean_speech"][r].abs().amax().item() == 0)
            item = {
                "target_absent": absent,
                "turn_taking": scalar(batch, "turn_taking", r),
                "foreground_drr": scalar(batch, "foreground_drr", r),
                "interferer_drr": scalar(batch, "strongest_interferer_drr", r),
                "drr_gap": scalar(batch, "drr_gap", r),
                "foreground_distance": scalar(batch, "foreground_distance", r),
                "interferer_distance": scalar(batch, "nearest_interferer_distance", r),
                "rt60": scalar(batch, "rt60", r),
                "noisy_db": rms_db(batch["noisy_speech"][r]),
            }
            if not absent:
                item["clean_db"] = rms_db(batch["clean_speech"][r])
                if "consistency_noise" in batch:
                    # residual = interferers + noise after the device chain, so this is
                    # the SIR/SNR the configured ranges actually delivered.
                    item["target_to_residual_db"] = item["clean_db"] - rms_db(batch["consistency_noise"][r])
            rows.append(item)
        if len(rows) >= n_items:
            break
    return rows[:n_items], temporal[:n_items], batches


def report_separability(rows: list[dict]) -> dict:
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    fg_drr, itf_drr = col("foreground_drr"), col("interferer_drr")
    gap = col("drr_gap") or [f - i for f, i in zip(fg_drr, itf_drr)]
    fg_med = float(np.median(fg_drr)) if fg_drr else float("nan")
    itf_med = float(np.median(itf_drr)) if itf_drr else float("nan")
    gap_med = float(np.median(gap)) if gap else float("nan")
    passed = bool(gap_med >= 6.0)

    n_absent = sum(1 for r in rows if r["target_absent"])
    n_turn = sum(1 for r in rows if r.get("turn_taking"))

    print("=" * 72)
    print(f"NEAR/FAR SEPARABILITY  (n={len(rows)} sampled items)")
    print("=" * 72)
    print(f"target-absent rows : {n_absent}/{len(rows)}")
    print(f"turn-taking rows   : {n_turn}/{len(rows)}")
    if fg_drr:
        print("\nForeground DRR (dB):")
        print(text_histogram(fg_drr))
    if itf_drr:
        print("\nStrongest-interferer DRR (dB):")
        print(text_histogram(itf_drr))
    print(f"\nForeground DRR median : {fg_med:6.2f} dB")
    print(f"Interferer DRR median : {itf_med:6.2f} dB")
    print(f"DRR gap median        : {gap_med:6.2f} dB   "
          f"({'PASS' if passed else 'FAIL'} vs >= 6 dB target)")
    for key, label in (("foreground_distance", "Foreground distance (m)"),
                       ("interferer_distance", "Nearest-interferer distance (m)"),
                       ("rt60", "RT60 (s)"),
                       ("target_to_residual_db", "Realized target-to-residual ratio (dB)")):
        vals = col(key)
        if vals:
            print(f"\n{label}:")
            print(text_histogram(vals))

    return {
        "n_items": len(rows),
        "target_absent": f"{n_absent}/{len(rows)}",
        "turn_taking": f"{n_turn}/{len(rows)}",
        "foreground_drr": summary(fg_drr),
        "interferer_drr": summary(itf_drr),
        "drr_median_gap_db": gap_med,
        "drr_gap_pass": passed,
        "foreground_distance": summary(col("foreground_distance")),
        "interferer_distance": summary(col("interferer_distance")),
        "rt60": summary(col("rt60")),
        "target_to_residual_db": summary(col("target_to_residual_db")),
    }


def dump_samples(batch: dict, vad_label_cfg: dict | None, n_dump: int, out_dir: Path) -> dict:
    noisy, clean = batch["noisy_speech"], batch["clean_speech"]
    noise = batch.get("consistency_noise")
    vad = batch.get("vad_target")
    hop = int((vad_label_cfg or {}).get("args", {}).get("hop_length", 160))
    # A Silero backend defers labeling to the training module (the dataset emits the
    # clean reference instead of frame labels); mirror it so the dump shows the exact
    # ground truth the loss would see.
    if vad is None and "vad_reference" in batch and vad_label_cfg:
        labeler = BatchedSileroVADLabeler(**vad_label_cfg.get("args", {}))
        vad = labeler(batch["vad_reference"], sample_rate=int(batch["sr"].reshape(-1)[0].item()))
        hop = labeler.hop_length

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_dump, noisy.shape[0])
    print("\n" + "=" * 72)
    print(f"SAMPLE DUMP  (n={n} -> {out_dir}/)")
    print("=" * 72)
    print(f"\n{'idx':>3} {'fgDRR':>7} {'itfDRR':>7} {'gap':>6} {'fgDist':>7} {'itfDist':>8} "
          f"{'clean dB':>9} {'noisy dB':>9} {'vad_act':>8} {'present':>8}")
    for i in range(n):
        absent = bool(clean[i].abs().amax().item() == 0)

        def fmt(key, prec=1):
            v = scalar(batch, key, i)
            return f"{v:.{prec}f}" if v is not None else "-"

        vact = f"{vad[i].float().mean().item():.2f}" if vad is not None else "-"
        print(f"{i:>3} {fmt('foreground_drr'):>7} {fmt('strongest_interferer_drr'):>7} "
              f"{fmt('drr_gap'):>6} {fmt('foreground_distance', 2):>7} "
              f"{fmt('nearest_interferer_distance', 2):>8} "
              f"{rms_db(clean[i]):>9.1f} {rms_db(noisy[i]):>9.1f} {vact:>8} {str(not absent):>8}")

        chans = [noisy[i].reshape(-1), clean[i].reshape(-1)]
        if noise is not None:
            chans.append(noise[i].reshape(-1))
        if vad is not None:
            # Frame-level VAD ground truth -> sample-level step signal (0.9 = active)
            # so it can be aligned with the clean waveform in an audio editor.
            vad_wave = (vad[i] > 0.5).float().repeat_interleave(hop) * 0.9
            pad = noisy.shape[-1] - vad_wave.shape[-1]
            if pad > 0:
                vad_wave = torch.nn.functional.pad(vad_wave, (0, pad))
            chans.append(vad_wave[: noisy.shape[-1]])
        AudioIO.save(wav=torch.stack(chans, dim=0), f_path=str(out_dir / f"sample-{i:02d}.wav"),
                     sr=int(batch["sr"].reshape(-1)[i].item()))

    ch_desc = "ch0=noisy ch1=clean" + (" ch2=noise" if noise is not None else "") \
              + (" ch3=vad(0.9/0)" if vad is not None else "")
    print(f"\nsaved {n} wavs ({ch_desc})")
    return {"n_dump": n, "channels": ch_desc}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config_path")
    p.add_argument("--n", type=int, default=64, help="items sampled for the separability check")
    p.add_argument("--split", choices=("train", "valid"), default="valid",
                   help="which loader to sample; the default keeps every existing "
                        "invocation reading exactly as it did")
    p.add_argument("--dump", type=int, default=8, help="sample wavs to dump (0 = none)")
    p.add_argument("--check-paths", type=int, default=200,
                   help="manifest audio paths to existence-check per split")
    p.add_argument("--out", default="data_report/data_check")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config_path = str(Path(args.config_path).resolve())
    out_dir = Path(args.out) if Path(args.out).is_absolute() else RECIPE_DIR / args.out
    os.chdir(RECIPE_DIR)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # The TRAIN sampler is unseeded (only validation gets per-item seeds), so its
    # batch composition comes out of the process's own `random` -- seed it or
    # `--split train` is not reproducible. Cannot change what `--split valid`
    # prints: that sampler builds its own `random.Random(valid_seed)` and the
    # synthesis is per-item seeded (task/ns.py: `random.seed(item_seed)`).
    random.seed(args.seed)

    # *_rest absorbs recipe-tuple growth; the trailing blocks are forwarded so real
    # recording rows appear here too when the recipe enables them.
    recipe = load_recipe(
        config_path, expected_task="voice_isolation", expected_purpose="train"
    )

    report = {"config": args.config_path}
    report["manifests"] = check_manifests(
        recipe.dataset, config_path, args.check_paths
    )

    recipe = with_overrides(recipe, trainer={"num_workers": args.num_workers})
    train_dl, valid_dl = recipe_main.init_dataloader(recipe)
    loader = train_dl if args.split == "train" else valid_dl
    rows, temporal, batches = collect(loader, args.n)
    report["split"] = args.split
    report["separability"] = report_separability(rows)
    report["temporal"] = report_temporal(temporal, args.split)
    if args.dump > 0 and batches:
        report["dump"] = dump_samples(batches[0], recipe.vad_label, args.dump, out_dir / "wavs")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport -> {report_path}")


if __name__ == "__main__":
    main()
