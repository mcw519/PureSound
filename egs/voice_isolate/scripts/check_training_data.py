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

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/check_training_data.py \
        egs/voice_isolate/config/train_dpcrn.yaml --n 64 --dump 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.audio.io import AudioIO  # noqa: E402
from puresound.audio.vad import BatchedSileroVADLabeler  # noqa: E402
from puresound.recipes import load_siso_recipe_config  # noqa: E402


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


def check_manifests(corpus_dict: dict, config_path: str, n_path_checks: int) -> dict:
    train_path = resolve_metafile(corpus_dict["train_metafile"], config_path)
    valid_path = resolve_metafile(corpus_dict["valid_metafile"], config_path)
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


def collect(loader, n_items: int) -> tuple[list[dict], list[dict]]:
    """Pull batches until n_items rows are seen; return (per-item rows, batches)."""
    rows: list[dict] = []
    batches: list[dict] = []
    for batch in loader:
        batches.append(batch)
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
    return rows[:n_items], batches


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

    # *_rest absorbs recipe-tuple growth; the trailing blocks are forwarded so real
    # recording rows appear here too when the recipe enables them.
    (corpus, trainer, _optim, _sched, _loss, _model, a_sp, a_no, a_rv, a_spd,
     a_ir, a_src, a_hpf, a_vol, a_cod, a_pl, a_ta, a_vad, *rest) = load_siso_recipe_config(config_path)
    a_realfar = rest[0] if len(rest) > 0 else None
    a_realnear = rest[1] if len(rest) > 1 else None

    report = {"config": args.config_path}
    report["manifests"] = check_manifests(corpus, config_path, args.check_paths)

    trainer["num_workers"] = args.num_workers
    _train_dl, valid_dl = recipe_main.init_dataloader(
        corpus, trainer, a_sp, a_no, a_rv, a_spd, a_ir, a_src, a_hpf,
        a_vol, a_cod, a_pl, a_ta, a_vad, a_realfar, a_realnear)

    rows, batches = collect(valid_dl, args.n)
    report["separability"] = report_separability(rows)
    if args.dump > 0 and batches:
        report["dump"] = dump_samples(batches[0], a_vad, args.dump, out_dir / "wavs")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport -> {report_path}")


if __name__ == "__main__":
    main()
