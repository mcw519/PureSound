"""Validate the voice_isolate data pipeline (plan section 9.6).

Complementary checks, all driven by the same config the training run uses:

0. Manifest integrity. Train/valid speaker sets must be disjoint (speaker
   overlap makes validation blind to speaker generalization) and a random
   subset of audio paths must exist on disk.

1. Scene-level DRR / distance / RT60 separability. Drives the shoebox room
   simulator (puresound/audio/room_simulator.py) directly: for many sampled
   rooms it generates a foreground RIR and an interferer RIR in the *same* room
   (mirroring source-level reverb) and reports DRR histograms per role. The
   headline check is the foreground-vs-interferer DRR median gap >= 6 dB -- the
   condition under which a single-channel model can learn "near = foreground".

2. Sample-level dump for human listening. Instantiates the real
   NoiseSuppressionDataset and saves a few multi-channel wavs
   [noisy | clean | noise | vad] -- the 4th channel (present when vad_label is
   configured) renders the frame-level VAD ground truth as a sample-level
   0.9/0.0 step signal so it can be eyeballed against the clean waveform --
   plus a per-sample table (query distance, VAD active ratio, target-absent),
   so you can confirm the foreground is clear and the target has no interferer
   leakage.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/validate_data.py \
        egs/voice_isolate/config/conformer_nearfield_rirbank.yaml --n 128 --dump 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.audio.room_simulator import RoomImpulseResponseSimulator
from puresound.audio.vad import BatchedSileroVADLabeler
from puresound.recipes import load_siso_recipe_config
from puresound.task.ns import (
    NoiseSuppressionCollateFunc,
    NoiseSuppressionDataset,
)
from puresound.task.sampler import SpeakerSampler
from puresound.task.voice_isolation import (
    VoiceIsolationCollateFunc,
    VoiceIsolationDataset,
)


def text_histogram(values: list[float], bins: int = 20, width: int = 40) -> str:
    if not values:
        return "  (no values)"
    arr = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for i in range(bins):
        bar = "#" * int(round(width * counts[i] / peak))
        lines.append(f"  [{edges[i]:7.2f}, {edges[i + 1]:7.2f})  {counts[i]:5d} |{bar}")
    return "\n".join(lines)


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
    """Metafile paths in the config are CWD-relative; fall back to the recipe
    dir (the config file's parent's parent) when not run from there."""
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


def validate_manifests(corpus_dict: dict, config_path: str, n_path_checks: int) -> dict:
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

    print("=" * 64)
    print("MANIFEST INTEGRITY")
    print("=" * 64)
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


def build_simulator(aug_reverb: dict) -> RoomImpulseResponseSimulator:
    sim = aug_reverb["simulator"]
    return RoomImpulseResponseSimulator(
        room_dim_range=sim["room_dim_range"],
        rt60_range=sim["rt60_range"],
        source_receiver_distance_range=sim["source_receiver_distance_range"],
        foreground_distance_range=sim.get("foreground_distance_range"),
        interferer_distance_range=sim.get("interferer_distance_range"),
        media_distance_range=sim.get("media_distance_range"),
        receiver_margin=sim.get("receiver_margin", 0.4),
        source_margin=sim.get("source_margin", 0.4),
        nsample=sim.get("nsample"),
        order=sim.get("order", -1),
        hp_filter=sim.get("hp_filter", True),
    )


def validate_scenes(
    aug_reverb: dict,
    sample_rate: int,
    n: int,
) -> dict:
    sim = build_simulator(aug_reverb)
    fg = {"drr": [], "dist": []}
    itf = {"drr": [], "dist": []}
    rt60s = []
    for _ in range(n):
        scene = sim.sample_scene()
        rt60s.append(float(scene["rt60"]))
        _, fg_meta = sim.generate(
            sample_rate,
            scene=scene,
            source_role="foreground",
        )
        _, itf_meta = sim.generate(
            sample_rate,
            scene=scene,
            source_role="interferer",
        )
        fg["drr"].append(fg_meta["drr_db"])
        fg["dist"].append(fg_meta["source_receiver_distance"])
        itf["drr"].append(itf_meta["drr_db"])
        itf["dist"].append(itf_meta["source_receiver_distance"])

    fg_med = float(np.median(fg["drr"]))
    itf_med = float(np.median(itf["drr"]))
    gap = fg_med - itf_med
    passed = gap >= 6.0

    print("=" * 64)
    print(f"SCENE-LEVEL DRR SEPARABILITY  (n={n} rooms, sr={sample_rate})")
    print("=" * 64)
    print("\nForeground DRR (dB):")
    print(text_histogram(fg["drr"]))
    print("\nInterferer DRR (dB):")
    print(text_histogram(itf["drr"]))
    print(f"\nForeground DRR median : {fg_med:6.2f} dB")
    print(f"Interferer DRR median : {itf_med:6.2f} dB")
    print(f"DRR median gap        : {gap:6.2f} dB   "
          f"({'PASS' if passed else 'FAIL'} vs >= 6 dB target)")

    print("\nForeground distance (m):")
    print(text_histogram(fg["dist"]))
    print("\nInterferer distance (m):")
    print(text_histogram(itf["dist"]))
    print("\nRT60 (s):")
    print(text_histogram(rt60s))

    return {
        "n_scenes": n,
        "foreground_drr": summary(fg["drr"]),
        "interferer_drr": summary(itf["drr"]),
        "drr_median_gap_db": gap,
        "drr_gap_pass": passed,
        "foreground_distance": summary(fg["dist"]),
        "interferer_distance": summary(itf["dist"]),
        "rt60": summary(rt60s),
    }


def rms_db(x: torch.Tensor) -> float:
    r = x.pow(2).mean().sqrt().item()
    return 20.0 * torch.log10(torch.tensor(r + 1e-12)).item()


def dump_samples(cfg, n_dump: int, out_dir: Path, config_path: str) -> dict:
    (
        corpus_dict, trainer_dict, _optim, _sched, _loss, _model,
        aug_speech, aug_noise, aug_reverb, aug_speed, aug_ir, aug_src,
        aug_hpf, aug_volume, aug_codec, aug_packet_loss, aug_target_absent,
        vad_label,
    ) = cfg

    task_name = corpus_dict.get("task", "noise_suppression")
    if task_name == "voice_isolation":
        dataset_cls = VoiceIsolationDataset
        collate_fn = VoiceIsolationCollateFunc()
    else:
        dataset_cls = NoiseSuppressionDataset
        collate_fn = NoiseSuppressionCollateFunc()

    dataset = dataset_cls(
        metafile_path=str(resolve_metafile(corpus_dict["train_metafile"], config_path)),
        min_utt_length_in_seconds=corpus_dict["filter_min_utterance_length"],
        min_utts_in_each_speaker=corpus_dict["filter_min_utterance_per_speaker"],
        target_sr=corpus_dict["target_sample_rate"],
        training_sample_length_in_seconds=corpus_dict["training_length_seconds"],
        audio_gain_nomalized_to=corpus_dict["gain_nomalized_to"],
        augmentation_speech_args=aug_speech,
        augmentation_noise_args=aug_noise,
        augmentation_reverb_args=aug_reverb,
        augmentation_speed_args=aug_speed,
        augmentation_ir_response_args=aug_ir,
        augmentation_src_args=aug_src,
        augmentation_hpf_args=aug_hpf,
        augmentation_volume_args=aug_volume,
        augmentation_codec_args=aug_codec,
        augmentation_packet_loss_args=aug_packet_loss,
        augmentation_target_absent_args=aug_target_absent,
        vad_label_args=vad_label,
    )
    sampler = SpeakerSampler(
        data=dataset.meta,
        total_batch=1,
        n_spks=n_dump,
        n_per=1,
        select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 64)
    print(f"SAMPLE-LEVEL DUMP  (n={n_dump} -> {out_dir}/)")
    print("=" * 64)

    sir = aug_speech.get("snr_range") if aug_speech else None
    snr = aug_noise.get("snr_range") if aug_noise else None
    print(f"configured SIR range (speech): {sir}")
    print(f"configured SNR range (noise) : {snr}")

    batch = next(iter(loader))
    noisy, clean, noise = batch["noisy_speech"], batch["clean_speech"], batch["consistency_noise"]
    qd, vad = batch.get("query_distance"), batch.get("vad_target")
    bg_vad = batch.get("background_vad_target")

    # With a Silero backend the dataset defers labeling to the training module
    # (it emits the clean reference instead of frame labels); mirror
    # on_after_batch_transfer + the labeler construction in
    # egs/noise_suppression/main.py so the dump gets the exact ground truth
    # the loss would see.
    vad_labeler = dataset.vad_labeler
    if vad is None and "vad_reference" in batch:
        vad_labeler = BatchedSileroVADLabeler(**vad_label.get("args", {}))
        vad = vad_labeler(
            batch["vad_reference"],
            sample_rate=int(batch["sr"].reshape(-1)[0].item()),
        )
    if bg_vad is None and "background_vad_reference" in batch:
        bg_labeler = BatchedSileroVADLabeler(**vad_label.get("args", {}))
        bg_vad = bg_labeler(
            batch["background_vad_reference"],
            sample_rate=int(batch["sr"].reshape(-1)[0].item()),
        )
    n = noisy.shape[0]
    n_absent = 0

    # Realized target-to-residual ratio: rms(clean) vs rms(noisy - clean).
    # The residual is interferers + noise + echo after the device chain, so this
    # measures the coverage the sampled SIR/SNR ranges actually deliver.
    trr_db = []
    drr_gap = []
    boundary_margin = []

    print(
        f"\n{'idx':>3} {'dist(m)':>8} {'drrGap':>8} {'bMarg':>7} "
        f"{'clean dB':>9} {'noisy dB':>9} {'trr dB':>8} "
        f"{'vad_act':>8} {'bg_vad':>7} {'present':>7}"
    )
    for i in range(n):
        absent = bool(clean[i].abs().amax().item() == 0)
        n_absent += int(absent)
        d = f"{qd[i].item():.2f}" if qd is not None else "-"
        vact = f"{vad[i].float().mean().item():.2f}" if vad is not None else "-"
        bgact = f"{bg_vad[i].float().mean().item():.2f}" if bg_vad is not None else "-"
        dg = batch.get("drr_gap")
        bm = batch.get("boundary_margin")
        present = batch.get("target_present")
        dg_s = f"{dg[i].item():.1f}" if dg is not None and torch.isfinite(dg[i]) else "-"
        bm_s = f"{bm[i].item():.2f}" if bm is not None and torch.isfinite(bm[i]) else "-"
        pr_s = f"{present[i].item():.0f}" if present is not None else str(not absent)
        if dg is not None and torch.isfinite(dg[i]):
            drr_gap.append(float(dg[i].item()))
        if bm is not None and torch.isfinite(bm[i]):
            boundary_margin.append(float(bm[i].item()))
        if not absent:
            trr = rms_db(clean[i]) - rms_db(noise[i])
            trr_db.append(trr)
            trr_s = f"{trr:.1f}"
        else:
            trr_s = "-"
        print(
            f"{i:>3} {d:>8} {dg_s:>8} {bm_s:>7} "
            f"{rms_db(clean[i]):>9.1f} {rms_db(noisy[i]):>9.1f} "
            f"{trr_s:>8} {vact:>8} {bgact:>7} {pr_s:>7}"
        )
        chans = [noisy[i], clean[i], noise[i]]
        if vad is not None:
            # Frame-level VAD ground truth -> sample-level step signal
            # (0.9 = voice active, 0.0 = inactive) so it can be visually
            # aligned with the clean waveform in an audio editor.
            hop = vad_labeler.hop_length
            vad_wave = (vad[i] > 0.5).float().repeat_interleave(hop) * 0.9
            pad = noisy.shape[-1] - vad_wave.shape[-1]
            if pad > 0:
                vad_wave = torch.nn.functional.pad(vad_wave, (0, pad))
            chans.append(vad_wave[: noisy.shape[-1]])
        AudioIO.save(
            wav=torch.stack(chans, dim=0),
            f_path=str(out_dir / f"sample-{i:02d}.wav"),
            sr=int(batch["sr"][i].item()),
        )
    ch_desc = "ch0=noisy ch1=clean ch2=noise" + (" ch3=vad(0.9/0)" if vad is not None else "")
    print(f"\nsaved {n} wavs ({ch_desc}). target-absent: {n_absent}/{n}")
    if trr_db:
        print("\nRealized target-to-residual ratio (dB):")
        print(text_histogram(trr_db, bins=10))
    return {
        "n_dump": n,
        "sir_range": sir,
        "snr_range": snr,
        "target_absent": f"{n_absent}/{n}",
        "target_to_residual_db": summary(trr_db),
        "drr_gap": summary(drr_gap),
        "boundary_margin": summary(boundary_margin),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config_path")
    p.add_argument("--n", type=int, default=128, help="number of rooms for DRR check")
    p.add_argument("--dump", type=int, default=8, help="number of sample wavs to dump")
    p.add_argument("--check-paths", type=int, default=200,
                   help="number of manifest audio paths to existence-check per split")
    p.add_argument("--out", default="./egs/voice_isolate/data_report")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_siso_recipe_config(args.config_path)
    aug_reverb = cfg[8]
    sample_rate = int(cfg[0]["target_sample_rate"])

    out_dir = Path(args.out)
    report = {"config": args.config_path}

    report["manifests"] = validate_manifests(cfg[0], args.config_path, args.check_paths)

    if aug_reverb and aug_reverb.get("simulator", {}).get("used"):
        report["scenes"] = validate_scenes(aug_reverb, sample_rate, args.n)
    else:
        print("augmentation_reverb.simulator not enabled; skipping DRR check.")

    if args.dump > 0:
        report["dump"] = dump_samples(cfg, args.dump, out_dir / "wavs", args.config_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport -> {report_path}")


if __name__ == "__main__":
    main()
