"""Dump a FROZEN conversational turn-taking (mix, target) test set as wav.

Forces overlap_control.turn_taking_prob=1.0 on the given config so every row is
a near/far alternating exchange, then saves (mix, target) pairs -- near and far
alternate in long turns (far gets continuous multi-second solos, incl. row-initial),
the target label is gated in lockstep with the mix. With a fixed --seed the dump is
reproducible, so once written to disk it is a frozen benchmark: KEEP the near user
during their turns, SUPPRESS the far competitor during its solo stretches.

A manifest.jsonl is written alongside the wavs (one row per item: turn_taking flag,
duration, mix/target RMS, target-active ratio) so the set can be scored / filtered.

Usage (from repo root):
    # small listening dump (original behaviour)
    uv run python egs/voice_isolate/scripts/dump_turntaking_samples.py \
        egs/voice_isolate/config/exp/eval_indomain_phase1.yaml \
        --out-dir data_report/turntaking_samples --n 6
    # frozen 100-item, ~10 s test set
    uv run python egs/voice_isolate/scripts/dump_turntaking_samples.py \
        egs/voice_isolate/config/exp/eval_indomain_phase1.yaml \
        --out-dir /data/audio/eval_noisy_data/turntaking_set \
        --n 100 --length-seconds 10 --seed 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _rms_dbfs(x) -> float:
    import torch
    return float(20.0 * torch.log10(x.reshape(-1).square().mean().sqrt().clamp_min(1e-12)))


def _active_ratio(x, sr: int, win_ms: float = 25.0, floor_db: float = -40.0) -> float:
    """Fraction of framed energy above floor_db relative to the per-clip peak frame.

    A turn-taking target is silent during the far solos, so this is ~= the near
    user's talk-time fraction -- a compact proxy for how much of the row is KEEP
    (target present) vs SUPPRESS (target silent, far only).
    """
    import torch
    x = x.reshape(-1)
    win = max(1, int(sr * win_ms / 1000.0))
    n = (x.numel() // win) * win
    if n == 0:
        return 0.0
    frames = x[:n].reshape(-1, win)
    fe = frames.square().mean(dim=1).sqrt()
    peak = fe.max().clamp_min(1e-12)
    fe_db = 20.0 * torch.log10((fe / peak).clamp_min(1e-12))
    return float((fe_db > floor_db).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config_path")
    ap.add_argument("--out-dir", default="data_report/turntaking_samples")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--length-seconds", type=float, default=None,
                    help="override dataset.training_length_seconds (e.g. 10 for a "
                         "~10 s test set). default: keep the config value.")
    ap.add_argument("--rir-folder", default=None,
                    help="override augmentation_reverb.simulator.pregenerated.folder "
                         "(e.g. a REAL-RIR bank like exp/but_real_rir_16k_office). "
                         "default: keep the config's bank.")
    args = ap.parse_args()

    os.chdir(REPO / "egs/voice_isolate")
    import torch
    from puresound.audio.io import AudioIO
    from puresound.recipes import load_siso_recipe_config
    import egs.noise_suppression.main as M

    cfg = load_siso_recipe_config(str(Path(args.config_path).resolve()))
    (corpus, trainer, _opt, _sch, _loss, _md, a_sp, a_no, a_rv, a_spd,
     a_ir, a_src, a_hpf, a_vol, a_cod, a_pl, a_ta, a_vad) = cfg
    trainer["num_workers"] = 0
    if args.length_seconds is not None:
        corpus["training_length_seconds"] = float(args.length_seconds)
    if args.rir_folder is not None:
        a_rv["simulator"]["pregenerated"]["folder"] = args.rir_folder
    a_sp["overlap_control"]["turn_taking_prob"] = 1.0  # force turn-taking on every row
    a_sp["prob"] = 1.0                                  # always add an interferer
    if a_ta:
        a_ta["used"] = False  # keep target present so the pair is audible

    torch.manual_seed(args.seed)
    _tr, valid_dl = M.init_dataloader(
        corpus, trainer, a_sp, a_no, a_rv, a_spd, a_ir, a_src, a_hpf,
        a_vol, a_cod, a_pl, a_ta, a_vad)

    sr = int(corpus.get("target_sample_rate", 16000))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = open(out / "manifest.jsonl", "w")
    saved = 0
    for batch in valid_dl:
        B = batch["clean_speech"].shape[0]
        for r in range(B):
            cl = batch["clean_speech"][r]
            if cl.abs().max() < 1e-6:
                continue
            mix = batch["noisy_speech"][r].reshape(1, -1).cpu()
            tgt = cl.reshape(1, -1).cpu()
            tt = float(batch["turn_taking"][r]) if "turn_taking" in batch else -1
            item_id = f"tt{saved:03d}_turn{int(tt)}"
            AudioIO.save(mix, str(out / f"{item_id}_mix.wav"), sr)
            AudioIO.save(tgt, str(out / f"{item_id}_target.wav"), sr)
            manifest.write(json.dumps({
                "id": item_id,
                "turn_taking": int(tt),
                "duration_s": round(tgt.numel() / sr, 3),
                "samples": int(tgt.numel()),
                "sample_rate": sr,
                "mix_rms_dbfs": round(_rms_dbfs(mix), 2),
                "target_rms_dbfs": round(_rms_dbfs(tgt), 2),
                "target_active_ratio": round(_active_ratio(tgt, sr), 3),
            }) + "\n")
            manifest.flush()
            saved += 1
            if saved >= args.n:
                manifest.close()
                print(f"wrote {saved} (mix,target) turn-taking pairs + manifest.jsonl -> {out.resolve()}")
                return
    manifest.close()
    print(f"wrote {saved} pairs (fewer than requested) + manifest.jsonl -> {out.resolve()}")


if __name__ == "__main__":
    main()
