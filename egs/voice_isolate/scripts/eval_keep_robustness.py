"""Channel-gradient KEEP probe: how much does a checkpoint preserve NEAR speech
as its channel moves away from the training near-field signature?

A model can anchor its keep class on the capture-chain signature of its training
near-field rows instead of on proximity. When it does, deletion grows monotonically
along the channel-domain ladder:

    dry clean -> synthetic near-RIR -> measured real near-RIR -> real end-to-end
    (same clean utterances convolved)                            (VOiCES near mics)

Every model input is RMS-normalized to the same level so the ONLY thing that
varies along the ladder is the channel. Reduction is measured on speech-active
spans of the input; keep-correct behaviour is ~0 dB everywhere. A steep ladder =
channel-signature anchoring; a channel-robust model keeps the ladder flat.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_keep_robustness.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt wide-ep19=egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
        --ckpt realE2E-ep19=... [--ckpt tag=path ...] \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_far_suppression import (  # noqa: E402
    active_spans,
    load_model,
    quantiles,
    reduction_db,
)

from puresound.audio.rir.bank.loader import PreGeneratedRoomBank

RECIPE_DIR = Path(__file__).resolve().parents[1]
SR = 16000
TARGET_RMS_DBFS = -28.0


def rms_normalize(wav: torch.Tensor, target_dbfs: float = TARGET_RMS_DBFS) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
    gain = 10.0 ** ((target_dbfs - 20.0 * torch.log10(rms)) / 20.0)
    return (wav * gain).clamp(-1.0, 1.0)


def convolve(wav: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    out = torchaudio.functional.fftconvolve(wav, rir, mode="full")
    return out[..., : wav.shape[-1]]


def load_clean_utts(metafile: Path, n: int, rng: random.Random, min_sec: float, max_sec: float):
    rows = []
    with open(metafile) as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            length, sr = int(parts[4]), int(parts[5])
            if sr == SR and length >= int(min_sec * SR):
                rows.append(parts[3])
    picks = rng.sample(rows, min(n, len(rows)))
    utts = []
    for p in picks:
        wav, sr = torchaudio.load(p)
        wav = wav[:1, : int(max_sec * SR)]
        utts.append(wav)
    return utts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", action="append", required=True,
                        help="tag=path; repeatable")
    parser.add_argument("--clean-metafile", default=str(RECIPE_DIR / "data/dns5-read.dev.list"))
    parser.add_argument("--syn-bank", default="/work/any_exp_link/puresound_exp/hybrid_rir_16k_levels/wide")
    parser.add_argument("--real-rir-manifest",
                        default="/work/any_exp_link/puresound_exp/real_rir_corpora/manifests_split/near_pool.train.jsonl")
    parser.add_argument("--voices-near-pool",
                        default=str(RECIPE_DIR / "data/realfar_pool/voices.near.heldout.jsonl"))
    parser.add_argument("--n-clean", type=int, default=30)
    parser.add_argument("--n-voices", type=int, default=40)
    parser.add_argument("--noise-folder",
                        default="/data/audio/dns-5/datasets_fullband_16k/noise_fullband")
    parser.add_argument("--noise-snr", type=float, default=10.0,
                        help="SNR for the +noise ladder variants. Keeping "
                        "isolated near speech tends to be channel-robust, while "
                        "deletion shows up only once the speech sits in a "
                        "mixture, so the diagnostic variants add the same noise "
                        "realization to every step of the ladder.")
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)

    print(f"# loading {args.n_clean} clean utts from {args.clean_metafile}", flush=True)
    cleans = load_clean_utts(Path(args.clean_metafile), args.n_clean, rng,
                             min_sec=4.0, max_sec=6.0)

    print("# indexing synthetic bank (near channels)...", flush=True)
    bank = PreGeneratedRoomBank(args.syn_bank)
    syn_rirs = []
    for _ in range(len(cleans)):
        scene = bank.sample_scene()
        imp, md, sr = bank.select_channel(scene, source_role="foreground")
        if sr != SR:
            imp = torchaudio.functional.resample(imp, sr, SR)
        syn_rirs.append((imp, md["source_receiver_distance"]))

    real_rirs = []
    entries = [json.loads(line) for line in open(args.real_rir_manifest)]
    entries = [e for e in entries if float(e.get("distance_m", 9.9)) < 1.0]
    for e in (entries if len(entries) <= len(cleans) else rng.sample(entries, len(cleans))):
        wav, sr = torchaudio.load(e["rir_path"])
        ch = min(int(e.get("channel", 0)), wav.shape[0] - 1)
        imp = wav[ch : ch + 1]
        if sr != SR:
            imp = torchaudio.functional.resample(imp, sr, SR)
        real_rirs.append((imp, float(e["distance_m"])))
    print(f"# {len(real_rirs)} measured real <1m RIRs", flush=True)

    voices_rows = [json.loads(line) for line in open(args.voices_near_pool)]
    voices_picks = rng.sample(voices_rows, min(args.n_voices, len(voices_rows)))
    voices_wavs = []
    for it in voices_picks:
        wav, sr = torchaudio.load(it["wav_path"])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        voices_wavs.append(wav[:1])
    print(f"# {len(voices_wavs)} VOiCES near recordings (held-out speakers)", flush=True)

    # Build the ladder: list of (variant_name, [(input, speech_ref)]). Same cleans
    # re-used for the three convolution variants; VOiCES rows are real recordings
    # (their own speech). Everything RMS-normalized to the same level.
    #
    # Two ladders:
    #   LONE:    input = speech only. Usually flat: keeping isolated near speech
    #            is channel-robust.
    #   +NOISE:  input = speech + noise @ --noise-snr (same noise realization for
    #            every step/model -> paired), the condition under which deletion
    #            actually appears. Spans/reference = the
    #            pre-noise speech, so we measure how much SPEECH survives (the
    #            residual-noise confound at SNR 10 inflates by <=+0.4 dB, far
    #            smaller than the deletion signal it is diagnosing).
    speech_sets: list[tuple[str, list[torch.Tensor]]] = [
        ("dry", [rms_normalize(w) for w in cleans]),
        ("syn_near_rir", [rms_normalize(convolve(w, syn_rirs[i % len(syn_rirs)][0]))
                          for i, w in enumerate(cleans)]),
        ("real_near_rir", [rms_normalize(convolve(w, real_rirs[i % len(real_rirs)][0]))
                           for i, w in enumerate(cleans)]),
        ("voices_near_e2e", [rms_normalize(w) for w in voices_wavs]),
    ]

    noise_files = sorted(Path(args.noise_folder).glob("*.wav"))
    noise_rng = random.Random(args.seed + 1)
    noise_picks = noise_rng.sample(noise_files, min(64, len(noise_files)))

    def add_noise(speech: torch.Tensor, idx: int, snr_db: float) -> torch.Tensor:
        nwav, nsr = torchaudio.load(noise_picks[idx % len(noise_picks)])
        nwav = nwav[:1]
        if nsr != SR:
            nwav = torchaudio.functional.resample(nwav, nsr, SR)
        L = speech.shape[-1]
        if nwav.shape[-1] < L:
            reps = L // nwav.shape[-1] + 1
            nwav = nwav.repeat(1, reps)
        nwav = nwav[..., :L]
        sp = speech.pow(2).mean()
        np_ = nwav.pow(2).mean().clamp_min(1e-10)
        nwav = nwav * torch.sqrt(sp / np_ / (10.0 ** (snr_db / 10.0)))
        return (speech + nwav).clamp(-1.0, 1.0)

    ladder: list[tuple[str, list[tuple[torch.Tensor, torch.Tensor]]]] = []
    for name, wavs in speech_sets:
        ladder.append((name, [(w, w) for w in wavs]))
    for name, wavs in speech_sets:
        ladder.append((f"{name}+noise", [(add_noise(w, i, args.noise_snr), w)
                                         for i, w in enumerate(wavs)]))

    results: dict[str, dict[str, list[float]]] = {}
    tags = []
    for spec in args.ckpt:
        tag, path = spec.split("=", 1)
        tags.append(tag)
        model = load_model(args.config_path, path, device)
        results[tag] = {}
        for variant, pairs in ladder:
            vals = []
            for wav, speech_ref in pairs:
                spans = active_spans(speech_ref, SR)
                if sum(b - a for a, b in spans) < SR:
                    continue
                with torch.no_grad():
                    enh = model(wav.to(device)).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
                vals.append(reduction_db(enh, speech_ref, spans))
            results[tag][variant] = vals
        print(f"# {tag} done", flush=True)

    print("\n# KEEP ladder: active-span reduction dB, median [p25, p75]; want ~0 everywhere")
    print("variant\t" + "\t".join(tags))
    for variant, _ in ladder:
        row = variant
        for tag in tags:
            vals = results[tag][variant]
            if vals:
                p25, med, p75 = quantiles(vals)
                row += f"\t{med:+.2f} [{p25:+.2f},{p75:+.2f}]"
            else:
                row += "\t-"
        print(row)
    print("# steep ladder = keep anchored on training channel signature; "
          "a channel-robust model stays near 0 at every step.")


if __name__ == "__main__":
    main()
