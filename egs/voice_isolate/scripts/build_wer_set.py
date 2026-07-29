"""Build a FROZEN real-RIR WER benchmark with REAL transcripts.

Foreground = LibriTTS test-clean utterances (ground-truth transcripts, clean English),
convolved with a BUT real-RIR NEAR channel (<1m). Interferers = other LibriTTS speakers
convolved with the same room's FAR channels (>1.5m). Plus DNS5 noise. All real acoustics.

Freezes per item: <id>_mix.wav, <id>_ref.wav (near-reverbed foreground = the speaker we
want to keep), and manifest.jsonl with the REAL transcript + scene metadata. WER ground
truth is the LibriTTS transcript (NOT whisper) -> reliable. eval_wer.py scores a model.

Usage:
    uv run python scripts/build_wer_set.py --n-items 200 \
        --out data_report/but_wer_set --seed 1234
"""
from __future__ import annotations
import argparse, sys, json, random, glob, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import numpy as np, soundfile as sf, librosa  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402
from puresound.audio.rir_bank import PreGeneratedRoomBank  # noqa: E402

# Default corpus locations -- overridable via CLI so the script is not pinned to one box.
DEFAULT_LT = "/data/audio/LibriTTS/test-clean"          # transcribed foreground corpus
DEFAULT_NOISE = "/data/audio/dns-5/datasets_fullband_16k/noise_fullband"
DEFAULT_RIR = "/work/any_exp_link/puresound_exp/but_real_rir_16k"  # PreGeneratedRoomBank folder
SR = 16000

def load16k(path, max_s=10.0, min_s=4.0):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(1)
    if sr != SR: y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    y = y.astype(np.float32)
    if len(y) < int(min_s*SR): return None
    return y[:int(max_s*SR)]

def rms(x): return float(np.sqrt(np.mean(x**2) + 1e-12))
def conv(sig, rir):  # rir: [L]
    out = fftconvolve(sig, rir)[:len(sig)]; return out.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--out", default="data_report/but_wer_set")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sir-range", type=float, nargs=2, default=[-5.0, 10.0])
    ap.add_argument("--snr-range", type=float, nargs=2, default=[5.0, 20.0])
    ap.add_argument("--libritts-dir", default=DEFAULT_LT, help="transcribed foreground corpus (LibriTTS-style: <spk>/<chap>/*.wav + *.normalized.txt)")
    ap.add_argument("--noise-dir", default=DEFAULT_NOISE, help="folder of noise *.wav")
    ap.add_argument("--rir-dir", default=DEFAULT_RIR, help="PreGeneratedRoomBank folder of real RIRs (near/far labelled)")
    args = ap.parse_args()
    LT, NOISE, RIR = args.libritts_dir, args.noise_dir, args.rir_dir
    rng = random.Random(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # LibriTTS utts with transcripts, grouped by speaker
    wavs = sorted(glob.glob(f"{LT}/*/*/*.wav"))
    by_spk = {}
    for w in wavs:
        txt = w[:-4] + ".normalized.txt"
        if not os.path.exists(txt): continue
        spk = Path(w).parts[-3]
        by_spk.setdefault(spk, []).append(w)
    spks = sorted(by_spk); rng.shuffle(spks)
    print(f"LibriTTS test-clean: {len(spks)} speakers, {sum(len(v) for v in by_spk.values())} utts")

    bank = PreGeneratedRoomBank(folder=RIR, near_labels=["near_0","near_1"],
                                far_labels=["far_0","far_1","far_2"], drr_window_ms=2.5, cache_size=16)
    noises = sorted(glob.glob(f"{NOISE}/*.wav"))
    print(f"BUT rooms: {len(bank)}, noises: {len(noises)}")

    man = open(out / "manifest.jsonl", "w", encoding="utf-8")
    made = 0; si = 0; used_fg = set()
    while made < args.n_items and si < args.n_items*8:
        spk = spks[si % len(spks)]; si += 1
        cands = [w for w in by_spk[spk] if w not in used_fg]
        rng.shuffle(cands)
        fg_wav = None
        for c in cands:
            y = load16k(c)
            if y is not None: used_fg.add(c); fg_wav = (c, y); break
        if fg_wav is None: continue
        fg_path, fg = fg_wav
        transcript = open(fg_path[:-4] + ".normalized.txt", encoding="utf-8").read().strip()

        scene = bank.sample_scene()
        nrir, nmeta, _ = bank.select_channel(scene, "foreground")
        near = conv(fg, nrir.reshape(-1).numpy())
        ref = near.copy()                                # keep target = near-reverbed foreground
        nrms = rms(near)
        # interferers: other speakers via far channels
        k = rng.choice([1, 1, 2])                        # 1-2 interferers
        itf_sum = np.zeros_like(near)
        n_far_used = 0
        for _ in range(k):
            oy = None
            for _try in range(8):                        # retry: many LibriTTS utts are short
                ospk = rng.choice([s for s in spks if s != spk])
                ow = rng.choice(by_spk[ospk]); oy = load16k(ow, min_s=1.5)
                if oy is not None: break
            if oy is None: continue
            oy = np.resize(oy, len(near))
            try: frir, fmeta, _ = bank.select_channel(scene, "interferer")
            except Exception: break
            far = conv(oy, frir.reshape(-1).numpy())
            sir = rng.uniform(*args.sir_range)           # foreground-to-interferer ratio
            scale = nrms / (rms(far) + 1e-9) / (10**(sir/20))
            itf_sum += scale * far; n_far_used += 1
        # noise
        nz = load16k(rng.choice(noises), max_s=10.0, min_s=0.5)
        if nz is None: nz = np.zeros_like(near)
        nz = np.resize(nz, len(near))
        snr = rng.uniform(*args.snr_range)
        nz = nz * (nrms / (rms(nz)+1e-9) / (10**(snr/20)))
        mix = near + itf_sum + nz
        peak = np.max(np.abs(mix)) + 1e-9
        if peak > 0.99: mix = mix * (0.99/peak); ref = ref * (0.99/peak)

        iid = f"butw_{made:04d}"
        sf.write(out / f"{iid}_mix.wav", mix.astype(np.float32), SR)
        sf.write(out / f"{iid}_ref.wav", ref.astype(np.float32), SR)
        man.write(json.dumps({"id": iid, "transcript": transcript, "spk": spk,
                              "near_dist": nmeta["source_receiver_distance"], "rt60": nmeta["rt60"],
                              "n_interferers": n_far_used, "fg_path": fg_path}, ensure_ascii=False) + "\n")
        man.flush(); made += 1
        if made % 25 == 0: print(f"  built {made}/{args.n_items}", flush=True)
    man.close()
    print(f"DONE: froze {made} items -> {out.resolve()}")

if __name__ == "__main__":
    main()
