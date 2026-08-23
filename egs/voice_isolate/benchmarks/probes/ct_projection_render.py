"""Render the PER-UTTERANCE projection (the recipe that passed) for listening.

ct_projection_poc.py measured per-utterance filters but only ever SYNTHESISED
the global-filter track, so the published excerpt undersold the method. This
builds the real thing: for every solo utterance, estimate the taps inside that
utterance and project it; regions with no solo utterance stay zero (factory v1
uses solo utterances only, so those regions are out of scope by design).

Prints solo-region-only numbers -- the ones that describe the factory's actual
product -- and writes device / projection / residual excerpts over the
solo-densest stretch, plus the same three for ONE isolated utterance.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
DEVICE_WAV = MTG / "mc_rockfall_0/ch0.wav"
SR, NFFT, HOP = 16000, 512, 256
I_PAST, J_FUT = 13, 1
K = I_PAST + 1 + J_FUT
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

gt = json.load(open(MTG / "gt_transcription.json"))
meta = json.load(open(MTG / "gt_meeting_metadata.json"))
spk2ct = meta["ParticipantAliasToCtDevice"]
dev, sr = sf.read(DEVICE_WAV); assert sr == SR
dev = torch.from_numpy(dev).float(); N = len(dev)
win = torch.hann_window(NFFT)
stft = lambda x: torch.stft(x, NFFT, HOP, window=win, return_complex=True, center=True)
istft = lambda X, n: torch.istft(X, NFFT, HOP, window=win, length=n)
DEV = stft(dev); F, T = DEV.shape
fr = lambda t: max(0, min(T, int(t * SR / HOP)))

utts = [(u["speaker_id"], u["start_time"], u["end_time"]) for u in gt]
speakers = sorted({s for s, _, _ in utts} & set(spk2ct))

def solos(spk):
    out = []
    for s, t0, t1 in utts:
        if s != spk or t1 - t0 < 1.0:
            continue
        if any(o < t1 and e > t0 for s2, o, e in utts if s2 != spk):
            continue
        out.append((t0, t1))
    return out

active = torch.zeros(T, dtype=torch.bool)
for _, t0, t1 in utts:
    active[fr(t0):fr(t1)] = True
silent = ~active

PROJ = torch.zeros_like(DEV)
solo_frames = torch.zeros(T, dtype=torch.bool)
spans = []
for spk in speakers:
    ct, _ = sf.read(MTG / "close_talk" / f"{spk2ct[spk]}.wav")
    ct = torch.from_numpy(ct).float()
    ct = torch.nn.functional.pad(ct, (0, max(0, N - len(ct))))[:N]
    CTp = torch.nn.functional.pad(stft(ct), (I_PAST, J_FUT))
    for t0, t1 in solos(spk):
        a, b = fr(t0), fr(t1)
        if b - a < 40:
            continue
        Du = DEV[:, a:b]
        for f in range(F):
            z = torch.stack([CTp[f, k + a:k + b] for k in range(K)], 1)
            G = z.conj().T @ z
            G += 3e-3 * torch.eye(K) * (G.diagonal().real.mean() + 1e-9)
            PROJ[f, a:b] += z @ torch.linalg.solve(G, z.conj().T @ Du[f])
        solo_frames[a:b] = True
        spans.append((spk, t0, t1))

proj = istft(PROJ, N)
resid = dev - proj

# solo-region numbers, the factory's actual product
floor_bin = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True)
Ds, Ps = DEV[:, solo_frames], PROJ[:, solo_frames]
m = (Ds.abs() ** 2) >= 10.0 * floor_bin
drop = 10 * torch.log10((Ds.abs() ** 2)[m].sum() / ((Ds - Ps).abs() ** 2)[m].sum())
smp = torch.zeros(N, dtype=torch.bool)
for _, t0, t1 in spans:
    smp[int(t0 * SR):min(N, int(t1 * SR))] = True
d_rms = float((dev[smp] ** 2).mean().sqrt()); r_rms = float((resid[smp] ** 2).mean().sqrt())
n_rms = float((resid[~active.repeat_interleave(HOP)[:N]] ** 2).mean().sqrt()) if (~active).any() else float("nan")
print(f"solo utterances projected: {len(spans)}  ({100*float(solo_frames.float().mean()):.0f}% of meeting frames)")
print(f"speech-bin residual drop (solo regions): {float(drop):.2f} dB")
print(f"device {20*np.log10(d_rms+1e-12):.1f} dB | residual {20*np.log10(r_rms+1e-12):.1f} dB "
      f"| room-noise-only {20*np.log10(n_rms+1e-12):.1f} dB")
print(f"-> residual sits {20*np.log10((r_rms+1e-12)/(d_rms+1e-12)):+.1f} dB under the device signal, "
      f"and {20*np.log10((r_rms+1e-12)/(n_rms+1e-12)):+.1f} dB above room noise")

def dump(tag, a, b):
    for name, x in (("1_device", dev), ("2_projection", proj), ("3_residual", resid)):
        y = x[a:b].clone()
        y = y / dev[a:b].abs().max().clamp_min(1e-9) * 0.6   # COMMON gain: levels comparable
        sf.write(OUT / f"{tag}_{name}.wav", y.numpy().astype(np.float32), SR)

sf_score = torch.nn.functional.avg_pool1d(solo_frames.float().view(1, 1, -1), 1250, 1).view(-1)
c = int(sf_score.argmax()) * HOP
dump("stretch", max(0, c), min(N, c + 20 * SR))
spk, t0, t1 = max(spans, key=lambda s: s[2] - s[1])
dump("oneutt", int(t0 * SR), min(N, int(t1 * SR)))
print(f"excerpts -> {OUT}  (stretch {c/SR:.0f}s; single utterance {spk} {t0:.0f}-{t1:.0f}s)")
print("NOTE: all three tracks share ONE gain, so the residual's loudness is directly comparable.")
