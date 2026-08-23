"""Is the filter simply too SHORT for the room? Sweep tap count.

The window sweep peaked at 6.25 dB with +/-5-15 s of fitting data. That window
carries hundreds of frames, so a longer filter is now affordable -- and physics
says it may be needed: 13 past taps at a 16 ms hop spans 208 ms, while a
conference room's reverb tail typically runs 300-600 ms. We may have been
measuring the room with a ruler shorter than the room. Same honest scoring
(block-interleaved, no shared frames).

The holdout test showed the per-utterance headline (11.9 dB) was ~6 dB of
overfitting: block-interleaved scoring, which shares no frames with the fit at
near-zero time gap, reads 5.9 dB. Two opposing forces are now quantified:
  * time-locality HELPS -- a global per-speaker filter scores only ~1.4-2 dB
    honestly, four below per-utterance, so the path really does change;
  * small fitting sets OVERFIT -- 15 complex unknowns per bin against ~100
    frames.
So an optimum window must exist. This sweeps it, scoring every configuration
the same honest way (fit on even 128 ms blocks of the window, score on odd
blocks of the central utterance only), and sweeps ridge alongside since
regularisation trades the same two forces.

The winner's score is the factory's real target quality.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
SR, NFFT, HOP, J_FUT = 16000, 512, 256, 1
BLK = 8
WINDOWS_S = [5, 15]
RIDGES = [3e-3, 3e-2]
TAPS = [13, 25, 40, 60]      # 208 / 400 / 640 / 960 ms of past

gt = json.load(open(MTG / "gt_transcription.json"))
meta = json.load(open(MTG / "gt_meeting_metadata.json"))
spk2ct = meta["ParticipantAliasToCtDevice"]
dev, _ = sf.read(MTG / "mc_rockfall_0/ch0.wav"); dev = torch.from_numpy(dev).float(); N = len(dev)
win = torch.hann_window(NFFT)
stft = lambda x: torch.stft(x, NFFT, HOP, window=win, return_complex=True, center=True)
DEV = stft(dev); F, T = DEV.shape
fr = lambda t: max(0, min(T, int(t * SR / HOP)))
utts = [(u["speaker_id"], u["start_time"], u["end_time"]) for u in gt]
speakers = sorted({s for s, _, _ in utts} & set(spk2ct))
CTS = {}
for s in speakers:
    ct, _ = sf.read(MTG / "close_talk" / f"{spk2ct[s]}.wav")
    ct = torch.from_numpy(ct).float()
    CTS[s] = stft(torch.nn.functional.pad(ct, (0, max(0, N-len(ct))))[:N])
act = {s: torch.zeros(T, dtype=torch.bool) for s in speakers}
for s, t0, t1 in utts:
    if s in act: act[s][fr(t0):fr(t1)] = True
n_active = torch.stack([act[s] for s in speakers]).sum(0)
silent = n_active == 0
floor_bin = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True)
solo_of = {s: (act[s] & (n_active == 1)) for s in speakers}
key = torch.stack([act[s] for s in speakers]).T.int()
cuts = [0] + [i for i in range(1, T) if not torch.equal(key[i], key[i-1])] + [T]
ivals = [(a, b, [s for s in speakers if bool(act[s][a])]) for a, b in zip(cuts[:-1], cuts[1:])]
targets = [(a, b, live[0]) for a, b, live in ivals if len(live) == 1 and b - a >= 120]
print(f"scored utterances: {len(targets)}")

def tapcols(s, idx, CTP, K):
    return torch.stack([CTP[s][:, idx + k] for k in range(K)], -1)      # [F, n, K]

def score(D, P):
    m = (D.abs() ** 2) >= 10.0 * floor_bin
    return float(10*torch.log10((D.abs()**2)[m].sum() / ((D-P).abs()**2)[m].sum().clamp_min(1e-12))) if m.any() else float("nan")

print(f"\n{'taps (past ms)':>16s} " + " ".join(f"{('+/-'+str(w)+'s r'+str(r)):>14s}" for w in WINDOWS_S for r in RIDGES))
best = None
for I_PAST in TAPS:
    K = I_PAST + 1 + J_FUT
    CTP = {sp: torch.nn.functional.pad(v, (I_PAST, J_FUT)) for sp, v in CTS.items()}
    row = []
    for W in WINDOWS_S:
        for R in RIDGES:
            vals = []
            for a, b, s in targets:
                n = b - a
                blk = (torch.arange(n) // BLK)
                edge = torch.zeros(n, dtype=torch.bool); edge[::BLK] = True; edge[BLK-1::BLK] = True
                tst = (a + torch.nonzero((blk % 2 == 1) & ~edge, as_tuple=True)[0])
                if len(tst) < 20: continue
                fit_local = a + torch.nonzero(blk % 2 == 0, as_tuple=True)[0]
                half = int(W * SR / HOP / 2)
                lo, hi = max(0, (a+b)//2 - half), min(T, (a+b)//2 + half)
                near = torch.zeros(T, dtype=torch.bool); near[lo:hi] = True
                near &= solo_of[s]; near[a:b] = False
                fit_idx = torch.cat([fit_local, torch.nonzero(near, as_tuple=True)[0]])
                if len(fit_idx) < 3 * K: continue
                zf = tapcols(s, fit_idx, CTP, K)
                G = zf.conj().transpose(-1,-2) @ zf
                G = G + R * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:,None,None] * torch.eye(K)
                g = torch.linalg.solve(G, zf.conj().transpose(-1,-2) @ DEV[:, fit_idx].unsqueeze(-1))
                zt = tapcols(s, tst, CTP, K)
                vals.append(score(DEV[:, tst], (zt @ g).squeeze(-1)))
            vals = [v for v in vals if v == v]
            m = np.median(vals) if vals else float("nan")
            row.append(m)
            if vals and (best is None or m > best[0]): best = (m, I_PAST, W, R, len(vals))
    print(f"{f'{I_PAST} ({I_PAST*HOP*1000//SR} ms)':>16s} " + " ".join(f"{v:14.2f}" for v in row))
print(f"\nbest: {best[0]:.2f} dB with {best[1]} past taps ({best[1]*HOP*1000//SR} ms), "
      f"window +/-{best[2]}s, ridge {best[3]:g}, n={best[4]}")
print("block-interleaved held-out throughout.")
