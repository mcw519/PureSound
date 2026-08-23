"""Is the projection's quality real, or in-sample overfitting?

Every number so far fitted the taps on an interval and scored them on the SAME
interval. With K=15 complex unknowns per frequency bin, part of the apparent
fit can be the filter absorbing that interval's noise realization. The
staleness curve (own 10.1 dB -> 1-5 s 4.5 dB) is consistent with either fast
physical drift OR overfitting, and those imply very different verdicts.

Split each long solo interval in half: fit on the first half, score on the
second (held-out, ~1 s away, no shared samples). Compare against fitting and
scoring on that same second half (in-sample). The gap IS the overfitting.
Also sweeps tap count, since fewer unknowns overfit less.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
SR, NFFT, HOP = 16000, 512, 256
OUT_TAPS = [(13, 1), (7, 1), (3, 1)]
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
CT = {}
for s in speakers:
    ct, _ = sf.read(MTG / "close_talk" / f"{spk2ct[s]}.wav")
    ct = torch.from_numpy(ct).float()
    CT[s] = stft(torch.nn.functional.pad(ct, (0, max(0, N - len(ct))))[:N])
act = {s: torch.zeros(T, dtype=torch.bool) for s in speakers}
for s, t0, t1 in utts:
    if s in act: act[s][fr(t0):fr(t1)] = True
n_active = torch.stack([act[s] for s in speakers]).sum(0)
silent = n_active == 0
floor_bin = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True)
key = torch.stack([act[s] for s in speakers]).T.int()
cuts = [0] + [i for i in range(1, T) if not torch.equal(key[i], key[i-1])] + [T]
ivals = [(a, b, [s for s in speakers if bool(act[s][a])]) for a, b in zip(cuts[:-1], cuts[1:])]

def score(D, P):
    m = (D.abs() ** 2) >= 10.0 * floor_bin
    if not m.any(): return float("nan")
    return float(10 * torch.log10((D.abs()**2)[m].sum() / ((D - P).abs()**2)[m].sum().clamp_min(1e-12)))

print(f"{'taps':>10s} {'n':>4s} {'in-sample':>10s} {'1s-later':>9s} {'gap':>6s} {'INTERLEAVED':>11s}")
for I_PAST, J_FUT in OUT_TAPS:
    K = I_PAST + 1 + J_FUT
    CTP = {s: torch.nn.functional.pad(v, (I_PAST, J_FUT)) for s, v in CT.items()}
    taps = lambda s, a, b: torch.stack([CTP[s][:, k+a:k+b] for k in range(K)], -1)
    def fit(s, a, b):
        z = taps(s, a, b)
        G = z.conj().transpose(-1,-2) @ z
        G = G + 3e-3 * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:,None,None] * torch.eye(K)
        return torch.linalg.solve(G, z.conj().transpose(-1,-2) @ DEV[:, a:b].unsqueeze(-1))
    def fit_idx(s, a, b, idx):
        z = taps(s, a, b)[:, idx]
        G = z.conj().transpose(-1,-2) @ z
        G = G + 3e-3 * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:,None,None] * torch.eye(K)
        return torch.linalg.solve(G, z.conj().transpose(-1,-2) @ DEV[:, a:b][:, idx].unsqueeze(-1))

    ins, held, inter = [], [], []
    BLK = 8                                    # 128 ms blocks: near-zero time gap
    for a, b, live in ivals:
        if len(live) != 1 or b - a < 8 * K:
            continue
        s = live[0]; mid = (a + b) // 2
        g1 = fit(s, a, mid)
        g2 = fit(s, mid, b)
        D2 = DEV[:, mid:b]; z2 = taps(s, mid, b)
        held.append(score(D2, (z2 @ g1).squeeze(-1)))
        ins.append(score(D2, (z2 @ g2).squeeze(-1)))
        # block-interleaved: fit on even blocks, score on odd blocks of the SAME span.
        # Drops the 2 frames flanking every boundary so the 50%-overlap STFT frames
        # of the fit set never touch the scored ones.
        n = b - a
        blk = torch.arange(n) // BLK
        fit_i = torch.nonzero((blk % 2 == 0), as_tuple=True)[0]
        tst_i = torch.nonzero((blk % 2 == 1), as_tuple=True)[0]
        edge = torch.zeros(n, dtype=torch.bool)
        edge[::BLK] = True; edge[BLK-1::BLK] = True
        tst_i = tst_i[~edge[tst_i]]
        if len(fit_i) < 4 * K or len(tst_i) < 20:
            continue
        gi = fit_idx(s, a, b, fit_i)
        zt = taps(s, a, b)[:, tst_i]
        inter.append(score(DEV[:, a:b][:, tst_i], (zt @ gi).squeeze(-1)))
    ins = [x for x in ins if x == x]; held = [x for x in held if x == x]
    inter = [x for x in inter if x == x]
    print(f"{f'{I_PAST}+1+{J_FUT}':>10s} {len(ins):4d} {np.median(ins):10.2f} {np.median(held):9.2f} "
          f"{np.median(ins)-np.median(held):6.2f} {np.median(inter):11.2f}")
print("\n1s-later  = fit on the utterance's first half, scored on its second half")
print("            (mixes overfitting AND ~1 s of physical drift)")
print("INTERLEAVED = fit on even 128 ms blocks, scored on odd ones, same span")
print("            (near-zero time gap, no shared frames) -> isolates OVERFITTING alone.")
print("If INTERLEAVED ~ in-sample, the 1 s drop is real drift, not overfitting.")
