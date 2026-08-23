"""Interval-joint projection: can overlapped speech get targets too?

The solo-only recipe covers 37% of a meeting. Overlap regions cannot be fitted
one speaker at a time -- least squares would explain the OTHER voices with this
speaker's close-talk signal. The general form fixes that: cut the meeting at
every speaker-activity change point, and on each interval solve for ALL active
speakers' filters JOINTLY (device = sum_s filter_s * CT_s + noise). Solo
intervals are the n=1 case of the same estimator.

Batched over frequency bins, so the whole meeting runs in seconds.

Reports coverage and speech-bin residual drop split by SOLO vs OVERLAP, and
renders an OVERLAP-DENSE excerpt: device / solo-only projection (the v1 hole) /
joint projection / joint residual, all on one shared gain.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
DEVICE_WAV = MTG / "mc_rockfall_0/ch0.wav"
SR, NFFT, HOP, I_PAST, J_FUT = 16000, 512, 256, 13, 1
K = I_PAST + 1 + J_FUT
MIN_FRAMES_PER_UNKNOWN = 4.0
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

gt = json.load(open(MTG / "gt_transcription.json"))
meta = json.load(open(MTG / "gt_meeting_metadata.json"))
spk2ct = meta["ParticipantAliasToCtDevice"]
dev, _ = sf.read(DEVICE_WAV); dev = torch.from_numpy(dev).float(); N = len(dev)
win = torch.hann_window(NFFT)
stft = lambda x: torch.stft(x, NFFT, HOP, window=win, return_complex=True, center=True)
DEV = stft(dev); F, T = DEV.shape
fr = lambda t: max(0, min(T, int(t * SR / HOP)))

utts = [(u["speaker_id"], u["start_time"], u["end_time"]) for u in gt]
speakers = sorted({s for s, _, _ in utts} & set(spk2ct))
CTP = {}
for s in speakers:
    ct, _ = sf.read(MTG / "close_talk" / f"{spk2ct[s]}.wav")
    ct = torch.from_numpy(ct).float()
    ct = torch.nn.functional.pad(ct, (0, max(0, N - len(ct))))[:N]
    CTP[s] = torch.nn.functional.pad(stft(ct), (I_PAST, J_FUT))

# per-frame activity per speaker
act = {s: torch.zeros(T, dtype=torch.bool) for s in speakers}
for s, t0, t1 in utts:
    if s in act:
        act[s][fr(t0):fr(t1)] = True
n_active = torch.stack([act[s] for s in speakers]).sum(0)
solo_f, over_f = n_active == 1, n_active >= 2

# cut at every activity change
key = torch.stack([act[s] for s in speakers]).T.int()
change = [0] + [i for i in range(1, T) if not torch.equal(key[i], key[i-1])] + [T]

def taps(s, a, b):
    return torch.stack([CTP[s][:, k+a:k+b] for k in range(K)], -1)      # [F, T, K]

PROJ_J = torch.zeros_like(DEV)
covered = torch.zeros(T, dtype=torch.bool)
skipped = 0
for a, b in zip(change[:-1], change[1:]):
    live = [s for s in speakers if bool(act[s][a])]
    if not live or b - a < MIN_FRAMES_PER_UNKNOWN * K * len(live):
        skipped += b - a if live else 0
        continue
    z = torch.cat([taps(s, a, b) for s in live], -1)                    # [F, Tint, K*n]
    G = z.conj().transpose(-1, -2) @ z
    ridge = 3e-3 * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:, None, None]
    G = G + ridge * torch.eye(z.shape[-1])
    rhs = z.conj().transpose(-1, -2) @ DEV[:, a:b].unsqueeze(-1)
    PROJ_J[:, a:b] = (z @ torch.linalg.solve(G, rhs)).squeeze(-1)
    covered[a:b] = True

# solo-only reference (v1 factory)
PROJ_S = torch.zeros_like(DEV)
for a, b in zip(change[:-1], change[1:]):
    live = [s for s in speakers if bool(act[s][a])]
    if len(live) != 1 or b - a < MIN_FRAMES_PER_UNKNOWN * K:
        continue
    z = taps(live[0], a, b)
    G = z.conj().transpose(-1, -2) @ z
    G = G + 3e-3 * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:, None, None] * torch.eye(K)
    rhs = z.conj().transpose(-1, -2) @ DEV[:, a:b].unsqueeze(-1)
    PROJ_S[:, a:b] = (z @ torch.linalg.solve(G, rhs)).squeeze(-1)

silent = n_active == 0
floor_bin = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True)
def drop(P, mask):
    if not mask.any(): return float("nan")
    D, Pm = DEV[:, mask], P[:, mask]
    m = (D.abs() ** 2) >= 10.0 * floor_bin
    if not m.any(): return float("nan")
    return float(10 * torch.log10((D.abs()**2)[m].sum() / ((D - Pm).abs()**2)[m].sum().clamp_min(1e-12)))

print(f"meeting frames: solo {100*float(solo_f.float().mean()):.0f}% | "
      f"overlap {100*float(over_f.float().mean()):.0f}% | silent {100*float(silent.float().mean()):.0f}%")
print(f"joint projection covers {100*float(covered.float().mean()):.0f}% of frames "
      f"({100*float((covered & over_f).float().sum() / over_f.float().sum().clamp_min(1)):.0f}% of overlap frames)")
print(f"\n{'region':22s} {'solo-only v1':>13s} {'interval-joint':>15s}")
for name, msk in (("solo frames", solo_f), ("OVERLAP frames", over_f & covered), ("all speech", (solo_f | over_f) & covered)):
    print(f"{name:22s} {drop(PROJ_S, msk):13.2f} {drop(PROJ_J, msk):15.2f}   dB")

istft = lambda X: torch.istft(X, NFFT, HOP, window=win, length=N)
proj_j, proj_s = istft(PROJ_J), istft(PROJ_S)
resid_j = dev - proj_j

# OVERLAP-dense 20 s window
score = torch.nn.functional.avg_pool1d(over_f.float().view(1,1,-1), 1250, 1).view(-1)
c = int(score.argmax()) * HOP
a3, b3 = max(0, c), min(N, c + 20*SR)
g = dev[a3:b3].abs().max().clamp_min(1e-9)
for name, x in (("1_device", dev), ("2_proj_soloonly", proj_s), ("3_proj_joint", proj_j), ("4_resid_joint", resid_j)):
    sf.write(OUT / f"{name}.wav", (x[a3:b3] / g * 0.6).numpy().astype(np.float32), SR)
ov = float(over_f[fr(a3/SR):fr(b3/SR)].float().mean())
print(f"\nexcerpt {a3/SR:.0f}-{b3/SR:.0f}s, {100*ov:.0f}% overlap frames -> {OUT}")
