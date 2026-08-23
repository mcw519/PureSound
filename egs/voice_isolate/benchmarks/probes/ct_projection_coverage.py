"""How much of a meeting can get a projected target -- and how fast do filters go stale?

Interval-joint estimation works wherever an interval is long enough
(11.19 dB on overlap frames) but only reaches 8% of overlap frames: natural
overlaps are short interjections. The fix needs no estimation at all --
TRANSPLANT each speaker's filter from their nearest long solo interval. That
has no length requirement, so coverage becomes a question of how fast a filter
goes stale, which this measures directly.

Reports the staleness curve (drop vs seconds since the source interval) and
final coverage/quality for the combined policy: joint where estimable,
transplant elsewhere. Renders the overlap-dense excerpt for listening.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
SR, NFFT, HOP, I_PAST, J_FUT = 16000, 512, 256, 13, 1
K = I_PAST + 1 + J_FUT
FPU = 4.0
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

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
CTP = {}
for s in speakers:
    ct, _ = sf.read(MTG / "close_talk" / f"{spk2ct[s]}.wav")
    ct = torch.from_numpy(ct).float()
    CTP[s] = torch.nn.functional.pad(stft(torch.nn.functional.pad(ct, (0, max(0, N-len(ct))))[:N]), (I_PAST, J_FUT))
act = {s: torch.zeros(T, dtype=torch.bool) for s in speakers}
for s, t0, t1 in utts:
    if s in act: act[s][fr(t0):fr(t1)] = True
n_active = torch.stack([act[s] for s in speakers]).sum(0)
solo_f, over_f, silent = n_active == 1, n_active >= 2, n_active == 0
key = torch.stack([act[s] for s in speakers]).T.int()
ivals = list(zip([0] + [i for i in range(1, T) if not torch.equal(key[i], key[i-1])],
                 [i for i in range(1, T) if not torch.equal(key[i], key[i-1])] + [T]))
taps = lambda s, a, b: torch.stack([CTP[s][:, k+a:k+b] for k in range(K)], -1)

def solve(z, d):
    G = z.conj().transpose(-1, -2) @ z
    G = G + 3e-3 * G.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:, None, None] * torch.eye(z.shape[-1])
    return torch.linalg.solve(G, z.conj().transpose(-1, -2) @ d.unsqueeze(-1))

# bank of per-speaker filters from long solo intervals
bank = {s: [] for s in speakers}
for a, b in ivals:
    live = [s for s in speakers if bool(act[s][a])]
    if len(live) == 1 and b - a >= FPU * K:
        bank[live[0]].append(((a + b) / 2, solve(taps(live[0], a, b), DEV[:, a:b])))
print("filter bank per speaker:", {s: len(v) for s, v in bank.items()})

def drop(P, mask):
    if not mask.any(): return float("nan")
    D, Pm = DEV[:, mask], P[:, mask]
    fb = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True)
    m = (D.abs() ** 2) >= 10.0 * fb
    return float(10*torch.log10((D.abs()**2)[m].sum() / ((D-Pm).abs()**2)[m].sum().clamp_min(1e-12))) if m.any() else float("nan")

# staleness: apply each solo interval's OWN filter vs a filter from k seconds away
stale = {}
for a, b in ivals:
    live = [s for s in speakers if bool(act[s][a])]
    if len(live) != 1 or b - a < FPU * K: continue
    s = live[0]; z = taps(s, a, b); ctr = (a + b) / 2
    for gc, g in bank[s]:
        gap = abs(gc - ctr) * HOP / SR
        bucket = 0 if gap < 1 else 1 if gap < 5 else 2 if gap < 20 else 3 if gap < 60 else 4
        P = torch.zeros_like(DEV); P[:, a:b] = (z @ g).squeeze(-1)
        m = torch.zeros(T, dtype=torch.bool); m[a:b] = True
        stale.setdefault(bucket, []).append(drop(P, m))
print(f"\n{'filter age':>14s} {'n':>5s} {'median drop':>12s}")
for k, lbl in ((0,"own (<1 s)"),(1,"1-5 s"),(2,"5-20 s"),(3,"20-60 s"),(4,">60 s")):
    v = [x for x in stale.get(k, []) if x == x]
    if v: print(f"{lbl:>14s} {len(v):5d} {np.median(v):12.2f} dB")

# combined policy: joint where estimable, transplant nearest otherwise
PROJ = torch.zeros_like(DEV); covered = torch.zeros(T, dtype=torch.bool); how = {}
for a, b in ivals:
    live = [s for s in speakers if bool(act[s][a])]
    if not live: continue
    if b - a >= FPU * K * len(live):
        z = torch.cat([taps(s, a, b) for s in live], -1)
        PROJ[:, a:b] = (z @ solve(z, DEV[:, a:b])).squeeze(-1); how[(a,b)] = "joint"
    else:
        acc = torch.zeros(F, b - a, dtype=DEV.dtype); ok = True
        for s in live:
            if not bank[s]: ok = False; break
            g = min(bank[s], key=lambda kv: abs(kv[0] - (a+b)/2))[1]
            acc += (taps(s, a, b) @ g).squeeze(-1)
        if not ok: continue
        PROJ[:, a:b] = acc; how[(a,b)] = "transplant"
    covered[a:b] = True

sp = solo_f | over_f
print(f"\ncoverage of speech frames: {100*float((covered & sp).float().sum()/sp.float().sum()):.0f}%"
      f"  (overlap frames: {100*float((covered & over_f).float().sum()/over_f.float().sum()):.0f}%)")
print(f"{'region':22s} {'drop':>8s}")
for name, m in (("solo", solo_f & covered), ("OVERLAP", over_f & covered), ("all speech", sp & covered)):
    print(f"{name:22s} {drop(PROJ, m):8.2f} dB")

istft = lambda X: torch.istft(X, NFFT, HOP, window=win, length=N)
proj = istft(PROJ); resid = dev - proj
score = torch.nn.functional.avg_pool1d(over_f.float().view(1,1,-1), 1250, 1).view(-1)
c = int(score.argmax()) * HOP; a3, b3 = max(0, c), min(N, c + 20*SR)
g = dev[a3:b3].abs().max().clamp_min(1e-9)
for name, x in (("1_device", dev), ("2_projection", proj), ("3_residual", resid)):
    sf.write(OUT / f"{name}.wav", (x[a3:b3]/g*0.6).numpy().astype(np.float32), SR)
print(f"excerpt {a3/SR:.0f}-{b3/SR:.0f}s ({100*float(over_f[fr(a3/SR):fr(b3/SR)].float().mean()):.0f}% overlap) -> {OUT}")
