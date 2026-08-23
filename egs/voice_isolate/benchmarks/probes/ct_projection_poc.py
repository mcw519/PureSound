"""CT->device projection POC (NOTSOFAR): can real-chain keep targets be made?

For each speaker: estimate a per-frequency FIR (I past + J future STFT taps,
FCP-style) mapping their close-talk mic to the distant device channel, on
their SOLO utterances; apply it to the whole CT track. The projection is that
speaker's image AT THE DEVICE -- aligned to the mixture, same capture chain,
no RIR convolution anywhere. Sum of images ~= the device mixture's speech.

Acceptance checks printed:
  A. solo fit      SI-SDR(projection, device) on the speaker's solo segments,
                   against the naive baseline (best single lag + scale on CT).
  B. sum test      SI-SDR(sum of projections, device) on speech-active frames;
                   residual level vs the meeting's silent-frame noise floor.
  C. bleed         on A's solo segments, level of the OTHER speakers'
                   projections relative to A's (cross-talk carried into targets).
  D. lag           gross CT->device lag per speaker (must sit inside the tap span).

Writes listening excerpts (device / summed projection / residual) beside the
cache. Run from egs/voice_isolate/. CPU only.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch

MTG = pathlib.Path("exp/real_e2e_corpora/notsofar/benchmark-datasets/train_set/240825.1_train/MTG/MTG_31060")
DEVICE_WAV = MTG / "mc_rockfall_0/ch0.wav"
SR, NFFT, HOP = 16000, 512, 256
I_PAST, J_FUT = 13, 1
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/ct_poc")
OUT.mkdir(parents=True, exist_ok=True)

gt = json.load(open(MTG / "gt_transcription.json"))
meta = json.load(open(MTG / "gt_meeting_metadata.json"))
spk2ct = meta["ParticipantAliasToCtDevice"]
speakers = sorted({u["speaker_id"] for u in gt} & set(spk2ct))

dev, sr = sf.read(DEVICE_WAV); assert sr == SR
dev = torch.from_numpy(dev).float()
N = len(dev)
win = torch.hann_window(NFFT)

def stft(x):
    return torch.stft(x, NFFT, HOP, window=win, return_complex=True, center=True)

def istft(X, length):
    return torch.istft(X, NFFT, HOP, window=win, length=length)

DEV = stft(dev)                                   # [F, T]
F, T = DEV.shape

def frames(t0, t1):
    return max(0, int(t0 * SR / HOP)), min(T, int(t1 * SR / HOP))

utts = [(u["speaker_id"], u["start_time"], u["end_time"]) for u in gt]
def solo_utts(spk):
    out = []
    for s, t0, t1 in utts:
        if s != spk or t1 - t0 < 1.0:
            continue
        if any(os_ < t1 and oe > t0 for s2, os_, oe in utts if s2 != spk):
            continue
        out.append((t0, t1))
    return out

active = torch.zeros(T, dtype=torch.bool)
for _, t0, t1 in utts:
    a, b = frames(t0, t1); active[a:b] = True
silent = ~active
noise_floor = float((DEV[:, silent].abs() ** 2).mean().sqrt()) if silent.any() else float("nan")

def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est * ref).sum() / (ref * ref).sum().clamp_min(1e-12)
    t = a * ref
    return float(10 * torch.log10((t * t).sum() / ((est - t) ** 2).sum().clamp_min(1e-12)))

print(f"meeting {meta['meeting_id']}  device {DEVICE_WAV.parent.name}/ch0  "
      f"dur {N/SR:.0f}s  noise floor (STFT rms) {20*np.log10(noise_floor+1e-12):.1f} dB")
projections = {}
report = []
for spk in speakers:
    ct_path = MTG / "close_talk" / f"{spk2ct[spk]}.wav"
    ct, _ = sf.read(ct_path)
    ct = torch.from_numpy(ct).float()
    ct = torch.nn.functional.pad(ct, (0, max(0, N - len(ct))))[:N]
    solos = solo_utts(spk)
    if not solos:
        print(f"{spk}: no solo utterances, skipped"); continue

    # D. gross lag on the longest solo (envelope xcorr)
    t0, t1 = max(solos, key=lambda ab: ab[1] - ab[0])
    a, b = int(t0 * SR), int(t1 * SR)
    e1 = torch.nn.functional.avg_pool1d(ct[a:b].abs().view(1,1,-1), 160, 160).view(-1)
    e2 = torch.nn.functional.avg_pool1d(dev[a:b].abs().view(1,1,-1), 160, 160).view(-1)
    e1, e2 = e1 - e1.mean(), e2 - e2.mean()
    xc = torch.nn.functional.conv1d(e2.view(1,1,-1), e1.view(1,1,-1), padding=len(e1)//2).view(-1)
    lag_ms = float((int(xc.argmax()) - len(e1)//2) * 160 / SR * 1000)

    CT = stft(ct)                                  # [F, T]
    K = I_PAST + 1 + J_FUT
    CTp = torch.nn.functional.pad(CT, (I_PAST, J_FUT))
    solo_mask = torch.zeros(T, dtype=torch.bool)
    for t0, t1 in solos:
        a2, b2 = frames(t0, t1); solo_mask[a2:b2] = True
    n_solo = int(solo_mask.sum())

    PROJ = torch.zeros_like(DEV)
    for f in range(F):                             # per-bin taps, solved on solo frames
        z = torch.stack([CTp[f, k:k+T] for k in range(K)], 1)      # [T, K]
        A = z[solo_mask]; y = DEV[f, solo_mask]
        G = A.conj().T @ A
        G += 1e-3 * torch.eye(K) * (G.diagonal().real.mean() + 1e-9)
        g = torch.linalg.solve(G, A.conj().T @ y)
        PROJ[f] = z @ g
    proj = istft(PROJ, N)
    projections[spk] = (proj, solo_mask, solos)

    # A. solo fit vs naive lag+scale baseline
    fits, bases, drops = [], [], []
    for t0, t1 in solos:
        a2, b2 = int(t0 * SR), int(t1 * SR)
        fits.append(si_sdr(proj[a2:b2], dev[a2:b2]))
        drops.append(float(10 * torch.log10(((dev[a2:b2]**2).mean()) /
                     (((dev[a2:b2] - proj[a2:b2])**2).mean()).clamp_min(1e-12))))
        s = int(lag_ms / 1000 * SR)
        ct_shift = torch.roll(ct, s)
        bases.append(si_sdr(ct_shift[a2:b2], dev[a2:b2]))
    # PER-UTTERANCE filters: re-estimate the taps inside each solo utterance
    # alone and project just that utterance -- the NTT recipe. If this drop far
    # exceeds the global-filter drop, the global filter's miss is time-variance
    # (movement / clock drift), not a modeling-family limit.
    pu_drops = []
    floor_bin_ = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True) if silent.any() else None
    for t0, t1 in solos:
        a2, b2 = frames(t0, t1)
        if b2 - a2 < 40:
            continue
        Du = DEV[:, a2:b2]
        Pu = torch.zeros_like(Du)
        for f in range(F):
            z = torch.stack([CTp[f, k + a2:k + b2] for k in range(K)], 1)
            G = z.conj().T @ z
            G += 3e-3 * torch.eye(K) * (G.diagonal().real.mean() + 1e-9)
            g = torch.linalg.solve(G, z.conj().T @ Du[f])
            Pu[f] = z @ g
        if floor_bin_ is not None:
            m = (Du.abs() ** 2) >= 10.0 * floor_bin_
            if m.any():
                pu_drops.append(float(10 * torch.log10(
                    (Du.abs() ** 2)[m].sum() / ((Du - Pu).abs() ** 2)[m].sum().clamp_min(1e-12))))
    pu_med = float(np.median(pu_drops)) if pu_drops else float("nan")

    # speech-dominant-bin residual drop: only T-F bins where the device carries
    # real speech energy (>= floor + 10 dB) on this speaker's solo frames --
    # separates "projection misses speech" from "device is mostly noise there".
    floor_bin = (DEV[:, silent].abs() ** 2).mean(1, keepdim=True) if silent.any() else None
    sd_drop = float("nan")
    if floor_bin is not None:
        Dsolo = DEV[:, solo_mask]; Psolo = PROJ[:, solo_mask]
        m = (Dsolo.abs() ** 2) >= 10.0 * floor_bin
        if m.any():
            num = (Dsolo.abs() ** 2)[m].sum()
            den = ((Dsolo - Psolo).abs() ** 2)[m].sum().clamp_min(1e-12)
            sd_drop = float(10 * torch.log10(num / den))
    report.append((spk, len(solos), n_solo, lag_ms, float(np.median(fits)), float(np.median(bases)), float(np.median(drops)), sd_drop, pu_med))

print(f"\n{'speaker':10s} {'solos':>5s} {'lag ms':>7s} {'proj SI-SDR':>11s} {'naive':>7s} {'global drop':>11s} {'speech-bin':>10s} {'PER-UTT bin':>11s}")
for spk, ns, nf, lag, fit, base, drop, sd, pu in report:
    print(f"{spk:10s} {ns:5d} {lag:7.1f} {fit:11.2f} {base:7.2f} {drop:11.2f} {sd:10.2f} {pu:11.2f}")

# B. sum test
sum_proj = sum(p for p, _, _ in projections.values())
act_samples = torch.zeros(N, dtype=torch.bool)
for _, t0, t1 in utts:
    act_samples[int(t0*SR):min(N, int(t1*SR))] = True
sisdr_sum = si_sdr(sum_proj[act_samples], dev[act_samples])
resid = dev - sum_proj
r_act = float((resid[act_samples]**2).mean().sqrt())
r_sil = float((resid[~act_samples]**2).mean().sqrt()) if (~act_samples).any() else float("nan")
print(f"\nsum test (speech-active): SI-SDR {sisdr_sum:.2f} dB | residual rms {20*np.log10(r_act+1e-12):.1f} dB "
      f"vs silent-frame floor {20*np.log10(r_sil+1e-12):.1f} dB (gap {20*np.log10((r_act+1e-12)/(r_sil+1e-12)):+.1f} dB)")

# C. bleed
print(f"\n{'bleed into':12s} " + " ".join(f"{s:>9s}" for s in projections))
for spk, (p, mask, solos) in projections.items():
    row = []
    a_ref = None
    for spk2, (p2, _, _) in projections.items():
        e = 0.0; n = 0
        for t0, t1 in solos:
            a2, b2 = int(t0*SR), int(t1*SR)
            e += float((p2[a2:b2]**2).sum()); n += b2 - a2
        rms = (e / max(n,1)) ** 0.5
        if spk2 == spk: a_ref = rms
        row.append(rms)
    print(f"{spk+':':12s} " + " ".join(f"{20*np.log10((r+1e-12)/(a_ref+1e-12)):+9.1f}" for r in row))
print("  (each row: during that speaker's solos, other projections' level rel. to his own, dB)")

# listening excerpts: 20 s around the busiest stretch
act_f = active.float()
score = torch.nn.functional.avg_pool1d(act_f.view(1,1,-1), 1250, 1).view(-1)  # ~20s windows
c = int(score.argmax()) * HOP
a3, b3 = max(0, c), min(N, c + 20*SR)
for name, x in (("1_device_mix", dev), ("2_projection_sum", sum_proj), ("3_residual", resid)):
    y = x[a3:b3]; y = y / y.abs().max().clamp_min(1e-9) * 0.6
    sf.write(OUT / f"{name}.wav", y.numpy().astype(np.float32), SR)
print(f"\nlistening excerpts ({a3/SR:.0f}-{b3/SR:.0f}s) -> {OUT}")
