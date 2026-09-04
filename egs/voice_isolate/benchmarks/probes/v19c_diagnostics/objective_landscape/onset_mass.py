"""How much of the loss mass sits in the first second after a foreground onset?

Proxy for "what fraction of an energy-pooled loss (SI-SDR / MR-STFT / OverSuppression)
is spent on onset frames" -- the frames the field probes say get deleted.

Reads data_report/wer_set_moderate_test/*_ref.wav (ref = near-reverb foreground).
Frame grid = 10 ms (the model's bottleneck rate). Activity = frame energy within
30 dB of the file's 95th-percentile frame energy AND above -60 dBFS.
"""
import glob, json, os, sys
import numpy as np
import soundfile as sf

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*_ref.wav")))
HOP = 160  # 10 ms @16k
rows = []
for f in files:
    x, sr = sf.read(f, dtype="float32")
    assert sr == 16000, sr
    if x.ndim > 1:
        x = x.mean(1)
    n = len(x) // HOP
    if n < 50:
        continue
    fr = x[: n * HOP].reshape(n, HOP)
    e = (fr ** 2).sum(1) + 1e-20
    ref95 = np.percentile(e, 95)
    act = (e > ref95 * 10 ** (-30 / 10)) & (e > HOP * 10 ** (-60 / 10))
    if act.sum() < 20:
        continue
    # onsets = active frame preceded by >= G frames of inactivity
    out = {"id": os.path.basename(f)[:-8], "dur_s": len(x) / sr,
           "active_s": act.sum() * 0.01}
    for gap_frames, tag in ((20, "g200ms"), (50, "g500ms"), (100, "g1s")):
        onset_idx = []
        run = gap_frames  # treat file start as preceded by silence
        for i, a in enumerate(act):
            if a:
                if run >= gap_frames:
                    onset_idx.append(i)
                run = 0
            else:
                run += 1
        m = np.zeros(n, bool)
        for i in onset_idx:
            m[i : i + 100] = True   # first 1.0 s after each onset
        m &= act
        out[f"n_onset_{tag}"] = len(onset_idx)
        out[f"frac_frames_{tag}"] = float(m.sum() / max(1, act.sum()))
        out[f"frac_energy_{tag}"] = float(e[m].sum() / e[act].sum())
    # longest internal silence with speech on both sides
    gaps, run = [], 0
    seen = False
    for a in act:
        if a:
            if seen and run > 0:
                gaps.append(run)
            seen, run = True, 0
        else:
            run += 1
    out["max_internal_gap_s"] = (max(gaps) * 0.01) if gaps else 0.0
    rows.append(out)

def med(k):
    return float(np.median([r[k] for r in rows]))

print(json.dumps({
    "n_files": len(rows),
    "median_dur_s": med("dur_s"),
    "median_active_s": med("active_s"),
    "median_max_internal_gap_s": med("max_internal_gap_s"),
    "frac_files_with_gap_ge_1s": float(np.mean([r["max_internal_gap_s"] >= 1.0 for r in rows])),
    "frac_files_with_gap_ge_3s": float(np.mean([r["max_internal_gap_s"] >= 3.0 for r in rows])),
    "frac_files_with_gap_ge_5s": float(np.mean([r["max_internal_gap_s"] >= 5.0 for r in rows])),
    **{k: med(k) for k in rows[0] if k.startswith(("frac_", "n_onset_"))},
}, indent=2))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "onset_mass_rows.json"), "w") as fh:
    json.dump(rows, fh)
