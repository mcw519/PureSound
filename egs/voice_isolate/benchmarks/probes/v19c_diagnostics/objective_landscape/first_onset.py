import glob, os, sys, json
import numpy as np, soundfile as sf
root = sys.argv[1]
HOP = 160
t0 = []
for f in sorted(glob.glob(os.path.join(root, "*_ref.wav"))):
    x, sr = sf.read(f, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    n = len(x)//HOP
    e = (x[:n*HOP].reshape(n,HOP)**2).sum(1)+1e-20
    act = (e > np.percentile(e,95)*10**-3.0) & (e > HOP*1e-6)
    idx = np.flatnonzero(act)
    if len(idx): t0.append(idx[0]*0.01)
t0 = np.array(t0)
print(json.dumps({"n": len(t0), "median_first_onset_s": float(np.median(t0)),
  "p90": float(np.percentile(t0,90)), "frac_ge_0.5s": float((t0>=0.5).mean()),
  "frac_ge_1s": float((t0>=1.0).mean()), "frac_ge_2s": float((t0>=2.0).mean()),
  "max": float(t0.max())}, indent=2))
