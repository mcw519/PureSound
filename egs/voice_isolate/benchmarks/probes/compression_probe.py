"""Does dynamic-range compression create the phantom near user?

Level was ruled out: renormalising the QVF clips by -14..-28 dB moved the
foreground estimate by 0.01 m. The next candidate is production processing --
publication clips are compressed, and compression flattens the envelope, which is
one of the near/far cues (a close talker has deeper peak-to-floor structure).

Applies the repo's OWN compressor (the |x|^p waveshaper in
`Augmentor.apply_media_coloring`, RMS-restored so level is not a confound) to the
DEVICE far clips at increasing strength, and reads the foreground slot. If
compression is the mechanism, the estimate walks from ~1.0 m toward the QVF
clips' ~0.7 m.
"""
import json, pathlib, sys
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
SRC = pathlib.Path("data_report/field_cases/test_vector_cases")
w = json.load(open(SRC / "windows.json"))
far_dev = [k for k, v in w.items()
           if not k.startswith(("_", "qvf")) and not k.endswith("_session")
           and "lone far" in v["role"]]

def compress(x, p):
    """The media-coloring waveshaper alone: no band-limiting, RMS restored."""
    t = torch.from_numpy(x).float()
    rms_in = t.pow(2).mean().sqrt().clamp_min(1e-8)
    peak = t.abs().amax().clamp_min(1e-8)
    y = torch.sign(t / peak) * (t / peak).abs().pow(p) * peak
    y = y * (rms_in / y.pow(2).mean().sqrt().clamp_min(1e-8))
    return y.numpy()

for p in (1.0, 0.8, 0.6, 0.4):
    out = pathlib.Path(f"/tmp/cases_comp_{p}")
    out.mkdir(parents=True, exist_ok=True)
    spec = {"_comment": f"device far clips, |x|^{p} compression, RMS restored"}
    for k in far_dev:
        x, sr = sf.read(SRC / f"{k}_raw.wav")
        sf.write(out / f"{k}_raw.wav", compress(x, p).astype(np.float32), sr)
        spec[k] = w[k]
    (out / "windows.json").write_text(json.dumps(spec, indent=1) + "\n")
print("built", [f"/tmp/cases_comp_{p}" for p in (1.0, 0.8, 0.6, 0.4)])
