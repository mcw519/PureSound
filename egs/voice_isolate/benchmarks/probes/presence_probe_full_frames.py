"""Full per-frame features plus per-frame waveform energy, labels derived later.

`frames.py` saved only the labelled frames, which loses the gaps and makes a label
shift impossible to test. This keeps everything and leaves labelling to the
analysis, so the span boundaries can be moved.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, torch
from puresound.audio.io import AudioIO
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from frames import build, frame_features, CASES

if __name__ == "__main__":
    dev = torch.device("cuda:0")
    m, cap = build("pretrained_ckpt/dpcrn_v8.ckpt", dev)
    windows = json.loads((CASES / "windows.json").read_text())
    out = {}
    for clip in ("90d_session", "270d_session"):
        wav, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None,
                               resample_to=16000)
        wav = wav.view(1, -1)
        feats = frame_features(m, cap, wav, dev)            # [C, T]
        fps = feats.shape[1] / windows[clip]["duration_s"]
        # Per-frame input energy, on the same grid: what the model had to work with.
        hop = int(round(16000 / fps))
        n = feats.shape[1]
        pad = np.zeros(n)
        w = wav[0].numpy()
        for t in range(n):
            seg = w[t * hop:(t + 1) * hop]
            pad[t] = 10 * np.log10(float((seg ** 2).mean()) + 1e-12) if seg.size else -120
        out[f"{clip}_X"] = feats.astype(np.float32)
        out[f"{clip}_dbfs"] = pad.astype(np.float32)
        out[f"{clip}_fps"] = np.array([fps])
        print(f"  {clip}: {feats.shape} at {fps:.2f} fps, "
              f"energy p10 {np.percentile(pad,10):.1f} / p90 {np.percentile(pad,90):.1f} dBFS",
              flush=True)
    np.savez_compressed(f"{sys.argv[1]}", **out)
    print("wrote", sys.argv[1])
