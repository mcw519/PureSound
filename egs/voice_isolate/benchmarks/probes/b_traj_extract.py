"""Per-frame bottleneck features for EVERY field clip, sessions and cold start alike.

The per-frame probe only ever extracted the two session recordings; the cold-start
clips were extracted utterance-pooled. The b-trajectory question needs both on the
same grid, because the whole point is whether one continuous quantity behaves
correctly on both.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

CFG = "config/train_dpcrn.yaml"
CASES = pathlib.Path("data_report/field_cases/test_vector_cases")


def build(ckpt, device):
    m = init_siso_model(load_recipe(CFG, expected_task="voice_isolation",
                                    expected_purpose="train").model)
    m.reload_checkpoint(torch.load(ckpt, map_location="cpu")["state_dict"],
                        load_loss_func=False)
    m = m.to(device).eval()
    cap = {}
    m.backbone.dist_head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach()))
    return m, cap


def frame_features(m, cap, wav, device, seg_s=20.0, sr=16000):
    out = []
    step = int(seg_s * sr)
    for start in range(0, wav.shape[-1], step):
        seg = wav[..., start:start + step]
        if seg.shape[-1] < sr // 4:
            break
        with torch.no_grad():
            m(seg.to(device))
        out.append(cap["x"].mean(dim=2)[0].cpu().numpy())
    return np.concatenate(out, axis=1)


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    m, cap = build("pretrained_ckpt/dpcrn_v8.ckpt", dev)
    windows = json.loads((CASES / "windows.json").read_text())
    out = {}
    for clip, spec in windows.items():
        if clip.startswith("_"):
            continue
        wav, _ = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                              target_lvl=None, resample_to=16000)
        wav = wav.view(1, -1)
        feats = frame_features(m, cap, wav, dev)
        dur = spec.get("duration_s", wav.shape[-1] / 16000)
        fps = feats.shape[1] / dur
        hop = int(round(16000 / fps))
        w = wav[0].numpy()
        dbfs = np.array([
            10 * np.log10(float((w[t*hop:(t+1)*hop] ** 2).mean()) + 1e-12)
            if w[t*hop:(t+1)*hop].size else -120.0
            for t in range(feats.shape[1])])
        out[f"{clip}_X"] = feats.astype(np.float32)
        out[f"{clip}_dbfs"] = dbfs.astype(np.float32)
        out[f"{clip}_fps"] = np.array([fps])
        print(f"  {clip:16s} {feats.shape[1]:5d} frames  {dur:7.2f}s  {fps:6.2f} fps",
              flush=True)
    np.savez_compressed(sys.argv[1], **out)
    print("wrote", sys.argv[1])
