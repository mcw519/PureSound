"""Per-frame near-presence: is "someone is talking close by" readable frame by frame?

The utterance-level probe answered "is there a near user in this clip". A gate needs
"is there one right now". This reads the same frozen bottleneck without the time
pooling.

Material is the two STREAM recordings, which alternate near and far in labelled
spans -- 25 spans, thousands of frames. Frames inside one span are nearly identical,
so cross-validation splits by SPAN, never by frame; a random frame split would score
~1.0 on noise. The permutation null permutes span labels for the same reason.
"""
import argparse, json, pathlib, sys, warnings
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
    m.reload_checkpoint(torch.load(ckpt, map_location="cpu")["state_dict"], load_loss_func=False)
    m = m.to(device).eval()
    cap = {}
    m.backbone.dist_head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("x", inp[0].detach()))
    return m, cap


def frame_features(m, cap, wav, device, seg_s=20.0, sr=16000):
    """[C, T_total] pooled over frequency, stitched from segments.

    Segmented because a 135 s session at once does not fit comfortably; each
    segment is independent, which costs the model's 3-frame warm-up at every
    boundary. Those frames are a rounding error against thousands and the
    alternative (one 135 s forward) is what makes the card thrash.
    """
    out = []
    step = int(seg_s * sr)
    for start in range(0, wav.shape[-1], step):
        seg = wav[..., start:start + step]
        if seg.shape[-1] < sr // 4:
            break
        with torch.no_grad():
            m(seg.to(device))
        out.append(cap["x"].mean(dim=2)[0].cpu().numpy())   # [C, T_seg]
    return np.concatenate(out, axis=1)


def labelled_frames(feats, spec, duration_s):
    """Frames inside a keep span (label 1) or a suppress span (label 0).

    Frames in neither are dropped: the gaps between spans are transitions and the
    label there is genuinely unknown. `group` is the span index, which is the unit
    cross-validation is allowed to split on.
    """
    n_frames = feats.shape[1]
    fps = n_frames / duration_s
    X, y, group = [], [], []
    span_id = 0
    for label, key in ((1, "keep"), (0, "suppress")):
        for a, b in spec.get(key, []):
            lo, hi = int(a * fps), min(int(b * fps), n_frames)
            if hi - lo < 2:
                continue
            X.append(feats[:, lo:hi].T)
            y.append(np.full(hi - lo, label))
            group.append(np.full(hi - lo, span_id))
            span_id += 1
    return np.concatenate(X), np.concatenate(y), np.concatenate(group), fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pretrained_ckpt/dpcrn_v8.ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    m, cap = build(a.ckpt, dev)
    windows = json.loads((CASES / "windows.json").read_text())
    Xs, ys, gs = [], [], []
    offset = 0
    for clip in ("90d_session", "270d_session"):
        spec = windows[clip]
        wav, _ = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None,
                              resample_to=16000)
        feats = frame_features(m, cap, wav.view(1, -1), dev)
        X, y, g, fps = labelled_frames(feats, spec, spec["duration_s"])
        print(f"  {clip}: {feats.shape[1]} frames at {fps:.1f} fps -> "
              f"{len(y)} labelled ({int(y.sum())} near / {int((1-y).sum())} far), "
              f"{len(np.unique(g))} spans", flush=True)
        Xs.append(X); ys.append(y); gs.append(g + offset); offset += g.max() + 1
    np.savez_compressed(a.out, X=np.concatenate(Xs), y=np.concatenate(ys),
                        g=np.concatenate(gs))
    print("wrote", a.out)
