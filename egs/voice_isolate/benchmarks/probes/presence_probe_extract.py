"""Frozen-bottleneck features + near-presence labels, from both domains.

(c): does the bottleneck of a model trained WITH real data carry "is there a near
user" -- the question 2026-07-10's gate-only experiment answered no for a model
trained without it.
"""
import argparse, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
from puresound.task.voice_isolation import VoiceIsolationDataset

CFG = "config/train_dpcrn.yaml"


def build(ckpt, device):
    m = init_siso_model(load_recipe(CFG, expected_task="voice_isolation",
                                    expected_purpose="train").model)
    ck = torch.load(ckpt, map_location="cpu")
    m.reload_checkpoint(ck["state_dict"], load_loss_func=False)
    m = m.to(device).eval()
    cap = {}
    m.backbone.dist_head.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0].detach()))
    return m, cap


def feats(m, cap, wav, device):
    with torch.no_grad():
        m(wav.to(device))
    x = cap["x"]                                  # [N, C, F, T]
    return x.mean(dim=(2, 3)).cpu().numpy()       # same pooling DistHead uses


def real_set(m, cap, device, cases="data_report/field_cases/test_vector_cases"):
    import pathlib
    cases = pathlib.Path(cases)
    windows = {k: v for k, v in json.loads((cases / "windows.json").read_text()).items()
               if not k.startswith("_")}
    X, y, names = [], [], []
    for clip, spec in windows.items():
        if "session" in clip:
            continue                              # long sessions mix both; excluded
        p = cases / f"{clip}_raw.wav"
        if not p.is_file():
            continue
        wav, _ = AudioIO.open(f_path=str(p), target_lvl=None, resample_to=16000)
        wav = wav.view(1, -1)[..., : 20 * 16000]
        # present = a near user is talking in this clip (lone-near or double-talk);
        # absent = lone bystander only.
        present = "keep" in spec
        X.append(feats(m, cap, wav, device)[0]); y.append(int(present)); names.append(clip)
    return np.stack(X), np.array(y), names


def _dataset(lone_far_prob):
    """The shipped recipe with the absent-row rate forced.

    Absent rows come from `realfar.lone_far_prob` (shipped 0.20 x 0.15 = 3% of
    items), so the two classes are generated in separate passes rather than hoping
    a random draw balances them. Everything else is the shipped recipe.
    """
    from puresound.config import with_overrides

    recipe = load_recipe(CFG, expected_task="voice_isolation", expected_purpose="train")
    recipe = with_overrides(
        recipe, augmentation_realfar={"prob": 1.0, "lone_far_prob": lone_far_prob}
    )
    c = recipe.dataset
    return VoiceIsolationDataset(
        metafile_path=c.valid_metafile, min_utt_length_in_seconds=c.filter_min_utterance_length,
        min_utts_in_each_speaker=c.filter_min_utterance_per_speaker, target_sr=c.target_sample_rate,
        training_sample_length_in_seconds=c.training_length_seconds,
        audio_gain_normalized_to=c.gain_normalized_to,
        dataset_role="validation", pipeline_role="train", **recipe.augmentation_kwargs())


def synth_set(m, cap, device, n_items):
    X, y = [], []
    for lone_far_prob, seed0, tag in ((0.0, 700000, "present"), (1.0, 800000, "absent")):
        ds = _dataset(lone_far_prob)
        spks = sorted(ds.total_spks)
        kept = 0
        for i in range(n_items):
            item = ds[(spks[i % len(spks)], 16000, seed0 + i)]
            # Trust the row, not the knob: `lone_far_prob` gates a row type, and a
            # row can end up with a silent target for other reasons.
            present = float(item["clean_speech"].abs().amax()) > 0
            if present != (tag == "present"):
                continue
            X.append(feats(m, cap, item["noisy_speech"].view(1, -1), device)[0])
            y.append(int(present)); kept += 1
        print(f"    synthetic {tag}: kept {kept}/{n_items}", flush=True)
    return np.stack(X), np.array(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="pretrained_ckpt/dpcrn_v10.ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-synth", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    m, cap = build(a.ckpt, dev)
    Xr, yr, names = real_set(m, cap, dev)
    print(f"  real: {Xr.shape}  present {int(yr.sum())} / absent {int((1-yr).sum())}", flush=True)
    Xs, ys = synth_set(m, cap, dev, a.n_synth)
    print(f"  synthetic: {Xs.shape}  present {int(ys.sum())} / absent {int((1-ys).sum())}", flush=True)
    np.savez_compressed(a.out, Xr=Xr, yr=yr, names=np.array(names), Xs=Xs, ys=ys)
    print("wrote", a.out)
