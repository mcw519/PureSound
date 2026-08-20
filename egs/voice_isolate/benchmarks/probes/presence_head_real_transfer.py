"""The v11 presence heads on the real field recordings -- the transfer test.

The stratified probe scores the heads on the distribution they trained on, where
a high number is nearly free: the 2026-07-10 gate head reached 0.751 in-domain
and never transferred to a real recording. This runs the SAME heads, with NOTHING
refitted, on the v3 field set -- different chain, different rooms, real end-to-end
captures instead of RIR convolution.

Truth is the clip's own role: near/double-talk clips have a user talking, lone-far
clips do not. Audible frames only (> -60 dBFS): a silent frame has no user to
detect and no bystander to reject.

`held_out` groups are reported SEPARATELY and never pooled into the headline --
180D is a device orientation nothing has been fitted on, so it is the only column
that speaks to generalisation rather than to memorisation.
"""
import argparse, json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
RECIPE = REPO / "egs/voice_isolate"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPE))

from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
AUDIBLE_DBFS = -60.0
SEG_S = 20.0


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    r = both.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def frame_dbfs(wav, n_frames, hop):
    w = wav.view(-1).numpy()
    return np.array([
        10 * np.log10(float((w[t * hop:(t + 1) * hop] ** 2).mean()) + 1e-12)
        if w[t * hop:(t + 1) * hop].size else -120.0 for t in range(n_frames)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.chdir(RECIPE)

    model = init_siso_model(load_recipe(a.config_path, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    sd = torch.load(a.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if model.backbone.vad_head is None:
        raise SystemExit("this checkpoint/config has no vad_head")
    model = model.eval().to(a.device)
    hop = model.encoder.hop_length

    windows = json.loads((CASES / "windows.json").read_text())
    rows = []
    with torch.no_grad():
        for clip, spec in windows.items():
            if clip.startswith("_") or clip.endswith("_session"):
                continue
            role = spec["role"]
            if "double-talk" in role:
                truth = 1
            elif "lone near" in role:
                truth = 1
            elif "lone far" in role:
                truth = 0
            else:
                continue
            wav, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                                   target_lvl=None, resample_to=16000)
            wav = wav.view(1, -1)
            near, bg = [], []
            for start in range(0, wav.shape[-1], int(SEG_S * sr)):
                seg = wav[..., start:start + int(SEG_S * sr)]
                if seg.shape[-1] < sr // 4:
                    break
                model(seg.to(a.device))
                near.append(model.backbone.last_vad_logits[0].float().cpu().numpy())
                b = model.backbone.last_background_vad_logits
                if b is not None:
                    bg.append(b[0].float().cpu().numpy())
            near = np.concatenate(near)
            bg = np.concatenate(bg) if bg else None
            e = frame_dbfs(wav, len(near), hop)
            aud = e > AUDIBLE_DBFS
            if aud.sum() < 10:
                continue
            rows.append(dict(clip=clip, group=spec["group"], truth=truth,
                             held_out=bool(spec.get("held_out")),
                             near=near[aud], bg=bg[aud] if bg is not None else None))

    def report(title, subset, key="near"):
        pres = np.concatenate([r[key] for r in subset if r["truth"] == 1] or [[]])
        absn = np.concatenate([r[key] for r in subset if r["truth"] == 0] or [[]])
        if len(pres) < 20 or len(absn) < 20:
            print(f"  {title:26s} -- thin ({len(pres)} pres / {len(absn)} abs) --")
            return
        print(f"  {title:26s} {len(pres):7d} {len(absn):7d} "
              f"{np.median(pres):9.2f} {np.median(absn):9.2f} {auc(pres, absn):7.3f}")

    for key, label in (("near", "NEAR-PRESENCE head"), ("bg", "BACKGROUND head")):
        if key == "bg" and rows[0]["bg"] is None:
            continue
        print(f"\n===== {label}: real field recordings, nothing refitted =====")
        print(f"  {'scope':26s} {'n_pres':>7s} {'n_abs':>7s} "
              f"{'med(pres)':>9s} {'med(abs)':>9s} {'AUC':>7s}")
        fitted = [r for r in rows if not r["held_out"]]
        report("fitted-on rooms (pooled)", fitted, key)
        for g in sorted({r["group"] for r in fitted}):
            report(f"  {g}", [r for r in fitted if r["group"] == g], key)
        held = [r for r in rows if r["held_out"]]
        if held:
            print("  " + "-" * 60)
            for g in sorted({r["group"] for r in held}):
                report(f"HELD OUT: {g}", [r for r in held if r["group"] == g], key)


if __name__ == "__main__":
    main()
