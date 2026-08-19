"""The same three blends on the COLD-START clips -- the actual defect.

The session result showed the gain from releasing the ceiling tracks how hard the
ceiling was binding: spans the mask already pushed to -18 dB gained 5 dB, spans it
only reached -6 dB gained 0.16 dB. The cold-start lone-bystander clips are the -1
to -2 dB case. If the ceiling was never binding there, releasing it buys nothing
on the one thing this is all for.
"""
import json, math, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
import numpy as np, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
from b_traj import integrate, role

S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))
CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
TAU_UP, TAU_DN, B_LO, B_HI = 0.05, 1.0, 0.35, 0.75
BASE, TOP = 0.9, 0.995


def span_dbfs(wav, spans, sr, limit):
    tot = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j > i:
            tot += float(wav[..., i:j].square().sum()); n += j - i
    return 10.0 * math.log10(tot / n + 1e-12) if n > 0 else float("nan")


if __name__ == "__main__":
    dev = torch.device("cuda:0")
    model = init_siso_model(load_recipe("config/infer_dpcrn.yaml",
                                        expected_task="voice_isolation").model)
    model.reload_checkpoint(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
                                       map_location="cpu")["state_dict"],
                            load_loss_func=False)
    model = model.to(dev).eval()
    windows = json.loads((CASES / "windows.json").read_text())
    probs = np.load(S / "probs.npz"); feats = np.load(S / "all_v8.npz")

    print(f"{'clip':>12s} {'role':>5s} {'median b':>9s} {'floor':>7s} {'in':>7s} | "
          f"{'fixed':>8s} {'b-driven':>9s} {'pure':>8s} | {'gain':>6s} {'resid':>7s}")
    print("-" * 92)
    tot = {"fixed": [], "b-driven": [], "pure": []}
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        r = role(clip, spec)
        spans = spec.get("suppress") or spec.get("keep")
        mix, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                               target_lvl=None, resample_to=16000)
        mix = mix.view(1, -1)
        with torch.no_grad():
            enh = model(mix.to(dev), dry_blend=1.0).detach().cpu().view(1, -1)
        n = min(enh.shape[-1], mix.shape[-1]); enh, mixn = enh[..., :n], mix[..., :n]
        fps = float(feats[f"{clip}_fps"][0])
        b = integrate(probs[clip], fps, TAU_UP, TAU_DN)
        d = BASE + np.clip((B_HI - b) / (B_HI - B_LO), 0, 1) * (TOP - BASE)
        hop = int(round(sr / fps))
        ds = torch.from_numpy(np.repeat(d, hop)).float()
        ds = (torch.cat([ds, ds[-1:].expand(max(0, n - len(ds)))]) if len(ds) < n
              else ds[:n]).view(1, -1)
        sysd = {"fixed": (BASE*enh + (1-BASE)*mixn).clamp(-1, 1),
                "b-driven": (ds*enh + (1-ds)*mixn).clamp(-1, 1),
                "pure": enh.clamp(-1, 1)}
        din = span_dbfs(mixn, spans, sr, n)
        v = {k: span_dbfs(w, spans, sr, n) for k, w in sysd.items()}
        floor = spec.get("floor_dbfs", float("nan"))
        if r == "far":
            for k in tot:
                tot[k].append(v[k] - din)
        print(f"{clip:>12s} {r:>5s} {np.median(b):9.3f} {floor:7.1f} {din:7.1f} | "
              f"{v['fixed']-din:8.2f} {v['b-driven']-din:9.2f} {v['pure']-din:8.2f} | "
              f"{v['b-driven']-v['fixed']:+6.2f} {v['b-driven']-floor:7.2f}")
    print("-" * 92)
    print(f"{'FAR median':>12s} {'':>5s} {'':>9s} {'':>7s} {'':>7s} | "
          f"{np.median(tot['fixed']):8.2f} {np.median(tot['b-driven']):9.2f} "
          f"{np.median(tot['pure']):8.2f} | "
          f"{np.median(tot['b-driven'])-np.median(tot['fixed']):+6.2f}")
    print("\n  gain  = b-driven minus fixed, dB deeper than today")
    print("  resid = b-driven output level above the recording's own noise floor")
