"""b as a GAIN instead of a blend, because the blend provably cannot do the job.

The cold-start result: b-driven blend gains a median 0.03 dB on the very clips the
whole exercise is for. Not because b is wrong -- b reads 0.05-0.48 on them, it has
them right -- but because the -20 dB ceiling was never what limited suppression
there. The mask reduces those clips by 0.39 dB. Releasing a ceiling that is not
binding does nothing.

A gain can:   out = g(b) * enh + (1 - g(b)) * 0     i.e. just g(b) * enh
with g flat at 1.0 above b_hi, so that segment stays bit-identical.

This is the actuator the 2026-07-10 gate-only experiment used, and that failed
because its readout could not tell near from far on real recordings (both ~0.9).
This readout can (0.958/0.969 held out by recording). So the question is whether
the actuator was ever the problem.

And the exposure question re-opens in a much sharper form: a blend release
provably cannot delete the user (worst case is the unblended mask, measured at
-0.08 dB). A GAIN can. The short near spans where b sinks are now real risk.
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
BASE = 0.9
KEEP_VIOLATION_DB = -3.0


def span_dbfs(wav, spans, sr, limit):
    tot = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j > i:
            tot += float(wav[..., i:j].square().sum()); n += j - i
    return 10.0 * math.log10(tot / n + 1e-12) if n > 0 else float("nan")


def curve(b, lo, hi, floor_gain):
    """1.0 above hi (bit-identical), floor_gain below lo, linear in dB between."""
    t = np.clip((hi - b) / (hi - lo), 0.0, 1.0)
    return 10.0 ** (t * (20.0 * np.log10(floor_gain)) / 20.0)


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
    GMIN = {"gain-20": 0.1, "gain-26": 0.05, "gain-40": 0.01}

    def systems(clip):
        mix, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                               target_lvl=None, resample_to=16000)
        mix = mix.view(1, -1)
        with torch.no_grad():
            enh = model(mix.to(dev), dry_blend=1.0).detach().cpu().view(1, -1)
        n = min(enh.shape[-1], mix.shape[-1]); enh, mixn = enh[..., :n], mix[..., :n]
        fps = float(feats[f"{clip}_fps"][0])
        b = integrate(probs[clip], fps, TAU_UP, TAU_DN)
        hop = int(round(sr / fps))

        def up(x):
            t = torch.from_numpy(np.repeat(x, hop)).float()
            return (torch.cat([t, t[-1:].expand(max(0, n - len(t)))]) if len(t) < n
                    else t[:n]).view(1, -1)

        out = {"fixed": (BASE*enh + (1-BASE)*mixn).clamp(-1, 1)}
        for name, gmin in GMIN.items():
            g = up(curve(b, B_LO, B_HI, gmin))
            out[name] = (g * (BASE*enh + (1-BASE)*mixn)).clamp(-1, 1)
        return out, mixn, b, sr, n

    names = ["fixed"] + list(GMIN)
    print("COLD START -- the defect.  suppress spans, dB vs input\n")
    print(f"{'clip':>12s} {'med b':>7s} " + " ".join(f"{k:>9s}" for k in names)
          + f"  {'resid(-26)':>11s}")
    agg = {k: [] for k in names}
    keepagg = {k: [] for k in names}
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        r = role(clip, spec)
        sysd, mixn, b, sr, n = systems(clip)
        spans = spec.get("suppress") or spec.get("keep")
        din = span_dbfs(mixn, spans, sr, n)
        v = {k: span_dbfs(w, spans, sr, n) - din for k, w in sysd.items()}
        if r == "far":
            for k in names:
                agg[k].append(v[k])
            print(f"{clip:>12s} {np.median(b):7.3f} " + " ".join(f"{v[k]:9.2f}" for k in names)
                  + f"  {span_dbfs(sysd['gain-26'], spans, sr, n) - spec['floor_dbfs']:11.2f}")
        else:
            for k in names:
                keepagg[k].append(v[k])
    print(f"{'MEDIAN':>12s} {'':>7s} " + " ".join(f"{np.median(agg[k]):9.2f}" for k in names))
    print("\nCOLD START -- keep (near + double-talk), dB vs input")
    print(f"{'':>12s} {'':>7s} " + " ".join(f"{k:>9s}" for k in names))
    print(f"{'MEDIAN':>12s} {'':>7s} " + " ".join(f"{np.median(keepagg[k]):9.2f}" for k in names))
    print(f"{'WORST':>12s} {'':>7s} " + " ".join(f"{min(keepagg[k]):9.2f}" for k in names))

    print("\n\nIN SESSION -- where a gain can actually hurt\n")
    for clip in ("90d_session", "270d_session"):
        spec = windows[clip]
        sysd, mixn, b, sr, n = systems(clip)
        fps = float(feats[f"{clip}_fps"][0])
        print(f"=== {clip} ===  keep spans")
        print(f"{'span':>14s} {'len':>6s} {'med b':>7s} " + " ".join(f"{k:>9s}" for k in names))
        for i, (x, y) in enumerate(spec.get("keep", [])):
            sp = [(x, y)]; din = span_dbfs(mixn, sp, sr, n)
            lo, hi = max(0, int(x*fps)), min(len(b), int(y*fps))
            v = {k: span_dbfs(w, sp, sr, n) - din for k, w in sysd.items()}
            bad = "  <-- KEEP-VIOLATION" if v["gain-26"] < KEEP_VIOLATION_DB else ""
            print(f"{'near '+str(i):>14s} {y-x:5.1f}s {np.median(b[lo:hi]):7.3f} "
                  + " ".join(f"{v[k]:9.2f}" for k in names) + bad)
        print(f"  suppress spans")
        for i, (x, y) in enumerate(spec.get("suppress", [])):
            sp = [(x, y)]; din = span_dbfs(mixn, sp, sr, n)
            lo, hi = max(0, int(x*fps)), min(len(b), int(y*fps))
            v = {k: span_dbfs(w, sp, sr, n) - din for k, w in sysd.items()}
            print(f"{'far  '+str(i):>14s} {y-x:5.1f}s {np.median(b[lo:hi]):7.3f} "
                  + " ".join(f"{v[k]:9.2f}" for k in names))
        print()
