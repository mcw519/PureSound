"""The operating curve: b_hi against what it costs and what it buys.

b_hi is the single decision parameter -- the edge of the dead zone. Above it the
output is bit-identical to what ships. Move it up and more bystander material gets
suppressed; move it up too far and in-session near spans, where the readout is
weak on short turns, start getting gated on the user.

Caches the forward pass so the sweep is free.
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
TAU_UP, TAU_DN, GMIN, BASE = 0.05, 1.0, 0.05, 0.9
KEEP_VIOLATION_DB = -3.0


def span_dbfs(wav, spans, sr, limit):
    tot = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j > i:
            tot += float(wav[..., i:j].square().sum()); n += j - i
    return 10.0 * math.log10(tot / n + 1e-12) if n > 0 else float("nan")


dev = torch.device("cuda:0")
model = init_siso_model(load_recipe("config/infer_dpcrn.yaml",
                                    expected_task="voice_isolation").model)
model.reload_checkpoint(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
                                   map_location="cpu")["state_dict"], load_loss_func=False)
model = model.to(dev).eval()
windows = json.loads((CASES / "windows.json").read_text())
probs = np.load(S / "probs.npz"); feats = np.load(S / "all_v8.npz")

cache = {}
for clip, spec in windows.items():
    if clip.startswith("_"):
        continue
    mix, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None,
                           resample_to=16000)
    mix = mix.view(1, -1)
    with torch.no_grad():
        enh = model(mix.to(dev), dry_blend=1.0).detach().cpu().view(1, -1)
    n = min(enh.shape[-1], mix.shape[-1])
    fps = float(feats[f"{clip}_fps"][0])
    cache[clip] = dict(base=(BASE*enh[..., :n] + (1-BASE)*mix[..., :n]).clamp(-1, 1),
                       mix=mix[..., :n], n=n, sr=sr, spec=spec,
                       b=integrate(probs[clip], fps, TAU_UP, TAU_DN), fps=fps,
                       hop=int(round(sr/fps)))
print("forward passes cached\n")

print(f"{'b_hi':>6s} | {'cold far':>9s} {'cold keep':>10s} | {'sess far':>9s} "
      f"{'sess near':>10s} {'worst near':>11s} {'KEEP-VIOL':>10s}")
print("-" * 78)
for b_hi in (0.50, 0.60, 0.65, 0.70, 0.75, 0.85):
    b_lo = max(0.05, b_hi - 0.40)
    cf, ck, sf, sn, worst, viol = [], [], [], [], 0.0, 0
    for clip, c in cache.items():
        t = np.clip((b_hi - c["b"]) / (b_hi - b_lo), 0, 1)
        g = torch.from_numpy(np.repeat(10.0 ** (t * 20*np.log10(GMIN) / 20), c["hop"])).float()
        g = (torch.cat([g, g[-1:].expand(max(0, c["n"]-len(g)))]) if len(g) < c["n"]
             else g[:c["n"]]).view(1, -1)
        out = (g * c["base"]).clamp(-1, 1)
        spec, sr, n = c["spec"], c["sr"], c["n"]
        sess = clip.endswith("_session")
        for key, pool in (("suppress", sf if sess else cf), ("keep", sn if sess else ck)):
            for x, y in spec.get(key, []):
                sp = [(x, y)]
                d = span_dbfs(out, sp, sr, n) - span_dbfs(c["mix"], sp, sr, n)
                pool.append(d)
                if key == "keep":
                    worst = min(worst, d)
                    viol += int(d < KEEP_VIOLATION_DB)
    print(f"{b_hi:6.2f} | {np.median(cf):9.2f} {np.median(ck):10.2f} | "
          f"{np.median(sf):9.2f} {np.median(sn):10.2f} {worst:11.2f} {viol:10d}")
print("\n  all columns are median dB vs input, per span.  b_lo tracks b_hi - 0.40.")
print(f"  KEEP-VIOL counts spans below {KEEP_VIOLATION_DB} dB, across all 27 clips "
      f"({sum(len(c['spec'].get('keep', [])) for c in cache.values())} keep spans).")
print(f"  today (b_hi = 0, mechanism off): cold far "
      f"{np.median([span_dbfs(c['base'], c['spec']['suppress'], c['sr'], c['n']) - span_dbfs(c['mix'], c['spec']['suppress'], c['sr'], c['n']) for k, c in cache.items() if not k.endswith('_session') and c['spec'].get('suppress')]):.2f} dB")
