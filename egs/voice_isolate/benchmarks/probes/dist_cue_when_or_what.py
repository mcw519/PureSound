"""Is the distance cue WHEN things happen, or WHAT frequencies are there?

Four conditions on the first 5 ms of every RIR, energy preserved in all of them:
    none                 baseline
    phase-randomised     magnitude exact, timing destroyed
    magnitude-flattened  phase exact, frequency content destroyed
    both                 sanity check -- must kill it

Scored as the near/far estimate gap, the thing the readout demonstrably can do.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, torch
from dist_cue_decompose import phase_randomise, magnitude_flatten
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

BANK = "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar"
NEAR, FAR = {"near_0", "near_1"}, {"far_0", "far_1", "far_2"}
SR, MS = 16000, 5.0
model = init_siso_model(load_recipe("config/train_dpcrn.yaml",
        expected_task="voice_isolation", expected_purpose="train").model)
model.load_state_dict(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
        map_location="cpu")["state_dict"], strict=False)
model = model.eval()
sp, _ = AudioIO.open(f_path="data_report/field_cases/test_vector_cases/90d_near1_raw.wav",
                     target_lvl=None, resample_to=SR)
sp = sp.view(1, 1, -1)[..., : 2 * SR]

def est(h):
    t = h.view(1, 1, -1).float()
    x = torch.nn.functional.conv1d(
        torch.nn.functional.pad(sp, (t.shape[-1] - 1, 0)), t.flip(-1)).view(1, -1)
    x = x / x.abs().amax().clamp_min(1e-8) * 0.3
    with torch.no_grad():
        model(x)
        return float(10.0 ** model.backbone.last_dist_preds[0, 1])

def treat(h, how, seed):
    n = int(MS * SR / 1000.0)
    peak = int(h.abs().argmax())
    end = min(h.shape[-1], peak + n)
    if end - peak < 8 or how == "none":
        return h
    out = h.clone()
    w = out[peak:end]
    g = torch.Generator().manual_seed(seed)
    if how == "phase":
        w = phase_randomise(w, g)
    elif how == "mag":
        w = magnitude_flatten(w)
    elif how == "both":
        w = phase_randomise(magnitude_flatten(w), g)
    out[peak:end] = w
    return out

CONDS = ("none", "phase", "mag", "both")
res = {c: {"near": [], "far": []} for c in CONDS}
items = sorted(glob.glob(f"{BANK}/items/*.json"))
np.random.default_rng(7).shuffle(items)
used = 0
for p in items:
    j = json.load(open(p))
    if "scene" not in j: continue
    try: w, _ = sf.read(p.replace(".json", ".wav"))
    except Exception: continue
    if w.ndim == 1: w = w[:, None]
    for chm in j["scene"]["channel_map"]:
        if chm["label"] not in NEAR | FAR: continue
        h = torch.from_numpy(np.ascontiguousarray(w[:, chm["channel"]])).float()
        side = "near" if chm["label"] in NEAR else "far"
        for c in CONDS:
            res[c][side].append(est(treat(h, c, 1234)))
    used += 1
    if used >= 10: break

print(f"rooms: {used}   window: first {MS} ms\n")
print(f"{'condition':22s} {'near':>8s} {'far':>8s} {'gap':>8s} {'gap kept':>9s}")
base = None
LBL = {"none": "none (baseline)", "phase": "timing destroyed",
       "mag": "spectrum destroyed", "both": "both destroyed"}
for c in CONDS:
    n, f = np.median(res[c]["near"]), np.median(res[c]["far"])
    gap = f - n
    if base is None: base = gap
    print(f"{LBL[c]:22s} {n:8.3f} {f:8.3f} {gap:8.3f} {100*gap/base:8.0f}%")
