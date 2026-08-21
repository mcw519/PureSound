"""Does the smear actually remove the cue it targets?

If the near/far separation survives it, the augmentation attacks nothing and
adding it to a recipe would only cost training signal.
"""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch
from puresound.audio.io import AudioIO
from puresound.audio.impulse_response import smear_direct_arrival
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

BANK = "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar"
NEAR, FAR = {"near_0", "near_1"}, {"far_0", "far_1", "far_2"}
SR = 16000
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

items = sorted(glob.glob(f"{BANK}/items/*.json"))
np.random.default_rng(7).shuffle(items)
SMEARS = (0.0, 2.0, 5.0)
res = {s: {"near": [], "far": []} for s in SMEARS}
used = 0
for p in items:
    j = json.load(open(p))
    if "scene" not in j: continue
    try: w, _ = sf.read(p.replace(".json", ".wav"))
    except Exception: continue
    if w.ndim == 1: w = w[:, None]
    for chm in j["scene"]["channel_map"]:
        if chm["label"] not in NEAR | FAR: continue
        h = torch.from_numpy(np.ascontiguousarray(w[:, chm["channel"]])).float().view(1, -1)
        side = "near" if chm["label"] in NEAR else "far"
        for ms in SMEARS:
            g = torch.Generator().manual_seed(1234)
            res[ms][side].append(est(smear_direct_arrival(h, SR, smear_ms=ms, generator=g)[0]))
    used += 1
    if used >= 10: break

print(f"rooms: {used}\n{'smear':>8s} {'near':>8s} {'far':>8s} {'gap':>8s} {'gap kept':>9s}")
base = None
for ms in SMEARS:
    n, f = np.median(res[ms]["near"]), np.median(res[ms]["far"])
    gap = f - n
    if base is None: base = gap
    print(f"{ms:6.1f}ms {n:8.3f} {f:8.3f} {gap:8.3f} {100*gap/base:8.0f}%")
