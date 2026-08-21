"""Which time window does the distance readout actually listen to?

The readout tracks true distance (rho +0.60 anechoic) but compressed ~1:60, and
it survives both a 24 dB level swing and the near/far DRR gap thinning by 3.6 dB.
So it rides on something those manipulations leave alone. Split every RIR into
three windows and destroy them one at a time:

    direct  0 - 2.5 ms      the direct arrival (the DRR window)
    early   2.5 - 50 ms     first wall reflections -- room geometry
    late    50 ms +         the diffuse tail

"Destroy" = replace that window with white noise at its own RMS, so the energy
in the window is preserved and only its STRUCTURE is gone. Scrambling the phase
would keep the magnitude spectrum; matching RMS keeps the level. Either way the
control is the same: whichever window matters, breaking it should collapse the
near/far separation the readout is known to have.

Scored as the gap between the near-channel and far-channel estimates, because
that two-group separation is the thing the readout demonstrably CAN do -- ranking
within a group it cannot (rho +0.227 in the trained bank, not significant).
"""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

CASES = "data_report/field_cases/test_vector_cases"
BANK = "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar"
NEAR, FAR = {"near_0", "near_1"}, {"far_0", "far_1", "far_2"}
SR = 16000
WINDOWS = {"none": None, "direct": (0.0, 2.5), "early": (2.5, 50.0), "late": (50.0, 1e9)}

model = init_siso_model(load_recipe("config/train_dpcrn.yaml",
        expected_task="voice_isolation", expected_purpose="train").model)
model.load_state_dict(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
        map_location="cpu")["state_dict"], strict=False)
model = model.eval()

speech, _ = AudioIO.open(f_path=f"{CASES}/90d_near1_raw.wav", target_lvl=None,
                         resample_to=SR)
speech = speech.view(1, 1, -1)[..., : 2 * SR]
rng = np.random.default_rng(0)

def damage(h, span):
    """Replace one window with noise at its own RMS: energy kept, structure gone."""
    if span is None:
        return h
    peak = int(np.argmax(np.abs(h)))
    a = peak + int(span[0] * SR / 1000.0)
    b = min(len(h), peak + int(span[1] * SR / 1000.0))
    if b <= a:
        return h
    out = h.copy()
    seg = out[a:b]
    rms = np.sqrt((seg ** 2).mean()) if seg.size else 0.0
    if rms > 0:
        n = rng.standard_normal(b - a)
        out[a:b] = n / np.sqrt((n ** 2).mean()) * rms
    return out

def estimate(h):
    t = torch.from_numpy(np.ascontiguousarray(h)).float().view(1, 1, -1)
    x = torch.nn.functional.conv1d(
        torch.nn.functional.pad(speech, (t.shape[-1] - 1, 0)), t.flip(-1)).view(1, -1)
    x = x / x.abs().amax().clamp_min(1e-8) * 0.3
    with torch.no_grad():
        model(x)
        return float(10.0 ** model.backbone.last_dist_preds[0, 1])

items = sorted(glob.glob(f"{BANK}/items/*.json"))
rng2 = np.random.default_rng(7); rng2.shuffle(items)
res = {k: {"near": [], "far": []} for k in WINDOWS}
used = 0
for p in items:
    j = json.load(open(p))
    if "scene" not in j:
        continue
    try:
        w, _ = sf.read(p.replace(".json", ".wav"))
    except Exception:
        continue
    if w.ndim == 1:
        w = w[:, None]
    for chm in j["scene"]["channel_map"]:
        lab = chm["label"]
        if lab not in NEAR | FAR:
            continue
        h = np.ascontiguousarray(w[:, chm["channel"]]).astype(np.float64)
        side = "near" if lab in NEAR else "far"
        for name, span in WINDOWS.items():
            res[name][side].append(estimate(damage(h, span)))
    used += 1
    if used >= 10:
        break

print(f"rooms used: {used}\n")
print(f"{'damaged window':16s} {'near est':>9s} {'far est':>9s} {'gap':>8s} {'gap kept':>9s}")
base = None
for name in WINDOWS:
    n = np.median(res[name]["near"]); f = np.median(res[name]["far"])
    gap = f - n
    if base is None:
        base = gap
    print(f"{name:16s} {n:9.3f} {f:9.3f} {gap:8.3f} {100*gap/base:8.0f}%")
print("\n  'gap kept' = how much of the undamaged near/far separation survives.")
print("  The window the readout depends on is the one whose damage keeps the least.")
