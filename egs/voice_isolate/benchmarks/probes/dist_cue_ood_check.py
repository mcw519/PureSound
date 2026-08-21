"""Is the drop from damaging a window "information removed", or "input the model
has never seen"?

The contradiction to explain: destroying BOTH magnitude and phase keeps 54% of
the near/far gap while destroying phase ALONE keeps 35%. Damaging more cannot
recover information, so at least part of that 35% is not missing information --
it is the manipulation producing something out of distribution.

Two independent readouts on the SAME manipulated inputs, neither of which is the
distance head:

  separation quality  SI-SDR of the enhanced output against the clean target.
                      An out-of-distribution input degrades the whole model, not
                      just one head. Information removal that is IN distribution
                      (a genuinely reverberant room) should leave it roughly alone.
  bottleneck norm     mean |activation| entering the heads. A normal input sits
                      in the range training produced; a pathological one does not.

If the phase-randomised condition wrecks BOTH of these while a real high-reverb
room does not, the 35% is contaminated and the damage numbers cannot be read as
an information budget.
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
from puresound.audio.impulse_response import wav_apply_rir

BANK = "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar"
NEAR = {"near_0", "near_1"}
SR, MS = 16000, 5.0
model = init_siso_model(load_recipe("config/train_dpcrn.yaml",
        expected_task="voice_isolation", expected_purpose="train").model)
model.load_state_dict(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
        map_location="cpu")["state_dict"], strict=False)
model = model.eval()
cap = {}
model.backbone.dist_head.register_forward_pre_hook(
    lambda m, i: cap.__setitem__("x", i[0].detach()))
sp, _ = AudioIO.open(f_path="data_report/field_cases/test_vector_cases/90d_near1_raw.wav",
                     target_lvl=None, resample_to=SR)
sp = sp.view(1, -1)[..., : 2 * SR]

def si_sdr(est, ref):
    est, ref = est.view(-1), ref.view(-1)
    n = min(est.shape[0], ref.shape[0]); est, ref = est[:n], ref[:n]
    ref = ref - ref.mean(); est = est - est.mean()
    a = (est * ref).sum() / ref.pow(2).sum().clamp_min(1e-12)
    t = a * ref
    return float(10 * torch.log10(t.pow(2).sum() / (est - t).pow(2).sum().clamp_min(1e-12)))

def treat(h, how, seed=1234):
    n = int(MS * SR / 1000.0)
    peak = int(h.abs().argmax()); end = min(h.shape[-1], peak + n)
    if end - peak < 8 or how == "none": return h
    out = h.clone(); w = out[peak:end]; g = torch.Generator().manual_seed(seed)
    if how == "phase": w = phase_randomise(w, g)
    elif how == "mag": w = magnitude_flatten(w)
    elif how == "both": w = phase_randomise(magnitude_flatten(w), g)
    out[peak:end] = w
    return out

CONDS = ("none", "phase", "mag", "both")
acc = {c: {"sisdr": [], "norm": []} for c in CONDS}
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
        if chm["label"] not in NEAR: continue
        h = torch.from_numpy(np.ascontiguousarray(w[:, chm["channel"]])).float()
        for c in CONDS:
            hh = treat(h, c).view(1, -1)
            # mixture = full RIR, target = early: the model's actual task
            noisy = wav_apply_rir(sp, hh, SR, rir_mode="full")
            clean = wav_apply_rir(sp, hh, SR, rir_mode="early")
            peak = noisy.abs().amax().clamp_min(1e-8)
            noisy, clean = noisy / peak * 0.3, clean / peak * 0.3
            with torch.no_grad():
                enh = model(noisy)
            acc[c]["sisdr"].append(si_sdr(enh, clean))
            acc[c]["norm"].append(float(cap["x"].abs().mean()))
    used += 1
    if used >= 8: break

print(f"rooms: {used}   (near channels only, so 'clean' is a real target)\n")
print(f"{'condition':22s} {'SI-SDR dB':>10s} {'vs none':>8s} {'bottleneck |x|':>15s} {'vs none':>8s}")
b_s = np.median(acc["none"]["sisdr"]); b_n = np.median(acc["none"]["norm"])
LBL = {"none": "none (baseline)", "phase": "timing destroyed",
       "mag": "spectrum destroyed", "both": "both destroyed"}
for c in CONDS:
    s = np.median(acc[c]["sisdr"]); n = np.median(acc[c]["norm"])
    print(f"{LBL[c]:22s} {s:10.2f} {s-b_s:+8.2f} {n:15.4f} {100*(n/b_n-1):+7.1f}%")
print("\n  A large SI-SDR drop = the whole model is off, not one head.")
