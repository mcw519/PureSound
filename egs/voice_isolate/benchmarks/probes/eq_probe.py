"""Does production EQ create the phantom near user? (chain factors, round 2)

Compression is convicted but only explains part of the shift: it walks the DRR
readout fully (-0.61 -> +5.19 dB, QVF reads +5.33) yet moves the foreground
distance only a third of the way (1.00 -> 0.90 m against QVF's 0.70). The next
suspect with a physical story is EQ: publication chains brighten (rumble cut,
presence boost), and spectral tilt is a proximity cue -- near speech is darker
(measured centroid 351 Hz near vs 620 Hz far), so BRIGHTENING should read as
NEARER only if the model reads tilt absolutely... which the first-principles
work says it does not do in isolation. This measures it either way.

Same protocol that convicted compression: the repo's own device far clips,
each manipulation RMS-restored so level is never the variable, foreground
slot read from dpcrn_v8's DistHead ([drr/10, log10 fg_dist, log10 itf_dist]).
`dark` inverts the tilt as the directional control: a causal tilt cue must
move the estimate the other way. `comp+broadcast` stacks the two convicted/
suspected chain stages, asking whether they close the remaining gap to QVF's
0.70 m together.
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch
import torchaudio.functional as AF
from puresound.audio.dsp import compressor_gain
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

SRC = pathlib.Path("data_report/field_cases/test_vector_cases")
CKPT = "pretrained_ckpt/dpcrn_v8.ckpt"
SR = 16000

w = json.load(open(SRC / "windows.json"))
far_dev = sorted(k for k, v in w.items()
                 if not k.startswith(("_", "qvf")) and not k.endswith("_session")
                 and "lone far" in v["role"])

def rms_restore(y, ref):
    return y * (ref.pow(2).mean().sqrt().clamp_min(1e-8)
                / y.pow(2).mean().sqrt().clamp_min(1e-8))

def eq_none(x): return x
def eq_hp100(x): return AF.highpass_biquad(x, SR, 100.0)
def eq_podcast(x):
    return AF.equalizer_biquad(AF.highpass_biquad(x, SR, 100.0), SR, 3000.0, 4.0, Q=1.0)
def eq_broadcast(x):
    y = AF.highpass_biquad(x, SR, 120.0)
    y = AF.bass_biquad(y, SR, -4.0, central_freq=200.0)
    return AF.equalizer_biquad(y, SR, 3000.0, 6.0, Q=1.0)
def eq_bright(x):
    return AF.treble_biquad(AF.bass_biquad(x, SR, -4.0, central_freq=250.0),
                            SR, 4.0, central_freq=2500.0)
def eq_dark(x): return AF.treble_biquad(x, SR, -4.0, central_freq=2500.0)
def comp(x):
    return compressor_gain(x.view(1, -1), SR, threshold_db=-28.0, ratio=4.0).view(-1)
def comp_broadcast(x): return eq_broadcast(comp(x))

CONDS = [("none", eq_none), ("hp100", eq_hp100), ("podcast", eq_podcast),
         ("broadcast", eq_broadcast), ("bright_tilt", eq_bright),
         ("dark_tilt (control)", eq_dark), ("comp", comp),
         ("comp+broadcast", comp_broadcast)]

model = init_siso_model(load_recipe("config/train_dpcrn.yaml",
        expected_task="voice_isolation", expected_purpose="train").model)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model = model.eval()

res = {name: {"dist": [], "drr": []} for name, _ in CONDS}
for k in far_dev:
    x, _ = sf.read(SRC / f"{k}_raw.wav")
    x = torch.from_numpy(np.ascontiguousarray(x)).float().view(-1)
    for name, fn in CONDS:
        y = rms_restore(fn(x.clone()), x).view(1, -1)
        with torch.no_grad():
            model(y)
            p = model.backbone.last_dist_preds[0]
        res[name]["dist"].append(float(10.0 ** p[1]))
        res[name]["drr"].append(float(p[0] * 10.0))
    print(f"  {k} done", flush=True)

print(f"\nclips: {len(far_dev)} device lone-far  |  references: device baseline ~1.00 m / -0.61 dB,"
      f"  QVF clips ~0.70 m / +5.33 dB,  compression alone 0.90 m / +5.19 dB\n")
print(f"{'condition':22s} {'fg_dist med':>11s} {'vs none':>8s} {'DRR med dB':>10s}")
base = np.median(res["none"]["dist"])
for name, _ in CONDS:
    d, r = np.median(res[name]["dist"]), np.median(res[name]["drr"])
    print(f"{name:22s} {d:11.3f} {d - base:+8.3f} {r:10.2f}")
