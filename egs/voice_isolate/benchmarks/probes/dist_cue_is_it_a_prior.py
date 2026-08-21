"""Is the distance readout measuring out of distribution, or reporting a prior?

Between the trained bank and the near-anechoic one the near/far DRR separation
thinned by 3.6 dB and the estimate moved 0.670 -> 0.665 m. Level moves it 0.000.
Either it rides on a cue neither manipulation touched, or it is not a measurement
at all outside the training range.

Decisive test: render the SAME speech through anechoic RIRs at several TRUE
distances and see whether the estimate tracks. A measurement tracks; a prior does
not. Run against the trained bank as the positive control -- the estimate is known
to separate there (near 0.61 / far 1.71 on the field set, 2.8x).
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, soundfile as sf, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

CASES = "data_report/field_cases/test_vector_cases"
BANKS = {
    "anechoic  (rt60 0.07-0.12)": "exp/hybrid_rir_16k_anechoic",
    "trained   (rt60 0.17-0.85)": "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar",
}
NEAR = {"near_0", "near_1"}
FAR = {"far_0", "far_1", "far_2"}

model = init_siso_model(load_recipe("config/train_dpcrn.yaml",
        expected_task="voice_isolation", expected_purpose="train").model)
model.load_state_dict(torch.load("pretrained_ckpt/dpcrn_v8.ckpt",
        map_location="cpu")["state_dict"], strict=False)
model = model.eval()

# one dry-ish speech excerpt, reused everywhere so speech content is not a variable
speech, sr = AudioIO.open(f_path=f"{CASES}/90d_near1_raw.wav", target_lvl=None,
                          resample_to=16000)
speech = speech.view(1, -1)[..., : 2 * 16000]

def render(rir, ch):
    h = torch.from_numpy(np.ascontiguousarray(rir[:, ch])).float().view(1, 1, -1)
    x = torch.nn.functional.conv1d(
        torch.nn.functional.pad(speech.view(1, 1, -1), (h.shape[-1] - 1, 0)), h.flip(-1))
    y = x.view(1, -1)
    return y / y.abs().amax().clamp_min(1e-8) * 0.3       # fixed peak: level is not the variable

for tag, bank in BANKS.items():
    items = sorted(glob.glob(f"{bank}/**/*.json", recursive=True))
    if not items:
        items = sorted(glob.glob(f"{bank}/items/*.json"))
    rows = []
    for p in items[:30]:
        j = json.load(open(p))
        if "scene" not in j:
            continue
        try:
            w, srr = sf.read(p.replace(".json", ".wav"))
        except Exception:
            continue
        if w.ndim == 1:
            w = w[:, None]
        for chm in j["scene"]["channel_map"]:
            if chm["label"] not in NEAR | FAR:
                continue
            d = float(chm.get("distance_m", float("nan")))
            if d != d:
                continue
            with torch.no_grad():
                model(render(w, chm["channel"]))
                est = float(10.0 ** model.backbone.last_dist_preds[0, 1])
            rows.append((d, est))
        if len(rows) >= 40:
            break
    true = np.array([r[0] for r in rows]); est = np.array([r[1] for r in rows])
    from scipy.stats import spearmanr
    rho, p = spearmanr(true, est)
    print(f"\n{tag}   n={len(rows)}")
    print(f"  {'true dist':>12s} {'n':>4s} {'estimate median':>16s}")
    for lo, hi in ((0.3, 0.7), (0.7, 1.1), (1.1, 2.0), (2.0, 3.0), (3.0, 6.0)):
        m = (true >= lo) & (true < hi)
        if m.sum() < 3:
            continue
        print(f"  {lo:.1f}-{hi:.1f} m    {m.sum():4d} {np.median(est[m]):16.3f}")
    print(f"  -> Spearman rho(true, estimate) = {rho:+.3f}  p = {p:.4f}")
    print(f"     estimate spread p10-p90: {np.percentile(est,10):.3f} - {np.percentile(est,90):.3f} m")
