"""Does the near/far decision fall back on absolute level when the room cue thins?

Established on reverberant real recordings: a -28 dB renormalisation moved the
foreground distance estimate 0.00 m. The model does NOT gamble on level THERE.
The open question is whether that holds when the reflected-sound cue is gone.

2x2, same rows, same model:
    {trained rt60 0.17-0.85, near-anechoic rt60 0.07-0.12} x {level, level +/- g}

If the anechoic column tracks the gain and the reverberant one does not, the
level-gambling hypothesis holds and it is conditional on the cue being absent.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound"); sys.path.insert(0, ".")
import numpy as np, torch
import torch.utils.data as _tud
_orig = _tud.DataLoader.__init__
def _no_pin(self, *a, **kw):
    kw["pin_memory"] = False
    return _orig(self, *a, **kw)
_tud.DataLoader.__init__ = _no_pin
from puresound.config import load_recipe, with_overrides
from puresound.recipes import init_siso_model
import main as M

CKPT = "pretrained_ckpt/dpcrn_v8.ckpt"      # the deployed default; v11b is still training
GAINS_DB = (-18.0, 0.0, +6.0)

def run(bank, n_batches=2):
    recipe = load_recipe("config/train_dpcrn.yaml", expected_task="voice_isolation",
                         expected_purpose="train")
    recipe = with_overrides(
        recipe,
        trainer={"num_workers": 6},
        augmentation_reverb={"simulator": {"pregenerated": {"folder": bank}}},
    )
    _t, valid = M.init_dataloader(recipe)
    model = init_siso_model(recipe.model)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
    model = model.eval()
    out = {g: [] for g in GAINS_DB}
    with torch.no_grad():
        for bi, batch in enumerate(valid):
            if bi >= n_batches: break
            x = batch["noisy_speech"]
            for g in GAINS_DB:
                model(x * (10.0 ** (g / 20.0)))
                p = model.backbone.last_dist_preds        # [N,3]
                out[g].append(p[:, 1].float().numpy())    # log10 fg_dist
    return {g: 10.0 ** np.concatenate(v) for g, v in out.items()}

for tag, bank in (("trained  (rt60 0.17-0.85)", "/work/any_exp_link/puresound_exp/hybrid_rir_16k_realfar"),
                  ("anechoic (rt60 0.07-0.12)", os.path.abspath("exp/hybrid_rir_16k_anechoic"))):
    r = run(bank)
    base = np.median(r[0.0])
    print(f"\n{tag}")
    print(f"  {'gain':>7s} {'median fg_dist':>15s} {'shift vs 0 dB':>14s}")
    for g in GAINS_DB:
        m = np.median(r[g])
        print(f"  {g:+7.0f} {m:15.3f} m {m - base:+13.3f} m")
    span = max(np.median(r[g]) for g in GAINS_DB) - min(np.median(r[g]) for g in GAINS_DB)
    print(f"  -> estimate moves {span:.3f} m across a 24 dB level swing")
