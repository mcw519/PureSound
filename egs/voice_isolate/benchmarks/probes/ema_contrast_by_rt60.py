"""Does the EMA bank's fast/slow contrast collapse as reverberation grows?

The hypothesis: a very long tail keeps the slow averages permanently elevated,
so the difference between the 50 ms and the 4 s average -- which is the modulation
depth the head was given the bank for -- flattens out, and the head loses the
onset transient.

Measured on the live training distribution, stratified by the row's own rt60, on
the v11 head's ACTUAL bank (the tensor it feeds to proj), not on energy.
Contrast is |fast - slow| relative to the instantaneous magnitude, so a collapse
shows up as a fall toward zero.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound"); sys.path.insert(0, ".")
import numpy as np, torch
from puresound.config import load_recipe, with_overrides
from puresound.recipes import init_siso_model
import main as M

recipe = load_recipe("config/exp/train_dpcrn_v11_presence.yaml",
                     expected_task="voice_isolation", expected_purpose="train")
recipe = with_overrides(recipe, trainer={"num_workers": 3})
# pin_memory touches CUDA even for a CPU run and the GPUs are busy training;
# the dataset is iterable-style so it cannot be rewrapped with a plain sampler.
import torch.utils.data as _tud
_orig = _tud.DataLoader.__init__
def _no_pin(self, *a, **kw):
    kw["pin_memory"] = False
    return _orig(self, *a, **kw)
_tud.DataLoader.__init__ = _no_pin
_t, valid = M.init_dataloader(recipe)
model = init_siso_model(recipe.model)
model.load_state_dict(torch.load("pretrained_ckpt/dpcrn_v11_ep19.ckpt",
                                 map_location="cpu")["state_dict"], strict=False)
model = model.eval()
head = model.backbone.vad_head
TAUS = head.ema_taus_s
print("EMA taus:", TAUS)

cap = {}
head.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0].detach()))
BANDS = ((0.0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9))
acc = {b: [] for b in BANDS}
with torch.no_grad():
    for bi, batch in enumerate(valid):
        if bi >= 6: break
        model(batch["noisy_speech"])
        x = cap["x"]                                  # [N,C,F,T]
        h = x.mean(dim=2)                             # [N,C,T]
        bank = head._ema_bank(h)                      # [N,(K+1)C,T]
        C = h.shape[1]
        inst = bank[:, :C]
        fast = bank[:, C:2*C]                         # tau[0] = 50 ms
        slow = bank[:, (len(TAUS))*C:(len(TAUS)+1)*C] # tau[-1] = 4 s
        num = (fast - slow).abs().mean(dim=(1,))      # [N,T]
        den = inst.abs().mean(dim=(1,)).clamp_min(1e-6)
        contrast = (num / den).float().numpy()  # [N,T]
        rt = batch.get("rt60")
        for r in range(contrast.shape[0]):
            v = float(rt.view(-1)[r]) if rt is not None else float("nan")
            if v != v: continue
            for lo, hi in BANDS:
                if lo <= v < hi:
                    acc[(lo,hi)].append(contrast[r]); break

print(f"\n{'rt60 band':12s} {'rows':>5s} {'fast-vs-slow contrast':>22s} {'p10':>7s} {'p90':>7s}")
for b in BANDS:
    if not acc[b]: continue
    v = np.concatenate(acc[b])
    print(f"  {b[0]:.1f}-{b[1]:.1f}     {len(acc[b]):5d} {np.median(v):22.4f} "
          f"{np.percentile(v,10):7.4f} {np.percentile(v,90):7.4f}")
print("\n  a collapse would show as this falling toward 0 with rt60.")
