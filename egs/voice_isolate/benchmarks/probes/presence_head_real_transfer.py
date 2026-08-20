"""The wall: v11's near-presence head on the REAL field recordings, no refit.

In-domain the head trains on the rows it's scored on, so a high AUC is expected.
The chain boundary is where every presence readout has died in both directions
(07-10 synthetic->real 0.500; v8 real->synthetic 0.529). This runs the v11
last_vad_logits on the 27 field clips and asks: present (keep) vs absent
(suppress) frames, per group, and does the score hold -- with NO fitting of
anything. This is the same present/absent labelling the b-trajectory probe used.
"""
import json, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
import numpy as np, torch
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
CKPT = "exp/dpcrn_v11_presence/lightning_logs/version_2/checkpoints/epoch=19-step=10000.ckpt"


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg]); order = both.argsort().argsort() + 1
    rp = order[:len(pos)].sum()
    return float((rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


dev = torch.device("cpu")
model = init_siso_model(load_recipe("config/exp/train_dpcrn_v11_presence.yaml",
                                    expected_task="voice_isolation",
                                    expected_purpose="train").model)
sd = torch.load(CKPT, map_location="cpu")["state_dict"]
model.load_state_dict(sd, strict=False)
model = model.eval().to(dev)
windows = json.loads((CASES / "windows.json").read_text())

def logits_for(clip, seg_s=20.0, sr=16000):
    wav, _ = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None, resample_to=16000)
    wav = wav.view(1, -1); out = []
    step = int(seg_s * sr)
    for a in range(0, wav.shape[-1], step):
        seg = wav[..., a:a+step]
        if seg.shape[-1] < sr // 4: break
        with torch.no_grad():
            model(seg.to(dev))
        out.append(model.backbone.last_vad_logits[0].cpu().numpy())
    return np.concatenate(out)

# cold-start clips: whole-clip present (near/dt) vs absent (far)
cold = {"present": [], "absent": []}
for clip, spec in windows.items():
    if clip.startswith("_") or clip.endswith("_session"): continue
    lg = logits_for(clip)
    role = "absent" if "lone far" in spec["role"] else "present"
    cold[role].append((clip, np.median(lg), lg))
print("=== COLD-START clips (whole-clip median near-presence logit) ===")
print(f"  {'clip':>12s} {'role':>8s} {'median':>8s}")
for role in ("present", "absent"):
    for clip, md, _ in sorted(cold[role], key=lambda r: r[1]):
        print(f"  {clip:>12s} {role:>8s} {md:8.2f}")
pres = np.concatenate([lg for _,_,lg in cold["present"]])
absn = np.concatenate([lg for _,_,lg in cold["absent"]])
print(f"\n  frame-level AUC present vs absent: {auc(pres, absn):.3f}  "
      f"(n_pres {len(pres)}, n_abs {len(absn)})")
clip_p = [md for _,md,_ in cold["present"]]; clip_a = [md for _,md,_ in cold["absent"]]
print(f"  clip-level: present median-of-medians {np.median(clip_p):+.2f}, "
      f"absent {np.median(clip_a):+.2f}, min present {min(clip_p):+.2f} vs max absent {max(clip_a):+.2f}")

# session clips: keep spans vs suppress spans
print("\n=== SESSION clips (per-frame, keep spans vs suppress spans) ===")
labeler_fps = None
for clip in ("90d_session", "270d_session"):
    spec = windows[clip]; lg = logits_for(clip); n = len(lg)
    fps = n / spec["duration_s"]
    P, A = [], []
    for a, b in spec.get("keep", []):
        P.append(lg[max(0,int(a*fps)):min(n,int(b*fps))])
    for a, b in spec.get("suppress", []):
        A.append(lg[max(0,int(a*fps)):min(n,int(b*fps))])
    P = np.concatenate(P) if P else np.array([]); A = np.concatenate(A) if A else np.array([])
    print(f"  {clip}: keep med {np.median(P):+.2f}  supp med {np.median(A):+.2f}  "
          f"AUC {auc(P, A):.3f}  (n_keep {len(P)}, n_supp {len(A)})")

# --- split rig-chain (0d/90d/180d/270d) vs qvf-chain ---
def is_qvf(c): return c.startswith("qvf")
for label, keep in (("RIG chain (0d/90d/180d/270d)", lambda c: not is_qvf(c)),
                    ("QVF chain (ai-coustics)", is_qvf)):
    p = np.concatenate([lg for c,_,lg in cold["present"] if keep(c)] or [np.array([])])
    a = np.concatenate([lg for c,_,lg in cold["absent"] if keep(c)] or [np.array([])])
    cp = [md for c,md,_ in cold["present"] if keep(c)]
    ca = [md for c,md,_ in cold["absent"] if keep(c)]
    print(f"\n  {label}:  frame AUC {auc(p,a):.3f}   "
          f"clip present med {np.median(cp):+.2f} / absent {np.median(ca):+.2f}   "
          f"min_pres {min(cp):+.2f} vs max_abs {max(ca):+.2f}")
