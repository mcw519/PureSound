"""What the self-calibrated threshold does through the REAL actuator.

`presence_selfcal_sim.py` scored raw per-frame thresholding, which overstates
damage: deployment runs the head's p through PresenceGate's asymmetric
integrator (fast up 0.05 s, slow down 1.0 s, dead zone at b_hi) before any gain
is applied, and turn-transition frames that read low for a moment are exactly
what that machinery absorbs. This replays the same cached frames through the
real `PresenceGate.trajectory()`/`gain()` math and scores in the scorecard's
units: per-span dB, keep violation at < -3 dB.

Arms, all on the v11a near head's logits:
  fixed       s = sigmoid(logit)               -- factory threshold 0
  cal-loose   calibrator (1 s/cluster, no hold) -- the effective-but-bitey config
  cal-strict  calibrator (3 s/cluster, 20 s hold) -- the safe config
While the calibrator is inactive s is forced to 1.0, so b sits at its ceiling
and the gain is exactly 1.0: inactive IS passthrough, by arithmetic.

Span dB = 10*log10(sum g^2 e / sum e) with e the frame energy -- the gain the
span's actual audio experienced, not the mean gain over its frames.
"""
import json, os, pathlib, sys
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from presence_selfcal_sim import Calibrator, frame_labels
from puresound.system.presence_gate import PresenceGate

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
CACHE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
KEEP_VIOL_DB = -3.0

ARMS = {
    "fixed": None,
    "cal-loose": dict(min_mass_s=1.0, min_elapsed_s=0.0),
    "cal-strict": dict(min_mass_s=3.0, min_elapsed_s=20.0),
}
gate = PresenceGate()   # deployment defaults: b_hi .5, b_lo .1, floor -26 dB


def frames_for(clip):
    z = np.load(CACHE / f"{clip}.npz")
    return z["lg"], z["dbfs"], float(z["fps"])


def s_stream(lg, dbfs, fps, arm):
    if ARMS[arm] is None:
        return 1.0 / (1.0 + np.exp(-lg))
    cal = Calibrator(fps, **ARMS[arm])
    s = np.ones(len(lg), dtype=np.float64)
    for i in range(len(lg)):
        active, thr = cal.step(lg[i], dbfs[i])
        if active:
            s[i] = 1.0 / (1.0 + np.exp(-(lg[i] - thr)))
    return s


def gains(s, fps):
    b = gate.trajectory(torch.from_numpy(np.asarray(s)).view(1, -1), fps)
    return gate.gain(b).view(-1).numpy()


def span_db(g, e, a, b_, fps, n):
    i, j = max(0, int(a * fps)), min(n, int(b_ * fps))
    if j <= i or e[i:j].sum() <= 0:
        return float("nan")
    return 10.0 * np.log10((g[i:j] ** 2 * e[i:j]).sum() / e[i:j].sum())


windows = json.loads((CASES / "windows.json").read_text())

print("=== SESSIONS through PresenceGate (per-span dB; keep violation < -3) ===")
print(f"{'session':22s} {'arm':10s} {'worst_keep':>10s} {'viol':>4s} {'med_supp':>8s} {'best_supp':>9s}")
for clip in sorted(k for k in windows if k.endswith("_session")):
    spec = windows[clip]
    lg, dbfs, fps = frames_for(clip)
    n = len(lg)
    e = (10.0 ** (dbfs / 20.0)) ** 2
    for arm in ARMS:
        g = gains(s_stream(lg, dbfs, fps, arm), fps)
        keeps = [span_db(g, e, a, b_, fps, n) for a, b_ in spec.get("keep", [])]
        supps = [span_db(g, e, a, b_, fps, n) for a, b_ in spec.get("suppress", [])]
        keeps = [k for k in keeps if k == k]; supps = [x for x in supps if x == x]
        viol = sum(1 for k in keeps if k < KEEP_VIOL_DB)
        print(f"{clip:22s} {arm:10s} {min(keeps) if keeps else float('nan'):10.2f} "
              f"{viol:>3d}/{len(keeps):<2d} "
              f"{np.median(supps) if supps else float('nan'):8.2f} "
              f"{min(supps) if supps else float('nan'):9.2f}")

print("\n=== COLD-START clips through PresenceGate (whole clip = its span) ===")
print(f"{'arm':10s} {'keep worst dB':>13s} {'keep viol':>9s} {'far med dB':>10s} {'far best dB':>11s} {'dt worst dB':>11s}")
for arm in ARMS:
    keep_db, far_db, dt_db, names = [], [], [], []
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        lg, dbfs, fps = frames_for(clip)
        e = (10.0 ** (dbfs / 20.0)) ** 2
        g = gains(s_stream(lg, dbfs, fps, arm), fps)
        db = 10.0 * np.log10((g ** 2 * e).sum() / e.sum())
        if "lone far" in spec["role"]:
            far_db.append(db)
        elif "double-talk" in spec["role"]:
            dt_db.append(db)
            if db < KEEP_VIOL_DB: names.append((clip, db))
        else:
            keep_db.append(db)
            if db < KEEP_VIOL_DB: names.append((clip, db))
    viol = len(names)
    print(f"{arm:10s} {min(keep_db):13.2f} {viol:>4d}/{len(keep_db)+len(dt_db):<3d} "
          f"{np.median(far_db):10.2f} {min(far_db):11.2f} {min(dt_db):11.2f}")
    for c, d in names:
        print(f"    !! {arm}: {c} {d:.2f} dB")
