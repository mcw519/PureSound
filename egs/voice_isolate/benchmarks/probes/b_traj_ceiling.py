"""What the mechanism buys, in the unit the benchmarks are written in.

b is not the deliverable; dry_blend(b) is, and dry_blend maps to a suppression
ceiling of 20*log10(1 - dry_blend). Today that is a flat -20.0 dB everywhere. The
question is how much deeper it gets on material that should be suppressed, and how
much of the deletion insurance is given up on material that must not be.
"""
import os
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
from b_traj import load, role, integrate, AUDIBLE_DBFS

# Derived features are not committed; PROBE_WORK says where they live.
S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))
TAU_UP, TAU_DN = 0.05, 1.0
clips = load()
probs = np.load(S / "probs.npz")


def blend(b, b_lo, b_hi, base=0.9, top=0.995):
    """Flat at today's 0.9 above b_hi -- that segment is bit-identical to now."""
    t = np.clip((b_hi - b) / (b_hi - b_lo), 0.0, 1.0)
    return base + t * (top - base)


def ceiling_db(d):
    return 20.0 * np.log10(np.maximum(1.0 - d, 1e-9))


pools = {"cold keep": [], "cold far": [], "sess near": [], "sess far": []}
for n in probs.files:
    c = clips[n]; r = role(n, c["spec"])
    b = integrate(probs[n], c["fps"], TAU_UP, TAU_DN)
    aud = c["dbfs"] > AUDIBLE_DBFS
    fps = c["fps"]
    if r == "session":
        nn = len(b)
        for key, tag in (("keep", "sess near"), ("suppress", "sess far")):
            for x, y in c["spec"].get(key, []):
                m = np.zeros(nn, bool)
                m[max(0, int(x*fps)):min(nn, int(y*fps))] = True
                m &= aud
                if m.sum() >= 5:
                    pools[tag].append(b[m])
    else:
        clk = (np.arange(len(b)) - (np.flatnonzero(aud)[0] if aud.any() else 0)) / fps
        m = aud & (clk >= 1.5)
        if m.sum():
            pools["cold keep" if r in ("near", "dt") else "cold far"].append(b[m])
V = {k: np.concatenate(v) for k, v in pools.items()}

print("=" * 78)
print(f"Suppression ceiling delivered   (tau_up {TAU_UP}s / tau_dn {TAU_DN}s, "
      f"blend 0.9 -> 0.995)")
print("=" * 78)
print("  today: a flat -20.0 dB on every frame of every clip\n")
for b_lo, b_hi in ((0.35, 0.75), (0.25, 0.75), (0.35, 0.85), (0.45, 0.90)):
    print(f"  b_lo {b_lo:.2f}  b_hi {b_hi:.2f}")
    for k in ("cold keep", "sess near", "cold far", "sess far"):
        d = blend(V[k], b_lo, b_hi)
        cb = ceiling_db(d)
        untouched = 100 * (d <= 0.9 + 1e-12).mean()
        print(f"     {k:10s}  frames bit-identical {untouched:5.1f}%   "
              f"ceiling median {np.median(cb):7.2f} dB   p10 {np.percentile(cb,10):7.2f}"
              f"   p90 {np.percentile(cb,90):7.2f}")
    print()

print("=" * 78)
print("The trade in one line, at b_lo 0.35 / b_hi 0.75")
print("=" * 78)
d_far = blend(V["cold far"], 0.35, 0.75)
d_keep = blend(V["cold keep"], 0.35, 0.75)
d_sn = blend(V["sess near"], 0.35, 0.75)
print(f"  cold-start bystander : ceiling median {np.median(ceiling_db(d_far)):.2f} dB "
      f"(was -20.00) -> {np.median(ceiling_db(d_far)) + 20:.2f} dB of headroom released")
print(f"  cold-start user      : {100*(d_keep<=0.9+1e-12).mean():.1f}% of frames untouched, "
      f"worst ceiling {ceiling_db(d_keep).min():.2f} dB")
print(f"  in-session user      : {100*(d_sn<=0.9+1e-12).mean():.1f}% of frames untouched, "
      f"worst ceiling {ceiling_db(d_sn).min():.2f} dB")
print()
print("  NB the ceiling is a bound, not an outcome: releasing it lets the model")
print("  suppress deeper, it does not make it. What it removes is the arithmetic")
print("  floor that every far-field residual on record was measured against.")
