"""Where the keep guardrail actually breaks, and whose fault it is.

M1 says the dead zone exists on cold-start material. M3 says short near spans
inside a session sink into it -- the user IS talking and b reads "absent". This
asks two things about that:

  * Is it the READOUT (s is low, the integrator is faithfully reporting it) or the
    INTEGRATOR (s is fine, b has not caught up)? Different fixes.
  * Does any (tau_up, tau_dn) buy the short spans back without letting far frames
    into the dead zone?
"""
import os
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
from b_traj import load, role, integrate, AUDIBLE_DBFS

# Derived features are not committed; PROBE_WORK says where they live.
S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))
B_HI = 0.75
clips = load()
probs = np.load(S / "probs.npz")


def spans(name):
    c = clips[name]; fps = c["fps"]; n = c["X"].shape[0]
    aud = c["dbfs"] > AUDIBLE_DBFS
    for key, tag in (("keep", "near"), ("suppress", "far")):
        for i, (x, y) in enumerate(c["spec"].get(key, [])):
            lo, hi = max(0, int(x * fps)), min(n, int(y * fps))
            m = np.zeros(n, bool); m[lo:hi] = True; m &= aud
            if m.sum() >= 5:
                yield tag, i, y - x, m


print("=" * 78)
print(f"Session NEAR spans -- readout vs integrator, at b_hi={B_HI}")
print("=" * 78)
print(f"  {'span':22s} {'len':>6s} {'median s':>9s} {'median b':>9s} "
      f"{'s<hi':>7s} {'b<hi':>7s}")
tot_f, tot_b = 0, 0
for name in ("90d_session", "270d_session"):
    b = integrate(probs[name], clips[name]["fps"], 0.15, 1.0)
    s = probs[name]
    for tag, i, dur, m in spans(name):
        if tag != "near":
            continue
        tot_f += int(m.sum()); tot_b += int((b[m] < B_HI).sum())
        print(f"  {name[:4]+' near '+str(i):22s} {dur:5.1f}s {np.median(s[m]):9.3f} "
              f"{np.median(b[m]):9.3f} {100*(s[m]<B_HI).mean():6.1f}% "
              f"{100*(b[m]<B_HI).mean():6.1f}%")
print(f"\n  all session near frames below b_hi: {100*tot_b/tot_f:.1f}%  "
      f"({tot_b}/{tot_f})")

print()
print("=" * 78)
print("By span length -- session near spans")
print("=" * 78)
buckets = {"< 2.5 s": [], "2.5-8 s": [], "> 8 s": []}
for name in ("90d_session", "270d_session"):
    b = integrate(probs[name], clips[name]["fps"], 0.15, 1.0)
    for tag, i, dur, m in spans(name):
        if tag != "near":
            continue
        k = "< 2.5 s" if dur < 2.5 else ("2.5-8 s" if dur < 8 else "> 8 s")
        buckets[k].append((b[m] < B_HI).mean())
for k, v in buckets.items():
    if v:
        print(f"  {k:9s} {len(v)} spans   frames below b_hi: "
              f"mean {100*np.mean(v):5.1f}%   worst {100*max(v):5.1f}%")

print()
print("=" * 78)
print("tau sweep -- can any pair protect short near spans without leaking far?")
print("=" * 78)
print(f"  {'tau_up':>7s} {'tau_dn':>7s} | {'cold keep':>10s} {'cold far':>9s} | "
      f"{'sess near':>10s} {'sess far':>9s}")
print(f"  {'':>7s} {'':>7s} | {'>=b_hi':>10s} {'>=b_hi':>9s} | {'>=b_hi':>10s} {'>=b_hi':>9s}")
for tau_up in (0.05, 0.15, 0.30):
    for tau_dn in (0.5, 1.0, 2.0, 4.0):
        pools = {"ck": [], "cf": [], "sn": [], "sf": []}
        for n in probs.files:
            c = clips[n]; r = role(n, c["spec"])
            b = integrate(probs[n], c["fps"], tau_up, tau_dn)
            aud = c["dbfs"] > AUDIBLE_DBFS
            if r == "session":
                for tag, i, dur, m in spans(n):
                    pools["sn" if tag == "near" else "sf"].append(b[m])
            else:
                clk = (np.arange(len(b)) - (np.flatnonzero(aud)[0]
                       if aud.any() else 0)) / c["fps"]
                m = aud & (clk >= 1.5)
                if m.sum():
                    pools["ck" if r in ("near", "dt") else "cf"].append(b[m])
        v = {k: np.concatenate(x) for k, x in pools.items()}
        print(f"  {tau_up:7.2f} {tau_dn:7.1f} | "
              f"{100*(v['ck']>=B_HI).mean():9.1f}% {100*(v['cf']>=B_HI).mean():8.1f}% | "
              f"{100*(v['sn']>=B_HI).mean():9.1f}% {100*(v['sf']>=B_HI).mean():8.1f}%")
