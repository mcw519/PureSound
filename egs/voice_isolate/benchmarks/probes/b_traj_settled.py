"""M1 done properly.

The first pass compared min(keep) against max(far) and got a negative gap. That
metric was wrong: every far clip's max is ~0.99 because b STARTS at 1.0 and has
not fallen yet. It measures the initial condition, not the mechanism.

Two corrections:

  * Only frames where the relevant voice is AUDIBLE. b sagging during a pause
    protects nothing and endangers nothing -- there is no speech there to keep or
    to suppress.
  * Only frames after the decision has had time to arrive. ~1 s of audible speech
    is the measured settling time; before that b is reporting its prior, which is
    deliberately "someone is there".

So the honest question is: once settled, on audible speech, do the two
distributions separate?
"""
import os
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
from b_traj import load, role, integrate, AUDIBLE_DBFS

# Derived features are not committed; PROBE_WORK says where they live.
S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))


def onset_clock(dbfs, fps):
    """Seconds since the first audible frame, per frame."""
    aud = np.flatnonzero(dbfs > AUDIBLE_DBFS)
    t0 = aud[0] if len(aud) else 0
    return (np.arange(len(dbfs)) - t0) / fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-up", type=float, default=0.15)
    ap.add_argument("--tau-dn", type=float, default=1.0)
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds of audible speech before b is read as a decision")
    a = ap.parse_args()

    clips = load()
    probs = np.load(S / "probs.npz")
    B, meta = {}, {}
    for n in probs.files:
        c = clips[n]
        B[n] = integrate(probs[n], c["fps"], a.tau_up, a.tau_dn)
        meta[n] = (role(n, c["spec"]), c["dbfs"] > AUDIBLE_DBFS,
                   onset_clock(c["dbfs"], c["fps"]))

    print("=" * 78)
    print(f"M1  b on AUDIBLE speech, settled (>{a.settle}s since onset)"
          f"   tau_up {a.tau_up}s / tau_dn {a.tau_dn}s")
    print("=" * 78)
    pools = {}
    for want, title in (("keep", "MUST NOT BREAK  (near + double-talk)"),
                        ("far", "MUST FIX        (cold-start lone bystander)")):
        names = [n for n in B if (meta[n][0] in ("near", "dt")) == (want == "keep")
                 and meta[n][0] != "session"]
        rows, pool = [], []
        for n in sorted(names):
            r, aud, clk = meta[n]
            m = aud & (clk >= a.settle)
            if m.sum() < 10:
                rows.append((n, r, int(m.sum()), None, None, None)); continue
            b = B[n][m]
            rows.append((n, r, int(m.sum()), b.min(), np.median(b), b.max()))
            pool.append(b)
        pools[want] = np.concatenate(pool) if pool else np.array([])
        print(f"\n{title}")
        print(f"    {'clip':14s} {'role':5s} {'n':>6s} {'min':>7s} {'median':>8s} {'max':>7s}")
        for n, r, cnt, mn, md, mx in rows:
            if mn is None:
                print(f"    {n:14s} {r:5s} {cnt:6d}   -- too short to settle --")
            else:
                print(f"    {n:14s} {r:5s} {cnt:6d} {mn:7.3f} {md:8.3f} {mx:7.3f}")

    k, f = pools["keep"], pools["far"]
    print(f"\n  pooled frames: keep n={len(k)}, far n={len(f)}")
    for q in (0, 1, 5, 10, 25, 50):
        print(f"    keep p{q:<2d} {np.percentile(k, q):.3f}"
              f"        far p{100-q:<2d} {np.percentile(f, 100-q):.3f}"
              f"        gap {np.percentile(k,q) - np.percentile(f,100-q):+.3f}")
    sep = (k > f.max()).mean()
    print(f"\n  keep frames above the worst far frame : {100*sep:5.1f}%")
    print(f"  far frames below the worst keep frame  : {100*(f < k.min()).mean():5.1f}%")

    # A dead zone needs one threshold that keeps ~all keep frames and still moves
    # on far frames. Report the trade directly instead of picking one.
    print(f"\n  {'b_hi':>6s} {'keep frames >= b_hi':>21s} {'far frames >= b_hi':>20s}")
    for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        print(f"  {thr:6.2f} {100*(k>=thr).mean():20.1f}% {100*(f>=thr).mean():19.1f}%")

    # ---------- settling curve, both groups ----------
    print()
    print("=" * 78)
    print("b against time since audible onset  (median over audible frames)")
    print("=" * 78)
    print(f"    {'window':>12s} {'keep':>18s} {'far':>18s}")
    for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 3), (3, 5), (5, 100)):
        cells = []
        for want in ("keep", "far"):
            vals = []
            for n in B:
                r, aud, clk = meta[n]
                if r == "session" or ((r in ("near", "dt")) != (want == "keep")):
                    continue
                m = aud & (clk >= lo) & (clk < hi)
                if m.sum():
                    vals.append(B[n][m])
            if vals:
                v = np.concatenate(vals)
                cells.append(f"{np.median(v):.3f} (n={len(v)})")
            else:
                cells.append("-")
        lab = f"{lo:g}-{hi:g}s" if hi < 100 else f">{lo:g}s"
        print(f"    {lab:>12s} {cells[0]:>18s} {cells[1]:>18s}")

    # ---------- session, audible only ----------
    print()
    print("=" * 78)
    print("M3  session spans, audible frames only")
    print("=" * 78)
    for name in ("90d_session", "270d_session"):
        c = clips[name]; b = B[name]; fps = c["fps"]; n = len(b)
        aud = c["dbfs"] > AUDIBLE_DBFS
        print(f"\n  {name}")
        for key, tag in (("keep", "near"), ("suppress", "far ")):
            for i, (x, y) in enumerate(c["spec"].get(key, [])):
                lo, hi = max(0, int(x * fps)), min(n, int(y * fps))
                m = np.zeros(n, bool); m[lo:hi] = True; m &= aud
                if m.sum() < 5:
                    continue
                seg = b[m]
                print(f"    {tag} span {i}  {x:6.1f}-{y:6.1f}s ({y-x:5.1f}s)  "
                      f"min {seg.min():.3f}  median {np.median(seg):.3f}  "
                      f"p90 {np.percentile(seg,90):.3f}")
