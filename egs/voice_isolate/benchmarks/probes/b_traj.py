"""The b trajectory: does one continuous near-presence quantity behave on real audio?

Three questions, in order of how fatal a bad answer is:

  M0  does the readout transfer across recordings at all?  Fitted on one room,
      applied to the other -- no session-adaptive calibration, no threshold picked
      on the test points. If this is chance the design stops here.
  M1  GO/NO-GO. Is there a gap between the b of material that must not break
      (a near user is actually talking) and the b of material that must be fixed
      (cold-start lone bystander)?  No gap, no dead zone, no design.
  M2  On a cold-start bystander clip, how long until b falls?  Longer than the
      clip means the mechanism cannot act in time.
  M3  Session far spans -- the anchored case. b should fall there too.

b is an asymmetric leaky integrator over the readout probability: fast up, slow
down, started at 1.0 (biased toward "someone is there", the safe direction and
the one C4 says the evidence already leans).
"""
import os
import argparse, json, pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Derived features are not committed; PROBE_WORK says where they live.
S = pathlib.Path(os.environ.get("PROBE_WORK", pathlib.Path(sys.argv[0]).parent))
CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
AUDIBLE_DBFS = -60.0     # the lag analysis used this; below it there is nothing to decide


def load():
    d = np.load(S / "all_v8.npz")
    w = json.loads((CASES / "windows.json").read_text())
    clips = {}
    for k, spec in w.items():
        if k.startswith("_"):
            continue
        clips[k] = dict(X=d[f"{k}_X"].T, dbfs=d[f"{k}_dbfs"],
                        fps=float(d[f"{k}_fps"][0]), spec=spec,
                        group=spec["group"], session=k.endswith("_session"),
                        held_out=bool(spec.get("held_out")))
    return clips


def role(name, spec):
    if name.endswith("_session"):
        return "session"
    r = spec["role"]
    if "double-talk" in r:
        return "dt"
    return "near" if "lone near" in r else "far"


def training_rows(clips, group):
    """Labelled frames from one recording. Audible frames only -- a silent frame
    carries no presence evidence and training on it teaches the readout the room
    tone.

    Refuses held-out material outright. A readout fitted on one room scored
    0.958 held-out *within* that room and then deleted the user everywhere else
    (full_gate/v8_gate050_VERDICT.md), so the reserve has to be enforced here
    rather than remembered at each call site.
    """
    X, y = [], []
    for name, c in clips.items():
        if c["group"] != group:
            continue
        if c["held_out"]:
            raise ValueError(
                f"{name} is held_out: nothing may be fitted on group {group!r}. "
                "Use it for evaluation only."
            )
        aud = c["dbfs"] > AUDIBLE_DBFS
        if c["session"]:
            fps, n = c["fps"], c["X"].shape[0]
            for label, key in ((1, "keep"), (0, "suppress")):
                for a, b in c["spec"].get(key, []):
                    lo, hi = max(0, int(a * fps)), min(n, int(b * fps))
                    m = np.zeros(n, bool); m[lo:hi] = True; m &= aud
                    if m.sum():
                        X.append(c["X"][m]); y.append(np.full(int(m.sum()), label))
        else:
            r = role(name, c["spec"])
            label = 0 if r == "far" else 1
            m = aud
            if m.sum():
                X.append(c["X"][m]); y.append(np.full(int(m.sum()), label))
    return np.concatenate(X), np.concatenate(y)


def integrate(s, fps, tau_up, tau_dn, b0=1.0):
    a_up = 1.0 - np.exp(-1.0 / (tau_up * fps))
    a_dn = 1.0 - np.exp(-1.0 / (tau_dn * fps))
    b = np.empty_like(s)
    cur = b0
    for t, v in enumerate(s):
        cur += (a_up if v > cur else a_dn) * (v - cur)
        b[t] = cur
    return b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-up", type=float, default=0.15)
    ap.add_argument("--tau-dn", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    clips = load()

    # ---------- M0: leave-one-recording-out readout ----------
    print("=" * 76)
    print("M0  readout fitted on ONE recording, applied to the OTHER")
    print("=" * 76)
    probs, roles = {}, {}
    for test_g, train_g in (("270d", "90d"), ("90d", "270d")):
        Xtr, ytr = training_rows(clips, train_g)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
        clf.fit(Xtr[::a.stride], ytr[::a.stride])
        ev_X, ev_y = [], []
        for name, c in clips.items():
            if c["group"] != test_g:
                continue
            p = clf.predict_proba(c["X"])[:, 1]
            probs[name] = p
            r = role(name, c["spec"]); roles[name] = r
            aud = c["dbfs"] > AUDIBLE_DBFS
            if r in ("near", "dt", "far"):
                ev_X.append(p[aud]); ev_y.append(np.full(int(aud.sum()),
                                                          0 if r == "far" else 1))
        ev_p, ev_y = np.concatenate(ev_X), np.concatenate(ev_y)
        acc = balanced_accuracy_score(ev_y, (ev_p > 0.5).astype(int))
        print(f"  train {train_g:5s} -> test {test_g:5s} : "
              f"cold-start frames n={len(ev_y):6d}  "
              f"balanced acc {acc:.3f}  AUC {roc_auc_score(ev_y, ev_p):.3f}")
        print(f"      mean p(present)  present {ev_p[ev_y==1].mean():.3f}   "
              f"absent {ev_p[ev_y==0].mean():.3f}")
    np.savez_compressed(S / "probs.npz", **probs)

    # ---------- b ----------
    B = {n: integrate(probs[n], clips[n]["fps"], a.tau_up, a.tau_dn) for n in probs}

    print()
    print("=" * 76)
    print(f"M1  GO/NO-GO -- b distributions (tau_up {a.tau_up}s, tau_dn {a.tau_dn}s, b0 1.0)")
    print("=" * 76)
    groups = {"MUST NOT BREAK  near (cold start)": [n for n in B if roles[n] == "near"],
              "MUST NOT BREAK  double-talk":       [n for n in B if roles[n] == "dt"],
              "MUST FIX        far (cold start)":  [n for n in B if roles[n] == "far"]}
    stats = {}
    for label, names in groups.items():
        rows = []
        for n in sorted(names):
            b = B[n]
            rows.append((n, b.min(), np.percentile(b, 5), np.median(b), b.max()))
        stats[label] = rows
        print(f"\n{label}   ({len(rows)} clips)")
        print(f"    {'clip':14s} {'min':>7s} {'p5':>7s} {'median':>8s} {'max':>7s}")
        for n, mn, p5, md, mx in rows:
            print(f"    {n:14s} {mn:7.3f} {p5:7.3f} {md:8.3f} {mx:7.3f}")

    keep_names = groups["MUST NOT BREAK  near (cold start)"] + groups["MUST NOT BREAK  double-talk"]
    far_names = groups["MUST FIX        far (cold start)"]
    keep_min = min(B[n].min() for n in keep_names)
    keep_p5 = min(np.percentile(B[n], 5) for n in keep_names)
    far_max = max(B[n].max() for n in far_names)
    far_p95 = max(np.percentile(B[n], 95) for n in far_names)
    print()
    print(f"  worst keep clip: min {keep_min:.3f}   (p5 across clips, worst {keep_p5:.3f})")
    print(f"  worst far  clip: max {far_max:.3f}   (p95 across clips, worst {far_p95:.3f})")
    print(f"  --> strict gap (keep.min - far.max)      = {keep_min - far_max:+.3f}")
    print(f"  --> tolerant gap (keep.p5 - far.p95)     = {keep_p5 - far_p95:+.3f}")

    # ---------- M2 ----------
    print()
    print("=" * 76)
    print("M2  how long until b falls, on a cold-start bystander clip")
    print("=" * 76)
    print(f"    {'clip':14s} {'len':>6s} {'audible@':>9s}  " +
          "  ".join(f"b<{t:.1f}" for t in (0.9, 0.7, 0.5, 0.3)))
    for n in sorted(far_names):
        c = clips[n]; b = B[n]; fps = c["fps"]
        aud = np.flatnonzero(c["dbfs"] > AUDIBLE_DBFS)
        t0 = aud[0] / fps if len(aud) else float("nan")
        cells = []
        for thr in (0.9, 0.7, 0.5, 0.3):
            hit = np.flatnonzero(b < thr)
            cells.append(f"{hit[0]/fps - t0:6.2f}s" if len(hit) else "  never")
        print(f"    {n:14s} {len(b)/fps:5.1f}s {t0:8.2f}s  " + "  ".join(cells))
    print("    (time is measured from the first audible frame, not from t=0)")

    # ---------- M3 ----------
    print()
    print("=" * 76)
    print("M3  session spans -- the anchored case")
    print("=" * 76)
    for name in ("90d_session", "270d_session"):
        if name not in B:
            continue
        c = clips[name]; b = B[name]; fps = c["fps"]; n = len(b)
        print(f"\n  {name}")
        for key, tag in (("keep", "near"), ("suppress", "far ")):
            for i, (x, y) in enumerate(c["spec"].get(key, [])):
                lo, hi = max(0, int(x * fps)), min(n, int(y * fps))
                seg = b[lo:hi]
                if len(seg) < 2:
                    continue
                print(f"    {tag} span {i}  {x:6.1f}-{y:6.1f}s  "
                      f"min {seg.min():.3f}  median {np.median(seg):.3f}  max {seg.max():.3f}")
