"""Where does the 250 ms lag come from?

A  label misalignment  -> shifting every span later recovers accuracy
B  fixed algorithmic delay -> same signature as A (lookahead is only 30 ms, so at
                              most a small part of it)
C  evidence accumulation -> no shift helps; the deficit tracks how much signal
                            there is, not how late the label is
"""
import json, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

S = pathlib.Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/7b76fa14-55b2-4c0a-b797-c8214d986949/scratchpad/presence")
CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
d = np.load(S / "full_v8.npz")
windows = json.loads((CASES / "windows.json").read_text())
STRIDE = 5


def build_set(shift_s):
    """Label every frame from the spans, moved by `shift_s` seconds."""
    Xs, ys, gs, es, offs = [], [], [], [], []
    sid = 0
    for clip in ("90d_session", "270d_session"):
        X = d[f"{clip}_X"]; dbfs = d[f"{clip}_dbfs"]; fps = float(d[f"{clip}_fps"][0])
        n = X.shape[1]
        for label, key in ((1, "keep"), (0, "suppress")):
            for a, b in windows[clip].get(key, []):
                lo = int((a + shift_s) * fps); hi = int((b + shift_s) * fps)
                lo, hi = max(0, lo), min(n, hi)
                if hi - lo < 4:
                    continue
                idx = np.arange(lo, hi)[::STRIDE]
                Xs.append(X[:, idx].T); ys.append(np.full(len(idx), label))
                gs.append(np.full(len(idx), sid)); es.append(dbfs[idx])
                offs.append((idx - lo) / fps)
                sid += 1
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(es), np.concatenate(offs))


def score(X, y, g):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    pred = cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=g)
    return balanced_accuracy_score(y, pred), (pred == y)


print("TEST A/B -- does moving every label later recover accuracy?\n")
print(f"{'label shift':>12s} {'overall':>9s} {'first 0.25 s':>13s}")
print("-" * 38)
best = None
for shift in (-0.10, 0.0, 0.10, 0.20, 0.25, 0.30, 0.40):
    X, y, g, e, off = build_set(shift)
    acc, correct = score(X, y, g)
    early = correct[off < 0.25].mean()
    print(f"{shift*1000:+9.0f} ms {acc:9.3f} {100*early:12.1f}%")
    if best is None or acc > best[1]:
        best = (shift, acc)
print(f"\n  best shift {best[0]*1000:+.0f} ms (overall {best[1]:.3f})")
print("  -> a shift near +250 ms would mean misalignment; near 0 means the lag is real.")

print("\n\nTEST C -- is the early deficit explained by there being no signal yet?\n")
X, y, g, e, off = build_set(0.0)
acc, correct = score(X, y, g)
early = off < 0.25
print(f"  frame energy, first 0.25 s : median {np.median(e[early]):6.1f} dBFS")
print(f"  frame energy, after 2 s    : median {np.median(e[off >= 2]):6.1f} dBFS")
print()
loud = e > np.percentile(e, 50)
print(f"{'window':>14s} {'all frames':>13s} {'louder half only':>18s}")
print("-" * 48)
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 100)):
    m = (off >= lo) & (off < hi)
    ml = m & loud
    lab = f"{lo:g}-{hi:g}s" if hi < 100 else f">{lo:g}s"
    print(f"{lab:>14s} {100*correct[m].mean():12.1f}% "
          f"{(f'{100*correct[ml].mean():.1f}% (n={ml.sum()})' if ml.sum() >= 20 else '-'):>18s}")
