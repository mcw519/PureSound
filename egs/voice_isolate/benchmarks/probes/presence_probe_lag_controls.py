"""Two controls on the lag result.

1. Extend the shift range. A real misalignment peaks at the true offset and then
   degrades as the window overshoots into the next turn. Monotone improvement with
   no peak means the shift is not finding an alignment, it is censoring the hard
   early frames -- a different conclusion entirely.

2. Re-time everything from AUDIBLE ONSET rather than the labelled boundary. If the
   first 250 ms is mostly the pause between turns, "250 ms of lag" is wall-clock,
   not leaked speech, and the product-relevant number is how long after the new
   voice is actually audible the decision lands.
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


def build(shift_s=0.0, onset_db=None):
    Xs, ys, gs, es, offs = [], [], [], [], []
    sid = 0
    for clip in ("90d_session", "270d_session"):
        X = d[f"{clip}_X"]; dbfs = d[f"{clip}_dbfs"]; fps = float(d[f"{clip}_fps"][0])
        n = X.shape[1]
        for label, key in ((1, "keep"), (0, "suppress")):
            for a, b in windows[clip].get(key, []):
                lo, hi = max(0, int((a + shift_s) * fps)), min(n, int((b + shift_s) * fps))
                if hi - lo < 4:
                    continue
                idx = np.arange(lo, hi)
                if onset_db is None:
                    zero = lo
                else:
                    # First frame in the span whose energy clears the threshold.
                    above = np.nonzero(dbfs[idx] > onset_db)[0]
                    if above.size == 0:
                        continue
                    zero = lo + above[0]
                idx = idx[::STRIDE]
                Xs.append(X[:, idx].T); ys.append(np.full(len(idx), label))
                gs.append(np.full(len(idx), sid)); es.append(dbfs[idx])
                offs.append((idx - zero) / fps)
                sid += 1
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(es), np.concatenate(offs))


def run(X, y, g):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    pred = cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=g)
    return balanced_accuracy_score(y, pred), (pred == y)


print("CONTROL 1 -- extend the shift. A misalignment peaks; censoring keeps rising.\n")
print(f"{'shift':>10s} {'overall':>9s}")
print("-" * 21)
for shift in (0.0, 0.25, 0.40, 0.60, 0.80, 1.20):
    X, y, g, _, _ = build(shift)
    acc, _ = run(X, y, g)
    print(f"{shift*1000:+7.0f} ms {acc:9.3f}")

print("\n\nCONTROL 2 -- re-timed from audible onset (energy > -60 dBFS)\n")
X, y, g, e, off = build(0.0, onset_db=-60.0)
acc, correct = run(X, y, g)
print(f"  overall balanced acc {acc:.3f}\n")
print(f"{'since onset':>14s} {'correct':>9s} {'n':>7s}")
print("-" * 33)
for lo, hi in ((-1, 0), (0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 100)):
    m = (off >= lo) & (off < hi)
    if m.sum() < 15:
        continue
    lab = "before onset" if hi == 0 else (f"{lo:g}-{hi:g}s" if hi < 100 else f">{lo:g}s")
    print(f"{lab:>14s} {100*correct[m].mean():8.1f}% {m.sum():7d}")
