"""How long does the decision take to settle after a turn changes?

The span-length correlation (rho +0.39, p=0.055 over 25 spans) only hinted at
this. Aligning every span to its own start instead gives hundreds of frames per
time bin, which is the properly-powered version of the same question -- and it is
the number a session-start calibration window has to be sized against.

Predictions come from leave-one-span-out, so a frame's own span is never in the
training set.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

S = "/tmp/claude-1001/-home-milowu-A4Audio-PureSound/7b76fa14-55b2-4c0a-b797-c8214d986949/scratchpad/presence"
FPS = 99.8
STRIDE = 2

d = np.load(f"{S}/frames_v8.npz")
X, y, g = d["X"][::STRIDE], d["y"][::STRIDE], d["g"][::STRIDE]
# Offset from the start of each span, in seconds.
off = np.zeros(len(y))
for s in np.unique(g):
    m = g == s
    off[m] = np.arange(m.sum()) * STRIDE / FPS

clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
pred = cross_val_predict(clf, X, y, cv=LeaveOneGroupOut(), groups=g)
correct = (pred == y)
print(f"{len(y)} frames, {len(np.unique(g))} spans, overall balanced acc "
      f"{balanced_accuracy_score(y, pred):.3f}\n")

EDGES = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 100]
print("判對率 vs 距離段落開頭的時間\n")
print(f"{'時間窗':>12s} {'全部':>16s} {'near 段':>16s} {'far 段':>16s}")
print("-" * 64)
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    m = (off >= lo) & (off < hi)
    if m.sum() < 20:
        continue
    cells = []
    for sel, _name in ((m, "all"), (m & (y == 1), "near"), (m & (y == 0), "far")):
        cells.append(f"{100*correct[sel].mean():5.1f}% (n={sel.sum():4d})" if sel.sum() >= 10 else "      -       ")
    label = f"{lo:g}-{hi:g}s" if hi < 100 else f">{lo:g}s"
    print(f"{label:>12s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")

print("\n第一次穩定超過 90% 的時間點（連續兩個窗都過）:")
run = 0
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    m = (off >= lo) & (off < hi)
    if m.sum() < 20:
        continue
    ok = correct[m].mean() >= 0.90
    run = run + 1 if ok else 0
    if run == 2:
        print(f"   ~{lo:g} 秒")
        break
else:
    print("   在測到的範圍內沒有穩定超過 90%")
