import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def mk(C=0.01):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10,
                    help="keep every Nth frame; at 100 fps consecutive frames are "
                         "redundant and 25 x (1+n_perm) fits on 21k rows is not worth it")
    a = ap.parse_args()
    d = np.load(a.npz)
    X, y, g = d["X"][::a.stride], d["y"][::a.stride], d["g"][::a.stride]
    spans = np.unique(g)
    print(f"{len(y)} frames, {len(spans)} spans, {int(y.sum())} near / {int((1-y).sum())} far\n")

    print("Why the split matters -- the same data, two ways of cutting it:\n")
    p = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"  split by FRAME (wrong): balanced acc {balanced_accuracy_score(y, p):.3f}")
    print("     neighbouring frames are nearly identical, so most of a test frame's")
    print("     own span is in the training set. This number is not a measurement.")
    p = cross_val_predict(mk(), X, y, cv=GroupKFold(5), groups=g)
    print(f"  split by SPAN  (right): balanced acc {balanced_accuracy_score(y, p):.3f}")

    print("\nLeave-one-span-out, and what shuffled span labels score:\n")
    pred = cross_val_predict(mk(), X, y, cv=LeaveOneGroupOut(), groups=g)
    prob = cross_val_predict(mk(), X, y, cv=LeaveOneGroupOut(), groups=g,
                             method="predict_proba")[:, 1]
    acc, auc = balanced_accuracy_score(y, pred), roc_auc_score(y, prob)
    print(f"  real labels : balanced acc {acc:.3f}   AUC {auc:.3f}")

    # Permute the label OF EACH SPAN, keeping every frame in a span together.
    span_label = {s: y[g == s][0] for s in spans}
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.n_perm):
        shuffled = rng.permutation([span_label[s] for s in spans])
        yp = np.empty_like(y)
        for s, lab in zip(spans, shuffled):
            yp[g == s] = lab
        null.append(balanced_accuracy_score(
            yp, cross_val_predict(mk(), X, yp, cv=GroupKFold(5), groups=g)))
    null = np.array(null)
    print(f"  shuffled    : mean {null.mean():.3f}  p95 {np.percentile(null,95):.3f}  "
          f"max {null.max():.3f}")
    print(f"  -> p = {((null >= acc).sum() + 1) / (a.n_perm + 1):.4f}")

    print("\nPer-span detail (leave-one-span-out):\n")
    for s in spans:
        m = g == s
        frac = float((pred[m] == y[m]).mean())
        print(f"    span {int(s):2d}  {'near' if y[m][0] else 'far ':4s}  "
              f"{m.sum():5d} frames  correct {100*frac:5.1f}%")
