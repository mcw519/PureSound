"""Is "is there a near user" linearly decodable from the frozen bottleneck?

Three questions, and only the third is new:

  (i)   fit synthetic -> test synthetic held-out.  Does the information exist
        in-domain?  2026-07-10's gate-only run answered yes (balanced_acc 0.90)
        for a model trained without real data.
  (ii)  fit synthetic -> test the 25 real field clips.  Does it transfer?  This
        is the axis that failed in 2026-07-10.
  (iii) fit on the real clips themselves, leave-one-out.  Is the information in
        the bottleneck on the real chain AT ALL, regardless of transfer?

(iii) is decisive.  Fails -> the representation does not carry it and an explicit
presence head repeats 2026-07-10.  Passes -> it carries it and the gap is
transfer/decision, which is fixable without new data.

n=25 with d=128 separates RANDOM labels, so every real-side number is reported
against a label-permutation baseline.  Without that this file would be a
generator of false positives.
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def model(C=0.01):
    # Heavy L2: d=128 against n=25 on the real side.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def loo_scores(X, y, C):
    pred = cross_val_predict(model(C), X, y, cv=LeaveOneOut())
    prob = cross_val_predict(model(C), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return balanced_accuracy_score(y, pred), roc_auc_score(y, prob)


def permuted(X, y, C, n_perm, rng):
    out = []
    for _ in range(n_perm):
        out.append(loo_scores(X, rng.permutation(y), C)[0])
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()
    d = np.load(a.npz)
    Xs, ys, Xr, yr = d["Xs"], d["ys"], d["Xr"], d["yr"]
    rng = np.random.default_rng(0)
    print(f"synthetic n={len(ys)} ({int(ys.sum())} present / {int((1-ys).sum())} absent)   "
          f"real n={len(yr)} ({int(yr.sum())} / {int((1-yr).sum())})\n")

    print("(i)  in-domain: fit synthetic, 5-fold CV on synthetic")
    pred = cross_val_predict(model(a.C), Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    prob = cross_val_predict(model(a.C), Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]
    print(f"     balanced acc {balanced_accuracy_score(ys, pred):.3f}   AUC {roc_auc_score(ys, prob):.3f}")

    print("\n(ii) transfer: fit ALL synthetic, test the 25 real clips")
    clf = model(a.C).fit(Xs, ys)
    pr = clf.predict(Xr); pp = clf.predict_proba(Xr)[:, 1]
    print(f"     balanced acc {balanced_accuracy_score(yr, pr):.3f}   AUC {roc_auc_score(yr, pp):.3f}")
    print(f"     mean p(present):  真的有人 {pp[yr==1].mean():.3f}   真的沒人 {pp[yr==0].mean():.3f}")

    print("\n(iii) real-chain decodability: leave-one-out ON the 25 real clips")
    acc, auc = loo_scores(Xr, yr, a.C)
    null = permuted(Xr, yr, a.C, a.n_perm, rng)
    p = float((null >= acc).sum() + 1) / (a.n_perm + 1)
    print(f"     balanced acc {acc:.3f}   AUC {auc:.3f}")
    print(f"     permutation null (n={a.n_perm}): mean {null.mean():.3f}  p95 {np.percentile(null,95):.3f}  max {null.max():.3f}")
    print(f"     -> p = {p:.4f}   {'ABOVE chance' if p < 0.05 else 'NOT distinguishable from chance'}")
