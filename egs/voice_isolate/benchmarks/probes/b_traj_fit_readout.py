"""Fit the presence readout once and persist it as system weights.

Fitted on the 90D recording ONLY. The cold-start clips are excerpts of the same
two source recordings the sessions are (windows.json: `source: real_record_test_*`),
so anything fitted on a recording is in-sample for every clip cut from it. Holding
out 270D keeps half the field set honest; the other eight benchmark stages are
different corpora entirely.

The StandardScaler is folded into the coefficients so the runtime is one dot
product:  z = (w/sigma) . x + (b0 - sum(w*mu/sigma))
"""
import json, os, pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
from b_traj import load, role, training_rows, AUDIBLE_DBFS

TRAIN_GROUP = "90d"
STRIDE = 5

clips = load()
X, y = training_rows(clips, TRAIN_GROUP)
pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
pipe.fit(X[::STRIDE], y[::STRIDE])

scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
w = lr.coef_.ravel() / scaler.scale_
b = float(lr.intercept_[0] - np.dot(lr.coef_.ravel(), scaler.mean_ / scaler.scale_))

# Fold-check: the folded form must reproduce the pipeline bit-for-bit-ish.
probe = X[:2000]
ref = pipe.predict_proba(probe)[:, 1]
got = 1.0 / (1.0 + np.exp(-(probe @ w + b)))
assert np.abs(ref - got).max() < 1e-6, np.abs(ref - got).max()
print(f"folded readout matches the pipeline to {np.abs(ref-got).max():.2e}")

# Held-out score, for the record.
ev_p, ev_y = [], []
for name, c in clips.items():
    if c["group"] == TRAIN_GROUP or c["session"]:
        continue
    r = role(name, c["spec"])
    aud = c["dbfs"] > AUDIBLE_DBFS
    if r in ("near", "dt", "far") and aud.any():
        p = 1.0 / (1.0 + np.exp(-(c["X"][aud] @ w + b)))
        ev_p.append(p); ev_y.append(np.full(int(aud.sum()), 0 if r == "far" else 1))
ev_p, ev_y = np.concatenate(ev_p), np.concatenate(ev_y)
print(f"held-out 270d cold-start frames n={len(ev_y)}  "
      f"balanced acc {balanced_accuracy_score(ev_y, (ev_p>0.5).astype(int)):.3f}  "
      f"AUC {roc_auc_score(ev_y, ev_p):.3f}")

out = pathlib.Path(sys.argv[1])
np.savez(out, weight=w.astype(np.float32), bias=np.float32(b))
meta = {
    "readout": "linear on the frozen dpcrn bottleneck, pooled over frequency",
    "checkpoint": "dpcrn_v8",
    "fitted_on": f"{TRAIN_GROUP} field recording only (270d held out)",
    "n_train_frames": int(len(y[::STRIDE])),
    "audible_dbfs": AUDIBLE_DBFS,
    "held_out_balanced_acc": round(float(balanced_accuracy_score(ev_y, (ev_p>0.5).astype(int))), 4),
    "held_out_auc": round(float(roc_auc_score(ev_y, ev_p)), 4),
    "dim": int(w.shape[0]),
}
out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", out, "and", out.with_suffix(".json"))
