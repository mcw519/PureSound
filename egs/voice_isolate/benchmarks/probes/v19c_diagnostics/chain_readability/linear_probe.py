"""Is the near/far information ABSENT from the bottleneck on the QVF chain, or present but
mapped the wrong way by the DistHead?

Fits a fresh linear probe (logistic regression) on the SAME cached pooled bottleneck frames the
DistHead reads, and reports:
  * device-fit  -> device-test  (leave-one-group-out): the ceiling on the rig chain
  * device-fit  -> QVF-test:    does a device-learned boundary transfer? (this is what the
                                DistHead effectively is: a boundary fit on device-like data)
  * QVF-fit     -> QVF-test     (leave-one-QVF-group-out): is the information THERE at all?
Frames come from `condition == none`, keep spans (near) vs suppress spans (far), same as
`anchor_gate_sim.py readability`. Fitting is per-frame; the resampling/holdout unit is the
recording group, so no group appears in both fit and test.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/milowu/A4Audio/PureSound/egs/voice_isolate/benchmarks/probes")
import anchor_gate_sim as ag
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
OUT = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability")
W = 1.0


def pooled_window(feat, act, n):
    """Mean of feat over the last n ACTIVE frames, evaluated at every active frame -- the exact
    input the sliding DistHead readout gets."""
    F = feat.shape[1]
    ai = np.flatnonzero(act[:F])
    X = np.full((F, feat.shape[0]), np.nan, dtype=np.float64)
    if ai.size == 0:
        return X
    A = feat[:, ai].T.astype(np.float64)
    cs = np.concatenate([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)], axis=0)
    k = np.arange(1, ai.size + 1)
    lo = np.maximum(0, k - n)
    X[ai] = (cs[k] - cs[lo]) / (k - lo)[:, None]
    return X


rows = []
for tag in ("v8", "v16", "v11b"):
    head = ag.load_head(CACHE, tag)
    data = {}      # group -> (X, y, chain)
    for e in ag.read_index(CACHE, tag, "field"):
        if e["condition"] != "none" or e["side"] not in ("near", "far"):
            continue
        d = np.load(e["file"], allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        feat = d["feat"].astype(np.float32)
        F = feat.shape[1]
        e_db = ag.frame_energy_db(d["mix"], F)
        floor = meta.get("floor_dbfs")
        thr = (floor if floor is not None else np.percentile(e_db, 5)) + ag.ACTIVE_MARGIN_DB
        act = ag.activity(e_db, thr)
        X = pooled_window(feat, act, int(W * ag.FPS))
        for kind, lab in (("keep", 1), ("suppress", 0)):
            spans = meta["spans"].get(kind) or []
            if not spans:
                continue
            b = ag.span_blocks(spans, meta["offset_s"], F)
            m = b & np.isfinite(X).all(axis=1)
            if not m.any():
                continue
            g = ag.group_of(e["clip"])
            gx, gy, _ = data.get(g, (None, None, None))
            xs = X[m]; ys = np.full(xs.shape[0], lab)
            data[g] = (xs if gx is None else np.vstack([gx, xs]),
                       ys if gy is None else np.concatenate([gy, ys]),
                       ag.chain_of(g))
    dev = [g for g, v in data.items() if v[2] == "device"]
    qvf = [g for g, v in data.items() if v[2] == "qvf"]

    def fit(groups):
        X = np.vstack([data[g][0] for g in groups]); y = np.concatenate([data[g][1] for g in groups])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(X, y)
        return clf

    def test(clf, groups):
        X = np.vstack([data[g][0] for g in groups]); y = np.concatenate([data[g][1] for g in groups])
        s = clf.decision_function(X)
        return ag.auc(s[y == 1], s[y == 0]), int((y == 1).sum()), int((y == 0).sum())

    # device leave-one-group-out
    a = []
    for g in dev:
        rest = [h for h in dev if h != g]
        if not rest or len({data[h][1][0] for h in [g]}) == 0:
            continue
        yg = data[g][1]
        if len(np.unique(yg)) < 2:
            continue
        a.append(test(fit(rest), [g])[0])
    rows.append({"tag": tag, "arm": "device-fit -> device-test (LOGO)", "auc": float(np.mean(a)),
                 "detail": ", ".join(f"{x:.3f}" for x in a), "n_groups": len(a)})
    clf_dev = fit(dev)
    au, nk, ns = test(clf_dev, qvf)
    rows.append({"tag": tag, "arm": "device-fit -> QVF-test (pooled)", "auc": au,
                 "detail": f"nK={nk} nS={ns}", "n_groups": len(qvf)})
    per = []
    for g in qvf:
        if len(np.unique(data[g][1])) < 2:
            continue
        per.append((g, test(clf_dev, [g])[0]))
    rows.append({"tag": tag, "arm": "device-fit -> QVF-test (per group)", "auc": float("nan"),
                 "detail": ", ".join(f"{g}={v:.3f}" for g, v in per), "n_groups": len(per)})
    # QVF leave-one-group-out
    b = []
    for g in qvf:
        rest = [h for h in qvf if h != g]
        if len(np.unique(data[g][1])) < 2 or len(np.unique(np.concatenate([data[h][1] for h in rest]))) < 2:
            continue
        b.append((g, test(fit(rest), [g])[0]))
    rows.append({"tag": tag, "arm": "QVF-fit -> QVF-test (LOGO)",
                 "auc": float(np.mean([v for _, v in b])) if b else float("nan"),
                 "detail": ", ".join(f"{g}={v:.3f}" for g, v in b), "n_groups": len(b)})
    # within-group (in-sample upper bound on 'is it linearly there')
    c = []
    for g in qvf:
        if len(np.unique(data[g][1])) < 2:
            continue
        c.append((g, test(fit([g]), [g])[0]))
    rows.append({"tag": tag, "arm": "QVF within-group in-sample (info ceiling)",
                 "auc": float(np.mean([v for _, v in c])) if c else float("nan"),
                 "detail": ", ".join(f"{g}={v:.3f}" for g, v in c), "n_groups": len(c)})

(OUT / "linear_probe.json").write_text(json.dumps(rows, indent=1))
print(f"{'tag':6s} {'arm':42s} {'AUC':7s} detail")
for r in rows:
    a = "n/a   " if r["auc"] != r["auc"] else f"{r['auc']:<7.3f}"
    print(f"{r['tag']:6s} {r['arm']:42s} {a} {r['detail']}")
