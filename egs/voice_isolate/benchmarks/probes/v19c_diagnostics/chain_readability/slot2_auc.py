"""The DistHead has three output slots [drr/10, log10 fg_m, log10 itf_m]; `readability` only ever
used slots 0 and 1. Does the INTERFERER-distance slot separate near from far on the QVF chain?"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/milowu/A4Audio/PureSound/egs/voice_isolate/benchmarks/probes")
import anchor_gate_sim as ag

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
W, n = 1.0, 100
rows = []
for tag in ("v8", "v16", "v11b"):
    head = ag.load_head(CACHE, tag)
    pools = {}
    for e in ag.read_index(CACHE, tag, "field"):
        if e["condition"] != "none" or e["side"] == "dt":
            continue
        d = np.load(e["file"], allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        feat = d["feat"].astype(np.float32)
        F = feat.shape[1]
        e_db = ag.frame_energy_db(d["mix"], F)
        thr = (meta.get("floor_dbfs") if meta.get("floor_dbfs") is not None
               else np.percentile(e_db, 5)) + ag.ACTIVE_MARGIN_DB
        act = ag.activity(e_db, thr)
        ai = np.flatnonzero(act[:F])
        if ai.size == 0:
            continue
        A = feat[:, ai].T.astype(np.float64)
        cs = np.concatenate([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)], axis=0)
        k = np.arange(1, ai.size + 1); lo = np.maximum(0, k - n)
        y = ag.head_apply(head, (cs[k] - cs[lo]) / (k - lo)[:, None])
        full = np.full((F, 3), np.nan); full[ai] = y
        ch = ag.chain_of(ag.group_of(e["clip"]))
        scope = "session" if e["side"] == "session" else "clip"
        for kind, spans in (("keep", meta["spans"].get("keep") or []),
                            ("suppress", meta["spans"].get("suppress") or [])):
            if not spans:
                continue
            b = ag.span_blocks(spans, meta["offset_s"], F)
            v = full[b]
            v = v[np.isfinite(v).all(axis=1)]
            if v.size:
                pools.setdefault((ch, scope, kind), []).append(v)
    for ch in ("device", "qvf"):
        for scope in ("clip", "session"):
            K, S = pools.get((ch, scope, "keep")), pools.get((ch, scope, "suppress"))
            if not K or not S:
                continue
            K, S = np.vstack(K), np.vstack(S)
            rows.append({"tag": tag, "chain": ch, "scope": scope,
                         "auc_drr": ag.auc(K[:, 0], S[:, 0]),
                         "auc_fg_dist": ag.auc(-K[:, 1], -S[:, 1]),
                         "auc_itf_dist": ag.auc(K[:, 2], S[:, 2]),
                         "med_itf_keep_m": float(np.median(10 ** K[:, 2])),
                         "med_itf_supp_m": float(np.median(10 ** S[:, 2]))})
Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability/slot2_auc.json").write_text(json.dumps(rows, indent=1))
print(f"{'tag':6s} {'chain':7s} {'scope':8s} {'AUCdrr':7s} {'AUCfg':7s} {'AUCitf':7s} {'itf_keep_m':10s} {'itf_supp_m':10s}")
for r in rows:
    print(f"{r['tag']:6s} {r['chain']:7s} {r['scope']:8s} {r['auc_drr']:<7.3f} {r['auc_fg_dist']:<7.3f} "
          f"{r['auc_itf_dist']:<7.3f} {r['med_itf_keep_m']:<10.3f} {r['med_itf_supp_m']:<10.3f}")
