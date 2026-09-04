"""Per-group / per-session breakdown of the readability AUC (W=1 s), all tags.
Same pools as anchor_gate_sim.cmd_readability but keyed by group instead of chain."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/milowu/A4Audio/PureSound/egs/voice_isolate/benchmarks/probes")
import anchor_gate_sim as ag

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
W = 1.0
rows = []
for tag in ("v8", "v16", "v11b"):
    head = ag.load_head(CACHE, tag)
    idx = [e for e in ag.read_index(CACHE, tag, "field") if e["condition"] == "none"]
    pools = {}
    for e in idx:
        if e["side"] == "dt":
            continue
        rec = ag.load_record(e, head, [W])
        g = ag.group_of(e["clip"])
        scope = "session" if e["side"] == "session" else "clip"
        for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
            if not blocks.any():
                continue
            v = rec.C[W][: blocks.size][blocks]
            vd = rec.Cdrr[W][: blocks.size][blocks]
            pools.setdefault((g, scope, kind), []).append((v[np.isfinite(v)], vd[np.isfinite(vd)]))
    for (g, scope, _kind) in sorted({(a, b, c) for (a, b, c) in pools}):
        pass
    keys = sorted({(g, s) for (g, s, k) in pools})
    for (g, s) in keys:
        pk = pools.get((g, s, "keep")); ps = pools.get((g, s, "suppress"))
        if not pk or not ps:
            continue
        K = np.concatenate([a for a, _ in pk]); S = np.concatenate([a for a, _ in ps])
        Kd = np.concatenate([b for _, b in pk]); Sd = np.concatenate([b for _, b in ps])
        rows.append({"tag": tag, "group": g, "scope": s, "chain": ag.chain_of(g),
                     "auc_dist": ag.auc(-K, -S), "auc_drr": ag.auc(Kd, Sd),
                     "keep_m": float(np.median(K)), "supp_m": float(np.median(S)),
                     "keep_drr": float(np.median(Kd)), "supp_drr": float(np.median(Sd)),
                     "nK": int(K.size), "nS": int(S.size)})
Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability/per_group_auc.json").write_text(json.dumps(rows, indent=1))
print(f"{'tag':6s} {'group':22s} {'scope':8s} {'AUCdist':8s} {'AUCdrr':8s} {'keep_m':7s} {'supp_m':7s} {'kDRR':7s} {'sDRR':7s} {'nK':6s} {'nS':6s}")
for r in rows:
    print(f"{r['tag']:6s} {r['group']:22s} {r['scope']:8s} {r['auc_dist']:<8.3f} {r['auc_drr']:<8.3f} "
          f"{r['keep_m']:<7.3f} {r['supp_m']:<7.3f} {r['keep_drr']:<7.2f} {r['supp_drr']:<7.2f} {r['nK']:<6d} {r['nS']:<6d}")
