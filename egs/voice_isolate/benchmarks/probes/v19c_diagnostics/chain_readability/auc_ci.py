"""Clip-level bootstrap CI on the readability AUC (W=1 s), so 'what number must a round hit'
has a noise band. Frames inside a clip are heavily correlated, so the resampling unit is the
CLIP (keep clips and suppress clips resampled independently, with replacement)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/milowu/A4Audio/PureSound/egs/voice_isolate/benchmarks/probes")
import anchor_gate_sim as ag

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
W = 1.0
B = 2000
rng = np.random.default_rng(0)
out = []
for tag in ("v8", "v16", "v11b"):
    head = ag.load_head(CACHE, tag)
    idx = [e for e in ag.read_index(CACHE, tag, "field") if e["condition"] == "none"]
    per = {}   # (chain, scope, kind) -> list of (dist_arr, drr_arr) per clip
    for e in idx:
        if e["side"] == "dt":
            continue
        rec = ag.load_record(e, head, [W])
        ch = ag.chain_of(ag.group_of(e["clip"]))
        scope = "session" if e["side"] == "session" else "clip"
        for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
            if not blocks.any():
                continue
            v = rec.C[W][: blocks.size][blocks]; vd = rec.Cdrr[W][: blocks.size][blocks]
            per.setdefault((ch, scope, kind), []).append((v[np.isfinite(v)], vd[np.isfinite(vd)]))
    for ch in ("device", "qvf"):
        for scope in ("clip", "session"):
            K = per.get((ch, scope, "keep")); S = per.get((ch, scope, "suppress"))
            if not K or not S:
                continue
            for name, j, sign in (("dist", 0, -1.0), ("drr", 1, 1.0)):
                point = ag.auc(sign * np.concatenate([a[j] for a in K]),
                               sign * np.concatenate([a[j] for a in S]))
                bs = []
                for _ in range(B):
                    ki = rng.integers(0, len(K), len(K)); si = rng.integers(0, len(S), len(S))
                    bs.append(ag.auc(sign * np.concatenate([K[i][j] for i in ki]),
                                     sign * np.concatenate([S[i][j] for i in si])))
                bs = np.array(bs)
                out.append({"tag": tag, "chain": ch, "scope": scope, "readout": name,
                            "auc": point, "lo95": float(np.percentile(bs, 2.5)),
                            "hi95": float(np.percentile(bs, 97.5)),
                            "n_keep_units": len(K), "n_supp_units": len(S)})
Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability/auc_ci.json").write_text(json.dumps(out, indent=1))
print(f"{'tag':6s} {'chain':7s} {'scope':8s} {'readout':8s} {'AUC':6s} {'95% CI':20s} {'nK':4s} {'nS':4s}")
for r in out:
    print(f"{r['tag']:6s} {r['chain']:7s} {r['scope']:8s} {r['readout']:8s} {r['auc']:<6.3f} "
          f"[{r['lo95']:.3f}, {r['hi95']:.3f}]{'':6s} {r['n_keep_units']:<4d} {r['n_supp_units']:<4d}")
