"""PAIRED clip-level bootstrap on AUC differences between checkpoints (W=1 s).
The same resampled clip indices are used for every tag, so the comparison is paired and the
between-recording variance that dominates the marginal CIs partly cancels."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/milowu/A4Audio/PureSound/egs/voice_isolate/benchmarks/probes")
import anchor_gate_sim as ag

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
W, B = 1.0, 4000
TAGS = ("v8", "v16", "v11b")
per = {t: {} for t in TAGS}       # tag -> (chain, scope, kind) -> {clip: (dist, drr)}
for tag in TAGS:
    head = ag.load_head(CACHE, tag)
    for e in ag.read_index(CACHE, tag, "field"):
        if e["condition"] != "none" or e["side"] == "dt":
            continue
        rec = ag.load_record(e, head, [W])
        ch = ag.chain_of(ag.group_of(e["clip"]))
        scope = "session" if e["side"] == "session" else "clip"
        for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
            if not blocks.any():
                continue
            v = rec.C[W][: blocks.size][blocks]; vd = rec.Cdrr[W][: blocks.size][blocks]
            per[tag].setdefault((ch, scope, kind), {})[e["clip"]] = (v[np.isfinite(v)], vd[np.isfinite(vd)])

rng = np.random.default_rng(7)
rows = []
for ch in ("device", "qvf"):
    for scope in ("clip", "session"):
        kk = per["v8"].get((ch, scope, "keep")); ss = per["v8"].get((ch, scope, "suppress"))
        if not kk or not ss:
            continue
        kclips = sorted(kk); sclips = sorted(ss)
        for name, j, sign in (("dist", 0, -1.0), ("drr", 1, 1.0)):
            def point(tag, ki, si):
                K = np.concatenate([per[tag][(ch, scope, "keep")][kclips[i]][j] for i in ki])
                S = np.concatenate([per[tag][(ch, scope, "suppress")][sclips[i]][j] for i in si])
                return ag.auc(sign * K, sign * S)
            base = {t: point(t, range(len(kclips)), range(len(sclips))) for t in TAGS}
            draws = {("v11b", "v8"): [], ("v11b", "v16"): [], ("v16", "v8"): []}
            for _ in range(B):
                ki = rng.integers(0, len(kclips), len(kclips)); si = rng.integers(0, len(sclips), len(sclips))
                p = {t: point(t, ki, si) for t in TAGS}
                for (a, b) in draws:
                    draws[(a, b)].append(p[a] - p[b])
            for (a, b), d in draws.items():
                d = np.array(d)
                rows.append({"chain": ch, "scope": scope, "readout": name, "pair": f"{a}-{b}",
                             "delta": base[a] - base[b], "lo95": float(np.percentile(d, 2.5)),
                             "hi95": float(np.percentile(d, 97.5)),
                             "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                             "n_keep_units": len(kclips), "n_supp_units": len(sclips),
                             **{f"auc_{t}": base[t] for t in TAGS}})
Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability/auc_delta.json").write_text(json.dumps(rows, indent=1))
print(f"{'chain':7s} {'scope':8s} {'rd':5s} {'pair':10s} {'v8':6s} {'v16':6s} {'v11b':6s} {'delta':7s} {'95% CI':18s} {'p':6s}")
for r in rows:
    print(f"{r['chain']:7s} {r['scope']:8s} {r['readout']:5s} {r['pair']:10s} {r['auc_v8']:<6.3f} {r['auc_v16']:<6.3f} "
          f"{r['auc_v11b']:<6.3f} {r['delta']:<+7.3f} [{r['lo95']:+.3f},{r['hi95']:+.3f}]   {r['p_two_sided']:.3f}")
