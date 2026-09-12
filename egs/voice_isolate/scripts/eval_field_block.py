"""Field benchmark, block protocol -- the whole set v3 table, scored so it can resolve.

Every field number this recipe has ever compared was one checkpoint on one cut. The
same run's neighbouring epochs (v16 ep8-19, no knob change) move the cold-far median
across -0.35..-3.39 dB, session means by sd 1.95 dB, KEEP violation counts across 2..5
-- so most single-point deltas on record were noise (COLDSTART_V2.md). This tool scores
a BLOCK of checkpoints (the run's last five, fixed a priori) over every clip in the set
and reports block means with the per-checkpoint spread; `compare` pairs two blocks per
clip and runs Wilcoxon signed-rank per group. Cold-start clips also get the 3 s ambient
lead-in condition from eval_coldstart_v2 (deployment steady state).

Score:   uv run python scripts/eval_field_block.py score config/infer_dpcrn.yaml \
             --ckpt A.ckpt --ckpt B.ckpt ... --tag v16ep19_block --out out.json
Compare: uv run python scripts/eval_field_block.py compare A.json B.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR / "scripts"))

from puresound.config import load_recipe                       # noqa: E402
from puresound.recipes import init_siso_model                  # noqa: E402
from eval_realcase import (span_dbfs, KEEP_VIOLATION_DB,       # noqa: E402
                           SUPPRESS_FAIL_DB, SUPPRESS_PARTIAL_DB)
from eval_coldstart_v2 import (CASES, PAD_SECONDS, SR,         # noqa: E402
                               ambient_pads, group_of, load_wav)


def chain_of(clip: str) -> str:
    return "qvf" if clip.startswith("qvf") else "device"


def score_ckpt(model, windows, device, with_ambient: bool):
    rows = []
    for clip, spec in windows.items():
        if clip.startswith("_"):
            continue
        keep, supp = spec.get("keep") or [], spec.get("suppress") or []
        if not keep and not supp:
            continue
        is_session = clip.endswith("_session")
        wav = load_wav(clip)
        floor = spec.get("floor_dbfs")
        near_ref = spec.get("near_ref_dbfs", float("nan"))
        conditions = [("none", None)]
        if with_ambient and not is_session:
            conditions += [(f"ambient{i}", p) for i, p in enumerate(ambient_pads(windows, clip))]
        for cond, pad in conditions:
            x = torch.cat([pad.view(1, -1), wav], dim=-1) if pad is not None else wav
            off = PAD_SECONDS if pad is not None else 0.0
            with torch.no_grad():
                out = model(x.to(device)).detach().cpu().view(1, -1)
            n = min(out.shape[-1], x.shape[-1])
            row = {"clip": clip, "chain": chain_of(clip), "group": group_of(clip),
                   "kind": "session" if is_session else ("dt" if "_dt" in clip else
                           "near" if keep else "far"),
                   "condition": "ambient" if cond.startswith("ambient") else "none", "draw": cond}
            if keep:
                sh = [[a + off, b + off] for a, b in keep]
                row["keep_db"] = span_dbfs(out, sh, SR, n) - span_dbfs(x, sh, SR, n)
            if supp:
                sh = [[a + off, b + off] for a, b in supp]
                out_db = span_dbfs(out, sh, SR, n)
                row["supp_db"] = out_db - span_dbfs(x, sh, SR, n)
                if floor is not None:
                    row["residual_db"] = out_db - floor
                if near_ref == near_ref:
                    row["sir_out_db"] = near_ref - out_db
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
def block_means(rec):
    """(clip, condition) -> {metric: (mean over ckpts+draws, sd over ckpts)}"""
    acc = {}
    for r in rec["rows"]:
        k = (r["clip"], r["condition"])
        for m in ("keep_db", "supp_db", "residual_db"):
            if m in r:
                acc.setdefault(k, {}).setdefault(m, {}).setdefault(r["ckpt"], []).append(r[m])
    out = {}
    for k, ms in acc.items():
        out[k] = {}
        for m, per_ck in ms.items():
            per = [float(np.mean(v)) for v in per_ck.values()]
            out[k][m] = (float(np.mean(per)), float(np.std(per)) if len(per) > 1 else 0.0)
    return out


def summarize(rec):
    bm = block_means(rec)
    meta = {r["clip"]: r for r in rec["rows"]}
    def pick(kind, cond, metric, chain=None):
        return {c: v[metric][0] for (c, cd), v in bm.items()
                if cd == cond and meta[c]["kind"] == kind and metric in v
                and (chain is None or meta[c]["chain"] == chain)}
    def spread(kind, cond, metric):
        s = [v[metric][1] for (c, cd), v in bm.items() if cd == cond and meta[c]["kind"] == kind and metric in v]
        return float(np.median(s)) if s else float("nan")
    print(f"\n== {rec['tag']}  (block of {len(rec['ckpts'])} ckpts)  "
          f"[per-ckpt sd, median over clips, shown as ±]")
    s = pick("session", "none", "supp_db")
    print(f"  sessions supp  median {np.median(list(s.values())):+7.2f} ±{spread('session','none','supp_db'):.2f}   "
          + "  ".join(f"{c.replace('_session','')} {v:+.2f}" for c, v in sorted(s.items())))
    for chain in ("device", "qvf"):
        for cond in ("none", "ambient"):
            f = pick("far", cond, "supp_db", chain)
            if not f:
                continue
            v = np.array(list(f.values()))
            res = pick("far", cond, "residual_db", chain)
            ok = sum(1 for c in f if c in res and res[c] <= SUPPRESS_PARTIAL_DB)
            fail = int((v > SUPPRESS_FAIL_DB).sum())
            print(f"  cold-far {chain:6s} {cond:8s} n={len(v):2d} median {np.median(v):+7.2f} "
                  f"±{spread('far',cond,'supp_db'):.2f}  ok {ok} / FAIL {fail}  residual med {np.median(list(res.values())):5.1f}")
    for kind in ("near", "dt"):
        for cond in ("none", "ambient"):
            k = pick(kind, cond, "keep_db")
            if not k:
                continue
            v = np.array(list(k.values()))
            viol = sorted(((c, x) for c, x in k.items() if x < KEEP_VIOLATION_DB), key=lambda t: t[1])
            print(f"  keep {kind:4s} {cond:8s} n={len(v):2d} median {np.median(v):+6.2f} ±{spread(kind,cond,'keep_db'):.2f}  "
                  f"violations {len(viol)}  worst {v.min():+.2f}"
                  + (f"  [{', '.join(f'{c}:{x:+.1f}' for c, x in viol[:4])}]" if viol else ""))
    sk = pick("session", "none", "keep_db")
    print(f"  sessions keep  median {np.median(list(sk.values())):+6.2f}")


def cmd_score(args):
    windows = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items()
               if not k.startswith("_")}
    device = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config_path, expected_task="voice_isolation").model)
    rows = []
    for ck in args.ckpt:
        st = torch.load(ck, map_location="cpu")
        model.reload_checkpoint(st.get("state_dict", st), load_loss_func=False)
        model = model.to(device).eval()
        r = score_ckpt(model, windows, device, not args.no_ambient)
        for x in r:
            x["ckpt"] = Path(ck).name
        rows.extend(r)
        print(f"  scored {Path(ck).name}: {len(r)} rows", flush=True)
    rec = {"tag": args.tag, "ckpts": [str(c) for c in args.ckpt], "rows": rows}
    Path(args.out).write_text(json.dumps(rec, indent=1))
    summarize(rec)
    print(f"\nwrote {args.out}")


def cmd_compare(args):
    from scipy.stats import wilcoxon
    a, b = (json.loads(Path(p).read_text()) for p in args.results)
    ma, mb = block_means(a), block_means(b)
    meta = {r["clip"]: r for r in a["rows"]}
    print(f"compare {a['tag']}  vs  {b['tag']}   (paired per clip on block means; delta = A - B)")
    groups = [("sessions supp", "session", None, "none", "supp_db"),
              ("cold-far device none", "far", "device", "none", "supp_db"),
              ("cold-far device ambient", "far", "device", "ambient", "supp_db"),
              ("cold-far qvf none", "far", "qvf", "none", "supp_db"),
              ("keep near none", "near", None, "none", "keep_db"),
              ("keep near ambient", "near", None, "ambient", "keep_db"),
              ("keep double-talk", "dt", None, "none", "keep_db"),
              ("sessions keep", "session", None, "none", "keep_db")]
    for label, kind, chain, cond, metric in groups:
        keys = sorted(k for k in ma if k in mb and k[1] == cond and meta[k[0]]["kind"] == kind
                      and (chain is None or meta[k[0]]["chain"] == chain)
                      and metric in ma[k] and metric in mb[k])
        if len(keys) < 3:
            continue
        xa = np.array([ma[k][metric][0] for k in keys]); xb = np.array([mb[k][metric][0] for k in keys])
        d = xa - xb
        p = float("nan")
        if len(keys) >= 6:
            try:
                p = wilcoxon(xa, xb).pvalue
            except ValueError:
                pass
        tag = "" if p != p else ("  **" if p < 0.05 else "")
        print(f"  {label:24s} n={len(keys):2d}  {np.median(xa):+7.2f} vs {np.median(xb):+7.2f}  "
              f"delta med {np.median(d):+6.2f}  p={p:.3f}{tag}")
    # severity table: keep clips that moved > 1 dB, by chain
    moved = []
    for k in ma:
        if k in mb and k[1] == "none" and "keep_db" in ma[k] and "keep_db" in mb[k]:
            d = ma[k]["keep_db"][0] - mb[k]["keep_db"][0]
            if abs(d) >= 1.0:
                moved.append((meta[k[0]]["chain"], k[0], mb[k]["keep_db"][0], ma[k]["keep_db"][0], d))
    if moved:
        print("  keep severity changes >= 1 dB (B -> A):")
        for chain, c, vb, va, d in sorted(moved, key=lambda t: t[4]):
            print(f"    [{chain:6s}] {c:26s} {vb:+7.2f} -> {va:+7.2f}  ({d:+.2f})")
    else:
        print("  keep severity: no clip moved >= 1 dB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score"); sc.add_argument("config_path")
    sc.add_argument("--ckpt", action="append", required=True); sc.add_argument("--tag", required=True)
    sc.add_argument("--out", required=True); sc.add_argument("--device", default="cuda")
    sc.add_argument("--no-ambient", action="store_true"); sc.set_defaults(func=cmd_score)
    cp = sub.add_parser("compare"); cp.add_argument("results", nargs=2); cp.set_defaults(func=cmd_compare)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
