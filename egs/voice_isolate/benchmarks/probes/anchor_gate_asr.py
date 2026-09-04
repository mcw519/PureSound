"""Step 3 of the anchor-qualification gate: real ASR deletion on Dawn Chorus.

`anchor_gate_README` §4 measured the gate's keep-side effect with an ENERGY PROXY
(output energy over reference-active frames, relative to the mix). This turns that
proxy into faster-whisper large-v3 WER / substitution / insertion / deletion, for
every (checkpoint tag) x (context: none | background) x (gate arm), paired per
utterance so the deletion difference can be tested directly.

Reads ONLY the Dawn cache written by `anchor_gate_cache.py dawn` (mix / enh / feat /
ref per utterance x context) and reuses `anchor_gate_sim`'s gate verbatim
(`load_head`, `read_index`, `anchor_track`, `sliding_estimates`, `block_sums`,
`gate_gain`) -- nothing about the gate is re-implemented here.

Arms (all: feature=drr, rule=rel, Wc=1.0, tau_dn=2.0, protect_no_anchor=True):
  ungated   the cached `enh` itself (dry_blend 1.0) -- the baseline
  pna       r = 1e6 dB, so the relative trigger can never fire: the ONLY frames the
            gate opens are the ones before the first anchor arms ("until a >= 1 s
            talker has been heard, do not delete")
  drr3      r = 3 dB  (the README's chosen point, fitted on v8)
  drr6      r = 6 dB  (v16's own frontier)
plus the raw mix, transcribed once per utterance (it does not depend on the context
prefix: the prefix only ever touched the model input).

Output slicing is exactly `eval_dawn_chorus.py --context`: out = g*mix + (1-g)*enh on
the model-input timeline, trimmed to the common mix/enh length, sliced from `offset_s`
for len(ref) samples, zero-padded if the STFT tail came up short.

Usage (from egs/voice_isolate):
  uv run python benchmarks/probes/anchor_gate_asr.py check --cache <dir> --tag v8 --n 6
  uv run python benchmarks/probes/anchor_gate_asr.py run   --cache <dir> --tag v8 \
      --device cuda:0 --out <dir>/asr_v8.jsonl
  uv run python benchmarks/probes/anchor_gate_asr.py analyze --tr <dir>/asr_v8.jsonl \
      --tr <dir>/asr_v16.jsonl --out <dir>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECIPE_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for p in (str(REPO_ROOT), str(RECIPE_DIR / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import anchor_gate_sim as ag  # noqa: E402

HOP = ag.HOP
SR = ag.SR
WC = 1.0

# feature/rule/Wc/tau_dn/pna are fixed; only the margin r changes between arms.
ARMS = {
    "pna": 1e6,
    "drr3": 3.0,
    "drr6": 6.0,
}
ARM_ORDER = ("ungated", "pna", "drr3", "drr6")


def arm_cfg(r: float) -> dict:
    return dict(rule="rel", r=r, Wc=WC, d_far=-1e9, tau_dn=2.0,
                protect_no_anchor=True, feature="drr")


# ------------------------------------------------------------------ per record


def build_record(entry: dict, head):
    """The same Record `anchor_gate_sim.cmd_dawn` builds, plus the raw signals."""
    d = np.load(entry["file"], allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    feat = d["feat"].astype(np.float32)
    F = feat.shape[1]
    mix, enh, ref = d["mix"], d["enh"], d["ref"]
    e_db = ag.frame_energy_db(mix, F)
    act = ag.activity(e_db, float(np.percentile(e_db, 5)) + ag.ACTIVE_MARGIN_DB)
    A, Adrr = ag.anchor_track(feat, act, head)
    C, Cdrr = ag.sliding_estimates(feat, act, head, [WC])[WC]
    Smm, Sme, See, nb, _ = ag.block_sums(mix, enh, F)
    rec = ag.Record(meta=meta, F=F, L=nb * HOP, act=act[:nb], A=A[:nb], Adrr=Adrr[:nb],
                    C={WC: C[:nb]}, Cdrr={WC: Cdrr[:nb]},
                    Smm=Smm, Sme=Sme, See=See,
                    keep_blocks=np.zeros(nb, bool), supp_blocks=np.zeros(nb, bool),
                    exact={})
    return rec, meta, mix, enh, ref


def blend(mix, enh, g, meta, ref, anomalies, tag_id):
    """out = g*mix + (1-g)*enh on the model-input timeline, then the clip slice."""
    L = min(mix.size, enh.size)
    m = mix[:L].astype(np.float64)
    h = enh[:L].astype(np.float64)
    if g is None:
        out = h
    else:
        gs = np.repeat(g, HOP)
        if gs.size < L:
            gs = np.concatenate([gs, np.full(L - gs.size, gs[-1])])
        out = gs[:L] * m + (1.0 - gs[:L]) * h
    off = int(round(meta["offset_s"] * SR))
    n = min(int(ref.size), L - off)
    if n <= 0:
        anomalies.append(f"{tag_id}: no clip region (off={off}, L={L})")
        return None
    if n < ref.size:
        anomalies.append(f"{tag_id}: clip {int(ref.size) - n} samples short (zero-padded)")
    seg = out[off:off + n]
    if n < ref.size:
        seg = np.concatenate([seg, np.zeros(int(ref.size) - n)])
    return seg.astype(np.float32)


def clip_mix(mix, meta, ref, anomalies, tag_id):
    off = int(round(meta["offset_s"] * SR))
    n = min(int(ref.size), mix.size - off)
    if n <= 0:
        anomalies.append(f"{tag_id}: mix has no clip region")
        return None
    if n < ref.size:
        anomalies.append(f"{tag_id}: mix clip {int(ref.size) - n} samples short (zero-padded)")
    seg = mix[off:off + n].astype(np.float64)
    if n < ref.size:
        seg = np.concatenate([seg, np.zeros(int(ref.size) - n)])
    return seg.astype(np.float32)


# ------------------------------------------------------------------ check


def cmd_check(args):
    """pna sanity: the only frames with g > 0 are the ones before the first anchor
    (plus the tau_dn release that follows it, which never rises again)."""
    cache = Path(args.cache)
    head = ag.load_head(cache, args.tag)
    index = ag.read_index(cache, args.tag, "dawn")
    if args.condition:
        index = [e for e in index if e["condition"] == args.condition]
    cfg = arm_cfg(ARMS["pna"])
    print(f"# {args.tag} pna check  cfg={cfg}")
    print(f"{'id':>22} {'cond':>11} {'nb':>5} {'a0':>5} {'g==1 before a0':>15} "
          f"{'max g at/after a0':>18} {'monotone down':>14} {'g>1e-3 frames':>14}")
    bad = 0
    for e in index[: args.n]:
        rec, meta, mix, enh, ref = build_record(e, head)
        g = ag.gate_gain(rec, **cfg)
        nb = g.size
        fin = np.flatnonzero(np.isfinite(rec.Adrr))
        a0 = int(fin[0]) if fin.size else -1
        if a0 < 0:
            pre_ok, mx_after, mono, npos = bool(np.all(g == 1.0)), float("nan"), True, int((g > 1e-3).sum())
        else:
            pre_ok = bool(np.all(g[:a0] == 1.0)) if a0 > 0 else True
            after = g[a0:]
            mx_after = float(after.max()) if after.size else float("nan")
            mono = bool(np.all(np.diff(after) <= 1e-12)) if after.size > 1 else True
            npos = int((after > 1e-3).sum())
        ok = pre_ok and mono
        bad += int(not ok)
        print(f"{e['id']:>22} {e['condition']:>11} {nb:>5} {a0:>5} {str(pre_ok):>15} "
              f"{mx_after:>18.4f} {str(mono):>14} {npos:>14}")
    print(f"violations (g not exactly 1 before the anchor, or g rising after it): {bad}/{min(args.n, len(index))}")


# ------------------------------------------------------------------ run


def cmd_run(args):
    from eval_dawn_chorus import init_asr

    cache = Path(args.cache)
    head = ag.load_head(cache, args.tag)
    index = ag.read_index(cache, args.tag, "dawn")
    if args.limit:
        index = index[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and args.resume:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["context"], r["arm"]))
        print(f"resume: {len(done)} rows already transcribed", flush=True)

    asr_name, transcribe = init_asr("faster-whisper", args.asr_model, args.device)
    print(f"ASR backend: {asr_name} on {args.device}", flush=True)

    anomalies = []
    n_empty = {}
    fh = out_path.open("a" if args.resume else "w", encoding="utf-8")
    by_id = {}
    for e in index:
        by_id.setdefault(e["id"], {})[e["condition"]] = e

    ids = list(by_id)
    for i, uid in enumerate(ids):
        conds = by_id[uid]
        mix_done = False
        for cond in ("none", "background"):
            e = conds.get(cond)
            if e is None:
                anomalies.append(f"{uid}: missing condition {cond}")
                continue
            rec, meta, mix, enh, ref = build_record(e, head)
            ref_txt = meta["transcript"]
            if not mix_done and cond == "none":
                # the raw mix does not depend on the context prefix; transcribe once
                seg = clip_mix(mix, meta, ref, anomalies, f"{uid}:{cond}:mix")
                if seg is not None and (uid, "-", "mix") not in done:
                    hyp = transcribe(seg, SR)
                    n_empty["mix"] = n_empty.get("mix", 0) + int(not hyp.strip())
                    fh.write(json.dumps({"id": uid, "context": "-", "arm": "mix",
                                         "ref": ref_txt, "hyp": hyp},
                                        ensure_ascii=False) + "\n")
                mix_done = True
            for arm in ARM_ORDER:
                if (uid, cond, arm) in done:
                    continue
                g = None if arm == "ungated" else ag.gate_gain(rec, **arm_cfg(ARMS[arm]))
                seg = blend(mix, enh, g, meta, ref, anomalies, f"{uid}:{cond}:{arm}")
                if seg is None:
                    continue
                hyp = transcribe(seg, SR)
                n_empty[arm] = n_empty.get(arm, 0) + int(not hyp.strip())
                fh.write(json.dumps({"id": uid, "context": cond, "arm": arm,
                                     "ref": ref_txt, "hyp": hyp},
                                    ensure_ascii=False) + "\n")
        fh.flush()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ids)} utterances  empty-hyp {n_empty}", flush=True)
    fh.close()
    kinds = {}
    for a in anomalies:
        kinds[a.split(": ", 1)[1]] = kinds.get(a.split(": ", 1)[1], 0) + 1
    Path(str(out_path) + ".anomalies.json").write_text(
        json.dumps({"tag": args.tag, "asr": asr_name, "n_utterances": len(ids),
                    "empty_hyp": n_empty, "n_anomalies": len(anomalies),
                    "anomaly_kinds": kinds, "anomalies": anomalies[:50]}, indent=1))
    print(f"wrote {out_path}  ({len(anomalies)} anomalies, kinds={kinds}, empty-hyp {n_empty})")


# ------------------------------------------------------------------ analyze


def _tf():
    import jiwer
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.ReduceToListOfListOfWords()])


def per_utt(ref: str, hyp: str) -> dict:
    """Same normaliser as `wer_breakdown`, one utterance."""
    import jiwer
    tf = _tf()
    n_ref = len(tf(ref)[0]) if tf(ref) else 0
    if n_ref == 0:
        return {"n": 0, "wer": float("nan"), "del": float("nan"),
                "sub": float("nan"), "ins": float("nan")}
    if not tf(hyp) or len(tf(hyp)[0]) == 0:
        return {"n": n_ref, "wer": 1.0, "del": 1.0, "sub": 0.0, "ins": 0.0}
    o = jiwer.process_words([ref], [hyp], reference_transform=tf, hypothesis_transform=tf)
    return {"n": n_ref, "wer": o.wer, "del": o.deletions / n_ref,
            "sub": o.substitutions / n_ref, "ins": o.insertions / n_ref}


def cmd_analyze(args):
    from eval_dawn_chorus import wer_breakdown
    from scipy.stats import wilcoxon

    out = {"tags": {}}
    for tr in args.tr:
        p = Path(tr)
        tag = p.stem.replace("asr_", "")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pool = {}
        for r in rows:
            pool.setdefault((r["context"], r["arm"]), {})[r["id"]] = r
        agg = {}
        pu = {}
        for key, d in pool.items():
            ids = sorted(d)
            refs = [d[i]["ref"] for i in ids]
            hyps = [d[i]["hyp"] for i in ids]
            agg["|".join(key)] = {**wer_breakdown(refs, hyps), "n_utt": len(ids),
                                  "n_empty_hyp": sum(1 for h in hyps if not h.strip())}
            pu[key] = {i: per_utt(d[i]["ref"], d[i]["hyp"]) for i in ids}
        paired = {}
        for cond in ("none", "background"):
            base = pu.get((cond, "ungated"))
            if base is None:
                continue
            for arm in ("pna", "drr3", "drr6"):
                cur = pu.get((cond, arm))
                if cur is None:
                    continue
                ids = sorted(set(base) & set(cur))
                a = np.array([base[i]["del"] for i in ids], float)
                b = np.array([cur[i]["del"] for i in ids], float)
                ok = np.isfinite(a) & np.isfinite(b)
                d = b[ok] - a[ok]
                nz = d[d != 0]
                pval = float(wilcoxon(nz).pvalue) if nz.size else float("nan")
                paired[f"{cond}|{arm}-ungated"] = {
                    "n": int(ok.sum()), "median_diff": float(np.median(d)),
                    "mean_diff": float(d.mean()),
                    "down": int((d < 0).sum()), "up": int((d > 0).sum()),
                    "unchanged": int((d == 0).sum()), "p": pval}
        bystander = {}
        for arm in ARM_ORDER:
            n_, b_ = pu.get(("none", arm)), pu.get(("background", arm))
            if not n_ or not b_:
                continue
            ids = sorted(set(n_) & set(b_))
            a = np.array([n_[i]["del"] for i in ids], float)
            b = np.array([b_[i]["del"] for i in ids], float)
            ok = np.isfinite(a) & np.isfinite(b)
            d = b[ok] - a[ok]
            nz = d[d != 0]
            bystander[arm] = {
                "n": int(ok.sum()),
                "aggregate_diff": agg[f"background|{arm}"]["deletion_rate"]
                - agg[f"none|{arm}"]["deletion_rate"],
                "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
                "down": int((d < 0).sum()), "up": int((d > 0).sum()),
                "unchanged": int((d == 0).sum()),
                "p": float(wilcoxon(nz).pvalue) if nz.size else float("nan")}
        out["tags"][tag] = {"aggregate": agg, "paired_del_vs_ungated": paired,
                            "bystander_penalty_del": bystander}
    Path(args.out).write_text(json.dumps(out, indent=1))

    for tag, t in out["tags"].items():
        print(f"\n### {tag}")
        print(f"{'arm':>9} | {'none WER':>8} {'sub':>6} {'ins':>6} {'del':>6} | "
              f"{'bg WER':>8} {'sub':>6} {'ins':>6} {'del':>6}")
        m = t["aggregate"].get("-|mix")
        if m:
            print(f"{'raw mix':>9} | {m['wer']:>8.3f} {m['substitution_rate']:>6.3f} "
                  f"{m['insertion_rate']:>6.3f} {m['deletion_rate']:>6.3f} | "
                  f"{'(same)':>8} {'':>6} {'':>6} {'':>6}")
        for arm in ARM_ORDER:
            a = t["aggregate"].get(f"none|{arm}")
            b = t["aggregate"].get(f"background|{arm}")
            if not a or not b:
                continue
            print(f"{arm:>9} | {a['wer']:>8.3f} {a['substitution_rate']:>6.3f} "
                  f"{a['insertion_rate']:>6.3f} {a['deletion_rate']:>6.3f} | "
                  f"{b['wer']:>8.3f} {b['substitution_rate']:>6.3f} "
                  f"{b['insertion_rate']:>6.3f} {b['deletion_rate']:>6.3f}")
        print("paired per-utterance deletion, arm - ungated:")
        for k, v in t["paired_del_vs_ungated"].items():
            print(f"  {k:>26}  median {v['median_diff']:+.3f}  mean {v['mean_diff']:+.3f}  "
                  f"{v['down']}down/{v['up']}up/{v['unchanged']}same  p={v['p']:.3g}  n={v['n']}")
        print("bystander penalty (background - none) deletion:")
        for k, v in t["bystander_penalty_del"].items():
            print(f"  {k:>26}  aggregate {v['aggregate_diff']:+.3f}  median {v['median_diff']:+.3f}  "
                  f"mean {v['mean_diff']:+.3f}  {v['down']}down/{v['up']}up/{v['unchanged']}same  "
                  f"p={v['p']:.3g}  n={v['n']}")
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--condition", default=None)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("run")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--asr-model", default="large-v3")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("analyze")
    s.add_argument("--tr", action="append", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
