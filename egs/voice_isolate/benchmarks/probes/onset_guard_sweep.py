"""Onset guard: decide the four free parameters of the "protect-when-no-anchor" rule by
measurement, with a CAUSAL speech detector.

`anchor_gate_README` §2 found that half the gate's keep-side benefit comes from one clause
that needs no distance readout at all: **until a talker has been heard for >= 1 s of
sustained speech, do not attenuate** (`pna`). §6 turned it into real ASR. This script turns
that clause into a candidate inference option ("onset guard") by (a) replacing the sim's
oracle speech detector -- which reads the record's floor, i.e. the whole file -- with a
causal minimum-statistics tracker, and (b) choosing T_arm / T_forget / tau_dn / F_win on the
FIT set with a stated selection rule instead of by hand.

Nothing here reads the DistHead or the bottleneck: the guard sees only the mix waveform.
The DistHead is loaded only to reproduce the sim's `pna` arm as a reference (`oracle_pna`),
which must land on `anchor_gate_README` §2's numbers.

State per stream (all causal except where noted in `detector`'s report):
  floor      running minimum of the frame energy over the last F_win s, allowed to rise at
             most `rise_db_per_s`; initialised from the first `init_s` of the file.
  active     energy > floor + margin_db, then the sim's 0.2 s hangover / 0.1 s min-run
             cleanup (`anchor_gate_sim.activity`, called on energy-minus-floor so the same
             code path runs against a time-varying threshold).
  confirmed  set once `active` has been continuous for >= T_arm s; cleared again after
             T_forget s with no activity (T_forget = inf reproduces the sim).
  g          target 1 (dry mix) while unconfirmed, 0 (model) once confirmed, through a
             first-order integrator with attack tau_up = 0.05 s and release tau_dn,
             identical in form to `anchor_gate_sim.gate_gain`.
  out        g * mix + (1 - g) * enh.

Selection rule (applied by `report`): on the FIT set, minimise keep violations subject to
keeping >= 90 % of the ungated suppress passes and losing <= 4 dB of the suppress median;
among ties prefer shorter T_arm, finite T_forget, shorter tau_dn.

Usage (from egs/voice_isolate; CPU only, no model calls):
  uv run python benchmarks/probes/onset_guard_sweep.py detector --cache <dir> --tag v8 \
      --which field dawn --out <dir>/detector.json          [--strict]
  uv run python benchmarks/probes/onset_guard_sweep.py field --cache <dir> --tag v8 \
      --split fit --fwin 2 --out <dir>/field_fit_v8.json
  uv run python benchmarks/probes/onset_guard_sweep.py report --sweep <dir>/field_fit_v8.json \
      --detail --out <dir>/report_fit_v8.json --tsv <dir>/report_fit_v8.tsv
  uv run python benchmarks/probes/onset_guard_sweep.py dawn --cache <dir> --tag v8 \
      --fwin 2 --t-arm 1.0 --t-forget 3 --tau-dn 2.0 --out <dir>/dawn_v8.json

The defaults below are the operating point this script measured (F_win 2 s, T_arm 1 s,
T_forget 3 s -- unresolved by the data, see the T_forget line `report` prints -- tau_dn 2 s).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECIPE_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for p in (str(REPO_ROOT), str(RECIPE_DIR / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import anchor_gate_sim as ag        # noqa: E402
import anchor_gate_asr as aga       # noqa: E402

FPS = ag.FPS
HOP = ag.HOP
SR = ag.SR
SNAP = ag.SNAP
MARGIN_DB = ag.ACTIVE_MARGIN_DB     # 8 dB, as the sim
RISE_DB_PER_S = 3.0
INIT_S = 0.2
TAU_UP = 0.05
WC = 1.0                            # the pna reference arm's W_c (anchor_gate_asr)

# the sim's `pna` arm, verbatim: DRR relative rule with a margin that can never fire, so the
# only open frames are the ones before the first anchor arms.
PNA_CFG = dict(rule="rel", r=1e6, Wc=WC, d_far=-1e9, tau_dn=2.0,
               protect_no_anchor=True, feature="drr")


# ------------------------------------------------------------------ causal detector


def running_min(x: np.ndarray, n: int) -> np.ndarray:
    """Causal minimum over the last `n` samples (monotonic deque, O(F))."""
    out = np.empty_like(x)
    dq: deque = deque()
    for i, v in enumerate(x):
        while dq and x[dq[-1]] >= v:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - n:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def causal_floor(e_db: np.ndarray, f_win: float, rise_db_per_s: float = RISE_DB_PER_S,
                 init_s: float = INIT_S) -> np.ndarray:
    """Minimum-statistics floor: the running minimum over the last `f_win` s, but allowed to
    rise at most `rise_db_per_s`; initialised from the first `init_s` of frame energy."""
    F = e_db.size
    n = max(1, int(round(f_win * FPS)))
    wmin = running_min(e_db, n)
    n0 = max(1, min(int(round(init_s * FPS)), F))
    cur = float(e_db[:n0].min())
    rise = rise_db_per_s / FPS
    out = np.empty(F)
    for t in range(F):
        c = wmin[t]
        cur = c if c < cur else min(c, cur + rise)
        out[t] = cur
    return out


def _win_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Causal sum over the last n samples."""
    cs = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
    k = np.arange(1, x.size + 1)
    return cs[k] - cs[np.maximum(0, k - n)]


def causal_activity(e_db: np.ndarray, f_win: float, margin_db: float = MARGIN_DB,
                    strict: bool = False):
    """(active, floor).

    Default: `ag.activity` is called on energy-minus-floor, so the sim's own hangover /
    min-run cleanup runs unchanged against a time-varying threshold. That cleanup is the
    sim's, and it is NOT strictly causal: it bridges a gap only once the next active frame
    has arrived (<= 0.2 s of lookahead) and it deletes a short run after the fact (<= 0.1 s).

    `strict=True`: the streaming form of the same two rules -- hangover becomes a 0.2 s HOLD
    after the last frame over threshold, and min-run becomes a 0.1 s confirmation DELAY
    before activity is declared. No lookahead anywhere.
    """
    fl = causal_floor(e_db, f_win)
    if not strict:
        return ag.activity(e_db - fl, margin_db), fl
    raw = (e_db > fl + margin_db).astype(np.float64)
    held = _win_sum(raw, int(ag.HANGOVER_S * FPS) + 1) > 0
    minr = max(1, int(ag.MIN_RUN_S * FPS))
    act = _win_sum(held.astype(np.float64), minr) >= minr
    return act, fl


def oracle_activity(e_db: np.ndarray, floor_dbfs) -> np.ndarray:
    """Exactly what `ag.load_record` / `ag.cmd_dawn` do."""
    thr = (floor_dbfs if floor_dbfs is not None else float(np.percentile(e_db, 5))) + MARGIN_DB
    return ag.activity(e_db, thr)


# ------------------------------------------------------------------ the guard


def guard_target(act: np.ndarray, t_arm: float, t_forget: float) -> np.ndarray:
    """1 while no anchor is confirmed, 0 once one is. t_forget = inf: never dropped."""
    F = act.size
    n_arm = max(1, int(round(t_arm * FPS)))
    n_forget = None if not np.isfinite(t_forget) else max(1, int(round(t_forget * FPS)))
    tgt = np.ones(F)
    run = sil = 0
    conf = False
    for t in range(F):
        if act[t]:
            run += 1
            sil = 0
            if not conf and run >= n_arm:
                conf = True
        else:
            run = 0
            sil += 1
            if conf and n_forget is not None and sil >= n_forget:
                conf = False
        if conf:
            tgt[t] = 0.0
    return tgt


def integrate(target: np.ndarray, tau_dn: float, tau_up: float = TAU_UP) -> np.ndarray:
    """First-order integrator, the same closed form as `anchor_gate_sim.gate_gain`."""
    nb = target.size
    a_up = 1.0 - math.exp(-1.0 / (tau_up * FPS))
    a_dn = 1.0 - math.exp(-1.0 / (tau_dn * FPS))
    g = np.empty(nb)
    cur = target[0]
    edges = np.flatnonzero(np.diff(target) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [nb]))
    for s, e in zip(starts, ends):
        v = target[s]
        n = e - s
        if abs(cur - v) <= SNAP:
            g[s:e] = v
            cur = v
            continue
        al = a_up if v > cur else a_dn
        k = np.arange(1, n + 1)
        traj = v + (cur - v) * (1.0 - al) ** k
        traj = np.where(np.abs(traj - v) <= SNAP, v, traj)
        g[s:e] = traj
        cur = g[e - 1]
    return g


def guard_gain(act: np.ndarray, t_arm: float, t_forget: float, tau_dn: float) -> np.ndarray:
    return integrate(guard_target(act, t_arm, t_forget), tau_dn)


def cfg_name(c: dict) -> str:
    tf = "inf" if not np.isfinite(c["t_forget"]) else f"{c['t_forget']:g}"
    return f"arm{c['t_arm']:g}/forget{tf}/tau{c['tau_dn']:g}"


def build_configs(t_arms, t_forgets, tau_dns, f_win) -> list:
    return [dict(t_arm=a, t_forget=f, tau_dn=d, f_win=f_win)
            for a in t_arms for f in t_forgets for d in tau_dns]


# ------------------------------------------------------------------ detector validation


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple:
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return prec, rec


def cmd_detector(args):
    cache = Path(args.cache)
    res = {"tag": args.tag, "f_wins": args.fwin, "rise_db_per_s": RISE_DB_PER_S,
           "init_s": INIT_S, "margin_db": MARGIN_DB,
           "strict_causal": bool(args.strict), "sets": {}}
    for which in args.which:
        index = ag.read_index(cache, args.tag, which)
        if which == "dawn" and args.dawn_limit:
            index = index[: args.dawn_limit]
        per = {w: {"agree": [], "prec": [], "rec": [], "n_agree": 0, "n_tot": 0,
                   "act_causal": 0, "act_oracle": 0, "onset_delta_s": []}
               for w in args.fwin}
        anomalies = []
        for i, e in enumerate(index):
            d = np.load(e["file"], allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            F = d["feat"].shape[1]
            mix = d["mix"]
            e_db = ag.frame_energy_db(mix, F)
            floor = meta.get("floor_dbfs") if which == "field" else None
            orc = oracle_activity(e_db, floor)
            if not orc.any():
                anomalies.append(f"{which}:{e.get('clip', e.get('id'))}:"
                                 f"{e.get('variant', e.get('condition'))}: oracle never active")
            for w in args.fwin:
                cau, _ = causal_activity(e_db, w, strict=args.strict)
                agree = float((cau == orc).mean())
                p, r = prf(cau, orc)
                per[w]["agree"].append(agree)
                per[w]["prec"].append(p)
                per[w]["rec"].append(r)
                per[w]["n_agree"] += int((cau == orc).sum())
                per[w]["n_tot"] += int(cau.size)
                per[w]["act_causal"] += int(cau.sum())
                per[w]["act_oracle"] += int(orc.sum())
                io, ic = np.flatnonzero(orc), np.flatnonzero(cau)
                if io.size and ic.size:
                    per[w]["onset_delta_s"].append((int(ic[0]) - int(io[0])) / FPS)
            if args.progress and (i + 1) % 100 == 0:
                print(f"  {which} {i+1}/{len(index)}", flush=True)
        rows = []
        for w in args.fwin:
            d = per[w]
            rows.append({
                "f_win": w, "n_records": len(d["agree"]),
                "frame_agreement_pooled": d["n_agree"] / max(1, d["n_tot"]),
                "frame_agreement_median": float(np.median(d["agree"])),
                "precision_median": float(np.nanmedian(d["prec"])),
                "recall_median": float(np.nanmedian(d["rec"])),
                "precision_mean": float(np.nanmean(d["prec"])),
                "recall_mean": float(np.nanmean(d["rec"])),
                "active_frac_causal": d["act_causal"] / max(1, d["n_tot"]),
                "active_frac_oracle": d["act_oracle"] / max(1, d["n_tot"]),
                "first_onset_delta_s_median": float(np.median(d["onset_delta_s"]))
                if d["onset_delta_s"] else float("nan"),
            })
        res["sets"][which] = {"rows": rows, "n": len(index), "anomalies": anomalies}
        print(f"\n# detector vs oracle, {args.tag} {which} (n={len(index)})")
        print(f"{'F_win':>6} {'frame agree':>12} {'median':>8} {'prec med':>9} {'rec med':>8} "
              f"{'act% causal':>12} {'act% oracle':>12} {'onset delta s':>14}")
        for r in rows:
            print(f"{r['f_win']:>6g} {r['frame_agreement_pooled']:>12.4f} "
                  f"{r['frame_agreement_median']:>8.4f} {r['precision_median']:>9.4f} "
                  f"{r['recall_median']:>8.4f} {100*r['active_frac_causal']:>12.2f} "
                  f"{100*r['active_frac_oracle']:>12.2f} {r['first_onset_delta_s_median']:>14.2f}")
        if anomalies:
            print(f"  anomalies: {len(anomalies)} (first 5: {anomalies[:5]})")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


# ------------------------------------------------------------------ field sweep


def cmd_field(args):
    cache = Path(args.cache)
    head = ag.load_head(cache, args.tag)
    index = ag.read_index(cache, args.tag, "field")
    if args.split == "fit":
        index = [e for e in index if ag._fit(e)]
    elif args.split == "held":
        index = [e for e in index if ag.group_of(e["clip"]) in ag.HELD_OUT_GROUPS]
    if args.configs_json:
        configs = [dict(t_arm=c["t_arm"], t_forget=float(c["t_forget"]),
                        tau_dn=c["tau_dn"], f_win=c["f_win"])
                   for c in json.loads(Path(args.configs_json).read_text())]
    else:
        configs = build_configs(args.t_arm, [float(x) for x in args.t_forget],
                                args.tau_dn, args.fwin)
    print(f"{args.tag} split={args.split}: {len(index)} records x {len(configs)} configs",
          flush=True)
    rows, duty, anomalies = [], [], []
    max_err = 0.0
    for i, e in enumerate(index):
        rec = ag.load_record(e, head, [WC])
        e_db = ag.frame_energy_db(np.load(e["file"], allow_pickle=True)["mix"], rec.F)[: rec.Smm.size]
        gains = {"oracle_pna": ag.gate_gain(rec, **PNA_CFG)}
        acts, targets = {}, {}
        for ci, c in enumerate(configs):
            fw = c["f_win"]
            if fw not in acts:
                acts[fw] = causal_activity(e_db, fw, strict=args.strict)[0]
            tk = (fw, c["t_arm"], c["t_forget"])
            if tk not in targets:
                targets[tk] = guard_target(acts[fw], c["t_arm"], c["t_forget"])
            gains[f"c{ci}"] = integrate(targets[tk], c["tau_dn"])
        base = {"clip": e["clip"], "group": ag.group_of(e["clip"]), "side": e["side"],
                "condition": e["condition"], "variant": e["variant"],
                "held_out": bool(e.get("held_out")), "sentinel": bool(e.get("sentinel"))}
        if not np.isfinite(rec.A).any():
            anomalies.append(f"{args.tag}:{e['clip']}:{e['variant']}: anchor never armed (oracle)")
        drow = {**base, "speech_s": float(rec.act.sum()) / FPS}
        for k, g in gains.items():
            drow[f"duty_{k}"] = float(((g > 0.5) & rec.act).sum()) / FPS
        duty.append(drow)
        for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
            if not blocks.any():
                continue
            mx = ag.mix_level(rec, blocks)
            r0 = {**base, "kind": kind, "mix_dbfs": mx,
                  "ungated": ag.span_level(rec, blocks, None) - mx}
            ex_m = rec.exact.get(f"mix_{kind}")
            if ex_m is not None and np.isfinite(ex_m):
                max_err = max(max_err, abs(ex_m - mx),
                              abs(rec.exact[f"enh_{kind}"] - (r0["ungated"] + mx)))
            for k, g in gains.items():
                r0[k] = ag.span_level(rec, blocks, g) - mx
            rows.append(r0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(index)}", flush=True)
    out = {"tag": args.tag, "split": args.split,
           "configs": [{**c, "name": cfg_name(c)} for c in configs],
           "rows": rows, "duty": duty, "anomalies": anomalies,
           "max_block_vs_exact_db": max_err,
           "detector": {"rise_db_per_s": RISE_DB_PER_S, "init_s": INIT_S,
                        "margin_db": MARGIN_DB, "tau_up": TAU_UP,
                        "strict_causal": bool(args.strict)}}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}; max |exact - block| = {max_err:.4f} dB; "
          f"{len(anomalies)} anomalies")


# ------------------------------------------------------------------ report


FAR_NONE = ("none",)
FAR_COST = ("stream", "near_speech")


def subset(rows, key, *, kind, sides=None, conds=None, sessions=False):
    v = []
    for r in rows:
        if r["kind"] != kind:
            continue
        is_sess = r["side"] == "session"
        if sessions:
            if not is_sess:
                continue
        else:
            if is_sess:
                continue
            if sides and r["side"] not in sides:
                continue
            if conds and r["condition"] not in conds:
                continue
        x = r[key]
        if np.isfinite(x):
            v.append(x)
    return np.array(v, float)


def extra_stats(rows, key) -> dict:
    out = {}
    for name, kw in (("none_far", dict(kind="suppress", sides=("far",), conds=FAR_NONE)),
                     ("cost_far", dict(kind="suppress", sides=("far",), conds=FAR_COST)),
                     ("session_supp", dict(kind="suppress", sessions=True)),
                     ("session_keep", dict(kind="keep", sessions=True))):
        v = subset(rows, key, **kw)
        if v.size == 0:
            out[name] = {"n": 0}
            continue
        if name == "session_keep":
            out[name] = {"n": int(v.size), "median": float(np.median(v)),
                         "viol": int((v < ag.KEEP_VIOLATION_DB).sum())}
        else:
            out[name] = {"n": int(v.size), "median": float(np.median(v)),
                         "pass": int((v <= ag.SUPPRESS_PASS_DB).sum())}
    return out


def duty_stats(duty, key) -> dict:
    if key == "ungated":          # g == 0 everywhere by definition
        v = np.zeros(len(duty))
    else:
        v = np.array([d[f"duty_{key}"] for d in duty], float)
    sp = np.array([d["speech_s"] for d in duty], float)
    frac = np.divide(v, sp, out=np.full(v.shape, np.nan), where=sp > 0)
    return {"median_s": float(np.median(v)), "mean_s": float(v.mean()),
            "p90_s": float(np.percentile(v, 90)),
            "median_frac": float(np.nanmedian(frac))}


def sort_key(cfg, stat):
    return (stat["kv"], cfg["t_arm"], 0 if np.isfinite(cfg["t_forget"]) else 1,
            cfg["t_forget"] if np.isfinite(cfg["t_forget"]) else 1e9, cfg["tau_dn"])


def cmd_report(args):
    data = json.loads(Path(args.sweep).read_text())
    rows, duty, configs = data["rows"], data["duty"], data["configs"]
    ung = ag.objective(rows, "ungated")
    pna = ag.objective(rows, "oracle_pna")
    keep_pass = args.keep_frac * ung["sp"]
    med_floor = ung["smed"] + args.max_med_loss          # dB, less negative = worse

    def line(name, s, extra=""):
        print(f"{name:>26} {str(s['kv'])+'/'+str(s['kn']):>8} {s['kmed']:>7.2f} "
              f"{str(s['sp'])+'/'+str(s['sn']):>7} {s['smed']:>7.2f} {s['smean']:>7.2f}  {extra}")

    print(f"# {data['tag']} split={data['split']}  n_rows={len(rows)}  "
          f"keep-obj={ag.KEEP_OBJ_CONDS}+sessions  supp-obj={ag.SUPP_OBJ_CONDS}+sessions")
    print(f"# constraint: spass >= {keep_pass:.1f} (={args.keep_frac:g} x ungated {ung['sp']}) "
          f"and smed <= {med_floor:.2f} dB")
    print(f"{'arm':>26} {'kviol':>8} {'kmed':>7} {'spass':>7} {'smed':>7} {'smean':>7}")
    line("ungated", ung)
    line("oracle_pna (sim §2)", pna)

    cand = []
    for ci, c in enumerate(configs):
        s = ag.objective(rows, f"c{ci}")
        ok = (s["sp"] >= keep_pass) and (s["smed"] <= med_floor)
        cand.append({"i": ci, "cfg": c, "stat": s, "feasible": bool(ok)})
    feas = [c for c in cand if c["feasible"]]
    feas.sort(key=lambda x: sort_key(x["cfg"], x["stat"]))
    allc = sorted(cand, key=lambda x: sort_key(x["cfg"], x["stat"]))
    print(f"\n## feasible configs: {len(feas)}/{len(cand)}   top {args.top} by "
          f"(keep viol, T_arm, finite T_forget, tau_dn)")
    for c in feas[: args.top]:
        line(c["cfg"]["name"], c["stat"])
    if not feas:
        print("  NONE feasible; top by the same order over all configs:")
        for c in allc[: args.top]:
            line(c["cfg"]["name"], c["stat"], "INFEASIBLE")

    # does T_forget change anything at all on this set?
    cells, split = {}, 0
    for c in cand:
        k = (c["cfg"]["t_arm"], c["cfg"]["tau_dn"])
        s = c["stat"]
        cells.setdefault(k, set()).add((s["kv"], s["sp"], round(s["smed"], 3),
                                        round(s["kmed"], 3)))
    for k, v in cells.items():
        split += int(len(v) > 1)
    print(f"\n## T_forget resolution: {split}/{len(cells)} (T_arm, tau_dn) cells where the "
          f"{len(set(c['cfg']['t_forget'] for c in cand))} T_forget values give different "
          f"(kviol, spass, smed, kmed)")

    chosen = feas[0] if feas else allc[0]
    if args.force_config:
        want = args.force_config
        m = [c for c in cand if c["cfg"]["name"] == want]
        if not m:
            raise SystemExit(f"no config named {want}")
        chosen = m[0]
    print(f"\n## chosen: {chosen['cfg']['name']}  feasible={chosen['feasible']}")
    detail = {"ungated": ("ungated", ung), "oracle_pna": ("oracle_pna", pna),
              chosen["cfg"]["name"]: (f"c{chosen['i']}", chosen["stat"])}
    print(f"{'arm':>26} {'none-far n/pass/med':>26} {'cost-far n/pass/med':>26} "
          f"{'sess supp':>20} {'sess keep':>20} {'duty med s':>11} {'frac':>6}")
    outj = {"tag": data["tag"], "split": data["split"],
            "ungated": ung, "oracle_pna": pna,
            "chosen": {"cfg": chosen["cfg"], "stat": chosen["stat"],
                       "feasible": chosen["feasible"]},
            "top": [{"cfg": c["cfg"], "stat": c["stat"]} for c in feas[: args.top]],
            "all": [{"cfg": c["cfg"], "stat": c["stat"], "feasible": c["feasible"]}
                    for c in allc],
            "extra": {}, "duty": {}}
    for name, (key, s) in detail.items():
        ex = extra_stats(rows, key)
        du = duty_stats(duty, key)
        outj["extra"][name] = ex
        outj["duty"][name] = du
        f = lambda d, hi: (f"{d['n']}/{d.get('pass', d.get('viol'))}/{d['median']:.2f}"
                           if d["n"] else "-")
        print(f"{name:>26} {f(ex['none_far'],0):>26} {f(ex['cost_far'],0):>26} "
              f"{f(ex['session_supp'],0):>20} {f(ex['session_keep'],0):>20} "
              f"{du['median_s']:>11.2f} {du['median_frac']:>6.2f}")
    if args.detail:
        keys = [("ungated", "ungated"), ("oracle_pna", "oracle_pna"),
                (chosen["cfg"]["name"], f"c{chosen['i']}")]
        print(f"\n## per condition x side, {' | '.join(k for k, _ in keys)}")
        print(f"{'condition':>12} {'side':>8} {'kind':>9} {'n':>4}  "
              + "  ".join(f"{'med':>7} {'v/p':>5}" for _ in keys))
        buckets = {}
        for r in rows:
            buckets.setdefault((r["condition"], r["side"], r["kind"]), []).append(r)
        det = {}
        for k in sorted(buckets):
            rs = buckets[k]
            cells_ = []
            for nm, key in keys:
                v = np.array([r[key] for r in rs], float)
                v = v[np.isfinite(v)]
                if k[2] == "keep":
                    c_ = int((v < ag.KEEP_VIOLATION_DB).sum())
                else:
                    c_ = int((v <= ag.SUPPRESS_PASS_DB).sum())
                cells_.append((float(np.median(v)), c_))
                det.setdefault("|".join(k), {})[nm] = {"n": int(v.size),
                                                       "median": float(np.median(v)),
                                                       "viol_or_pass": c_}
            print(f"{k[0]:>12} {k[1]:>8} {k[2]:>9} {len(rs):>4}  "
                  + "  ".join(f"{m:>7.2f} {c_:>5}" for m, c_ in cells_))
        outj["detail"] = det
    if args.also_config:
        print(f"\n## also (v8's choice on this tag): {args.also_config}")
        for c in cand:
            if c["cfg"]["name"] == args.also_config:
                line(c["cfg"]["name"], c["stat"], f"feasible={c['feasible']}")
                ex, du = extra_stats(rows, f"c{c['i']}"), duty_stats(duty, f"c{c['i']}")
                outj["extra"][c["cfg"]["name"]] = ex
                outj["duty"][c["cfg"]["name"]] = du
                outj["also"] = {"cfg": c["cfg"], "stat": c["stat"], "feasible": c["feasible"]}
                print(f"{'':>26} none-far {ex['none_far']} cost-far {ex['cost_far']}")
                print(f"{'':>26} sess supp {ex['session_supp']} sess keep {ex['session_keep']}")
                print(f"{'':>26} duty {du}")
    if args.out:
        Path(args.out).write_text(json.dumps(outj, indent=1))
        print(f"\nwrote {args.out}")
    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("arm\tkviol\tkn\tkmed\tspass\tsn\tsmed\tsmean\tfeasible\n")
            for nm, s, fe in [("ungated", ung, ""), ("oracle_pna", pna, "")] + \
                    [(c["cfg"]["name"], c["stat"], int(c["feasible"])) for c in allc]:
                fh.write(f"{nm}\t{s['kv']}\t{s['kn']}\t{s['kmed']:.3f}\t{s['sp']}\t{s['sn']}\t"
                         f"{s['smed']:.3f}\t{s['smean']:.3f}\t{fe}\n")
        print(f"wrote {args.tsv}")


# ------------------------------------------------------------------ dawn


def cmd_dawn(args):
    cache = Path(args.cache)
    head = ag.load_head(cache, args.tag)
    index = ag.read_index(cache, args.tag, "dawn")
    if args.limit:
        index = index[: args.limit]
    guard = dict(t_arm=args.t_arm, t_forget=float(args.t_forget), tau_dn=args.tau_dn,
                 f_win=args.fwin, strict_causal=bool(args.strict))
    rows, anomalies = [], []
    for i, e in enumerate(index):
        rec, meta, mix, enh, ref = aga.build_record(e, head)
        nb = rec.Smm.size
        e_db = ag.frame_energy_db(mix, rec.F)[:nb]
        cau = causal_activity(e_db, args.fwin, strict=args.strict)[0]
        gains = {"oracle_pna": ag.gate_gain(rec, **PNA_CFG),
                 "guard": guard_gain(cau, guard["t_arm"], guard["t_forget"], guard["tau_dn"])}
        if np.isfinite(guard["t_forget"]):
            # the same guard with the re-arming clause disabled, to see whether T_forget
            # is measurable here at all (it is a no-op on the field set)
            gains["guard_forget_inf"] = guard_gain(cau, guard["t_arm"], float("inf"),
                                                   guard["tau_dn"])
        L = min(mix.size, enh.size)
        m = mix[:L].astype(np.float64)
        h = enh[:L].astype(np.float64)
        off = int(round(meta["offset_s"] * SR))
        n = min(int(ref.size), L - off)
        if n <= 0:
            anomalies.append(f"{e['id']}:{e['condition']}: no clip region")
            continue
        if n < ref.size:
            anomalies.append(f"{e['id']}:{e['condition']}: clip {int(ref.size)-n} samples short")
        R = ref[:n].astype(np.float64)
        nfr = n // HOP
        rf = R[: nfr * HOP].reshape(-1, HOP)
        re_db = 10.0 * np.log10((rf * rf).mean(1) + 1e-12)
        ra = re_db > (re_db.max() - 40.0)
        mb = (m[off:off + nfr * HOP].reshape(-1, HOP) ** 2).sum(1)
        row = {"id": e["id"], "condition": e["condition"],
               "sisdr_mix": ag.si_sdr(m[off:off + n], R),
               "sisdr_enh": ag.si_sdr(h[off:off + n], R)}
        sigs = {"enh": h}
        for name, g in gains.items():
            gs = np.repeat(g, HOP)
            if gs.size < L:
                gs = np.concatenate([gs, np.full(L - gs.size, gs[-1])])
            sigs[name] = gs[:L] * m + (1.0 - gs[:L]) * h
            row[f"sisdr_{name}"] = ag.si_sdr(sigs[name][off:off + n], R)
            o0 = off // HOP
            gg = g[o0:o0 + nfr]
            if gg.size < nfr:      # the STFT tail: pad with the last gain, as the blend does
                gg = np.concatenate([gg, np.full(nfr - gg.size, g[-1])])
            if nfr > 0:
                dn = int(((gg > 0.5) & ra).sum())
                row[f"duty_s_{name}"] = dn / FPS
                row[f"duty_frac_{name}"] = dn / max(1, int(ra.sum()))
            else:
                anomalies.append(f"{e['id']}:{e['condition']}: empty clip region for duty")
                row[f"duty_s_{name}"] = float("nan")
                row[f"duty_frac_{name}"] = float("nan")
        for name, sig in sigs.items():
            sb = (sig[off:off + nfr * HOP].reshape(-1, HOP) ** 2).sum(1)
            for tag_, sel in (("del", ra), ("leak", ~ra)):
                num, den = sb[sel].sum(), mb[sel].sum()
                row[f"{tag_}_{name}"] = (10.0 * math.log10(num / den + 1e-12)
                                         if den > 0 else float("nan"))
        row["ref_active_frac"] = float(ra.mean())
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(index)}", flush=True)
    arms = ["enh"] + [k for k in ("oracle_pna", "guard", "guard_forget_inf")
                      if f"del_{k}" in (rows[0] if rows else {})]
    summ = {}
    for cond in ("none", "background"):
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        f = lambda k: np.array([r[k] for r in rs], float)
        s = {"n": len(rs), "sisdr_mix": float(np.median(f("sisdr_mix")))}
        for a in arms:
            key = "enh" if a == "enh" else a
            s[a] = {"del_median": float(np.nanmedian(f(f"del_{key}"))),
                    "del_mean": float(np.nanmean(f(f"del_{key}"))),
                    "leak_median": float(np.nanmedian(f(f"leak_{key}"))),
                    "leak_mean": float(np.nanmean(f(f"leak_{key}"))),
                    "sisdr_median": float(np.nanmedian(f(f"sisdr_{key}" if a != "enh" else "sisdr_enh")))}
            if a != "enh":
                d = f(f"del_{key}") - f("del_enh")
                sd = f(f"sisdr_{key}") - f("sisdr_enh")
                lk = f(f"leak_{key}") - f("leak_enh")
                s[a].update({"del_paired_median": float(np.nanmedian(d)),
                             "sisdr_paired_median": float(np.nanmedian(sd)),
                             "leak_paired_median": float(np.nanmedian(lk)),
                             "duty_s_median": float(np.nanmedian(f(f"duty_s_{key}"))),
                             "duty_frac_median": float(np.nanmedian(f(f"duty_frac_{key}")))})
        summ[cond] = s
    kinds = {}
    for a in anomalies:
        k = a.split(": ", 1)[1]
        k = k.split(" samples")[0].split()[-1] + " samples short" if "samples short" in k else k
        kinds[k] = kinds.get(k, 0) + 1
    out = {"tag": args.tag, "guard": guard, "pna_cfg": PNA_CFG, "summary": summ,
           "rows": rows, "anomaly_kinds": kinds, "anomalies": anomalies[:50]}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out} ({len(rows)} rows, {len(anomalies)} anomalies, kinds={kinds})")
    for cond, s in summ.items():
        print(f"\n{args.tag} {cond}  n={s['n']}  SI-SDR mix {s['sisdr_mix']:.2f}")
        print(f"{'arm':>12} {'del med':>8} {'del mean':>9} {'leak med':>9} {'leak mean':>10} "
              f"{'SI-SDR':>8} {'d del':>7} {'d leak':>7} {'d sisdr':>8} {'duty s':>7} {'duty f':>7}")
        for a in arms:
            v = s[a]
            print(f"{a:>12} {v['del_median']:>8.2f} {v['del_mean']:>9.2f} "
                  f"{v['leak_median']:>9.2f} {v['leak_mean']:>10.2f} {v['sisdr_median']:>8.2f} "
                  + (f"{v['del_paired_median']:>7.2f} {v['leak_paired_median']:>7.2f} "
                     f"{v['sisdr_paired_median']:>8.2f} {v['duty_s_median']:>7.2f} "
                     f"{v['duty_frac_median']:>7.2f}" if a != "enh" else ""))


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("detector")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--which", nargs="+", default=["field"], choices=["field", "dawn"])
    s.add_argument("--fwin", nargs="+", type=float, default=[2.0, 4.0, 8.0])
    s.add_argument("--dawn-limit", type=int, default=None)
    s.add_argument("--strict", action="store_true",
                   help="strictly causal hangover/min-run (hold + confirmation delay)")
    s.add_argument("--progress", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_detector)

    s = sub.add_parser("field")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--split", default="fit", choices=["fit", "held", "all"])
    s.add_argument("--fwin", type=float, default=2.0)
    s.add_argument("--t-arm", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    s.add_argument("--t-forget", nargs="+", default=["3", "5", "10", "inf"])
    s.add_argument("--tau-dn", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    s.add_argument("--configs-json", default=None)
    s.add_argument("--strict", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_field)

    s = sub.add_parser("report")
    s.add_argument("--sweep", required=True)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--keep-frac", type=float, default=0.9)
    s.add_argument("--max-med-loss", type=float, default=4.0)
    s.add_argument("--force-config", default=None)
    s.add_argument("--also-config", default=None)
    s.add_argument("--detail", action="store_true")
    s.add_argument("--out", default=None)
    s.add_argument("--tsv", default=None)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("dawn")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--fwin", type=float, default=2.0)
    s.add_argument("--t-arm", type=float, default=1.0)
    s.add_argument("--t-forget", default="3")
    s.add_argument("--strict", action="store_true")
    s.add_argument("--tau-dn", type=float, default=2.0)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_dawn)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
