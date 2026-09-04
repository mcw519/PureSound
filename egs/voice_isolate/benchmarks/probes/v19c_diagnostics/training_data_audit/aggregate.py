"""Aggregate rows.jsonl into the tables the audit reports."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
rows = [json.loads(l) for l in path.open()]

def rowtype(r):
    if r["target_absent"]:
        base = "lone_far(absent)"
    elif r["fg_drr"] is None and r["fg_dist"] is not None:
        base = "real_near_keep"
    elif r["itf_drr"] is None and (r["far_count"] or 0) > 0:
        base = "real_far_itf"
    elif (r["far_count"] or 0) > 0:
        base = "synth_near+far"
    else:
        base = "synth_near_only"
    return base

for r in rows:
    r["rtype"] = rowtype(r)

def stats(vals, pcts=(10, 50, 90)):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "mean": float(v.mean())}

def frac(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)

def fmt(s):
    if s is None:
        return "     --   "
    return f"{s['median']:7.2f} [{s['p10']:6.2f},{s['p90']:6.2f}] n={s['n']}"

def nominal(sec):
    # speed perturbation (0.95-1.05) shortens/lengthens the row before the
    # fixed-length snip, so 30 s rows can arrive at 28.57 s
    for b in (3.0, 6.0, 12.0, 30.0):
        if sec <= b * 1.15:
            return b
    return 30.0


def report(subset, label, det="pipeline"):
    tkey = f"target_{det}" if det != "db25" else "target_db25"
    ikey = "itf_pipeline" if det == "pipeline" else "itf_db40"
    akey = "any_speech" if det == "pipeline" else "any_speech_db25"
    pre = f"itf_before_onset_s_{'pipeline' if det=='pipeline' else 'db25'}"
    present = [r for r in subset if not r["target_absent"]]
    out = {}
    out["n_rows"] = len(subset)
    out["n_target_present"] = len(present)
    out["frac_target_absent"] = round(len(subset and [r for r in subset if r['target_absent']]) / max(1, len(subset)), 4)
    out["row_seconds"] = stats([r["row_seconds"] for r in subset])
    out["target_onset_s"] = stats([r[tkey]["onset_s"] for r in present])
    out["target_active_frac"] = stats([r[tkey]["active_frac"] for r in present])
    onsets = [r[tkey]["onset_s"] for r in present if r[tkey]["onset_s"] is not None]
    out["frac_onset_le_0.5s"] = (round(float(np.mean([o <= 0.5 for o in onsets])), 4), len(onsets)) if onsets else None
    out["frac_onset_le_0.1s"] = (round(float(np.mean([o <= 0.1 for o in onsets])), 4), len(onsets)) if onsets else None
    out["frac_onset_ge_1s"] = (round(float(np.mean([o >= 1.0 for o in onsets])), 4), len(onsets)) if onsets else None
    out["itf_before_onset_s"] = stats([r[pre] for r in present])
    ib = [r[pre] for r in present if r[pre] is not None]
    out["frac_itf_ge_1s_before_onset"] = (round(float(np.mean([x >= 1.0 for x in ib])), 4), len(ib)) if ib else None
    out["frac_itf_any_before_onset"] = (round(float(np.mean([x > 0.05 for x in ib])), 4), len(ib)) if ib else None
    out["target_longest_gap_s"] = stats([r[tkey]["longest_gap_s"] for r in present])
    out["target_longest_interior_gap_s"] = stats([r[tkey]["longest_interior_gap_s"] for r in present])
    out["frac_target_reentry_after_5s"] = frac([r[tkey]["reentry_after_5s"] for r in present])
    for th in (2.0, 3.0, 5.0, 10.0):
        out[f"frac_target_gap_ge_{th}s"] = frac([r[tkey]["longest_gap_s"] >= th for r in present])
        out[f"frac_target_interior_gap_ge_{th}s"] = frac(
            [r[tkey]["longest_interior_gap_s"] >= th for r in present])
    gi = [r[tkey]["longest_interior_gap_s"] for r in present]
    out["max_target_interior_gap_s"] = max(gi) if gi else None
    out["frac_onset_ge_2s"] = frac([ (r[tkey]["onset_s"] or 0.0) >= 2.0 for r in present])
    out["frac_row_opens_with_1s_no_speech"] = frac(
        [ (r[akey]["onset_s"] if r[akey]["onset_s"] is not None else 99.0) >= 1.0 for r in subset])
    out["frac_row_opens_with_0.5s_no_speech"] = frac(
        [ (r[akey]["onset_s"] if r[akey]["onset_s"] is not None else 99.0) >= 0.5 for r in subset])
    out["any_speech_longest_gap_s"] = stats([r[akey]["longest_gap_s"] for r in subset])
    out["any_speech_onset_s"] = stats([r[akey]["onset_s"] for r in subset])
    out["frac_anyspeech_gap_ge_2s"] = frac([r[akey]["longest_gap_s"] >= 2.0 for r in subset])
    out["frac_anyspeech_gap_ge_5s"] = frac([r[akey]["longest_gap_s"] >= 5.0 for r in subset])
    out["itf_active_frac"] = stats([r[ikey]["active_frac"] for r in subset])
    out["turn_taking_frac"] = frac([bool(r["turn_taking"]) for r in subset])
    return label, out

groups = {"ALL": rows}
for t in sorted({r["rtype"] for r in rows}):
    groups[f"type={t}"] = [r for r in rows if r["rtype"] == t]
for sec in (3.0, 6.0, 12.0, 30.0):
    groups[f"bucket={sec}s"] = [r for r in rows if nominal(r["row_seconds"]) == sec]
groups["turn_taking=1"] = [r for r in rows if r["turn_taking"]]
groups["turn_taking=0"] = [r for r in rows if not r["turn_taking"]]

res = {}
for det in ("pipeline", "db25"):
    for label, sub in groups.items():
        if not sub:
            continue
        l, o = report(sub, label, det)
        res.setdefault(det, {})[l] = o
print(json.dumps(res, indent=1))
