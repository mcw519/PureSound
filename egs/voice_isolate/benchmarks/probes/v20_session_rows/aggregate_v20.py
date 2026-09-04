"""Turn `sample_rows_v20.py`'s rows.jsonl into the README's tables.

Same shape as `v19c_diagnostics/training_data_audit/aggregate.py` -- the v16
columns are computed identically so the two runs are comparable -- plus the
session columns. Markdown on stdout.

    uv run python benchmarks/probes/v20_session_rows/aggregate_v20.py \
        --label v16 <scratch>/v16_rows.jsonl \
        --label r1a <scratch>/r1a_rows.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


SHAPE_NAMES = {0.0: "none", 1.0: "user_first", 2.0: "bystander_first",
               3.0: "user_gap", 4.0: "overlap"}


def rowtype(r: dict) -> str:
    if r.get("session_row"):
        return "session"
    if r["target_absent"]:
        return "lone_far(absent)"
    if r["fg_drr"] is None and r["fg_dist"] is not None:
        return "real_near_keep"
    if r["itf_drr"] is None and (r["far_count"] or 0) > 0:
        return "real_far_itf"
    if (r["far_count"] or 0) > 0:
        return "synth_near+far"
    return "synth_near_only"


def nominal(sec: float) -> float:
    for b in (3.0, 6.0, 12.0, 30.0):
        if sec <= b * 1.15:
            return b
    return 30.0


def stats(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"n": int(v.size), "p5": float(np.percentile(v, 5)),
            "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "mean": float(v.mean())}


def frac(vals):
    v = [bool(x) for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def f3(x, digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def med(s, digits=2):
    if s is None:
        return "--"
    return f"{s['median']:.{digits}f} [{s['p10']:.{digits}f},{s['p90']:.{digits}f}]"


def shape_row(rows, det="pipeline") -> dict:
    tkey = "target_pipeline" if det == "pipeline" else "target_db25"
    pre = f"itf_before_onset_s_{det}"
    present = [r for r in rows if not r["target_absent"]]
    onsets = [r[tkey]["onset_s"] for r in present if r[tkey]["onset_s"] is not None]
    ib = [r[pre] for r in present if r[pre] is not None]
    out = {
        "n": len(rows),
        "row_s": stats([r["row_seconds"] for r in rows]),
        "tgt_absent": frac([r["target_absent"] for r in rows]),
        "onset": stats(onsets),
        "onset<=0.5": frac([o <= 0.5 for o in onsets]),
        "onset>=1": frac([o >= 1.0 for o in onsets]),
        "onset>=2": frac([o >= 2.0 for o in onsets]),
        "itf_pre": stats(ib),
        "itf>=1s_pre": frac([x >= 1.0 for x in ib]),
        "itf_any_pre": frac([x > 0.05 for x in ib]),
        "tgt_gap": stats([r[tkey]["longest_gap_s"] for r in present]),
        "tgt_int_gap": stats([r[tkey]["longest_interior_gap_s"] for r in present]),
        "gap>=5s": frac([r[tkey]["longest_gap_s"] >= 5.0 for r in present]),
        "int_gap>=5s": frac([r[tkey]["longest_interior_gap_s"] >= 5.0 for r in present]),
        "reentry>=5s": frac([r[tkey]["reentry_after_5s"] for r in present]),
        "tgt_active": stats([r[tkey]["active_frac"] for r in present]),
        "turn_frac": frac([bool(r["turn_taking"]) for r in rows]),
    }
    return out


HEAD = ("| group | n | row_s | tgt_onset_s | onset<=0.5 | onset>=1s | itf_pre_s | "
        "itf>=1s pre | tgt_gap_s | tgt_int_gap_s | gap>=5s | int_gap>=5s | "
        "reentry>=5s | tgt_active | tgt_absent |")
SEP = "|---" * 15 + "|"


def line(label: str, o: dict) -> str:
    return (
        f"| {label} | {o['n']} | {med(o['row_s'])} | {med(o['onset'])} | "
        f"{f3(o['onset<=0.5'])} | {f3(o['onset>=1'])} | {med(o['itf_pre'])} | "
        f"{f3(o['itf>=1s_pre'])} | {med(o['tgt_gap'])} | {med(o['tgt_int_gap'])} | "
        f"{f3(o['gap>=5s'])} | {f3(o['int_gap>=5s'])} | {f3(o['reentry>=5s'])} | "
        f"{med(o['tgt_active'])} | {f3(o['tgt_absent'], 4)} |"
    )


def session_table(rows) -> str:
    sessions = [r for r in rows if r.get("session_row")]
    if not sessions:
        return "_no session rows in this sample._\n"
    lines = ["| quantity | value |", "|---|---|",
             f"| session rows | {len(sessions)} of {len(rows)} "
             f"({len(sessions)/len(rows):.3f}) |"]
    shares = {}
    for r in sessions:
        name = SHAPE_NAMES.get(r.get("session_shape"), "?")
        shares[name] = shares.get(name, 0) + 1
    for name in ("user_first", "bystander_first", "user_gap", "overlap"):
        count = shares.get(name, 0)
        lines.append(f"| shape {name} | {count} ({count/len(sessions):.3f}) |")
    def add(label, value):
        lines.append(f"| {label} | {value} |")
    add("RIR move share", f3(frac([r["session_rir_move"] for r in sessions])))
    moved = [r["session_move_distance_delta"] for r in sessions
             if r.get("session_rir_move") and r.get("session_move_distance_delta") is not None]
    add("move distance delta, m", med(stats(moved)))
    two = [r["session_move_channels"] for r in sessions
           if r.get("session_rir_move") and r.get("session_move_channels") is not None]
    add("moves that landed on 2 distinct channels",
        f3(frac([v >= 2 for v in two])) + f" (n={len(two)})")
    add("distance-matched bystander share",
        f3(frac([r["session_matched_bystander"] for r in sessions])))
    matched = [r["itf_dist"] for r in sessions if r.get("session_matched_bystander")]
    add("nearest interferer distance on matched rows, m", med(stats(matched)))
    unmatched = [r["itf_dist"] for r in sessions if not r.get("session_matched_bystander")]
    add("nearest interferer distance elsewhere, m", med(stats(unmatched)))
    add("user distance, m", med(stats([r["fg_dist"] for r in sessions])))
    add("turns per row", med(stats([r["session_n_turns"] for r in sessions]), 1))
    add("user turns per row", med(stats([r["session_n_user_turns"] for r in sessions]), 1))
    gaps = [r["session_gap_seconds"] for r in sessions if (r.get("session_gap_seconds") or 0) > 0]
    add("scripted user gap, s (rows with one)", med(stats(gaps)))
    add("rows with a scripted gap >= 5 s",
        f3(frac([(r.get("session_gap_seconds") or 0.0) >= 5.0 for r in sessions])))
    sir = stats([r["session_sir_db"] for r in sessions])
    add("drawn SIR, dB", med(sir))
    add("P(drawn SIR <= -5 dB)",
        f3(frac([(r["session_sir_db"] or 99) <= -5.0 for r in sessions])))
    add("achieved SIR over user-active frames, dB",
        med(stats([r["achieved_sir_user_frames_db"] for r in sessions])))
    add("achieved SIR over double-talk frames, dB",
        med(stats([r.get("achieved_sir_overlap_db") for r in sessions])))
    add("level ratio (user while it talks vs bystander while it talks), dB",
        med(stats([r.get("achieved_level_ratio_db") for r in sessions])))
    add("P(level ratio <= -5 dB)", f3(frac(
        [r["achieved_level_ratio_db"] <= -5.0 for r in sessions
         if r.get("achieved_level_ratio_db") is not None])))
    add("P(double-talk SIR <= -5 dB)", f3(frac(
        [r["achieved_sir_overlap_db"] <= -5.0 for r in sessions
         if r.get("achieved_sir_overlap_db") is not None])))
    add("forced floor level drawn, dBFS", med(stats([r["session_floor_dbfs"] for r in sessions])))
    add("gap frame level, dBFS (median over rows)",
        med(stats([r["gap_floor_median_dbfs"] for r in sessions])))
    add("gap frame level minimum, dBFS", med(stats([r["gap_floor_min_dbfs"] for r in sessions])))
    add("rows whose quiet frames are digital silence",
        f3(frac([r["gap_is_digital_silence"] for r in sessions])))
    add("label user_active == vad_target",
        f3(frac([r["user_label_matches_vad"] for r in sessions])))
    add("rows with a broken turn_id run",
        f3(frac([bool(r.get("turn_ids_broken")) for r in sessions])))
    add("turn_id frame coverage", med(stats([r.get("turn_id_covered_frac") for r in sessions])))
    add("distinct chain ids per row", med(stats([r.get("n_chain_ids") for r in sessions]), 2))
    add("distinct user speaker ids per row", med(stats([r.get("n_user_ids") for r in sessions]), 2))
    add("rows where a bystander id is also the user id",
        f3(frac([r.get("user_is_also_bystander") for r in sessions])))
    paired = [r for r in sessions if (r.get("row_source_id") or -1) >= 0]
    add("paired rows (row_source_id >= 0)",
        f"{len(paired)} ({len(paired)/len(sessions):.3f})")
    per_batch = {r["batch"]: r.get("batch_pairs", 0) for r in rows}
    add("batches with at least one within-batch pair",
        f"{sum(1 for v in per_batch.values() if v > 0)} of {len(per_batch)}")
    add("within-batch pairs, total", f"{sum(per_batch.values())}")
    return "\n".join(lines) + "\n"


def sir_table(rows) -> str:
    """The level relationship, on both configs, by row type.

    ``realized_speech_sir`` is what the recipe drew (NaN on the 'physical' mix
    mode, which applies no rescale); ``level_ratio`` is measured from the
    signals -- the user's level while it talks against the bystander's level
    while it talks -- and is the one comparable across row types.
    """
    lines = ["| group | n | drawn SIR dB | P(drawn<=-5) | level ratio dB | "
             "P(ratio<=-5) | p5 ratio |", "|---|---|---|---|---|---|---|"]
    groups = {"ALL": rows}
    for t in sorted({rowtype(r) for r in rows}):
        groups[f"type={t}"] = [r for r in rows if rowtype(r) == t]
    for label, sub in groups.items():
        drawn = stats([r.get("realized_sir") for r in sub])
        ratio = stats([r.get("achieved_level_ratio_db") for r in sub])
        p_drawn = frac([r["realized_sir"] <= -5.0 for r in sub
                        if r.get("realized_sir") is not None])
        p_ratio = frac([r["achieved_level_ratio_db"] <= -5.0 for r in sub
                        if r.get("achieved_level_ratio_db") is not None])
        p5 = "--" if ratio is None else f"{ratio['p5']:.2f}"
        lines.append(
            f"| {label} | {len(sub)} | {med(drawn)} | {f3(p_drawn)} | "
            f"{med(ratio)} | {f3(p_ratio)} | {p5} |"
        )
    return "\n".join(lines) + "\n"


def identity_table(rows) -> str:
    """Does the same speaker appear in both roles across the sample?"""
    sessions = [r for r in rows if r.get("session_row")]
    if not sessions:
        return ""
    as_user, as_bystander = set(), set()
    for r in sessions:
        if r.get("user_id") is not None and r["user_id"] >= 0:
            as_user.add(int(r["user_id"]))
        for value in r.get("bystander_ids") or []:
            if value >= 0:
                as_bystander.add(int(value))
    both = as_user & as_bystander
    lines = ["| quantity | value |", "|---|---|",
             f"| distinct speakers as U | {len(as_user)} |",
             f"| distinct speakers as B | {len(as_bystander)} |",
             f"| speakers seen in BOTH roles | {len(both)} "
             f"({len(both)/max(1, len(as_user | as_bystander)):.3f} of the pool seen) |"]
    return "\n".join(lines) + "\n"


def batch_table(rows) -> str:
    """The diversity budget: rows, audio-seconds and independent talkers per batch.

    ``spkid`` is the row's foreground speaker and is emitted by every row type,
    so the "distinct foreground speakers" line compares the two configs
    directly. Interferer identities are NOT emitted outside session rows, so the
    talker-draw line counts draws (1 + far_count) rather than distinct ids, and
    the labelled-id line is session-rows-only by construction.

    Only whole batches are counted: a truncated last batch would understate
    both rows and speakers.
    """
    by_batch: dict[int, list] = {}
    for r in rows:
        by_batch.setdefault(r["batch"], []).append(r)
    complete = [v for v in by_batch.values() if len(v) == v[0]["batch_rows"]]
    if not complete:
        complete = list(by_batch.values())
    rows_per_batch, seconds, foreground, draws = [], [], [], []
    labelled, sessions_per_batch, pairs_per_batch = [], [], []
    for value in complete:
        rows_per_batch.append(len(value))
        seconds.append(sum(r["row_seconds"] for r in value))
        foreground.append(len({r["spkid"] for r in value if r.get("spkid") is not None}))
        draws.append(sum(1 + int(r.get("far_count") or 0) for r in value))
        ids = set()
        for r in value:
            if r.get("user_id") is not None and r["user_id"] >= 0:
                ids.add(int(r["user_id"]))
            for b in r.get("bystander_ids") or []:
                if b >= 0:
                    ids.add(int(b))
        labelled.append(len(ids))
        sessions_per_batch.append(sum(1 for r in value if r.get("session_row")))
        pairs_per_batch.append(value[0].get("batch_pairs", 0))
    lines = [f"_{len(complete)} complete batches._", "",
             "| quantity | median [p10,p90] | mean |", "|---|---|---|"]
    for label, s in (("rows per batch", stats(rows_per_batch)),
                     ("audio-seconds per batch", stats(seconds)),
                     ("distinct foreground speakers per batch", stats(foreground)),
                     ("talker draws per batch (1 + far_count)", stats(draws)),
                     ("labelled speaker ids per batch (session rows only)", stats(labelled)),
                     ("session rows per batch", stats(sessions_per_batch)),
                     ("within-batch source pairs", stats(pairs_per_batch))):
        mean = "--" if s is None else f"{s['mean']:.2f}"
        lines.append(f"| {label} | {med(s, 2)} | {mean} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    labels = args.label or [Path(p).stem for p in args.paths]

    for label, path in zip(labels, args.paths):
        rows = [json.loads(line) for line in Path(path).open()]
        for r in rows:
            r["rtype"] = rowtype(r)
        print(f"\n### {label}  (n={len(rows)}, {path})\n")
        print("#### temporal shape, detector = pipeline (vad_target / background_vad_target)\n")
        print(HEAD)
        print(SEP)
        groups = {"ALL": rows}
        for t in sorted({r["rtype"] for r in rows}):
            groups[f"type={t}"] = [r for r in rows if r["rtype"] == t]
        for sec in (3.0, 6.0, 12.0, 30.0):
            sub = [r for r in rows if nominal(r["row_seconds"]) == sec]
            if sub:
                groups[f"bucket={sec:g}s"] = sub
        for name, sub in groups.items():
            if sub:
                print(line(name, shape_row(sub)))
        print("\n#### row-type mix\n")
        print("| row type | n | share |")
        print("|---|---|---|")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["rtype"]] = counts.get(r["rtype"], 0) + 1
        for name in sorted(counts, key=lambda k: -counts[k]):
            print(f"| {name} | {counts[name]} | {counts[name]/len(rows):.4f} |")
        print("\n#### level relationship\n")
        print(sir_table(rows))
        print("\n#### session rows\n")
        print(session_table(rows))
        ident = identity_table(rows)
        if ident:
            print("\n#### speaker roles across the sample\n")
            print(ident)
        print("\n#### per batch\n")
        print(batch_table(rows))


if __name__ == "__main__":
    main()
