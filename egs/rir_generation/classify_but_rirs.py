"""Index a BUT ReverbDB tree by reverb tier and room type.

Emits, without copying any audio:
  * ``by_reverb/<tier>/<room>/{near,far}/`` and ``by_type/<env_type>/<room>/{near,far}/``
    symlink trees for browsing and for picking rooms to build a bank from
  * ``but_rir_index.jsonl`` -- one row per RIR (room, tier, distance, RT20/30/60, path)
  * ``but_rir_rooms.csv``   -- one row per room, sorted by median RT30

The tier is set on median RT30, not RT60: BUT's RT60 column is the inflated
(-5,-65) dB projection, while RT30/RT20 describe the early decay a near/far
decision actually depends on.

Usage:
  uv run python egs/rir_generation/classify_but_rirs.py \
      --but-root exp/BUT_ReverbDB --out exp/BUT_ReverbDB_classified
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
import statistics as st
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent


def tier_of(rt30_med: float) -> str:
    if rt30_med <= 0.75:
        return "office"
    if rt30_med <= 1.5:
        return "high"
    return "extreme"


def parse_meta(mp: Path):
    d = {}
    for line in mp.read_text(errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("$"):
            d[parts[0][1:]] = parts[1].strip()
    mic_id = d.get("EnvMicID")
    dist = rt20 = rt30 = rt60 = None
    if mic_id is not None:
        def g(suf):
            v = d.get(f"EnvMic{mic_id}{suf}")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        dist = g("RelDistance")
        rt20, rt30, rt60 = g("RelRT20"), g("RelRT30"), g("RelRT60")
    return {
        "env_type": d.get("EnvType", "?").strip().title() or "?",
        "env_subtype": d.get("EnvSubType", "?"),
        "volume_m3": float(d["EnvVolume"]) if "EnvVolume" in d else None,
        "mic_id": mic_id,
        "distance_m": dist,
        "rt20": rt20, "rt30": rt30, "rt60": rt60,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--but-root", type=Path, default=RECIPE_DIR / "exp/BUT_ReverbDB",
                    help="extracted BUT ReverbDB root (default: exp/BUT_ReverbDB)")
    ap.add_argument("--out", type=Path, default=RECIPE_DIR / "exp/BUT_ReverbDB_classified",
                    help="output index + symlink trees (re-runs replace the trees)")
    ap.add_argument("--near-far-split-m", type=float, default=1.0,
                    help="distance that separates near from far (default: 1.0)")
    args = ap.parse_args()
    BUT, OUT, D0 = args.but_root.resolve(), args.out.resolve(), args.near_far_split_m
    if not BUT.is_dir():
        raise SystemExit(f"BUT ReverbDB root not found: {BUT}")

    rows = []
    for mp in sorted(BUT.glob("*/MicID*/SpkID*/*/mic_meta.txt")):
        room = mp.relative_to(BUT).parts[0]
        spk_m = re.search(r"(SpkID\d+)", str(mp))
        spk = spk_m.group(1) if spk_m else "SpkID"
        meta = parse_meta(mp)
        rirs = sorted((mp.parent / "RIR").glob("*.wav"))
        for rir in rirs:
            rows.append({
                "room": room, "spk": spk, **meta,
                "rir_path": str(rir.resolve()),
            })

    # per-room aggregation
    rooms = {}
    for r in rows:
        rooms.setdefault(r["room"], []).append(r)

    room_summ = {}
    for room, rs in rooms.items():
        rt30s = [r["rt30"] for r in rs if r["rt30"] is not None]
        rt20s = [r["rt20"] for r in rs if r["rt20"] is not None]
        rt60s = [r["rt60"] for r in rs if r["rt60"] is not None]
        dists = [r["distance_m"] for r in rs if r["distance_m"] is not None]
        rt30_med = st.median(rt30s) if rt30s else float("nan")
        near = sum(1 for d in dists if d < D0)
        far = sum(1 for d in dists if d >= D0)
        room_summ[room] = {
            "env_type": rs[0]["env_type"], "env_subtype": rs[0]["env_subtype"],
            "volume_m3": rs[0]["volume_m3"],
            "n_rir": len(rs),
            "rt20_med": round(st.median(rt20s), 3) if rt20s else None,
            "rt30_med": round(rt30_med, 3) if rt30s else None,
            "rt60_med": round(st.median(rt60s), 3) if rt60s else None,
            "dist_min": round(min(dists), 2) if dists else None,
            "dist_max": round(max(dists), 2) if dists else None,
            "n_near_lt1m": near, "n_far_ge1m": far,
            "tier": tier_of(rt30_med) if rt30s else "?",
        }

    # write outputs (idempotent: wipe prior symlink trees so re-runs don't stack)
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("by_reverb", "by_type"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(exist_ok=True)

    # per-RIR index jsonl
    with open(OUT / "but_rir_index.jsonl", "w") as f:
        for r in rows:
            tier = room_summ[r["room"]]["tier"]
            nf = ("near" if (r["distance_m"] is not None and r["distance_m"] < D0)
                  else "far" if r["distance_m"] is not None else "unknown")
            f.write(json.dumps({
                "room": r["room"], "tier": tier, "env_type": r["env_type"],
                "spk": r["spk"], "mic_id": r["mic_id"],
                "distance_m": round(r["distance_m"], 3) if r["distance_m"] is not None else None,
                "near_far": nf,
                "rt20": r["rt20"], "rt30": r["rt30"], "rt60": r["rt60"],
                "rir_path": r["rir_path"],
            }) + "\n")

    # per-room summary csv
    with open(OUT / "but_rir_rooms.csv", "w", newline="") as f:
        cols = ["room", "tier", "env_type", "env_subtype", "volume_m3", "n_rir",
                "rt20_med", "rt30_med", "rt60_med", "dist_min", "dist_max",
                "n_near_lt1m", "n_far_ge1m"]
        w = csv.writer(f)
        w.writerow(cols)
        for room in sorted(room_summ, key=lambda k: room_summ[k]["rt30_med"] or 9):
            s = room_summ[room]
            w.writerow([room] + [s.get(c) for c in cols[1:]])

    # symlink trees
    n_links = 0
    for r in rows:
        s = room_summ[r["room"]]
        tier = s["tier"]
        nf = ("near" if (r["distance_m"] is not None and r["distance_m"] < D0)
              else "far" if r["distance_m"] is not None else "unknown")
        d = r["distance_m"]
        stem = f"{r['room']}_{r['spk']}_mic{r['mic_id']}_d{d:.2f}m" if d is not None \
            else f"{r['room']}_{r['spk']}_mic{r['mic_id']}"
        name = stem + Path(r["rir_path"]).suffix
        for base in (OUT / "by_reverb" / tier / r["room"] / nf,
                     OUT / "by_type" / r["env_type"] / r["room"] / nf):
            base.mkdir(parents=True, exist_ok=True)
            link = base / name
            # dedup identical stem (multiple RIR versions v00/v01): keep suffixing
            i = 0
            while link.exists() or link.is_symlink():
                i += 1
                link = base / f"{stem}.v{i:02d}{Path(r['rir_path']).suffix}"
            os.symlink(r["rir_path"], link)
            n_links += 1

    print(f"indexed {len(rows)} RIRs across {len(room_summ)} rooms; {n_links} symlinks")
    print("tiers: " + ", ".join(
        f"{t}={sum(1 for s in room_summ.values() if s['tier']==t)} rooms"
        for t in ("office", "high", "extreme")))
    print("output ->", OUT)
    # echo the room table for the summary
    for room in sorted(room_summ, key=lambda k: room_summ[k]["rt30_med"] or 9):
        s = room_summ[room]
        print(f"  [{s['tier']:7}] {room:38} type={s['env_type']:8} "
              f"vol={s['volume_m3']:.0f}m3 RT30={s['rt30_med']} "
              f"n={s['n_rir']} near={s['n_near_lt1m']} far={s['n_far_ge1m']} "
              f"dist={s['dist_min']}-{s['dist_max']}m")


if __name__ == "__main__":
    main()
