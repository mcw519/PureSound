"""Cold-start protocol v2 -- the field cold-start instrument that can actually resolve.

WHY THIS EXISTS (2026-08-31, benchmarks/probes/v17_round_design.md §2): scoring one
checkpoint on one zero-context cut of each clip cannot resolve version differences --
across v16's own epochs 8-19 (no knob change) the cold-far median spans -0.35..-3.39 dB
and single clips swing up to 19.9 dB. And a 3 s lead-in of the recording's own room tone
moves the median from -1.21 to -16.11 dB, so "cold start" is two different questions:
the zero-context transient (stream start) and the calibrated steady state (deployment).

So a reading here is: a BLOCK of checkpoints (the run's last five, fixed a priori --
never picked by this score) x context conditions {none, 3 s ambient x3 seeded draws} x
both sides of the clip roster (lone-far suppress AND lone-near keep), reported in
absolute terms (residual over floor, sir_out) with whole-span and steady-state (t>=3 s)
views. `compare` mode pairs two result files per (clip, condition) on block means and
runs a Wilcoxon signed-rank test, which is the only sanctioned way to call a winner.

Score:   uv run python scripts/eval_coldstart_v2.py score config/infer_dpcrn.yaml \
             --ckpt <ckpt> [--ckpt ...] --tag v16_ep19_block --out <json>
Compare: uv run python scripts/eval_coldstart_v2.py compare A.json B.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR / "scripts"))

from puresound.audio.io import AudioIO           # noqa: E402
from puresound.config import load_recipe         # noqa: E402
from puresound.recipes import init_siso_model    # noqa: E402
from eval_realcase import span_dbfs              # noqa: E402

CASES = RECIPE_DIR / "data_report/field_cases/test_vector_cases"
PAD_SECONDS = 3.0
AMBIENT_DRAWS = 3
STEADY_FROM = 3.0
SR = 16000


def group_of(clip: str) -> str:
    for tail in ("_far", "_near", "_dt", "_session"):
        if tail in clip:
            return clip[: clip.rindex(tail)]
    return clip


def load_wav(clip: str) -> torch.Tensor:
    wav, _ = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"), target_lvl=None, resample_to=SR)
    return wav.view(1, -1)


def ambient_pads(windows: dict, clip: str) -> list:
    """3 s room-tone segments for this clip's recording: from the session clip of the
    same group when one exists, else from the clip's own file -- always OUTSIDE every
    annotated span. Deterministic per (clip, draw): provenance beats variety here."""
    group = group_of(clip)
    sessions = {group_of(c): c for c in windows if c.endswith("_session")}
    source = sessions.get(group, clip)
    spec = windows[source]
    wav = load_wav(source).view(-1)
    occupied = (spec.get("keep") or []) + (spec.get("suppress") or [])
    pad_n = int(PAD_SECONDS * SR)

    def free(t0: float) -> bool:
        return not any(a - 0.2 <= t <= b + 0.2 for a, b in occupied
                       for t in (t0, t0 + PAD_SECONDS / 2, t0 + PAD_SECONDS))

    seed = int(hashlib.sha1(clip.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    pads, tries = [], 0
    horizon = wav.numel() / SR - PAD_SECONDS - 0.1
    while len(pads) < AMBIENT_DRAWS and tries < 5000 and horizon > 0:
        tries += 1
        t0 = float(rng.uniform(0, horizon))
        if free(t0):
            a = int(t0 * SR)
            pads.append(wav[a : a + pad_n].clone())
    return pads


def score_ckpt(model, windows: dict, device: torch.device) -> list:
    rows = []
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        role = spec.get("role", "")
        side = "far" if "lone far" in role else "near" if "lone near" in role else None
        if side is None:
            continue
        spans = spec["suppress"] if side == "far" else spec["keep"]
        wav = load_wav(clip)
        floor = spec.get("floor_dbfs")
        near_ref = spec.get("near_ref_dbfs", float("nan"))

        conditions = [("none", None)]
        for i, pad in enumerate(ambient_pads(windows, clip)):
            conditions.append((f"ambient{i}", pad))

        for cond, pad in conditions:
            x = torch.cat([pad.view(1, -1), wav], dim=-1) if pad is not None else wav
            off = PAD_SECONDS if pad is not None else 0.0
            with torch.no_grad():
                out = model(x.to(device)).detach().cpu().view(1, -1)
            n = min(out.shape[-1], x.shape[-1])
            sh = [[a + off, b + off] for a, b in spans]
            sh_steady = [[max(a, off + STEADY_FROM), b] for a, b in sh]
            sh_steady = [s for s in sh_steady if s[1] - s[0] > 0.2]
            in_db = span_dbfs(x, sh, SR, n)
            out_db = span_dbfs(out, sh, SR, n)
            row = {
                "clip": clip, "side": side, "condition": "ambient" if cond.startswith("ambient") else cond,
                "draw": cond, "delta_db": out_db - in_db,
                "residual_db": (out_db - floor) if (side == "far" and floor is not None) else None,
                "sir_out_db": (near_ref - out_db) if (side == "far" and near_ref == near_ref) else None,
            }
            if sh_steady:
                row["delta_steady_db"] = span_dbfs(out, sh_steady, SR, n) - span_dbfs(x, sh_steady, SR, n)
            rows.append(row)
    return rows


def summarize(records: dict) -> None:
    rows = records["rows"]
    def sel(side, cond):
        per_clip = {}
        for r in rows:
            if r["side"] == side and r["condition"] == cond:
                per_clip.setdefault(r["clip"], []).append(r["delta_db"])
        return {c: float(np.mean(v)) for c, v in per_clip.items()}   # mean over ckpts+draws
    print(f"\n== {records['tag']}  (block of {len(records['ckpts'])} ckpts, "
          f"{AMBIENT_DRAWS} ambient draws, pad {PAD_SECONDS:.0f} s) ==")
    for side, bar in (("far", -6.0), ("near", -3.0)):
        for cond in ("none", "ambient"):
            m = sel(side, cond)
            if not m:
                continue
            v = np.array(list(m.values()))
            hits = int((v <= bar).sum())
            label = "suppression" if side == "far" else "preservation"
            extra = f"  <= {bar:.0f} dB: {hits}/{len(v)}" if side == "far" else \
                    f"  violations(< {bar:.0f}): {hits}/{len(v)}  worst {v.min():+.2f}"
            print(f"  {side:4s} {cond:8s} {label:12s} block-median {np.median(v):+7.2f}{extra}")


def cmd_score(args) -> None:
    windows = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items()
               if not k.startswith("_")}
    device = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config_path, expected_task="voice_isolation").model)
    all_rows = []
    for ck in args.ckpt:
        state = torch.load(ck, map_location="cpu")
        model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
        model = model.to(device).eval()
        rows = score_ckpt(model, windows, device)
        for r in rows:
            r["ckpt"] = Path(ck).name
        all_rows.extend(rows)
        print(f"  scored {Path(ck).name}: {len(rows)} rows", flush=True)
    records = {"tag": args.tag, "ckpts": [str(c) for c in args.ckpt],
               "pad_seconds": PAD_SECONDS, "rows": all_rows}
    Path(args.out).write_text(json.dumps(records, indent=1))
    summarize(records)
    print(f"\nwrote {args.out}")


def cmd_compare(args) -> None:
    from scipy.stats import wilcoxon
    a, b = (json.loads(Path(p).read_text()) for p in (args.results[0], args.results[1]))
    def block_means(rec):
        acc = {}
        for r in rec["rows"]:
            acc.setdefault((r["clip"], r["side"], r["condition"]), []).append(r["delta_db"])
        return {k: float(np.mean(v)) for k, v in acc.items()}
    ma, mb = block_means(a), block_means(b)
    print(f"compare {a['tag']}  vs  {b['tag']}  (paired per clip, block means)")
    for side in ("far", "near"):
        for cond in ("none", "ambient"):
            keys = sorted(k for k in ma if k in mb and k[1] == side and k[2] == cond)
            if len(keys) < 6:
                continue
            xa = np.array([ma[k] for k in keys]); xb = np.array([mb[k] for k in keys])
            d = xa - xb
            try:
                stat = wilcoxon(xa, xb)
                p = stat.pvalue
            except ValueError:
                p = float("nan")
            print(f"  {side:4s} {cond:8s} n={len(keys):2d}  medians {np.median(xa):+7.2f} vs "
                  f"{np.median(xb):+7.2f}  paired-delta median {np.median(d):+6.2f}  Wilcoxon p={p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("config_path")
    sc.add_argument("--ckpt", action="append", required=True)
    sc.add_argument("--tag", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--device", default="cuda")
    sc.set_defaults(func=cmd_score)
    cp = sub.add_parser("compare")
    cp.add_argument("results", nargs=2)
    cp.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
