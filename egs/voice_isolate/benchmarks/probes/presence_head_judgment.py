"""Judge a presence-head checkpoint on the v3 field set, against prior caches.

Official protocol (reproduces the v11a ep19 table in
`presence_head_v11_README.md` to +-0.005): per-frame near-head logits, 20 s
segments, audibility = frame dBFS > -60 (fixed, NOT per-clip floor -- on the
quiet rig captures this bites, on the loud QVF captures it passes everything,
and that is what the table on record was measured with). Cold-start truth =
the clip's role (lone far = absent, else present), grouped by recording;
session truth = the hand-labelled keep/suppress spans.

Usage:
  presence_head_judgment.py --ckpt CKPT --cache DIR [--recipe YAML]
                            [--compare name=dir ...]

Extracts per-frame (logit, dBFS, fps) npz per clip into --cache (skipping ones
already there -- caches are derived data, never committed), then prints the
cold-start per-group and session AUC tables for the new cache and every
--compare cache side by side. Run from egs/voice_isolate/.
"""
import argparse, json, pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch

from presence_selfcal_sim import extract, frame_labels, auc

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")


def group_of(clip):
    for tail in ("_near", "_far", "_dt"):
        if tail in clip:
            return clip[: clip.rindex(tail)]
    return clip


def cold_table(cache, windows, clips):
    pools = {}
    for clip in clips:
        if clip.endswith("_session"):
            continue
        z = np.load(cache / f"{clip}.npz")
        lg = z["lg"][z["dbfs"] > -60.0]
        role = 0 if "lone far" in windows[clip]["role"] else 1
        pools.setdefault(group_of(clip), {0: [], 1: []})[role].append(lg)
    return {g: auc(np.concatenate(d[1]), np.concatenate(d[0]))
            for g, d in pools.items() if d[0] and d[1]}


def sess_table(cache, windows, clips):
    out = {}
    for clip in clips:
        if not clip.endswith("_session"):
            continue
        z = np.load(cache / f"{clip}.npz")
        lg, dbfs, fps = z["lg"], z["dbfs"], float(z["fps"])
        y = frame_labels(windows[clip], len(lg), fps)
        m = (y >= 0) & (dbfs > -60.0)
        out[clip] = auc(lg[m & (y == 1)], lg[m & (y == 0)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--recipe", default="config/exp/train_dpcrn_v11b_compinv.yaml",
                    help="recipe whose .model section matches the checkpoint")
    ap.add_argument("--compare", action="append", default=[],
                    metavar="NAME=DIR", help="existing cache to print beside")
    args = ap.parse_args()

    windows = json.loads((CASES / "windows.json").read_text())
    clips = [k for k in windows if not k.startswith("_")]

    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if not (cache / f"{c}.npz").exists()]
    if todo:
        from puresound.config import load_recipe
        from puresound.recipes import init_siso_model
        model = init_siso_model(load_recipe(args.recipe, expected_task="voice_isolation",
                                            expected_purpose="train").model)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"],
                              strict=False)
        model = model.eval()
        for i, clip in enumerate(todo):
            lg, dbfs, fps = extract(clip, model)
            np.savez(cache / f"{clip}.npz", lg=lg, dbfs=dbfs, fps=fps)
            print(f"  extracted {i + 1}/{len(todo)} {clip}", flush=True)

    caches = {}
    for spec in args.compare:
        name, _, d = spec.partition("=")
        caches[name] = pathlib.Path(d)
    caches["THIS RUN"] = cache

    cold = {k: cold_table(v, windows, clips) for k, v in caches.items()}
    sess = {k: sess_table(v, windows, clips) for k, v in caches.items()}
    hdr = " ".join(f"{k:>12s}" for k in caches)
    print(f"\n=== COLD-START per-group AUC (-60 dBFS protocol) ===")
    print(f"{'group':16s} {hdr}")
    for g in sorted(cold["THIS RUN"]):
        print(f"{g:16s} " + " ".join(f"{cold[k].get(g, float('nan')):12.3f}" for k in caches))
    print(f"\n=== SESSIONS keep-vs-suppress AUC ===")
    for s in sorted(sess["THIS RUN"]):
        print(f"{s:24s} " + " ".join(f"{sess[k].get(s, float('nan')):12.3f}" for k in caches))


if __name__ == "__main__":
    main()
