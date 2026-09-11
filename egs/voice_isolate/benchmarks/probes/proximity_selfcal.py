"""Is the proximity readout usable across recordings, and what has to be subtracted?

The head is trained on ORDERING WITHIN A ROW: for two turns of one synthetic row it
must read the nearer one higher. Nothing in that objective fixes where a recording's
readout sits, so reading it against one absolute threshold is reading a quantity the
training never defined. This measures the two halves of that separately:

* **within a recording** -- do the hand-labelled near spans read above the far spans
  of the SAME recording, and by how much;
* **across recordings** -- how far the whole distribution moves between them, which
  is what an absolute threshold pays for.

Three readouts are scored on the same frames. `raw` is the head's own output.
`floor` subtracts the recording's own non-speech level, an anchor that exists in every
recording including one with no far talker in it. `quantile` maps the readout onto its
own recording's distribution, which needs both classes present to mean anything.

    uv run python benchmarks/probes/proximity_selfcal.py --ckpt CKPT [--ckpt ...] \
        [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

RECIPE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = RECIPE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from puresound.config import load_recipe  # noqa: E402
from puresound.recipes import init_siso_model  # noqa: E402

SR = 16000
HOP = 160  # the head's frame rate follows the encoder hop
SPANS = RECIPE_DIR / "benchmarks/field_test_vector/spans"
AUDIO = RECIPE_DIR / "test_vec"
NEAR, FAR, BOTH = "near / keep", "far / suppress", "double-talk"


def load_recording(spans_path: Path):
    meta = json.loads(spans_path.read_text())
    wav_path = AUDIO / meta["file"]
    if not wav_path.exists():
        return None
    x, sr = sf.read(wav_path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    if sr != SR:  # the labels carry their own rate; resample to the model's
        import torchaudio

        x = torchaudio.functional.resample(torch.as_tensor(x), sr, SR).numpy()
    return x, meta["spans"]


def frame_labels(spans, n_frames: int) -> np.ndarray:
    """0 unlabelled (gaps, room tone), 1 near, 2 far, 3 double-talk."""
    out = np.zeros(n_frames, dtype=np.int8)
    for span in spans:
        a = int(round(span["start_s"] * SR / HOP))
        b = int(round(span["end_s"] * SR / HOP))
        code = {NEAR: 1, FAR: 2, BOTH: 3}.get(span["tag"], 0)
        out[max(a, 0):min(b, n_frames)] = code
    return out


def readout(model, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        model(torch.as_tensor(x).unsqueeze(0).to(device), dry_blend=0.9)
        return model.backbone.last_proximity.squeeze(0).float().cpu().numpy()


def separability(near: np.ndarray, far: np.ndarray) -> float:
    """P(a near frame reads above a far frame) -- AUC, chance is 0.5."""
    if near.size == 0 or far.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([near, far]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: near.size].sum() - near.size * (near.size + 1) / 2)
                 / (near.size * far.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_curriculum_v1.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recipe = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    per_ckpt = []
    for ckpt in args.ckpt:
        model = init_siso_model(recipe.model)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
        model.eval().to(args.device)
        rows = {}
        for spans_path in sorted(SPANS.glob("*.json")):
            loaded = load_recording(spans_path)
            if loaded is None:
                continue
            x, spans = loaded
            p = readout(model, x, args.device)
            lab = frame_labels(spans, p.size)
            rows[spans_path.name.split(".")[0]] = {
                "near": p[lab == 1], "far": p[lab == 2],
                "both": p[lab == 3], "unlabelled": p[lab == 0],
            }
        per_ckpt.append(rows)

    names = sorted(per_ckpt[0])
    stats = {}
    print("per recording: class medians, each class's mean quantile inside that "
          "recording's own distribution, and the distance from its own floor\n")
    for name in names:
        near = np.concatenate([c[name]["near"] for c in per_ckpt])
        far = np.concatenate([c[name]["far"] for c in per_ckpt]) if per_ckpt[0][name]["far"].size else np.array([])
        floor = np.concatenate([c[name]["unlabelled"] for c in per_ckpt])
        fl = float(np.median(floor)) if floor.size else float("nan")
        auc = separability(near, far)
        # The quantile readout: where the near frames sit inside this recording's
        # own distribution. A self-calibrated rule can only work if they sit high.
        allf = np.concatenate([c[name][k] for c in per_ckpt
                               for k in ("near", "far", "both", "unlabelled") if c[name][k].size])
        q_near = float(np.mean(np.searchsorted(np.sort(allf), near) / allf.size)) if near.size else float("nan")
        q_far = float(np.mean(np.searchsorted(np.sort(allf), far) / allf.size)) if far.size else float("nan")
        stats[name] = {
            "near_quantile": q_near, "far_quantile": q_far,
            "near_median": float(np.median(near)) if near.size else float("nan"),
            "near_p10": float(np.percentile(near, 10)) if near.size else float("nan"),
            "far_median": float(np.median(far)) if far.size else float("nan"),
            "floor_median": fl, "auc": auc,
            "n_near": int(near.size), "n_far": int(far.size),
        }
        s = stats[name]
        print(f"  {name[:26]:28s} near {s['near_median']:6.2f} (q{q_near*100:4.0f}) "
              f"far {s['far_median']:6.2f} (q{q_far*100:4.0f})  floor {fl:6.2f}  "
              f"AUC {auc:5.2f}  near-floor {s['near_median'] - fl:6.2f}")

    # What an absolute threshold has to straddle, against what a floor-referenced
    # one does: the spread of the class medians across recordings is the price.
    near_med = np.array([stats[n]["near_median"] for n in names])
    far_med = np.array([stats[n]["far_median"] for n in names if np.isfinite(stats[n]["far_median"])])
    near_rel = np.array([stats[n]["near_median"] - stats[n]["floor_median"] for n in names])
    far_rel = np.array([stats[n]["far_median"] - stats[n]["floor_median"] for n in names
                        if np.isfinite(stats[n]["far_median"])])
    print(f"\n{'':22s} {'near spread':>22s} {'far spread':>22s} {'separated?':>12s}")
    near_q = np.array([stats[n]["near_quantile"] for n in names if np.isfinite(stats[n]["near_quantile"])])
    far_q = np.array([stats[n]["far_quantile"] for n in names if np.isfinite(stats[n]["far_quantile"])])
    for label, nv, fv in (("raw readout", near_med, far_med),
                          ("minus own floor", near_rel, far_rel),
                          ("own-recording quantile", near_q, far_q)):
        sep = "YES" if np.nanmin(nv) > np.nanmax(fv) else "no (overlap)"
        print(f"  {label:20s} {np.nanmin(nv):7.2f} .. {np.nanmax(nv):7.2f}   "
              f"{np.nanmin(fv):7.2f} .. {np.nanmax(fv):7.2f}   {sep:>12s}")

    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
