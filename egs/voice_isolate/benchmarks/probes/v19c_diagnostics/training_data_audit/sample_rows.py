"""Sample training rows through the REAL v16 pipeline and measure onset/anchor stats.

Writes one JSON object per row to rows.jsonl. Read-only w.r.t. the repo.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

RECIPE_DIR = Path("/home/milowu/A4Audio/PureSound/egs/voice_isolate")
REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa
from puresound.config import load_recipe, with_overrides  # noqa

SR = 16000
HOP = 160
FRAME = 400
FPS = SR / HOP  # 100


def frames_energy_db(wav: torch.Tensor) -> np.ndarray:
    """Per-frame power in dB relative to the row's own peak frame (same grid as
    the pipeline's EnergyVADLabeler: frame 400, hop 160)."""
    x = wav.reshape(1, -1)
    if x.shape[-1] < FRAME:
        x = torch.nn.functional.pad(x, (0, FRAME - x.shape[-1]))
    fr = x.unfold(-1, FRAME, HOP)
    p = fr.square().mean(dim=-1).squeeze(0)
    ref = p.max().clamp_min(1e-20)
    return (10.0 * torch.log10(p.clamp_min(1e-30) / ref)).numpy()


def runs_of_zero(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] runs where mask == 0."""
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == 0:
            j = i
            while j < n and mask[j] == 0:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def describe(mask: np.ndarray) -> dict:
    """mask: 0/1 frame activity."""
    act = np.flatnonzero(mask > 0)
    d = {"active_frac": float(mask.mean()) if len(mask) else float("nan")}
    if act.size == 0:
        d.update(onset_s=None, offset_s=None, longest_gap_s=float(len(mask) / FPS),
                 longest_interior_gap_s=0.0, reentry_after_5s=False, n_spans=0)
        return d
    zr = runs_of_zero(mask)
    longest = max((e - s for s, e in zr), default=0)
    interior = [e - s for s, e in zr if s > 0 and e < len(mask)]
    longest_int = max(interior, default=0)
    # spans of activity
    diff = np.diff(np.concatenate(([0], (mask > 0).astype(int), [0])))
    n_spans = int((diff == 1).sum())
    d.update(
        onset_s=float(act[0] / FPS),
        offset_s=float(act[-1] / FPS),
        longest_gap_s=float(longest / FPS),
        longest_interior_gap_s=float(longest_int / FPS),
        reentry_after_5s=bool(longest_int >= 5 * FPS),
        n_spans=n_spans,
    )
    return d


def sc(batch, key, r):
    v = batch.get(key)
    if v is None:
        return None
    x = v.reshape(-1)[r]
    f = float(x)
    return f if np.isfinite(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v16_lengthmix.yaml"))
    ap.add_argument("--n-rows", type=int, default=420)
    ap.add_argument("--max-batches", type=int, default=400)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(RECIPE_DIR)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)

    recipe = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    recipe = with_overrides(recipe, trainer={"num_workers": args.num_workers})
    train_dl, _ = recipe_main.init_dataloader(recipe)

    fh = open(args.out, "w", encoding="utf-8")
    n = 0
    t0 = time.time()
    for bi, batch in enumerate(train_dl):
        if bi >= args.max_batches or n >= args.n_rows:
            break
        clean = batch["clean_speech"]
        noisy = batch["noisy_speech"]
        far = batch.get("far_target")
        vad = batch.get("vad_target")
        bvad = batch.get("background_vad_target")
        B = clean.shape[0]
        for r in range(B):
            L = int(batch["length"].reshape(-1)[r].item())
            c = clean[r, :L]
            absent = bool(c.abs().amax().item() == 0.0)
            row = {
                "batch": bi, "row": r, "batch_rows": B,
                "row_seconds": round(L / SR, 4),
                "target_absent": absent,
                "scalar_target_absent": sc(batch, "target_absent", r),
                "scalar_target_present": sc(batch, "target_present", r),
                "turn_taking": sc(batch, "turn_taking", r),
                "mix_mode": sc(batch, "mix_mode", r),
                "far_count": sc(batch, "far_count", r),
                "overlap_fraction": sc(batch, "overlap_fraction", r),
                "has_background_speech": sc(batch, "has_background_speech", r),
                "fg_drr": sc(batch, "foreground_drr", r),
                "fg_dist": sc(batch, "foreground_distance", r),
                "itf_drr": sc(batch, "strongest_interferer_drr", r),
                "itf_dist": sc(batch, "nearest_interferer_distance", r),
                "rt60": sc(batch, "rt60", r),
                "realized_sir": sc(batch, "realized_speech_sir", r),
                "noise_snr": sc(batch, "noise_snr", r),
            }
            # --- target activity, three detectors -----------------------
            if absent:
                tgt40 = np.zeros(int(L / HOP) or 1)
                tgt30 = tgt40
                tgt_pipeline = tgt40
                row["target_floor_db"] = None
            else:
                db = frames_energy_db(c)
                row["target_floor_db"] = float(np.percentile(db, 5))
                tgt40 = (db > -40.0).astype(float)   # == pipeline EnergyVADLabeler
                tgt30 = (db > -25.0).astype(float)   # stricter
                tgt_pipeline = tgt40
            if vad is not None:
                vp = vad[r].numpy()
                vp = vp[: max(1, int(L / HOP))]
                tgt_pipeline = (vp > 0.5).astype(float)
            # --- interferer activity ------------------------------------
            if far is not None:
                f = far[r, : min(L, far.shape[-1])]
                if f.abs().amax().item() == 0.0:
                    itf = np.zeros(int(L / HOP) or 1)
                    row["itf_floor_db"] = None
                else:
                    dbf = frames_energy_db(f)
                    row["itf_floor_db"] = float(np.percentile(dbf, 5))
                    itf = (dbf > -40.0).astype(float)
            else:
                itf = np.zeros(int(L / HOP) or 1)
                row["itf_floor_db"] = None
            if bvad is not None:
                bp = bvad[r].numpy()[: max(1, int(L / HOP))]
                itf_pipeline = (bp > 0.5).astype(float)
            else:
                itf_pipeline = itf

            m = min(len(tgt_pipeline), len(itf_pipeline), len(tgt40), len(tgt30), len(itf))
            tgt_pipeline, itf_pipeline = tgt_pipeline[:m], itf_pipeline[:m]
            tgt40, tgt30, itf = tgt40[:m], tgt30[:m], itf[:m]

            row["n_frames"] = int(m)
            row["target_pipeline"] = describe(tgt_pipeline)
            row["target_db40"] = describe(tgt40)
            row["target_db25"] = describe(tgt30)
            row["itf_pipeline"] = describe(itf_pipeline)
            row["itf_db40"] = describe(itf)

            # interferer active BEFORE the first target onset
            for tag, tm, im in (("pipeline", tgt_pipeline, itf_pipeline),
                                ("db25", tgt30, itf)):
                onset = row[f"target_{'pipeline' if tag=='pipeline' else 'db25'}"]["onset_s"]
                if onset is None:
                    pre = float(im.sum() / FPS)
                    row[f"itf_before_onset_s_{tag}"] = pre
                    row[f"itf_before_onset_frac_{tag}"] = float(im.mean()) if m else None
                else:
                    k = int(round(onset * FPS))
                    row[f"itf_before_onset_s_{tag}"] = float(im[:k].sum() / FPS)
                    row[f"itf_before_onset_frac_{tag}"] = float(im[:k].mean()) if k > 0 else 0.0
            # any-speech gap
            any_sp = np.maximum(tgt_pipeline, itf_pipeline)
            row["any_speech"] = describe(any_sp)
            any25 = np.maximum(tgt30, itf)
            row["any_speech_db25"] = describe(any25)
            fh.write(json.dumps(row) + "\n")
            n += 1
        fh.flush()
        if bi % 5 == 0:
            print(f"batch {bi} rows={n} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"DONE rows={n} batches={bi+1} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
