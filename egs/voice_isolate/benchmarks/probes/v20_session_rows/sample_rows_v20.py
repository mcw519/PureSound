"""Sample training rows through a REAL recipe and measure their temporal shape.

The method is `v19c_diagnostics/training_data_audit/sample_rows.py` -- copied
rather than imported, so the v16 numbers already on the record stay reproducible
from that file while this one grows the columns v20 needs:

* the session scalars the row type emits (shape, turns, gap, RIR move,
  distance-matched bystander, drawn SIR, forced floor level, `row_source_id`);
* the per-frame label tensors (`user_active`, `bystander_active`, `turn_id`),
  which are what the losses will actually read -- so the audit measures the
  labels, not a reconstruction of them;
* the achieved SIR over the frames where the user is speaking, because the
  drawn SIR is applied by `add_bg_noise` over the whole row and a row where the
  user talks 40% of the time does not deliver it as drawn;
* the noise floor inside the user's gap, which is the "never digital silence"
  contract stated as a number;
* the within-batch `row_source_id` collisions, i.e. how many cross-chain pairs
  actually land in one optimiser step.

Read-only with respect to the repo. Writes one JSON object per row to
`--out`; `aggregate_v20.py` turns that into the README's tables.

    cd egs/voice_isolate
    uv run python benchmarks/probes/v20_session_rows/sample_rows_v20.py \
        --config config/exp/train_dpcrn_v20_r1a.yaml --n-rows 600 --seed 7 \
        --out <scratch>/r1a_rows.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

RECIPE_DIR = Path("/home/milowu/A4Audio/PureSound/egs/voice_isolate")
REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
sys.path.insert(0, str(REPO_ROOT))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.config import load_recipe, with_overrides  # noqa: E402

SR = 16000
HOP = 160
FRAME = 400
FPS = SR / HOP  # 100

#: Session scalars, read straight off the batch when the recipe emits them.
SESSION_SCALARS = (
    "session_row",
    "session_shape",
    "session_n_turns",
    "session_n_user_turns",
    "session_rir_move",
    "session_move_distance_delta",
    "session_move_channels",
    "session_matched_bystander",
    "session_gap_seconds",
    "session_sir_db",
    "session_floor_dbfs",
)


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


def frames_power(wav: torch.Tensor) -> np.ndarray:
    """Per-frame power, absolute (not row-relative). Same grid."""
    x = wav.reshape(1, -1)
    if x.shape[-1] < FRAME:
        x = torch.nn.functional.pad(x, (0, FRAME - x.shape[-1]))
    fr = x.unfold(-1, FRAME, HOP)
    return fr.square().mean(dim=-1).squeeze(0).numpy()


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


def turn_stats(batch, r, n_frames: int) -> dict:
    """What the per-turn label vectors say about this row."""
    out: dict = {}
    role = batch.get("turn_role")
    speaker = batch.get("turn_speaker")
    chain = batch.get("turn_chain")
    if role is None:
        return out
    roles = role[r].numpy()
    speakers = speaker[r].numpy() if speaker is not None else np.zeros_like(roles)
    chains = chain[r].numpy() if chain is not None else np.zeros_like(roles)
    live = roles > 0
    out["n_turns_label"] = int(live.sum())
    out["n_user_turns_label"] = int((roles == 1).sum())
    out["n_bystander_turns_label"] = int((roles == 2).sum())
    user_ids = sorted({int(s) for role_, s in zip(roles, speakers) if role_ == 1})
    bys_ids = sorted({int(s) for role_, s in zip(roles, speakers) if role_ == 2})
    out["n_user_ids"] = len(user_ids)
    out["n_bystander_ids"] = len(bys_ids)
    out["user_id"] = user_ids[0] if user_ids else None
    out["bystander_ids"] = bys_ids
    out["user_is_also_bystander"] = bool(set(user_ids) & set(bys_ids))
    out["n_chain_ids"] = len({int(c) for role_, c in zip(roles, chains) if role_ > 0})
    turn_id = batch.get("turn_id")
    if turn_id is not None:
        ids = turn_id[r].numpy()[:n_frames]
        out["turn_id_max"] = int(ids.max()) if ids.size else 0
        out["turn_id_covered_frac"] = float((ids > 0).mean()) if ids.size else None
        # contiguity: every id must occupy exactly one run
        broken = 0
        for k in range(1, int(out["turn_id_max"]) + 1):
            runs = runs_of_zero((ids != k).astype(int))
            if len([1 for s, e in runs if e > s]) > 1:
                broken += 1
        out["turn_ids_broken"] = int(broken)
    return out


def achieved_sir_db(target: torch.Tensor, far: torch.Tensor, user_mask: np.ndarray,
                    bys_mask: np.ndarray, n_frames: int) -> dict:
    """Three ways to read the level relationship a row actually delivered.

    The drawn SIR is applied by ``add_bg_noise`` against the WHOLE row's RMS,
    and on a turn-taking row neither talker occupies the whole row -- so the
    drawn number is not what the model hears anywhere in particular:

    * ``user_frames`` -- SIR over the frames the user is active in. On a
      turn-taking row this is high by construction (the bystander is mostly
      quiet while the user talks) and says little about difficulty.
    * ``overlap`` -- SIR over the double-talk frames. This is where a loud
      bystander is actually a keep/delete problem.
    * ``level_ratio`` -- the user's level while IT talks against the
      bystander's level while IT talks. This is the turn-taking analogue of
      "the bystander is louder than the user", and it is the number that should
      be read against the drawn SIR.
    """
    tp = frames_power(target)[:n_frames]
    fp = frames_power(far)[:n_frames]
    m = min(len(tp), len(fp), len(user_mask), len(bys_mask), n_frames)
    tp, fp = tp[:m], fp[:m]
    user = user_mask[:m] > 0
    bys = bys_mask[:m] > 0
    out: dict = {"user_frames": None, "overlap": None, "level_ratio": None}

    def ratio(num, den):
        if num <= 0 or den <= 0:
            return None
        return float(10.0 * np.log10(num / den))

    if user.sum():
        out["user_frames"] = ratio(float(tp[user].mean()), float(fp[user].mean()))
    both = user & bys
    if both.sum():
        out["overlap"] = ratio(float(tp[both].mean()), float(fp[both].mean()))
    if user.sum() and bys.sum():
        out["level_ratio"] = ratio(float(tp[user].mean()), float(fp[bys].mean()))
    return out


def gap_floor_dbfs(noisy: torch.Tensor, user_mask: np.ndarray, bys_mask: np.ndarray,
                   n_frames: int) -> tuple[float | None, float | None]:
    """(median frame level, minimum frame level) in dBFS where nobody talks.

    ``-inf`` would be digital silence; that is the thing the row type promises
    never to deliver.
    """
    p = frames_power(noisy)[:n_frames]
    m = min(len(p), len(user_mask), len(bys_mask), n_frames)
    quiet = (user_mask[:m] == 0) & (bys_mask[:m] == 0)
    if quiet.sum() == 0:
        return None, None
    vals = p[:m][quiet]
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(vals, 1e-300))
    return float(np.median(db)), float(db.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v20_r1a.yaml"))
    ap.add_argument("--n-rows", type=int, default=600)
    ap.add_argument("--max-batches", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only-bucket", type=float, default=None,
                    help="force every batch to this row length (session-focused pass)")
    ap.add_argument("--only-bucket-n-spk", type=int, default=2)
    args = ap.parse_args()

    os.chdir(RECIPE_DIR)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)

    recipe = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    recipe = with_overrides(recipe, trainer={"num_workers": args.num_workers})
    if args.only_bucket is not None:
        recipe = with_overrides(recipe, trainer={"length_schedule": [
            {"seconds": args.only_bucket, "n_spk": args.only_bucket_n_spk, "prob": 1.0}
        ]})
    train_dl, _ = recipe_main.init_dataloader(recipe)

    fh = open(args.out, "w", encoding="utf-8")
    n = 0
    bi = -1
    t0 = time.time()
    for bi, batch in enumerate(train_dl):
        if bi >= args.max_batches or n >= args.n_rows:
            break
        clean = batch["clean_speech"]
        noisy = batch["noisy_speech"]
        far = batch.get("far_target")
        vad = batch.get("vad_target")
        bvad = batch.get("background_vad_target")
        ua = batch.get("user_active")
        ba = batch.get("bystander_active")
        rsid = batch.get("row_source_id")
        B = clean.shape[0]
        # within-batch cross-chain pairs: how many rows share a source id
        pair_counts: dict[int, int] = {}
        if rsid is not None:
            for value in rsid.reshape(-1).tolist():
                if int(value) >= 0:
                    pair_counts[int(value)] = pair_counts.get(int(value), 0) + 1
        batch_pairs = sum(count // 2 for count in pair_counts.values())
        batch_paired_rows = sum(count for count in pair_counts.values() if count >= 2)
        for r in range(B):
            L = int(batch["length"].reshape(-1)[r].item())
            c = clean[r, :L]
            absent = bool(c.abs().amax().item() == 0.0)
            row = {
                "batch": bi, "row": r, "batch_rows": B,
                "row_seconds": round(L / SR, 4),
                # The row's own foreground speaker index -- emitted by every row
                # type, so speakers/batch is comparable between the two configs
                # (a session row overwrites it with the user it rendered).
                "spkid": int(batch["spkid"].reshape(-1)[r].item()),
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
                "batch_pairs": batch_pairs,
                "batch_paired_rows": batch_paired_rows,
            }
            for key in SESSION_SCALARS:
                row[key] = sc(batch, key, r)
            row["row_source_id"] = (
                int(rsid.reshape(-1)[r].item()) if rsid is not None else None
            )
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

            # ---- v20 additions: the emitted labels, read as labels ------
            if ua is not None:
                ua_r = (ua[r].numpy()[:m] > 0.5).astype(float)
                row["user_active_label"] = describe(ua_r)
                row["user_label_matches_vad"] = bool(
                    np.array_equal(ua_r, tgt_pipeline[: len(ua_r)])
                )
            else:
                ua_r = tgt_pipeline
                row["user_active_label"] = None
                row["user_label_matches_vad"] = None
            if ba is not None:
                ba_r = (ba[r].numpy()[:m] > 0.5).astype(float)
                row["bystander_active_label"] = describe(ba_r)
            else:
                ba_r = itf_pipeline
                row["bystander_active_label"] = None
            row.update(turn_stats(batch, r, m))
            if far is not None and not absent:
                sirs = achieved_sir_db(
                    c, far[r, : min(L, far.shape[-1])], ua_r, ba_r, m
                )
            else:
                sirs = {"user_frames": None, "overlap": None, "level_ratio": None}
            row["achieved_sir_user_frames_db"] = sirs["user_frames"]
            row["achieved_sir_overlap_db"] = sirs["overlap"]
            row["achieved_level_ratio_db"] = sirs["level_ratio"]
            median_db, min_db = gap_floor_dbfs(noisy[r, :L], ua_r, ba_r, m)
            row["gap_floor_median_dbfs"] = median_db
            row["gap_floor_min_dbfs"] = min_db
            row["gap_is_digital_silence"] = (
                None if min_db is None else bool(min_db < -200.0)
            )
            fh.write(json.dumps(row) + "\n")
            n += 1
        fh.flush()
        if bi % 5 == 0:
            print(f"batch {bi} rows={n} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"DONE rows={n} batches={bi+1} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
