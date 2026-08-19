"""Presence margin by RT60 band -- the measurement v11 is judged against.

WHY THIS EXISTS (full_gate/v8_gate050_VERDICT.md): the presence gate's deletion
cost rose monotonically with reverberation because the readout was DRR-shaped,
and DRR falls with distance AND with RT60. A presence signal is only usable if
its margin HOLDS ACROSS RT60 -- a single-room or single-band number cannot see
this failure, which is how the field scorecard passed a gate that deleted the
user everywhere else.

Frames come from the live training distribution (the same synthetic rows the
recipe trains on), with per-row rt60 from the bank metadata. Truth per frame is
the energy labeler on the row's own stems:

    present = target speech active            (the near user is talking)
    absent  = target silent AND far active    (a bystander is talking, user not)
    dropped = neither active                  (nothing to decide)

Two sources for the score, so the same table reads "before" and "after":

  --readout X.npz    the offline linear readout on the frozen bottleneck
                     (the b-trajectory probe's artefact) -- v8's "before"
  --head             backbone.last_vad_logits (and last_background_vad_logits
                     when present) -- the v11 heads' "after"

Verdict column: per-band AUC of present vs absent frames. The confound signature
is the near-frame score falling with rt60; a head that holds its margin in the
0.5-0.9 bands (where bank_drr_overlap_README.md shows DRR cannot answer) has
learned something DRR is not.

Usage, from the recipe dir:
    uv run python benchmarks/probes/presence_stratified_probe.py \
        config/exp/eval_indomain_phase1.yaml --ckpt pretrained_ckpt/dpcrn_v8.ckpt \
        --readout benchmarks/probes/presence_readout_v8.npz --n-batches 60
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR))

from puresound.audio.vad import EnergyVADLabeler
from puresound.config import load_recipe, with_overrides
from puresound.recipes import init_siso_model

BANDS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 9.9))


def auc(pos, neg):
    """Rank AUC, no sklearn needed at this size."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort().argsort() + 1          # ranks, ties ignored (scores are dense)
    rp = order[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--readout", default=None,
                    help="offline linear readout .npz; omit with --head")
    ap.add_argument("--head", action="store_true",
                    help="score with backbone.last_vad_logits instead")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=3)
    args = ap.parse_args()
    if bool(args.readout) == bool(args.head):
        ap.error("exactly one of --readout / --head")

    config_path = str(Path(args.config_path).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())
    readout_path = str(Path(args.readout).resolve()) if args.readout else None
    os.chdir(RECIPE_DIR)
    torch.manual_seed(0)

    import main as M
    recipe = load_recipe(config_path, expected_task="voice_isolation",
                         expected_purpose="train")
    recipe = with_overrides(recipe, trainer={"num_workers": args.num_workers})
    _train, valid_dl = M.init_dataloader(recipe)

    model = init_siso_model(recipe.model)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.eval().to(args.device)

    cap = {}
    if readout_path:
        rd = np.load(readout_path)
        w = torch.from_numpy(rd["weight"]).float().to(args.device)
        bias = float(rd["bias"])
        hook_point = getattr(model.backbone, "dist_head", None)
        if hook_point is None:
            raise SystemExit("--readout needs a train-config backbone with dist_head "
                             "to hook; use the train yaml, not infer")
        hook_point.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("x", inp[0].detach()))

    # The same labeler family the losses use, on the row's own stems.
    labeler = EnergyVADLabeler(frame_length=400, hop_length=160)

    rows = {"near": [], "bg": []}
    with torch.no_grad():
        for bi, batch in enumerate(valid_dl):
            if bi >= args.n_batches:
                break
            noisy = batch["noisy_speech"].to(args.device)
            model(noisy)
            if readout_path:
                pooled = cap["x"].mean(dim=2)                       # [N, C, T]
                score = torch.einsum("nct,c->nt", pooled.float(), w) + bias
            else:
                score = model.backbone.last_vad_logits
                if score is None:
                    raise SystemExit("--head: this checkpoint has no vad_head")
            bg_score = getattr(model.backbone, "last_background_vad_logits", None)

            rt60 = batch.get("rt60")
            clean = batch["clean_speech"]
            far = batch.get("far_target")
            for r in range(clean.shape[0]):
                rv = float(rt60.view(-1)[r].item()) if rt60 is not None else float("nan")
                if rv != rv:
                    continue
                tgt_act = labeler(clean[r].view(-1), sample_rate=16000).bool()
                far_act = (labeler(far[r].view(-1), sample_rate=16000).bool()
                           if far is not None else torch.zeros_like(tgt_act))
                n = min(score.shape[-1], tgt_act.shape[-1])
                s = score[r, :n].float().cpu().numpy()
                ta, fa = tgt_act[:n].numpy(), far_act[:n].numpy()
                present, absent = ta, (~ta) & fa
                if present.any() or absent.any():
                    rows["near"].append((rv, s[present], s[absent]))
                if bg_score is not None:
                    sb = bg_score[r, :n].float().cpu().numpy()
                    # bg truth: far active vs far silent (target state irrelevant)
                    rows["bg"].append((rv, sb[fa], sb[~fa]))

    for name, data in rows.items():
        if not data:
            continue
        print(f"\n===== {name} head/readout: score by rt60 band =====")
        print(f"  {'band':>10s} {'n_pres':>8s} {'n_abs':>8s} "
              f"{'med(pres)':>10s} {'med(abs)':>9s} {'AUC':>6s}")
        for lo, hi in BANDS:
            pres = np.concatenate([p for rv, p, a in data if lo <= rv < hi] or [[]])
            absn = np.concatenate([a for rv, p, a in data if lo <= rv < hi] or [[]])
            if len(pres) < 50 or len(absn) < 50:
                print(f"  {lo:.1f}-{hi:.1f}   {len(pres):8d} {len(absn):8d}   -- thin --")
                continue
            print(f"  {lo:.1f}-{hi:.1f}   {len(pres):8d} {len(absn):8d} "
                  f"{np.median(pres):10.2f} {np.median(absn):9.2f} "
                  f"{auc(pres, absn):6.3f}")
        pres = np.concatenate([p for _, p, a in data])
        absn = np.concatenate([a for _, p, a in data])
        print(f"  {'ALL':>10s} {len(pres):8d} {len(absn):8d} "
              f"{np.median(pres):10.2f} {np.median(absn):9.2f} {auc(pres, absn):6.3f}")


if __name__ == "__main__":
    main()
