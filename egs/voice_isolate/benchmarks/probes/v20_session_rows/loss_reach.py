"""How often is each R1a loss term actually non-zero, at the R1a knobs?

Runs the REAL train loader of the R1a recipe and calls the two new loss terms on
every batch, with a randomly-initialised head over a random bottleneck of the
grid the backbone would produce (T_label - 1, verified). What is being measured
is the *structure* of the batch -- how many turns are eligible, whether any
speaker holds two of them, whether two rows share a source id -- not a value the
model would produce.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path("/home/milowu/A4Audio/PureSound")
RECIPE_DIR = REPO / "egs/voice_isolate"
sys.path.insert(0, str(REPO))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.config import load_recipe, with_overrides  # noqa: E402
from puresound.nnet.lobe.heads import IdentityHead, ProximityHead  # noqa: E402
from puresound.nnet.loss import (  # noqa: E402
    IdentityContrastiveLoss,
    RelativeProximityLoss,
)

C = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RECIPE_DIR / "config/exp/train_dpcrn_v20_r1a.yaml"))
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--only-bucket", type=float, default=None)
    ap.add_argument("--only-bucket-n-spk", type=int, default=2)
    args = ap.parse_args()
    os.chdir(RECIPE_DIR)
    torch.manual_seed(args.seed)

    recipe = load_recipe(args.config, expected_task="voice_isolation", expected_purpose="train")
    recipe = with_overrides(recipe, trainer={"num_workers": args.num_workers})
    if args.only_bucket is not None:
        recipe = with_overrides(recipe, trainer={"length_schedule": [
            {"seconds": args.only_bucket, "n_spk": args.only_bucket_n_spk, "prob": 1.0}
        ]})
    train_dl, _ = recipe_main.init_dataloader(recipe)

    identity_head = IdentityHead(enc_channels=C, dim=64, kernel_t=5)
    proximity_head = ProximityHead(enc_channels=C, hidden=64)
    identity_loss = IdentityContrastiveLoss(temperature=0.1, momentum=0.99, min_turn_frames=1)
    proximity_loss = RelativeProximityLoss(margin=1.0, consistency_weight=1.0, min_turn_frames=1)
    identity_loss.train()
    proximity_loss.train()

    stats = Counter()
    eligible_hist = Counter()
    n = 0
    for index, batch in enumerate(train_dl):
        if index >= args.batches:
            break
        n += 1
        frames = int(batch["user_active"].shape[1])
        rows = int(batch["user_active"].shape[0])
        bott = torch.randn(rows, C, frames - 1, requires_grad=True)
        emb = identity_head(bott)
        prox = proximity_head(bott)
        li = identity_loss(
            identity_emb=emb, identity_head=identity_head, bottleneck=bott, batch=batch
        )
        lp = proximity_loss(proximity=prox, batch=batch)
        stats["batches"] += 1
        stats["session"] += int(bool((batch["session_row"] > 0).any()))
        stats["identity_nonzero"] += int(abs(float(li)) > 0.0)
        stats["proximity_nonzero"] += int(abs(float(lp)) > 0.0)
        # eligible turns: a distinct non-zero turn_id, minus overlap frames
        turn_id = batch["turn_id"].clone()
        overlap = (batch["user_active"] > 0.5) & (batch["bystander_active"] > 0.5)
        turn_id[overlap[:, : turn_id.shape[1]]] = 0
        turn_id = turn_id[:, : frames - 1]
        total = 0
        repeated = False
        speakers_all = []
        for row in range(rows):
            ids = sorted({int(v) for v in turn_id[row].unique().tolist() if v > 0})
            total += len(ids)
            spk = [int(batch["turn_speaker"][row, k - 1]) for k in ids]
            speakers_all += spk
            if any(count >= 2 for count in Counter(spk).values()):
                repeated = True
        if Counter(speakers_all) and max(Counter(speakers_all).values()) >= 2:
            repeated = True
        eligible_hist[total] += 1
        stats["eligible_ge3"] += int(total >= 3)
        stats["repeated_speaker"] += int(repeated)
        stats["distinct_ge2"] += int(len(set(speakers_all)) >= 2)
        ids = batch["row_source_id"].view(-1).tolist()
        counts = Counter(v for v in ids if v >= 0)
        stats["within_batch_pair"] += int(any(c >= 2 for c in counts.values()))
        if index % 10 == 0:
            print(f"  batch {index}: rows={rows} eligible={total} "
                  f"id={float(li):.4f} prox={float(lp):.4f}", flush=True)

    out = {k: v for k, v in stats.items()}
    out["eligible_turns_hist"] = dict(sorted(eligible_hist.items()))
    for key in ("session", "identity_nonzero", "proximity_nonzero", "eligible_ge3",
                "repeated_speaker", "distinct_ge2", "within_batch_pair"):
        out[key + "_share"] = round(stats[key] / max(1, n), 4)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
