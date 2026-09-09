"""Measure the realised v20 row distribution without constructing a model.

The audit consumes the production training dataloader and records exactly the
requested rows (300 by default), including every turn's rendered distance,
presence labels, nested paired-view counts, and the exact proximity eligibility
mask. It performs no forward, backward, optimizer, or checkpoint operation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def _scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return value


def run(config_path: str, *, rows: int, out: str | None, num_workers: int):
    os.chdir(ROOT / "egs" / "voice_isolate")
    from puresound.config import load_recipe, with_overrides
    from puresound.nnet.loss import RelativeProximityLoss
    import egs.voice_isolate.main as recipe_main

    recipe = load_recipe(config_path, expected_task="voice_isolation", expected_purpose="train")
    if rows < 1:
        raise ValueError("rows must be positive")
    # Bucket probabilities are per batch. Preserve their speaker counts so the
    # realised row distribution matches training, including small long buckets.
    schedule = recipe.trainer.length_schedule or []
    bucket_values = [float(bucket.seconds) for bucket in schedule]
    min_batch_rows = min((bucket.n_spk for bucket in schedule), default=recipe.trainer.n_spk_per_batch)
    n_batches = (rows + min_batch_rows - 1) // min_batch_rows
    recipe = with_overrides(recipe, trainer={
        "train_iter_per_epoch": n_batches,
        "num_workers": num_workers,
    })
    loader, _ = recipe_main.init_dataloader(recipe)
    proximity = RelativeProximityLoss(
        min_turn_frames=recipe.loss_func[[x.type for x in recipe.loss_func].index("RelativeProximityLoss")].args.get("min_turn_frames", 1),
        min_distance_gap_m=recipe.loss_func[[x.type for x in recipe.loss_func].index("RelativeProximityLoss")].args.get("min_distance_gap_m", .25),
        scale_free=False,
    )
    report = {
        "config": str(Path(config_path).resolve()),
        "requested_rows": rows,
        "bucket_schedule": [bucket.model_dump() for bucket in schedule],
        "rows": 0,
        "session_rows": 0,
        "paired_rows": 0,
        "effective_ordering_pairs": 0,
        "known_turn_distances": [],
        "turn_distance_unknown": 0,
        "target_present": Counter(),
        "shape": Counter(),
        "bucket_seconds": Counter(),
        "vad_active_fraction": [],
        "paired_rows_by_bucket": Counter(),
    }
    for batch_index, batch in enumerate(loader, start=1):
        size = int(batch["noisy_speech"].shape[0])
        take = min(size, rows - report["rows"])
        if take <= 0:
            break
        for key in ("session_row", "session_shape", "session_n_turns", "audio_length", "length", "target_present"):
            if key not in batch:
                continue
        lengths = batch.get("length", batch.get("audio_length"))
        if lengths is not None:
            for value in lengths[:take].reshape(-1).tolist():
                actual = float(value) / 16000
                bucket_value = min(bucket_values, key=lambda candidate: abs(candidate - actual)) if bucket_values else actual
                bucket = f"{bucket_value:g}s"
                report["bucket_seconds"][bucket] += 1
        if "session_row" in batch:
            session = batch["session_row"][:take]
            report["session_rows"] += int((session > .5).sum())
            if "session_shape" in batch:
                for value in batch["session_shape"][:take].tolist():
                    report["shape"][str(int(value))] += 1
            if "target_present" in batch:
                for value in batch["target_present"][:take].tolist():
                    report["target_present"][str(int(value > .5))] += 1
        if "turn_distance" in batch:
            roles = batch.get("turn_role")
            for row_index, row in enumerate(batch["turn_distance"][:take]):
                role_row = roles[row_index] if roles is not None else None
                for turn_index, distance in enumerate(row.tolist()):
                    if role_row is not None and int(role_row[turn_index]) == 0:
                        continue
                    if distance == distance and distance > 0:
                        report["known_turn_distances"].append(float(distance))
                    else:
                        report["turn_distance_unknown"] += 1
        if "vad_target" in batch:
            report["vad_active_fraction"].extend(batch["vad_target"][:take].float().mean(-1).tolist())
        paired = batch.get("paired_view")
        if paired is not None:
            indices = paired["source_indices"][paired["source_indices"] < take]
            count = int(indices.numel())
            report["paired_rows"] += count
            if lengths is not None:
                for index in indices.tolist():
                    actual = float(lengths[index]) / 16000
                    bucket_value = min(bucket_values, key=lambda candidate: abs(candidate - actual)) if bucket_values else actual
                    report["paired_rows_by_bucket"][f"{bucket_value:g}s"] += 1
            # A nested view count is already provenance-safe: source_indices
            # points back into this batch, so no random source-id collision is
            # inferred here.
        label_batch = {key: value[:take] for key, value in batch.items()
                       if torch.is_tensor(value) and value.ndim and value.shape[0] == size}
        if "turn_id" in label_batch and label_batch["turn_id"].shape[1]:
            zeros = torch.zeros_like(label_batch["turn_id"], dtype=torch.float32)
            pair_data = proximity._pairs(zeros, label_batch)
            if pair_data is not None:
                report["effective_ordering_pairs"] += int(pair_data[1].sum())
        report["rows"] += take
        print(f"audit batch {batch_index}: {report['rows']}/{rows} rows", flush=True)
        if report["rows"] == rows:
            break
    if report["rows"] != rows:
        raise RuntimeError(f"audit exhausted after {report['rows']} of {rows} rows")
    report["target_present"] = dict(report["target_present"])
    report["shape"] = dict(report["shape"])
    report["bucket_seconds"] = dict(report["bucket_seconds"])
    report["paired_rows_by_bucket"] = dict(report["paired_rows_by_bucket"])
    distances = report["known_turn_distances"]
    report["turn_distance_summary_m"] = {
        "count": len(distances),
        "min": min(distances) if distances else None,
        "median": sorted(distances)[len(distances) // 2] if distances else None,
        "max": max(distances) if distances else None,
    }
    if out:
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp/train_dpcrn_v20_r1a.yaml")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default="benchmarks/probes/v20_session_rows/audit_300.json")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    print(json.dumps(run(args.config, rows=args.rows, out=args.out, num_workers=args.num_workers), indent=2))


if __name__ == "__main__":
    main()
