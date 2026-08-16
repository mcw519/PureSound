"""Frame-level gate-head scorecard on a held-out synthetic validation set.

The gate recipe freezes a pretrained separator and optimizes ONLY the causal
near-field VAD gate head with VADHeadBCELoss. The
mask path never changes, so enhanced-energy scorecards (eval_turntaking.py /
eval_realcase.py) cannot show gate progress. This script instead reads the
gate logits (backbone.last_vad_logits) directly and scores them against vad_target:

  * KEEP recall (TPR on near-active frames)   -- gate should stay ON for the user.
                 Low recall = the gate would mute a legitimate near speaker.
  * SUPPRESS specificity (TNR on far/absent)  -- gate should turn OFF on far-only /
                 target-absent frames. Low specificity = far competitor leaks.
  * balanced_accuracy = (recall + specificity) / 2  -- the headline number; robust to
                 the label imbalance of far-only rows.
  * BCE (mean, with logits) -- comparable to the training/valid loss.

This is a SYNTHETIC in-domain smoke test (pipeline validation), not evidence of
transfer to end-to-end real recordings.

Usage (from the recipe dir egs/voice_isolate):
    uv run python scripts/eval_gate.py config/exp/train_dpcrn_gate.yaml \
        --ckpt exp/dpcrn_gate_synth/lightning_logs/version_0/checkpoints/epoch=1-step=500.ckpt \
        --device cuda --n-batches 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json-out", default=None, help="append one-line JSON record here")
    args = parser.parse_args()

    config_path = str(Path(args.config_path).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())
    os.chdir(RECIPE_DIR)
    torch.manual_seed(args.seed)

    (
        corpus,
        trainer,
        _optim,
        _scheduler,
        _loss,
        model_dict,
        aug_speech,
        aug_noise,
        aug_reverb,
        aug_speed,
        aug_ir,
        aug_src,
        aug_hpf,
        aug_volume,
        aug_codec,
        aug_packet_loss,
        aug_target_absent,
        vad_label,
        *rest,  # absorb recipe-tuple growth (new augmentation blocks append)
    ) = load_siso_recipe_config(config_path)
    aug_realfar = rest[0] if len(rest) > 0 else None
    aug_realnear = rest[1] if len(rest) > 1 else None

    corpus["training_length_seconds"] = args.seconds
    trainer.update(
        {
            "valid_iter_per_epoch": args.n_batches,
            "valid_seed": args.seed,
            "n_spk_per_batch": args.batch_size,
            "n_utt_per_speaker": 1,
            "num_workers": 8,
        }
    )
    _, valid_loader = recipe_main.init_dataloader(
        corpus,
        trainer,
        aug_speech,
        aug_noise,
        aug_reverb,
        aug_speed,
        aug_ir,
        aug_src,
        aug_hpf,
        aug_volume,
        aug_codec,
        aug_packet_loss,
        aug_target_absent,
        vad_label,
        aug_realfar,
        aug_realnear,
    )

    model = init_siso_model(model_dict)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("loss_func_list.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [ckpt] {len(missing)} missing key(s) kept at init: {missing[:4]}")
    if unexpected:
        print(f"  [ckpt] {len(unexpected)} unexpected key(s) ignored: {unexpected[:4]}")
    model.to(args.device).eval()

    tp = fp = tn = fn = 0
    bce_sum = 0.0
    frame_count = 0
    pos_frames = 0
    seen = 0
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    with torch.no_grad():
        for batch in valid_loader:
            if seen >= args.n_batches:
                break
            seen += 1
            noisy = batch["noisy_speech"].to(args.device)
            target = batch.get("vad_target")
            if target is None:
                raise RuntimeError("batch has no 'vad_target'; check vad_label config")
            target = target.to(args.device).float()
            model(noisy)
            logits = model.backbone.last_vad_logits
            if logits is None:
                raise RuntimeError("backbone.last_vad_logits is None; vad_head disabled?")
            logits = logits.reshape(logits.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
            n = min(logits.shape[-1], target.shape[-1])
            logits = logits[..., :n]
            target = target[..., :n]

            label = target > 0.5
            pred = torch.sigmoid(logits) >= args.threshold
            tp += int((pred & label).sum())
            fp += int((pred & ~label).sum())
            tn += int((~pred & ~label).sum())
            fn += int((~pred & label).sum())
            bce_sum += float(bce(logits, label.float()))
            frame_count += label.numel()
            pos_frames += int(label.sum())

    recall = tp / (tp + fn) if (tp + fn) else float("nan")          # KEEP near-active
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")     # SUPPRESS far/absent
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    accuracy = (tp + tn) / frame_count if frame_count else float("nan")
    bal_acc = (recall + specificity) / 2

    record = {
        "ckpt": ckpt_path,
        "batches": seen,
        "frames": frame_count,
        "positive_rate": round(pos_frames / frame_count, 4) if frame_count else None,
        "bce": round(bce_sum / frame_count, 4) if frame_count else None,
        "accuracy": round(accuracy, 4),
        "keep_recall": round(recall, 4),
        "suppress_specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "balanced_accuracy": round(bal_acc, 4),
    }
    print(json.dumps(record, indent=2))
    if args.json_out:
        with open(args.json_out, "a") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
