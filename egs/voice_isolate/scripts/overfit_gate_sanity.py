"""Overfit the DPCRN gate head on one fixed synthetic batch.

This is an engineering smoke test only. It verifies that the configured data
produces foreground-activity labels, that a separator checkpoint warm-starts
with only the new head missing, and that the frozen-bottleneck gate can reduce
BCE. It does not test transfer to real recordings.

Run from the repository root::

    uv run python egs/voice_isolate/scripts/overfit_gate_sanity.py \
      egs/voice_isolate/config/exp/train_dpcrn_gate.yaml \
      --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
      --device cpu --steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import egs.noise_suppression.main as recipe_main  # noqa: E402
from puresound.nnet.loss import VADHeadBCELoss  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    config_path = str(Path(args.config_path).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())
    os.chdir(RECIPE_DIR)
    torch.manual_seed(0)

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
        *_rest,  # tolerate recipe-tuple growth (new augmentation blocks append)
    ) = load_siso_recipe_config(config_path)

    corpus["training_length_seconds"] = args.sample_seconds
    trainer.update(
        {
            "train_iter_per_epoch": 1,
            "valid_iter_per_epoch": args.max_batches,
            "n_spk_per_batch": args.batch_size,
            "n_utt_per_speaker": 1,
            "num_workers": 0,
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
    )

    batch = None
    for candidate in valid_loader:
        labels = candidate.get("vad_target")
        if (
            labels is not None
            and bool((labels > 0.5).any())
            and bool((labels <= 0.5).any())
        ):
            batch = candidate
            break
    if batch is None:
        raise RuntimeError(
            "could not sample a batch containing both active and inactive gate labels"
        )

    model = init_siso_model(model_dict)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("loss_func_list.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = {
        f"backbone.vad_head.{name}"
        for name, _ in model.backbone.vad_head.named_parameters()
    }
    if set(missing) != expected_missing:
        raise RuntimeError(f"unexpected missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")

    model.to(args.device).train()
    noisy = batch["noisy_speech"].to(args.device)
    target = batch["vad_target"].to(args.device)

    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs) -> None:
        captured["bottleneck"] = inputs[0].detach()

    handle = model.backbone.vad_head.register_forward_pre_hook(capture_input)
    with torch.no_grad():
        model(noisy)
    handle.remove()
    bottleneck = captured["bottleneck"]

    head = model.backbone.vad_head
    params = list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    loss_fn = VADHeadBCELoss(balance_per_batch=True)
    losses = []
    for _ in range(args.steps + 1):
        logits = head(bottleneck)
        loss = loss_fn(logits, target)
        losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(bottleneck)
        n = min(logits.shape[-1], target.shape[-1])
        prediction = torch.sigmoid(logits[..., :n]) >= 0.5
        accuracy = float((prediction == (target[..., :n] > 0.5)).float().mean())

    print(
        {
            "batch_shape": tuple(noisy.shape),
            "positive_rate": float((target > 0.5).float().mean()),
            "trainable_params": sum(parameter.numel() for parameter in params),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "fixed_batch_accuracy": accuracy,
        }
    )
    if not losses[-1] < losses[0]:
        raise RuntimeError("gate head failed to reduce fixed-batch BCE")


if __name__ == "__main__":
    main()
