"""Overfit one fixed batch: can this model/pipeline learn at all?

Two modes, both driven by the same recipe config the training run uses. Neither says
anything about generalization -- they isolate "is the plumbing capable of learning?"
from "is the data/recipe right?", which is worth knowing before spending a full run.

Separator mode (default)
    Builds the model from scratch (or warm-starts with --ckpt), grabs ONE fixed batch
    from the valid dataloader (the hardest of --pick-hardest-of, so passthrough is not
    already a winning answer), freezes it, and trains on just that batch with the real
    training loss. Prints SI-SDR(enh, target) against the passthrough baseline.
      * climbs well above the mix baseline -> the model CAN separate; a full-run
        passthrough collapse is then a data/difficulty problem.
      * stalls at the mix baseline           -> gradient flow / architecture / loss bug.

Gate mode (--gate)
    Loads a separator checkpoint into a config with vad_head enabled, checks that the
    only missing weights are the new head's, captures one frozen bottleneck batch, and
    trains ONLY the causal gate head on its BCE. Verifies the label pipeline produces
    both active and inactive frames and that the head can fit them.

Usage (from anywhere):
    uv run python egs/voice_isolate/scripts/overfit_check.py \
        egs/voice_isolate/config/train_dpcrn.yaml --steps 800 --device cuda

    uv run python egs/voice_isolate/scripts/overfit_check.py \
        <gate-recipe>.yaml --gate \
        --ckpt egs/voice_isolate/pretrained_ckpt/backup/dpcrn_v6.ckpt --device cpu --steps 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

RECIPE_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> voice_isolate/
REPO = Path(__file__).resolve().parents[3]  # -> PureSound repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import egs.voice_isolate.main as recipe_main  # noqa: E402
from puresound.nnet.loss import VADHeadBCELoss  # noqa: E402
from puresound.recipes import (  # noqa: E402
    init_loss_func, init_siso_model)
from puresound.config import load_recipe, with_overrides


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.reshape(-1) - est.reshape(-1).mean()
    ref = ref.reshape(-1) - ref.reshape(-1).mean()
    alpha = (est @ ref) / ((ref @ ref) + eps)
    s = alpha * ref
    e = est - s
    return float(10.0 * torch.log10(((s @ s) + eps) / ((e @ e) + eps)))


def batch_sisdr(enh: torch.Tensor, ref: torch.Tensor) -> tuple[float, int]:
    """Mean SI-SDR over foreground-present rows (silent targets are undefined)."""
    vals = []
    for r in range(ref.shape[0]):
        if ref[r].abs().max() <= 1e-6:
            continue
        t = min(enh[r].shape[-1], ref[r].shape[-1])
        vals.append(si_sdr(enh[r, ..., :t], ref[r, ..., :t]))
    return sum(vals) / max(len(vals), 1), len(vals)


def build_valid_loader(config_path: str, num_workers: int, overrides: dict | None = None):
    """Recipe config -> valid dataloader, i.e. exactly the training synthesis path."""
    recipe = with_overrides(
        load_recipe(
            config_path,
            expected_task="voice_isolation",
            expected_purpose="train",
        ),
        trainer={"num_workers": num_workers, **(overrides or {}).get("trainer", {})},
        dataset=(overrides or {}).get("corpus", {}),
    )
    _train_dl, valid_dl = recipe_main.init_dataloader(recipe)
    return valid_dl, recipe


def run_separator(args, valid_dl, recipe) -> None:
    model = init_siso_model(recipe.model)
    # attach the real training losses (init_siso_model does not) -- same as main.py
    loss_list, loss_w = init_loss_func(recipe.loss_func)
    model.register_loss_func(loss_list, loss_w)
    vad_label = recipe.vad_label
    if vad_label is not None and vad_label.used and vad_label.backend == "silero":
        from puresound.audio.vad import BatchedSileroVADLabeler
        model.register_gpu_vad_labeler(BatchedSileroVADLabeler(**vad_label.args))
    if args.ckpt:
        state = torch.load(str(Path(args.ckpt).resolve()), map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"warm-start: {args.ckpt}")
    else:
        print("random init (true capacity test)")
    model.to(args.device)

    # fixed batch: the hardest of N, so passthrough is not already a good answer
    it = iter(valid_dl)
    batch = None
    hardest = 1e9
    for _ in range(max(1, args.pick_hardest_of)):
        cand = next(it)
        d, _ = batch_sisdr(cand["noisy_speech"], cand["clean_speech"])
        if d < hardest:
            hardest, batch = d, cand
    print(f"picked hardest of {args.pick_hardest_of}: mean SI-SDR(mix,target)={hardest:+.2f} dB", flush=True)

    batch = model.ensure_vad_targets(batch) if hasattr(model, "ensure_vad_targets") else batch
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(args.device)
    noisy, clean = batch["noisy_speech"], batch["clean_speech"]

    mix_vs_tgt, n_present = batch_sisdr(noisy, clean)
    print(f"fixed batch: {n_present} fg-present rows | SI-SDR(mix,target)={mix_vs_tgt:+.2f} dB "
          f"(this is what passthrough already gets; enh must beat it)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr)

    # match training precision (the recipes use bf16-mixed; fp32 can OOM)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda")

    model.train()
    for step in range(args.steps + 1):
        with amp:
            enh = model(noisy)
            total_loss, _losses = model.compute_loss(
                enhanced=enh, target=clean,
                vad_target=batch.get("vad_target"), batch=batch)
        opt.zero_grad()
        total_loss.backward()  # bf16 (not fp16) -> no GradScaler needed
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9)  # measure, don't clip
        opt.step()
        if step % args.print_every == 0:
            model.eval()
            with torch.no_grad(), amp:
                enh_e = model(noisy)
                sdr_t, _ = batch_sisdr(enh_e.float(), clean)
                sdr_m, _ = batch_sisdr(enh_e.float(), noisy)
            model.train()
            print(f"step {step:4d} | loss {float(total_loss):+.3f} | "
                  f"SI-SDR(enh,target) {sdr_t:+6.2f} | SI-SDR(enh,mix) {sdr_m:+6.2f} | "
                  f"gradnorm {float(gnorm):.2e}", flush=True)

    print("\nVERDICT:")
    print("  enh,target well above the mix baseline -> the model CAN separate; a passthrough")
    print("    collapse on full data is a data problem (mixtures too easy).")
    print("  enh,target stuck at the mix baseline   -> gradient/architecture/loss bug.")


def run_gate(args, valid_dl, model_dict) -> None:
    if not args.ckpt:
        sys.exit("--gate needs --ckpt (a trained separator to freeze)")

    batch = None
    for candidate in valid_dl:
        labels = candidate.get("vad_target")
        if labels is not None and bool((labels > 0.5).any()) and bool((labels <= 0.5).any()):
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("could not sample a batch containing both active and inactive gate labels")

    model = init_siso_model(model_dict)
    if getattr(model.backbone, "vad_head", None) is None:
        sys.exit("--gate needs a config whose backbone has vad_head enabled")
    state_dict = torch.load(str(Path(args.ckpt).resolve()), map_location="cpu")["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_func_list.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = {f"backbone.vad_head.{name}" for name, _ in model.backbone.vad_head.named_parameters()}
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
        loss = loss_fn(head(bottleneck), target)
        losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(bottleneck)
        n = min(logits.shape[-1], target.shape[-1])
        accuracy = float(((torch.sigmoid(logits[..., :n]) >= 0.5) == (target[..., :n] > 0.5)).float().mean())

    print({
        "batch_shape": tuple(noisy.shape),
        "positive_rate": float((target > 0.5).float().mean()),
        "trainable_params": sum(p.numel() for p in params),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "fixed_batch_accuracy": accuracy,
    })
    if not losses[-1] < losses[0]:
        raise RuntimeError("gate head failed to reduce fixed-batch BCE")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config_path")
    p.add_argument("--gate", action="store_true",
                   help="train only the frame-level gate head on a frozen bottleneck")
    p.add_argument("--ckpt", default=None,
                   help="warm-start (separator mode) / required frozen separator (--gate)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pick-hardest-of", type=int, default=1,
                   help="scan this many batches and overfit the HARDEST one (lowest mean "
                        "SI-SDR(mix,target)); easy batches cannot test separation")
    p.add_argument("--batch-size", type=int, default=None, help="speakers per batch override")
    p.add_argument("--sample-seconds", type=float, default=None,
                   help="clip length override (a short clip is enough for --gate)")
    args = p.parse_args()

    config_path = str(Path(args.config_path).resolve())
    os.chdir(RECIPE_DIR)
    torch.manual_seed(0)

    overrides: dict = {"corpus": {}, "trainer": {}}
    if args.sample_seconds is not None:
        overrides["corpus"]["training_length_seconds"] = float(args.sample_seconds)
    if args.batch_size is not None:
        overrides["trainer"].update({"n_spk_per_batch": args.batch_size, "n_utt_per_speaker": 1})
    if args.gate:
        overrides["trainer"].setdefault("valid_iter_per_epoch", 8)

    valid_dl, recipe = build_valid_loader(
        config_path, args.num_workers, overrides)

    if args.gate:
        if args.lr == 1e-3:
            args.lr = 1e-2      # head-only fit wants a larger step
        run_gate(args, valid_dl, recipe.model)
    else:
        run_separator(args, valid_dl, recipe)


if __name__ == "__main__":
    main()
