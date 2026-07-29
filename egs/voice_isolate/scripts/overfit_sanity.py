"""Overfit-tiny-batch sanity test: can the model learn separation AT ALL?

Builds the model FROM SCRATCH (random init, no ckpt), grabs ONE fixed batch
from the valid dataloader, freezes it (same tensors every step -> no
augmentation randomness), and trains on just that batch with the model's REAL
training loss (model.compute_loss, same path as training_step). Prints
SI-SDR(enh, target) over steps.

Interpretation:
  - SI-SDR(enh,target) climbs high (+15..+30 dB)  -> model/pipeline CAN learn
    separation; the full-data passthrough collapse is a DATA problem (mixtures
    too easy -> passthrough rewarded). Fix: harden the SNR/difficulty.
  - SI-SDR(enh,target) stalls near the mix level    -> fundamental bug
    (gradient flow / architecture / loss). No data change will help.

Usage (GPU, from anywhere):
    uv run python egs/voice_isolate/scripts/overfit_sanity.py \
        egs/voice_isolate/config/exp/train_dpcrn_wide_antisup.yaml \
        --steps 800 --device cuda
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

from puresound.recipes import (  # noqa: E402
    init_loss_func, init_siso_model, load_siso_recipe_config)
import egs.noise_suppression.main as M  # noqa: E402


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.reshape(-1) - est.reshape(-1).mean()
    ref = ref.reshape(-1) - ref.reshape(-1).mean()
    alpha = (est @ ref) / ((ref @ ref) + eps)
    s = alpha * ref
    e = est - s
    return float(10.0 * torch.log10(((s @ s) + eps) / ((e @ e) + eps)))


def batch_sisdr(enh, ref, eps=1e-8):
    vals = []
    for r in range(ref.shape[0]):
        if ref[r].abs().max() <= 1e-6:
            continue
        t = min(enh[r].shape[-1], ref[r].shape[-1])
        vals.append(si_sdr(enh[r, ..., :t], ref[r, ..., :t]))
    return sum(vals) / max(len(vals), 1), len(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config_path")
    p.add_argument("--ckpt", default=None,
                   help="optional warm-start; default = random init (true capacity test)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pick-hardest-of", type=int, default=1,
                   help="scan this many batches and overfit the HARDEST one "
                        "(lowest mean SI-SDR(mix,target)). Easy batches can't "
                        "test separation -- passthrough already wins there.")
    args = p.parse_args()

    config_path = str(Path(args.config_path).resolve())
    ckpt_path = str(Path(args.ckpt).resolve()) if args.ckpt else None
    os.chdir(RECIPE_DIR)
    torch.manual_seed(0)

    cfg = load_siso_recipe_config(config_path)
    (corpus, trainer, _opt, _sch, _loss, model_dict, a_sp, a_no, a_rv, a_spd,
     a_ir, a_src, a_hpf, a_vol, a_cod, a_pl, a_ta, a_vad) = cfg
    trainer["num_workers"] = args.num_workers

    _train_dl, valid_dl = M.init_dataloader(
        corpus, trainer, a_sp, a_no, a_rv, a_spd, a_ir, a_src, a_hpf,
        a_vol, a_cod, a_pl, a_ta, a_vad)

    model = init_siso_model(model_dict)
    # attach the real training losses (init_siso_model does not) -- same as main.py
    loss_list, loss_w = init_loss_func(hparam_conf=_loss)
    model.register_loss_func(loss_list, loss_w)
    if a_vad and a_vad.get("used") and a_vad.get("backend", "energy").lower() == "silero":
        from puresound.audio.vad import BatchedSileroVADLabeler
        model.register_gpu_vad_labeler(BatchedSileroVADLabeler(**a_vad.get("args", {})))
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"warm-start: {ckpt_path}")
    else:
        print("random init (true capacity test)")
    model.to(args.device)

    # pick a fixed batch (the HARDEST of N, so passthrough is NOT already good),
    # frozen for the whole run
    it = iter(valid_dl)
    batch = None
    best_diff = 1e9
    for _ in range(max(1, args.pick_hardest_of)):
        cand = next(it)
        nb = cand["noisy_speech"]
        cb = cand["clean_speech"]
        d, _ = batch_sisdr(nb, cb)
        if d < best_diff:
            best_diff, batch = d, cand
    print(f"picked hardest of {args.pick_hardest_of}: mean SI-SDR(mix,target)={best_diff:+.2f} dB", flush=True)
    batch = model.ensure_vad_targets(batch) if hasattr(model, "ensure_vad_targets") else batch
    noisy = batch["noisy_speech"].to(args.device)
    clean = batch["clean_speech"].to(args.device)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(args.device)

    # passthrough baselines on this fixed batch
    mix_vs_tgt, npres = batch_sisdr(noisy, clean)
    print(f"fixed batch: {npres} fg-present rows | SI-SDR(mix,target)={mix_vs_tgt:+.2f} dB "
          f"(this is what passthrough already gets; we want enh to BEAT it)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr)

    # match training precision (config uses bf16-mixed; fp32 here OOMs)
    use_amp = args.device == "cuda"
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

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
    print("  enh,target climbed high (>>mix baseline) -> model CAN separate; passthrough is a DATA problem (mixtures too easy).")
    print("  enh,target stuck near mix baseline       -> fundamental bug (gradient/architecture/loss).")


if __name__ == "__main__":
    main()
