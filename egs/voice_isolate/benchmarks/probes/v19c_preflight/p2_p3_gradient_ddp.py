"""P2 + P3 -- gradient share of the new term, and DDP safety on a real batch.

v19c_round_design.md 3.3.

P2: is it a targeted hinge or a global reweighting? Gradient-norm share of the
new term at init on real training batches, at the configured weight 0.25.
    kill   > 50% of total (v18 in reverse: v18 made every suppression number
             shallower by relaxing a floor; the mirror failure is a new term
             that owns the gradient)
    rescale > 20%
Reported two ways, because they answer different questions:
  * as the distribution delivers it -- most batches have no eligible row at all,
    so the term's gradient is exactly zero there;
  * conditional on the batch containing at least one eligible row -- which is
    what "targeted or global" actually asks.

P3: a batch with no eligible row must return a graph-carrying zero, and a
forward/backward through `siso.compute_loss` with the term configured must not
raise. A `None` or a NaN here severs a DDP job, as the background-VAD crash
already did once.

fp32, not bf16-mixed: a gradient-norm ratio read under autocast would be
comparing terms whose per-op precision differs. The ratio is a shape property of
the objective, not a number the training run reproduces bit for bit.

    cd egs/voice_isolate && uv run python \
      benchmarks/probes/v19c_preflight/p2_p3_gradient_ddp.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_common as pc  # noqa: E402

from puresound.nnet.loss import AnchorInheritanceLoss  # noqa: E402
from puresound.system.base import invoke_loss  # noqa: E402


def _grad_norm(scalar: torch.Tensor, params) -> float:
    """||dL/dtheta||_2, with the graph released.

    One term per forward pass, deliberately. Holding eight retained graphs plus
    the two SSL encoders OOMs a 23 GB card at the 6 s bucket (measured: 10.7 GiB
    reserved-but-unallocated at the point of failure), and the peak this way is
    the peak the training step actually has. The module is in eval mode, so
    dropout is off and BatchNorm statistics are frozen -- every re-forward
    produces identical activations and the per-term norms stay comparable.
    """
    grads = torch.autograd.grad(
        scalar, params, retain_graph=False, allow_unused=True
    )
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return float(np.sqrt(total))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=pc.CONFIG)
    ap.add_argument("--ckpt", default=pc.V16_EP19)
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--max-seconds", type=float, default=6.5,
                    help="skip longer buckets: the two SSL encoders plus a "
                         "backward do not fit beside a 12 s or 30 s bucket on a "
                         "23 GB card at fp32 (training runs bf16-mixed on two)")
    ap.add_argument("--firing-only", action="store_true",
                    help="measure only batches with >=1 eligible row. The share "
                         "conditional on firing is the number that answers "
                         "'targeted or global', and at 1-8% eligibility per row a "
                         "plain sweep spends most of its batches on exact zeros. "
                         "P3's no-eligible-row case is then not exercised -- run "
                         "once without this flag for that.")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.seed_everything(args.seed)

    model, recipe, device = pc.load_model(
        args.config, args.ckpt, args.device, with_losses=True
    )
    # cuDNN refuses `RNN backward` outside training mode ("cudnn RNN backward can
    # only be called in training mode"), so the module has to be in train mode to
    # measure a gradient at all. Dropout and BatchNorm are then pinned back to
    # eval by hand: every pass has to see identical activations or the per-term
    # gradient norms are not comparable, and a BatchNorm whose running statistics
    # move while being measured is measuring itself.
    model.train(True)
    frozen = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.dropout._DropoutNd,
                               torch.nn.modules.batchnorm._BatchNorm,
                               torch.nn.modules.instancenorm._InstanceNorm)):
            module.eval()
            frozen += 1
    print(f"[p2] train mode with {frozen} dropout/normalisation modules pinned "
          "to eval", flush=True)
    names = [type(loss).__name__ for loss in model.loss_func_list]
    weights = list(model.loss_func_list_w)
    new_index = names.index("AnchorInheritanceLoss")
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[p2] {len(params)} parameter tensors; terms={list(zip(names, weights))}",
          flush=True)

    train, _valid, _recipe = pc.build_loaders(args.config, args.num_workers)
    probe = AnchorInheritanceLoss(
        margin_db=model.loss_func_list[new_index].margin_db
    )

    records: list[dict] = []
    p3 = {"no_eligible_batches": 0, "graph_carrying_zero": 0,
          "backward_ok": 0, "nan_or_none": 0, "errors": []}
    t0 = time.time()
    for bi, batch in enumerate(train):
        if bi >= args.max_batches or len(records) >= args.n_batches:
            break
        seconds = pc.bucket(batch, 0)
        if seconds > args.max_seconds:
            continue
        on_device = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        noisy = on_device["noisy_speech"]
        clean = on_device["clean_speech"]
        vad = on_device.get("vad_target")

        def one_pass(term_index: int | None):
            """A fresh forward, then one weighted term (or the total)."""
            model.zero_grad(set_to_none=True)
            enhanced = model(noisy)
            # the same alignment compute_loss does, so per-term values match it
            length = min(enhanced.shape[-1], clean.shape[-1])
            target = clean[..., :length]
            providers = model._loss_providers(
                enhanced=enhanced[..., :length], target=target, vad_target=vad,
                batch=on_device, inactive_labels=target.abs().amax(dim=-1) == 0,
            )
            if term_index is None:
                value = None
                for i, loss_func in enumerate(model.loss_func_list):
                    weighted = weights[i] * invoke_loss(loss_func, providers)
                    value = weighted if value is None else value + weighted
            else:
                value = weights[term_index] * invoke_loss(
                    model.loss_func_list[term_index], providers
                )
            return value, enhanced[..., :length], target, providers

        with torch.no_grad():
            enhanced_only = model(noisy)
            length_only = min(enhanced_only.shape[-1], clean.shape[-1])
            scores = probe.row_scores(
                enhanced_only[..., :length_only], clean[..., :length_only],
                on_device, vad
            )
        n_eligible = int(scores["eligible"].sum())
        del enhanced_only
        if args.firing_only and n_eligible == 0:
            torch.cuda.empty_cache()
            continue

        try:
            per_term, per_norm = [], []
            for i in range(len(model.loss_func_list)):
                value, _e, _t, _p = one_pass(i)
                per_term.append(float(value))
                per_norm.append(_grad_norm(value, params))
                del value, _e, _t, _p
            total, _e, _t, _p = one_pass(None)
            total_norm = _grad_norm(total, params)
            del total, _e, _t, _p
            torch.cuda.empty_cache()
        except Exception as exc:                     # pragma: no cover - reported
            p3["errors"].append(f"batch {bi} (P2): {type(exc).__name__}: {exc}")
            print(f"[p2] ERROR on batch {bi}: {exc}", flush=True)
            torch.cuda.empty_cache()
            continue

        new_value = per_term[new_index]
        new_norm = per_norm[new_index]
        record = {
            "batch": bi,
            "row_seconds": seconds,
            "batch_rows": int(clean.shape[0]),
            "n_eligible": n_eligible,
            "terms": dict(zip(names, per_term)),
            "grad_norms": dict(zip(names, per_norm)),
            "total_grad_norm": total_norm,
            "new_term_value": new_value,
            "new_term_grad_norm": new_norm,
            "share_of_total_norm": (new_norm / total_norm) if total_norm > 0 else 0.0,
            "share_of_summed_norms": (new_norm / sum(per_norm)) if sum(per_norm) > 0
            else 0.0,
        }
        records.append(record)
        print(f"batch {bi} sec={seconds:g} rows={record['batch_rows']} "
              f"eligible={n_eligible} L={new_value:.4f} "
              f"|g_new|={new_norm:.4g} |g_tot|={total_norm:.4g} "
              f"share={record['share_of_total_norm'] * 100:.2f}%", flush=True)

        # ---- P3, on this batch: a real forward/backward through compute_loss
        try:
            if n_eligible == 0:
                p3["no_eligible_batches"] += 1
                _v, enhanced, target, providers = one_pass(new_index)
                term = invoke_loss(model.loss_func_list[new_index], providers)
                ok = bool(term.grad_fn is not None and term.requires_grad
                          and float(term) == 0.0 and torch.isfinite(term))
                p3["graph_carrying_zero"] += int(ok)
                if not ok:
                    p3["nan_or_none"] += 1
                    print(f"[p3] batch {bi}: NOT a graph-carrying zero: "
                          f"value={float(term)} grad_fn={term.grad_fn}", flush=True)
                del _v, term, providers
            else:
                model.zero_grad(set_to_none=True)
                full = model(noisy)
                cut = min(full.shape[-1], clean.shape[-1])
                enhanced, target = full[..., :cut], clean[..., :cut]
            model.zero_grad(set_to_none=True)
            total_again, values = model.compute_loss(
                enhanced=enhanced, target=target,
                vad_target=vad, batch=on_device,
            )
            if not torch.isfinite(total_again):
                p3["nan_or_none"] += 1
                print(f"[p3] batch {bi}: non-finite total {float(total_again)}",
                      flush=True)
            total_again.backward()
            p3["backward_ok"] += 1
            record["compute_loss_total"] = float(total_again)
            record["compute_loss_terms"] = values
        except Exception as exc:                     # pragma: no cover - reported
            p3["errors"].append(f"batch {bi} (P3): {type(exc).__name__}: {exc}")
            print(f"[p3] ERROR on batch {bi}: {exc}", flush=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    print(f"DONE batches={len(records)} elapsed={time.time() - t0:.0f}s", flush=True)
    report = summarize(records, names, weights, new_index, p3)
    (out_dir / "p2_p3.json").write_text(
        json.dumps({"batches": records, **report}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nreport -> {out_dir / 'p2_p3.json'}")
    return 0


def summarize(records, names, weights, new_index, p3) -> dict:
    print("\n" + "=" * 78)
    print(f"P2  gradient-norm share of AnchorInheritanceLoss   (n={len(records)} "
          "real train batches, fp32)")
    print("=" * 78)
    if not records:
        print("  (no batches measured)")
        return {"verdict": "NO DATA", "p3": p3}

    def shares(subset):
        return pc.describe([r["share_of_total_norm"] * 100.0 for r in subset])

    all_batches = shares(records)
    fired = [r for r in records if r["n_eligible"] > 0]
    # The stratum that matters: an eligible row whose hinge is already satisfied
    # contributes an exact zero and no gradient at all, so "conditional on an
    # eligible row" still averages in zeros. The term only exists on the batches
    # where L > 0.
    active = [r for r in records if r["new_term_value"] > 0.0]
    conditional = shares(fired) if fired else {"n": 0}
    active_shares = shares(active) if active else {"n": 0}
    print(f"all batches            {pc.fmt(all_batches)}   (% of ||g_total||)")
    print(f"batches with >=1 elig. {pc.fmt(conditional)}   "
          f"({len(fired)}/{len(records)} batches)")
    print(f"batches with L > 0     {pc.fmt(active_shares)}   "
          f"({len(active)}/{len(records)} batches)")
    print("\nper-term gradient norms, median over the measured batches:")
    for i, name in enumerate(names):
        values = [r["grad_norms"][name] for r in records]
        term_values = [r["terms"][name] for r in records]
        print(f"  {name:26s} w={weights[i]:<5g} |g| med={np.median(values):10.4g}  "
              f"weighted loss med={np.median(term_values):10.4g}")
    print(f"  {'||g_total||':26s}          med="
          f"{np.median([r['total_grad_norm'] for r in records]):10.4g}")

    worst = max(r["share_of_total_norm"] * 100.0 for r in records)
    median = active_shares.get("median", 0.0) if active else 0.0
    kill = worst > 50.0
    rescale = median > 20.0
    print("\n" + "-" * 78)
    print(f"P2  worst single batch share  : {worst:6.2f}%  "
          f"(kill > 50%)  -> {'KILL' if kill else 'pass'}")
    print(f"P2  median share when L > 0   : {median:6.2f}%  "
          f"(rescale > 20%) -> {'RESCALE' if rescale else 'pass'}")
    verdict = "KILL" if kill else ("RESCALE" if rescale else "PASS")
    print(f"P2 VERDICT: {verdict}")

    print("\n" + "=" * 78)
    print("P3  DDP safety")
    print("=" * 78)
    print(f"batches with no eligible row      : {p3['no_eligible_batches']}")
    print(f"  returned a graph-carrying zero  : {p3['graph_carrying_zero']}")
    print(f"  returned None / NaN / no graph  : {p3['nan_or_none']}")
    print(f"forward+backward completed        : {p3['backward_ok']}/{len(records)}")
    print(f"exceptions                        : {len(p3['errors'])} {p3['errors'][:3]}")
    p3_pass = (p3["nan_or_none"] == 0 and not p3["errors"]
               and p3["graph_carrying_zero"] == p3["no_eligible_batches"]
               and p3["backward_ok"] == len(records))
    print(f"P3 VERDICT: {'PASS' if p3_pass else 'FAIL'}")
    print("-" * 78)

    return {
        "share_all_batches_pct": all_batches,
        "share_when_fired_pct": conditional,
        "share_when_active_pct": active_shares,
        "n_batches_fired": len(fired),
        "n_batches_active": len(active),
        "worst_batch_share_pct": worst,
        "kill_gt_50pct": kill,
        "rescale_gt_20pct": rescale,
        "verdict": verdict,
        "p3": {**p3, "verdict": "PASS" if p3_pass else "FAIL"},
    }


if __name__ == "__main__":
    raise SystemExit(main())
