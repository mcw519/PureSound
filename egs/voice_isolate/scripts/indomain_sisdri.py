"""In-domain SI-SDRi: the realistic separation metric.

Reuses the EXACT training/valid synthesis pipeline (init_dataloader) and
measures SI-SDR improvement of the model output vs the early-reverb target
(``clean_speech``), on foreground-present rows only (silent target-absent /
distance-gated rows are skipped -- SI-SDR is undefined there).

Why this and not diagnose_distance.py: diagnose's near reference is full-reverb
while the model target is early-reverb, so its absolute SI-SDR is confounded.
And the raw valid SDR loss is misleading -- it averages in the silent rows, so
it drops as the model learns to silence, not to separate. This script isolates
"can the model actually separate foreground from interferer+noise in-domain".

Auto-adapts to query-conditioned vs fixed models: it passes whatever
``query_distance`` the dataloader emits (None for the no-query nearfield config).

Usage (from anywhere; CPU by default so it never touches the training GPUs):
    uv run python egs/voice_isolate/scripts/indomain_sisdri.py \
        egs/voice_isolate/config/train_dpcrn_wide_antisup.yaml \
        --ckpt <ckpt> --n-batches 40 --device cpu
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

# repo root (.../PureSound) and recipe dir (.../egs/voice_isolate)
RECIPE_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> voice_isolate/
REPO = Path(__file__).resolve().parents[3]  # -> PureSound repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from puresound.audio.io import AudioIO  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402
import egs.noise_suppression.main as M  # noqa: E402


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.reshape(-1) - est.reshape(-1).mean()
    ref = ref.reshape(-1) - ref.reshape(-1).mean()
    alpha = (est @ ref) / ((ref @ ref) + eps)
    s = alpha * ref
    e = est - s
    return float(10.0 * torch.log10(((s @ s) + eps) / ((e @ e) + eps)))


# mix_mode float codes -> names (mirror puresound/task/voice_isolation.py).
MIX_MODE_NAMES = {0.0: "none", 1.0: "legacy", 2.0: "physical",
                  3.0: "moderate", 4.0: "counter_level"}
_META_KEYS = ("drr_gap", "foreground_distance", "rt60", "n_interferers",
              "near_count", "far_count", "overlap_fraction", "mix_mode",
              "realized_speech_sir", "noise_snr")


def _pull(batch, key, r):
    """Per-row scalar metadata as float, or None if absent / NaN."""
    t = batch.get(key)
    if t is None:
        return None
    try:
        v = float(t.view(-1)[r].item())
    except Exception:
        return None
    return None if v != v else v  # drop NaN


def _row_meta(batch, r):
    return {k: _pull(batch, k, r) for k in _META_KEYS}


def _binned(v, edges, names):
    """Map v into len(edges)+1 labelled bins (names indexed by edge crossings)."""
    if v is None:
        return None
    i = 0
    for edge in edges:
        if v < edge:
            break
        i += 1
    return names[i]


def _report_buckets(title, rows, key_of):
    groups = defaultdict(list)
    for row in rows:
        k = key_of(row)
        if k is not None:
            groups[k].append(row["sisdri"])
    if not groups:
        return
    print(f"  -- by {title} --")
    for k in sorted(groups, key=str):
        v = groups[k]
        print(f"     {str(k):16s} n={len(v):4d}  "
              f"median={st.median(v):+6.2f}  mean={st.mean(v):+6.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config_path")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-batches", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=3,
                   help="override dataloader workers for the quick eval")
    p.add_argument("--silence-eps", type=float, default=1e-6,
                   help="rows whose clean target max-abs <= this are treated as "
                        "target-absent and skipped")
    p.add_argument("--dump-distribution", action="store_true",
                   help="print percentiles of SI-SDR(mix,target) etc. -- if mix "
                        "SI-SDR is already high, passthrough is rewarded and the "
                        "model has no incentive to separate")
    p.add_argument("--by-bucket", action="store_true",
                   help="report SI-SDRi grouped by composition / far_count / "
                        "mix_mode / drr_gap / overlap_fraction / realized SIR, "
                        "plus F-only (silent-target) power-reduction + false-near "
                        "rate -- the honest slice the aggregate median hides")
    p.add_argument("--save-audio", type=int, default=0,
                   help="save this many (mix, target, enhanced) wav triples")
    p.add_argument("--audio-out", default="./data_report/indomain_audio",
                   help="dir for saved audio (relative to the recipe dir)")
    args = p.parse_args()

    # resolve user paths BEFORE chdir (they may be relative to the launch cwd)
    config_path = str(Path(args.config_path).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())
    # config metafile paths (data/...) are relative to the recipe dir
    os.chdir(RECIPE_DIR)
    torch.manual_seed(0)

    cfg = load_siso_recipe_config(config_path)
    (corpus, trainer, _opt, _sch, _loss, model_dict, a_sp, a_no, a_rv, a_spd,
     a_ir, a_src, a_hpf, a_vol, a_cod, a_pl, a_ta, a_qd, a_vad) = cfg
    trainer["num_workers"] = args.num_workers

    _train_dl, valid_dl = M.init_dataloader(
        corpus, trainer, a_sp, a_no, a_rv, a_spd, a_ir, a_src, a_hpf,
        a_vol, a_cod, a_pl, a_ta, a_qd, a_vad)

    model = init_siso_model(model_dict)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
              f"(ok if only loss-fn buffers)")
    model.eval().to(args.device)
    print(f"ckpt: {ckpt_path}", flush=True)

    sr = int(corpus.get("target_sample_rate", 16000))
    out_dir = Path(args.audio_out)
    if args.save_audio:
        out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    mix_sisdr, enh_sisdr, sisdri = [], [], []
    rows, fonly = [], []  # per-row metadata for --by-bucket
    n_present = n_total = 0
    with torch.no_grad():
        for bi, batch in enumerate(valid_dl):
            if bi >= args.n_batches:
                break
            noisy = batch["noisy_speech"].to(args.device)
            clean = batch["clean_speech"].to(args.device)
            qd = batch.get("query_distance")
            qd = qd.to(args.device) if qd is not None else None
            enh = model(noisy, query_distance=qd)
            B = clean.shape[0]
            n_total += B
            for r in range(B):
                cl = clean[r]
                if cl.abs().max() <= args.silence_eps:  # target-absent / silent
                    # SI-SDRi is undefined vs a silent target. For F-only buckets
                    # score suppression instead: how far below the input the
                    # output sits (DISTANCE_PARENT proposal §7).
                    if args.by_bucket:
                        tl = min(enh[r].shape[-1], noisy[r].shape[-1])
                        in_rms = float(noisy[r, ..., :tl].pow(2).mean().sqrt())
                        out_rms = float(enh[r, ..., :tl].pow(2).mean().sqrt())
                        if in_rms > 0:
                            meta = _row_meta(batch, r)
                            meta["power_reduction_db"] = 20.0 * float(
                                np.log10((out_rms + 1e-8) / (in_rms + 1e-8)))
                            fonly.append(meta)
                    continue
                n_present += 1
                t = min(enh[r].shape[-1], noisy[r].shape[-1], cl.shape[-1])
                m = si_sdr(noisy[r, ..., :t], cl[..., :t])
                e = si_sdr(enh[r, ..., :t], cl[..., :t])
                mix_sisdr.append(m)
                enh_sisdr.append(e)
                sisdri.append(e - m)
                if args.by_bucket:
                    meta = _row_meta(batch, r)
                    meta.update(sisdri=e - m, mix=m, enh=e)
                    rows.append(meta)
                if saved < args.save_audio:
                    meta = _row_meta(batch, r)
                    nc, fc = meta.get("near_count"), meta.get("far_count")
                    comp = (
                        f"{int(nc)}N+{int(fc)}F"
                        if nc is not None and fc is not None else "NA"
                    )
                    mm = MIX_MODE_NAMES.get(meta.get("mix_mode"), "na")
                    sirv = meta.get("realized_speech_sir")
                    sir_s = f"sir{sirv:+.0f}" if sirv is not None else "sirNA"
                    tag = f"case{saved:02d}_{comp}_{mm}_{sir_s}_si{e - m:+.1f}dB"
                    AudioIO.save(noisy[r, ..., :t].reshape(1, -1).cpu(),
                                 str(out_dir / f"{tag}_mix.wav"), sr)
                    AudioIO.save(cl[..., :t].reshape(1, -1).cpu(),
                                 str(out_dir / f"{tag}_target.wav"), sr)
                    AudioIO.save(enh[r, ..., :t].reshape(1, -1).cpu(),
                                 str(out_dir / f"{tag}_enh.wav"), sr)
                    saved += 1
            if (bi + 1) % 10 == 0 and sisdri:
                print(f"  {bi+1}/{args.n_batches} batches | present={n_present} | "
                      f"running SI-SDRi mean={st.mean(sisdri):+.2f} "
                      f"median={st.median(sisdri):+.2f}", flush=True)

    if not sisdri:
        if fonly:
            print("no foreground-present rows (target_absent probe) -- F-only suppression only:")
            red = [row["power_reduction_db"] for row in fonly]
            false_near = float(np.mean(np.array(red) > -6.0))
            print(f"  n={len(red)}  median power reduction={st.median(red):+.1f} dB "
                  f"(more negative = better suppression)")
            print(f"  false-near rate (reduction > -6 dB) = {false_near:.0%}")
        else:
            print("no foreground-present rows found -- nothing to score")
        return
    print("=" * 62)
    print("IN-DOMAIN (synthetic valid, vs early-reverb target, fg-present only)")
    print("=" * 62)
    print(f"rows: {n_present} present / {n_total} total")
    print(f"SI-SDR  mix -> enh : {st.mean(mix_sisdr):+.2f} -> {st.mean(enh_sisdr):+.2f} dB")
    print(f"SI-SDRi : mean {st.mean(sisdri):+.2f} dB, median {st.median(sisdri):+.2f} dB")

    if args.dump_distribution:
        pcts = [10, 25, 50, 75, 90]
        def row(name, arr):
            qs = np.percentile(arr, pcts)
            return f"  {name:18s}" + "  ".join(f"p{p}={q:+6.2f}" for p, q in zip(pcts, qs))
        print("-" * 62)
        print("DISTRIBUTION (dB) -- if mix SI-SDR is mostly high, foreground is")
        print("dominant and passthrough is already 'good enough' -> no separation incentive")
        print(row("SI-SDR(mix,tgt)", mix_sisdr))
        print(row("SI-SDR(enh,tgt)", enh_sisdr))
        print(row("SI-SDRi", sisdri))
        frac_easy = float(np.mean(np.array(mix_sisdr) > 5.0))
        print(f"  fraction of rows with mix SI-SDR > +5 dB (passthrough already good): {frac_easy:.0%}")

    if args.by_bucket:
        print("-" * 62)
        print("PER-BUCKET SI-SDRi (median/mean) -- watch the hard buckets, not the")
        print("aggregate median: easy buckets can hide damage on hard ones")
        _report_buckets("composition (near+far)", rows, lambda r: (
            None if r.get("near_count") is None or r.get("far_count") is None
            else f"{int(r['near_count'])}N+{int(r['far_count'])}F"))
        _report_buckets("far_count", rows, lambda r: (
            None if r.get("far_count") is None else int(r["far_count"])))
        _report_buckets("mix_mode", rows,
                        lambda r: MIX_MODE_NAMES.get(r.get("mix_mode")))
        _report_buckets("drr_gap (dB)", rows, lambda r: _binned(
            r.get("drr_gap"), [0, 3, 6], ["<0", "0-3", "3-6", ">6"]))
        _report_buckets("overlap_fraction", rows, lambda r: _binned(
            r.get("overlap_fraction"), [0.1, 0.4, 0.7],
            ["<.1", ".1-.4", ".4-.7", ">.7"]))
        _report_buckets("realized_sir (dB)", rows, lambda r: _binned(
            r.get("realized_speech_sir"), [-3, 3, 9],
            ["<-3", "-3..3", "3..9", ">9"]))
        print("  -- F-only (silent-target) suppression --")
        if fonly:
            red = [row["power_reduction_db"] for row in fonly]
            false_near = float(np.mean(np.array(red) > -6.0))
            print(f"     n={len(red)}  median power reduction={st.median(red):+.1f} dB "
                  f"(more negative = better suppression)")
            print(f"     false-near rate (reduction > -6 dB) = {false_near:.0%}")
        else:
            print("     no silent-target rows in this eval (target_absent off)")

    if args.save_audio:
        print(f"\nsaved {saved} (mix,target,enh) triples -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
