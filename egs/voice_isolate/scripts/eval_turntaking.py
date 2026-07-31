"""Keep/suppress scorecard on a frozen turn-taking (mix, target) set.

Where eval_realcase.py needs hand-annotated spans on a handful of clips, this derives
keep / suppress spans automatically from the TARGET energy of every item in a set built
by build_turntaking_set.py:

  * KEEP spans     = target ACTIVE -> the near speaker's turns. Output should preserve
                     them: keep_preservation_db = 10log10( E(enh)/E(target) ) ~ 0 dB.
                     << 0 = KEEP-VIOLATION (the near speaker was killed).
  * SUPPRESS spans = target SILENT while the MIX is active -> the far speaker's solo.
                     Output should silence them: suppress_reduction_db =
                     10log10( E(enh)/E(mix) ), very negative = good.
                     > threshold = SUPPRESS-FAIL (a far voice leaked through).

Plus whole-clip SI-SDR(enh, target). Runs on any set of <id>_mix.wav + <id>_target.wav,
so building the set with a measured-RIR bank scores real-room turn-taking instead of
simulated reverb.

With ``--gate`` the frame-level gate head is applied to the output as a multiplicative
gain, adding gate_soft / gate_hard systems next to the mask-only one (gate training
leaves the separator untouched, so the mask-only row cannot show a gate's effect).

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_turntaking.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
        --set-dir /data/audio/eval_noisy_data/turntaking_set_realrir --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402

KEEP_VIOLATION_DB = -3.0   # keep-span preservation below this = user/foreground killed
SUPPRESS_FAIL_DB = -6.0    # suppress-span reduction shallower than this = far leak


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = init_siso_model(load_siso_recipe_config(config_path)[5])
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(sd, load_loss_func=False)
    else:
        model.load_state_dict(sd)
    return model.to(device).eval()


def gate_gain(logits: torch.Tensor, n_samples: int, hop: int, threshold: float | None) -> torch.Tensor:
    """Upsample per-frame gate logits to a sample-rate multiplicative gain in [0,1]."""
    prob = torch.sigmoid(logits.reshape(-1).float())
    if threshold is not None:
        prob = (prob >= threshold).float()
    gain = prob.repeat_interleave(hop)
    if gain.shape[-1] < n_samples:
        gain = torch.cat([gain, gain[-1:].expand(n_samples - gain.shape[-1])])
    return gain[:n_samples].view(1, -1)


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.reshape(-1) - est.reshape(-1).mean()
    ref = ref.reshape(-1) - ref.reshape(-1).mean()
    a = (est @ ref) / ((ref @ ref) + eps)
    s = a * ref
    e = est - s
    return float(10.0 * torch.log10(((s @ s) + eps) / ((e @ e) + eps)))


def frame_spans(target: torch.Tensor, mix: torch.Tensor, sr: int,
                win_ms: float = 25.0, floor_db: float = -40.0):
    """Split the clip into KEEP (target active) and SUPPRESS (target silent, mix
    active) span lists [(a_s, b_s), ...] from framed energy, floor relative to the
    per-clip peak frame of each signal."""
    win = max(1, int(sr * win_ms / 1000.0))
    x, m = target.reshape(-1), mix.reshape(-1)
    n = (min(x.numel(), m.numel()) // win) * win
    if n == 0:
        return [], []
    tf = x[:n].reshape(-1, win).square().mean(1).sqrt()
    mf = m[:n].reshape(-1, win).square().mean(1).sqrt()
    t_db = 20 * torch.log10((tf / tf.max().clamp_min(1e-12)).clamp_min(1e-12))
    m_db = 20 * torch.log10((mf / mf.max().clamp_min(1e-12)).clamp_min(1e-12))
    t_active = t_db > floor_db
    m_active = m_db > floor_db
    keep_flag = t_active                      # target present -> keep
    supp_flag = (~t_active) & m_active        # target silent but mix has energy -> suppress

    def to_spans(flag):
        spans, run = [], None
        for i, f in enumerate(flag.tolist()):
            if f and run is None:
                run = i
            elif not f and run is not None:
                spans.append((run * win / sr, i * win / sr))
                run = None
        if run is not None:
            spans.append((run * win / sr, len(flag) * win / sr))
        return spans

    return to_spans(keep_flag), to_spans(supp_flag)


def span_power_db(num: torch.Tensor, den: torch.Tensor, spans, sr: int) -> float:
    """10*log10( sum(num^2) / sum(den^2) ) over the concatenated spans."""
    n = min(num.shape[-1], den.shape[-1])
    ne = de = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), n)
        if j <= i:
            continue
        ne += float(num[..., i:j].square().sum())
        de += float(den[..., i:j].square().sum())
    if de <= 0.0:
        return float("nan")
    return 10.0 * torch.log10(torch.tensor(ne / de + 1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-blend", type=float, default=1.0,
                    help="inference over-suppression relief: enh*b + mix*(1-b)")
    ap.add_argument("--spec-floor", type=float, default=0.0,
                    help="clamp enhanced |bin| to >= floor * |mix bin|")
    ap.add_argument("--gate", action="store_true",
                    help="also score the output multiplied by the frame-level gate head")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--hop", type=int, default=160, help="gate frame hop in samples")
    args = ap.parse_args()

    device = torch.device(args.device)
    set_dir = Path(args.set_dir)
    mixes = sorted(set_dir.glob("*_mix.wav"))
    if args.limit:
        mixes = mixes[: args.limit]
    if not mixes:
        sys.exit(f"no *_mix.wav under {set_dir}")

    model = load_model(args.config_path, args.ckpt, device)
    systems = ["ours"] + (["gate_soft", "gate_hard"] if args.gate else [])
    keep: dict[str, list[float]] = {s: [] for s in systems}
    supp: dict[str, list[float]] = {s: [] for s in systems}
    sisdr: dict[str, list[float]] = {s: [] for s in systems}
    kviol = {s: 0 for s in systems}
    sfail = {s: 0 for s in systems}
    out_rows = []

    for mp in mixes:
        item = mp.name[: -len("_mix.wav")]
        tp = set_dir / f"{item}_target.wav"
        if not tp.is_file():
            continue
        mix, sr = AudioIO.open(f_path=str(mp), target_lvl=None, resample_to=16000)
        tgt, _ = AudioIO.open(f_path=str(tp), target_lvl=None, resample_to=16000)
        mix, tgt = mix.view(1, -1), tgt.view(1, -1)
        with torch.no_grad():
            enh = model(mix.to(device), dry_blend=args.dry_blend,
                        spec_floor=args.spec_floor).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
            logits = getattr(model.backbone, "last_vad_logits", None)

        # DPCRN look-ahead can shorten the output by a few samples -- align lengths.
        L = min(enh.shape[-1], mix.shape[-1], tgt.shape[-1])
        mix, tgt, enh = mix[..., :L], tgt[..., :L], enh[..., :L]

        outs = {"ours": enh}
        if args.gate:
            if logits is None:
                raise RuntimeError("--gate needs a backbone with vad_head enabled (see config/exp/train_dpcrn_gate.yaml)")
            logits = logits.detach().cpu()
            outs["gate_soft"] = (enh * gate_gain(logits, L, args.hop, None)).clamp(-1.0, 1.0)
            outs["gate_hard"] = (enh * gate_gain(logits, L, args.hop, args.gate_threshold)).clamp(-1.0, 1.0)

        keep_spans, supp_spans = frame_spans(tgt, mix, sr)
        row = {"id": item,
               "keep_span_s": round(sum(b - a for a, b in keep_spans), 2),
               "suppress_span_s": round(sum(b - a for a, b in supp_spans), 2)}
        for s in systems:
            kp = span_power_db(outs[s], tgt, keep_spans, sr) if keep_spans else float("nan")
            sd_ = span_power_db(outs[s], mix, supp_spans, sr) if supp_spans else float("nan")
            ss = si_sdr(outs[s], tgt)
            if kp == kp:
                keep[s].append(kp)
                kviol[s] += int(kp < KEEP_VIOLATION_DB)
            if sd_ == sd_:
                supp[s].append(sd_)
                sfail[s] += int(sd_ > SUPPRESS_FAIL_DB)
            sisdr[s].append(ss)
            row[f"{s}_keep_preservation_db"] = round(kp, 2) if kp == kp else None
            row[f"{s}_suppress_reduction_db"] = round(sd_, 2) if sd_ == sd_ else None
            row[f"{s}_si_sdr_db"] = round(ss, 2)
        out_rows.append(row)

    (set_dir / "turntaking_scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n")

    def med(v):
        return st.median(v) if v else float("nan")

    def mean(v):
        return st.mean(v) if v else float("nan")

    n = len(out_rows)
    print("=" * 78)
    print(f"Turn-taking keep/suppress scorecard  (n={n})   set={set_dir.name}")
    print(f"  ckpt : {args.ckpt}")
    if args.dry_blend != 1.0 or args.spec_floor:
        print(f"  knobs: dry_blend={args.dry_blend} spec_floor={args.spec_floor}")
    print("=" * 78)
    print(f"{'system':<10} {'KEEP med/mean':>19} {'ok/viol':>9}   "
          f"{'SUPPRESS med/mean':>20} {'ok/fail':>9}   {'SI-SDR med':>10}")
    for s in systems:
        nk, ns = len(keep[s]), len(supp[s])
        print(f"{s:<10} {med(keep[s]):+8.2f}/{mean(keep[s]):+6.2f} dB "
              f"{nk - kviol[s]:>4}/{kviol[s]:<4}   "
              f"{med(supp[s]):+9.2f}/{mean(supp[s]):+7.2f} dB "
              f"{ns - sfail[s]:>4}/{sfail[s]:<4}   {med(sisdr[s]):+9.2f}")
    print()
    print(f"# KEEP wants ~0 dB (< {KEEP_VIOLATION_DB} = KEEP-VIOLATION); "
          f"SUPPRESS wants << 0 dB (> {SUPPRESS_FAIL_DB} = SUPPRESS-FAIL)")
    print(f"# per-item scores -> {set_dir / 'turntaking_scores.jsonl'}")


if __name__ == "__main__":
    main()
