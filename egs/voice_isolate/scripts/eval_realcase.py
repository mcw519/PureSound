"""Two-sided keep/suppress scorecard on real recordings, scored in absolute levels.

Both failure directions matter and a single number hides one of them:
  * killing speech that must be kept produces deletions -- for a voice agent, a
    dropped turn;
  * leaking speech that must be suppressed (a bystander, a TV, echo) produces
    stray words and broken turn-taking.

So each clip is annotated span by span and scored on both sides separately.

WHY NOT ATTENUATION ALONE.  ``reduction_db`` is relative, so it is only comparable
between spans that started at comparable levels.  A far voice already close to the
capture noise floor has almost no removable energy: the model can attenuate it by 30 dB
and the measured reduction still bottoms out at the floor, and conversely a span that
*is* the floor scores 0 dB no matter what the model does.  Every suppress span is
therefore reported three ways:

  reduction_db   out/in over the span            -- how hard the model pushed
  residual_db    out - floor_dbfs                -- what is still above room tone;
                                                    this is "did the bystander go away"
  sir_out_db     near_ref_dbfs - out             -- how far the leftover sits below the
                                                    user's own voice, i.e. what an ASR
                                                    downstream actually competes with

and a span whose input is less than ``--min-headroom`` dB above the floor is reported
NOT-SCORABLE rather than given a misleading number.  ``floor_dbfs`` (5th-percentile
frame energy of the recording) and ``near_ref_dbfs`` (energy of its keep spans) are
written into windows.json by the cases-dir build script; without them the floor is
estimated from the clip itself and ``sir_out_db`` is left blank.

windows.json marks, per clip:
  * ``keep``     spans -> target output ~= input. ``preservation_db`` should be ~0;
                 a large negative value is a KEEP-VIOLATION (the model is killing a
                 legitimate user / foreground -- bot goes deaf).
  * ``suppress`` spans -> target output ~ silence. SUPPRESS-FAIL when the model barely
                 pushed at all; SUPPRESS-PARTIAL when it pushed but the residual is
                 still well above room tone (audible bystander, just quieter).

No clean reference / transcript needed -- this is a segment-energy scorecard that runs
on any harvested real clip. When a case directory also ships another system's output
(``<clip>_qvf22.wav``), it is scored alongside as ``reference``.

With ``--gate`` the model's frame-level gate head is applied to the output as a
multiplicative gain, adding two more systems (the separator is unchanged by gate
training, so the mask-only row alone cannot show a gate's effect):

    gate_soft[t] = enhanced[t] * sigmoid(gate_logit[frame(t)])
    gate_hard[t] = enhanced[t] * (sigmoid(gate_logit[frame(t)]) >= threshold)

Usage (from repo root):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python egs/voice_isolate/scripts/eval_realcase.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
        --cases-dir egs/voice_isolate/data_report/field_cases/test_vector_cases --device cuda:0

    # gate checkpoint: pass a config whose backbone has vad_head enabled
    uv run python egs/voice_isolate/scripts/eval_realcase.py \
        egs/voice_isolate/config/exp/train_dpcrn_gate.yaml \
        --ckpt <gate ckpt> --gate --gate-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gate_flags import (add_presence_gate_arg, build_presence_gate,
                         add_onset_guard_arg, build_onset_guard)

SUPPRESS_FAIL_DB = -6.0     # suppress-span reduction shallower than this = leak
SUPPRESS_PARTIAL_DB = 12.0  # residual this far above the floor = audible bystander
KEEP_VIOLATION_DB = -3.0    # keep-span preservation below this = user/foreground killed
FLOOR_PERCENTILE = 0.05
FRAME, HOP = 1600, 800      # 100 ms / 50 ms, for the floor estimate


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = init_siso_model(
        load_recipe(config_path, expected_task="voice_isolation").model
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict, load_loss_func=False)
    else:
        model.load_state_dict(state_dict)
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


def span_dbfs(wav: torch.Tensor, spans, sr: int, limit: int) -> float:
    """Mean-square level of the concatenated spans, in dBFS."""
    total = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j <= i:
            continue
        total += float(wav[..., i:j].square().sum())
        n += j - i
    if n <= 0:
        return float("nan")
    return 10.0 * math.log10(total / n + 1e-12)


def estimate_floor(wav: torch.Tensor) -> float:
    frames = wav.view(-1).unfold(-1, FRAME, HOP).square().mean(-1)
    return float(10.0 * torch.log10(torch.quantile(frames, FLOOR_PERCENTILE) + 1e-12))


def fmt(v: float, width: int = 8) -> str:
    return " " * (width - 3) + "n/a" if v != v else f"{v:{width}.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cases-dir", default=str(REPO_ROOT / "egs/voice_isolate/data_report/field_cases/test_vector_cases"))
    parser.add_argument("--windows", default=None, help="windows.json (default: <cases-dir>/windows.json)")
    parser.add_argument("--device", default="cpu")
    add_presence_gate_arg(parser)
    add_onset_guard_arg(parser)
    parser.add_argument("--dry-blend", type=float, default=1.0,
                        help="inference over-suppression relief: enh*b + mix*(1-b)")
    parser.add_argument("--spec-floor", type=float, default=0.0,
                        help="clamp enhanced |bin| to >= floor * |mix bin|")
    parser.add_argument("--min-headroom", type=float, default=6.0,
                        help="a suppress span less than this far above the noise floor is NOT-SCORABLE")
    parser.add_argument("--gate", action="store_true",
                        help="also score the output multiplied by the frame-level gate head")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="hard-gate threshold on the gate probability")
    parser.add_argument("--hop", type=int, default=160, help="gate frame hop in samples")
    args = parser.parse_args()

    device = torch.device(args.device)
    cases_dir = Path(args.cases_dir)
    windows_path = Path(args.windows) if args.windows else cases_dir / "windows.json"
    windows = {k: v for k, v in json.loads(windows_path.read_text()).items() if not k.startswith("_")}

    model = load_model(args.config_path, args.ckpt, device)
    gate = build_presence_gate(args)
    guard = build_onset_guard(args)

    header = ["clip", "role", "system", "keep_preserv_dB", "keep_out_dBFS",
              "supp_reduc_dB", "supp_out_dBFS", "residual_dB", "sir_out_dB", "headroom_dB", "verdict"]
    print("\t".join(header))
    fails: dict[str, list[str]] = {}
    for clip, spec in windows.items():
        raw_path = cases_dir / f"{clip}_raw.wav"
        reference_path = cases_dir / f"{clip}_qvf22.wav"
        if not raw_path.is_file():
            print(f"# skip {clip}: missing {raw_path.name}")
            continue

        raw_wav, sr = AudioIO.open(f_path=str(raw_path), target_lvl=None, resample_to=16000)
        raw_wav = raw_wav.view(1, -1)
        with torch.no_grad():
            # A head-driven gate is applied below as its own system, so the
            # `ours` row stays mask-only and the comparison has a baseline.
            ours = model(raw_wav.to(device), dry_blend=args.dry_blend,
                         spec_floor=args.spec_floor,
                         presence_gate=None if (gate is not None and gate.head_driven)
                         else gate, onset_guard=guard).detach().cpu().view(1, -1).clamp(min=-1.0, max=1.0)
            logits = getattr(model.backbone, "last_vad_logits", None)

        systems = {"ours": ours}
        if gate is not None and gate.head_driven:
            if logits is None:
                raise RuntimeError(
                    "--presence-gate-head needs a backbone with vad_head enabled "
                    "(use a config whose backbone_args declares it)")
            systems["presence_gate"] = gate.apply(
                ours, hop=args.hop, logits=logits.detach().cpu()
            ).clamp(-1.0, 1.0)
        if args.gate:
            if logits is None:
                raise RuntimeError("--gate needs a backbone with vad_head enabled (see config/exp/train_dpcrn_gate.yaml)")
            logits = logits.detach().cpu()
            n = ours.shape[-1]
            systems["gate_soft"] = (ours * gate_gain(logits, n, args.hop, None)).clamp(-1.0, 1.0)
            systems["gate_hard"] = (ours * gate_gain(logits, n, args.hop, args.gate_threshold)).clamp(-1.0, 1.0)
        if reference_path.is_file():
            reference_wav, _ = AudioIO.open(f_path=str(reference_path), target_lvl=None, resample_to=16000)
            systems["reference"] = reference_wav.view(1, -1)

        keep_spans = spec.get("keep", [])
        supp_spans = spec.get("suppress", [])
        role = spec.get("role", "")
        floor = spec.get("floor_dbfs")
        if floor is None:
            floor = estimate_floor(raw_wav)
        near_ref = spec.get("near_ref_dbfs", float("nan"))

        n = raw_wav.shape[-1]
        keep_in = span_dbfs(raw_wav, keep_spans, sr, n) if keep_spans else float("nan")
        supp_in = span_dbfs(raw_wav, supp_spans, sr, n) if supp_spans else float("nan")
        headroom = supp_in - floor if supp_in == supp_in else float("nan")

        for sysname, processed in systems.items():
            m = min(processed.shape[-1], n)
            keep_out = span_dbfs(processed, keep_spans, sr, m) if keep_spans else float("nan")
            supp_out = span_dbfs(processed, supp_spans, sr, m) if supp_spans else float("nan")
            keep = keep_out - keep_in if keep_out == keep_out else float("nan")
            supp = supp_out - supp_in if supp_out == supp_out else float("nan")
            residual = supp_out - floor if supp_out == supp_out else float("nan")
            sir_out = near_ref - supp_out if supp_out == supp_out and near_ref == near_ref else float("nan")

            flags = []
            if keep == keep and keep < KEEP_VIOLATION_DB:
                flags.append("KEEP-VIOLATION")
                fails.setdefault(sysname, []).append(f"{clip}:keep")
            if supp == supp:
                if headroom == headroom and headroom < args.min_headroom:
                    flags.append("NOT-SCORABLE")
                elif supp > SUPPRESS_FAIL_DB:
                    flags.append("SUPPRESS-FAIL")
                    fails.setdefault(sysname, []).append(f"{clip}:suppress")
                elif residual == residual and residual > SUPPRESS_PARTIAL_DB:
                    flags.append("SUPPRESS-PARTIAL")
            fails.setdefault(sysname, [])
            verdict = ",".join(flags) if flags else "ok"
            print(f"{clip}\t{role}\t{sysname}\t{fmt(keep)}\t{fmt(keep_out)}\t{fmt(supp)}\t"
                  f"{fmt(supp_out)}\t{fmt(residual)}\t{fmt(sir_out)}\t{fmt(headroom)}\t{verdict}")

    print()
    print(f"# keep_preserv_dB ~ 0 dB; < {KEEP_VIOLATION_DB} dB = KEEP-VIOLATION (bot goes deaf to a user)")
    print(f"# supp_reduc_dB very negative; > {SUPPRESS_FAIL_DB} dB = SUPPRESS-FAIL (non-user voice leaks)")
    print(f"# residual_dB = output level above the recording's noise floor; > {SUPPRESS_PARTIAL_DB} dB with a")
    print("#   passing reduction = SUPPRESS-PARTIAL (pushed, but the bystander is still above room tone)")
    print("# sir_out_dB = how far the residual sits below the user's own voice (bigger is better)")
    print(f"# headroom_dB = input level above the floor; < {args.min_headroom} dB = NOT-SCORABLE, there was")
    print("#   nothing to remove and reduction_dB would only measure the floor")
    for sysname, f in fails.items():
        print(f"# {sysname}: {'PASS (all clips correct)' if not f else 'FAIL -> ' + ', '.join(f)}")


if __name__ == "__main__":
    main()
