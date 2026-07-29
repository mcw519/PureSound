"""Two-sided keep/suppress scorecard on real recordings.

Both failure directions matter and a single number hides one of them:
  * killing speech that must be kept produces deletions -- for a voice agent, a
    dropped turn;
  * leaking speech that must be suppressed (a bystander, a TV, echo) produces
    stray words and broken turn-taking.

So each clip is annotated span by span and scored on both sides separately.

windows.json marks, per clip:
  * ``keep``     spans -> target output ~= input. ``preservation_db`` should be ~0;
                 a large negative value is a KEEP-VIOLATION (the model is killing a
                 legitimate user / foreground -- bot goes deaf).
  * ``suppress`` spans -> target output ~ silence. ``reduction_db`` should be very
                 negative; reduction > -6 dB is a SUPPRESS-FAIL (a non-user voice
                 leaked into the transcript).

No clean reference / transcript needed -- this is a segment-energy scorecard that
runs on any harvested real clip. When a case directory also ships another system's
output (``<clip>_qvf22.wav``), it is scored alongside as ``reference``.

With ``--gate`` the model's frame-level gate head is applied to the output as a
multiplicative gain, adding two more systems (the separator is unchanged by gate
training, so the mask-only row alone cannot show a gate's effect):

    gate_soft[t] = enhanced[t] * sigmoid(gate_logit[frame(t)])
    gate_hard[t] = enhanced[t] * (sigmoid(gate_logit[frame(t)]) >= threshold)

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_realcase.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v7.ckpt --dry-blend 0.9 \
        --cases-dir egs/voice_isolate/data_report/qvf22_real_cases --device cpu

    # gate checkpoint: pass a config whose backbone has vad_head enabled
    uv run python egs/voice_isolate/scripts/eval_realcase.py \
        egs/voice_isolate/config/exp/train_dpcrn_gate.yaml \
        --ckpt <gate ckpt> --gate --gate-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.recipes import init_siso_model, load_siso_recipe_config

SUPPRESS_FAIL_DB = -6.0     # suppress-span reduction shallower than this = leak
KEEP_VIOLATION_DB = -3.0    # keep-span preservation below this = user/foreground killed


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = init_siso_model(load_siso_recipe_config(config_path)[5])
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


def window_power_db(processed: torch.Tensor, raw: torch.Tensor, spans, sr: int) -> float:
    """10*log10( sum(processed^2) / sum(raw^2) ) over the concatenated spans."""
    n = min(processed.shape[-1], raw.shape[-1])
    p_e = r_e = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), n)
        if j <= i:
            continue
        p_e += float(processed[..., i:j].square().sum())
        r_e += float(raw[..., i:j].square().sum())
    if r_e <= 0.0:
        return float("nan")
    return 10.0 * torch.log10(torch.tensor(p_e / r_e + 1e-12)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cases-dir", default="data_report/qvf22_real_cases")
    parser.add_argument("--windows", default=None, help="windows.json (default: <cases-dir>/windows.json)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-blend", type=float, default=1.0,
                        help="inference over-suppression relief: enh*b + mix*(1-b)")
    parser.add_argument("--spec-floor", type=float, default=0.0,
                        help="clamp enhanced |bin| to >= floor * |mix bin|")
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

    header = ["clip", "role", "system", "keep_preserv_dB", "suppress_reduc_dB", "verdict"]
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
            ours = model(raw_wav.to(device), dry_blend=args.dry_blend,
                         spec_floor=args.spec_floor).detach().cpu().view(1, -1).clamp(min=-1.0, max=1.0)
            logits = getattr(model.backbone, "last_vad_logits", None)

        systems = {"ours": ours}
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
        for sysname, processed in systems.items():
            keep = window_power_db(processed, raw_wav, keep_spans, sr) if keep_spans else float("nan")
            supp = window_power_db(processed, raw_wav, supp_spans, sr) if supp_spans else float("nan")
            flags = []
            if keep == keep and keep < KEEP_VIOLATION_DB:  # not NaN and too suppressed
                flags.append("KEEP-VIOLATION")
                fails.setdefault(sysname, []).append(f"{clip}:keep")
            if supp == supp and supp > SUPPRESS_FAIL_DB:
                flags.append("SUPPRESS-FAIL")
                fails.setdefault(sysname, []).append(f"{clip}:suppress")
            fails.setdefault(sysname, [])
            verdict = ",".join(flags) if flags else "ok"
            ks = f"{keep:8.2f}" if keep == keep else "     n/a"
            ss = f"{supp:8.2f}" if supp == supp else "     n/a"
            print(f"{clip}\t{role}\t{sysname}\t{ks}\t{ss}\t{verdict}")

    print()
    print(f"# keep-span preservation should be ~0 dB; < {KEEP_VIOLATION_DB} dB = KEEP-VIOLATION (bot goes deaf to a user)")
    print(f"# suppress-span reduction should be very negative; > {SUPPRESS_FAIL_DB} dB = SUPPRESS-FAIL (non-user voice leaks)")
    for sysname, f in fails.items():
        print(f"# {sysname}: {'PASS (all clips correct)' if not f else 'FAIL -> ' + ', '.join(f)}")


if __name__ == "__main__":
    main()
