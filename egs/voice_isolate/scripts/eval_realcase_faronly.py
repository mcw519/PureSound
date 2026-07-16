"""Voicebot-correctness scorecard on the ai-coustics Voice Focus 2.2 real cases.

The criterion is the VOICEBOT EXPERIENCE, not "near=keep / far=suppress":
  * KEEP the conversational user -- even when they are far / reverberant / briefly
    quiet. Killing them = the bot goes deaf (dropped turns, deletions).
  * SUPPRESS anything that would corrupt ASR / turn-taking -- a bystander, a TV,
    echo. Leaking them = stray inserted words, broken turn-taking, polluted LLM
    context.

So a lone far-field user must be KEPT (not suppressed just for being far), while a
far competitor next to a near user must be SUPPRESSED. Distance alone cannot decide
it -- the same "far voice" is keep in one scene and suppress in another.

windows.json marks, per clip:
  * ``keep``     spans -> target output ~= input. ``preservation_db`` should be ~0;
                 a large negative value is a KEEP-VIOLATION (the model is killing a
                 legitimate user / foreground -- bot goes deaf).
  * ``suppress`` spans -> target output ~ silence. ``reduction_db`` should be very
                 negative; reduction > -6 dB is a SUPPRESS-FAIL (a non-user voice
                 leaked into the transcript).

No clean reference / transcript needed -- this is a segment-energy scorecard that
runs on any harvested real clip. Reports our model and QVF2.2 (reference ceiling)
side by side.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_realcase_faronly.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
        --cases-dir egs/voice_isolate/data_report/qvf22_real_cases --device cpu
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
    (
        _corpus_dict,
        _trainer_dict,
        _optim_dict,
        _scheduler_dict,
        _loss_dict,
        model_dict,
        *_rest,
    ) = load_siso_recipe_config(config_path)
    model = init_siso_model(model_dict)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


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
    args = parser.parse_args()

    device = torch.device(args.device)
    cases_dir = Path(args.cases_dir)
    windows_path = Path(args.windows) if args.windows else cases_dir / "windows.json"
    windows = {k: v for k, v in json.loads(windows_path.read_text()).items() if not k.startswith("_")}

    model = load_model(args.config_path, args.ckpt, device)

    header = ["clip", "role", "system", "keep_preserv_dB", "suppress_reduc_dB", "verdict"]
    print("\t".join(header))
    fails = {"ours": [], "qvf22": []}
    for clip, spec in windows.items():
        raw_path = cases_dir / f"{clip}_raw.wav"
        qvf22_path = cases_dir / f"{clip}_qvf22.wav"
        if not raw_path.is_file():
            print(f"# skip {clip}: missing {raw_path.name}")
            continue

        raw_wav, sr = AudioIO.open(f_path=str(raw_path), target_lvl=None, resample_to=16000)
        raw_wav = raw_wav.view(1, -1)
        with torch.no_grad():
            ours = model(raw_wav.to(device)).detach().cpu().view(1, -1).clamp(min=-1.0, max=1.0)

        systems = {"ours": ours}
        if qvf22_path.is_file():
            qvf22_wav, _ = AudioIO.open(f_path=str(qvf22_path), target_lvl=None, resample_to=16000)
            systems["qvf22"] = qvf22_wav.view(1, -1)

        keep_spans = spec.get("keep", [])
        supp_spans = spec.get("suppress", [])
        role = spec.get("role", "")
        for sysname, processed in systems.items():
            keep = window_power_db(processed, raw_wav, keep_spans, sr) if keep_spans else float("nan")
            supp = window_power_db(processed, raw_wav, supp_spans, sr) if supp_spans else float("nan")
            flags = []
            if keep == keep and keep < KEEP_VIOLATION_DB:  # not NaN and too suppressed
                flags.append("KEEP-VIOLATION")
                fails[sysname].append(f"{clip}:keep")
            if supp == supp and supp > SUPPRESS_FAIL_DB:
                flags.append("SUPPRESS-FAIL")
                fails[sysname].append(f"{clip}:suppress")
            verdict = ",".join(flags) if flags else "ok"
            ks = f"{keep:8.2f}" if keep == keep else "     n/a"
            ss = f"{supp:8.2f}" if supp == supp else "     n/a"
            print(f"{clip}\t{role}\t{sysname}\t{ks}\t{ss}\t{verdict}")

    print()
    print(f"# keep-span preservation should be ~0 dB; < {KEEP_VIOLATION_DB} dB = KEEP-VIOLATION (bot goes deaf to a user)")
    print(f"# suppress-span reduction should be very negative; > {SUPPRESS_FAIL_DB} dB = SUPPRESS-FAIL (non-user voice leaks)")
    for sysname in ("ours", "qvf22"):
        if sysname in fails:
            f = fails[sysname]
            print(f"# {sysname}: {'PASS (all clips voicebot-correct)' if not f else 'FAIL -> ' + ', '.join(f)}")


if __name__ == "__main__":
    main()
