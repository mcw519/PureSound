"""What does the model *believe* about distance, on clips it fails to suppress?

The suppression ladder across capture domains -- VOiCES -14.5 dB, RealMAN -7.3, our field
recordings -1.2 (v8, >1 m / cold-start far) -- has two possible causes, and they need
opposite fixes:

  A. the encoder still recovers the distance cue on the failing chain, but the mask ignores
     it.  Fix: couple mask depth to the distance estimate.
  B. the cue is gone by the time it reaches the bottleneck.  Fix: data from that chain, or
     a representation that survives it.  No amount of loss shaping helps.

The DPCRN carries a training-only utterance-level DistHead regressing, in order:

    0. foreground DRR / drr_scale     1. log10(foreground distance)
    2. log10(nearest interferer distance)

Inference never reads it, which is exactly why it is a clean probe: it was never tuned to
make the field set look good. Run it on clips whose true distance you know and compare.

    uv run python egs/voice_isolate/scripts/probe_distance_head.py \
        egs/voice_isolate/config/train_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/backup/dpcrn_v10.ckpt \
        --cases-dir egs/voice_isolate/data_report/field_cases/test_vector_cases --device cuda:0

Pass the TRAINING config, not infer_dpcrn.yaml -- the inference config builds no dist head
and the probe would have nothing to read.
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

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.config import load_recipe  # noqa: E402
from puresound.recipes import init_siso_model  # noqa: E402

DRR_SCALE = 10.0   # must match DistHeadRegressionLoss(drr_scale=...) used in training


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    model = init_siso_model(
        load_recipe(
            config_path,
            expected_task="voice_isolation",
            expected_purpose="train",
        ).model
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.reload_checkpoint(state, load_loss_func=False)
    model = model.to(device).eval()
    if getattr(model.backbone, "dist_head", None) is None:
        raise SystemExit(
            "this config builds no dist_head -- pass the training config "
            "(config/train_dpcrn.yaml), not the inference one"
        )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases-dir", default=str(REPO_ROOT / "egs/voice_isolate/data_report/field_cases/test_vector_cases"))
    ap.add_argument("--windows", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-blend", type=float, default=0.9)
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="trim long clips; the head is utterance-level so a session would average away")
    args = ap.parse_args()

    device = torch.device(args.device)
    cases = Path(args.cases_dir)
    windows_path = Path(args.windows) if args.windows else cases / "windows.json"
    windows = {k: v for k, v in json.loads(windows_path.read_text()).items() if not k.startswith("_")}
    model = load_model(args.config_path, args.ckpt, device)

    print("clip\trole\tlabel\tpred_drr_dB\tpred_fg_dist_m\tpred_itf_dist_m\treduction_dB")
    for clip, spec in windows.items():
        raw_path = cases / f"{clip}_raw.wav"
        if not raw_path.is_file():
            continue
        wav, sr = AudioIO.open(f_path=str(raw_path), target_lvl=None, resample_to=16000)
        wav = wav.view(1, -1)[..., : int(args.max_seconds * 16000)]
        with torch.no_grad():
            out = model(wav.to(device), dry_blend=args.dry_blend).detach().cpu().view(1, -1)
            preds = getattr(model.backbone, "last_dist_preds", None)
        if preds is None:
            raise SystemExit("backbone produced no dist preds; is dist_head.enabled true?")
        p = preds.detach().cpu().reshape(-1)[:3].tolist()
        n = min(out.shape[-1], wav.shape[-1])
        red = 10.0 * math.log10(
            float(out[..., :n].square().sum()) / max(float(wav[..., :n].square().sum()), 1e-12) + 1e-12
        )
        role = "far" if "suppress" in spec else ("dt" if "_dt" in clip else "near")
        print(f"{clip}\t{role}\t{spec.get('label','')}\t{p[0]*DRR_SCALE:8.2f}\t"
              f"{10 ** p[1]:8.2f}\t{10 ** p[2]:8.2f}\t{red:8.2f}")

    print()
    print("# Reading it: on a lone far clip the FOREGROUND slot is meaningless (there is no near")
    print("#   speaker); look at pred_itf_dist_m. If the model places a 3 m bystander at ~3 m and")
    print("#   still passes it through, the cue survives and the mask is ignoring it -> couple mask")
    print("#   depth to the estimate. If it places it under a metre, the cue is gone on this capture")
    print("#   chain -> no loss or dosage change will fix it; that needs data from the chain.")


if __name__ == "__main__":
    main()
