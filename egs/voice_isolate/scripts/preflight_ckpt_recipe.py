"""Prove a checkpoint loads WHOLE into every recipe a benchmark will use.

`reload_checkpoint` (and `load_state_dict(strict=False)`) keep going when a
recipe and a checkpoint disagree: keys the model does not have are warned about
and dropped, and modules the checkpoint does not carry stay at their random
init. Either way the run produces numbers -- for a model nobody trained. This
preflight is strict by default. Warm-starts must explicitly name any missing
heads with --allow-missing-head; existing weights must still match shape.

    uv run python scripts/preflight_ckpt_recipe.py --ckpt CKPT CONFIG [CONFIG ...]

Exit 1 if a recipe would leave a parameter at its init, OR if it drops trained
backbone weights on the floor -- the second direction is the one that bites when
a new module is added, because every LSTM-era recipe loads such a checkpoint
"successfully" while ignoring the new branch. Dropped auxiliary heads are exempt
because inference recipes may omit them; missing heads are never exempt unless
explicitly allowed for a warm-start. Loss-function buffers are ignored for the same reason.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from puresound.config import load_recipe
from puresound.recipes import init_siso_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--allow-dropped", action="store_true",
                    help="downgrade dropped backbone weights to a warning")
    ap.add_argument("--allow-missing-head", action="append", default=[],
                    choices=("vad_head", "background_vad_head", "dist_head",
                             "identity_head", "proximity_head"),
                    help="warm-start only: allow this head to initialise (repeatable)")
    args = ap.parse_args()
    allowed_missing = tuple(f"backbone.{head}." for head in args.allow_missing_head)

    # Inference recipes may intentionally leave these side heads unbuilt.
    # This dropped-key exemption does not authorise missing recipe weights.
    new_heads = (
        "backbone.vad_head.", "backbone.background_vad_head.",
        "backbone.dist_head.", "backbone.identity_head.",
        "backbone.proximity_head.",
    )

    blob = torch.load(args.ckpt, map_location="cpu")
    ckpt = blob.get("state_dict", blob)
    bad = 0
    for rel in args.configs:
        model = init_siso_model(load_recipe(rel, expected_task="voice_isolation").model)
        keys = set(model.state_dict())
        missing = sorted(k for k in keys if k not in ckpt)
        missing_new = [k for k in missing if k.startswith(allowed_missing)]
        missing_required = [k for k in missing if not k.startswith(allowed_missing)]
        mismatched = sorted(
            k for k in keys & set(ckpt)
            if tuple(model.state_dict()[k].shape) != tuple(ckpt[k].shape)
        )
        dropped = sorted(k for k in ckpt
                         if k not in keys and k.startswith("backbone.")
                         and not k.startswith(new_heads))
        status = "ok"
        if missing_required:
            status = f"MISSING REQUIRED {len(missing_required)}: {missing_required[:3]}"
            bad += 1
        elif mismatched:
            status = f"SHAPE MISMATCH {len(mismatched)}: {mismatched[:3]}"
            bad += 1
        elif dropped:
            status = f"DROPS {len(dropped)} trained: {dropped[:3]}"
            bad += 0 if args.allow_dropped else 1
        if status == "ok" and missing_new:
            status = f"ok (explicitly allowed heads initialise: {len(missing_new)})"
        print(f"  {pathlib.Path(rel).name:42s} {status}")
    if bad:
        print(f"\npreflight FAILED for {bad} recipe(s): the benchmark would score a "
              f"model that is not the checkpoint.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
