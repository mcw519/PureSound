"""Prove a checkpoint loads WHOLE into every recipe a benchmark will use.

`reload_checkpoint` (and `load_state_dict(strict=False)`) keep going when a
recipe and a checkpoint disagree: keys the model does not have are warned about
and dropped, and modules the checkpoint does not carry stay at their random
init. Either way the run produces numbers -- for a model nobody trained. That
failure has already cost this project one round, so a benchmark that spans six
recipes checks all of them before it spends a GPU hour.

    uv run python scripts/preflight_ckpt_recipe.py --ckpt CKPT CONFIG [CONFIG ...]

Exit 1 if a recipe would leave a parameter at its init, OR if it drops trained
backbone weights on the floor -- the second direction is the one that bites when
a new module is added, because every LSTM-era recipe loads such a checkpoint
"successfully" while ignoring the new branch. The auxiliary heads are exempt:
they are training-time / side-information modules that inference recipes are
meant not to build. Loss-function buffers are ignored for the same reason.
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
    args = ap.parse_args()

    # Modules a benchmark recipe is SUPPOSED to leave unbuilt.
    aux = ("backbone.vad_head.", "backbone.background_vad_head.", "backbone.dist_head.")

    blob = torch.load(args.ckpt, map_location="cpu")
    ckpt = blob.get("state_dict", blob)
    bad = 0
    for rel in args.configs:
        model = init_siso_model(load_recipe(rel, expected_task="voice_isolation").model)
        keys = set(model.state_dict())
        missing = sorted(k for k in keys if k not in ckpt)
        dropped = sorted(k for k in ckpt
                         if k not in keys and k.startswith("backbone.")
                         and not k.startswith(aux))
        status = "ok"
        if missing:
            status = f"MISSING {len(missing)}: {missing[:3]}"
            bad += 1
        elif dropped:
            status = f"DROPS {len(dropped)} trained: {dropped[:3]}"
            bad += 0 if args.allow_dropped else 1
        print(f"  {pathlib.Path(rel).name:42s} {status}")
    if bad:
        print(f"\npreflight FAILED for {bad} recipe(s): the benchmark would score a "
              f"model that is not the checkpoint.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
