"""noise_suppression -- single-channel speech enhancement, training entry point.

One speaker, noise and reverberation removed, no notion of distance: the target is
whatever speech the row was built from. The task-specific part is
``NoiseSuppressionDataset``; everything downstream lives in
``puresound.system.runner``.

For the near-field isolation task -- keep the speaker inside ~1 m, suppress voices
beyond it -- use ``egs/voice_isolate/main.py`` instead. That recipe has its own dataset
and its own real-recording row types; this one will refuse its configs.

Run from this directory -- the config's metafile and work-folder paths are relative to
it::

    cd egs/noise_suppression
    uv run python main.py config/dpcrn.yaml --training
    uv run python main.py config/dpcrn.yaml --dump_training_samples
"""

from pathlib import Path
import sys

import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.config import load_recipe  # noqa: E402
from puresound.system import runner  # noqa: E402
from puresound.task.ns import (  # noqa: E402
    NoiseSuppressionCollateFunc,
    NoiseSuppressionDataset,
)

TASK_NAME = "noise_suppression"

runner.configure_torch_backends()


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe."""
    return runner.build_dataloaders(
        dataset_cls=NoiseSuppressionDataset,
        collate_fn=NoiseSuppressionCollateFunc(),
        recipe=recipe,
    )


if __name__ == "__main__":
    args = runner.build_arg_parser(__doc__.splitlines()[0]).parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    # The loader rejects another recipe's config by name, and the
    # NoiseSuppressionRecipe model has no realfar/realnear fields at all, so the
    # hand-written guards these lines used to carry are now schema-level.
    recipe = load_recipe(
        args.config_path, expected_task=TASK_NAME, expected_purpose="train"
    )

    train_dataloader = valid_dataloader = None
    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(recipe)

    runner.run_stages(args, recipe, train_dataloader, valid_dataloader)
