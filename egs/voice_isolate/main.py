"""voice_isolate -- near-field foreground voice isolation, training entry point.

Keep the speaker within ~1 m of the microphone, suppress every voice beyond it plus
noise, with no enrollment. The task-specific part of this recipe is the dataset:
``VoiceIsolationDataset`` synthesises each row from a near channel and one or more far
channels, and it accepts two row types no other recipe has -- the real far-field
(``augmentation_realfar``) and real near-field (``augmentation_realnear``) recording
pools. Everything downstream of that lives in ``puresound.system.runner``.

Run from this directory -- the config's metafile and work-folder paths are relative to
it::

    cd egs/voice_isolate
    uv run python main.py config/train_dpcrn.yaml --training
    uv run python main.py config/train_dpcrn.yaml --training \
        --pretrained_ckpt_path pretrained_ckpt/dpcrn_v8.ckpt
    uv run python main.py config/train_dpcrn.yaml --dump_training_samples
"""

from pathlib import Path
import sys
import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.config import load_recipe  # noqa: E402
from puresound.system import runner  # noqa: E402
from puresound.task.voice_isolation import (  # noqa: E402
    VoiceIsolationCollateFunc,
    VoiceIsolationDataset,
)

TASK_NAME = "voice_isolation"

runner.configure_torch_backends()


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe.

    Part of the recipe's surface: scripts/ imports it to rebuild the exact
    training distribution for evaluation and data audits. It takes the typed
    recipe -- the sixteen positional dicts it used to take were the config
    tunnel, and the recipe now carries its own names.
    """
    return runner.build_dataloaders(
        dataset_cls=VoiceIsolationDataset,
        collate_fn=VoiceIsolationCollateFunc(),
        recipe=recipe,
    )


if __name__ == "__main__":
    args = runner.build_arg_parser(__doc__.splitlines()[0]).parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    # expected_task makes the loader reject another recipe's config by name,
    # so the mismatch is caught before anything is built.
    recipe = load_recipe(
        args.config_path, expected_task=TASK_NAME, expected_purpose="train"
    )

    train_dataloader = valid_dataloader = None
    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(recipe)

    runner.run_stages(args, recipe, train_dataloader, valid_dataloader)
