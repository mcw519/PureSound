"""target_speaker_extraction -- extract one enrolled speaker, training entry point.

Two branches: the mixture goes through the separation path, an enrollment
utterance goes through a conditioning path, and the conditioning embedding steers
the mask. Its dataset is ``TargetSpeakerExtractDataset`` and its module is a
conditioned (MISO) one; every stage downstream lives in
``puresound.system.runner``, the same one the single-branch recipes use.

Run from this directory -- the config's metafile and work-folder paths are
relative to it::

    cd egs/target_speaker_extraction
    uv run python main.py config/default_config.yaml --training
    uv run python main.py config/default_config.yaml --inference \
        --ckpt_path path/to.ckpt
"""

from pathlib import Path
import sys

import lightning as L
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.config import load_recipe  # noqa: E402
from puresound.system import runner  # noqa: E402
from puresound.task.tse import (  # noqa: E402
    TargetSpeakerExtractCollateFunc,
    TargetSpeakerExtractDataset,
)

TASK_NAME = "target_speaker_extraction"

#: The eval corpus carries an enrollment list alongside the audio; the
#: single-branch tasks read nothing extra.
FOLDER_CONTENT = {"wav2enroll": "wav2enroll.txt"}

runner.configure_torch_backends()


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe."""
    return runner.build_dataloaders(
        dataset_cls=TargetSpeakerExtractDataset,
        collate_fn=TargetSpeakerExtractCollateFunc(),
        recipe=recipe,
        task_kwargs={"enroll_speech_args": recipe.enroll_speech},
    )


def write_batch(batch, file_name):
    """Mixture / target / enrollment. The enrollment is a different utterance of
    a different length, so it cannot be stacked with the pair -- separate file."""
    noisy_speech = batch["noisy_speech"]
    for i in range(noisy_speech.shape[0]):
        AudioIO.save(
            wav=torch.stack([noisy_speech[i], batch["clean_speech"][i]], dim=0),
            f_path=f"{file_name}-{str(i).zfill(2)}.wav",
            sr=batch["sr"][i],
        )
        AudioIO.save(
            wav=batch["enroll_speech"][i : i + 1],
            f_path=f"{file_name}-{str(i).zfill(2)}-enroll.wav",
            sr=batch["sr"][i],
        )


if __name__ == "__main__":
    args = runner.build_arg_parser(__doc__.splitlines()[0]).parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    recipe = load_recipe(
        args.config_path, expected_task=TASK_NAME, expected_purpose="train"
    )

    train_dataloader = valid_dataloader = None
    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(recipe)

    # No `init_model=`: `run_stages` resolves the conditioned module from
    # `recipe.task`, so this entry point cannot pass the wrong one.
    runner.run_stages(
        args,
        recipe,
        train_dataloader,
        valid_dataloader,
        write_batch=write_batch,
        folder_content=FOLDER_CONTENT,
    )
