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
from typing import Dict

import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.system import runner  # noqa: E402
from puresound.task.voice_isolation import (  # noqa: E402
    VoiceIsolationCollateFunc,
    VoiceIsolationDataset,
)

TASK_NAME = "voice_isolation"

runner.configure_torch_backends()


def init_dataloader(
    corpus_dict: Dict,
    trainer_dict: Dict,
    aug_speech_dict: Dict,
    aug_noise_dict: Dict,
    aug_reverb_dict: Dict,
    aug_speed_dict: Dict,
    aug_ir_dict: Dict,
    aug_src_dict: Dict,
    aug_hpf_dict: Dict,
    aug_volume_dict: Dict,
    aug_codec_dict: Dict,
    aug_packet_loss_dict: Dict,
    aug_target_absent_dict: Dict,
    vad_label_dict: Dict,
    aug_realfar_dict: Dict = None,
    aug_realnear_dict: Dict = None,
):
    """Train / valid dataloaders for this recipe.

    The positional signature is part of the recipe's surface: scripts/ imports this to
    rebuild the exact training distribution for evaluation and data audits.
    """
    return runner.build_dataloaders(
        dataset_cls=VoiceIsolationDataset,
        collate_fn=VoiceIsolationCollateFunc(),
        corpus_dict=corpus_dict,
        trainer_dict=trainer_dict,
        aug_speech_dict=aug_speech_dict,
        aug_noise_dict=aug_noise_dict,
        aug_reverb_dict=aug_reverb_dict,
        aug_speed_dict=aug_speed_dict,
        aug_ir_dict=aug_ir_dict,
        aug_src_dict=aug_src_dict,
        aug_hpf_dict=aug_hpf_dict,
        aug_volume_dict=aug_volume_dict,
        aug_codec_dict=aug_codec_dict,
        aug_packet_loss_dict=aug_packet_loss_dict,
        aug_target_absent_dict=aug_target_absent_dict,
        vad_label_dict=vad_label_dict,
        task_kwargs={
            "augmentation_realfar_args": aug_realfar_dict,
            "augmentation_realnear_args": aug_realnear_dict,
        },
    )


if __name__ == "__main__":
    args = runner.build_arg_parser(__doc__.splitlines()[0]).parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    cfg = runner.RecipeConfig.load(args.config_path)
    if cfg.task != TASK_NAME:
        raise SystemExit(
            f"{args.config_path} declares dataset.task: {cfg.task!r}, which is not this "
            "recipe. Noise-suppression configs run with egs/noise_suppression/main.py."
        )

    train_dataloader = valid_dataloader = None
    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(
            cfg.corpus,
            cfg.trainer,
            cfg.aug_speech,
            cfg.aug_noise,
            cfg.aug_reverb,
            cfg.aug_speed,
            cfg.aug_ir,
            cfg.aug_src,
            cfg.aug_hpf,
            cfg.aug_volume,
            cfg.aug_codec,
            cfg.aug_packet_loss,
            cfg.aug_target_absent,
            cfg.vad_label,
            cfg.aug_realfar,
            cfg.aug_realnear,
        )

    runner.run_stages(args, cfg, train_dataloader, valid_dataloader)
