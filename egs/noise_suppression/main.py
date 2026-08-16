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
from typing import Dict

import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.system import runner  # noqa: E402
from puresound.task.ns import (  # noqa: E402
    NoiseSuppressionCollateFunc,
    NoiseSuppressionDataset,
)

TASK_NAME = "noise_suppression"

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
):
    """Train / valid dataloaders for this recipe."""
    return runner.build_dataloaders(
        dataset_cls=NoiseSuppressionDataset,
        collate_fn=NoiseSuppressionCollateFunc(),
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
            "recipe. Voice-isolation configs run with egs/voice_isolate/main.py."
        )
    if (cfg.aug_realfar and cfg.aug_realfar.get("used")) or (
        cfg.aug_realnear and cfg.aug_realnear.get("used")
    ):
        raise SystemExit(
            "augmentation_realfar/realnear are voice-isolation row types; run this "
            "config with egs/voice_isolate/main.py."
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
        )

    runner.run_stages(args, cfg, train_dataloader, valid_dataloader)
