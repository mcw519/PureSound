"""Training / scoring / inference driver shared by the SISO recipes.

A recipe's ``main.py`` says *what* it trains -- which dataset class, which collate
function, and any arguments only its own dataset takes. Everything downstream of that
choice is identical between recipes: the sampler, the two dataloaders, the CLI, the
Lightning module wiring, DDP strategy, checkpoint callbacks, and the scoring and
inference stages. That part lives here so the recipes stay independent of each other
without each carrying its own copy to drift.

The split is deliberate. A recipe owns its task; it does not own argparse.

Wiring a recipe::

    from puresound.system import runner

    def init_dataloader(corpus_dict, trainer_dict, ...):
        return runner.build_dataloaders(
            dataset_cls=MyDataset, collate_fn=MyCollateFunc(),
            corpus_dict=corpus_dict, trainer_dict=trainer_dict, ...
        )

    if __name__ == "__main__":
        args = runner.build_arg_parser("...").parse_args()
        cfg = runner.RecipeConfig.load(args.config_path)
        runner.run_stages(args, cfg, *(init_dataloader(...) if training else (None, None)))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from puresound.audio.io import AudioIO
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.metrics import Metrics
from puresound.recipes import init_loss_func, init_siso_model, load_siso_recipe_config
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.sampler import SpeakerSampler
from puresound.utils import create_folder


def configure_torch_backends() -> None:
    """Enable cuDNN autotuning and TF32.

    Training uses a fixed sample length, so input tensor shapes are constant across
    steps. That makes cuDNN autotuning a pure win: it benchmarks each conv shape once
    and reuses the fastest algorithm, instead of the default heuristic -- which picks a
    pathologically slow dgrad algorithm for the dilated encoder convs (measured ~3.6x
    slower per step end to end). TF32 lets the attention / linear matmuls use tensor
    cores at negligible precision cost on Ampere+.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class RecipeConfig:
    """Named view over ``load_siso_recipe_config``'s positional tuple.

    That function stays as it is -- ten scripts unpack its tuple -- but a recipe reading
    twenty positional fields cannot be checked by eye, so the driver uses names.
    """

    corpus: Dict
    trainer: Dict
    optimizer: Dict
    scheduler: Dict
    loss: Any
    model: Dict
    aug_speech: Optional[Dict]
    aug_noise: Optional[Dict]
    aug_reverb: Optional[Dict]
    aug_speed: Optional[Dict]
    aug_ir: Optional[Dict]
    aug_src: Optional[Dict]
    aug_hpf: Optional[Dict]
    aug_volume: Optional[Dict]
    aug_codec: Optional[Dict]
    aug_packet_loss: Optional[Dict]
    aug_target_absent: Optional[Dict]
    vad_label: Optional[Dict]
    aug_realfar: Optional[Dict] = None
    aug_realnear: Optional[Dict] = None

    @classmethod
    def load(cls, config_path: str) -> "RecipeConfig":
        return cls(*load_siso_recipe_config(config_path))

    @property
    def task(self) -> str:
        return self.corpus.get("task", "noise_suppression")


def build_dataloaders(
    *,
    dataset_cls,
    collate_fn,
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
    task_kwargs: Optional[Dict] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train and validation dataloaders for a dynamic-synthesis dataset.

    ``task_kwargs`` carries the arguments only one task's dataset accepts (the
    real-recording pools, for instance); everything else is common.
    """
    common = dict(
        min_utt_length_in_seconds=corpus_dict["filter_min_utterance_length"],
        min_utts_in_each_speaker=corpus_dict["filter_min_utterance_per_speaker"],
        target_sr=corpus_dict["target_sample_rate"],
        training_sample_length_in_seconds=corpus_dict["training_length_seconds"],
        audio_gain_normalized_to=corpus_dict["gain_normalized_to"],
        augmentation_speech_args=aug_speech_dict,
        augmentation_noise_args=aug_noise_dict,
        augmentation_reverb_args=aug_reverb_dict,
        augmentation_speed_args=aug_speed_dict,
        augmentation_ir_response_args=aug_ir_dict,
        augmentation_src_args=aug_src_dict,
        augmentation_hpf_args=aug_hpf_dict,
        augmentation_volume_args=aug_volume_dict,
        augmentation_codec_args=aug_codec_dict,
        augmentation_packet_loss_args=aug_packet_loss_dict,
        augmentation_target_absent_args=aug_target_absent_dict,
        vad_label_args=vad_label_dict,
        **(task_kwargs or {}),
    )
    select_by_sr_first = False if corpus_dict["target_sample_rate"] else True

    train_dataset = dataset_cls(metafile_path=corpus_dict["train_metafile"], **common)
    train_sampler = SpeakerSampler(
        data=train_dataset.meta,
        total_batch=trainer_dict["train_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
        select_by_sr_first=select_by_sr_first,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=collate_fn,
    )

    valid_dataset = dataset_cls(metafile_path=corpus_dict["valid_metafile"], **common)
    # Seeded sampler -> same valid batches every epoch, and per-item seeds make the
    # on-the-fly synthesis reproducible, so val metrics are comparable across epochs
    # and runs.
    valid_sampler = SpeakerSampler(
        data=valid_dataset.meta,
        total_batch=trainer_dict["valid_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
        select_by_sr_first=select_by_sr_first,
        seed=trainer_dict.get("valid_seed", 1234),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_sampler=valid_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config_path", type=str)
    parser.add_argument("--set_seed", type=int, default=None, help="set random seed.")
    parser.add_argument(
        "--training", action="store_true", default=False, help="start training new model."
    )
    parser.add_argument(
        "--scoring", action="store_true", default=False, help="compute metrics."
    )
    parser.add_argument(
        "--inference", action="store_true", default=False, help="inference audios."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="choose a exist checkpoint for resume trainig, caculte scores or inferencing.",
    )
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=str,
        default=None,
        help="choose a exist checkpoint for training which replacing from scratch, \
            and with new optimizer, learning rate scheduler and loss etc.",
    )
    parser.add_argument(
        "--dump_training_samples",
        action="store_true",
        default=False,
        help="generate some training samples.",
    )
    parser.add_argument(
        "--inference_sr",
        type=int,
        default=None,
        help="If given, all processs would work on this sr.",
    )
    return parser


def dump_training_samples(train_dataloader, out_folder: str = "./dummy_samples", n_batches: int = 3) -> None:
    """Write a few synthesised batches to disk -- the fastest way to hear a recipe."""
    create_folder(folder_name=out_folder)
    dataiter = iter(train_dataloader)
    for iteration in range(n_batches):
        file_name = f"{out_folder}/batch_{str(iteration).zfill(2)}"
        batch = next(dataiter)
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        noise = batch["consistency_noise"]
        for i in range(noisy_speech.shape[0]):
            AudioIO.save(
                wav=torch.stack([noisy_speech[i], clean_speech[i], noise[i]], dim=0),
                f_path=f"{file_name}-{str(i).zfill(2)}.wav",
                sr=batch["sr"][i],
            )


def run_training(args, cfg: RecipeConfig, train_dataloader, valid_dataloader) -> None:
    loss_func_list, loss_func_list_w = init_loss_func(hparam_conf=cfg.loss)

    lightning_model = init_siso_model(cfg.model)
    lightning_model.register_loss_func(loss_func_list, loss_func_list_w)

    # Silero VAD labels are computed batched on GPU (lifted out of the DataLoader
    # workers); the dataset emits `vad_reference` and the module labels the batch in
    # on_after_batch_transfer.
    if (
        cfg.vad_label
        and cfg.vad_label.get("used")
        and cfg.vad_label.get("backend", "energy").lower() == "silero"
    ):
        from puresound.audio.vad import BatchedSileroVADLabeler

        lightning_model.register_gpu_vad_labeler(
            BatchedSileroVADLabeler(**cfg.vad_label.get("args", {}))
        )

    param_groups = lightning_model.get_total_param_groups()
    optimizer, scheduler = create_optimizer_and_scheduler(
        overall_params_and_lr_factor=param_groups,
        optimizer_args=cfg.optimizer,
        scheduler_args=cfg.scheduler,
    )
    lightning_model.register_optimizer(optimizer)
    lightning_model.register_scheduler(scheduler)
    lightning_model.register_warmup_step(cfg.scheduler["warmup_step"])

    if args.pretrained_ckpt_path:
        print("Loading the pretrained params only.")
        state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")["state_dict"]
        # strict=False: warm-starting a model that ADDED params (e.g. new aux heads for
        # a curriculum stage) must keep those new params at init rather than error on
        # missing keys. Mismatches are reported.
        missing, unexpected = lightning_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [pretrained] {len(missing)} new param(s) kept at init: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
        if unexpected:
            print(f"  [pretrained] {len(unexpected)} ckpt param(s) ignored: {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    ckpt_monitor = ModelCheckpoint(
        save_on_train_epoch_end=True, every_n_epochs=1, save_top_k=-1
    )

    # gradient_as_bucket_view=True makes DDP all-reduce read gradients in place from the
    # bucket, which removes the "grad strides do not match bucket view strides" warning
    # (triggered by cuDNN's 1x1-conv weight-grad layout) and lowers memory. Only
    # meaningful with >1 device; fall back to Lightning's auto strategy otherwise.
    #
    # find_unused_parameters is config-driven (trainer.find_unused_parameters, default
    # False). Models with data-dependent parameter groups (e.g. a cold-started VAD gate
    # head that some batches never exercise) need it True or DDP errors. Plain
    # noise-suppression configs leave it False and avoid the per-step graph walk.
    strategy = (
        DDPStrategy(
            gradient_as_bucket_view=True,
            find_unused_parameters=cfg.trainer.get("find_unused_parameters", False),
        )
        if cfg.trainer["num_gpus"] > 1
        else "auto"
    )
    # Precision defaults to full precision (Lightning's 32-true) and is config-driven:
    # to trade a little accuracy for speed/memory, add `precision: bf16-mixed` under
    # trainer.lightning_trainer_args -- it threads through the spread below. bf16 (not
    # fp16) is preferred for the complex-spectral magnitude/division ops (needs fp32
    # range, no GradScaler); measured ~1.57x faster steps and ~40% less activation
    # memory on Ampere when enabled.
    trainer = L.Trainer(
        **cfg.trainer["lightning_trainer_args"],
        accelerator="gpu" if cfg.trainer["num_gpus"] > 0 else "cpu",
        devices=cfg.trainer["num_gpus"],
        strategy=strategy,
        limit_train_batches=cfg.trainer["train_iter_per_epoch"],
        limit_val_batches=cfg.trainer["valid_iter_per_epoch"],
        use_distributed_sampler=False,
        default_root_dir=cfg.trainer["work_folder"],
        callbacks=[lr_monitor, ckpt_monitor],
        profiler="simple",
        sync_batchnorm=True,
    )

    fit_kwargs = {"ckpt_path": args.ckpt_path} if args.ckpt_path is not None else {}
    trainer.fit(
        lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        **fit_kwargs,
    )


def _test_dataloader(cfg: RecipeConfig, mode: str, resample_to: Optional[int]):
    dataset = KaldiFormBaseDataset(
        folder=cfg.corpus["test_folder"], mode=mode, resample_to=resample_to
    )
    return torch.utils.data.DataLoader(
        dataset=dataset, pin_memory=True, num_workers=4, batch_size=1, shuffle=False
    )


def run_scoring(args, cfg: RecipeConfig) -> None:
    test_dataloader = _test_dataloader(cfg, "dev", args.inference_sr)
    trainer = L.Trainer(inference_mode=True)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    lightning_model = init_siso_model(cfg.model)
    lightning_model.reload_checkpoint(state_dict)
    lightning_model.register_metrics_func(
        {
            "pesq_wb": {"func": Metrics.pesq_wb, "sr": 16000},
            "pesq_nb": {"func": Metrics.pesq_nb, "sr": 8000},
            "stoi": {"func": Metrics.stoi, "sr": None},
            "estoi": {"func": Metrics.estoi, "sr": None},
            "sisnr": {"func": Metrics.sisnr, "sr": None},
            "bss_sdr": {"func": Metrics.bss_sdr, "sr": None},
            "dnsmos_p835": {"func": Metrics.dnsmos_p835, "sr": 16000},
        }
    )
    trainer.test(lightning_model, dataloaders=test_dataloader)


def run_inference(args, cfg: RecipeConfig) -> None:
    test_dataloader = _test_dataloader(cfg, "eval", args.inference_sr)
    trainer = L.Trainer(
        inference_mode=True, default_root_dir=cfg.corpus["proc_output_folder"]
    )
    state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    lightning_model = init_siso_model(cfg.model)
    lightning_model.reload_checkpoint(state_dict)
    create_folder(cfg.corpus["proc_output_folder"])
    lightning_model.register_proc_output_folder(cfg.corpus["proc_output_folder"])
    trainer.predict(lightning_model, dataloaders=test_dataloader)


def run_stages(args, cfg: RecipeConfig, train_dataloader=None, valid_dataloader=None) -> None:
    """Run whichever stages the CLI asked for, in the order they depend on each other."""
    if args.dump_training_samples:
        dump_training_samples(train_dataloader)
    if args.training:
        run_training(args, cfg, train_dataloader, valid_dataloader)
    if args.scoring:
        run_scoring(args, cfg)
    if args.inference:
        run_inference(args, cfg)
