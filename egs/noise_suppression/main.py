import argparse
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from puresound.audio.io import AudioIO
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.metrics import Metrics
from puresound.recipes import init_loss_func, init_siso_model, load_siso_recipe_config
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset
from puresound.task.voice_isolation import VoiceIsolationCollateFunc, VoiceIsolationDataset
from puresound.task.sampler import SpeakerSampler
from puresound.utils import create_folder, load_hparam


# Training uses a fixed sample length, so input tensor shapes are constant
# across steps. That makes cuDNN autotuning a pure win: it benchmarks each conv
# shape once and reuses the fastest algorithm, instead of the default heuristic
# -- which picks a pathologically slow dgrad algorithm for the dilated encoder
# convs (measured ~3.6x slower per step end to end). TF32 lets the attention /
# linear matmuls use tensor cores at negligible precision cost on Ampere+.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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
    task_name = corpus_dict.get("task", "noise_suppression")
    if task_name == "voice_isolation":
        dataset_cls = VoiceIsolationDataset
        collate_fn = VoiceIsolationCollateFunc()
    elif task_name == "noise_suppression":
        dataset_cls = NoiseSuppressionDataset
        collate_fn = NoiseSuppressionCollateFunc()
    else:
        raise ValueError(f"Unsupported dataset.task: {task_name}")

    train_dataset = dataset_cls(
        metafile_path=corpus_dict["train_metafile"],
        min_utt_length_in_seconds=corpus_dict["filter_min_utterance_length"],
        min_utts_in_each_speaker=corpus_dict["filter_min_utterance_per_speaker"],
        target_sr=corpus_dict["target_sample_rate"],
        training_sample_length_in_seconds=corpus_dict["training_length_seconds"],
        audio_gain_nomalized_to=corpus_dict["gain_nomalized_to"],
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
    )

    train_sampler = SpeakerSampler(
        data=train_dataset.meta,
        total_batch=trainer_dict["train_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
        select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=collate_fn,
    )

    valid_dataset = dataset_cls(
        metafile_path=corpus_dict["valid_metafile"],
        min_utt_length_in_seconds=corpus_dict["filter_min_utterance_length"],
        min_utts_in_each_speaker=corpus_dict["filter_min_utterance_per_speaker"],
        target_sr=corpus_dict["target_sample_rate"],
        training_sample_length_in_seconds=corpus_dict["training_length_seconds"],
        audio_gain_nomalized_to=corpus_dict["gain_nomalized_to"],
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
    )

    # Seeded sampler -> same valid batches every epoch, and per-item seeds make
    # the on-the-fly synthesis reproducible, so val metrics are comparable
    # across epochs and runs.
    valid_sampler = SpeakerSampler(
        data=valid_dataset.meta,
        total_batch=trainer_dict["valid_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
        select_by_sr_first=False if corpus_dict["target_sample_rate"] else True,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--set_seed", type=int, default=None, help="set random seed.")
    parser.add_argument(
        "--training",
        action="store_true",
        default=False,
        help="start training new model.",
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
    args = parser.parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    (
        corpus_dict,
        trainer_dict,
        optim_dict,
        scheduler_dict,
        loss_dict,
        model_dict,
        aug_speech_dict,
        aug_noise_dict,
        aug_reverb_dict,
        aug_speed_dict,
        aug_ir_dict,
        aug_src_dict,
        aug_hpf_dict,
        aug_volume_dict,
        aug_codec_dict,
        aug_packet_loss_dict,
        aug_target_absent_dict,
        vad_label_dict,
    ) = load_siso_recipe_config(args.config_path)

    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(
            corpus_dict,
            trainer_dict,
            aug_speech_dict,
            aug_noise_dict,
            aug_reverb_dict,
            aug_speed_dict,
            aug_ir_dict,
            aug_src_dict,
            aug_hpf_dict,
            aug_volume_dict,
            aug_codec_dict,
            aug_packet_loss_dict,
            aug_target_absent_dict,
            vad_label_dict,
        )

    # Stage of dump the training samples
    if args.dump_training_samples:
        create_folder(folder_name="./dummy_samples")
        dataiter = iter(train_dataloader)
        for iteration in range(3):
            file_name = f"./dummy_samples/batch_{str(iteration).zfill(2)}"
            batch = next(dataiter)
            noisy_speech = batch["noisy_speech"]
            clean_speech = batch["clean_speech"]
            noise = batch["consistency_noise"]
            for i in range(noisy_speech.shape[0]):
                AudioIO.save(
                    wav=torch.stack(
                        [noisy_speech[i], clean_speech[i], noise[i]], dim=0
                    ),
                    f_path=f"{file_name}-{str(i).zfill(2)}.wav",
                    sr=batch["sr"][i],
                )

    # Stage of training a new model
    if args.training:
        # Initialize loss function
        loss_func_list, loss_func_list_w = init_loss_func(hparam_conf=loss_dict)

        # PL-Model
        lighting_model = init_siso_model(model_dict)
        lighting_model.register_loss_func(loss_func_list, loss_func_list_w)

        # Silero VAD labels are computed batched on GPU (lifted out of the
        # DataLoader workers); the dataset emits `vad_reference` and the module
        # labels the batch in on_after_batch_transfer.
        if (
            vad_label_dict
            and vad_label_dict.get("used")
            and vad_label_dict.get("backend", "energy").lower() == "silero"
        ):
            from puresound.audio.vad import BatchedSileroVADLabeler

            lighting_model.register_gpu_vad_labeler(
                BatchedSileroVADLabeler(**vad_label_dict.get("args", {}))
            )
        param_groups = lighting_model.get_total_param_groups()
        optimizer, scheduler = create_optimizer_and_scheduler(
            overall_params_and_lr_factor=param_groups,
            optimizer_args=optim_dict,
            scheduler_args=scheduler_dict,
        )
        lighting_model.register_optimizer(optimizer)
        lighting_model.register_scheduler(scheduler)
        lighting_model.register_warmup_step(scheduler_dict["warmup_step"])

        # Loading exists state_dicts
        if args.pretrained_ckpt_path:
            print("Loading the pretrained params only.")
            state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")[
                "state_dict"
            ]
            # strict=False: warm-starting a model that ADDED params (e.g. new aux
            # heads for a curriculum stage) must keep those new params at init
            # rather than error on missing keys. Mismatches are reported.
            missing, unexpected = lighting_model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [pretrained] {len(missing)} new param(s) kept at init: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
            if unexpected:
                print(f"  [pretrained] {len(unexpected)} ckpt param(s) ignored: {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")

        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        ckpt_monitor = ModelCheckpoint(
            save_on_train_epoch_end=True, every_n_epochs=1, save_top_k=-1
        )

        # gradient_as_bucket_view=True makes DDP all-reduce read gradients in
        # place from the bucket, which removes the "grad strides do not match
        # bucket view strides" warning (triggered by cuDNN's 1x1-conv weight-grad
        # layout) and lowers memory. Only meaningful with >1 device; fall back to
        # Lightning's auto strategy otherwise.
        #
        # find_unused_parameters is config-driven (trainer.find_unused_parameters,
        # default False). Models with data-dependent parameter groups (e.g. a
        # cold-started VAD gate head that some batches never exercise) need it
        # True or DDP errors. Plain noise-suppression configs leave it False and
        # avoid the per-step graph walk.
        strategy = (
            DDPStrategy(
                gradient_as_bucket_view=True,
                find_unused_parameters=trainer_dict.get(
                    "find_unused_parameters", False
                ),
            )
            if trainer_dict["num_gpus"] > 1
            else "auto"
        )
        # Precision defaults to full precision (Lightning's 32-true) and is
        # config-driven: to trade a little accuracy for speed/memory, add
        # `precision: bf16-mixed` under trainer.lighting_trainer_args -- it
        # threads through the spread below. bf16 (not fp16) is preferred for the
        # complex-spectral magnitude/division ops (needs fp32 range, no
        # GradScaler); measured ~1.57x faster steps and ~40% less activation
        # memory on Ampere when enabled.
        trainer = L.Trainer(
            **trainer_dict["lighting_trainer_args"],
            accelerator="gpu" if trainer_dict["num_gpus"] > 0 else "cpu",
            devices=trainer_dict["num_gpus"],
            strategy=strategy,
            limit_train_batches=trainer_dict["train_iter_per_epoch"],
            limit_val_batches=trainer_dict["valid_iter_per_epoch"],
            use_distributed_sampler=False,
            default_root_dir=trainer_dict["work_folder"],
            callbacks=[lr_monitor, ckpt_monitor],
            profiler="simple",
            sync_batchnorm=True,
        )

        if args.ckpt_path is not None:
            trainer.fit(
                lighting_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=args.ckpt_path,
            )
        else:
            trainer.fit(
                lighting_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
            )

    # Stage of caculating the metric scores
    if args.scoring:
        test_dataset = KaldiFormBaseDataset(
            folder=corpus_dict["test_folder"],
            mode="dev",
            resample_to=args.inference_sr,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(inference_mode=True)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        lighting_model = init_siso_model(model_dict)
        lighting_model.reload_checkpoint(state_dict)
        lighting_model.register_metrics_func(
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
        trainer.test(lighting_model, dataloaders=test_dataloader)

    # Stage of inferencing audio only
    if args.inference:
        test_dataset = KaldiFormBaseDataset(
            folder=corpus_dict["test_folder"],
            mode="eval",
            resample_to=args.inference_sr,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(
            inference_mode=True, default_root_dir=corpus_dict["proc_output_folder"]
        )
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        lighting_model = init_siso_model(model_dict)
        lighting_model.reload_checkpoint(state_dict)
        create_folder(corpus_dict["proc_output_folder"])
        lighting_model.register_proc_output_folder(corpus_dict["proc_output_folder"])
        trainer.predict(lighting_model, dataloaders=test_dataloader)
