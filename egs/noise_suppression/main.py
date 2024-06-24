import argparse
from typing import Dict, List

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from puresound import nnet, system
from puresound.audio.io import AudioIO
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.metrics import Metrics
from puresound.nnet import loss as ploss
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset
from puresound.task.sampler import SpeakerSampler
from puresound.utils import create_folder, load_hparam, str2bool


def load_config(f_path: str):
    config = load_hparam(file_path=f_path)
    corpus_dict = config["dataset"]
    trainer_dict = config["trainer"]
    optim_dict = config["optimizer"]
    scheduler_dict = config["scheduler"]
    loss_dict = config["loss_func"]
    model_dict = config["model"]

    aug_speech_dict = None
    aug_noise_dict = None
    aug_reverb_dict = None
    aug_speed_dict = None
    aug_ir_dict = None
    aug_src_dict = None
    aug_hpf_dict = None
    aug_volume_dict = None

    if "augmentation_speech" in config:
        aug_speech_dict = config["augmentation_speech"]
    if "augmentation_noise" in config:
        aug_noise_dict = config["augmentation_noise"]
    if "augmentation_speech" in config:
        aug_reverb_dict = config["augmentation_reverb"]
    if "augmentation_speech" in config:
        aug_speed_dict = config["augmentation_speed"]
    if "augmentation_speech" in config:
        aug_ir_dict = config["augmentation_ir_response"]
    if "augmentation_speech" in config:
        aug_src_dict = config["augmentation_src"]
    if "augmentation_hpf" in config:
        aug_hpf_dict = config["augmentation_hpf"]
    if "augmentation_volume" in config:
        aug_volume_dict = config["augmentation_volume"]

    return (
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
    )


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
):
    train_dataset = NoiseSuppressionDataset(
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
    )

    train_sampler = SpeakerSampler(
        data=train_dataset.meta,
        total_batch=trainer_dict["train_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=NoiseSuppressionCollateFunc(),
    )

    valid_dataset = NoiseSuppressionDataset(
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
    )

    valid_sampler = SpeakerSampler(
        data=valid_dataset.meta,
        total_batch=trainer_dict["valid_iter_per_epoch"],
        n_spks=trainer_dict["n_spk_per_batch"],
        n_per=trainer_dict["n_utt_per_speaker"],
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_sampler=valid_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=NoiseSuppressionCollateFunc(),
    )

    return train_dataloader, valid_dataloader


def init_model(model_dict):
    lighting_module = getattr(system, model_dict["lighting_module"]["type"])
    encoder = getattr(nnet, model_dict["encoder"]["type"])(
        **model_dict["encoder"]["encoder_args"]
    )
    feature_encoder = nnet.FeatureEncoder(**model_dict["features"])
    backbone = getattr(nnet, model_dict["backbone"]["type"])(
        **model_dict["backbone"]["backbone_args"]
    )
    model = lighting_module(
        encoder,
        feature_encoder,
        backbone,
        **model_dict["lighting_module"]["module_args"],
    )
    return model


def init_loss_func(hparam_conf: List):
    """
    Returns:
        loss_list contain [[loss_1, weighted_1], [loss_2, weighted_2], ...]
    """
    loss_list = []
    for item in hparam_conf:
        loss_func = getattr(ploss, item["type"])(**item["args"])
        loss_list.append([loss_func, item["weighted"]])

    return loss_list


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--set_seed", type=int, default=None)
    parser.add_argument("--training", type=str2bool, default=False)
    parser.add_argument("--scoring", type=str2bool, default=False)
    parser.add_argument("--inference", type=str2bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dump_training_samples", type=str2bool, default=False)
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
    ) = load_config(args.config_path)

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
        )

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
                    sr=corpus_dict["target_sample_rate"],
                )

    if args.training:
        # Initialize loss function
        loss_func_list = init_loss_func(hparam_conf=loss_dict)
        lighting_model = init_model(model_dict)
        lighting_model.register_loss_func(loss_func_list)
        param_groups = lighting_model.get_total_param_groups()
        optimizer, scheduler = create_optimizer_and_scheduler(
            overall_params_and_lr_factor=param_groups,
            optimizer_args=optim_dict,
            scheduler_args=scheduler_dict,
        )
        lighting_model.register_optimizer(optimizer)
        lighting_model.register_scheduler(scheduler)
        lighting_model.register_warmup_step(scheduler_dict["warmup_step"])

        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        ckpt_monitor = ModelCheckpoint(
            save_on_train_epoch_end=True, every_n_epochs=1, save_top_k=-1
        )

        trainer = L.Trainer(
            **trainer_dict["lighting_trainer_args"],
            accelerator="gpu" if trainer_dict["num_gpus"] > 0 else "cpu",
            devices=trainer_dict["num_gpus"],
            limit_train_batches=trainer_dict["train_iter_per_epoch"],
            limit_val_batches=trainer_dict["valid_iter_per_epoch"],
            use_distributed_sampler=False,
            default_root_dir=trainer_dict["work_folder"],
            callbacks=[lr_monitor, ckpt_monitor],
            profiler="simple",
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

    if args.scoring:
        test_dataset = KaldiFormBaseDataset(folder=corpus_dict["test_folder"])
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer()
        lighting_model = init_model(model_dict)
        # ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # lighting_model.load_state_dict(ckpt["state_dict"])
        lighting_model.register_metrics_func(
            {
                "pesq_wb": {"func": Metrics.pesq_wb, "sr": 16000},
                "pesq_nb": {"func": Metrics.pesq_nb, "sr": 8000},
                "stoi": {"func": Metrics.stoi, "sr": 16000},
                "sisnr": {"func": Metrics.sisnr, "sr": None},
                "bss_sdr": {"func": Metrics.bss_sdr, "sr": None},
            }
        )
        trainer.test(
            lighting_model, dataloaders=test_dataloader, ckpt_path=args.ckpt_path
        )

    if args.inference:
        test_dataset = KaldiFormBaseDataset(folder=corpus_dict["test_folder"])
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer()
        lighting_model = init_model(model_dict)
        # ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # lighting_model.load_state_dict(ckpt["state_dict"])
        create_folder(corpus_dict["proc_output_folder"])
        lighting_model.register_proc_output_folder(corpus_dict["proc_output_folder"])
        trainer.predict(
            lighting_model, dataloaders=test_dataloader, ckpt_path=args.ckpt_path
        )
