import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from puresound import nnet, system
from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.metrics import Metrics
from puresound.recipes import init_loss_func
from puresound.system import runner
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.tse import TargetSpeakerExtractCollateFunc, TargetSpeakerExtractDataset
from puresound.utils import create_folder, str2bool


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe.

    Delegates to ``runner.build_dataloaders``; this file used to carry its own
    copy. ``enroll_speech`` is the one argument the shared builder does not know
    about, so it rides in through ``task_kwargs``.
    """
    return runner.build_dataloaders(
        dataset_cls=TargetSpeakerExtractDataset,
        collate_fn=TargetSpeakerExtractCollateFunc(),
        recipe=recipe,
        task_kwargs={"enroll_speech_args": recipe.enroll_speech},
    )


def init_model(model_conf):
    lightning_module = getattr(system, model_conf["lightning_module"]["type"])

    encoder = getattr(nnet, model_conf["encoder"]["type"])(
        **model_conf["encoder"]["encoder_args"]
    )
    if not model_conf["lightning_module"]["module_args"]["siamese_encoder"]:
        c_encoder = getattr(nnet, model_conf["c_encoder"]["type"])(
            **model_conf["c_encoder"]["encoder_args"]
        )
    else:
        c_encoder = None

    feature_args = dict(model_conf["features"])
    if "freq_eq" in model_conf:
        peq_module = getattr(nnet, model_conf["freq_eq"]["type"])(
            **model_conf["freq_eq"]["eq_args"]
        )
        # register peq inside the feature module
        feature_args["peq_module"] = peq_module

    feature_encoder = nnet.FeatureEncoder(**feature_args)
    if not model_conf["lightning_module"]["module_args"]["siamese_encoder"]:
        c_feature_encoder = nnet.FeatureEncoder(**model_conf["c_features"])
    else:
        c_feature_encoder = None

    backbone = getattr(nnet, model_conf["backbone"]["type"])(
        **model_conf["backbone"]["backbone_args"]
    )
    c_backbone = getattr(nnet, model_conf["c_backbone"]["type"])(
        **model_conf["c_backbone"]["backbone_args"]
    )

    model = lightning_module(
        encoder,
        feature_encoder,
        backbone,
        c_backbone,
        c_encoder=c_encoder,
        c_feats=c_feature_encoder,
        **model_conf["lightning_module"]["module_args"],
    )
    return model


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--set_seed", type=int, default=None, help="set random seed.")
    parser.add_argument(
        "--training", type=str2bool, default=False, help="start training new model."
    )
    parser.add_argument(
        "--scoring", type=str2bool, default=False, help="compute metrics."
    )
    parser.add_argument(
        "--inference", type=str2bool, default=False, help="inference audios."
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
        type=str2bool,
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

    recipe = load_recipe(
        args.config_path,
        expected_task="target_speaker_extraction",
        expected_purpose="train",
    )

    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(recipe)

    # Stage of dump the training samples
    if args.dump_training_samples:
        create_folder(folder_name="./dummy_samples")
        dataiter = iter(train_dataloader)
        for iteration in range(3):
            file_name = f"./dummy_samples/batch_{str(iteration).zfill(2)}"
            batch = next(dataiter)
            noisy_speech = batch["noisy_speech"]
            clean_speech = batch["clean_speech"]
            enroll_speech = batch["conditional_speech"]
            noise = batch["consistency_noise"]
            for i in range(noisy_speech.shape[0]):
                AudioIO.save(
                    wav=torch.stack(
                        [noisy_speech[i], clean_speech[i], noise[i]], dim=0
                    ),
                    f_path=f"{file_name}-{str(i).zfill(2)}.wav",
                    sr=batch["sr"][i],
                )
                AudioIO.save(
                    wav=enroll_speech[i].view(1, -1),
                    f_path=f"{file_name}-{str(i).zfill(2)}-enroll.wav",
                    sr=batch["sr"][i],
                )

    # Stage of training a new model
    if args.training:
        # Initialize loss function
        signal_loss_func_list, signal_loss_func_list_w = init_loss_func(
            recipe.signal_loss_func
        )
        if recipe.class_loss_func is not None:
            class_loss_func_list, class_loss_func_list_w = init_loss_func(
                recipe.class_loss_func
            )
        else:
            class_loss_func_list, class_loss_func_list_w = None, None

        # PL-Model
        lightning_model = init_model(recipe.model)
        lightning_model.register_loss_func(
            signal_loss_func_list,
            signal_loss_func_list_w,
            class_loss_func_list,
            class_loss_func_list_w,
        )

        # Load pretrained speaker model which trained by puresound framwork
        if recipe.model["c_backbone"]["pretrained_ckpt"] is not None:
            self_state = lightning_model.state_dict()
            loaded_spknet_state = torch.load(
                recipe.model["c_backbone"]["pretrained_ckpt"], map_location="cpu"
            )["state_dict"]

            for name, param in loaded_spknet_state.items():
                if (
                    not recipe.model["lightning_module"]["module_args"]["siamese_encoder"]
                    and "encoder.encoder" in name
                ):
                    name = name.replace("encoder.encoder", "c_encoder.encoder")

                if (
                    not recipe.model["lightning_module"]["module_args"]["siamese_feats"]
                    and "feats" in name
                ):
                    name = name.replace("feats.", "c_feats.")

                if "backbone" in name:
                    name = name.replace("backbone.", "c_backbone.")

                if "loss_func_list" in name:
                    continue

                self_state[name].copy_(param)

        param_groups = lightning_model.get_total_param_groups()
        optimizer, scheduler = create_optimizer_and_scheduler(
            overall_params_and_lr_factor=param_groups,
            optimizer_args=recipe.optimizer,
            scheduler_args=recipe.scheduler,
        )
        lightning_model.register_optimizer(optimizer)
        lightning_model.register_scheduler(scheduler)
        lightning_model.register_warmup_step(recipe.scheduler.warmup_step)

        # Loading exists state_dicts
        if args.pretrained_ckpt_path:
            print("Loading the pretrained params only.")
            state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")[
                "state_dict"
            ]
            lightning_model.load_state_dict(state_dict)

        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        ckpt_monitor = ModelCheckpoint(
            save_on_train_epoch_end=True, every_n_epochs=1, save_top_k=-1
        )

        trainer = L.Trainer(
            **recipe.trainer.lightning_trainer_args,
            accelerator="gpu" if recipe.trainer.num_gpus > 0 else "cpu",
            devices=recipe.trainer.num_gpus,
            limit_train_batches=recipe.trainer.train_iter_per_epoch,
            limit_val_batches=recipe.trainer.valid_iter_per_epoch,
            use_distributed_sampler=False,
            default_root_dir=recipe.trainer.work_folder,
            callbacks=[lr_monitor, ckpt_monitor],
            profiler="simple",
            sync_batchnorm=True,
        )

        if args.ckpt_path is not None:
            trainer.fit(
                lightning_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=args.ckpt_path,
            )
        else:
            trainer.fit(
                lightning_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
            )

    # Stage of caculating the metric scores
    if args.scoring:
        test_dataset = KaldiFormBaseDataset(
            folder=recipe.dataset.test_folder,
            mode="dev",
            resample_to=args.inference_sr,
        )
        test_dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(inference_mode=True)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        lightning_model = init_model(recipe.model)
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

    # Stage of inferencing audio only
    if args.inference:
        test_dataset = KaldiFormBaseDataset(
            folder=recipe.dataset.test_folder,
            mode="eval",
            resample_to=args.inference_sr,
        )
        test_dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(
            inference_mode=True, default_root_dir=recipe.dataset.proc_output_folder
        )
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        lightning_model = init_model(recipe.model)
        lightning_model.reload_checkpoint(state_dict)
        create_folder(recipe.dataset.proc_output_folder)
        lightning_model.register_proc_output_folder(recipe.dataset.proc_output_folder)
        trainer.predict(lightning_model, dataloaders=test_dataloader)
