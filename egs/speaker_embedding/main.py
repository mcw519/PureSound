import argparse
from typing import Dict, List

import lightning as L
import numpy as np
import onnxruntime
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from puresound import nnet, system
from puresound.audio.io import AudioIO
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.metrics import Metrics
from puresound.nnet import loss as ploss
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.sampler import SpeakerSampler
from puresound.task.sv import SpeakerEmbeddingCollateFunc, SpeakerEmbeddingDataset
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
        if config["augmentation_speech"]["used"]:
            aug_speech_dict = config["augmentation_speech"]
    if "augmentation_noise" in config:
        if config["augmentation_noise"]["used"]:
            aug_noise_dict = config["augmentation_noise"]
    if "augmentation_reverb" in config:
        if config["augmentation_reverb"]["used"]:
            aug_reverb_dict = config["augmentation_reverb"]
    if "augmentation_speed" in config:
        if config["augmentation_speed"]["used"]:
            aug_speed_dict = config["augmentation_speed"]
    if "augmentation_ir_response" in config:
        if config["augmentation_ir_response"]["used"]:
            aug_ir_dict = config["augmentation_ir_response"]
    if "augmentation_src" in config:
        if config["augmentation_src"]["used"]:
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
    train_dataset = SpeakerEmbeddingDataset(
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
        select_by_sr_first=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=SpeakerEmbeddingCollateFunc(),
    )

    valid_dataset = SpeakerEmbeddingDataset(
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
        select_by_sr_first=False,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_sampler=valid_sampler,
        pin_memory=True,
        num_workers=trainer_dict["num_workers"],
        collate_fn=SpeakerEmbeddingCollateFunc(),
    )

    return train_dataloader, valid_dataloader


def init_model(model_dict):
    lighting_module = getattr(system, model_dict["lighting_module"]["type"])
    encoder = getattr(nnet, model_dict["encoder"]["type"])(
        **model_dict["encoder"]["encoder_args"]
    )
    if "freq_eq" in model_dict.keys():
        peq_module = getattr(nnet, model_dict["freq_eq"]["type"])(
            **model_dict["freq_eq"]["eq_args"]
        )
        # register peq inside the feature module
        model_dict["features"]["peq_module"] = peq_module

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
        loss_list contain ModuleList([loss_1, loss_2, ...])
        loss_list_w contain [w1, w2, ...]
    """
    loss_list = torch.nn.ModuleList([])
    loss_list_w = []
    for item in hparam_conf:
        loss_func = getattr(ploss, item["type"])(**item["args"])
        loss_list.append(loss_func)
        loss_list_w.append(item["weighted"])

    return loss_list, loss_list_w


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
        help="choose a exist checkpoint, including params, optimizer, learning rate scheduler and loss functions.",
    )
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=str,
        default=None,
        help="choose a exist checkpoint with its model params only for training, \
            and continuous training with new optimizer, learning rate scheduler and loss etc.",
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
    parser.add_argument(
        "--split_to_chunks_with_size",
        type=float,
        default=None,
        help="If given, chunking the input audio (seconds).",
    )
    parser.add_argument(
        "--export_onnx", type=str2bool, default=False, help="export model to onnx form"
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

    # Stage of dump the training samples
    if args.dump_training_samples:
        create_folder(folder_name="./dummy_samples")
        dataiter = iter(train_dataloader)
        for iteration in range(3):
            file_name = f"./dummy_samples/batch_{str(iteration).zfill(2)}"
            batch = next(dataiter)
            noisy_speech = batch["noisy_speech"]
            target = batch["target"]
            for i in range(noisy_speech.shape[0]):
                spkid = str(target[i])
                AudioIO.save(
                    wav=noisy_speech,
                    f_path=f"{file_name}-{str(i).zfill(2)}-{spkid.zfill(5)}.wav",
                    sr=batch["sr"][i],
                )

    # Stage of training a new model
    if args.training:
        # Initialize loss function
        loss_func_list, loss_func_list_w = init_loss_func(hparam_conf=loss_dict)

        # PL-Model
        lighting_model = init_model(model_dict)
        lighting_model.register_loss_func(loss_func_list, loss_func_list_w)
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
            lighting_model.reload_checkpoint(
                loaded_state=state_dict, load_loss_func=False
            )

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
        lighting_model = init_model(model_dict)
        # TODO
        raise NotImplementedError

    # Stage of inferencing audio only
    if args.inference:
        test_dataset = KaldiFormBaseDataset(
            folder=corpus_dict["test_folder"],
            mode="eval",
            resample_to=args.inference_sr,
            split_to_chunks_with_size=args.split_to_chunks_with_size,
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
        lighting_model = init_model(model_dict)
        lighting_model.reload_checkpoint(state_dict)
        create_folder(corpus_dict["proc_output_folder"])
        lighting_model.register_proc_output_folder(corpus_dict["proc_output_folder"])
        trainer.predict(lighting_model, dataloaders=test_dataloader)

    # Stage of export model to ONNX
    if args.export_onnx and args.pretrained_ckpt_path:
        if args.inference_sr:
            sample_input = torch.rand(1, args.inference_sr * 5)
        else:
            sample_input = torch.rand(1, 16000 * 5)

        save_path = f"{args.pretrained_ckpt_path}.onnx"

        lighting_model = init_model(model_dict)
        print("Loading the pretrained params only.")
        state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")[
            "state_dict"
        ]
        lighting_model.reload_checkpoint(loaded_state=state_dict, load_loss_func=False)
        lighting_model.eval()

        torch.onnx.export(
            lighting_model,
            (sample_input,),
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=[
                "Audio",
            ],
            output_names=[
                "Embedding",
            ],
            dynamic_axes={
                "Audio": {0: "batch_size", 1: "sequence_length"},
                "Embedding": {0: "batch_size"},
            },
            verbose=False,
        )

        # Test onnx model
        with torch.no_grad():
            torch_out = lighting_model(sample_input)
            torch_out = torch_out.numpy()

        ort_session = onnxruntime.InferenceSession(save_path)
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: sample_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        atol = 1e-4
        while True:
            try:
                assert np.allclose(torch_out, ort_outs[0], rtol=1e-5, atol=atol)
                print(f">>> ONNX model accuracy pass {atol} spec.")
                atol /= 10
            except:
                print(f">>> Export done")
                print(f">>> ONNX model accuracy can't pass {atol} spec.")
                break
