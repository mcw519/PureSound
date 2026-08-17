import argparse

import lightning as L
import numpy as np
import onnxruntime
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.recipes import init_loss_func, init_siso_model
from puresound.system import runner
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.task.sv import SpeakerEmbeddingCollateFunc, SpeakerEmbeddingDataset
from puresound.utils import create_folder, str2bool


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe.

    Delegates to ``runner.build_dataloaders``; this file used to carry its own
    copy of it, which is what let the two drift.
    """
    return runner.build_dataloaders(
        dataset_cls=SpeakerEmbeddingDataset,
        collate_fn=SpeakerEmbeddingCollateFunc(),
        recipe=recipe,
    )


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

    recipe = load_recipe(
        args.config_path,
        expected_task="speaker_embedding",
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
        loss_func_list, loss_func_list_w = init_loss_func(recipe.loss_func)

        # PL-Model
        lightning_model = init_siso_model(recipe.model)
        lightning_model.register_loss_func(loss_func_list, loss_func_list_w)
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
            lightning_model.reload_checkpoint(
                loaded_state=state_dict, load_loss_func=False
            )

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
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            pin_memory=True,
            num_workers=4,
            batch_size=1,
            shuffle=False,
        )
        trainer = L.Trainer(inference_mode=True)
        lightning_model = init_siso_model(recipe.model)
        # TODO
        raise NotImplementedError

    # Stage of inferencing audio only
    if args.inference:
        test_dataset = KaldiFormBaseDataset(
            folder=recipe.dataset.test_folder,
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
            inference_mode=True, default_root_dir=recipe.dataset.proc_output_folder
        )
        state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
        lightning_model = init_siso_model(recipe.model)
        lightning_model.reload_checkpoint(state_dict)
        create_folder(recipe.dataset.proc_output_folder)
        lightning_model.register_proc_output_folder(recipe.dataset.proc_output_folder)
        trainer.predict(lightning_model, dataloaders=test_dataloader)

    # Stage of export model to ONNX
    if args.export_onnx and args.pretrained_ckpt_path:
        if args.inference_sr:
            sample_input = torch.rand(1, args.inference_sr * 5)
        else:
            sample_input = torch.rand(1, 16000 * 5)

        save_path = f"{args.pretrained_ckpt_path}.onnx"

        lightning_model = init_siso_model(recipe.model)
        print("Loading the pretrained params only.")
        state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")[
            "state_dict"
        ]
        lightning_model.reload_checkpoint(loaded_state=state_dict, load_loss_func=False)
        lightning_model.eval()

        torch.onnx.export(
            lightning_model,
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
            torch_out = lightning_model(sample_input)
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
            except AssertionError:
                print(">>> Export done")
                print(f">>> ONNX model accuracy can't pass {atol} spec.")
                break
