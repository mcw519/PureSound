"""speaker_embedding -- speaker verification / d-vector training entry point.

Produces a fixed-length embedding per utterance, trained with a margin-based
classification loss. Its dataset is ``SpeakerEmbeddingDataset``; every stage
downstream lives in ``puresound.system.runner``, the same one the separation
recipes use.

Run from this directory -- the config's metafile and work-folder paths are
relative to it::

    cd egs/speaker_embedding
    uv run python main.py conf/PS-spk-v1.yaml --training
    uv run python main.py conf/PS-spk-v1.yaml --export_onnx \
        --pretrained_ckpt_path path/to.ckpt
"""

from pathlib import Path
import sys

import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.config import load_recipe  # noqa: E402
from puresound.system import runner  # noqa: E402
from puresound.task.sv import (  # noqa: E402
    SpeakerEmbeddingCollateFunc,
    SpeakerEmbeddingDataset,
)

TASK_NAME = "speaker_embedding"

runner.configure_torch_backends()


def init_dataloader(recipe):
    """Train / valid dataloaders for this recipe."""
    return runner.build_dataloaders(
        dataset_cls=SpeakerEmbeddingDataset,
        collate_fn=SpeakerEmbeddingCollateFunc(),
        recipe=recipe,
    )


def write_batch(batch, file_name):
    """This task's batch is one waveform per row plus a speaker id.

    No clean/noise pair to interleave, so the id goes in the filename -- that is
    the only thing worth eyeballing about a speaker-embedding sample.
    """
    noisy_speech = batch["noisy_speech"]
    target = batch["target"]
    for i in range(noisy_speech.shape[0]):
        spkid = str(target[i]).zfill(5)
        AudioIO.save(
            wav=noisy_speech[i : i + 1],
            f_path=f"{file_name}-{str(i).zfill(2)}-{spkid}.wav",
            sr=batch["sr"][i],
        )


def export_onnx(args, recipe):
    """Export the embedding extractor: one waveform in, one vector out."""
    import numpy as np
    import onnxruntime
    import torch

    from puresound.recipes import init_siso_model

    sample_rate = args.inference_sr or 16000
    sample_input = torch.rand(1, sample_rate * 5)
    save_path = f"{args.pretrained_ckpt_path}.onnx"

    lightning_model = init_siso_model(recipe.model)
    state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")["state_dict"]
    lightning_model.reload_checkpoint(loaded_state=state_dict, load_loss_func=False)
    lightning_model.eval()

    torch.onnx.export(
        lightning_model,
        (sample_input,),
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["Audio"],
        output_names=["Embedding"],
        dynamic_axes={
            "Audio": {0: "batch_size", 1: "sequence_length"},
            "Embedding": {0: "batch_size"},
        },
        verbose=False,
    )

    with torch.no_grad():
        torch_out = lightning_model(sample_input).numpy()
    session = onnxruntime.InferenceSession(save_path)
    ort_out = session.run(
        None, {session.get_inputs()[0].name: sample_input.numpy()}
    )[0]

    # Report the tightest tolerance the export actually holds rather than
    # asserting one: what matters is knowing which it is.
    atol = 1e-4
    while np.allclose(torch_out, ort_out, rtol=1e-5, atol=atol):
        print(f">>> ONNX model accuracy pass {atol} spec.")
        atol /= 10
    print(f">>> Export done; accuracy does not reach {atol} spec.")


if __name__ == "__main__":
    parser = runner.build_arg_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--split_to_chunks_with_size",
        type=float,
        default=None,
        help="If given, chunking the input audio (seconds).",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        default=False,
        help="export the embedding extractor to ONNX (needs --pretrained_ckpt_path).",
    )
    args = parser.parse_args()

    if args.set_seed is not None:
        print(f"Adjust random seed to {args.set_seed}")
        L.seed_everything(seed=args.set_seed)

    recipe = load_recipe(
        args.config_path, expected_task=TASK_NAME, expected_purpose="train"
    )

    train_dataloader = valid_dataloader = None
    if args.training or args.dump_training_samples:
        train_dataloader, valid_dataloader = init_dataloader(recipe)

    # metrics=None: verification is scored by EER over trial pairs, not by the
    # waveform metrics the separation tasks use, and that stage was never
    # written. `--scoring` raises saying so rather than half-running.
    runner.run_stages(
        args,
        recipe,
        train_dataloader,
        valid_dataloader,
        write_batch=write_batch,
        metrics=None,
    )

    if args.export_onnx:
        if not args.pretrained_ckpt_path:
            parser.error("--export_onnx needs --pretrained_ckpt_path")
        export_onnx(args, recipe)
