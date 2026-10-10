"""Small Gradio client for the catalog-backed speaker-verification runtime."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

from puresound.inference import (
    InferenceResult,
    ModelZoo,
    SpeakerVerificationRuntime,
    load_model,
)


def cos_similarity(s1: torch.Tensor, s2: torch.Tensor):
    """Compatibility helper retained for callers of the old demo module."""
    return (s1 * s2).sum(axis=-1) / np.sqrt(
        (s1 * s1).sum(axis=-1) * (s2 * s2).sum(axis=-1)
    )


def inference(
    onnx_sess: Any,
    anchor_wav: str,
    test_wav: str,
    sr: int = 16000,
    norm_gain: float = -22,
    threshold: float = 0.8,
) -> str:
    """Run speaker verification through a runtime or a catalog model path/id."""
    localtime = time.strftime("%Y-%m-%d-%I-%M-%S", time.localtime())
    print(f"Uploaded audio: {anchor_wav}, {localtime}")
    if hasattr(onnx_sess, "infer"):
        runtime = onnx_sess
    elif isinstance(onnx_sess, (str, Path)):
        runtime = load_model(str(onnx_sess), provider="auto")
    elif hasattr(onnx_sess, "run"):
        # Old callers may still pass an already-created InferenceSession.
        runtime = SpeakerVerificationRuntime.from_session(onnx_sess)
    else:
        raise TypeError("onnx_sess must be a catalog model id/path or runtime")
    result: InferenceResult = runtime.infer(
        inputs={"enrollment": anchor_wav, "test": test_wav},
        parameters={"target_dbfs": float(norm_gain), "threshold": float(threshold)},
    )
    score = float(result.scores["cosine_similarity"])
    verdict = bool(result.scores["verdict"])
    return f"{'PASS' if verdict else 'Fail'}, Cosine similarity is: {score:.3f}"


def main(args):
    zoo = ModelZoo.default()
    models = zoo.list(task="speaker_embedding")
    models.sort(key=lambda model: ("default" not in model.roles, model.id))
    choices = [(model.display_name, model.id) for model in models]
    default_model = choices[0][1] if choices else "speaker-verification-ps-spk-v1-1"
    selected = getattr(args, "_selected_model_id", None) or getattr(args, "onnx_path", None)
    if selected:
        try:
            default_model = zoo.get(str(selected)).id
        except KeyError:
            found = zoo.find_by_artifact(str(selected))
            if found is not None:
                default_model = found[0].id
    runtime_cache: dict[str, Any] = {}

    def wrapped_inference(model_id, enroll_wav, test_wav, norm_gain, threshold):
        if not model_id:
            model_id = default_model
        runtime = runtime_cache.get(model_id)
        if runtime is None:
            runtime = load_model(model_id, provider="auto", zoo=zoo)
            runtime_cache[model_id] = runtime
        return inference(runtime, enroll_wav, test_wav, norm_gain=norm_gain, threshold=threshold)

    gr.Interface(
        fn=wrapped_inference,
        title="PureSound's Speaker Verification",
        description="Upload two speech samples; audio is converted to mono 16 kHz by the Model Zoo processor.",
        inputs=[
            gr.Dropdown(label="Model", choices=choices, value=default_model),
            gr.Audio(label="Upload Enrollment Speech", type="filepath", buttons=["download"]),
            gr.Audio(label="Upload Testing Speech", type="filepath", buttons=["download"]),
            gr.Slider(label="Normalized gain (dB)", minimum=-40, maximum=-16, value=-22, step=2),
            gr.Slider(label="Cosine Threshold", minimum=0.1, maximum=0.95, value=0.8, step=0.05),
        ],
        outputs=[gr.Textbox(label="Scores")],
    ).launch(server_name=args.address, server_port=args.port, share=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "onnx_path",
        type=str,
        nargs="?",
        help="legacy ONNX path or catalog model id (defaults to PS-spk-v1-1)",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--address", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6666)
    args = parser.parse_args()
    if args.model_id:
        # Keep one source of truth for the Gradio model selector while allowing
        # ``--model-id`` to be the preferred spelling in new scripts.
        args._selected_model_id = args.model_id
    elif args.onnx_path:
        args._selected_model_id = args.onnx_path
    else:
        args._selected_model_id = "speaker-verification-ps-spk-v1-1"
    main(args)
