"""The catalog's int8 (flash) graphs on the machine running the tests.

int8 kernels differ by instruction set, and on some an int8 product can
saturate: x86 without VNNI multiplies uint8 activations by int8 weights through
16-bit pairs. Each flash artifact is therefore held to its float graph here, on
whatever CPU runs the suite, over real speech.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from puresound.inference import ModelZoo
from puresound.inference.benchmark import benchmark_clip
from puresound.streaming.quantization import INT8_MAX_RELATIVE_ERROR


def _flash_pairs():
    zoo = ModelZoo.default()
    pairs = []
    for model in zoo.list():
        variants = {artifact.variant: artifact for artifact in model.artifacts}
        if "flash" in variants and "default" in variants:
            pairs.append(pytest.param(zoo, variants["default"], variants["flash"], id=model.id))
    return pairs


def _enhanced(path, manifest, frames):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    state = {name: np.zeros(shape, np.float32) for name, shape in manifest["state_shapes"].items()}
    out = []
    for frame in frames:
        result = session.run(manifest["output_names"], {"noisy_frame": frame, **state})
        out.append(result[0].reshape(-1))
        state.update(zip(manifest["state_input_names"], result[1 + len(manifest["extra_output_names"]):]))
    return np.concatenate(out)


@pytest.mark.parametrize("zoo, default, flash", _flash_pairs())
def test_every_flash_graph_stays_near_its_float_graph_on_this_cpu(zoo, default, flash):
    manifest = json.loads(zoo.path_for(default.manifest).read_text(encoding="utf-8"))
    window = np.hanning(manifest["win_length"] + 1)[:-1].astype(np.float32)
    clip = benchmark_clip(2.5, int(manifest["sample_rate"]))
    frames = []
    for start in range(0, clip.size - manifest["win_length"], manifest["hop_length"]):
        spectrum = np.fft.rfft(clip[start:start + manifest["win_length"]] * window, n=manifest["fft_length"])
        frames.append(np.stack([spectrum.real, spectrum.imag], -1).reshape(1, -1, 2).astype(np.float32))

    reference = _enhanced(zoo.path_for(default.path), manifest, frames)
    quantized = _enhanced(zoo.path_for(flash.path), manifest, frames)
    error = float(np.linalg.norm(quantized - reference) / np.linalg.norm(reference))

    assert error < INT8_MAX_RELATIVE_ERROR, (
        f"{flash.path}: relative error {error:.3f} against its float graph on this CPU"
    )
