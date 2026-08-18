"""What a streaming frame model is, minus the backbone it wraps.

`dpcrn` and `dparn` each turn an offline encoder/decoder into a model that
consumes one STFT frame and carries its own state. What genuinely differs
between them is the shape of that state and what one block does with it. What
does not differ -- and was copied line for line -- is the naming of the state
ports, the causal down-convolution step, the ONNX export and its round-trip
check, and the whole ORT runtime.

The split follows what a measurement said rather than what the two files look
like. Comparing them unit by unit with the backbone names normalised away: the
export path and the loader are byte-identical, `_down_step` is byte-identical,
the port-name properties differ only by DPCRN's lookahead ports, and `forward`
differs only in whether it flattens the state through a helper. Everything else
-- the block step, the config validators, the state dataclasses -- is genuinely
different, and pulling those up would mean inventing a shared abstraction over
things that are not the same. They stay where they are.

`_up_step` is here because the difference between the two copies turned out to
be a bug rather than a design. Both overlap-add a transpose conv whose kernel
spans two time taps, and the conv adds its bias to both of them, so the sum
counts it twice where PyTorch's offline `conv_transpose` counts it once. DPCRN
subtracted the extra copy; DPARN did not, and streamed 1.15e-01 relative away
from its own offline forward until it did. Once corrected the two were the same
code, so it lives here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puresound.utils import load_hparam


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


class StreamingFrameModelBase(nn.Module):
    """The parts of a frame model that do not depend on its backbone.

    A subclass supplies `n_down`, `n_up`, `n_blocks`, `initial_state`,
    `_state_to_tuple`, `state_from_tensors` and `forward_frame`; it may add
    ports beyond the four standard groups through `_extra_state_names`.

    The port names are a published interface, not an implementation detail:
    they are written into the export manifest and the ORT runtime feeds the
    session by name. Renaming one silently breaks every model already exported.
    """

    def _extra_state_names(self, prefix: str = "") -> list[str]:
        """Ports beyond down/up/h/c. DPCRN's lookahead variant carries skip
        caches, a noisy cache and a frame counter; DPARN carries none."""
        return []

    @property
    def state_input_names(self) -> list[str]:
        return (
            [f"down_cache_{i}" for i in range(self.n_down)]
            + [f"up_cache_{i}" for i in range(self.n_up)]
            + [f"h_{i}" for i in range(self.n_blocks)]
            + [f"c_{i}" for i in range(self.n_blocks)]
            + self._extra_state_names("")
        )

    @property
    def state_output_names(self) -> list[str]:
        return (
            [f"next_down_cache_{i}" for i in range(self.n_down)]
            + [f"next_up_cache_{i}" for i in range(self.n_up)]
            + [f"next_h_{i}" for i in range(self.n_blocks)]
            + [f"next_c_{i}" for i in range(self.n_blocks)]
            + self._extra_state_names("next_")
        )

    def _state_to_tuple(self, state) -> tuple[torch.Tensor, ...]:
        """Flatten the state in the order `state_input_names` announces."""
        raise NotImplementedError

    def initial_state_tensors(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, ...]:
        return self._state_to_tuple(
            self.initial_state(batch_size=batch_size, device=device)
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Flat tensors in, flat tensors out -- the shape ONNX can express."""
        noisy_frame, *state_tensors = inputs
        state = self.state_from_tensors(state_tensors)
        enhanced, next_state = self.forward_frame(noisy_frame, state)
        return tuple([enhanced] + list(self._state_to_tuple(next_state)))

    def _down_step(self, layer: nn.Sequential, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = layer[0].padding
        conv = layer[1]
        x_ctx = torch.cat([cache, x], dim=-1) if cache.shape[-1] else x
        x_ctx = F.pad(x_ctx, (0, 0, pad[2], pad[3]))
        y = conv(x_ctx)
        y = layer[2](y)
        y = layer[3](y)
        y = layer[4](y)
        next_cache = x[..., -cache.shape[-1] :] if cache.shape[-1] else cache
        return y, next_cache

    def _up_step(
        self, layer: nn.Sequential, x: torch.Tensor, pending: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv = layer[0]
        raw = conv(x)
        # The transpose conv adds its bias to BOTH time taps of each frame's
        # splat, so overlap-adding tap[0] with the previous frame's tap[1] would
        # count the bias twice. PyTorch's offline conv_transpose adds bias once
        # per output position -> subtract one bias copy from the overlap sum.
        completed = raw[..., :1] + pending
        if conv.bias is not None:
            completed = completed - conv.bias.view(1, -1, 1, 1)
        next_pending = raw[..., 1:2]
        if len(layer) > 1:
            completed = layer[1](completed)
            completed = layer[2](completed)
        return completed, next_pending


@dataclass(frozen=True)
class StreamingVariant:
    """What the shared load and export paths need to know about one backbone.

    Three values, because that is all the two copies actually differed by: the
    name that goes into the manifest, the config validator, and the frame model
    to build.
    """

    name: str
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    frame_model: type[StreamingFrameModelBase]


def load_streaming_model(
    variant: StreamingVariant,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> StreamingFrameModelBase:
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    config_path = Path(config_path)
    config = load_hparam(str(config_path))
    variant.validate(config)
    model_dict = load_recipe(config_path).model
    system_model = init_siso_model(model_dict)
    if checkpoint_path:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        system_model.reload_checkpoint(state_dict, load_loss_func=False)
    return variant.frame_model(system_model.eval())


def export_streaming_onnx(
    variant: StreamingVariant,
    config_path: str | Path,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 17,
) -> dict[str, Any]:
    import onnxruntime

    frame_model = load_streaming_model(variant, config_path, checkpoint_path)
    frame_model.eval()
    onnx_path = Path(onnx_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else onnx_path.with_suffix(".json")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    geometry = variant.validate(load_hparam(str(config_path)))

    noisy_frame = torch.randn(1, int(geometry["freq_bins"]), 2)
    state = frame_model.initial_state_tensors(batch_size=1)

    input_names = ["noisy_frame"] + frame_model.state_input_names
    output_names = ["enhanced_frame"] + frame_model.state_output_names
    dynamic_axes = {
        "noisy_frame": {0: "batch_size"},
        "enhanced_frame": {0: "batch_size"},
    }
    for name in frame_model.state_input_names + frame_model.state_output_names:
        if name.startswith(("h_", "c_", "next_h_", "next_c_")):
            dynamic_axes[name] = {1: "batch_freq"}
        else:
            dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        frame_model,
        (noisy_frame, *state),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    providers = [provider for provider in providers if provider in available]
    session = onnxruntime.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
    ort_inputs = {"noisy_frame": noisy_frame.numpy()}
    for name, tensor in zip(frame_model.state_input_names, state):
        ort_inputs[name] = tensor.numpy()
    ort_out = session.run(None, ort_inputs)
    with torch.no_grad():
        torch_out = frame_model(noisy_frame, *state)
    if not np.allclose(torch_out[0].numpy(), ort_out[0], rtol=1e-4, atol=1e-4):
        raise AssertionError("exported ONNX frame output does not match PyTorch output")

    manifest = {
        "model_type": f"{variant.name}_streaming_frame",
        "processor": "stft_frame_ort",
        "created_at": int(time.time()),
        "onnx_path": str(onnx_path),
        "sample_rate": int(geometry["sample_rate"]),
        "fft_length": int(geometry["fft_length"]),
        "win_length": int(geometry["win_length"]),
        "hop_length": int(geometry["hop_length"]),
        "freq_bins": int(geometry["freq_bins"]),
        "feature_bins": int(geometry["feature_bins"]),
        "streaming_delay_frames": frame_model.streaming_delay,
        "input_names": input_names,
        "output_names": output_names,
        "state_input_names": frame_model.state_input_names,
        "state_output_names": frame_model.state_output_names,
        "state_shapes": {
            name: tensor_shape(tensor)
            for name, tensor in zip(frame_model.state_input_names, state)
        },
        "providers": providers or ["CPUExecutionProvider"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class StreamingOrt:
    def __init__(
        self,
        onnx_path: str | Path,
        manifest_path: str | Path | None = None,
        provider: str = "auto",
    ):
        import onnxruntime

        self.onnx_path = Path(onnx_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.onnx_path.with_suffix(".json")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        providers = self._resolve_providers(provider, onnxruntime.get_available_providers())
        self.session = onnxruntime.InferenceSession(str(self.onnx_path), providers=providers)
        self.providers = self.session.get_providers()
        self.sample_rate = int(self.manifest["sample_rate"])
        self.fft_length = int(self.manifest["fft_length"])
        self.win_length = int(self.manifest["win_length"])
        self.hop_length = int(self.manifest["hop_length"])
        self.freq_bins = int(self.manifest["freq_bins"])
        self.window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)
        self.reset()

    @staticmethod
    def _resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
        provider = provider.lower()
        if provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if provider == "auto":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        raise ValueError("provider must be one of: auto, cpu, cuda")

    def reset(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("StreamingOrt currently supports batch_size=1 for waveform streaming")
        self.state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.manifest["state_shapes"].items()
        }
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(0, dtype=np.float32)
        self.ola_norm = np.zeros(0, dtype=np.float32)

    def run_frame(self, noisy_frame: np.ndarray) -> np.ndarray:
        noisy_frame = np.asarray(noisy_frame, dtype=np.float32)
        if noisy_frame.shape != (1, self.freq_bins, 2):
            raise ValueError(f"noisy_frame must have shape (1, {self.freq_bins}, 2)")
        ort_inputs = {"noisy_frame": noisy_frame}
        ort_inputs.update(self.state)
        outputs = self.session.run(self.manifest["output_names"], ort_inputs)
        enhanced = outputs[0]
        for name, value in zip(self.manifest["state_input_names"], outputs[1:]):
            self.state[name] = value
        return enhanced

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window, n=self.fft_length).astype(np.complex64)
        noisy_frame = np.stack([spec.real, spec.imag], axis=-1).reshape(1, self.freq_bins, 2).astype(np.float32)
        enhanced = self.run_frame(noisy_frame)[0]
        enhanced_complex = enhanced[:, 0] + 1j * enhanced[:, 1]
        wav = np.fft.irfft(enhanced_complex, n=self.fft_length).astype(np.float32)[: self.win_length]
        return wav * self.window

    def _add_ola_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.ola.size < self.win_length:
            pad = self.win_length - self.ola.size
            self.ola = np.pad(self.ola, (0, pad))
            self.ola_norm = np.pad(self.ola_norm, (0, pad))
        self.ola[: self.win_length] += frame
        self.ola_norm[: self.win_length] += self.window * self.window
        emit = self.ola[: self.hop_length].copy()
        norm = self.ola_norm[: self.hop_length].copy()
        emit = emit / np.maximum(norm, 1e-8)
        self.ola = self.ola[self.hop_length :]
        self.ola_norm = self.ola_norm[self.hop_length :]
        return emit.astype(np.float32)

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.input_buffer = np.concatenate([self.input_buffer, samples])
        chunks = []
        while self.input_buffer.shape[0] >= self.win_length:
            frame = self.input_buffer[: self.win_length]
            self.input_buffer = self.input_buffer[self.hop_length :]
            chunks.append(self._add_ola_frame(self._process_frame(frame)))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        chunks = []
        while self.input_buffer.size > 0:
            frame = np.zeros(self.win_length, dtype=np.float32)
            n = min(self.input_buffer.size, self.win_length)
            frame[:n] = self.input_buffer[:n]
            self.input_buffer = self.input_buffer[min(self.hop_length, self.input_buffer.size) :]
            chunks.append(self._add_ola_frame(self._process_frame(frame)))
        if self.ola.size:
            tail = self.ola / np.maximum(self.ola_norm, 1e-8)
            chunks.append(tail.astype(np.float32))
            self.ola = np.zeros(0, dtype=np.float32)
            self.ola_norm = np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
