import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puresound.nnet.dparn import DPARN
from puresound.nnet.masker import Masker
from puresound.utils import load_hparam


@dataclass
class DparnStreamingState:
    down_caches: list[torch.Tensor]
    up_caches: list[torch.Tensor]
    h_states: list[torch.Tensor]
    c_states: list[torch.Tensor]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def validate_streaming_dparn_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    encoder = model.get("encoder", {})
    features = model.get("features", {})
    backbone = model.get("backbone", {})
    encoder_args = encoder.get("encoder_args", {})
    backbone_args = backbone.get("backbone_args", {})

    _require(dataset.get("target_sample_rate") == 16000, "streaming DPARN requires dataset.target_sample_rate=16000")
    _require(encoder.get("type") == "ConvEncDec", "streaming DPARN requires ConvEncDec encoder")
    _require(str(encoder_args.get("win_type", "")).lower() == "hann", "streaming DPARN requires a Hann window")
    _require(encoder_args.get("sr") == 16000, "streaming DPARN requires encoder sr=16000")
    _require(encoder_args.get("fmax") == 8000, "streaming DPARN requires encoder fmax=8000")
    _require(encoder_args.get("trainable") is False, "streaming DPARN requires a fixed Hann frontend: encoder.trainable=False")

    # STFT geometry is now read from the config so non-default win/hop (e.g.
    # voice_isolate v6 uses hop=160) are honoured by the manifest and the
    # runtime's OLA buffer alignment.
    fft_length = int(encoder_args.get("fft_length", 512))
    win_length = int(encoder_args.get("win_length", fft_length))
    hop_length = int(encoder_args.get("hop_length", win_length // 4))
    _require(win_length <= fft_length, "win_length must be <= fft_length")
    _require(hop_length > 0, "hop_length must be positive")

    _require(features.get("feats_type") == "complex", "streaming DPARN requires complex features")
    _require(features.get("drop_stft_first_bin") is True, "streaming DPARN requires drop_stft_first_bin=True")
    _require(features.get("trainable") is False, "streaming DPARN requires features.trainable=False")
    _require(not features.get("include_specaug", False), "streaming DPARN does not support specaug")

    _require(backbone.get("type") == "DPARN", "streaming DPARN requires a DPARN backbone")
    _require(backbone_args.get("input_dim") == 256, "streaming DPARN requires input_dim=256")
    # bN2d streams too: at eval BatchNorm2d uses fixed running statistics, so it
    # degenerates to a per-frame affine. DPCRN's validator already accepts it and
    # both share one runtime (StreamingDpcrnOrt is StreamingDparnOrt), so keeping
    # DPARN stricter only rejected configs that would have exported correctly --
    # including egs/noise_suppression/config/dparn.yaml, the repo's only DPARN recipe.
    _require(backbone_args.get("norm_type") in {"cLN", "iLN", "bN2d"}, "streaming DPARN requires cLN/iLN/bN2d normalization")
    _require(not backbone_args.get("skip_conv", False), "streaming DPARN currently expects skip_conv=False")
    _require(all(v == 1 for v in _as_list(backbone_args.get("stride_t", []))), "streaming DPARN requires stride_t=1 for every down layer")
    _require(all(v == 1 for v in _as_list(backbone_args.get("dilation_t", []))), "streaming DPARN requires dilation_t=1 for every down layer")

    # Causality hints. transpose_delay=False / delay!=0 mean the offline
    # forward peeks at future time steps, so the exported per-frame graph is
    # functionally still correct (the comparison check at export time
    # verifies PyTorch vs ONNX match) but the resulting model is NOT real
    # streaming -- live audio will not have the future frames the model
    # expects. Emit warnings rather than blocking the export so we can ship
    # ONNX for v6/v7 evaluation while a streaming-friendly recipe is being
    # retrained.
    import warnings

    if backbone_args.get("transpose_delay") is not True:
        warnings.warn(
            "transpose_delay=False: exported ONNX is per-frame but not truly causal. "
            "Streaming inference will leak future frames; retrain with transpose_delay=True for real streaming.",
            stacklevel=2,
        )
    if not all(v == 0 for v in _as_list(backbone_args.get("delay", []))):
        warnings.warn(
            f"delay={backbone_args.get('delay')}: down layers peek at future frames; "
            "exported ONNX will not be real streaming.",
            stacklevel=2,
        )

    # Hann window with rfft -> bins = fft_length // 2 + 1; feature bins drop
    # the DC bin (drop_stft_first_bin=True).
    freq_bins = fft_length // 2 + 1
    return {
        "sample_rate": 16000,
        "fft_length": fft_length,
        "win_length": win_length,
        "hop_length": hop_length,
        "freq_bins": freq_bins,
        "feature_bins": freq_bins - 1,
    }


class StreamingDparnFrameModel(nn.Module):
    """One-frame DPARN feature model with explicit streaming state."""

    def __init__(self, system_model: nn.Module):
        super().__init__()
        _require(hasattr(system_model, "backbone"), "system model must expose a DPARN backbone")
        _require(hasattr(system_model, "feats"), "system model must expose a feature encoder")
        _require(system_model.mask_type == "complex", "streaming DPARN supports complex masking only")
        _require(isinstance(system_model.backbone, DPARN), "streaming frame model requires a DPARN backbone")
        self.system_model = system_model
        self.backbone: DPARN = system_model.backbone
        self.feats = system_model.feats
        self.n_down = len(self.backbone.cnn_down)
        self.n_up = len(self.backbone.cnn_up)
        self.n_blocks = len(self.backbone.dparn_block)
        self.streaming_delay = self.n_up
        self.eval()

    @property
    def state_input_names(self) -> list[str]:
        return (
            [f"down_cache_{i}" for i in range(self.n_down)]
            + [f"up_cache_{i}" for i in range(self.n_up)]
            + [f"h_{i}" for i in range(self.n_blocks)]
            + [f"c_{i}" for i in range(self.n_blocks)]
        )

    @property
    def state_output_names(self) -> list[str]:
        return (
            [f"next_down_cache_{i}" for i in range(self.n_down)]
            + [f"next_up_cache_{i}" for i in range(self.n_up)]
            + [f"next_h_{i}" for i in range(self.n_blocks)]
            + [f"next_c_{i}" for i in range(self.n_blocks)]
        )

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> DparnStreamingState:
        device = torch.device(device)
        dtype = next(self.parameters()).dtype
        down_freqs, up_freqs = self.backbone.shape_info()
        down_caches = []
        for i, layer in enumerate(self.backbone.cnn_down):
            conv = layer[1]
            cache_t = int(conv.kernel_size[1] - 1)
            down_caches.append(
                torch.zeros(batch_size, conv.in_channels, down_freqs[i], cache_t, dtype=dtype, device=device)
            )

        up_caches = []
        for i, layer in enumerate(self.backbone.cnn_up):
            conv = layer[0]
            up_caches.append(
                torch.zeros(batch_size, conv.out_channels, up_freqs[i + 1], 1, dtype=dtype, device=device)
            )

        bottleneck_freq = down_freqs[-1]
        h_states = []
        c_states = []
        for block in self.backbone.dparn_block:
            hidden_size = block.inter_rnn.hidden_size
            state_shape = (1, batch_size * bottleneck_freq, hidden_size)
            h_states.append(torch.zeros(state_shape, dtype=dtype, device=device))
            c_states.append(torch.zeros(state_shape, dtype=dtype, device=device))
        return DparnStreamingState(down_caches, up_caches, h_states, c_states)

    def initial_state_tensors(self, batch_size: int = 1, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
        state = self.initial_state(batch_size=batch_size, device=device)
        return tuple(state.down_caches + state.up_caches + state.h_states + state.c_states)

    def state_from_tensors(self, tensors: Sequence[torch.Tensor]) -> DparnStreamingState:
        n_down = self.n_down
        n_up = self.n_up
        n_blocks = self.n_blocks
        tensors = list(tensors)
        return DparnStreamingState(
            down_caches=tensors[:n_down],
            up_caches=tensors[n_down : n_down + n_up],
            h_states=tensors[n_down + n_up : n_down + n_up + n_blocks],
            c_states=tensors[n_down + n_up + n_blocks : n_down + n_up + 2 * n_blocks],
        )

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

    def _dparn_block_step(
        self,
        block: nn.Module,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_intra_skip = x
        n_batch, channels, freq, n_frames = x.shape
        x = x.transpose(1, -1).reshape(n_batch * n_frames, freq, channels)
        x = block.intra_atten1(x.permute(0, 2, 1), causal=False).permute(0, 2, 1)
        x = block.intra_atten2(x.permute(0, 2, 1), causal=False).permute(0, 2, 1)
        x = block.intra_fc(x)
        x = block.intra_norm(x)
        x = x.reshape(n_batch, n_frames, freq, -1).transpose(1, -1)
        x = x_intra_skip + x

        x_inter_skip = x
        seq = x.permute(0, 2, 3, 1).reshape(n_batch * freq, n_frames, channels)
        rnn_out, (next_h, next_c) = block.inter_rnn.rnn(seq, (h, c))
        rnn_out = block.inter_rnn.drop(rnn_out)
        rnn_out = block.inter_rnn.proj(rnn_out.contiguous().view(-1, rnn_out.shape[2])).view(seq.shape)
        rnn_out = block.inter_norm(rnn_out)
        x = rnn_out.permute(0, 2, 1).reshape(n_batch, freq, channels, n_frames).permute(0, 2, 1, 3)
        return x_inter_skip + x, next_h, next_c

    def _up_step(self, layer: nn.Sequential, x: torch.Tensor, pending: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = layer[0](x)
        completed = raw[..., :1] + pending
        next_pending = raw[..., 1:2]
        if len(layer) > 1:
            completed = layer[1](completed)
            completed = layer[2](completed)
        return completed, next_pending

    def _forward_feature_frame(
        self,
        features: torch.Tensor,
        state: DparnStreamingState,
    ) -> tuple[torch.Tensor, DparnStreamingState]:
        if self.backbone.spectral_compress:
            raise ValueError("streaming DPARN does not support spectral_compress=True")

        x = self.backbone.input_norm(features)
        skip = [x]
        next_down = []
        for i, layer in enumerate(self.backbone.cnn_down):
            x, cache = self._down_step(layer, x, state.down_caches[i])
            next_down.append(cache)
            skip.append(x)

        next_h = []
        next_c = []
        for i, block in enumerate(self.backbone.dparn_block):
            x, h, c = self._dparn_block_step(block, x, state.h_states[i], state.c_states[i])
            next_h.append(h)
            next_c.append(c)

        next_up = []
        for i, layer in enumerate(self.backbone.cnn_up):
            x = torch.cat([x, skip[-i - 1]], dim=1)
            x, pending = self._up_step(layer, x, state.up_caches[i])
            next_up.append(pending)

        return x, DparnStreamingState(next_down, next_up, next_h, next_c)

    def forward_frame(
        self,
        noisy_frame: torch.Tensor,
        state: DparnStreamingState,
    ) -> tuple[torch.Tensor, DparnStreamingState]:
        if noisy_frame.dim() != 3 or noisy_frame.shape[-1] != 2:
            raise ValueError("noisy_frame must have shape [B, 257, 2]")
        tf_frame = noisy_frame.unsqueeze(2)
        features, features_for_enhanced = self.feats(tf_frame)
        mask, next_state = self._forward_feature_frame(features, state)
        enhanced = Masker.apply_complex_mask_on_reim(features_for_enhanced, mask)
        enhanced = self.feats.back_forward(enhanced)
        enhanced = enhanced.squeeze(-1).permute(0, 2, 1).contiguous()
        real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
        return torch.cat([real, imag], dim=-1).reshape(noisy_frame.shape), next_state

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        noisy_frame = inputs[0]
        state = self.state_from_tensors(inputs[1:])
        enhanced, next_state = self.forward_frame(noisy_frame, state)
        return tuple([enhanced] + next_state.down_caches + next_state.up_caches + next_state.h_states + next_state.c_states)


def create_streaming_dparn_model(system_model: nn.Module) -> StreamingDparnFrameModel:
    return StreamingDparnFrameModel(system_model.eval())


def load_streaming_dparn_model(config_path: str | Path, checkpoint_path: str | Path | None = None) -> StreamingDparnFrameModel:
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    config_path = Path(config_path)
    config = load_hparam(str(config_path))
    validate_streaming_dparn_config(config)
    model_dict = load_recipe(config_path).model
    system_model = init_siso_model(model_dict)
    if checkpoint_path:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        system_model.reload_checkpoint(state_dict, load_loss_func=False)
    return create_streaming_dparn_model(system_model)


def _tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def export_streaming_dparn_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 17,
) -> dict[str, Any]:
    import onnxruntime

    frame_model = load_streaming_dparn_model(config_path, checkpoint_path)
    frame_model.eval()
    onnx_path = Path(onnx_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else onnx_path.with_suffix(".json")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    geometry = validate_streaming_dparn_config(load_hparam(str(config_path)))

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
        "model_type": "dparn_streaming_frame",
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
            name: _tensor_shape(tensor)
            for name, tensor in zip(frame_model.state_input_names, state)
        },
        "providers": providers or ["CPUExecutionProvider"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class StreamingDparnOrt:
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
            raise ValueError("StreamingDparnOrt currently supports batch_size=1 for waveform streaming")
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
