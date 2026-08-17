"""Per-frame streaming + ONNX export for the DPCRN backbone.

Mirrors ``puresound/streaming/dparn.py`` (both backbones subclass the same
``Unet`` base, so the encoder/decoder stepping, state layout, feature/mask/iSTFT
front-end, export and manifest are identical). The ONLY structural difference is
the bottleneck block: DPCRN's intra path is a bidirectional LSTM over the
frequency axis (time-independent, carries no cross-frame state), whereas DPARN
uses self-attention. The inter path (unidirectional LSTM over time) is the same
``SingleRNN`` and is the only operator whose ``(h, c)`` must persist across frames.

Look-ahead note: the voice_isolate recipe uses ``delay=[1,1,1]`` +
``transpose_delay=False`` (30 ms look-ahead). The per-frame graph reproduces the
offline forward up to a fixed output delay (validated end-to-end by the
full-utterance parity test); the runtime supplies the future frames via its ring
buffer, so the streamed output is the offline result delayed by that many frames.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.masker import Masker
from puresound.utils import load_hparam

# The ORT runtime is fully manifest-driven (dispatched by processor
# "stft_frame_ort", state read from JSON), so the DPARN runtime handles a DPCRN
# manifest unchanged. Re-exported under a DPCRN name for callers/CLIs.
from puresound.streaming.dparn import StreamingDparnOrt as StreamingDpcrnOrt  # noqa: F401


@dataclass
class DpcrnStreamingState:
    down_caches: list[torch.Tensor]
    up_caches: list[torch.Tensor]
    h_states: list[torch.Tensor]
    c_states: list[torch.Tensor]
    # Look-ahead streaming extras (empty / None for a causal model, delay=[0,0,0]).
    # skip_caches: per-up-layer FIFO delay lines that re-align each U-Net skip with
    #   the main path's cumulative bottleneck latency.
    # noisy_cache: delay line for the mask-application spectrum (delay = bottleneck).
    # counter: frame index; gates the inter-LSTM state to zero during the startup
    #   warmup so the causal down-path's phantom frames never contaminate it.
    skip_caches: list[torch.Tensor] = field(default_factory=list)
    noisy_cache: torch.Tensor | None = None
    counter: torch.Tensor | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def validate_streaming_dpcrn_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    encoder = model.get("encoder", {})
    features = model.get("features", {})
    backbone = model.get("backbone", {})
    encoder_args = encoder.get("encoder_args", {})
    backbone_args = backbone.get("backbone_args", {})

    _require(dataset.get("target_sample_rate") == 16000, "streaming DPCRN requires dataset.target_sample_rate=16000")
    _require(encoder.get("type") == "ConvEncDec", "streaming DPCRN requires ConvEncDec encoder")
    _require(str(encoder_args.get("win_type", "")).lower() == "hann", "streaming DPCRN requires a Hann window")
    _require(encoder_args.get("sr") == 16000, "streaming DPCRN requires encoder sr=16000")
    _require(encoder_args.get("fmax") == 8000, "streaming DPCRN requires encoder fmax=8000")
    _require(encoder_args.get("trainable") is False, "streaming DPCRN requires a fixed Hann frontend: encoder.trainable=False")

    fft_length = int(encoder_args.get("fft_length", 512))
    win_length = int(encoder_args.get("win_length", fft_length))
    hop_length = int(encoder_args.get("hop_length", win_length // 4))
    _require(win_length <= fft_length, "win_length must be <= fft_length")
    _require(hop_length > 0, "hop_length must be positive")

    _require(features.get("feats_type") == "complex", "streaming DPCRN requires complex features")
    _require(features.get("drop_stft_first_bin") is True, "streaming DPCRN requires drop_stft_first_bin=True")
    _require(features.get("trainable") is False, "streaming DPCRN requires features.trainable=False")
    _require(not features.get("include_specaug", False), "streaming DPCRN does not support specaug")

    _require(backbone.get("type") == "DPCRN", "streaming DPCRN requires a DPCRN backbone")
    _require(backbone_args.get("input_dim") == 256, "streaming DPCRN requires input_dim=256")
    # bN2d (BatchNorm2d) is allowed: in eval() it applies fixed running stats
    # per (freq,time) location, so it is frame-independent (no cross-time state).
    _require(backbone_args.get("norm_type") in {"cLN", "iLN", "bN2d"}, "streaming DPCRN requires cLN/iLN/bN2d normalization")
    _require(not backbone_args.get("skip_conv", False), "streaming DPCRN currently expects skip_conv=False")
    _require(all(v == 1 for v in _as_list(backbone_args.get("stride_t", []))), "streaming DPCRN requires stride_t=1 for every down layer")
    _require(all(v == 1 for v in _as_list(backbone_args.get("dilation_t", []))), "streaming DPCRN requires dilation_t=1 for every down layer")

    # Causality note (non-blocking). delay=[0,0,0] streams bit-exact with ZERO
    # latency. A look-ahead down-path (delay>0) is also supported and bit-exact,
    # but the streamed output carries a fixed algorithmic latency of D = sum(delay)
    # frames -- the frame model buffers the future frames via extra state (U-Net
    # skip delay lines + noisy-spectrum delay) and gates the inter-LSTM during the
    # D-frame startup warmup. transpose_delay must stay False; transpose_delay=True
    # makes the decoder anti-causal (uses future frames) and breaks streaming parity.
    import warnings

    delay = _as_list(backbone_args.get("delay", []))
    if not all(v == 0 for v in delay):
        warnings.warn(
            f"delay={backbone_args.get('delay')} (look-ahead): streams bit-exactly but with a fixed "
            f"algorithmic latency of {sum(int(v) for v in delay)} frames (~{sum(int(v) for v in delay) * hop_length / 16:.0f} ms at "
            f"hop {hop_length}); the exported graph carries extra look-ahead buffering state.",
            stacklevel=2,
        )
    if backbone_args.get("transpose_delay") is True:
        warnings.warn(
            "transpose_delay=True makes the decoder anti-causal (uses future frames) and breaks "
            "streaming parity. Use transpose_delay=False.",
            stacklevel=2,
        )

    freq_bins = fft_length // 2 + 1
    return {
        "sample_rate": 16000,
        "fft_length": fft_length,
        "win_length": win_length,
        "hop_length": hop_length,
        "freq_bins": freq_bins,
        "feature_bins": freq_bins - 1,
    }


class StreamingDpcrnFrameModel(nn.Module):
    """One-frame DPCRN feature model with explicit streaming state."""

    def __init__(self, system_model: nn.Module):
        super().__init__()
        _require(hasattr(system_model, "backbone"), "system model must expose a DPCRN backbone")
        _require(hasattr(system_model, "feats"), "system model must expose a feature encoder")
        _require(system_model.mask_type == "complex", "streaming DPCRN supports complex masking only")
        _require(isinstance(system_model.backbone, DPCRN), "streaming frame model requires a DPCRN backbone")
        self.system_model = system_model
        self.backbone: DPCRN = system_model.backbone
        self.feats = system_model.feats
        # DPCRN keeps its two DPRNN blocks as separate attributes (not a ModuleList).
        self.blocks = [self.backbone.dprnn_block1, self.backbone.dprnn_block2]
        self.n_down = len(self.backbone.cnn_down)
        self.n_up = len(self.backbone.cnn_up)
        self.n_blocks = len(self.blocks)

        # Look-ahead streaming geometry. Each down layer peeks `delay[i]` future
        # frames (offline right time-pad); the causal per-frame step realises that
        # as a fixed output delay. Cumulative look-ahead at down layer k = cum[k];
        # the bottleneck (and thus the whole main path) is delayed by D = cum[-1].
        # A U-Net skip from down layer k must be delayed by (D - cum[k]) to line up
        # with the main path at the up layer that consumes it. delay=[0,0,0] => all
        # zero => the causal fast path (no extra state, no warmup gate).
        delay_cfg = [int(v) for v in _as_list(getattr(self.backbone, "delay", [0] * self.n_down))]
        cum, run = [], 0
        for d in delay_cfg[: self.n_down]:
            run += d
            cum.append(run)
        self.bottleneck_delay = cum[-1] if cum else 0
        # up layer i consumes skip[-i-1] == down layer (n_down-1-i)
        self.skip_delay = [self.bottleneck_delay - cum[self.n_down - 1 - i] for i in range(self.n_up)]
        self.skip_delay_layers = [i for i in range(self.n_up) if self.skip_delay[i] > 0]
        self.is_lookahead = self.bottleneck_delay > 0
        self.warmup_frames = self.bottleneck_delay
        # Algorithmic output latency in frames (0 for a causal model).
        self.streaming_delay = self.bottleneck_delay
        self.eval()

    def _lookahead_state_suffix(self, prefix: str = "") -> list[str]:
        if not self.is_lookahead:
            return []
        names = [f"{prefix}skip_cache_{i}" for i in self.skip_delay_layers]
        names.append(f"{prefix}noisy_cache")
        names.append(f"{prefix}counter")
        return names

    @property
    def state_input_names(self) -> list[str]:
        return (
            [f"down_cache_{i}" for i in range(self.n_down)]
            + [f"up_cache_{i}" for i in range(self.n_up)]
            + [f"h_{i}" for i in range(self.n_blocks)]
            + [f"c_{i}" for i in range(self.n_blocks)]
            + self._lookahead_state_suffix("")
        )

    @property
    def state_output_names(self) -> list[str]:
        return (
            [f"next_down_cache_{i}" for i in range(self.n_down)]
            + [f"next_up_cache_{i}" for i in range(self.n_up)]
            + [f"next_h_{i}" for i in range(self.n_blocks)]
            + [f"next_c_{i}" for i in range(self.n_blocks)]
            + self._lookahead_state_suffix("next_")
        )

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> DpcrnStreamingState:
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
        for block in self.blocks:
            hidden_size = block.inter_rnn.hidden_size
            state_shape = (1, batch_size * bottleneck_freq, hidden_size)
            h_states.append(torch.zeros(state_shape, dtype=dtype, device=device))
            c_states.append(torch.zeros(state_shape, dtype=dtype, device=device))

        skip_caches: list[torch.Tensor] = []
        noisy_cache = None
        counter = None
        if self.is_lookahead:
            skip_shapes, noisy_shape = self._delay_line_shapes(batch_size, down_freqs)
            for i in self.skip_delay_layers:
                c, f = skip_shapes[i]
                skip_caches.append(
                    torch.zeros(batch_size, c, f, self.skip_delay[i], dtype=dtype, device=device)
                )
            nc, nf = noisy_shape
            noisy_cache = torch.zeros(batch_size, nc, nf, self.bottleneck_delay, dtype=dtype, device=device)
            counter = torch.zeros(1, dtype=dtype, device=device)
        return DpcrnStreamingState(down_caches, up_caches, h_states, c_states, skip_caches, noisy_cache, counter)

    def _delay_line_shapes(self, batch_size, down_freqs):
        """(channels, freq) of each up-layer skip tensor and of the mask-application
        spectrum, captured from a dummy down-pass (robust to config geometry)."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        in_ch = self.backbone.cnn_down[0][1].in_channels
        dummy = torch.zeros(batch_size, in_ch, down_freqs[0], 1, dtype=dtype, device=device)
        with torch.no_grad():
            x = self.backbone.input_norm(dummy)
            skip = [x]
            state = DpcrnStreamingState(
                [torch.zeros(batch_size, l[1].in_channels, down_freqs[i], int(l[1].kernel_size[1] - 1),
                             dtype=dtype, device=device) for i, l in enumerate(self.backbone.cnn_down)],
                [], [], [],
            )
            for i, layer in enumerate(self.backbone.cnn_down):
                x, _ = self._down_step(layer, x, state.down_caches[i])
                skip.append(x)
        skip_shapes = {}
        for i in range(self.n_up):
            t = skip[-i - 1]
            skip_shapes[i] = (t.shape[1], t.shape[2])
        # mask-application spectrum == features_for_enhanced, same [C, F] as features
        noisy_shape = (in_ch, down_freqs[0])
        return skip_shapes, noisy_shape

    def _state_to_tuple(self, state: DpcrnStreamingState) -> tuple[torch.Tensor, ...]:
        tensors = list(state.down_caches + state.up_caches + state.h_states + state.c_states)
        if self.is_lookahead:
            tensors += list(state.skip_caches)
            tensors.append(state.noisy_cache)
            tensors.append(state.counter)
        return tuple(tensors)

    def initial_state_tensors(self, batch_size: int = 1, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
        return self._state_to_tuple(self.initial_state(batch_size=batch_size, device=device))

    def state_from_tensors(self, tensors: Sequence[torch.Tensor]) -> DpcrnStreamingState:
        n_down = self.n_down
        n_up = self.n_up
        n_blocks = self.n_blocks
        tensors = list(tensors)
        base = n_down + n_up + 2 * n_blocks
        skip_caches: list[torch.Tensor] = []
        noisy_cache = None
        counter = None
        if self.is_lookahead:
            n_skip = len(self.skip_delay_layers)
            skip_caches = tensors[base : base + n_skip]
            noisy_cache = tensors[base + n_skip]
            counter = tensors[base + n_skip + 1]
        return DpcrnStreamingState(
            down_caches=tensors[:n_down],
            up_caches=tensors[n_down : n_down + n_up],
            h_states=tensors[n_down + n_up : n_down + n_up + n_blocks],
            c_states=tensors[n_down + n_up + n_blocks : base],
            skip_caches=skip_caches,
            noisy_cache=noisy_cache,
            counter=counter,
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

    def _dprnn_block_step(
        self,
        block: nn.Module,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # intra: bidirectional LSTM over the frequency axis -- time-independent,
        # carries no cross-frame state (verbatim from DPRNNblock2D.forward).
        x_intra_skip = x
        n_batch, channels, freq, n_frames = x.shape
        xi = x.transpose(1, -1).reshape(n_batch * n_frames, freq, channels)
        xi = block.intra_rnn(xi.permute(0, 2, 1))
        xi = xi.permute(0, 2, 1)
        xi = block.intra_norm(xi)
        xi = xi.reshape(n_batch, n_frames, freq, -1).transpose(1, -1)
        x = x_intra_skip + xi

        # inter: unidirectional LSTM over time -- the only per-block state (h, c).
        x_inter_skip = x
        seq = x.permute(0, 2, 3, 1).reshape(n_batch * freq, n_frames, channels)
        rnn_out, (next_h, next_c) = block.inter_rnn.rnn(seq, (h, c))
        rnn_out = block.inter_rnn.drop(rnn_out)
        rnn_out = block.inter_rnn.proj(rnn_out.contiguous().view(-1, rnn_out.shape[2])).view(seq.shape)
        rnn_out = block.inter_norm(rnn_out)
        x = rnn_out.permute(0, 2, 1).reshape(n_batch, freq, channels, n_frames).permute(0, 2, 1, 3)
        return x_inter_skip + x, next_h, next_c

    def _up_step(self, layer: nn.Sequential, x: torch.Tensor, pending: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv = layer[0]
        raw = conv(x)
        # The transpose conv adds its bias to BOTH time taps of each frame's splat,
        # so overlap-adding tap[0] with the previous frame's tap[1] would count the
        # bias twice. PyTorch's offline conv_transpose adds bias once per output
        # position -> subtract one bias copy from the overlap sum.
        completed = raw[..., :1] + pending
        if conv.bias is not None:
            completed = completed - conv.bias.view(1, -1, 1, 1)
        next_pending = raw[..., 1:2]
        if len(layer) > 1:
            completed = layer[1](completed)
            completed = layer[2](completed)
        return completed, next_pending

    @staticmethod
    def _shift(cache: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FIFO delay line of length cache.shape[-1]: emit the oldest frame, push
        `new`. Emitting oldest == `new` delayed by cache.shape[-1] frames."""
        buf = torch.cat([cache, new], dim=-1)
        return buf[..., :1], buf[..., 1:]

    def _forward_feature_frame(
        self,
        features: torch.Tensor,
        state: DpcrnStreamingState,
    ) -> tuple[torch.Tensor, DpcrnStreamingState]:
        if getattr(self.backbone, "spectral_compress", False):
            raise ValueError("streaming DPCRN does not support spectral_compress=True")

        x = self.backbone.input_norm(features)
        skip = [x]
        next_down = []
        for i, layer in enumerate(self.backbone.cnn_down):
            x, cache = self._down_step(layer, x, state.down_caches[i])
            next_down.append(cache)
            skip.append(x)

        next_h = []
        next_c = []
        for i, block in enumerate(self.blocks):
            x, h, c = self._dprnn_block_step(block, x, state.h_states[i], state.c_states[i])
            next_h.append(h)
            next_c.append(c)

        next_skip: list[torch.Tensor] = []
        next_counter = state.counter
        if self.is_lookahead:
            # Warmup gate: hold the inter-LSTM state at zero until the causal down
            # pipeline has flushed its `warmup_frames` phantom startup frames, so the
            # LSTM's first real input is offline's bottleneck[0] from a clean state.
            keep = (state.counter >= float(self.warmup_frames)).to(x.dtype).view(1, 1, 1)
            next_h = [h * keep for h in next_h]
            next_c = [c * keep for c in next_c]
            next_counter = state.counter + 1.0

        next_up = []
        skip_cache_idx = 0
        for i, layer in enumerate(self.backbone.cnn_up):
            sk = skip[-i - 1]
            if self.skip_delay[i] > 0:
                sk, new_cache = self._shift(state.skip_caches[skip_cache_idx], sk)
                next_skip.append(new_cache)
                skip_cache_idx += 1
            x = torch.cat([x, sk], dim=1)
            x, pending = self._up_step(layer, x, state.up_caches[i])
            next_up.append(pending)

        return x, DpcrnStreamingState(
            next_down, next_up, next_h, next_c, next_skip, state.noisy_cache, next_counter
        )

    def forward_frame(
        self,
        noisy_frame: torch.Tensor,
        state: DpcrnStreamingState,
    ) -> tuple[torch.Tensor, DpcrnStreamingState]:
        if noisy_frame.dim() != 3 or noisy_frame.shape[-1] != 2:
            raise ValueError("noisy_frame must have shape [B, freq_bins, 2]")
        tf_frame = noisy_frame.unsqueeze(2)
        features, features_for_enhanced = self.feats(tf_frame)
        mask, next_state = self._forward_feature_frame(features, state)
        # The mask leaves _forward_feature_frame already delayed by `bottleneck_delay`
        # frames (look-ahead compensation). Apply it to the equally-delayed noisy
        # spectrum so the mask and the spectrum are the same offline frame.
        noisy_cache = state.noisy_cache
        if self.is_lookahead:
            features_for_enhanced, noisy_cache = self._shift(state.noisy_cache, features_for_enhanced)
        next_state = DpcrnStreamingState(
            next_state.down_caches, next_state.up_caches, next_state.h_states,
            next_state.c_states, next_state.skip_caches, noisy_cache, next_state.counter,
        )
        enhanced = Masker.apply_complex_mask_on_reim(features_for_enhanced, mask)
        enhanced = self.feats.back_forward(enhanced)
        enhanced = enhanced.squeeze(-1).permute(0, 2, 1).contiguous()
        real, imag = torch.chunk(enhanced, chunks=2, dim=-1)
        return torch.cat([real, imag], dim=-1).reshape(noisy_frame.shape), next_state

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        noisy_frame = inputs[0]
        state = self.state_from_tensors(inputs[1:])
        enhanced, next_state = self.forward_frame(noisy_frame, state)
        return tuple([enhanced] + list(self._state_to_tuple(next_state)))


def create_streaming_dpcrn_model(system_model: nn.Module) -> StreamingDpcrnFrameModel:
    return StreamingDpcrnFrameModel(system_model.eval())


def load_streaming_dpcrn_model(config_path: str | Path, checkpoint_path: str | Path | None = None) -> StreamingDpcrnFrameModel:
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    config_path = Path(config_path)
    config = load_hparam(str(config_path))
    validate_streaming_dpcrn_config(config)
    model_dict = load_recipe(config_path).model
    system_model = init_siso_model(model_dict)
    if checkpoint_path:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        system_model.reload_checkpoint(state_dict, load_loss_func=False)
    return create_streaming_dpcrn_model(system_model)


def _tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def export_streaming_dpcrn_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 17,
) -> dict[str, Any]:
    import onnxruntime

    frame_model = load_streaming_dpcrn_model(config_path, checkpoint_path)
    frame_model.eval()
    onnx_path = Path(onnx_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else onnx_path.with_suffix(".json")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    geometry = validate_streaming_dpcrn_config(load_hparam(str(config_path)))

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
        "model_type": "dpcrn_streaming_frame",
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
