from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from puresound.nnet.dparn import DPARN
from puresound.streaming.base import (
    IDENTITY,
    Postprocessor,
    StreamingFrameModelBase,
    StreamingOrt,
    StreamingVariant,
    as_list,
    export_streaming_onnx,
    load_streaming_model,
    require,
)
from puresound.nnet.masker import Masker


@dataclass
class DparnStreamingState:
    down_caches: list[torch.Tensor]
    up_caches: list[torch.Tensor]
    h_states: list[torch.Tensor]
    c_states: list[torch.Tensor]


def validate_streaming_dparn_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    encoder = model.get("encoder", {})
    features = model.get("features", {})
    backbone = model.get("backbone", {})
    encoder_args = encoder.get("encoder_args", {})
    backbone_args = backbone.get("backbone_args", {})

    require(dataset.get("target_sample_rate") == 16000, "streaming DPARN requires dataset.target_sample_rate=16000")
    require(encoder.get("type") == "ConvEncDec", "streaming DPARN requires ConvEncDec encoder")
    require(str(encoder_args.get("win_type", "")).lower() == "hann", "streaming DPARN requires a Hann window")
    require(encoder_args.get("sr") == 16000, "streaming DPARN requires encoder sr=16000")
    require(encoder_args.get("fmax") == 8000, "streaming DPARN requires encoder fmax=8000")
    require(encoder_args.get("trainable") is False, "streaming DPARN requires a fixed Hann frontend: encoder.trainable=False")

    # STFT geometry is now read from the config so non-default win/hop (e.g.
    # voice_isolate v6 uses hop=160) are honoured by the manifest and the
    # runtime's OLA buffer alignment.
    fft_length = int(encoder_args.get("fft_length", 512))
    win_length = int(encoder_args.get("win_length", fft_length))
    hop_length = int(encoder_args.get("hop_length", win_length // 4))
    require(win_length <= fft_length, "win_length must be <= fft_length")
    require(hop_length > 0, "hop_length must be positive")

    require(features.get("feats_type") == "complex", "streaming DPARN requires complex features")
    require(features.get("drop_stft_first_bin") is True, "streaming DPARN requires drop_stft_first_bin=True")
    require(features.get("trainable") is False, "streaming DPARN requires features.trainable=False")
    require(not features.get("include_specaug", False), "streaming DPARN does not support specaug")

    require(backbone.get("type") == "DPARN", "streaming DPARN requires a DPARN backbone")
    require(backbone_args.get("input_dim") == 256, "streaming DPARN requires input_dim=256")
    # bN2d streams too: at eval BatchNorm2d uses fixed running statistics, so it
    # degenerates to a per-frame affine. DPCRN's validator already accepts it and
    # both share one runtime (StreamingDpcrnOrt is StreamingDparnOrt), so keeping
    # DPARN stricter only rejected configs that would have exported correctly --
    # including egs/noise_suppression/config/dparn.yaml, the repo's only DPARN recipe.
    require(backbone_args.get("norm_type") in {"cLN", "iLN", "bN2d"}, "streaming DPARN requires cLN/iLN/bN2d normalization")
    require(not backbone_args.get("skip_conv", False), "streaming DPARN currently expects skip_conv=False")
    require(all(v == 1 for v in as_list(backbone_args.get("stride_t", []))), "streaming DPARN requires stride_t=1 for every down layer")
    require(all(v == 1 for v in as_list(backbone_args.get("dilation_t", []))), "streaming DPARN requires dilation_t=1 for every down layer")

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
    if not all(v == 0 for v in as_list(backbone_args.get("delay", []))):
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


class StreamingDparnFrameModel(StreamingFrameModelBase):
    """One-frame DPARN feature model with explicit streaming state."""

    def __init__(self, system_model: nn.Module):
        super().__init__()
        require(hasattr(system_model, "backbone"), "system model must expose a DPARN backbone")
        require(hasattr(system_model, "feats"), "system model must expose a feature encoder")
        require(system_model.mask_type == "complex", "streaming DPARN supports complex masking only")
        require(isinstance(system_model.backbone, DPARN), "streaming frame model requires a DPARN backbone")
        self.system_model = system_model
        self.backbone: DPARN = system_model.backbone
        self.feats = system_model.feats
        self.n_down = len(self.backbone.cnn_down)
        self.n_up = len(self.backbone.cnn_up)
        self.n_blocks = len(self.backbone.dparn_block)
        self.streaming_delay = self.n_up
        self.eval()

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

    def _state_to_tuple(self, state: DparnStreamingState) -> tuple[torch.Tensor, ...]:
        return tuple(
            state.down_caches + state.up_caches + state.h_states + state.c_states
        )

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

def create_streaming_dparn_model(system_model: nn.Module) -> StreamingDparnFrameModel:
    return StreamingDparnFrameModel(system_model.eval())


#: DPARN carries no ports beyond the four standard groups, so the base class's
#: `_extra_state_names` default is exactly right and it declares nothing.
_VARIANT = StreamingVariant(
    "dparn", validate_streaming_dparn_config, StreamingDparnFrameModel
)

#: One runtime serves both backbones -- it drives the session by the port names
#: in the manifest and never looks at what produced them.
StreamingDparnOrt = StreamingOrt


def load_streaming_dparn_model(
    config_path: str | Path, checkpoint_path: str | Path | None = None
) -> StreamingDparnFrameModel:
    return load_streaming_model(_VARIANT, config_path, checkpoint_path)


def export_streaming_dparn_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 17,
    postprocess: Postprocessor = IDENTITY,
) -> dict[str, Any]:
    return export_streaming_onnx(
        _VARIANT,
        config_path,
        checkpoint_path,
        onnx_path,
        manifest_path,
        opset_version,
        postprocess,
    )
