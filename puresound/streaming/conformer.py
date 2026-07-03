"""Per-chunk streaming runner for the TSConformer backbone.

The TSConformer is designed so that, with ``right_lookahead_frames=0``, every
module is either per-frame (encoder/decoder/freq-conformer, all time-kernel=1 and
channel-wise norms) or carries a small, well-defined time state (the
TimeConformer attention KV cache + causal depthwise-conv ring buffer, and the VAD
head's causal conv). This module exploits that to process audio chunk-by-chunk
while reproducing the full-utterance forward *exactly* (see
``verify_streaming_consistency``), which both validates the streaming design and
is the basis for a later per-chunk ONNX export.

Consistency relies on:
  * chunk-causal attention (R=0): all queries in a chunk share the key window
    ``[chunk_start - L, chunk_end]``, so per-chunk attention is plain attention
    over ``[kv_cache || chunk]`` with no intra-chunk mask. The KV cache grows to
    ``L = left_context_chunks * chunk_size`` frames then slides -- matching the
    bounded training mask, including the warm-up region (cache starts empty).
  * causal depthwise convs: ring buffer of ``kernel-1`` frames, initialised to
    zeros to match the training left-pad.

ONNX note: ONNX needs fixed-shape state. The variable-length KV cache here would
become a fixed ``[B, L, C]`` buffer plus a key-padding mask covering the not-yet-
filled positions; that export is deferred until a checkpoint converges.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from puresound.nnet.conformer import TSConformer


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_streaming_tsconformer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Assert a recipe config is streaming-compatible and return STFT geometry."""
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    encoder = model.get("encoder", {})
    features = model.get("features", {})
    backbone = model.get("backbone", {})
    encoder_args = encoder.get("encoder_args", {})
    backbone_args = backbone.get("backbone_args", {})

    _require(dataset.get("target_sample_rate") == 16000, "streaming TSConformer requires dataset.target_sample_rate=16000")
    _require(encoder.get("type") == "ConvEncDec", "streaming TSConformer requires ConvEncDec encoder")
    _require(str(encoder_args.get("win_type", "")).lower() == "hann", "streaming TSConformer requires a Hann window")
    _require(encoder_args.get("trainable") is False, "streaming TSConformer requires a fixed frontend: encoder.trainable=False")
    _require(features.get("feats_type") == "complex", "streaming TSConformer requires complex features")
    _require(features.get("drop_stft_first_bin") is True, "streaming TSConformer requires drop_stft_first_bin=True")
    _require(not features.get("include_specaug", False), "streaming TSConformer does not support specaug")
    _require(backbone.get("type") == "TSConformer", "streaming wrapper requires a TSConformer backbone")
    _require(model.get("lighting_module", {}).get("module_args", {}).get("mask_type") == "mapping",
             "streaming TSConformer requires mask_type=mapping")

    r = int(backbone_args.get("right_lookahead_frames", 0))
    n_blocks = int(backbone_args.get("n_blocks", 4))
    chunk_size = int(backbone_args.get("chunk_size", 4))
    fft_length = int(encoder_args.get("fft_length", 512))
    win_length = int(encoder_args.get("win_length", fft_length))
    hop_length = int(encoder_args.get("hop_length", win_length // 4))
    frame_ms = 1000.0 * hop_length / 16000.0

    if r > 0:
        warnings.warn(
            f"right_lookahead_frames={r}: cross-chunk lookahead compounds across "
            f"blocks, so algorithmic latency is ~{n_blocks * r} frames "
            f"({n_blocks * r * frame_ms:.0f} ms) on top of the {chunk_size}-frame "
            f"({chunk_size * frame_ms:.0f} ms) chunk buffering. Set it to 0 for a "
            "clean non-compounding streaming model.",
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
        "chunk_size": chunk_size,
        "buffering_latency_ms": chunk_size * frame_ms,
        "lookahead_latency_ms": n_blocks * r * frame_ms,
    }


class StreamingTSConformer(nn.Module):
    """Chunk-by-chunk runner over a trained TSConformer backbone.

    Operates directly on complex-spectrum frames ``[N, 2, F, T]`` (the backbone's
    input/output space). Call ``reset()`` then ``process_chunk()`` repeatedly.
    """

    def __init__(self, backbone: TSConformer):
        super().__init__()
        _require(isinstance(backbone, TSConformer), "expected a TSConformer backbone")
        _require(
            getattr(backbone, "time_causal", True),
            "streaming runner requires a causal backbone (time_causal=True); "
            "a non-causal upper-bound model cannot be streamed",
        )
        _require(
            backbone.right_lookahead_frames == 0,
            "streaming runner requires right_lookahead_frames=0 (chunk-causal)",
        )
        self.bb = backbone.eval()
        self.left = backbone.left_context_chunks * backbone.chunk_size
        self.kt = backbone.conv_kernel_time
        self.reset()

    def reset(self, batch_size: int = 1, device: torch.device | str = "cpu") -> None:
        device = torch.device(device)
        n_blocks = len(self.bb.blocks)
        # variable-length attention KV caches (grow to self.left, then slide)
        self._kv = [None] * n_blocks
        # causal depthwise-conv ring buffers, zero-initialised to match left-pad
        self._conv = [None] * n_blocks
        self._vad_cache = None
        self._bs = batch_size
        self._device = device

    def _time_step(self, idx: int, x_chunk: torch.Tensor) -> torch.Tensor:
        tc = self.bb.blocks[idx].time_conformer
        n, c, f, t = x_chunk.shape
        seq = x_chunk.permute(0, 2, 3, 1).reshape(n * f, t, c)  # [N*F, T, C]
        seq = seq + 0.5 * tc.ffn1(seq)

        residual = seq
        h = tc.attn_norm(seq)
        kv = h if self._kv[idx] is None else torch.cat([self._kv[idx], h], dim=1)
        out, _ = tc.attn(h, kv, kv, need_weights=False)
        seq = residual + out
        self._kv[idx] = kv[:, -self.left :, :].detach()

        residual = seq
        hc = tc.conv_norm(seq).transpose(1, 2)  # [B, C, T]
        hc = F.glu(tc.pwconv1(hc), dim=1)
        if self._conv[idx] is None:
            self._conv[idx] = hc.new_zeros(hc.shape[0], hc.shape[1], self.kt - 1)
        hc_ctx = torch.cat([self._conv[idx], hc], dim=2)
        self._conv[idx] = hc_ctx[:, :, -(self.kt - 1) :].detach()
        hc = tc.conv_act(tc.dwconv(hc_ctx))
        hc = tc.pwconv2(hc).transpose(1, 2)
        seq = residual + hc

        seq = seq + 0.5 * tc.ffn2(seq)
        seq = tc.out_norm(seq)
        return seq.reshape(n, f, t, c).permute(0, 3, 1, 2)

    def _vad_step(self, feat: torch.Tensor) -> torch.Tensor:
        head = self.bb.vad_head
        h = feat.mean(dim=2)  # [N, C, T]
        h = head.proj(h.transpose(1, 2)).transpose(1, 2)  # [N, hidden, T]
        if self._vad_cache is None:
            self._vad_cache = h.new_zeros(h.shape[0], h.shape[1], head.kernel_t - 1)
        h_ctx = torch.cat([self._vad_cache, h], dim=2)
        self._vad_cache = h_ctx[:, :, -(head.kernel_t - 1) :].detach()
        h = head.act(head.dwconv(h_ctx))
        return head.out(h).squeeze(1)  # [N, T]

    @torch.no_grad()
    def process_chunk(
        self,
        chunk: torch.Tensor,
        query_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """chunk: [N, 2, F, Tc] complex spectrum. Returns enhanced [N, 2, F, Tc].

        The VAD logits for the chunk (if the head is enabled) are stashed on
        ``self.last_vad_logits``.
        """
        bb = self.bb
        if chunk.dim() == 3:
            chunk = chunk.unsqueeze(1)
        x_orig = chunk
        re, im = x_orig[:, 0:1], x_orig[:, 1:2]
        mag = torch.sqrt(re * re + im * im + bb.eps)
        cos_p, sin_p = re / mag, im / mag

        feat = bb.compress(x_orig)
        feat, skips = bb.encoder(feat)

        d_emb = None
        if bb.distance_embedding is not None and query_distance is not None:
            d_emb = bb.distance_embedding(query_distance.to(feat.dtype))

        # DistanceFiLM is time-stateless (1x1 convs + per-cell norm), so applying
        # it per chunk matches the full-utterance forward exactly -- parity holds.
        for i in range(len(bb.blocks)):
            if d_emb is not None:
                feat = bb.distance_films[i](feat, d_emb)
            feat = self._time_step(i, feat)
            feat = bb.blocks[i].freq_conformer(feat)

        self.last_vad_logits = self._vad_step(feat) if bb.vad_head is not None else None

        mask = torch.sigmoid(bb.mask_decoder(feat, skips))
        mag_hat = mask * mag
        y_real = mag_hat * cos_p
        y_imag = mag_hat * sin_p
        if bb.complex_decoder is not None:
            residual = bb.complex_decoder(feat, skips)
            y_real = y_real + residual[:, 0:1]
            y_imag = y_imag + residual[:, 1:2]
        return torch.cat([y_real, y_imag], dim=1)

    @torch.no_grad()
    def process_full(
        self,
        x: torch.Tensor,
        query_distance: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Run a full spectrum [N,2,F,T] through the streaming loop, chunk-by-chunk."""
        chunk_size = chunk_size or self.bb.chunk_size
        if x.dim() == 3:
            x = x.unsqueeze(1)
        t = x.shape[-1]
        outs = []
        vad_outs = []
        for start in range(0, t, chunk_size):
            outs.append(
                self.process_chunk(x[..., start : start + chunk_size], query_distance)
            )
            if self.last_vad_logits is not None:
                vad_outs.append(self.last_vad_logits)
        # expose the concatenated per-chunk VAD logits, not just the last chunk's
        self.last_vad_logits = torch.cat(vad_outs, dim=-1) if vad_outs else None
        return torch.cat(outs, dim=-1)


@torch.no_grad()
def verify_streaming_consistency(
    backbone: TSConformer,
    x: torch.Tensor,
    query_distance: torch.Tensor | None = None,
) -> float:
    """Max abs diff between the full-utterance forward and the chunk-by-chunk
    streaming run. Should be ~1e-5 when the design is correct."""
    backbone = backbone.eval()
    full = backbone(x, query_distance=query_distance)
    runner = StreamingTSConformer(backbone)
    runner.reset(batch_size=x.shape[0])
    streamed = runner.process_full(x, query_distance=query_distance)
    n = min(full.shape[-1], streamed.shape[-1])
    return (full[..., :n] - streamed[..., :n]).abs().max().item()
