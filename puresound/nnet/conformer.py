"""Streaming, chunk-based TS-Conformer backbone for near-field voice isolation.

Design (see egs/voice_isolate): a single-output, distance-cued foreground
extractor built around two-stage (time + frequency) Conformer blocks, mirroring
the mobile distance-separation architecture of arXiv:2501.03045 (CMGAN-style
dual decoder) but reworked for low-latency streaming:

    [N,2,256,T] re/im
      -> ComplexCompress         power-law magnitude compression (internal feats)
      -> DenseEncoder            per-frame freq down-sample 256->64, ch 2->C
      -> n_blocks x [ DistanceFiLM (zero-init residual, per-cell mult+add)
                      TSConformerBlock ]
            TimeConformer         block-causal MHSA (left KV cache) + causal conv
            FreqConformer         intra-frame full freq MHSA + freq conv
      -> DualDecoder             magnitude mask + complex residual, fused in
                                 spectrum space using the *uncompressed* input
      -> enhanced spectrum [N,2,256,T]   (mask_type=mapping)
      VADHead -> self.last_vad_logits [N,T]

Streaming contract: the only modules that carry time state are the TimeConformer
blocks (attention KV cache + depthwise-conv ring buffer) and the VAD head; the
encoder/decoder use time-kernel=1 (per-frame) so they are stateless in time, and
the frequency axis is fully intra-frame.

Latency: attention is chunk-causal -- frame ``t`` attends within its own chunk
(plus cached left context), so a frame early in a chunk sees up to
``chunk_size - 1`` future frames for free, and the chunk boundary is shared by
all blocks so this lookahead does NOT compound with depth. The algorithmic
latency of chunked streaming is therefore ``chunk_size`` frames of buffering
(``chunk_size x 10 ms`` at 16 kHz / 160 hop). ``right_lookahead_frames > 0`` adds
an extra cross-chunk overhang that DOES compound across blocks (total
``n_blocks x R`` frames); leave it at 0 unless you accept that cost.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dparn import DistanceEmbeddingGenerator


def build_block_causal_mask(
    n_frames: int,
    chunk_size: int,
    left_context_chunks: int,
    right_lookahead_frames: int,
    time_causal: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Additive attention mask [T, T] for block-causal streaming attention.

    Query frame ``t`` (in chunk ``c = t // chunk_size``) may attend to key frame
    ``s`` iff ``chunk_start - L <= s <= chunk_end + R`` where ``chunk_start`` is
    ``c * chunk_size``, ``chunk_end`` is ``chunk_start + chunk_size - 1`` and
    ``L = left_context_chunks * chunk_size``. ``left_context_chunks <= 0`` means
    unbounded left context. Allowed positions are 0.0, disallowed -inf.

    Using the *same* mask in full-utterance training and in the per-chunk
    streaming loop is what guarantees train/stream parity.

    ``time_causal=False`` returns an all-zero (fully non-causal) mask so the time
    attention sees the whole utterance bidirectionally. This is the offline
    upper-bound mode (NOT streamable) used to prove a separation task is even
    achievable; the chunk/context args are ignored in that case.
    """
    if not time_causal:
        return torch.zeros(n_frames, n_frames, device=device)
    idx = torch.arange(n_frames, device=device)
    chunk_id = idx // chunk_size
    chunk_start = chunk_id * chunk_size
    chunk_end = chunk_start + (chunk_size - 1)

    key = idx.unsqueeze(0)  # [1, S]
    upper = (chunk_end + right_lookahead_frames).unsqueeze(1)  # [T, 1]
    allowed = key <= upper
    if left_context_chunks > 0:
        lower = (chunk_start - left_context_chunks * chunk_size).unsqueeze(1)
        allowed = allowed & (key >= lower)

    mask = torch.zeros(n_frames, n_frames, device=device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask


class ChannelNorm2d(nn.Module):
    """LayerNorm over the channel dim for a [N, C, F, T] tensor, per (N, F, T)
    cell. Unlike GroupNorm/InstanceNorm it never pools over the time axis, so it
    is strictly per-frame and safe for causal streaming."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class ComplexCompress(nn.Module):
    """Power-law magnitude compression that preserves phase. Stateless / per-point.

    ``y = mag**alpha * [cos, sin]`` with ``mag = sqrt(re^2 + im^2)``. Used only to
    feed the network; the decoder reconstructs against the *uncompressed* input.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2, F, T]
        re, im = x[:, 0:1], x[:, 1:2]
        mag = torch.sqrt(re * re + im * im + self.eps)
        gain = mag.pow(self.alpha) / mag
        return torch.cat([re * gain, im * gain], dim=1)


class _FreqConvNorm(nn.Module):
    """Conv2d on (freq, time) with time-kernel=1 (per-frame), symmetric freq pad,
    LayerNorm over channels, and activation. Stateless in time."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_f: int = 3,
        stride_f: int = 1,
        dilation_f: int = 1,
        activation: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        pad_f = (kernel_f // 2) * dilation_f
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_f, 1),
            stride=(stride_f, 1),
            padding=(pad_f, 0),
            dilation=(dilation_f, 1),
        )
        # ChannelNorm2d normalises across channels per (F,T) cell. On a 1-channel
        # output (the magnitude mask) that DEGENERATES to a constant (mean==x,
        # var==0 -> (x-mean)/sqrt(eps)==0 -> bias), so the mask cannot vary with
        # input/freq/time and the model is forced into a constant passthrough.
        # On a 2-channel (re,im) output it also distorts the spectrum. The output
        # projection must therefore be a bare conv (norm=False).
        self.norm = ChannelNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.PReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DenseEncoder(nn.Module):
    """Per-frame frequency encoder: 2 -> C channels, F -> F // freq_down_factor.

    A dilated dense stack on the frequency axis followed by ``log2(freq_down)``
    stride-2 frequency-downsampling convs. All time kernels are 1, so this module
    holds no streaming state. Returns the bottleneck plus skip activations for the
    decoder.
    """

    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        n_dense_layers: int,
        dense_dilations: Tuple[int, ...],
        freq_down_factor: int,
    ):
        super().__init__()
        self.in_proj = _FreqConvNorm(in_channels, enc_channels, kernel_f=1)

        self.dense = nn.ModuleList()
        for i in range(n_dense_layers):
            dil = dense_dilations[i % len(dense_dilations)]
            # dense: input is enc_channels*(i+1) concatenated, output enc_channels
            self.dense.append(
                _FreqConvNorm(
                    enc_channels * (i + 1),
                    enc_channels,
                    kernel_f=3,
                    dilation_f=dil,
                )
            )

        n_down = 0
        f = freq_down_factor
        while f > 1:
            assert f % 2 == 0, "freq_down_factor must be a power of 2"
            n_down += 1
            f //= 2
        self.down = nn.ModuleList(
            [
                _FreqConvNorm(enc_channels, enc_channels, kernel_f=3, stride_f=2)
                for _ in range(n_down)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.in_proj(x)
        feats = [x]
        for layer in self.dense:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        x = feats[-1]

        skips = []
        for layer in self.down:
            skips.append(x)
            x = layer(x)
        return x, skips


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        return self.net(x)


class FreqConformer(nn.Module):
    """Intra-frame Conformer over the frequency axis. Non-causal over F, but
    fully present every frame, so it carries no time state."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_expansion: int,
        conv_kernel_freq: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        pad = conv_kernel_freq // 2
        self.dwconv = nn.Conv1d(dim, dim, conv_kernel_freq, padding=pad, groups=dim)
        self.pwconv1 = nn.Conv1d(dim, 2 * dim, 1)
        self.pwconv2 = nn.Conv1d(dim, dim, 1)
        self.conv_act = nn.SiLU()
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> operate over F per (N, T)
        n, c, f, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(n * t, f, c)  # [N*T, F, C]

        x = x + 0.5 * self.ffn1(x)

        residual = x
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = residual + h

        residual = x
        h = self.conv_norm(x).transpose(1, 2)  # [B, C, F]
        h = self.pwconv1(h)
        h = F.glu(h, dim=1)
        h = self.conv_act(self.dwconv(h))
        h = self.pwconv2(h).transpose(1, 2)
        x = residual + h

        x = x + 0.5 * self.ffn2(x)
        x = self.out_norm(x)

        x = x.reshape(n, t, f, c).permute(0, 3, 2, 1)  # [N, C, F, T]
        return x


class TimeConformer(nn.Module):
    """Block-causal Conformer over the time axis, per frequency bin.

    Attention is masked block-causally (chunk + left context + right lookahead).
    The depthwise temporal conv is strictly causal (left pad only). This module
    owns the streaming time state: attention K/V cache and the conv ring buffer.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_expansion: int,
        conv_kernel_time: int,
        dropout: float,
        chunk_size: int,
        left_context_chunks: int,
        right_lookahead_frames: int,
        time_causal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.conv_kernel_time = conv_kernel_time
        self.chunk_size = chunk_size
        self.left_context_chunks = left_context_chunks
        self.right_lookahead_frames = right_lookahead_frames
        self.time_causal = time_causal

        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, 2 * dim, 1)
        # causal depthwise conv: pad manually on the left in forward
        self.dwconv = nn.Conv1d(dim, dim, conv_kernel_time, padding=0, groups=dim)
        self.pwconv2 = nn.Conv1d(dim, dim, 1)
        self.conv_act = nn.SiLU()
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> operate over T per (N, F)
        n, c, f, t = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(n * f, t, c)  # [N*F, T, C]

        seq = seq + 0.5 * self.ffn1(seq)

        residual = seq
        h = self.attn_norm(seq)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        seq = residual + h

        residual = seq
        h = self.conv_norm(seq).transpose(1, 2)  # [B, C, T]
        h = F.glu(self.pwconv1(h), dim=1)
        if self.time_causal:
            h = F.pad(h, (self.conv_kernel_time - 1, 0))  # causal left pad
        else:
            # non-causal upper bound: symmetric pad so the temporal conv sees
            # both past and future (offline only, not streamable).
            pad_l = (self.conv_kernel_time - 1) // 2
            pad_r = (self.conv_kernel_time - 1) - pad_l
            h = F.pad(h, (pad_l, pad_r))
        h = self.conv_act(self.dwconv(h))
        h = self.pwconv2(h).transpose(1, 2)
        seq = residual + h

        seq = seq + 0.5 * self.ffn2(seq)
        seq = self.out_norm(seq)

        return seq.reshape(n, f, t, c).permute(0, 3, 1, 2)  # [N, C, F, T]


class TSConformerBlock(nn.Module):
    def __init__(self, time_conformer: TimeConformer, freq_conformer: FreqConformer):
        super().__init__()
        self.time_conformer = time_conformer
        self.freq_conformer = freq_conformer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.time_conformer(x, attn_mask)
        x = self.freq_conformer(x)
        return x


class _FreqUpConvNorm(nn.Module):
    """ConvTranspose2d on freq (stride-2 up), time-kernel=1. Stateless in time."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_f: int = 3, activation: bool = True
    ):
        super().__init__()
        pad = kernel_f // 2
        op = 2 - kernel_f + 2 * pad  # output_padding so out_f = in_f * 2
        self.conv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_f, 1),
            stride=(2, 1),
            padding=(pad, 0),
            output_padding=(op, 0),
        )
        self.norm = ChannelNorm2d(out_ch)
        self.act = nn.PReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _DecoderBranch(nn.Module):
    """Upsample bottleneck back to (out_ch, F, T) with skip connections.

    Mirrors DenseEncoder.down with transposed freq convs; time-kernel=1 keeps it
    stateless in time. Each up step upsamples then fuses the matching-resolution
    encoder skip via a 1x1 conv. ``out_ch`` is 1 for the magnitude-mask branch and
    2 for the complex-residual branch.
    """

    def __init__(self, enc_channels: int, n_up: int, out_ch: int):
        super().__init__()
        self.up = nn.ModuleList(
            [_FreqUpConvNorm(enc_channels, enc_channels) for _ in range(n_up)]
        )
        self.fuse = nn.ModuleList(
            [
                _FreqConvNorm(enc_channels * 2, enc_channels, kernel_f=1)
                for _ in range(n_up)
            ]
        )
        # bare conv: no ChannelNorm (degenerates a 1-ch mask to a constant) and
        # no activation -- the raw mask logit / complex spectrum prediction.
        self.out_proj = _FreqConvNorm(
            enc_channels, out_ch, kernel_f=1, activation=False, norm=False
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # skips are encoder pre-downsample activations, fine -> coarse: [F, F/2, ...]
        for i, (up, fuse) in enumerate(zip(self.up, self.fuse)):
            x = up(x)  # upsample freq x2
            skip = skips[-i - 1]  # matching (now upsampled) resolution
            x = fuse(torch.cat([x, skip], dim=1))
        return self.out_proj(x)


class VADHead(nn.Module):
    """Frame-level speech-activity logits from the bottleneck. Causal in time."""

    def __init__(self, enc_channels: int, hidden: int, kernel_t: int):
        super().__init__()
        self.kernel_t = kernel_t
        self.proj = nn.Linear(enc_channels, hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_t, padding=0, groups=1)
        self.act = nn.SiLU()
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> pool over F -> [N, C, T]
        h = x.mean(dim=2)  # [N, C, T]
        h = self.proj(h.transpose(1, 2)).transpose(1, 2)  # [N, hidden, T]
        h = F.pad(h, (self.kernel_t - 1, 0))  # causal
        h = self.act(self.dwconv(h))
        return self.out(h).squeeze(1)  # [N, T]


class ScalarAuxHead(nn.Module):
    """Utterance-level scalar prediction head from bottleneck features."""

    def __init__(self, enc_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> [N]
        h = x.mean(dim=(2, 3))
        return self.net(h).squeeze(-1)


class DistanceFiLM(nn.Module):
    """Zero-init residual FiLM that conditions a ``[N, C, F, T]`` feature map on a
    distance embedding.

    The scale and bias are produced per ``(F, T)`` cell from that cell's own
    feature concatenated with the distance embedding, so the modulation is
    multiplicative AND input-dependent: it can pass, attenuate or shift a channel
    differently at every time/frequency cell. A single broadcast additive bias
    (the previous mechanism) could only shift the whole map by a constant and so
    could not express "follow the near talker, drop the far one" frame by frame.

    The two ``1x1`` convs and the per-cell LayerNorm carry no time state, so the
    module is identical applied to a full utterance or chunk by chunk -- it is
    streaming-safe and preserves train/stream parity.

    Both conv weights are zero-initialised, so at the start the residual is
    exactly zero and the block is a no-op (DPARN parity); the conditioning grows
    from there, but -- unlike the old single entrance bias -- it is injected
    before every Conformer block.
    """

    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.to_scale = nn.Conv1d(channels + embed_dim, channels, 1, bias=False)
        self.to_bias = nn.Conv1d(channels + embed_dim, channels, 1, bias=False)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_bias.weight)

    def forward(self, feat: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
        n, c, f, t = feat.shape
        flat = feat.reshape(n, c, f * t)
        h = self.norm(flat.transpose(1, 2)).transpose(
            1, 2
        )  # LayerNorm over C, per cell
        cond = d_emb.unsqueeze(-1).expand(-1, -1, f * t)
        h = torch.cat([h, cond], dim=1)
        scale = self.to_scale(h)
        bias = self.to_bias(h)
        out = flat + scale * flat + bias  # zero-init -> out == flat at start
        return out.reshape(n, c, f, t)


class TSConformer(nn.Module):
    """Two-stage Conformer backbone returning an enhanced complex spectrum.

    Use with ``mask_type: mapping`` in EncDecMaskBase: ``forward`` returns the
    final complex spectrum ``[N, 2, F, T]`` directly (no external masking step).

    Args:
        input_dim: number of frequency bins seen by the backbone (FFT//2 + 1 - 1
            with ``drop_stft_first_bin=True``; 256 for a 512-point FFT).
        distance_embedding_dim: 0 disables distance conditioning. When > 0 the
            scalar query distance is mapped to an embedding and fused before every
            Conformer block by a zero-init residual FiLM (DistanceFiLM): per-cell
            multiplicative + additive modulation, no-op at init.
    """

    def __init__(
        self,
        input_dim: int = 256,
        in_channels: int = 2,
        enc_channels: int = 48,
        n_dense_layers: int = 2,
        dense_dilations: Tuple[int, ...] = (1, 2),
        freq_down_factor: int = 4,
        n_blocks: int = 4,
        attn_heads: int = 4,
        ffn_expansion: int = 2,
        conv_kernel_time: int = 4,
        conv_kernel_freq: int = 31,
        dropout: float = 0.1,
        chunk_size: int = 4,
        left_context_chunks: int = 25,
        right_lookahead_frames: int = 0,
        time_causal: bool = True,
        compress_alpha: float = 0.5,
        dual_decoder: bool = True,
        output_head: str = "mask",
        distance_embedding_dim: int = 0,
        distance_max_metres: float = 2.0,
        vad_head: Optional[Dict] = None,
        background_vad_head: Optional[Dict] = None,
        aux_heads: Optional[Dict] = None,
        far_decoder: Optional[Dict] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.n_dense_layers = n_dense_layers
        self.dense_dilations = tuple(dense_dilations)
        self.freq_down_factor = freq_down_factor
        self.n_blocks = n_blocks
        self.attn_heads = attn_heads
        self.ffn_expansion = ffn_expansion
        self.conv_kernel_time = conv_kernel_time
        self.conv_kernel_freq = conv_kernel_freq
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.left_context_chunks = left_context_chunks
        self.right_lookahead_frames = right_lookahead_frames
        self.time_causal = time_causal
        self.compress_alpha = compress_alpha
        self.dual_decoder = dual_decoder
        self.distance_embedding_dim = distance_embedding_dim
        self.distance_max_metres = distance_max_metres
        self.vad_head_args = vad_head
        self.background_vad_head_args = background_vad_head
        self.aux_heads_args = aux_heads
        self.far_decoder_args = far_decoder
        self.eps = eps

        self.compress = ComplexCompress(alpha=compress_alpha, eps=eps)
        self.encoder = DenseEncoder(
            in_channels,
            enc_channels,
            n_dense_layers,
            self.dense_dilations,
            freq_down_factor,
        )
        n_down = len(self.encoder.down)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            time_conformer = TimeConformer(
                enc_channels,
                attn_heads,
                ffn_expansion,
                conv_kernel_time,
                dropout,
                chunk_size,
                left_context_chunks,
                right_lookahead_frames,
                time_causal,
            )
            freq_conformer = FreqConformer(
                enc_channels, attn_heads, ffn_expansion, conv_kernel_freq, dropout
            )
            self.blocks.append(TSConformerBlock(time_conformer, freq_conformer))

        # Output head. "mask" (default): sigmoid magnitude mask * noisy + complex
        # residual -- has a trivial constant-mask passthrough that collapses the
        # encoder (see FAILED_NOTE.md). "mapping": one decoder emits the full
        # [N,2,F,T] enhanced spectrum directly (no noisy multiply), so a constant
        # cannot reconstruct the target and the encoder/blocks MUST carry signal.
        self.output_head = output_head
        if output_head in ("mapping", "mapping_raw", "complex_mask"):
            # both use a single 2-ch decoder; "mapping" treats it as the full
            # spectrum, "complex_mask" as a complex ratio mask applied to noisy.
            self.mask_decoder = None
            self.complex_decoder = _DecoderBranch(enc_channels, n_down, out_ch=2)
        elif output_head == "mask":
            self.mask_decoder = _DecoderBranch(enc_channels, n_down, out_ch=1)
            if dual_decoder:
                self.complex_decoder = _DecoderBranch(enc_channels, n_down, out_ch=2)
            else:
                self.complex_decoder = None
        else:
            raise ValueError(f"unknown output_head: {output_head!r} (use 'mask' or 'mapping')")

        # Optional far ("background parent") decoder (DISTANCE_PARENT P1): a
        # second head off the SAME bottleneck that predicts the far/interferer
        # speech. Training-only -- forward still returns only the near spectrum;
        # the far spectrum is stashed in ``last_far_spec`` for the system to
        # iSTFT and route to FarReconstructionLoss. Mirrors the near head, so it
        # inherits the same ChannelNorm-free out_proj (see FAILED_NOTE.md §1).
        if far_decoder is not None and far_decoder.get("enabled", False):
            if output_head in ("mapping", "mapping_raw", "complex_mask"):
                self.far_mask_decoder = None
                self.far_complex_decoder = _DecoderBranch(enc_channels, n_down, out_ch=2)
            else:
                self.far_mask_decoder = _DecoderBranch(enc_channels, n_down, out_ch=1)
                self.far_complex_decoder = (
                    _DecoderBranch(enc_channels, n_down, out_ch=2) if dual_decoder else None
                )
        else:
            self.far_mask_decoder = None
            self.far_complex_decoder = None

        if distance_embedding_dim > 0:
            self.distance_embedding = DistanceEmbeddingGenerator(
                out_dim=distance_embedding_dim,
                max_distance=distance_max_metres,
            )
            # One zero-init residual FiLM per block (replaces the single broadcast
            # additive bias that previously sat once at the bottleneck entrance).
            self.distance_films = nn.ModuleList(
                [
                    DistanceFiLM(enc_channels, distance_embedding_dim)
                    for _ in range(n_blocks)
                ]
            )
        else:
            self.distance_embedding = None
            self.distance_films = None

        if vad_head is not None and vad_head.get("enabled", False):
            self.vad_head = VADHead(
                enc_channels,
                hidden=vad_head.get("hidden", enc_channels),
                kernel_t=vad_head.get("kernel_t", conv_kernel_time),
            )
        else:
            self.vad_head = None

        if background_vad_head is not None and background_vad_head.get(
            "enabled", False
        ):
            self.background_vad_head = VADHead(
                enc_channels,
                hidden=background_vad_head.get("hidden", enc_channels),
                kernel_t=background_vad_head.get("kernel_t", conv_kernel_time),
            )
        else:
            self.background_vad_head = None

        self.aux_heads = nn.ModuleDict()
        if aux_heads is not None and aux_heads.get("enabled", False):
            head_cfg = aux_heads.get("heads", {})
            for name, cfg in head_cfg.items():
                self.aux_heads[name] = ScalarAuxHead(
                    enc_channels,
                    hidden=cfg.get("hidden", enc_channels),
                )

        # populated each forward so EncDecMaskBase can route it to a BCE loss
        self.last_vad_logits: Optional[torch.Tensor] = None
        self.last_background_vad_logits: Optional[torch.Tensor] = None
        self.last_aux_outputs: Dict[str, torch.Tensor] = {}
        # far-decoder spectrum, stashed each training forward for the system to
        # iSTFT + route to FarReconstructionLoss (None when disabled / at eval).
        self.last_far_spec: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        query_distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [N, 2, F, T] (re, im). Returns enhanced spectrum [N, 2, F, T]."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Normalise to standard (channels-first) contiguous layout. A
        # channels_last input makes the 1x1 ``in_proj`` conv emit gradients with
        # non-standard strides, which DDP's reducer flags as a grad/bucket-view
        # stride mismatch ("Grad strides do not match bucket view strides").
        # Forcing contiguity here keeps every downstream conv's grad layout
        # matching its bucket view; the copy is cheap (2-channel input).
        x = x.contiguous()

        x_orig = x  # uncompressed noisy spectrum, used for reconstruction
        re, im = x_orig[:, 0:1], x_orig[:, 1:2]
        mag = torch.sqrt(re * re + im * im + self.eps)
        cos_p = re / mag
        sin_p = im / mag

        feat = self.compress(x_orig)
        feat, skips = self.encoder(feat)  # [N, C, F', T]

        d_emb = None
        if self.distance_embedding is not None and query_distance is not None:
            d_emb = self.distance_embedding(query_distance.to(feat.dtype))  # [N, E]

        attn_mask = build_block_causal_mask(
            feat.shape[-1],
            self.chunk_size,
            self.left_context_chunks,
            self.right_lookahead_frames,
            time_causal=self.time_causal,
            device=feat.device,
        )
        for i, block in enumerate(self.blocks):
            if d_emb is not None:
                feat = self.distance_films[i](feat, d_emb)
            feat = block(feat, attn_mask)

        if self.vad_head is not None:
            self.last_vad_logits = self.vad_head(feat)
        else:
            self.last_vad_logits = None
        if self.background_vad_head is not None:
            self.last_background_vad_logits = self.background_vad_head(feat)
        else:
            self.last_background_vad_logits = None
        self.last_aux_outputs = {
            name: head(feat) for name, head in self.aux_heads.items()
        }

        # Direct complex mapping: the decoder reconstructs the enhanced spectrum
        # from `feat` with no noisy multiply, so a constant output cannot
        # passthrough -- this forces the encoder/blocks to carry the input
        # signal (the "mask"/"complex_mask" heads have an identity-passthrough
        # attractor that starves the encoder until it collapses to constant).
        # The decoder predicts in the COMPRESSED magnitude domain (~unit scale,
        # matching its normalised features) and we decompress to the raw
        # spectrum; predicting raw large-magnitude bins directly fails to train.
        near = self._apply_head(
            feat, skips, re, im, mag, cos_p, sin_p,
            self.mask_decoder, self.complex_decoder,
        )

        # Far parent: training-only side output (DISTANCE_PARENT P1). Skipped at
        # eval/inference (self.training False) so deployment cost is unchanged.
        if self.training and (
            self.far_mask_decoder is not None or self.far_complex_decoder is not None
        ):
            self.last_far_spec = self._apply_head(
                feat, skips, re, im, mag, cos_p, sin_p,
                self.far_mask_decoder, self.far_complex_decoder,
            )
        else:
            self.last_far_spec = None

        return near

    def _apply_head(
        self, feat, skips, re, im, mag, cos_p, sin_p, mask_decoder, complex_decoder
    ) -> torch.Tensor:
        """Decode bottleneck features into an enhanced [N,2,F,T] spectrum for one
        head. Shared by the near and far parents so both go through identical
        output logic (and the same ChannelNorm-free out_proj)."""
        if self.output_head == "mapping":
            # decoder predicts the COMPRESSED magnitude domain; decompress here.
            y_c = complex_decoder(feat, skips)  # [N, 2, F, T] compressed
            yc_re, yc_im = y_c[:, 0:1], y_c[:, 1:2]
            mag_c = torch.sqrt(yc_re * yc_re + yc_im * yc_im + self.eps)
            gain = mag_c.pow(1.0 / self.compress_alpha - 1.0)  # mag_c^(1/a)/mag_c
            return torch.cat([yc_re * gain, yc_im * gain], dim=1)

        if self.output_head == "mapping_raw":
            # decoder output IS the enhanced re/im directly -- no noisy multiply
            # (no identity attractor) and no power decompression. Use with a
            # SCALE-INVARIANT loss (SI-SNR) so absolute scale need not be predicted.
            return complex_decoder(feat, skips)  # [N, 2, F, T]

        if self.output_head == "complex_mask":
            # complex ratio mask: enh = (m_re + j m_im) * (re + j im). Scale is
            # inherited from the noisy spectrum; a constant (1, 0) mask is
            # passthrough, so separation requires input-dependent masks.
            cm = complex_decoder(feat, skips)  # [N, 2, F, T]
            m_re, m_im = cm[:, 0:1], cm[:, 1:2]
            y_real = m_re * re - m_im * im
            y_imag = m_re * im + m_im * re
            return torch.cat([y_real, y_imag], dim=1)

        # magnitude-mask branch (pure attenuation, [0, 1]); keeps noisy phase
        mask = torch.sigmoid(mask_decoder(feat, skips))  # [N, 1, F, T]
        mag_hat = mask * mag
        y_real = mag_hat * cos_p
        y_imag = mag_hat * sin_p

        if complex_decoder is not None:
            residual = complex_decoder(feat, skips)  # [N, 2, F, T]
            y_real = y_real + residual[:, 0:1]
            y_imag = y_imag + residual[:, 1:2]

        return torch.cat([y_real, y_imag], dim=1)  # [N, 2, F, T]

    @property
    def get_args(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "in_channels": self.in_channels,
            "enc_channels": self.enc_channels,
            "n_dense_layers": self.n_dense_layers,
            "dense_dilations": list(self.dense_dilations),
            "freq_down_factor": self.freq_down_factor,
            "n_blocks": self.n_blocks,
            "attn_heads": self.attn_heads,
            "ffn_expansion": self.ffn_expansion,
            "conv_kernel_time": self.conv_kernel_time,
            "conv_kernel_freq": self.conv_kernel_freq,
            "dropout": self.dropout,
            "chunk_size": self.chunk_size,
            "left_context_chunks": self.left_context_chunks,
            "right_lookahead_frames": self.right_lookahead_frames,
            "time_causal": self.time_causal,
            "compress_alpha": self.compress_alpha,
            "dual_decoder": self.dual_decoder,
            "output_head": self.output_head,
            "distance_embedding_dim": self.distance_embedding_dim,
            "distance_max_metres": self.distance_max_metres,
            "vad_head": self.vad_head_args,
            "background_vad_head": self.background_vad_head_args,
            "aux_heads": self.aux_heads_args,
            "far_decoder": self.far_decoder_args,
        }
