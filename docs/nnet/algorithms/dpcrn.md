# puresound.nnet.dpcrn

DPCRN — dual-path convolutional recurrent network on a `Unet` chassis: CNN
down/up stacks over frequency with two `DPRNNblock2D` blocks at the bottleneck
(bidirectional intra-frequency LSTM, unidirectional inter-time LSTM, optional
FiLM conditioning). The released voice-isolate backbone.

## Class: `DPCRN`

### Constructor

```python
DPCRN(
    input_dim: int = 512,
    dvec_dim: Optional[int] = None,        # speaker-embedding FiLM conditioning
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),   # CNN stack; channels[-1] = bottleneck C
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t / stride_t / dilation_t: Tuple,      # per down-layer, time axis
    kernel_f / stride_f / dilation_f: Tuple,      # per down-layer, frequency axis
    delay: Tuple = (0, 0, ...),            # per-layer look-ahead; sum(delay) frames of
                                           # algorithmic latency when streaming
    rnn_hidden: int = 128,
    spectral_compress: bool = False,
    vad_head:  Optional[Dict] = None,      # {enabled, hidden, kernel_t} -> lobe.heads.VADHead
    dist_head: Optional[Dict] = None,      # {enabled, hidden}           -> lobe.heads.DistHead
)
```

`forward(x [N, CH, C, T], dvec=None) -> [N, CH, C, T]` predicts a mask in the
feature domain (`EncDecMaskBase` applies it). When the heads are enabled the
backbone exposes side outputs after each forward: `last_vad_logits [N, T]` and
`last_dist_preds [N, 3]` (see [nnet.lobe.heads](../lobe/heads.md)); both default
to disabled and add no cost otherwise.

**Streaming**: `delay=[0,...]` streams bit-exact with zero latency;
`delay > 0` (the released recipes use `[1,1,1]` = 30 ms at hop 160) is exported
with future-buffering baked into the ONNX graph — see
[streaming/dpcrn_onnx](../../streaming/dpcrn_onnx.md).

## Class: `DPRNNblock2D`

The bottleneck block (intra-frequency BiLSTM + inter-time LSTM + optional FiLM).

```python
DPRNNblock2D(input_size: int, hidden_size: int, dropout: float = 0.0,
             embedding_size: Optional[int] = None, fused_type: Optional[str] = None)
```
