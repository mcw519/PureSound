# puresound.nnet.conv_tasnet

繁體中文版本：[conv_tasnet.zh-TW.md](conv_tasnet.zh-TW.md)

Status: library — reachable from a config as `type: ConvTasNet`, covered by
forward tests in `test/nnet/test_backbone.py`, not used by a maintained
recipe.

**Reference:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency
Magnitude Masking for Speech Separation," IEEE/ACM TASLP, 2019.

The temporal convolutional network (TCN) mask estimator of Conv-TasNet, without
the paper's learned waveform encoder/decoder. It is the backbone stage of
`Encoder -> Features -> Backbone -> Masker -> Decoder`
([nnet.features](features.md), [nnet.masker](masker.md)), so any front end
(`ConvEncDec`, `FreeEncDec`) can be paired with it. The `TCN` and `GatedTCN`
blocks are also the bottleneck of [`UnetTcn`](unet.md).

## Class: `TCN`

One residual block: `1x1 conv -> depthwise dilated conv -> 1x1 conv`, plus the
input.

```python
TCN(
    in_channels: int,        # input/output width (residual add)
    hid_channels: int,       # width inside the block
    kernel: int,
    dilation: int,
    dropout: float = 0.0,    # after the depthwise conv
    emb_dim: int = 0,        # > 0: embed is concatenated onto x before in_conv
    causal: bool = False,    # left-only padding in the depthwise conv
    tcn_norm: str = "gLN",   # norm after in_conv (lobe.norm.get_norm)
    dconv_norm: str = "gGN", # norm inside the DepthwiseSeparableConv1d
)
# forward(x [N, in_channels, T], embed [N, emb_dim] | None) -> [N, in_channels, T]
```

```
x' = cat(x, embed broadcast over T)          # only when embed is given
y  = out_conv(dropout(DSConv(PReLU(norm(in_conv(x'))))))
return y + x
```

The depthwise stage is [`DepthwiseSeparableConv1d`](lobe/cnn.md). With
`causal=True` that layer rejects the global norms `gLN` and `gGN` (they read
the whole sequence), so `dconv_norm` must be a per-frame norm such as `cLN`.

## Class: `GatedTCN`

Two parallel dilated convs multiplied together, the WaveNet-style gated
activation with PReLU in place of tanh: `left_conv` (conv, norm, PReLU,
dropout) times `right_conv` (the same, then a sigmoid).

```python
GatedTCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",   # norm inside both branches
    use_film: bool = False,  # how embed conditions right_conv (below)
)
# forward(x [N, in_channels, T], embed [N, emb_dim] | None) -> [N, in_channels, T]
```

```
h   = in_conv(x)
h_r = cat(h, embed)                                   # use_film=False
h_r = cond_scale(embed) * h + cond_bias(embed)        # use_film=True
y   = out_conv(left_conv(h) * right_conv(h_r))
return y + x
```

- `use_film=False`: `right_conv` takes `hid_channels + emb_dim` channels.
- `use_film=True`: two `1x1` convs map `embed` to a scale and a bias (FiLM);
  `right_conv` takes `hid_channels`.
- There is no `dconv_norm`.
- `causal=True`: both convs pad `(kernel - 1) * dilation` on each side and the
  trailing `(kernel - 1) * dilation` frames are cropped before the residual
  add, so each output frame depends only on the current and past inputs.

## Class: `ConvTasNet`

`repeat_tcn` stacks of `per_tcn_stack` blocks. Block `i` of every stack has
dilation `tcn_dilated_basic ** i`, so the default `per_tcn_stack=5`, base `2`
gives dilations `1, 2, 4, 8, 16` in each stack.

```python
ConvTasNet(
    input_dim: int = 512,          # feature width in and out (shape-preserving)
    embed_dim: int = 256,          # speaker-embedding width
    embed_norm: bool = False,      # L2-normalize dvec first
    tcn_layer: str = "normal",     # "normal" -> TCN, "gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,            # hid_channels of every block
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,
    repeat_tcn: int = 4,
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # per-block flag; len == per_tcn_stack (asserted)
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",       # TCN only; not passed to GatedTCN
    causal: bool = False,
)
```

`tcn_with_embed[i] == 1` makes block `i` of every stack take `dvec`
(`emb_dim=embed_dim`); the other blocks are unconditioned. `ConvTasNet` builds
`GatedTCN` with its default `use_film=False`. An unknown `tcn_layer` raises
`NameError`.

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, input_dim, T]
# dvec: [N, embed_dim], needed only when some tcn_with_embed[i] == 1
# returns [N, input_dim, T], a mask in the feature domain
```

`get_args` returns every constructor argument as a dict, used to rebuild the
model from a checkpoint.

### Example

```python
from puresound.nnet import ConvTasNet

# unconditioned
model = ConvTasNet(
    input_dim=512, embed_dim=0, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[0] * 8,
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100))                      # [1, 512, 100]

# speaker-conditioned: the first 3 of 8 blocks in each stack see dvec
model = ConvTasNet(
    input_dim=512, embed_dim=192, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100), torch.rand(1, 192))  # [1, 512, 100]
```

### Design notes

- Exponentially growing dilation gives a receptive field of
  `repeat_tcn * (kernel - 1) * (base^per_tcn_stack - 1) / (base - 1) + 1`
  frames (for base > 1) with a parameter count linear in the number of blocks.
- The encoder/decoder is left out so the same estimator can run on an STFT or
  a learned front end.
- Conditioning is per block, so a recipe chooses how early and how often the
  speaker embedding enters.
