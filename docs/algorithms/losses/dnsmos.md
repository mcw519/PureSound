# puresound.nnet.loss.dnsmos

繁體中文版本：[dnsmos.zh-TW.md](dnsmos.zh-TW.md)

DNSMOS P.835 (Reddy et al., 2022) as a differentiable training term. The
published `sig_bak_ovr.onnx` judge is executed with torch ops through
[`OnnxModule`](../models/lobe/onnx_module.md), so the enhancer receives a
gradient from the same network the evaluation reports SIG / BAK / OVRL from.
No clean reference is needed: the term works on real recordings.

## Class: `DNSMOSLoss`

### What it computes

Per row of the enhanced batch:

```
x      = row brought to reference_rms by a gradient-free factor (clamped to [1/max_scale, max_scale])
raw    = judge(tile(x, 144160 samples))             # 9.01 s at 16 kHz; sig, bak, ovr before calibration
MOS_k  = polyfit_k(raw_k)                           # the metric's own calibration, so values are MOS units
loss   = sum_k w_k * (5 - MOS_k) / 4                # over the scores with a non-zero weight
       + floor_weight * relu(sig_floor - MOS_sig)   # only when sig_floor is set
```

averaged over the rows that were scored. A row whose RMS is below
`silence_rms` is not sent to the judge (inside the graph it would divide 0 by
0 and poison every row's gradient); a batch with nothing to score returns
`enh.sum() * 0.0`, a zero that keeps a graph edge. Rows shorter than 9.01 s
are tiled, longer rows use their first 9.01 s. The judge runs in fp32 with
autocast disabled and is kept in eval mode whatever the parent module does.

The judge scores absolute level along with everything else (its input is the
raw log power spectrogram): on real recordings a 3 dB change moves SIG by
0.2-0.3 on average. Fixing the level before scoring is what keeps the term
from rewarding a quieter or louder output instead of a cleaner one.

### Constructor

```python
DNSMOSLoss(
    model_path: str = "~/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx",  # torchmetrics' download location
    sig_weight: float = 0.0,      # weights on (5 - MOS)/4 per score
    bak_weight: float = 1.0,
    ovr_weight: float = 0.0,
    sig_floor: float | None = None,   # hinge below this SIG (MOS units)
    floor_weight: float = 1.0,
    sample_rate: int = 16000,     # anything else raises; the loss does not resample
    max_rows: int | None = None,  # score a random subset of rows per call (each row costs ~0.45 GB of activations)
    reference_rms: float | None = 0.05,  # -26 dBFS; None scores rows at their natural level
    silence_rms: float = 1e-3,    # rows quieter than this are not scored
    max_scale: float = 20.0,      # cap on the per-row level factor
)
```

The DNSMOS file is downloaded by torchmetrics the first time any DNSMOS
metric runs; a missing file raises `FileNotFoundError`. A configuration with
no weight and no floor raises `ValueError`.

### Inputs

`forward(enh) -> Tensor`, waveform `[N, T]` (a singleton channel axis is
squeezed). `required_inputs = ("enhanced",)`.

`scores(wav) -> (dict, mask)` returns the calibrated `sig` / `bak` / `ovr`
tensors `[N]` with gradient and the `[N]` mask of rows that were scored; an
excluded row reads 0.

### Config usage

A no-reference judge can be gamed on its own, so the term belongs beside the
reference-based terms and a learned critic, with a modest weight:

```yaml
loss_func:
  - type: DNSMOSLoss
    weighted: 1.0
    args:
      sig_weight: 1.0
      bak_weight: 1.0
      max_rows: 6
```
