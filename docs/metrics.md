# puresound.metrics

繁體中文版本：[metrics.zh-TW.md](metrics.zh-TW.md)

Audio quality evaluation metrics for speech enhancement and source
separation.

## Class: `Metrics`

A collection of static methods for objective audio quality assessment. Some
methods' own type hints say `np.array`, but every method ultimately routes
through `check_shape`, which calls `.detach()` — so in practice **every
input must be a `torch.Tensor`**, not a plain NumPy array.

### `check_shape(clean, enhanced, retun_as_tensor=False)`

```python
check_shape(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    retun_as_tensor: bool = False,
) -> Tuple[np.ndarray, np.ndarray]   # or (Tensor, Tensor) if retun_as_tensor=True
```

Used internally by every other method below — its behavior therefore
affects all of them:

- If either tensor's first dimension isn't `1`, keeps only channel `0`
  (`clean[0, ...]`) — for a genuinely multi-channel input this silently
  discards every channel but the first.
- Squeezes to 1-D.
- Aligns length by truncating the **longer** of the two to match the
  shorter, from the start (no cross-correlation alignment).
- Converts to NumPy, then **peak-normalizes each of the two signals
  independently** by its own `abs().max()` (`clean = clean /
  abs(clean).max()`, same for `enhanced`) — the two signals are *not*
  rescaled by one shared factor. This means every metric below is computed
  on independently peak-normalized signals, not on the original absolute
  levels — including `noise_reduction`, where it makes the result a
  normalized-power ratio rather than a literal before/after energy
  comparison.
- `retun_as_tensor=True` (this is the real parameter name, not a
  typo to fix here) converts back to `torch.Tensor` before returning,
  instead of leaving NumPy arrays.

An all-zero input divides by zero here (`abs().max() == 0`), which can
surface as NaN in a downstream metric — e.g. `f1_score` on a frame-label
tensor with no positive frames at all.

### `pesq_wb(clean, enhanced) -> float`

Wide-band PESQ. Sample rate is **hardcoded to 16000 inside the function** —
there is no `sr` parameter to override it. Score range: -0.5 to 4.5 (higher
is better).

### `pesq_nb(clean, enhanced) -> float`

Narrow-band PESQ. Sample rate is **hardcoded to 8000** — no `sr` parameter
here either. Score range: 1.0 to 4.5.

### `stoi(clean, enhanced, sr=16000) -> float`

Short-Time Objective Intelligibility (STOI). Score range: 0 to 1 (higher is
better). Unlike the PESQ wrappers above, `sr` **is** a real parameter here.

### `estoi(clean, enhanced, sr=16000) -> float`

Extended STOI (`stoi(..., extended=True)`), better suited to very low SNR
conditions. Score range: 0 to 1.

### `bss_sdr(clean, enhanced) -> float`

BSS Eval Signal-to-Distortion Ratio, via
`mir_eval.separation.bss_eval_sources(clean, enhanced, False)[0][0]`
(`compute_permutation=False`, single source assumed). Returns SDR in dB.

### `sisnr(clean, enhanced) -> float`

Scale-Invariant Signal-to-Noise Ratio, computed as `si_snr(enhanced, clean)`
(`puresound.nnet.loss.sdr.si_snr`). Higher is better, in dB.

### `sisnr_imp(clean, enhanced, noisy) -> float`

SI-SNR improvement over the noisy baseline:

```
SI-SNRi = SI-SNR(enhanced, clean) - SI-SNR(noisy, clean)
```

`clean` is shape-aligned against `enhanced` and against `noisy` in two
separate `check_shape` calls — harmless when all three signals share one
length (the usual case), but worth knowing if you ever pass mismatched
lengths.

### `dnsmos_p835(clean, enhanced, sr=16000, personalized=False) -> Dict[str, float]`

Non-intrusive DNSMOS P.835 score, via
`torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore`.
**`clean` is accepted but immediately discarded** (`del clean`) — it's kept
only so the call signature matches the other metrics; DNSMOS is
reference-free and only scores `enhanced`.

`enhanced` is coerced through an internal `_mono_audio_tensor` helper first:
detached, moved to CPU, cast to float, repeatedly squeezed down from a
leading batch dim of size 1, channel 0 kept if still multi-channel, then
clamped to `[-1, 1]`. Note this does **not** go through `check_shape`, so it
is not peak-normalized the way the other metrics are.

Returns a dict, **not a tuple**:

```python
{"dnsmos_p808": ..., "dnsmos_sig": ..., "dnsmos_bak": ..., "dnsmos_ovr": ...}
```

The underlying `torchmetrics` metric instance is cached per `(sr,
personalized)` pair at module level, so repeated calls with the same
settings do not re-instantiate it. If `torchmetrics`' audio extras aren't
installed, raises `ModuleNotFoundError` with an install hint (`uv sync`, or
`librosa` + `onnxruntime` + `requests` manually).

### `f1_score(y_true, y_pred) -> Dict[str, float]`

Binary classification metrics (e.g. frame-level VAD labels) — note the
parameter names here are `y_true`/`y_pred`, not `clean`/`enhanced` like the
audio metrics above. Both still go through `check_shape` (so the same
shape/align/peak-normalize rules apply — for 0/1 labels the normalization is
a no-op as long as each tensor has at least one positive frame).

Returns a dict, **not a tuple**:

```python
{"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
```

### `noise_reduction(noisy, enhanced) -> Tensor`

Note the first parameter is `noisy`, not `clean`. Power ratio in dB between
the (peak-normalized, per `check_shape`) enhanced and noisy signals:

```
10 * log10(sum(enhanced ** 2) / sum(noisy ** 2))
```

## Example

```python
from puresound.metrics import Metrics

sisnr_val = Metrics.sisnr(clean_wav, enhanced_wav)
pesq_val  = Metrics.pesq_wb(clean_wav, enhanced_wav)
stoi_val  = Metrics.stoi(clean_wav, enhanced_wav, sr=16000)
dnsmos    = Metrics.dnsmos_p835(clean_wav, enhanced_wav)  # clean_wav is ignored
```
