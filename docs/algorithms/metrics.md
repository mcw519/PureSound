# puresound.metrics

繁體中文版本：[metrics.zh-TW.md](metrics.zh-TW.md)

Objective metrics for speech enhancement and separation:

- `puresound.metrics.Metrics` — reference-based quality and intelligibility
  scores, reference-free DNSMOS, and frame-label F1. `--scoring` runs read them
  through `WAVEFORM_METRICS` in `puresound/system/runner.py`; the gate's
  `reference` and `noreference` stages call them directly.
- `puresound.evaluation.spectral` — two measures of what a model's bottleneck
  loses (harmonic contrast, transient correlation), reported by the `reference`
  stage.
- `puresound.evaluation.tools.wer` — word error rates and their deletion
  component, used by the `wer` stage.

The benchmark protocol, paired statistics and verdicts are described in
[evaluation](../usage/evaluation.md).

## Class: `Metrics`

Static methods. Inputs are `torch.Tensor`s shaped `[1, T]` (or `[C, T]`, of
which channel 0 is scored), whatever the `np.array` type hints say.

### Input handling: `check_shape`

```python
Metrics.check_shape(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    retun_as_tensor: bool = False,   # the parameter's real spelling
) -> Tuple[np.ndarray, np.ndarray]   # (Tensor, Tensor) when retun_as_tensor=True
```

Every method except `dnsmos_p835` passes its inputs through this:

1. If the first axis is not of size 1, keep index 0 of it (`x[0, ...]`). A
   1-D `[T]` tensor therefore collapses to its first sample — pass `[1, T]`.
2. Squeeze to 1-D.
3. Truncate the longer signal to the shorter one, from the start (no
   cross-correlation alignment).
4. Convert to NumPy and **peak-normalise each signal by its own**
   `abs().max()`.

Scores are therefore independent of the output level. An all-zero input
divides by zero in step 4 and surfaces as NaN downstream.

### Reference-based scores

| method | computes | returns |
| --- | --- | --- |
| `pesq_wb(clean, enhanced)` | wide-band PESQ (ITU-T P.862.2) via the `pesq` package; the sample rate is fixed at 16000 | MOS-LQO, higher is better |
| `pesq_nb(clean, enhanced)` | narrow-band PESQ (ITU-T P.862); the sample rate is fixed at 8000, inputs must already be 8 kHz | MOS-LQO, higher is better |
| `stoi(clean, enhanced, sr=16000)` | STOI (Taal et al., 2011) via `pystoi` | `[0, 1]`, higher is better |
| `estoi(clean, enhanced, sr=16000)` | extended STOI (Jensen & Taal, 2016), `stoi(..., extended=True)` | `[0, 1]`, higher is better |
| `bss_sdr(clean, enhanced)` | BSS Eval SDR (Vincent et al., 2006), `mir_eval.separation.bss_eval_sources(clean, enhanced, False)[0][0]`, one source, no permutation search | dB |
| `sisnr(clean, enhanced)` | SI-SNR (below) | dB, float |
| `sisnr_imp(clean, enhanced, noisy)` | `SI-SNR(enhanced, clean) − SI-SNR(noisy, clean)`; `clean` is aligned against each of the two separately | dB, float |
| `noise_reduction(noisy, enhanced)` | `10 log10(Σ enhanced² / Σ noisy²)` on the peak-normalised pair — a ratio of normalised powers, not of absolute energies | `Tensor [1]` |

SI-SNR (Le Roux et al., 2019), from `puresound.nnet.loss.sdr.si_snr`, with
both signals made zero-mean:

```
s_target = <ŝ, s> / <s, s> · s
e        = ŝ − s_target
SI-SNR   = 10 log10(‖s_target‖² / ‖e‖²)
```

### `dnsmos_p835`

```python
Metrics.dnsmos_p835(
    clean: torch.Tensor,           # ignored; callers without a reference pass None
    enhanced: torch.Tensor,
    sr: int = 16000,
    personalized: bool = False,
    num_threads: int | None = None,  # onnxruntime intra/inter-op threads
) -> Dict[str, float]
# {"dnsmos_p808": ..., "dnsmos_sig": ..., "dnsmos_bak": ..., "dnsmos_ovr": ...}
```

Reference-free DNSMOS P.835 (Reddy et al., 2022) plus the P.808 overall score,
via `torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore` on CPU.
`enhanced` is detached, cast to float, reduced to one channel (leading size-1
axes squeezed, then channel 0) and clamped to `[-1, 1]`; it is **not**
peak-normalised. One metric instance is cached per `(sr, personalized,
num_threads)`.

`num_threads=None` uses every core, which suits a single process. A worker pool
must pass `1`: `torch.set_num_threads(1)` in a worker does not reach
onnxruntime, so each worker would otherwise start a full thread set and
oversubscribe the machine. `puresound.evaluation.tools.noreference` passes 1.
Without the torchmetrics audio extras the call raises `ModuleNotFoundError`
with an install hint (`uv sync`).

### `f1_score`

```python
Metrics.f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]
# {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
```

Binary frame labels (nonzero = positive), counted as TP / TN / FP / FN:
`precision = TP / (TP + FP + 1e-7)`, `recall = TP / (TP + FN + 1e-7)`,
`F1 = 2PR / (P + R + 1e-7)` clamped to `[1e-7, 1 − 1e-7]`. The labels go through
`check_shape`, so a tensor with no positive frame divides by zero there.

### Example

```python
from puresound.metrics import Metrics

clean, enhanced = clean_wav.view(1, -1), enhanced_wav.view(1, -1)
sisnr_val = Metrics.sisnr(clean, enhanced)
pesq_val  = Metrics.pesq_wb(clean, enhanced)
stoi_val  = Metrics.stoi(clean, enhanced, sr=16000)
dnsmos    = Metrics.dnsmos_p835(None, enhanced)
```

## Bottleneck measures: `puresound.evaluation.spectral`

Both are computed on a Hann-windowed, centred STFT in dB (`20 log10 |X|`) and
compare the enhanced signal against the target of the same item.

```python
harmonic_contrast_db(wav, sample_rate=16000, *, n_fft=512, hop=160,
                     band_hz=(100.0, 2000.0), active_percentile=0.6) -> float
harmonic_contrast_gap_db(enhanced, target, sample_rate=16000, **kwargs) -> float
transient_correlation(enhanced, target, sample_rate=16000, *, n_fft=128, hop=32) -> float
```

- **`harmonic_contrast_db`** keeps the frames whose mean log-magnitude is at or
  above the `active_percentile` quantile, finds local spectral peaks and valleys
  (greater or smaller than both neighbours) inside `band_hz`, and averages
  `mean(peaks) − mean(valleys)` over those frames.
- **`harmonic_contrast_gap_db`** = `contrast(enhanced) − contrast(target)`,
  reported by the `reference` stage as `harmonic_gap_db`. Negative means the
  harmonic comb was smeared.
- **`transient_correlation`** takes each signal's per-frame log energy (mean dB
  over bins) at 8 ms / 2 ms (at 16 kHz), its first difference, and returns the
  Pearson correlation of the two slopes, in `[-1, 1]`. Reported as
  `transient_corr`.

Each returns NaN when there is nothing to measure: fewer than 3 (harmonic) or 4
(transient) frames, a band narrower than 5 bins, no peak–valley pair, or zero
slope variance.

**Why.** Voiced speech is a comb of harmonics. A mask from a bottleneck too
coarse in frequency cannot follow the comb, so peaks and valleys flatten toward
each other; an energy ratio adds "noise removed" and "speech smeared" together,
this measure separates them. The gap, not the absolute contrast, is reported
because the absolute value depends on the speaker and utterance. Transients
(keyboard, door) are milliseconds long; a bottleneck too coarse in time smears
their edges. The slope rather than the envelope is correlated because two
signals with the same loudness and different attack sharpness agree on the
envelope, and the window is finer than any frame the model computes on, so the
measure sees smearing the model's own frame rate cannot.

## Word error rates: `puresound.evaluation.tools.wer`

```python
normalise(text: str) -> str                  # lower-case, "-" -> " ", drop chars outside [a-z0-9' ]
edit_counts(reference: str, hypothesis: str) -> dict   # {"sub", "del", "ins", "hit", "ref_words"}
rates(counts: dict) -> dict                  # {"wer", "del", "ins", "sub"}
loop_count(rows, ratio: float = LOOP_RATIO) -> int     # LOOP_RATIO = 1.5
```

`edit_counts` is a word-level Levenshtein alignment with backtrace (ties resolve
to match/substitution, then deletion, then insertion). With
`N = max(ref_words, 1)`:

```
WER = (S + D + I) / N      del = D / N      ins = I / N      sub = S / N
```

`loop_count` counts hypotheses longer than `ratio × ref_words` words: a
recogniser that cannot parse a segment repeats a phrase, and one such item's
insertion rate can dominate a corpus mean.

**Why.** Deletions are reported on their own because they are the failure with
a direction: substitutions and insertions also come from the recogniser, but a
rising deletion rate against the same reference is the model removing speech.
The reference is the corpus transcript, never a recogniser's output on clean
audio, which would measure agreement with itself.
