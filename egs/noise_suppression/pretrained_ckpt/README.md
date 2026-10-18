# noise_suppression — pretrained checkpoints

Traditional Chinese: [`README.zh-TW.md`](README.zh-TW.md)

Single-channel 16 kHz speech enhancement, streaming, no enrollment. All versions
use DPCRN with a Mamba temporal path over 24 ERB bands. Benchmark stages:
[`../benchmarks/stages.md`](../benchmarks/stages.md).

```bash
uv run puresound infer noise-suppression-dpcrn-mamba-v3 \
    --input audio=in.wav --output audio=out.wav --provider cpu
```

v1/v2 use [`../config/infer_dpcrn.yaml`](../config/infer_dpcrn.yaml).
v3 requires [`../config/infer_dpcrn_mamba_wide.yaml`](../config/infer_dpcrn_mamba_wide.yaml):
channels `[2,48,96,128]` and hidden size 128, versus `[2,32,64,96]` and 96.
All streaming exports record `dry_blend: 1.0`, with no onset guard or post-graph
relief. Re-export with `--dry-blend 1.0`.

## Versions

| Version | Zoo id / role | Training | Choose it when |
|---|---|---|---|
| `dpcrn_mamba_v3.ckpt` | `noise-suppression-dpcrn-mamba-v3`, released candidate | wider model + mixed-source coverage training + ActiveBin + MetricGAN with a DNSMOS P.835 term | higher PESQ, STOI and DNSMOS matter; see the ASR and CPU trade-offs below |
| **`dpcrn_mamba_v2.ckpt`** | `noise-suppression-dpcrn-mamba-v2`, **default** | v1 + 4 epochs with a PESQ critic | balanced quality and CPU cost |
| `dpcrn_mamba_v1.ckpt` | `noise-suppression-dpcrn-mamba-v1`, candidate | compact curriculum + ActiveBin calibration | deletion on clean read speech matters most |
| `dpcrn_mamba_v0.ckpt` | retired (`backup/`, not tracked) | compact curriculum | -- |

v3 also ships an int8 `flash` variant for tighter CPU budgets; see [v3 flash](#v3-flash-int8).

## Measurements

All values use `dry_blend: 1.0`. Columns report means on the same sets, not
paired confidence intervals between released versions.

| Set | Metric | Unprocessed | v1 | v2 | v3 |
|---|---|---|---|---|---|
| VCTK-DEMAND (n=824) | PESQ-WB | 1.968 | 2.551 | 2.567 | **2.600** |
| | STOI | 0.9210 | 0.9213 | 0.9231 | **0.9295** |
| | SI-SDR (dB) | 8.45 | **18.38** | 18.04 | 16.79 |
| | deletion rate (Whisper large-v3) | 0.00744 | **0.00643** | 0.00870 | 0.00721 |
| frozen synthetic (500 clips; PESQ n=499) | PESQ-WB | 1.471 | 2.354 | 2.409 | **2.440** |
| DNS-5 dev (n=921) | DNSMOS SIG / BAK / OVR / P.808 | 2.99 / 2.56 / 2.21 / 2.91 | 3.18 / 3.82 / 2.81 / 3.38 | 3.19 / 3.85 / 2.83 / 3.42 | **3.24 / 3.89 / 2.88 / 3.44** |
| hard WER set (n=500) | raw WER, including recognizer loops | 0.12251 | -- | 0.16006 | 0.17482 |
| | deletion rate | 0.04235 | -- | 0.04728 | 0.04669 |
| -- | CPU RTF, 1 thread (checkpoint gate) | -- | 0.171 | 0.164 | 0.249 |
| -- | CPU RTF, 1 thread (original streaming ONNX, 10 s probe) | -- | -- | 0.408 | 0.570 |
| -- | CPU RTF, 1 thread (current optimized ONNX, 5 s speech) | -- | 0.299 | 0.311 | 0.446 |
| -- | CPU RTF, 1 thread (current native SSM companion, 5 s speech) | -- | 0.257 | 0.266 | 0.390 |
| -- | Graph look-ahead (ms) | -- | 30 | 30 | 30 |
| -- | Streaming algorithmic buffering (ms, excluding compute) | -- | 52–62 | 52–62 | 52–62 |

Records: [v3](../benchmarks/records/ns_dpcrn-mamba_v3dm_mg_ep3.json),
[v2](../benchmarks/records/ns_dpcrn-mamba_metricgan_ft_ep3.json),
[v1](../benchmarks/records/ns_dpcrn-mamba_activebin_ft_ep1.json).

## Release decisions

- v3 was updated on 2026-10-18: the same wide model and training ladder, with a
  DNSMOS P.835 term beside the PESQ critic in the final stage (run
  `ns_dpcrn-mamba_v3dm_mg`). Paired against the previous v3 weights on DNS-5
  dev: SIG +0.050, BAK +0.090, OVR +0.076, P.808 +0.028, all resolvable and in
  every noise category; VCTK-DEMAND PESQ −0.031 and SI-SDR −0.34 dB; word
  deletions unchanged within resolution. The previous weights stay under
  `backup/` (not tracked).
- v3 remains a selectable candidate; v2 remains the default. v3 passes both
  PESQ gates and the checkpoint CPU budget, but both deletion gates are
  `no-resolution` (intervals cross zero). This does not establish
  non-inferiority or a word-preservation improvement.
- v3's raw hard-set WER includes 6 recognizer loops, versus 1 on the input.
  Excluding a pair when either hypothesis exceeds 1.5 times the reference word
  count leaves 493 pairs: model WER **0.143382**, input **0.116340**, delta
  **+0.027042**. These trimmed means use a different subset from earlier
  releases and must not be treated as a paired v3-versus-v2 ranking. Keep loop
  failures visible as a separate deployment metric.
- v2 was released over a narrow VCTK deletion failure, with the accepted
  trade-off recorded in its benchmark lineage. v1 remains available for its
  lower VCTK deletion rate.

## flash (int8) variants

`flash` is an artifact variant of each version, not a new checkpoint: the same
weights, with the LSTM and matrix-product weights stored as int8 and activations
quantized at run time (`export --quantize int8`), in both the standard graph and
the native companion. Select it with `--variant flash`. Each column below is the
streaming ONNX output on the same items, scored as paired differences; the float
columns therefore differ slightly from the checkpoint records.

### v3

| Measure | v3 | v3 flash | Paired difference (95% CI) |
|---|---|---|---|
| CPU RTF, 1 thread, standard / native graph | 0.454 / 0.387 | 0.370 / 0.304 | −18.5% / −21.5% |
| frozen synthetic PESQ-WB | 2.499 | 2.496 | −0.0028 (−0.0038, −0.0019) |
| VCTK-DEMAND PESQ-WB | 2.598 | 2.591 | −0.0070 (−0.0097, −0.0051) |
| VCTK-DEMAND SI-SDR (dB) | 16.79 | 16.77 | −0.016 (−0.027, −0.006) |
| DNS-5 dev DNSMOS OVR | 2.884 | 2.883 | −0.0004 (−0.0014, +0.0005) |
| VCTK-DEMAND deletion rate | 0.0061 | 0.0061 | +0.0000 (+0.0000, +0.0000) |
| hard-set deletion rate | 0.0451 | 0.0439 | −0.0009 (−0.0043, +0.0025) |

PESQ and SI-SDR are resolvably lower by a few thousandths; DNSMOS OVR and word
deletions show no resolvable change. Use it when the CPU budget is tight. The
timings are medians of nine interleaved repeats on a 5-second noisy-speech
sample, one thread on an AVX-512 Xeon, measured on the 2026-10-05 export; the
2026-10-18 weights run in a graph of the same shape and operator count. Read the
ratios, and since int8 kernels differ by ISA, measure on the target CPU.
Browser/WASM payloads keep the float graph. Word rates used Whisper large-v3 in
`float16` (no recognizer loops on this run), so they are not comparable with the
table above.

### v1 and v2

| Measure | v1 float → flash | Paired difference (95% CI) | v2 float → flash | Paired difference (95% CI) |
|---|---|---|---|---|
| CPU RTF, 1 thread, standard / native graph | 0.312 / 0.256 → 0.276 / 0.224 | −12% / −12% | 0.310 / 0.258 → 0.278 / 0.228 | −10% / −12% |
| frozen synthetic PESQ-WB | 2.433 → 2.428 | −0.0054 (−0.0064, −0.0045) | 2.490 → 2.491 | +0.0015 (+0.0006, +0.0023) |
| VCTK-DEMAND PESQ-WB | 2.553 → 2.550 | −0.0032 (−0.0050, −0.0016) | 2.572 → 2.568 | −0.0039 (−0.0081, −0.0009) |
| VCTK-DEMAND SI-SDR (dB) | 18.41 → 18.38 | −0.031 (−0.036, −0.027) | 18.14 → 18.13 | −0.005 (−0.010, +0.000) |
| DNS-5 dev DNSMOS OVR | 2.808 → 2.807 | −0.0009 (−0.0016, −0.0001) | 2.828 → 2.828 | −0.0002 (−0.0009, +0.0005) |
| VCTK-DEMAND deletion rate | 0.0077 → 0.0076 | −0.0001 (−0.0003, +0.0000) | 0.0077 → 0.0077 | +0.0000 |
| VCTK-DEMAND WER | 0.0359 → 0.0349 | −0.0002 (−0.0017, +0.0019) | 0.0338 → 0.0352 | +0.0018 (−0.0001, +0.0044) |

The same picture as v3, a little smaller on CPU: quality moves by thousandths and
words do not move resolvably. Timings are medians of three interleaved rounds on
the same 5-second sample and CPU as v3's; word rates here used Whisper large-v3 in
`float16`.

## PCS post-filter (optional)

`pcs: true` applies perceptual contrast stretching (Chao et al., Interspeech
2022) to the enhanced spectrum before the inverse STFT: a fixed per-band gamma
of 1.0-1.4 on the log1p magnitude, largest over 400-4400 Hz, phase untouched.
It is a request parameter, off by default, applied per frame by the streaming
runtime with no added latency (`--param pcs=true` on the CLI; `pcs` in the web
UI). Each column below is the streaming ONNX output of v3 on the same items
(records `ns_dpcrn-mamba_v3dm_mg_ep3_ort` and `..._ort_pcs`).

| Measure | v3 | v3 + PCS | Paired difference (95% CI) |
|---|---|---|---|
| frozen synthetic PESQ-WB | 2.499 | **2.703** | +0.204 (+0.192, +0.218) |
| VCTK-DEMAND PESQ-WB | 2.598 | **2.994** | +0.397 (+0.385, +0.409) |
| VCTK-DEMAND harmonic contrast gap (dB) | −0.24 | **+0.31** | +0.556 (+0.528, +0.586) |
| VCTK-DEMAND STOI / ESTOI | 0.929 / 0.837 | 0.926 / 0.834 | −0.0031 / −0.0034 |
| DNS-5 dev DNSMOS SIG | 3.241 | 3.118 | −0.123 (−0.131, −0.114) |
| DNS-5 dev DNSMOS BAK | 3.891 | 3.864 | −0.028 (−0.035, −0.021) |
| DNS-5 dev DNSMOS OVR | 2.884 | 2.778 | −0.106 (−0.115, −0.097) |
| DNS-5 dev DNSMOS P.808 | 3.437 | 3.347 | −0.089 (−0.102, −0.077) |
| VCTK-DEMAND WER / deletion rate | 0.041 / 0.0070 | 0.041 / 0.0071 | +0.000 (−0.003, +0.003) / +0.0002 (−0.0010, +0.0012) |
| hard-set raw WER / insertion rate | 0.157 / 0.024 | 0.158 / 0.026 | +0.001 (−0.013, +0.018) / +0.002 (−0.012, +0.018) |

What it buys: sharper formant peaks, which PESQ and the harmonic-contrast gap
reward strongly -- v3 + PCS scores above every model in this table on PESQ. What
it costs: every DNSMOS scale is resolvably lower (speech quality most) and STOI
slips a little. On the previous v3 weights PCS also doubled the recognizer's
insertions on the hard set; on the current weights that change is no longer
resolvable.
SI-SDR is not listed because the stretch is non-linear and the metric no longer
measures distortion. Turn it on when PESQ is the number being compared; leave
it off for listening and for ASR.

## Reproduce v3

Run these recipes in order with seed 1234, warm-starting each stage from the
previous stage's last checkpoint. The run names remain experiment names; v3 is
assigned only to the released artifact.

| Recipe | Warm start | Epochs / last checkpoint |
|---|---|---|
| [`train_dpcrn_mamba_capW_s0.yaml`](../config/train_dpcrn_mamba_capW_s0.yaml) | scratch; DNS-5 data, SNR [0,20] | 20 / ep19, step40000 |
| [`train_dpcrn_mamba_capWL3_s1.yaml`](../config/train_dpcrn_mamba_capWL3_s1.yaml) | capW_s0 ep19; mixed speech/noise, coverage sampler, SNR [-10,40], 4-second rows | 20 / ep19, step80000 |
| [`train_dpcrn_mamba_capWL3_ft.yaml`](../config/train_dpcrn_mamba_capWL3_ft.yaml) | capWL3_s1 ep19; ActiveBin calibration | 2 / ep1, step4000 |
| [`train_dpcrn_mamba_capWL3_mg.yaml`](../config/train_dpcrn_mamba_capWL3_mg.yaml) | capWL3_ft ep1; PESQ MetricGAN + DNSMOS P.835 term (`DNSMOSLoss`) | 4 / ep3, step8000 |

The recipes are snapshots of the configurations used for this run. Point corpus
and RIR paths at your copies; the short stages use the pool with silent files
removed. `target_clipping: own_quantile` is preserved to match training.
The released checkpoint contains inference weights and source provenance,
excluding the critic, losses, optimizer and sampler state. Its inference weights
are unchanged from `ns_dpcrn-mamba_v3dm_mg` ep3.

## Export validation

The release's inference output is bit-identical to the source checkpoint on
random input, a frozen NS mixture and a hard-set mixture. CPU ONNX streaming
parity is 103.1 dB on the export's verification probe after aligning the
480-sample look-ahead delay; maximum sample error is below 7e-7 (2026-10-18
export; the graph has the same shape as the 2026-10-04 re-export, so its
timings carry over).

A historical release-environment benchmark used the original streaming ONNX runtime with
one CPU thread, 10-second input, two warm-ups and three interleaved repeats:
v2 RTF **0.408**, v3 **0.570**. That original v3 export exceeded the **0.5**
streaming budget. Its full-utterance checkpoint measured **0.270** there; the
original checkpoint gate above measured **0.242**. These are different execution
paths, and the checkpoint gate does not establish the streaming runtime's cost.
Full validation results are preserved in the v3 record's release lineage.

All three NS releases have since been re-exported with fixed batch=1 and
frequency-major layout. Their standard ONNX graphs are now the model-zoo defaults;
the sidecars additionally reference SHA256-validated native SSM companions.
On the same 5-second noisy-speech sample, an interleaved old/new measurement gave:
v1 **0.393 → 0.299 / 0.257**, v2 **0.406 → 0.311 / 0.266**, and v3
**0.572 → 0.446 / 0.390** (standard / native). v3 now meets the 0.5 budget on
this CPU. The [re-export report](../../../model_zoo/benchmarks/onnx_reexport_20261004.json)
records the sample, timing repeats, hashes and 30-second stateful checks. These
5-second speech timings and the older 10-second probe are separate benchmarks.
See [CPU setup](../../../docs/usage/streaming/dpcrn_onnx.md) for library selection.

The checkpoint gate processes a whole 10-second waveform in one PyTorch call.
The streaming runtime processes one 10 ms hop at a time, with roughly 1,000
ONNX calls per 10 seconds plus state exchange, STFT/iSTFT and overlap-add.
These paths differ in batching, backend and orchestration; the measurements do
not isolate each contribution. RTF is compute time divided by audio duration:
0.570 means about 5.7 seconds of computation for 10 seconds of audio. It does
not measure the wait for future input and must be reported separately from latency.

v3 has a 160-sample hop (10 ms), a 512-sample analysis window (32 ms) and
three look-ahead frames (30 ms). Streaming algorithmic buffering is therefore
`lookahead + window − hop` to `lookahead + window`, or **52–62 ms**, depending
on sample position within the hop. Budget **62 ms** before compute time,
additional input chunk buffering, audio device buffering and transport. v1/v2
have the same geometry and algorithmic latency. The 480-sample (30 ms) waveform
alignment offset remains the graph look-ahead; it is not the complete streaming
buffering delay. These derived values are recorded under `algorithmic_latency`
in the v3 ONNX manifest and release validation record.

The generated device payload also passes ONNX Runtime Web 1.24.3 WASM
validation on Node: normalized RMS error versus native streaming is 3.30e-7;
irregular chunk sizes and a stream reset reproduce the same output.
