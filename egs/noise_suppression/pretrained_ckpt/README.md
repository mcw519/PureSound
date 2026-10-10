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
| `dpcrn_mamba_v3.ckpt` | `noise-suppression-dpcrn-mamba-v3`, released candidate | wider model + mixed-source coverage training + ActiveBin + MetricGAN | higher PESQ and STOI matter; see the DNSMOS, ASR and CPU trade-offs below |
| **`dpcrn_mamba_v2.ckpt`** | `noise-suppression-dpcrn-mamba-v2`, **default** | v1 + 4 epochs with a PESQ critic | balanced quality and CPU cost |
| `dpcrn_mamba_v1.ckpt` | `noise-suppression-dpcrn-mamba-v1`, candidate | compact curriculum + ActiveBin calibration | deletion on clean read speech matters most |
| `dpcrn_mamba_v0.ckpt` | retired (`backup/`, not tracked) | compact curriculum | -- |

v3 also ships an int8 `flash` variant for tighter CPU budgets; see [v3 flash](#v3-flash-int8).

## Measurements

All values use `dry_blend: 1.0`. Columns report means on the same sets, not
paired confidence intervals between released versions.

| Set | Metric | Unprocessed | v1 | v2 | v3 |
|---|---|---|---|---|---|
| VCTK-DEMAND (n=824) | PESQ-WB | 1.968 | 2.551 | 2.567 | **2.631** |
| | STOI | 0.9210 | 0.9213 | 0.9231 | **0.9310** |
| | SI-SDR (dB) | 8.45 | **18.38** | 18.04 | 17.13 |
| | deletion rate (Whisper large-v3) | 0.00744 | **0.00643** | 0.00870 | 0.00825 |
| frozen synthetic (500 clips; PESQ n=499) | PESQ-WB | 1.471 | 2.354 | 2.409 | **2.449** |
| DNS-5 dev (n=921) | DNSMOS SIG / BAK / OVR / P.808 | 2.99 / 2.56 / 2.21 / 2.91 | 3.18 / 3.82 / 2.81 / 3.38 | 3.19 / 3.85 / 2.83 / 3.42 | 3.19 / 3.80 / 2.81 / 3.41 |
| hard WER set (n=500) | raw WER, including recognizer loops | 0.12251 | -- | 0.16006 | 0.17651 |
| | deletion rate | 0.04235 | -- | 0.04728 | 0.04838 |
| -- | CPU RTF, 1 thread (checkpoint gate) | -- | 0.171 | 0.164 | 0.242 |
| -- | CPU RTF, 1 thread (original streaming ONNX, 10 s probe) | -- | -- | 0.408 | 0.570 |
| -- | CPU RTF, 1 thread (current optimized ONNX, 5 s speech) | -- | 0.299 | 0.311 | 0.446 |
| -- | CPU RTF, 1 thread (current native SSM companion, 5 s speech) | -- | 0.257 | 0.266 | 0.390 |
| -- | Graph look-ahead (ms) | -- | 30 | 30 | 30 |
| -- | Streaming algorithmic buffering (ms, excluding compute) | -- | 52–62 | 52–62 | 52–62 |

Records: [v3](../benchmarks/records/ns_dpcrn-mamba_capWL3_mg_ep3.json),
[v2](../benchmarks/records/ns_dpcrn-mamba_metricgan_ft_ep3.json),
[v1](../benchmarks/records/ns_dpcrn-mamba_activebin_ft_ep1.json).

## Release decisions

- v3 is published as a selectable candidate; v2 remains the default. v3 passes
  both PESQ gates and the checkpoint CPU budget, but both deletion gates are
  `no-resolution` (intervals cross zero). This does not establish
  non-inferiority or a word-preservation improvement.
- v3's raw hard-set WER includes 7 recognizer loops, versus 1 on the input.
  Excluding a pair when either hypothesis exceeds 1.5 times the reference word
  count leaves 493 pairs: model WER **0.144154**, input **0.120800**, delta
  **+0.023354**. These trimmed means use a different subset from earlier
  releases and must not be treated as a paired v3-versus-v2 ranking. Keep loop
  failures visible as a separate deployment metric.
- v2 was released over a narrow VCTK deletion failure, with the accepted
  trade-off recorded in its benchmark lineage. v1 remains available for its
  lower VCTK deletion rate.

## v3 flash (int8)

`flash` is an artifact variant of v3, not a new checkpoint: the same weights, with
the LSTM and matrix-product weights stored as int8 and activations quantized at
run time (`export --quantize int8`). Select it with `--variant flash`. Each column
below is the streaming ONNX output on the same items, scored as paired
differences; the v3 column therefore differs slightly from the checkpoint record.

| Measure | v3 | v3 flash | Paired difference (95% CI) |
|---|---|---|---|
| CPU RTF, 1 thread, standard / native graph | 0.454 / 0.387 | 0.370 / 0.304 | −18.5% / −21.5% |
| frozen synthetic PESQ-WB | 2.508 | 2.504 | −0.0044 (−0.0054, −0.0035) |
| VCTK-DEMAND PESQ-WB | 2.629 | 2.624 | −0.0053 (−0.0065, −0.0041) |
| VCTK-DEMAND SI-SDR (dB) | 17.14 | 17.10 | −0.038 (−0.047, −0.028) |
| DNS-5 dev DNSMOS OVR | 2.808 | 2.805 | −0.0024 (−0.0035, −0.0013) |
| VCTK-DEMAND deletion rate | 0.0086 | 0.0084 | −0.0002 (−0.0019, +0.0013) |
| hard-set deletion rate, loops excluded | 0.0434 | 0.0434 | +0.0000 (−0.0025, +0.0025) |

Every quality metric is resolvably lower, by about a tenth of v3's PESQ gain over
v2; word deletions show no resolvable change. Use it when the CPU budget is
tight. The timings are medians of nine interleaved repeats on a 5-second
noisy-speech sample, one thread on an AVX-512 Xeon; read the ratios, and since
int8 kernels differ by ISA, measure on the target CPU. Browser/WASM payloads keep
the float graph. Deletion rates used Whisper large-v3 in `int8_float16`, so they
are not comparable with the table above.

## Reproduce v3

Run these recipes in order with seed 1234, warm-starting each stage from the
previous stage's last checkpoint. The run names remain experiment names; v3 is
assigned only to the released artifact.

| Recipe | Warm start | Epochs / last checkpoint |
|---|---|---|
| [`train_dpcrn_mamba_capW_s0.yaml`](../config/train_dpcrn_mamba_capW_s0.yaml) | scratch; DNS-5 data, SNR [0,20] | 20 / ep19, step40000 |
| [`train_dpcrn_mamba_capWL3_s1.yaml`](../config/train_dpcrn_mamba_capWL3_s1.yaml) | capW_s0 ep19; mixed speech/noise, coverage sampler, SNR [-10,40], 4-second rows | 20 / ep19, step80000 |
| [`train_dpcrn_mamba_capWL3_ft.yaml`](../config/train_dpcrn_mamba_capWL3_ft.yaml) | capWL3_s1 ep19; ActiveBin calibration | 2 / ep1, step4000 |
| [`train_dpcrn_mamba_capWL3_mg.yaml`](../config/train_dpcrn_mamba_capWL3_mg.yaml) | capWL3_ft ep1; PESQ MetricGAN | 4 / ep3, step8000 |

The recipes are snapshots of the configurations used for this run. Point corpus
and RIR paths at your copies; the short stages use the pool with silent files
removed. `target_clipping: own_quantile` is preserved to match training.
The released checkpoint contains inference weights and source provenance,
excluding the critic, losses, optimizer and sampler state. Its inference weights
are unchanged from `ns_dpcrn-mamba_capWL3_mg` ep3.

## Export validation

The release's inference output is bit-identical to the source checkpoint on
random input, a frozen NS mixture and a hard-set mixture. CPU ONNX streaming
parity is 92.2 / 116.7 / 111.6 dB after aligning the 480-sample look-ahead delay;
maximum sample error is below 5e-7.

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
