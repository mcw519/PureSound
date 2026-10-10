# voice_isolate — pretrained checkpoints

Traditional Chinese: [`README.zh-TW.md`](README.zh-TW.md)

Near-field (<1 m) foreground voice isolation, single channel, no enrollment: keep
the near speaker, suppress far and competing speakers and noise.

**`dpcrn_curriculum_v1.ckpt` is the model-zoo default**, with `dpcrn_curriculum_v2.ckpt`
and `dpcrn_v8.ckpt` beside it as candidates; those three are the versions registered in
[`model_zoo/catalog.yaml`](../../../model_zoo/catalog.yaml). All three are DPCRN
(complex ratio mask, 16 kHz, 30 ms look-ahead). `dpcrn_v8` and `dpcrn_curriculum_v1`
share one width and load with `config/infer_dpcrn.yaml`; `dpcrn_curriculum_v2` is
wider and loads with `config/infer_dpcrn_wide.yaml`:

```bash
cd egs/voice_isolate
uv run python scripts/demo.py --config_path config/infer_dpcrn.yaml        # dpcrn_v8, dpcrn_curriculum_v1
uv run python scripts/demo.py --config_path config/infer_dpcrn_wide.yaml   # dpcrn_curriculum_v2
```

The dropdown lists every checkpoint in this directory whichever config you pass;
pick one whose width matches that config.

## The released blend: `dry_blend = 0.9`

The runtime blend is **part of the released configuration** of every shipped
version, not an optional extra:

```python
enhanced = model(wav, dry_blend=0.9)      # out = 0.9 * enhanced + 0.1 * input
```

It bounds attenuation at any point to -20 dB. That costs a little residual
interferer and buys a large drop in deletions on capture chains outside the
training data: with it the model lowers ASR error on real recordings, without it
the same checkpoint raises WER on one of the reverberant test sets. The eval
scripts expose `--dry-blend`; the streaming manifest carries the value under
`recommended_inference`.

Measured on `dpcrn_curriculum_v2` (ep39, full benchmark, same code for both rows):

| stage | `dry_blend 1.0` | `dry_blend 0.9` |
|---|---|---|
| moderate-reverb WER (primary gate) | 0.416 | **0.386** |
| Dawn WER / deletion (unprocessed 0.184 / 0.082) | 0.215 / 0.132 | **0.174 / 0.084** |
| BUT-OFFICE WER (unprocessed 0.563) | 0.597 | **0.528** |
| BUT extreme-reverb WER (unprocessed 0.658) | 0.739 | **0.658** |
| turn-taking KEEP ok / violations | 88 / 12 | **90 / 10** |
| turn-taking SUPPRESS (median) | **−26.0 dB** | −18.6 dB |

Take `dry_blend 1.0` only when a distant voice must be removed as completely as
possible and the near talker's words matter less; every word-error figure is worse
there. curriculum-v1 moves the same way (Dawn WER 0.232 at 1.0, 0.172 at 0.9).

Known limitation: a blend of 0.9 cannot produce full silence (a -20 dB floor by
construction), and suppression of far speech recorded through chains very unlike
the training corpora is shallower than in-domain. Hard muting would need a gate,
not a blend.

## Versions

| Version | Zoo id / role | Training | Choose it when |
|---|---|---|---|
| **`dpcrn_curriculum_v1.ckpt`** | `voice-isolate-dpcrn-curriculum-v1`, **default** | `config/train_dpcrn_curriculum_v1.yaml`, warm-started from the cold-start recipe | the capture hardware is the one the model was tuned for |
| `dpcrn_curriculum_v2.ckpt` | `voice-isolate-dpcrn-curriculum-v2`, candidate | the same two steps on a wider model (`train_dpcrn_curriculum_v2_base.yaml`, then `train_dpcrn_curriculum_v2.yaml`); loads with `config/infer_dpcrn_wide.yaml` | lower WER and deeper suppression are worth 1.4x curriculum-v1's CPU |
| `dpcrn_v8.ckpt` | `voice-isolate-dpcrn-v8`, candidate | a multi-stage warm-start ladder; recipe not published | the capture hardware is unknown or unlike the training corpora |
| `dpcrn_curriculum_v0` | not distributed | `config/train_dpcrn.yaml`, from scratch, taken at ep99 | not for deployment; it is the warm start of curriculum-v1 and -v2 |

## Measurements

All at `dry_blend 0.9`. The sets below read fixed audio, so the figures compare
across versions.

| Set | Metric | Unprocessed | v8 | curriculum-v1 | curriculum-v2 |
|---|---|---|---|---|---|
| moderate-reverb (primary WER gate) | WER | 0.582 | 0.411 | 0.427 | **0.386** |
| Dawn Chorus | WER / deletion | 0.184 / 0.082 | 0.180 / 0.094 | **0.172** / 0.089 | 0.174 / **0.084** |
| BUT-OFFICE (monitor) | WER | 0.563 | 0.539 | 0.547 | **0.528** |
| real-RIR turn-taking | SUPPRESS median (ok / fail) | -- | −16.14 dB (87 / 13) | −17.10 dB (94 / 6) | **−18.60 dB (95 / 5)** |
| | KEEP ok / violations | -- | **94 / 6** | 89 / 11 | 90 / 10 |
| -- | streaming CPU RTF, 1 thread | -- | 0.26 | 0.26 | 0.36 |

On an internal field recording set (not distributed), curriculum-v1 and -v2
suppress a lone distant talker heard before any near talker (−7.1 and −12.4 dB);
v8 does not. On recordings from capture chains unlike the training data, both can
attenuate the near talker; v8 does not.

BUT-OFFICE is a 200-utterance set whose bootstrap interval is usually too wide to
separate a model from leaving the audio alone, so it is a monitor; the primary WER
gate is `wer_set_moderate_test` ([`../scripts/WER_SETS.md`](../scripts/WER_SETS.md)).

## Choosing a version

Take `dpcrn_curriculum_v1.ckpt` with `dry_blend 0.9` -- the model-zoo default; it
suppresses a far voice even before any near talker has spoken. Take
`dpcrn_v8.ckpt` when the capture chain is unknown or unlike the training corpora:
it gives up that cold-start gain but does not regress near-talker keep on such
recordings, which is curriculum-v1's open defect. Take `dpcrn_curriculum_v2.ckpt`,
loaded with `config/infer_dpcrn_wide.yaml`, when the CPU budget allows 1.4x
curriculum-v1: it beats or ties v1 on every benchmark stage, but shares its
cross-chain keep defect.

**What none of them do.** Far speech recorded through a capture chain very unlike
the training corpora is still barely suppressed. That gap is a property of the
recording chain, not of distance, and no version here closes it.

**Judging convention.** The scheduler (`CosineAnnealingWarmRestarts`, `T_0=20`)
restarts every 20 epochs, so checkpoints are only comparable at the cosine
troughs -- ep19 / ep39 / ep59. Every number above comes from a trough epoch.

## `streaming/` — per-frame ONNX exports

`dpcrn_v8`, `dpcrn_curriculum_v1` and `dpcrn_curriculum_v2` (`.onnx` + `.json`), the
registered versions, built with `../scripts/streaming_onnx.py export`. All carry a **30 ms
(3-frame) algorithmic latency** from the look-ahead (52–62 ms input to output with
the 32 ms analysis window), handled by future-buffering inside the graph, and all
record `dry_blend 0.9` under `recommended_inference`. Load them with
`puresound.streaming.StreamingOrt` or the SDK's `PureSoundStreamingRuntime`
(`processor: stft_frame_ort`); both apply the blend on the output with the input
delayed by `streaming_delay_frames`, at no extra latency.

The `dpcrn_v8` and `dpcrn_curriculum_v1` exports use batch=1 frequency-major layout and a vectorized
single-time inter-LSTM update, combining the trained input/hidden projections
into one matrix multiplication over all frequency positions. The trained intra
bidirectional LSTMs and all state ports are retained. On the same 5-second
noisy-speech sample, interleaved single-thread CPU RTF improved from **0.366 to
0.258** for each release. The [re-export report](../../../model_zoo/benchmarks/onnx_reexport_20261004.json)
includes original/new hashes and 30-second streaming parity checks. `dpcrn_curriculum_v2` was exported the same way afterwards: offline-versus-streaming
parity 95.2 dB on a real recording (curriculum-v1 95.7), and streaming RTF **0.362**
against curriculum-v1's 0.256 on the same sample, interleaved, single thread.

Any offline-versus-streaming comparison **must** align by the reported latency and
trim the edges, otherwise the delay reads as error; `verify` does this. Judge a new
export on real speech (`--input_audio`): the default white-noise probe is a stress
signal whose parity figure tracks how aggressively a version modulates its mask,
not streaming correctness.

```bash
cd egs/voice_isolate
uv run python scripts/streaming_onnx.py export \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_curriculum_v1.ckpt /tmp/model.onnx \
    --optimization portable --dry-blend 0.9
uv run python scripts/streaming_onnx.py verify \
    config/infer_dpcrn.yaml pretrained_ckpt/dpcrn_v8.ckpt \
    pretrained_ckpt/streaming/dpcrn_v8.onnx --input_audio speech.wav --provider cpu
```

Retraining: [`../README.md`](../README.md#train). The checkpoints here are the
judged trough epochs of their training runs.
