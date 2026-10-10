# Device chain

繁體中文版本：[device_chain.zh-TW.md](device_chain.zh-TW.md)

`puresound.task.device_chain.DeviceChain` applies what happens to a finished
mixture between the room and the model: resampling, transducer response, rumble
filter, preamp gain or overload, compression, the A/D converter, a VoIP codec
and packet loss. It knows nothing about talkers or rooms. The noise-suppression,
voice-isolation and target-speaker-extraction datasets build one chain with
`device_chain_from_blocks(augmentor, dataset)` and call
`apply(noisy, target, sample_rate=...)` once per row, after the noise stage.

## Structure

```
SRC → 2nd-order IIR → HPF → volume / clipping → compressor     analogue path: the target follows
──────────────────── _analogue_to_digital ────────────────────   the A/D boundary
→ codec → packet loss                                           transmission damage: mixture only
```

The group a stage belongs to decides whether the clean target goes through it.

**Analogue path: the target follows.** Transducer response, filtering, preamp
gain and compression are (time-varying) linear operators. The target is defined
as what the model should recover *through* this channel, so each stage is
applied to the mixture and the target with identical parameters. Linearity keeps
the pair consistent:

```
H(near + far + noise) = H(near) + H(far) + H(noise)
```

After `H` the mixture is still the sum of its components, at the SIR and SNR
the recipe drew.

**Transmission: mixture only.** Codec artefacts and dropped packets are damage
the model should repair, not a channel it should reproduce. If the target
carried them, the optimal model would keep them.

## Stages

| Stage | Models | Computation | Target |
|---|---|---|---|
| `_sample_rate_conversion` | a lower sample rate somewhere in the chain | `fs → src_sr → fs`; `src_sr` drawn from `src_range` with weights `prob_each`; the resampler is chosen 50/50 between the fixed and the randomised backend ([Spectral and channel effects](spectral_channel.md)) | same backend; the randomised backend reuses the mixture's filter parameters |
| `_second_order_iir` | transducer frequency response | random stable second-order pole/zero filter | same `(a, b)` |
| `_high_pass` | rumble filter | RBJ high-pass; cutoff drawn from `cutoff` with weights `prob_each`; `Q ~ N(0.707, 0.1²)` clipped to [0.3, 1.3] | same cutoff and Q |
| `_volume` | preamp gain, or analogue overload | one draw against `clipping_prob` picks the branch: gain `g ~ U(perturbed_range)` as a plain multiply, or quantile clipping ([Level and dynamics](level_dynamics.md)) | gain: same `g`; clipping: per `target_clipping` (below) |
| `_compressor` | broadcast or conferencing compression | time-varying gain curve from `compressor_gain`, computed on the mixture | same curve |
| `_analogue_to_digital` | gain staging at the converter | divide the pair by their shared peak when it exceeds 1 | same divisor |
| `_codec` | VoIP or telephony coding | real encode → decode round trip | mixture only |
| `_packet_loss` | network packet loss | per-packet Bernoulli drop, zero fill | mixture only |

### Clipping and the target

The clipping branch draws quantile levels `min_q ~ U(clipping_range.min)` and
`max_q ~ U(clipping_range.max)` and clips the mixture to
`[Q_min_q(noisy), Q_max_q(noisy)]`. `augmentation_volume.target_clipping` sets
what happens to the target:

| `target_clipping` | target clipped to | effect |
|---|---|---|
| `mixture_level` (default) | the mixture's thresholds `[Q_min_q(noisy), Q_max_q(noisy)]` | a target quieter than the thresholds is unchanged |
| `own_quantile` | `[Q_min_q(target), Q_max_q(target)]`, the same quantile levels on its own samples | the target is clipped as hard as the mixture whatever its level; what the shipped noise-suppression recipes use |

Either way the mixture is clipped identically and the branch draws the same
numbers. This is the one nonlinearity the analogue path allows, and on a clipped
row the target is not exactly the near component of the mixture.
Voice-isolation recipes set `clipping_prob: 0`; noise-suppression recipes enable
the branch.

## The A/D boundary

`_analogue_to_digital` is the only point in synthesis where digital full scale
exists.

- **Upstream is acoustic and analogue.** Level there is sound pressure through a
  transducer and a preamp; a peak above 1.0 is a loud room, not an error. No
  upstream stage may treat 1.0 as a ceiling, which is why the filters run
  through `apply_linear` or with `clamp=False`
  ([Spectral and channel effects](spectral_channel.md)).
- **Downstream is digital.** A codec cannot encode beyond full scale, and the
  models clamp their output to [−1, 1], so a target above full scale could never
  be reached.

Crossing the boundary is gain staging, the engineer setting the preamp so the
converter is not driven into its rails:

```
peak = max(max|noisy|, max|target|)
if peak > 1:  noisy ← noisy / peak;  target ← target / peak
```

One shared divisor keeps both the pair's level relationship and superposition.
Deliberate overload lives in `_volume` instead, where it is probability
controlled and recorded as `volume_clipped`; clipping here as well would clip
those rows a second time with no record of it.

The step consumes no randomness and records `overload_rescaled`.
`DeviceChain(..., overload_guard=False)` removes it; every task builds its chain
with it on, and tests use the flag to isolate the analogue stages.

## Codec

`AudioEffectAugmentor.apply_codec` encodes to a temporary file with TorchCodec's
`AudioEncoder` and decodes it with `AudioDecoder(path, sample_rate=sr)`. The
encoder takes no codec argument, so the container extension selects the codec:
`libopus` → `.opus`, `g722` → `.g722`. The distortion is the codec's real
behaviour (quantisation, band limiting, psychoacoustic shaping), not a filter
approximation.

- **libopus** encodes at 48 kHz internally; the bitrate is an integer drawn
  uniformly from `bitrate_range.libopus`, in bit/s.
- **g722** is ITU-T G.722 sub-band ADPCM at 16 kHz and a fixed 64 kbit/s; it has
  no bitrate knob.

The codec is chosen from `codecs` with weights `prob_each` (uniformly when
omitted). Encoder delay and framing change the length, so the decoded signal is
truncated or zero-padded back to the input length.

## Packet loss

```
packet_samples = max(1, round(fs · packet_ms / 1000))
n_packets      = floor(T / packet_samples)
drop_k ~ Bernoulli(loss_rate),  k < n_packets;  dropped packets are zeroed
```

`packet_ms` is drawn from `packet_ms_choices` (20 ms is the WebRTC default,
60 ms is common for low-bitrate Opus) and `loss_rate ~ U(loss_rate_range)`. A
trailing partial packet is never dropped.

Two deliberate simplifications:

- **Zero fill, no concealment.** Packet-loss concealment is endpoint behaviour
  and differs between implementations; the model should cope with audio that is
  genuinely missing.
- **Independent losses.** Real network loss is bursty, so long gaps are
  under-represented. A Gilbert–Elliott two-state model is the natural extension;
  none is implemented.

## Contracts

**Stage order is fixed.** Every stage draws from the shared RNG streams, so
swapping two stages changes what a seeded recipe produces even when neither stage
changed. The order also follows the physical chain: transducer before preamp,
compression before the converter, converter before the codec.

**A disabled stage draws nothing.** Every guard is
`block is not None and block.used and torch.rand(1) < block.prob`, so disabling a
stage, or a task not having its block, leaves every later draw unchanged. See
[Engineering contract](engineering_contract.md).

**A stage without a block is absent.** `device_chain_from_blocks` reads the
dataset's `augmentation_src_args`, `augmentation_ir_response_args`,
`augmentation_hpf_args` and `augmentation_volume_args`, and the compressor, codec
and packet-loss blocks through `getattr`; a task without a block gets `None`, and
the stage costs nothing. Target-speaker extraction therefore has no codec or
packet-loss stage.

## Provenance

`apply` returns `ChainResult(noisy, target, applied)`; `applied` holds one float
per key of `DEVICE_CHAIN_SCALARS`:

| Keys | Value |
|---|---|
| `src_applied`, `iir_applied`, `hpf_applied`, `volume_applied`, `volume_clipped`, `compressor_applied`, `codec_applied`, `packet_loss_applied`, `overload_rescaled` | 0.0 or 1.0 |
| `src_target_sr`, `hpf_cutoff`, `volume_gain`, `compressor_ratio`, `compressor_threshold_db`, `codec_bitrate`, `packet_loss_rate` | the drawn value; NaN where the stage did not fire |
| `codec_kind` | `CODEC_CODES`: libopus 1.0, g722 2.0 |

Every row carries every key, so the collate can concatenate one tensor per key;
strings cannot take that path, which is why the codec is a float code. The random
IIR records only that it fired, not its coefficients. Evaluation buckets rows by
these values, for example `egs/voice_isolate/scripts/eval_indomain.py --by-bucket`.

## Paired chain views

`puresound.task.paired_views.apply_chain_views` can run a second, independent
draw of the chain on the identical post-noise pair, for consistency supervision.
Voice isolation enables it on session rows through
`augmentation_session_rows.paired_view_prob` (rows at least
`paired_view_min_seconds` long); the second view is collated under `paired_view`.
With probability 0 it performs no clones and no draws.

## Configuration

| Stage | YAML block | Schema (`puresound.config.augmentation`) | Tasks |
|---|---|---|---|
| SRC | `augmentation_src`: `src_range`, `prob_each` | `SourceRateAugmentation` | NS, VI, TSE |
| IIR | `augmentation_ir_response` | `SimpleProbAugmentation` | NS, VI, TSE |
| HPF | `augmentation_hpf`: `cutoff`, `prob_each` | `HighPassAugmentation` | NS, VI, TSE |
| volume | `augmentation_volume`: `perturbed_range`, `clipping_prob`, `clipping_range.min`, `clipping_range.max`, `target_clipping` | `VolumeAugmentation` | NS, VI, TSE |
| compressor | `augmentation_compressor`: `threshold_db_range`, `ratio_range`, `attack_ms_range`, `release_ms_range` | `CompressorAugmentation` | NS, VI, TSE |
| codec | `augmentation_codec`: `codecs`, `prob_each`, `bitrate_range` | `CodecAugmentation` | NS, VI |
| packet loss | `augmentation_packet_loss`: `packet_ms_choices`, `loss_rate_range` | `PacketLossAugmentation` | NS, VI |

```yaml
augmentation_src:
  used: True
  prob: 0.5
  src_range: [8000, 16000]
  prob_each: [0.2, 0.8]
augmentation_hpf:
  used: True
  prob: 0.25
  cutoff: [100, 200, 300]
  prob_each: [0.4, 0.4, 0.2]
```

A `src_range` entry equal to the row's sample rate is a no-op resample, but it
still sets `src_applied` and consumes the same draws.

## Notes

- The VAD reference is snapshotted before the chain (`ns.py`), so chain damage
  never changes activity labels.
- The compressor curve and the signals are aligned by the shortest length; any
  uncovered tail samples keep their values.
- Any behavioural change to the chain changes the synthesis distribution, even a
  fix; see [Engineering contract](engineering_contract.md).
- `_fires(block, probability_draw=False)` honours `used` without a probability
  draw. No stage uses it; using it changes that row's RNG consumption.
