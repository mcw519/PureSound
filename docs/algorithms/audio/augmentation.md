# puresound.audio.augmentation

繁體中文版本：[augmentation.zh-TW.md](augmentation.zh-TW.md)

`AudioEffectAugmentor` is the per-dataset toolbox the synthesis pipeline draws
on: noise pools, RIR sources (a folder, the on-the-fly simulator, or a
pre-generated bank), two RIR-channel augmentations, and the level, speed,
pitch, filter, codec and packet-loss operators. The dataset layer builds one
instance per split in `init_augmentor` (`puresound/dataset/dynamic_base.py`);
the task datasets, `NoiseStage` and `DeviceChain` call its methods. Where each
operator sits in a training row is described in
[data augmentation](../augmentation/index.md).

Conventions:

- Waveforms are `[C, L]`, almost always mono `[1, L]`.
- Every operator returns `(wav, info)`, where `info` is what was randomised
  (ids, coefficients, gains). Passing `info` back in applies the identical
  perturbation to a paired signal, which is how a mixture and its target get the
  same filter or the same RIR.

## State

| attribute | filled by |
| --- | --- |
| `bg_noise`, `bg_noise_groups` | `load_bg_noise_from_folder`, `load_bg_noise_sources` |
| `rir` | `load_rir_from_folder` |
| `room_simulator` | `init_room_simulator` |
| `room_bank`, `room_bank_kind` (`"room"`, `"release"`, `"union"`) | `init_room_bank` |
| `drr_contrast`, `direct_smear` | `init_drr_contrast`, `init_direct_smear` |
| `simulated_rir` | `apply_rir`: LRU cache of served impulses, 32 entries |
| `_last_rir_meta` | `apply_rir`; for interactive inspection only, nothing reads it |

## Noise pools

- `load_bg_noise_from_folder(folder, suffix=".wav")`: registers every file under
  `folder` (recursive) by path; nothing is loaded yet. The key is the file name
  without extension, inner dots joined by `_`. YAML:
  `augmentation_noise.noise_folder`.
- `load_bg_noise_sources([(name, folder, weight), ...])`: one pool from several
  corpora. Keys are prefixed `name/`, so equal basenames do not collide. A draw
  first picks a source with probability proportional to `weight`, then a file
  uniformly within it, so a small corpus keeps its share. YAML:
  `augmentation_noise.noise_sources: [{name, folder, weight}]`.

## RIR sources

`apply_rir` takes its impulse from the first available of: a bank, the
simulator, the cache (when a known `rir_id` is passed), the folder pool. A
recipe enables at most one of bank and simulator.

- `load_rir_from_folder(folder)`: static WAV pool, `augmentation_reverb.rir_folder`.
- `init_room_simulator(config)`: builds
  [`RoomImpulseResponseSimulator`](room_simulator.md) from
  `augmentation_reverb.simulator`, after removing the keys the dataset layer
  owns (`used`, `source_level`, `pregenerated`).
- `init_room_bank(config)`: builds a pre-generated bank from
  `augmentation_reverb.simulator.pregenerated`, which holds either one bank
  (`folder`) or a `banks:` list. See [RIR bank loaders](rir_bank.md) for the
  classes.
  - `bank_type` defaults to `"release"` when `recipe_id` is given, else
    `"room"` (a directory bank).
  - A release bank needs `usage_role ∈ {train, validation, test}`, a non-empty
    `recipe_id`, and `split == usage_role`, so a training reader cannot pull
    test rooms. The dataset layer sets `usage_role` from its pipeline role and
    rejects a configured value that disagrees.
  - A room bank rejects release-only keys (`recipe_id`,
    `release_manifest_name`, `require_production`, `production_decision_name`,
    `audit`, `audit_cache`).
  - A `banks:` list builds each member the same way and serves them through
    `UnionRoomBank` at the members' `weight`s (sampling probabilities, not item
    counts). Only `usage_role` may be set beside the list.
- `sample_room_scene()`: a bank scene if a bank is wired, else a simulator
  scene, else `None`. A scene fixes the room and receiver without a source, so
  one scene serves the foreground and every interferer of a row.

## RIR-channel augmentations

Both act on a freshly served impulse **before it is cached**. The target
re-fetches the same `rir_id` to convolve the same impulse in a shorter window,
so the `"early"` target stays the near component of the `"full"` mixture. Both
consume no randomness when disabled.

**`init_drr_contrast(config)`** (`augmentation_reverb.drr_contrast`) scales the
tail past `peak + direct_window_ms` (default 2.5 ms, the pipeline's DRR window)
by `10^(−Δ/20)`, which shifts the channel's DRR by exactly `+Δ` dB; the metadata
gets `drr_contrast_shift_db` and the recomputed `drr_db`.

| `mode` | Δ | randomness |
| --- | --- | --- |
| `"random"` (default) | with probability `prob` (0.5): `+U(near_boost_db)` for `foreground`, `−U(far_cut_db)` for `interferer`/`media`/`echo` (both `[0, 4]` dB by default); other roles untouched | Python `random` |
| `"deterministic"` | `extra_db_per_decade · log10(d / pivot_m)` from the channel's realised distance `d` (`pivot_m` default 1 m); role-free | none |

The random mode widens the near/far DRR gap per draw, which also decouples DRR
from distance; the deterministic mode steepens the pool's distance→DRR slope
while keeping one consistent mapping.

**`init_direct_smear(config)`** (`augmentation_reverb.direct_smear`): with
probability `prob`, applies
[`smear_direct_arrival`](impulse_response.md) with `smear_ms ~ U(smear_ms_range)`
and records `direct_smear_ms` in the metadata.

## `apply_rir(wav, rir_mode="full", sr=16000, rir_id=None, room_scene=None, source_role="source", distance_range_override=None) -> RirApplied`

`rir_mode` must be `"full"` (the default), `"early"` or `"direct"`; any other
value fails the assertion in [`wav_apply_rir`](impulse_response.md).
(`target_rir_type: anechoic` in a recipe is handled by the dataset layer, which
then skips the call.)

1. **Bank** (`room_bank` set, `rir_id is None`): reuses `room_scene` if it is a
   bank scene (`room_scene["_bank"]`), else draws one; selects a channel for
   `source_role`, closest to `distance_range_override` if given; resamples to
   `sr`; applies the channel augmentations; caches it as `bank-<n>`.
2. **Simulator** (`room_simulator` set, `rir_id is None`): `generate(sr, scene,
   source_role, distance_range_override)`, the override honoured exactly;
   channel augmentations; cached as `simulated-<n>`.
3. **Cache** (`rir_id` in `simulated_rir`): the stored impulse, resampled if
   `sr` differs.
4. **Folder**: a random key of `rir`, or the given `rir_id`.

The paired-target pattern is two calls: the first with `rir_id=None` and
`rir_mode="full"` returns an id; the second passes that id with the recipe's
`target_rir_type`. `apply_source_level_target_reverb` in the dataset layer does
exactly this.

Returns `RirApplied(wav, RirDetail(rir_id, {"mode": rir_mode, "metadata": ...}))`,
two nested `NamedTuple`s, so `wav, (rir_id, info) = aug.apply_rir(...)` unpacks.
`result.detail.metadata` is the simulator or bank placement (room, receiver,
source, `rt60`, `source_role`, `source_receiver_distance`, `drr_db`, plus the
augmentation fields), or `None` for a folder RIR. Take metadata from the return
value; `_last_rir_meta` is overwritten by every call and mis-attributes as soon
as another convolution happens in between.

## Level, speed and pitch

Each takes one resolved value; the recipe layer draws it from its range.

| method | operation |
| --- | --- |
| `sox_volume_perturbed(wav, vol_ratio, sr)` | `wav · vol_ratio`, a linear gain that cannot clip |
| `sox_speed_perturbed(wav, speed, sr)` | `torchaudio.functional.resample(orig=int(sr·speed), new=sr)`: length scales by `1/speed`, pitch by `speed` (sox `speed` semantics) |
| `sox_pitch_perturbed(wav, shift_ratio, sr)` | `torchaudio.functional.pitch_shift` with `bins_per_octave=1200`, so `shift_ratio` is in cents; a phase vocoder, duration unchanged; 0 is a no-op |

The names keep sox's vocabulary; none of them calls sox.

## Noise

- `add_bg_noise(wav, snr_list, sr, dynamic_type=False, noise_id=None, noise_transform=None)`:
  draws a noise id (or takes `noise_id`: the id a previous call returned, or a list of ids), loads and resamples it to
  `sr`, applies `noise_transform` (for example a room channel so the noise
  shares the speech's room), then mixes with [`noise.add_bg_noise`](noise.md) at
  every SNR in `snr_list`. `dynamic_type=True` draws two clips and concatenates
  them into one bed. Returns `(noisy_list, (added_noise_list, noise_id, snr_list))`.
- `add_bg_white_noise(wav, snr_list)`: [`noise.add_bg_white_noise`](noise.md);
  returns `(noisy_list, (noise_list, snr_list))`.

## Filters and channel damage

| method | operation | info |
| --- | --- | --- |
| `apply_2nd_iir_response(wav, a_coeffs=None, b_coeffs=None)` | random stable biquad, [`rand_add_2nd_filter_response`](impulse_response.md) | `(a, b)` |
| `apply_hpf(wav, sr, cutoff_freq, q_factor)` | `highpass_biquad` under [`apply_linear`](dsp.md) | `(cutoff, Q)` |
| `apply_src_effect(wav, sr, src_sr, src_backend)` | `sr → src_sr → sr` with [`wav_resampling`](dsp.md); with `"torchaudio"` the random filter drawn for the first leg is reused for the second | resampler return values |
| `apply_gain_distortion(wav, sr)` | [`rand_gain_distortion`](volume.md): a random segment gain, then a clip to [-1, 1] | `(start, length, gain)` |
| `apply_clipping_distortion(wav, min_quantile, max_quantile)` | [`wav_clipping`](volume.md) at the signal's own quantiles | the two quantile fractions |
| `apply_media_coloring(wav, sr, hp_cutoff, lp_cutoff, compress_power=None)` | TV/loudspeaker playback: high-pass and low-pass biquads (Q 0.707) under `apply_linear`, then optional `sign(x)·abs(x/peak)^p·peak` waveshaping for `p < 1`; RMS restored to the input's | `(hp, lp, p)` |
| `apply_codec(wav, sr, codec_name, bit_rate=None)` | encode and decode through TorchCodec | `(codec, bit_rate)` |
| `apply_packet_loss(wav, sr, packet_ms=20, loss_rate=0.05)` | zero whole packets, each dropped independently with `loss_rate` (torch generator) | `(packet_ms, loss_rate, n_dropped)` |

`apply_media_coloring` restores RMS so the SIR scaling applied afterwards is not
changed by the coloring. Its waveshaper is a distortion of a playback device in
the room, applied to the interferer before mixing; it is not a model of
recording-chain compression, which is a gain curve (see
[`compressor_gain`](dsp.md)).

`apply_codec` supports `supported_codecs() == ["libopus", "g722"]`. TorchCodec's
encoder selects the codec by container, so the round trip writes a temporary
`.opus` or `.g722` file; decoding at `sample_rate=sr` undoes the codec's
internal rate (Opus encodes at 48 kHz, G.722 at 16 kHz), and the output is cut
or zero-padded to the input length. `bit_rate` is ignored by G.722.

## Example

```python
from puresound.audio.augmentation import AudioEffectAugmentor

aug = AudioEffectAugmentor()
aug.load_bg_noise_from_folder("/path/to/noise")
aug.init_room_simulator({
    "room_dim_range": [[3.0, 8.0], [3.0, 8.0], [2.4, 3.5]],
    "rt60_range": [0.15, 0.8],
    "source_receiver_distance_range": [0.3, 4.0],
})

scene = aug.sample_room_scene()
mix = aug.apply_rir(clean, rir_mode="full", sr=16000, room_scene=scene, source_role="foreground")
target = aug.apply_rir(clean, rir_mode="early", sr=16000, rir_id=mix.detail.rir_id).wav
(noisy,), _ = aug.add_bg_noise(mix.wav, snr_list=[5.0], sr=16000)
```
