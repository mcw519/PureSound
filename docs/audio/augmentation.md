# puresound.audio.augmentation

繁體中文版本：[`augmentation.zh-TW.md`](augmentation.zh-TW.md)

Composable audio augmentation for speech-processing datasets: noise
injection, reverberation (folder RIRs, an on-the-fly physics simulator, or a
pre-generated bank), sox-style volume/speed/pitch perturbation, channel
coloring (biquad, codec round-trip, packet loss, media-device EQ), and
sample-rate-conversion artifacts. `puresound/dataset/*.py` and
`puresound/task/*.py` drive all of it through one `AudioEffectAugmentor`
instance per dataset split.

Conventions used throughout this module:
- Waveforms are `[..., L]` (time last); most call sites pass mono `[1, L]`.
- Every augmentation method returns **`(augmented_wav, info)`** — `info` is
  whatever was actually randomized (ids, coefficients, gains...), so a
  caller can log it or re-apply the identical perturbation to a paired
  signal (e.g. the same RIR convolved into both a mixture and its clean
  target — see `apply_rir`'s `rir_id` below). This isn't decorative; several
  recipes depend on it.

## Class: `AudioEffectAugmentor`

### Constructor

```python
AudioEffectAugmentor()
```

Takes no arguments. State is empty until you call one of the loader methods:

| attribute | initial value | populated by |
|---|---|---|
| `bg_noise` | `{}` | `load_bg_noise_from_folder` |
| `rir` | `{}` | `load_rir_from_folder` |
| `room_simulator` | `None` | `init_room_simulator` |
| `room_bank` / `room_bank_kind` | `None` / `None` | `init_room_bank` |
| `simulated_rir` | `OrderedDict()` | `apply_rir` (LRU cache, see below) |
| `_last_rir_meta` | `None` | every `apply_rir` call |

`_last_rir_meta` is a debug/introspection convenience: the metadata of the
most recent `apply_rir` call. **Nothing in the library reads it.** Callers take
the metadata off the return value (`RirApplied.detail.metadata`), which is the
only way to attribute it to a specific call — the attribute mis-attributes as
soon as anything else convolves in between. `puresound/task/ns.py` used to read
it for per-interferer RIR bookkeeping and no longer does.

### Loading noise / RIR pools

#### `load_bg_noise_from_folder(folder: str, suffix: str = ".wav")`
#### `load_rir_from_folder(folder: str, suffix: str = ".wav")`

Both recursively walk `folder` (via `puresound.utils.recursive_read_folder`)
for files ending in `suffix` and register them **by path only** — nothing is
loaded into memory yet. `bg_noise`/`rir` end up as `{uttid: {"wav_path":
path}}`, where `uttid` is the filename with its extension stripped (any
remaining dots are rejoined with `_`, so `Room042.00093.wav` becomes the key
`Room042_00093`).

### Room acoustics: 3 mutually exclusive backends, checked in priority order

`apply_rir` sources its impulse response from, in priority order: (1) a
pre-generated bank, (2) an on-the-fly physics simulator, (3) a per-instance
cache of previously-simulated RIRs keyed by `rir_id`, (4) the static folder
pool loaded above. A recipe wires at most one of (1)/(2); (3)/(4) are always
reachable as fallbacks.

#### `init_room_simulator(config: dict)`

```python
aug.init_room_simulator({
    "room_dim_range": [[3.0, 8.0], [3.0, 8.0], [2.4, 3.5]],
    "rt60_range": [0.15, 0.8],
    "source_receiver_distance_range": [0.3, 4.0],
    "foreground_distance_range": [0.3, 1.2],
    "interferer_distance_range": [1.2, 4.0],
    "receiver_margin": 0.4,
    "source_margin": 0.4,
    "nsample": 8192,
    "hp_filter": True,
})
```

Builds `RoomImpulseResponseSimulator(**config)` (see
[room_simulator.md](room_simulator.md)) after popping keys that matter to
the *training recipe* but that the dataclass itself doesn't accept: `used`,
`source_level`, `pregenerated`. Those are read by the dataset layer
(`puresound/dataset/dynamic_base.py`) to decide whether to call this method
at all, and whether reverb should be applied per-source ("source-level" —
foreground and each interferer convolved with their own draw from the same
room) rather than once on the finished mixture.

#### `init_room_bank(config: dict)`

Wires a pre-generated RIR bank instead of simulating on the fly — see
[rir_bank.md](rir_bank.md) for the bank classes themselves and the full
training-config YAML contract. This method's own job is resolving *which*
bank class to build:

- `bank_type` defaults to `"release"` when `config` has a `recipe_id`, else
  `"room"` (the legacy directory-of-WAV bank).
- `bank_type: release` requires `usage_role` (one of `train`/`validation`/
  `test`), a non-empty `recipe_id`, and a `split` that must string-equal
  `usage_role` — this stops a train-time reader from silently pulling
  test-split rooms.
- `bank_type: room` rejects release-only keys outright (`recipe_id`,
  `release_manifest_name`, `require_production`,
  `production_decision_name`, `usage_role`), so a legacy config can't
  half-opt into release semantics by accident.

Sets `self.room_bank` and `self.room_bank_kind` (`"release"` or `"room"`).

#### `sample_room_scene() -> Optional[dict]`

Returns `room_bank.sample_scene()` if a bank is wired, else
`room_simulator.sample_scene()` if a simulator is wired, else `None`. A
"scene" fixes the room geometry and receiver (mic) position **without a
source position yet** — see [room_simulator.md](room_simulator.md) for why:
`apply_rir`'s `source_role` argument decides the source's distance range at
convolution time, so the *same* scene can be reused to place a foreground
speaker close and an interferer far apart, inside one room.

### Sox-based single-value perturbations

Unlike `add_bg_noise`/`apply_rir`, these three each take one already-resolved
value, not a range — the recipe layer samples a value from its configured
range once per item and passes the resolved value in.

#### `sox_volume_perturbed(wav: Tensor, vol_ratio: float, sr: int) -> (Tensor, vol_ratio)`

Sox `vol` effect: a **linear** amplitude multiplier (not dB). The source
docstring's quoted typical range is `[0.125, 2]`. Falls back to plain
`wav * vol_ratio` if the installed torchaudio has no `sox_effects` backend.

#### `sox_speed_perturbed(wav: Tensor, speed: float, sr: int) -> (Tensor, speed)`

Sox `speed`+`rate` chain: changes duration **and** pitch together (unlike a
tempo-only stretch). Typical range `[0.8, 1.2]`. Fallback (no sox):
`torchaudio.functional.resample` at a shifted rate — same duration/pitch
trade-off, lower filter quality.

#### `sox_pitch_perturbed(wav: Tensor, shift_ratio: int, sr: int) -> (Tensor, shift_ratio)`

Sox `pitch` effect: shift in **cents** (100 cents = one semitone). Typical
range `[-100, 100]`. **No fallback** — if torchaudio has no sox backend,
this silently returns `wav` **unmodified** (unlike the two methods above,
which approximate the effect a different way when sox is unavailable).

### Noise injection

#### `add_bg_noise(wav, snr_list: List[float], sr: int, dynamic_type: bool = False, noise_id: Optional[List[str]] = None, noise_transform=None) -> (List[Tensor], (added_noise, noise_id, snr_list))`

Pool-aware wrapper over [`noise.add_bg_noise`](noise.md): picks (or accepts
an explicit) noise id from `self.bg_noise`, loads and resamples it to `sr`,
optionally passes it through `noise_transform` **before** SNR mixing (e.g.
convolve the noise with a room channel so it shares the speech's acoustic
space), then mixes it at *every* SNR in `snr_list` in one call.

- `dynamic_type=True` samples **2** noise ids instead of 1; `noise.add_bg_noise`
  concatenates them end to end into one noise bed before mixing.
- `noise_id`, when given explicitly, must be a `list` — needed when the
  caller wants a specific, reproducible noise (e.g. to reuse the same bed
  across two variants of a mixture).
- Returns a **list** of noisy waveforms, one per `snr_list` entry — not a
  single tensor matching `wav`'s shape.

#### `add_bg_white_noise(wav, snr_list: List[float]) -> (List[Tensor], (noise, snr_list))`

Thin wrapper over [`noise.add_bg_white_noise`](noise.md); same
multi-SNR/list-output contract, no pool involved.

### Reverberation

#### `apply_rir(wav, rir_mode: str = "image", sr: int = 16000, rir_id: Optional[str] = None, room_scene: Optional[dict] = None, source_role: str = "source", distance_range_override: Optional[List[float]] = None) -> RirApplied`

> **`rir_mode="image"` is not itself a valid mode.** Every real call site in
> this repo passes `rir_mode` explicitly — `"full"`, `"direct"`, or
> `"early"` (see [impulse_response.md](impulse_response.md) for what each
> trims). `apply_rir(wav)` with no `rir_mode` raises an `AssertionError`
> inside `wav_apply_rir`. A fourth value, `"anechoic"`, appears in recipe
> YAML (`target_rir_type`) but is handled by the *caller*
> (`puresound/dataset/dynamic_base.py`) as "skip reverb, return the dry
> signal" — it is never passed down to `apply_rir` itself.

RIR source, in priority order (first match wins):

1. **Bank** (`self.room_bank` set, `rir_id is None`): reuses `room_scene` if
   it's a bank-tagged scene (`room_scene["_bank"]` truthy), else draws a
   fresh one; selects a channel for `source_role`, optionally narrowed by
   `distance_range_override` — though a pre-generated bank can only pick the
   *closest available* channel to that range, not sample it exactly (see
   [room_simulator.md](room_simulator.md)); caches the result under a new
   `"bank-{n}"` id.
2. **Simulator** (`self.room_simulator` set, `rir_id is None`): calls
   `room_simulator.generate(sample_rate=sr, scene=room_scene,
   source_role=source_role, distance_range_override=...)` (see
   [room_simulator.md](room_simulator.md) for how `source_role` picks a
   distance range, and how `distance_range_override` is honored *exactly*
   here, unlike in the bank path); caches the result under a new
   `"simulated-{n}"` id.
3. **Simulated-RIR cache hit** (`rir_id` names an entry already in
   `self.simulated_rir`): reuses that exact impulse response (resampling if
   `sr` differs from when it was cached). This is how a dataset builds a
   noisy/clean pair sharing one room: call once with `rir_id=None` to get an
   id back, then call again with that `rir_id` and a *different* `rir_mode`
   (e.g. `"full"` for the noisy mixture, the recipe's `target_rir_type` —
   often `"early"` or `"direct"` — for the clean target) to convolve the
   *same* RIR two ways. See `apply_source_level_target_reverb` in
   `puresound/dataset/dynamic_base.py` for exactly this pattern. The cache
   is an `OrderedDict` LRU capped at `simulated_rir_cache_size` (32) entries.
4. **Static folder pool** (fallback): a random key from `self.rir` (loaded
   via `load_rir_from_folder`) if `rir_id is None`, else a direct lookup.

Returns `RirApplied(wav, RirDetail(rir_id, {"mode": rir_mode, "metadata": rir_metadata}))`.

Both are `NamedTuple`s of exactly two fields, so the historical
`wav, (rir_id, info) = aug.apply_rir(...)` unpacking and `info["metadata"]`
indexing are unchanged. New code should prefer the named access —
`result.wav`, `result.detail.rir_id`, `result.detail.metadata` (a property that
digs `"metadata"` out of the info dict, `None` for folder RIRs).
`rir_metadata` is `None` for folder-pool RIRs, and the simulator/bank
metadata dict otherwise (room dims, receiver, source, rt60, `source_role`,
`source_receiver_distance`, `drr_db` — see [room_simulator.md](room_simulator.md)).

### Channel coloring / distortion

#### `apply_2nd_iir_response(wav, a_coeffs: Optional[Tensor] = None, b_coeffs: Optional[Tensor] = None) -> (Tensor, (a_coeffs, b_coeffs))`

Random 2nd-order IIR coloring (mic/channel simulation), from *A Hybrid
DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement*. When
`a_coeffs`/`b_coeffs` are `None` they're drawn uniformly from `[-3/8, 3/8]`
(3 free coefficients each; `a[0]=b[0]=1` implicit). Pass the returned
coefficients back in to apply the *same* random filter to a second signal.

#### `apply_gain_distortion(wav, sr: int) -> (Tensor, (start_sample, duration_samples, gain))`

Delegates to `volume.rand_gain_distortion` (see [volume.md](volume.md)) —
distorts a random *segment* of `wav`, not the whole signal, with a random
gain, then clips to `[-1, 1]`. There are no min/max-gain parameters here;
segment placement, duration, and gain are all randomized inside
`rand_gain_distortion`.

#### `apply_clipping_distortion(wav, min_quantile: float, max_quantile: float) -> (Tensor, (min_quantile, max_quantile))`

Delegates to `volume.wav_clipping` (see [volume.md](volume.md)) — note that
module's default bounds are asymmetric; here both bounds are required
arguments.

#### `apply_src_effect(wav, sr: int, src_sr: int, src_backend: str) -> (Tensor, info_list)`

Down/up sample-rate-conversion round trip (`sr → src_sr → sr`) to simulate
bandwidth-limited transmission. `info_list` is whatever
[`dsp.wav_resampling`](dsp.md) returned beyond the waveform — for
`src_backend="torchaudio"` that includes the randomized anti-aliasing
filter params, which are deliberately re-used for the up-conversion leg
(same `lp_width`/`rolloff`/`window`) so the pair of resamples behaves like
one coherent low-quality resampler rather than two independently-randomized
ones.

#### `apply_hpf(wav, sr: int, cutoff_freq: int, q_factor: float) -> (Tensor, (cutoff_freq, q_factor))`

`torchaudio.functional.highpass_biquad` at a single cutoff — not a range,
same "recipe resolves the range" convention as the sox perturbations above.

#### `apply_media_coloring(wav, sr: int, hp_cutoff: float, lp_cutoff: float, compress_power: Optional[float] = None) -> (Tensor, (hp_cutoff, lp_cutoff, compress_power))`

Simulates playback through a TV/loudspeaker-class device: band-limits to
`[hp_cutoff, lp_cutoff]` (two cascaded biquads), then optional
`compress_power in (0, 1]` peak-normalized waveshaping (`|x|^p`; `1.0`/`None`
= no compression, models broadcast-chain light dynamic-range compression).
RMS is restored to the pre-coloring level afterward, so downstream SIR/level
scaling isn't confounded by the coloring step itself. Used in
`puresound/task/ns.py` to color interferers flagged as media sources — see
`source_role="media"` in [room_simulator.md](room_simulator.md).

#### `apply_codec(wav, sr: int, codec_name: str, bit_rate: Optional[int] = None) -> (Tensor, (codec_name, bit_rate))`

Encode/decode round trip through `torchaudio.io.AudioEffector` to simulate a
VoIP/telephony codec. `codec_name` must be one of `supported_codecs()`
(currently `"libopus"`, `"g722"`, each mapped internally to the container
format its encoder needs — `ogg`/`matroska` respectively, picked for
round-trip reliability rather than Opus/AAC-in-MKV quirks). Output is
cropped or zero-padded back to the input's exact sample count (a codec can
change length via internal resampling/framing). `bit_rate` is ignored for
codecs with no bitrate knob (`g722`).

#### `supported_codecs() -> List[str]` *(staticmethod)*

Returns the codec names `apply_codec` accepts.

#### `apply_packet_loss(wav, sr: int, packet_ms: int = 20, loss_rate: float = 0.05) -> (Tensor, (packet_ms, loss_rate, n_dropped))`

Zeroes random `packet_ms`-sized chunks to mimic VoIP drop-outs (20 ms is the
WebRTC default; 60 ms is typical for low-bandwidth Opus). Each packet is
dropped independently with probability `loss_rate`. `n_dropped` in the
returned info is the *actual* number zeroed (0 both when the random draw
happens to drop nothing and when `loss_rate <= 0` — both return a cloned,
otherwise-untouched waveform).

## Example

```python
from puresound.audio.augmentation import AudioEffectAugmentor

aug = AudioEffectAugmentor()
aug.load_bg_noise_from_folder("/data/musan/noise")
aug.load_rir_from_folder("/data/rirs")

noisy_list, (added_noise, noise_id, snr_list) = aug.add_bg_noise(
    wav=clean_wav, snr_list=[0.0], sr=16000
)
reverberant, (rir_id, rir_info) = aug.apply_rir(
    wav=noisy_list[0], rir_mode="full", sr=16000
)
```
