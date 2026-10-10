# Distance and timing cues

繁體中文版本：[distance_cues.zh-TW.md](distance_cues.zh-TW.md)

Voice isolation keeps the near-field talker and suppresses the rest, so what a
mixture reveals about source distance decides what the model can learn. This
page lists the physical distance cues, states which ones the pipeline keeps and
which it removes, and describes the three techniques that manipulate them: DRR
contrast, direct-arrival smear and `distance_level` mixing.

Code: `AudioEffectAugmentor._apply_drr_contrast` / `_apply_direct_smear`
(`puresound/audio/augmentation.py`), `smear_direct_arrival`
(`puresound/audio/impulse_response.py`), `VoiceIsolationDataset._distance_level_sir`
(`puresound/task/voice_isolation.py`).

## Algorithm

### 1. The distance cues

#### 1.1 Level (inverse distance law)

```
p(d) ∝ 1/d      →      L(d) = L_ref − 20·log10(d / d_ref)
```

6 dB per doubling of distance; 0.5 m against 3 m is `20·log10(6) ≈ 15.6 dB`.

**Pipeline: removed.** Corpus utterances are RMS-rescaled at load to
`dataset.gain_normalized_to` when a recipe sets it, and every source's RIR is
peak-normalised on its own ([room acoustics](room_acoustics.md) §3.2), so the
distance gain between sources is gone. Level is also the cue most contaminated
in real recordings (vocal effort, mouth directivity, microphone gain); removing
it stops the model from using "quiet means far" and makes it rely on channel
properties. `distance_level` mixing (§5) puts it back on a share of rows.

#### 1.2 Arrival time between sources

`Δt = Δd / c`, about 7.3 ms between 3 m and 0.5 m.

**Pipeline: removed.** Each convolved source is aligned on its own direct-path
peak ([room acoustics](room_acoustics.md) §3.3).

#### 1.3 Direct-to-reverberant ratio (DRR)

Direct energy falls as `1/d²` while the diffuse reverberant energy is roughly
independent of position, so

```
DRR_dB(d) = 20·log10(r_c / d)
```

with `r_c` the room's critical distance ([room acoustics](room_acoustics.md) §5).

**Pipeline: kept**; DRR contrast (§3) can widen it. With level removed it is the
main distance cue.

#### 1.4 Fine timing inside the direct window

In the first few milliseconds after the direct sound, single reflections off the
floor, the desk or a nearby wall arrive. Their delays relative to the direct
sound depend on geometry: for a close source the first reflection travels a
proportionally much longer path (later, weaker), for a distant source the path
difference shrinks (earlier, relatively stronger). The shape of this impulse
cluster — spacing and relative amplitudes — carries distance information that
is separate from the window's spectrum.

**Pipeline: kept**; direct smear (§4) can destroy it.

#### 1.5 Decay shape (RT60, EDC)

The decay rate is set by the room, not by distance. It tells how reverberant the
room is, which calibrates DRR: the same DRR means different distances in rooms
with different RT60.

**Pipeline: kept.**

#### 1.6 Spectral tilt

A distant source has relatively less high-frequency energy because air
absorption rises with frequency (mainly above 4 kHz indoors) and most surfaces
absorb more at high frequencies, and a far-field signal is a larger share of
reflected energy.

**Pipeline: kept**, with the fidelity of the RIR source: the image-source
simulator has frequency-independent walls and no air absorption
([room acoustics](room_acoustics.md) §7); the hybrid bank models both.

#### 1.7 Summary

| Cue | Physical origin | Pipeline |
|---|---|---|
| Level | `p ∝ 1/d` | Removed (load-time RMS, per-source RIR peak normalisation); `distance_level` restores it |
| Arrival time between sources | `Δd / c` | Removed (per-source peak alignment) |
| DRR | Direct `1/d²` against a uniform reverberant field | Kept; DRR contrast can widen it |
| Direct-window fine timing | First-reflection path difference | Kept; direct smear can destroy it |
| Decay shape | Room absorption and volume | Kept |
| Spectral tilt | Air and surface absorption | Kept, as far as the RIR source models it |

### 2. Two kinds of intervention

* **Amplify a cue** (DRR contrast, `distance_level`): make near and far more
  separable in the training distribution than they are naturally.
* **Remove a cue** (direct smear): stop the model from relying on one cue, so it
  must use others where that cue is masked.

They act on different cues and can be combined.

### 3. DRR contrast

#### 3.1 Operation

On a freshly served RIR channel, scale the whole tail after the direct window:

```
tail_start = peak + max(1, round(w · fs / 1000)),   peak = argmax|h|,  w = direct_window_ms
h[n] ← h[n] · 10^(−s/20)    for n ≥ tail_start
```

`s` is the DRR shift in dB.

#### 3.2 The shift is exactly s

With direct energy `E_d` and tail energy `E_r`, the scaled tail has energy
`E_r · 10^(−s/10)`, so

```
DRR' = 10·log10( E_d / (E_r · 10^(−s/10)) ) = DRR + s
```

This holds only if the scaled interval is the measured one. `direct_window_ms`
defaults to 2.5 ms, the same window and the same rounding as `compute_drr_db`;
set both to the same value if either is changed.

#### 3.3 Why the tail and not the direct path

Either changes DRR, but the RIR is peak-normalised downstream. Scaling the
direct path changes `max|h|`, so after normalisation it amounts to changing the
tail's absolute level too; scaling the tail leaves the peak and the
normalisation factor alone. Only the ratio reaches the model input, and the
operation stays single-purpose.

#### 3.4 Modes

**`random`**: the direction follows the source role, applied with probability
`prob`.

```
foreground:                  s = +U(near_boost_db)
interferer / media / echo:   s = −U(far_cut_db)
any other role:              untouched
```

**`deterministic`**: the shift is a fixed function of the channel's realised
distance, independent of role, with no probability draw and no randomness.

```
s = extra_db_per_decade · log10(d / pivot_m)
```

With the natural `DRR ≈ −20·log10(d) + C`:

```
DRR'(d) = −(20 − extra_db_per_decade)·log10(d) + C'
```

A negative `extra_db_per_decade` steepens the whole pool's distance-to-DRR
slope; `pivot_m` is the distance left unchanged (`s = 0` at `d = pivot_m`).

Random mode draws a new shift per row, so the same distance gets different DRR
values on different rows: it decouples DRR from distance and the pool no longer
has a single distance-to-DRR mapping. Deterministic mode keeps one mapping,
which is what a model reading DRR as distance needs; that is also why it
consumes no randomness — the same channel always gets the same shift.

#### 3.5 Parameters

| Key | Mode | Default | Meaning |
|---|---|---|---|
| `mode` | — | `random` | `random` or `deterministic` |
| `direct_window_ms` | both | 2.5 | Direct window, ms (> 0) |
| `prob` | random | 0.5 | Probability per eligible channel, in (0, 1] |
| `near_boost_db` | random | `[0, 4]` | DRR increase range for `foreground`, dB, `0 ≤ low ≤ high` |
| `far_cut_db` | random | `[0, 4]` | DRR decrease range for far roles, dB, `0 ≤ low ≤ high` |
| `extra_db_per_decade` | deterministic | required | Slope change, dB per decade of distance |
| `pivot_m` | deterministic | 1.0 | Distance with zero shift, m (> 0) |

A key that belongs to the other mode is rejected.

#### 3.6 Where it is applied

In `apply_rir`, on the bank and simulator paths, immediately after the channel
is served and before it enters the RIR cache: the target re-fetches the same
impulse by `rir_id`, so mixture and target see the same shift. Metadata records
`drr_contrast_shift_db` and the recomputed `drr_db`, so evaluation can bucket
rows by the DRR they actually had. Every channel served on those paths is
eligible, including an echo channel and the channel used to colour noise
(role `interferer`).

### 4. Direct-arrival smear

#### 4.1 Purpose

Destroy the direct-window timing structure of §1.4 on a fraction of rows, so a
distance readout that depends on it alone fails there. Reverberation masks this
cue naturally (strong reflections follow the direct window in a reverberant
room); whether a substitute such as spectral tilt is then learnable is open,
because timing and spectrum are read jointly. No released recipe enables it.

#### 4.2 Operation

`smear_direct_arrival(impaulse, sample_rate, smear_ms=..., generator=None)`, per
channel:

```
n = round(smear_ms · fs / 1000)             (no-op if n < 2)
window = h[peak : peak + n],  peak = argmax|h|
E = Σ window²

k = randn(n);  k ← k / ||k||₂                # random unit-energy kernel
window ← causal_conv(window, k)             # left zero-padding only

if |window[0]| < max|window|:                # keep the peak where it was
    window[0] ← sign(window[0]) · max|window| · 1.001

window ← window · sqrt(E / Σ window²)        # restore the window's energy
h[peak : peak + n] ← window
```

#### 4.3 Design

* **Random unit-energy kernel.** Convolving with a random sequence spreads the
  few concentrated impulses over the window and destroys their spacing. Unit L2
  norm keeps the convolution roughly energy-preserving, so the restoration step
  corrects little.
* **Causal.** Only left padding, so no energy appears before the direct
  arrival; that would be unphysical and would move every offset derived from
  the peak.
* **Peak index preserved.** `wav_apply_rir` cuts the target window and aligns
  propagation delay on the peak; moving it would move every downstream offset.
  The dominance fix comes before energy restoration: fixing afterwards would add
  energy back.
* **Window energy preserved.** The operation changes timing, not level.
* **Tail untouched.** Samples after `smear_ms` are unchanged, so RT60 is
  unchanged and DRR nearly so.

#### 4.4 Parameters and application point

| Key | Meaning |
|---|---|
| `prob` | Probability per served channel, in (0, 1] (required) |
| `smear_ms_range` | `[low, high]` ms, `0 < low ≤ high` (required); `smear_ms ~ U(low, high)` |

Applied right after DRR contrast and before the cache, for the same reason: the
`early` target must be the near part of the smeared `full` mixture. Metadata
records `direct_smear_ms`.

### 5. `distance_level` mixing

#### 5.1 Operation

Reinstate the level cue of §1.1 at the mixing stage from the scene geometry:

```
SIR_dB = 20 · log10(d_itf / max(d_fg, 1e-3)) + U(jitter_db)
```

`d_fg` is the foreground's realised distance and `d_itf` the distance of the
**nearest** interferer. The foreground and the summed interferers are then
mixed at that SIR with the same `add_bg_noise` mechanics as every hard-SIR row
([level and dynamics](level_dynamics.md)).

#### 5.2 Meaning of the terms

* `20·log10(d_itf / d_fg)` is the free-field amplitude ratio of two point sources
  at the microphone: 0.5 m against 3 m gives about +15.6 dB.
* **Nearest interferer**: under `1/r` it is the loudest and sets the effective
  SIR; a mean distance would let several distant interferers raise the SIR and
  understate the interference.
* **`jitter_db`** (default `[-3, 3]` dB) stands in for what level depends on
  besides distance — mouth directivity, vocal effort, head orientation,
  microphone pattern. Without it SIR would be a deterministic function of
  distance, a shortcut that real recordings do not offer.

#### 5.3 Fallback

When either distance is missing (folder RIRs, rows without source-level
reverb), `_distance_level_sir` returns `None` and the row falls back to the
hard-SIR draw from `augmentation_speech.snr_range`. The row still trains, without
the level cue.

## Engineering

### Config mapping

| Knob | Schema | Where it lands |
|---|---|---|
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | `AudioEffectAugmentor._apply_drr_contrast`, bank and simulator paths, before the cache |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | `AudioEffectAugmentor._apply_direct_smear`, same place |
| `augmentation_speech.mix_mode.modes[].distance_level` | `MixModeEntry` | `VoiceIsolationDataset._distance_level_sir` |

Both RIR knobs are off unless their block sets `used: true`. `mix_mode` exists
only in the voice-isolation task; a mode entry combines `name`, `prob` and
`distance_level: true`:

```yaml
augmentation_speech:
  mix_mode:
    used: true
    modes:
      - {name: physical,       prob: 0.4, physical: true}
      - {name: distance_level, prob: 0.2, distance_level: true, jitter_db: [-3, 3]}
      - {name: moderate,       prob: 0.4, sir_range: [-3, 6]}
```

Validation: deterministic DRR contrast requires `extra_db_per_decade`; random
mode ranges must be non-negative (the role gives the sign); `smear_ms_range`
must start above 0 ms (0 ms is a no-op already covered by `prob < 1`).

### Ordering and RNG

* On each served channel: **DRR contrast → direct smear → cache**.
* Both probability draws sit inside their short circuits, so a disabled knob
  consumes nothing and other recipes regenerate bit-identically
  ([engineering contract](engineering_contract.md)).
* Random DRR contrast and the smear draw (`prob`, `smear_ms`) use Python
  `random`; the smear kernel uses the global `torch` stream (`generator` is
  available for an independent one, but the pipeline does not pass it).
  Deterministic DRR contrast draws nothing.
* The `distance_level` jitter uses the `torch` stream, drawn after the mode
  selection.

### Pitfalls

* Folder RIRs carry no metadata: deterministic DRR contrast skips them and
  `distance_level` falls back. Check the RIR source first when a knob seems to
  do nothing.
* Random DRR contrast ignores roles other than `foreground`, `interferer`,
  `media` and `echo`. The whole-mixture reverb uses role `source`, so random
  mode leaves it alone; deterministic mode, being role-free, still shifts it
  when the channel has a distance.
* Rows with real-far interferers (real-far rows, and real-near rows, which draw
  their interferers from the real-far pool) skip `mix_mode` entirely,
  `distance_level` included: the level ratio between two differently recorded
  and normalised signals has no physical meaning, so those rows use the hard
  SIR. Session rows use their own SIR draw
  ([scene construction](scene_construction.md) §4).
