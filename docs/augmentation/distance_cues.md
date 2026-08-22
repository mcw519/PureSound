# Distance and timing cues

繁體中文版本：[`distance_cues.zh-TW.md`](distance_cues.zh-TW.md)

The near/far decision is the core judgement in voice isolation. This chapter
derives the physical origin of every distance cue available in a mixture,
states which ones the pipeline keeps and which it removes, and analyses the
three techniques that manipulate them directly: DRR contrast,
direct-arrival smear, and `distance_level` mixing.

## Algorithm

### 1. The physical origin of each distance cue

Six classes of information are available when judging source distance. Each is
derived below, annotated with its status in the pipeline.

#### 1.1 Level (the inverse square law)

Sound pressure from a point source in a free field falls inversely with
distance:

```
p(d) ∝ 1/d      →      L(d) = L_ref − 20·log10(d / d_ref)
```

That is 6 dB per doubling of distance. The level difference between 0.5 m and
3 m is `20·log10(3/0.5) ≈ 15.6 dB`.

**Pipeline status: removed.** Two steps compound to cause this. At load time
`AudioIO.open` applies an RMS rescale to `target_lvl`, flattening the level of
every source; at convolution time the per-channel peak normalisation
([ch2](room_acoustics.md) §3.1) erases the distance gain ratios between
channels.

This is the most direct of all distance cues, and also the one most easily
contaminated by other real-world factors (vocal effort, mouth directivity,
microphone gain). The trade-off in removing it: the model cannot take the
shortcut "quiet means far" and must learn channel characteristics instead; the
cost is the loss of a genuinely present cue. The `distance_level` mixing mode
(§5) is the mechanism for putting it back explicitly.

#### 1.2 Inter-source arrival time difference

Sources at different distances have different speech onset times, with
`Δt = Δd / c`. Between 3 m and 0.5 m that is about 7.3 ms.

**Pipeline status: removed.** Each channel is aligned by its own `argmax|h|`
after convolution (ch2 §3.2), pulling every source's speech onset to the same
instant.

#### 1.3 Direct-to-reverberant ratio (DRR)

Direct-path energy falls as `1/d²`, while diffuse-field theory holds that
reverberant energy density is approximately uniform in the room and independent
of distance. Therefore:

```
DRR_dB(d) ≈ −20·log10(d) + C
```

About 6 dB per doubling of distance, with `C` set by the room's absorption and
volume. The distance at which `DRR = 0` is the critical distance, around 1–2 m
in a typical meeting room.

**Pipeline status: kept**, and can be amplified explicitly by DRR contrast
(§3). With the level cue removed this is the primary distance cue.

#### 1.4 Fine timing structure inside the direct window

Within the first few milliseconds after the direct sound arrives, components
reflected once off the floor, the desk, or a nearby wall arrive in succession.
Their path differences vary with source distance: a near-field source has a
large first-reflection path difference (longer relative delay, weaker
amplitude), while a far-field source has a small one (hugging the direct
sound, relatively stronger).

The "impulse cluster shape" inside the direct window — the intervals between
impulses and their relative amplitudes — therefore carries distance
information. This is a time-domain fine-structure cue, distinct from the
spectral shape of the same window.

**Pipeline status: kept**, and can be destroyed deliberately by direct smear
(§4).

#### 1.5 Decay shape (RT60, EDC)

The decay rate of the reverberant tail is determined mainly by the room, not
by distance. What it provides is "how reverberant this room is" rather than
"how far this source is", but jointly with DRR it helps calibration: the same
DRR corresponds to different distances in rooms with different RT60.

**Pipeline status: kept.**

#### 1.6 Spectral tilt

Two mechanisms leave a far-field source with relatively less high-frequency
content:

* **Air absorption**: the absorption coefficient rises with frequency
  (over indoor distances this mainly affects content above 4 kHz).
* **Frequency dependence of surface absorption**: most building materials
  absorb more at high frequencies, so components that underwent many
  reflections lose more treble. A far-field signal has a higher proportion of
  reflected energy and therefore a darker overall spectrum.

**Pipeline status: kept**, though the accuracy of spectral tilt in synthetic
RIRs is limited by the ISM assumptions (ch2 §7: frequency-independent
reflection coefficient).

#### 1.7 Summary table

| Cue | Physical origin | Pipeline status |
|---|---|---|
| Level | Inverse square law `p ∝ 1/d` | Removed (load-time RMS + RIR peak normalisation); `distance_level` can restore it |
| Inter-source arrival difference | `Δd / c` | Removed (per-channel argmax alignment) |
| DRR | Direct `1/d²` against a uniform reverberant field | Kept; DRR contrast can amplify |
| Direct-window fine timing | First-reflection path difference varies with distance | Kept; direct smear can destroy |
| Decay shape | Room absorption and volume | Kept |
| Spectral tilt | Air absorption plus surface HF absorption | Kept (limited by ISM assumptions) |

Which cues the model actually reads, and with what weight, is an experimental
question outside the scope of this document. The measurement definitions and
results of the window-ablation probe are in
`egs/voice_isolate/benchmarks/probes/dist_cue_anatomy_README.md`.

### 2. Why manipulate cues at all

Given the list in §1, there are two directions of intervention available at the
data layer, and the pipeline offers both:

* **Amplify a cue** (DRR contrast, `distance_level`): make the near/far
  separation in the training distribution more pronounced than it is naturally,
  lowering the learning difficulty. Used to establish basic near/far
  selectivity.
* **Remove a cue** (direct smear): prevent the model from relying on a single
  cue, forcing it to look for alternative evidence. Used to improve robustness
  in situations where that cue is masked.

The two are not opposites — they act on different cues and can be used
together.

### 3. DRR contrast

#### 3.1 Algorithm

For a freshly generated RIR, multiply the entire tail after the direct window
by a fixed gain:

```
tail_start = peak + round(w · fs / 1000),    w = direct_window_ms (default 2.5 ms)
h[n] ← h[n] · 10^(−s/20)    for all n ≥ tail_start
```

`s` is the DRR shift to apply, in dB.

#### 3.2 Why the shift is exactly s

Let the original direct energy be `E_d` and the tail energy `E_r`. Scaling the
tail amplitude by `10^(−s/20)` makes its energy `E_r · 10^(−s/10)`. The new
DRR:

```
DRR' = 10·log10( E_d / (E_r · 10^(−s/10)) )
     = 10·log10(E_d / E_r) + 10·log10(10^(s/10))
     = DRR + s
```

The shift equals `s` exactly, with no iteration or correction needed. This
clean result depends on one condition: **the tail boundary must coincide with
the DRR measurement boundary**. The implementation therefore defaults
`direct_window_ms` to 2.5 ms, matching `compute_drr_db`'s default window
(ch2 §5). If the two disagree, the scaled interval and the measured interval
differ and the shift no longer equals `s`.

#### 3.3 Why scale the tail rather than the direct path

Either would change DRR, but there is another peak normalisation downstream
(ch2 §3.1). Scaling the direct path changes `max|h|`, so after normalisation it
amounts to changing the tail's absolute level as well; scaling the tail leaves
the peak, and therefore the normalisation factor, untouched. Only the DRR ratio
survives to the model input, so the implementation picks the operation that
leaves the peak alone, keeping the effect single-purpose and predictable.

#### 3.4 The two modes

**random mode**: the shift direction follows the source role, applied with
probability `prob`.

```
foreground:                  s = +U(near_boost_db)
interferer / media / echo:   s = −U(far_cut_db)
any other role:              untouched
```

The effect is to push near-field DRR up and far-field DRR down, widening the
gap between them.

**deterministic mode**: the shift is a fixed function of the channel's realised
distance, ignoring role, drawing no probability, and consuming no RNG.

```
s = extra_db_per_decade · log10(d / pivot_m)
```

Deriving its effect: from §1.3 the natural relation is
`DRR ≈ −20·log10(d) + C`. With the shift added:

```
DRR'(d) = −20·log10(d) + C + extra · log10(d / pivot)
        = −(20 − extra)·log10(d) + C'
```

For negative `extra`, the magnitude of the gradient `|20 − extra|` grows —
**the whole pool's distance→DRR gradient steepens**, widening the near/far DRR
gap consistently at every distance. `pivot_m` is the pivot distance left
unaffected (`s = 0` at `d = pivot`).

#### 3.5 The design difference between the modes

The reason deterministic mode exists is recorded in the `init_drr_contrast`
comments: random mode's per-draw sampling gives the same distance a different
DRR shift on different rows, i.e. it **decouples** DRR from distance. The pool
loses its single distance→DRR mapping, and one DRR value comes to correspond to
an interval of distances rather than a single one.

If the goal is for the model to learn one clear distance→DRR correspondence,
the shift has to be a function of distance rather than an independent random
variable. That is the starting point for deterministic mode, and the reason it
deliberately consumes no RNG — the same channel gets the same shift under all
circumstances.

#### 3.6 When it is applied

Both modes are applied **before** the RIR enters the cache, immediately after
generation. The reason is in ch2 §6: the target re-fetches the same RIR by
`rir_id`, so applying the shift after caching would leave mixture and target
built on two different impulse responses, breaking the premise that the target
is the mixture's near-field component.

Metadata records `drr_contrast_shift_db` and the recomputed `drr_db`, letting
eval bucket rows by actual DRR.

### 4. Direct-arrival smear

#### 4.1 Purpose

To destroy the direct-window fine timing structure of §1.4, disabling any
readout that depends on that cue on a fraction of training rows. The design
intent is to force the model to look for alternative cues (spectral tilt, decay
shape), improving robustness in conditions where the cue is masked —
reverberation masks it inherently, because a reverberant near-field signal has
strong reflections immediately after the direct window.

#### 4.2 Algorithm

```
n = round(smear_ms · fs / 1000)
window = h[peak : peak+n]
E_orig = Σ window²

k = randn(n);  k ← k / ||k||₂            # random unit-energy kernel
window ← causal_conv(window, k)           # left zero-pad, causal

if |window[0]| < max|window|:              # restore first-sample dominance
    window[0] ← sign(window[0]) · max|window| · 1.001

window ← window · sqrt(E_orig / Σ window²) # restore energy
h[peak : peak+n] ← window
```

#### 4.3 Why a random unit-energy kernel

Convolving with a random sequence spreads energy that was concentrated in a few
samples across the whole window, destroying the timing relationships between
impulses. Normalising the kernel to unit L2 energy (`||k||₂ = 1`) makes the
convolution approximately energy-preserving (in the Parseval sense), reducing
how much the subsequent energy-restoration step has to correct.

**Causality**: only the left side is zero-padded before convolving, ensuring no
output appears in samples earlier than the direct arrival. A non-causal smear
would place energy before the direct sound arrives, which is physically
impossible and would affect every downstream offset computed from `argmax`.

#### 4.4 Three deliberate invariants

The design goal is to change **only** the timing structure, so three things must
stay fixed:

**(a) The peak position.** `wav_apply_rir` uses `argmax|h|` for two purposes:
the start of the truncation window, and the propagation-delay alignment offset
(ch2 §3.2–3.3). Move the peak and every downstream offset moves with it.
Smearing can leave a later sample larger than the first, hence the dominance
restoration step in §4.2.

The **order** of that step matters: it must happen before energy restoration.
Restoring energy first and then fixing the first sample would add energy back,
leaving the window's total energy different from the original.

**(b) Window energy.** After smearing, multiply by
`sqrt(E_orig / E_smeared)`. This guarantees the operation is not a level change
— the level cue has already been removed (§1.1), and introducing a level change
here would conflate two effects.

**(c) The late tail.** Samples after `smear_ms` are untouched, so RT60 and the
approximate DRR are unchanged and the operation acts strictly inside the direct
window.

#### 4.5 Application point and shipping state

Same position as DRR contrast (before the RIR enters the cache) for the same
reason: the target re-fetches the same RIR by `rir_id`, which is what makes the
`early` target the near-field component of the smeared `full` mixture.

This knob is **disabled by default and enabled by no recipe**. The design
motivation, the measurements to read before enabling it, and the open question
of whether a substitute cue is learnable are documented in the
`DirectSmearConfig` docstring and the probe README referenced in §1.7. The knob
exists in the codebase so that running that experiment takes a one-line config
change, rather than to pre-decide the matter for every recipe.

### 5. The distance_level mixing mode

#### 5.1 Algorithm

Reinstate the level cue removed in §1.1, using scene geometry, at the mixing
stage:

```
SIR = 20 · log10(d_itf / d_fg) + U(jitter_db)
```

`d_fg` is the foreground's realised distance, `d_itf` that of the **nearest**
interferer.

#### 5.2 Derivation and parameter meaning

From the inverse square law in §1.1, the amplitude ratio of two sources at the
microphone is `d_itf / d_fg`, which in dB is `20·log10(d_itf / d_fg)`. A
foreground at 0.5 m against an interferer at 3 m gives about +15.6 dB,
consistent with free-field point sources.

**Why the nearest interferer**: under the 1/r law the nearest interferer is the
loudest and therefore sets the effective SIR. Using the mean distance would let
several distant interferers pull the SIR up, underestimating the actual
interference.

**What jitter is for**: in reality level is not determined by distance alone —
mouth directivity, vocal effort, body orientation, and microphone polar pattern
all contribute. `jitter_db` supplies a perturbation range for these unmodelled
factors, preventing SIR and distance from forming a fully deterministic
relationship (which would let the model infer distance from SIR, a shortcut
that does not exist in the real world).

#### 5.3 Fallback behaviour

When geometry is unavailable (folder RIRs with no metadata, or a bank that
supplies no distance), `_distance_level_sir` returns `None` and the caller falls
back to the legacy hard-SIR draw. The row remains usable for training, just
without a level cue.

## Engineering

### Config mapping

| Knob | Schema | Where it lands |
|---|---|---|
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | `Augmentor._apply_drr_contrast`; both the bank and simulator paths, before the cache |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | `Augmentor._apply_direct_smear`; same |
| `augmentation_speech.mix_mode.modes[].distance_level` | `MixModeEntry` | `VoiceIsolationDataset._distance_level_sir` |

Schema-level validation constraints:

* `DrrContrastConfig`: `deterministic` mode requires `extra_db_per_decade`;
  in `random` mode the lower bounds of `near_boost_db` / `far_cut_db` may not
  be negative (the direction comes from the role, not the sign of the number).
* `DirectSmearConfig`: the lower bound of `smear_ms_range` must exceed 0 (0 ms
  is a no-op, a case already covered by `prob < 1`).

### Ordering and RNG

* The order of the two RIR knobs is fixed: **DRR contrast → direct smear →
  enter cache**.
* Both probability draws sit inside their short circuits, so a disabled knob
  consumes no RNG and older recipes regenerate bit-identically
  ([ch8](engineering_contract.md)).
* Deterministic DRR contrast consumes no RNG at all, which is part of the
  "same channel always gets the same shift" property.
* Direct smear's random kernel uses `torch.randn` and optionally accepts a
  `generator` for an independent stream (the pipeline currently uses the global
  stream).

### Pitfalls

* Folder RIRs carry no metadata, so deterministic DRR contrast and
  `distance_level` both silently skip or fall back. Check the RIR source first
  when debugging.
* Random mode only handles known roles (`foreground` / `interferer` / `media` /
  `echo`). Call paths that leave the role at its default `"source"` are
  unaffected — the non-source-level whole-mix reverb is one such case.
* Real-far rows ([ch7](scene_construction.md)) **skip mix_mode entirely**,
  including `distance_level`: the level ratio between simulated near-field
  speech and a real far-field recording is not physically meaningful, so those
  rows use hard SIR.
