# Direct/early/later attribution — `puresound.audio.rir.metrics.attribution`

繁體中文版本：[`rir_attribution.zh-TW.md`](rir_attribution.zh-TW.md)

Splits a rendered response into direct, early-reflection, and later-reflection
components that sum back to the original exactly, for auditing a renderer's
own internal energy allocation (not a perceptual metric). Policy string:
`ATTRIBUTION_SCHEMA_VERSION = "puresound.rir_attribution.v1"`.

## Why the direct component is supplied, not windowed

`decompose_direct_early_later(full_output, direct_anchor_output, *,
sample_rate_hz, split_center_s, transition_width_s)` takes the direct-path
component as an explicit second array rather than inferring it from a fixed
time window. A time window breaks when the source pulse is wider than the
direct-to-first-reflection delay — the window would clip part of the direct
pulse into what should be an early reflection. Supplying `direct_anchor_output`
(typically a free-field or anechoic render of the same source through the same
path) sidesteps that failure mode entirely.

The residual `full - direct` is then split by a complementary early/later mask
pair, so the decomposition is **exactly reconstructive by construction** —
`direct + early_reflections + later_reflections == full` up to floating-point
rounding, never approximately.

Returns a dict: `direct`, `early_reflections`, `later_reflections`,
`early_cumulative` (`direct + early_reflections`), `full`.

## The complementary mask

```python
complementary_early_late_masks(
    *, num_samples, sample_rate_hz, split_center_s, transition_width_s,
) -> (early: np.ndarray, later: np.ndarray)
```

A raised-cosine crossfade: `early` is `1.0` before the transition window,
falls smoothly to `0.0` across `transition_width_s` centered on
`split_center_s`, and is `0.0` after. `later = 1.0 - early` by construction, so
`early + later == 1.0` sample-for-sample — this identity, not the shape of the
crossfade, is what makes the decomposition exact.

## Verifying the reconstruction

```python
reconstruction_error(components: Mapping[str, np.ndarray]) -> dict
# {"maximum_absolute_error": float, "nrmse": float}
```

Requires `direct`, `early_reflections`, `later_reflections`, `full` in
`components` (exactly what `decompose_direct_early_later` returns); sums the
three parts and reports both the absolute and normalized (by `full`'s L2 norm)
reconstruction error against `full`. Used to assert the "exactly
reconstructive" claim numerically rather than just by construction.

## Wiring

Used by
`egs/rir_generation/phases/m3_wave_path/scripts/validate_direct_early_later_attribution.py`
to audit the M3 PathEvent renderer's own direct/early/later split — see
[`rir_realism_algorithm.md`](rir_realism_algorithm.md) §4.2 for how the M3
backend produces the `direct_anchor_output` this module consumes.
