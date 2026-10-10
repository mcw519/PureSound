# Direct/early/later attribution — `puresound.audio.rir.metrics.attribution`

繁體中文版本：[rir_attribution.zh-TW.md](rir_attribution.zh-TW.md)

Splits a rendered response into direct, early-reflection and later-reflection
components that sum back to the original exactly. It audits how a renderer
allocates energy between those parts; it is not a perceptual metric. Schema
string: `ATTRIBUTION_SCHEMA_VERSION = "puresound.rir_attribution.v1"`.

## Decomposition

```python
decompose_direct_early_later(
    full_output,            # 1-D response
    direct_anchor_output,   # 1-D direct-path component, same length
    *, sample_rate_hz, split_center_s, transition_width_s,
) -> dict  # direct, early_reflections, later_reflections, early_cumulative, full
```

```text
r      = full − direct
early  = r · m_early(t)
later  = r · m_later(t),        m_later = 1 − m_early
early_cumulative = direct + early
```

**The direct component is supplied, not windowed.** A fixed time window fails
when the source pulse is wider than the delay between the direct path and the
first reflection: part of the direct pulse would be counted as a reflection.
The caller therefore renders the direct path on its own — for PathEvents, the
single `direct` event (`partition_path_events_by_arrival` separates direct,
early and later events by arrival time) — and passes it as
`direct_anchor_output`.

**The decomposition is exact by construction.** The masks are complementary
sample by sample, so `direct + early + later == full` up to floating-point
rounding, whatever the crossfade shape.

## Masks

```python
complementary_early_late_masks(
    *, num_samples, sample_rate_hz, split_center_s, transition_width_s,
) -> (early, later)
```

`early` is 1 before `split_center_s − transition_width_s / 2`, falls as a raised
cosine `0.5 + 0.5 cos(π p)` across the transition (`p` from 0 to 1), and is 0
afterwards; `later = 1 − early`.

## Checking the reconstruction

```python
reconstruction_error(components) -> {"maximum_absolute_error": float, "nrmse": float}
```

Requires `direct`, `early_reflections`, `later_reflections` and `full` (exactly
what `decompose_direct_early_later` returns), sums the three parts and reports
the maximum absolute error and the error norm divided by `‖full‖₂`. It turns
the exactness claim into a number a validator can assert.

## Use

The PathEvent renderer validation uses it to audit the direct/early/later split
of the coherent high band; `test/rir/test_rir_attribution.py` pins the
reconstruction.
