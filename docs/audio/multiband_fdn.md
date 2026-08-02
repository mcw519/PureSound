# `audio.multiband_fdn`

`puresound.audio.multiband_fdn` is the isolated M4.3 dense late-field core.
It designs and renders a deterministic mono multiband feedback delay network
(FDN), then validates it against the measured M4.2 octave-band target report.
M4.4 now couples this core to PathEvent early response through the separate
opt-in [`audio.rir_late_coupling`](rir_late_coupling.md) module.

## Design contract

The reference design uses parallel octave-band FDN branches. Branches share
16 distinct prime delay lengths and an orthogonal feedback matrix but have
independent states, output taps, and delay-proportional attenuation.

For delay length `d_i`, sample rate `fs`, and octave target `T60_b`, the
pressure gain per delay traversal is

```text
g[b, i] = 10 ** (-3 * d_i / (fs * T60_b))
```

The feedback operator for branch `b` is `U @ diag(g[b])`. `U` is a normalized,
signed, and permuted Hadamard matrix, so `U.T @ U = I` and
`||U @ diag(g[b])||2 = max(g[b]) < 1`.

In this module, *passive* means that the zero-input internal feedback update is
strictly contractive and cannot create runaway state energy. It does not mean
that the complete input/output transfer has energy gain at most one under all
tap normalizations; output taps observe state without removing energy from the
delay buffers.

The default delay interval is tied to the requested mixing time:
`0.125 * mixing_time` through `0.75 * mixing_time`. Nearby distinct primes are
selected deterministically. Prime delays reduce short common periods but do
not, by themselves, guarantee an uncolored tail.

## Public API

| Symbol | Purpose |
|--------|---------|
| `design_multiband_fdn(...)` | Build a validated, deterministic design from octave RT60 targets. |
| `render_multiband_fdn(...)` | Process a one-dimensional excitation. |
| `render_multiband_fdn_impulse(...)` | Render an isolated FDN impulse response. |
| `select_prime_delay_lengths(...)` | Select seeded, distinct prime delays. |
| `randomized_hadamard_matrix(...)` | Create a seeded signed/permuted orthogonal matrix. |
| `delay_proportional_loop_gains(...)` | Compute exact per-traversal pressure gains. |
| `analyze_fdn_coloration(...)` | Report within-octave spectral flatness and p95/median ripple diagnostics. |

Example:

```python
from puresound.audio.multiband_fdn import (
    design_multiband_fdn,
    render_multiband_fdn_impulse,
)

design = design_multiband_fdn(
    16_000,
    {
        500.0: 0.548,
        1_000.0: 0.463,
        2_000.0: 0.472,
        4_000.0: 0.474,
    },
    target_mixing_time_s=0.024,
    delay_line_count=16,
    seed=20260731,
)
render = render_multiband_fdn_impulse(design, duration_s=1.5)
rir = render.rir
```

The exact coefficient contract can be serialized with
`design.to_dict(include_coefficients=True)`. The default report omits full
matrices but retains the selected delays, orthogonality error, band targets,
loop-gain bounds, feedback norms, and band-weight energy.

## Validation

Run the formal M4.3 validator:

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_fdn.py \
  --target-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json \
  --output-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_fdn_report.json \
  --output-wav egs/rir_generation/exp/rir_realism/m4/rir_m4_fdn_core/rir_m4_fdn_core.wav
```

The validator reads measured median T20 targets and measured p10–p90 mixing
time/late normalized echo-density envelopes from the M4.2 report. It gates:

- distinct prime delays;
- orthogonal feedback and strict internal contraction;
- unit-energy band weights;
- exact same-seed rendering;
- finite output, causal delayed onset, and end-of-render decay;
- T20 relative error plus measured mixing-time and late-density envelopes.

The M4.3 qualified bands are 500 Hz through 4 kHz. At 16 kHz and seed
`20260731`, all 12 qualified acoustic checks and all 8 structural checks pass.
The 125/250 Hz branches remain diagnostic because the full hybrid low/modal
branch and crossover are not present in this isolated render.

Spectral coloration metrics are diagnostics, not a perceptual exit. M4.4 has
rerun causality, C50, early timing, decay, and density on the coupled hybrid;
controlled listening remains open, while M4.5 owns multi-receiver spatial
output.
