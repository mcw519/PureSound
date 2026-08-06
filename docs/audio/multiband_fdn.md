# Multiband FDN — `puresound.audio.rir.render.multiband_fdn`

繁體中文版本：[`multiband_fdn.zh-TW.md`](multiband_fdn.zh-TW.md)

Deterministic passive feedback delay network with per-octave RT60 control.
The late-field engine behind the M4 backend; design map in
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §4.3.

## Design contract

- `design_multiband_fdn(...)` → `MultibandFDNDesign`;
  `render_multiband_fdn(...)` / `render_multiband_fdn_impulse(...)` →
  `MultibandFDNRender`. Policy string: `MULTIBAND_FDN_POLICY`.
- Feedback matrix: `randomized_hadamard_matrix` (orthogonal → passive loop).
- Delays: `select_prime_delay_lengths` (mutually prime, density without
  periodicity).
- Per-band decay: `delay_proportional_loop_gains` realizes each octave's
  target RT60 exactly for its loop length.
- Filterbank: cascaded binary split, endpoint-complete — the top band is a
  true highpass to Nyquist, so late energy has no spectral hole below fs/2
  (`fdn_filterbank_power_response` verifies flatness).
- Everything is seeded; same inputs render byte-identical tails.

## Diagnostics

`analyze_fdn_coloration` reports modal coloration of a rendered tail;
`puresound.audio.rir.metrics.analyze_multiband_late_field` checks a tail
against its per-octave targets.

## Caveat

The FDN realizes whatever target it is given. Target quality is the caller's
responsibility — see `RIR_EXP_LOG.md` §6.6 for the measured limits of
Sabine-derived targets.
