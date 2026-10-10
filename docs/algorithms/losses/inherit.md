# puresound.nnet.loss.inherit

繁體中文版本：[inherit.zh-TW.md](inherit.zh-TW.md)

Anchor-inheritance hinge. It guards against a model that treats whoever spoke
last as the foreground and hands the next near talker the gain it chose for the
previous one: over the user's onset window, the gain achieved on the user must
sit at least `margin_db` above the gain the model applied to what was there
before the user arrived.

## Class: `AnchorInheritanceLoss`

Per frame, on the `EnergyVADLabeler` grid (`frame_length` 400, `hop_length`
160, i.e. 100 fps at 16 kHz — the grid `vad_target` is on), with `E` the
enhanced frame, `R` the target (the user) and `N = batch[reference_key]`
(`consistency_noise = noisy − target`):

```
a_t = <E_t, R_t> / max(<R_t, R_t>, eps_ratio)     # achieved gain on the user
q_t = <E_t, N_t> / max(<N_t, N_t>, eps_ratio)     # achieved gain on what was there before
pos(z) = softplus(β z) / β                         # β = softplus_beta
P_t = clamp(10 log10(pos(a_t)^2 + eps_db), floor_db, 0)
Q_t = same, on q_t
```

Two frame sets per row:

- `O` — the first `onset_frames` active frames of `vad_target` from the user's
  first active frame.
- `B` — frames before the onset with `vad_target == 0` whose
  `consistency_noise` energy is within `pre_energy_window_db` of the prefix's
  own peak frame.

```
Pbar_on  = Σ_O w_t P_t / Σ_O w_t                      w_t  = <R_t, R_t>
Qbar_pre = detach( Σ_B w'_t Q_t / Σ_B w'_t )          w'_t = <N_t, N_t>
L        = mean over eligible rows of relu(margin_db − (Pbar_on − Qbar_pre))
```

A row is eligible when `n_interferers ≥ 1`, the user is active, the onset is at
least `min_onset_frame` into the row, at least `onset_frames` active frames
follow it, `B` is not empty, and `background_vad_target` shows at least
`min_pre_interferer_frames` interferer frames strictly before the onset. A batch
with no eligible row returns the graph-carrying zero `enhanced.sum() * 0.0`.

### Constructor

```python
AnchorInheritanceLoss(
    margin_db: float = 10.0,                   # required onset-vs-prefix gain contrast
    frame_length: int = 400,                   # the EnergyVADLabeler grid; change only in tests
    hop_length: int = 160,
    onset_frames: int = 50,                    # |O|; 0.5 s of speech; > 0
    min_onset_frame: int = 100,                # onset at least 1 s into the row
    min_pre_interferer_frames: int = 100,      # 1 s of interferer speech before the onset
    fallback_pre_interferer_frames: int = 50,  # looser rule, reported by row_scores only;
                                               #   must not exceed min_pre_interferer_frames
    pre_energy_window_db: float = 25.0,        # admission window for B
    floor_db: float = -30.0,                   # floor on P and Q; must be < 0
    softplus_beta: float = 16.0,               # sharpness of pos()
    eps_ratio: float = 1e-10,
    eps_db: float = 1e-12,
    reference_key: str = "consistency_noise",  # batch key holding noisy - target
)
```

### Inputs

`required_inputs = ("enhanced", "target", "batch", "vad_target")`.
`enhanced` and `target` are `[B, T]` (a `[B, C, T]` input is averaged over
channels, a `[T]` input gets a batch axis). `vad_target` needs `vad_label` in
the recipe; the energy backend is enough. Batch keys: `reference_key`
(required), `n_interferers` (required, `[B]`) and `background_vad_target`
(optional; absent means no row carries background speech, so no row is
eligible). A missing required input raises `ValueError` / `KeyError`. Scoring
runs in float32.

### Methods

- `row_scores(enhanced, target, batch, vad_target) -> Dict[str, Tensor]` — the
  same arithmetic with nothing reduced, every value `[B]`: `hinge`, `pbar_on`,
  `qbar_pre`, `contrast`, `eligible`, `eligible_fallback`, `onset_frame`,
  `pre_interferer_frames`, `n_onset_frames`, `n_pre_frames`,
  `n_active_after_onset`, `n_interferers`, `n_frames`. Values are meaningful
  only where `eligible` (or `eligible_fallback`) is true; mask before
  aggregating. Preflight checks and validation monitors read this, so the
  monitored number and the trained number share one implementation.
- `floored_gain_db(ratio)` — the floored envelope
  `clamp(10 log10(pos(ratio)^2 + eps_db), floor_db, 0)`, public so its values
  can be pinned by a test.

### Config usage

No shipped recipe enables this loss.

```yaml
vad_label:
  used: True
  backend: energy
  args: {frame_length: 400, hop_length: 160}
loss_func:
  - type: AnchorInheritanceLoss
    weighted: 0.25
    args: {margin_db: 10.0}
```

## Design notes

- **Only a difference.** There is no absolute dB threshold: an absolute
  calibration does not carry across capture chains or checkpoints, a relative
  one does.
- **Detached prefix term.** The hinge cannot be met by suppressing the
  preceding interferer less, which would trade far-field suppression for onset
  keep.
- **Onset window, not row mean.** A 1 s event barely moves a row mean, so a
  row-mean form carries almost no gradient.
- **The floor bounds the statistic.** Without it the pooled hinge is dominated
  by a few onsets whose output is anti-correlated with the reference.
  `softplus` alone does not remove the dead zone for negative ratios; the clamp
  zeroes the gradient from `a ≤ −0.026` down (at the defaults). Summarise
  per-row values with a median.
- **`B` needs energy.** A noise-only or silent prefix would make the hinge
  trivially satisfiable, hence the `pre_energy_window_db` admission rule.
- **Which reference.** `consistency_noise` is built from the final,
  post-speed-perturbation waveforms, and during the pre-onset frames the user is
  silent by construction, so it is the interferer plus noise. The pre-speed
  `background_vad_target` is used only for the row-level 1 s count, which a few
  percent of timing error cannot flip.
- **No parameters.** The loss changes neither the checkpoint layout nor the
  streaming export.
