# M5 controlled-room RIR measurement and inverse calibration contract

繁體中文版本：`rir_measurement_campaign.zh-TW.md`

Status: M5.1-M5.6 implementation complete; the controlled measured/listening/downstream empirical exit is not yet ready
Schema: `puresound.rir_measurement_campaign.v1`
Loss policy: `puresound.rir_calibration_loss.v1`

## 1. The problem this solves

M1-M4 (see [`rir_scene_v2.md`](rir_scene_v2.md), [`spatial_rir.md`](spatial_rir.md),
and [`rir_realism_algorithm.md`](rir_realism_algorithm.md)) can already
synthesize a usable RIR from room geometry, materials, environment, and
direct/early paths plus a multiband late field. M5's question is no longer
"can we produce reverberation at all" but:

> Given a small but fully traceable set of real-room measurements, can we
> invert the renderer's physical parameters so that unmeasured positions —
> and even unseen rooms — end up closer to reality than an uncalibrated
> model?

Formally, a measured RIR is written `h_meas[r, s, m](t)`, where `r` is a
physical room, `s` is a source pose, and `m` is a receiver. The renderer is

```text
h_hat(t) = F(theta_room, theta_mat, theta_dir, theta_late; g, s, m, e)
```

where `g` is geometry and `e` is temperature, humidity, and pressure. M5
looks for the parameters

```text
theta* = argmin_theta  sum_(s,m) in T_r   L(h_meas[r, s, m], F(theta))
```

and then evaluates *only* at positions and rooms that took no part in
fitting. If a measurement conflates loudspeaker, microphone, geometry error,
and clock delay together with the room's own response, optimization can
converge and still have simply mistaken device error for wall material.
That is why M5.1 first freezes **what data even qualifies to enter
inversion**, before any fitting code is trusted.

## 2. Why repeated exponential sine sweeps

Every source/receiver configuration is captured with at least two
exponential sine sweep (ESS) recordings. An ESS's instantaneous frequency
sweeps exponentially from `f1` to `f2`:

```text
x(t) = sin[ 2*pi*f1 * T/ln(f2/f1) * (exp(t*ln(f2/f1)/T) - 1) ]
```

The recording `y(t)` is deconvolved against the matching inverse filter
`x^-1(t)`:

```text
h_tilde(t) = y(t) * x^-1(t)
```

The value of this pipeline is that per-band energy is controllable, and a
device's harmonic distortion separates from the linear impulse response
after deconvolution. The method traces back to Farina's
[AES ESS paper](https://angelofarina.it/Public/Papers/134-AES00.PDF). This
project retains more than the final `rir.wav`; it requires, per capture:

- at least two raw sweep recordings;
- the inverse filter used to play the sweep;
- a background-noise recording of the same configuration;
- the deconvolution window, harmonic-separation, and latency-correction
  settings;
- the final deconvolved RIR.

Repeating the measurement is not just so the two takes can be averaged
away. It lets the pipeline estimate repeatability, the noise floor,
time-varying interference, and reject data where "one recording happened to
look reasonable" — all of which `SweepCapture` in
`puresound.audio.rir.calibration.measured_campaign` enforces structurally:
its constructor requires `method == "exponential_sine_sweep"`, rejects
fewer than two `raw_recordings`, and requires the inverse-filter and
noise-recording assets to be present alongside the deconvolved RIR, all at
one shared sample rate.

## 3. Physical evidence that must be retained

### 3.1 The physical room and its coordinate frame

Every `MeasuredRoom` must carry a stable `room_id`, a room type, a
right-handed coordinate-frame description, a `geometry_uncertainty_m`, and
at least one of:

- measurable `dimensions_m`; or
- a coarse or detailed mesh asset with a SHA-256.

The same physical room must never be renamed just because a different set
of source positions was used for it, or the room-disjoint test (§4) leaks.

### 3.2 Source and receiver

`CalibratedTransducer` records the source's and receiver's manufacturer,
model, serial number, reference axis, calibration date, calibration-response
asset, and provenance *separately*. The room's transfer function and the
transducer's own response must not be absorbed into one combined synthetic
EQ.

Every pose (`MeasurementPose`) carries:

- `position_m = [x, y, z]`;
- `orientation_ypr_deg = [yaw, pitch, roll]`;
- a one-standard-deviation uncertainty for both position and orientation
  (`position_std_m`, `orientation_std_deg`).

Distance alone, without an absolute position and orientation, is not enough
to fit wall reflections, directivity, or spatial coherence.

### 3.3 Environment and synchronization

Every capture retains temperature, relative humidity, and pressure, because
they change sound speed and air absorption (`MeasuredRIRRecord` validates
`-50 <= temperature_c <= 60`, `0 <= relative_humidity_percent <= 100`, and a
positive `pressure_pa`). A multi-receiver capture must be
sample-synchronized: every WAV channel must represent the *same* source
excitation arriving at a different receiver, not several source positions
packed together as channels. The schema enforces this directly — if a
record's `receiver_ids` has more than one entry, `synchronized_receivers`
must be `True` or the record's constructor raises. A record that packs
multiple source positions as channels can still support mono acoustic-metric
comparison, but it cannot be used to estimate inter-channel coherence or
IACC.

### 3.4 Immutable asset identity

Every retained asset uses a campaign-relative safe path (`MeasurementAsset`
rejects absolute paths and any `..` path component) together with a SHA-256
digest (validated as exactly 64 hexadecimal characters). `audit_measurement_campaign()`
checks that every referenced file exists, that every hash matches the file
on disk, and that the same path is never claimed by two conflicting hashes —
which makes "the metadata did not change but the WAV was silently
re-rendered" a detectable, named error
(`asset_paths_have_no_conflicting_hashes`, `all_assets_exist`,
`all_retained_asset_hashes_match`).

## 4. Splits are room-disjoint, not item-disjoint

The schema allows only three room splits — `train`, `validation`, `test`
(`ROOM_SPLITS`) — and every `room_id` is assigned to exactly one of them;
`RIRMeasurementCampaign` raises if `room_splits` does not cover the room set
exactly, or assigns any label outside that set. For a `train` room, the
positions that did not participate in fitting must still be withheld to
form a **position holdout**; a `validation` or `test` room does not
participate in parameter fitting, loss weighting, or early stopping *at
all*.

`deterministic_position_assignments()` in
`puresound.audio.rir.calibration.measured_runner` implements this exactly:
every `train` room needs at least two records (it raises otherwise), and a
`holdout_fraction` of them — 0.25 by default, constrained to `(0, 0.5)` — is
withheld as `position_holdout`. Which specific records are withheld is
chosen deterministically, by sorting on the SHA-256 digest of
`f"{campaign_id}:{room_id}:{measurement_id}"`, not by redrawing a random
split on every run. Every record of a `validation` or `test` room instead
gets the label `room_<split>` (for example `room_test`), signaling that the
*entire room*, not just some of its positions, is held out.

The minimum data volume is currently set at 12 records per room, with 12-30
recommended configurations. This is not a magic acoustic constant; it is
the first operational gate. With too little data, many combinations of
material, scattering, directivity, and late-field parameters can produce
similar-looking RIRs, and the inverse problem is not identifiable. M5.2
tests, with synthetic recovery, which parameters can actually be recovered
before a real campaign commits to more positions, orientations, or receiver
spacing (§7.2). `audit_measurement_campaign(..., minimum_records_per_room=12)`
checks `minimum_records_per_room_met` and, separately,
`minimum_unique_configurations_per_room_met` — the two differ whenever
repeated ESS captures reuse the same source/receiver pose, so a room's raw
record count is not automatically its number of *distinct* configurations.
It also requires `at_least_one_synchronized_spatial_record` across the whole
campaign, since M5.4 (§7.3) has nothing to calibrate against otherwise.

## 5. Multi-objective calibration loss

A single waveform L1/L2 distance is dominated by tiny time shifts; comparing
only RT60 ignores early reflections, frequency coloration, and spatial
structure. The reference loss is therefore

```text
L = w_stft * L_stft + w_edc * L_edc + w_arr * L_arr + w_oct * L_oct
    + w_sp * L_sp + w_causal * R_causal + w_decay * R_decay
```

implemented as `analyze_rir_calibration_loss()` in
`puresound.audio.rir.calibration.loss`. Every term is reported individually —
never collapsed to only a total scalar (`CalibrationLossReport.terms` and
`.diagnostics`) — and `CalibrationLossWeights`'s defaults are
`multiresolution_stft=1.0`, `energy_decay=1.0`, `arrival_timing=1.0`,
`octave_acoustics=1.0`, `spatial_coherence=1.0`, `decay_regularization=1.0`,
and, deliberately an order of magnitude larger than the rest,
`causality=10.0`. The implementation is explicitly labeled
`"numpy_scipy_reference_not_autograd"`: it is a metric oracle used to score
and compare fits, not a differentiable training loss.

### 5.1 Multiresolution STFT

Spectral convergence and log-magnitude L1 are compared at several FFT sizes
(256/512/1024 by default):

```text
L_SC  = norm(|S| - |M|) / norm(|M|)
L_log = mean( |log(|S| + eps) - log(|M| + eps)| )
```

Short windows are more sensitive to early/transient structure; long windows
give finer frequency resolution. This kind of multiresolution spectral
objective is also common in neural waveform generators, e.g.
[Parallel WaveGAN](https://arxiv.org/abs/1910.11480).

### 5.2 Direct-relative energy-decay curve

The measured and synthetic direct arrivals are located separately, the two
signals are aligned from their own direct sample, and a Schroeder backward
integration is compared:

```text
E(t) = sum_{tau=t}^{T} h(tau)^2
D(t) = 10 * log10( E(t) / E(0) )
```

The comparison is the decay curve's RMSE (normalized by an explicit
`floor_db`, -80 dB by default), not a single linear slope, so a two-slope
decay, excess early energy, or tail flattening cannot hide behind one RT60
number.

### 5.3 Arrival timing

Each channel's direct-arrival sample is compared, converted to milliseconds,
and normalized by an explicit tolerance (1.0 ms by default). This term keeps
absolute time-of-flight information: it is computed from the raw
direct-sample difference, not from the direct-relative alignment that the
EDC and octave terms use, so it does not let a geometric timing error cancel
itself out.

### 5.4 Octave acoustic metrics

Direct-relative band energy and qualified T20 are compared in the valid
125 Hz-4 kHz octave bands (default centers 125/250/500/1000/2000/4000 Hz,
filtered to strictly below Nyquist). A channel's T20 only counts toward the
decay term when *both* the measured and synthetic bands yield a qualified
decay-time estimate; a band without enough decay range is recorded as
unqualified rather than silently defaulted to zero or a placeholder RT60,
and `octave_acoustic_distance()` reports `qualified_t20_channels` alongside
the energy error so the two failure modes stay distinguishable.

### 5.5 Spatial coherence

Only evaluable with synchronized multi-receiver capture (§3.3). In a late
window (80 ms after the earlier of the pair's direct arrivals, by default),
every receiver pair's normalized complex cross-spectrum is compared:

```text
Gamma_ij(f) = S_ij(f) / sqrt( S_ii(f) * S_jj(f) )
```

A mono bank, or one where the channels are really different source
positions rather than simultaneous receivers, reports
`"evaluable": false` with an explicit `"reason"` (for example
`"fewer_than_two_synchronized_receivers"`) instead of a spuriously perfect
spatial loss of `0`.

### 5.6 Physical regularization

`causality_penalty()` measures the synthetic energy fraction that falls
*before* the geometric or measured physical-arrival bound
(`prearrival_energy_fraction_by_channel`); `decay_growth_penalty()` measures
whether late-window energy keeps growing anomalously, penalizing sustained
growth beyond an allowed rate (1.0 dB per hop by default, measured on 20 ms
windows with a 10 ms hop starting 80 ms after the direct sample). Both exist
to rule out solutions that move every other term closer to the target while
physically manufacturing a pre-echo or an unstable tail — which is also why
`causality`'s default weight (10.0) is set an order of magnitude above every
other term's: a pre-arrival energy leak is penalized far more heavily than
any other single mismatch.

## 6. What M5.1 has already completed

(This section originally recorded specific results; they now live in the
[`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) appendix.)

## 7. Reference implementations: identifiability, the runner, spatial calibration, and the residual

The reference loss in §5 is the shared evaluation oracle used everywhere
below, but §5 alone does not answer three practical questions: which
parameters are safe to fit from limited real data, what stops a fit from
starting on evidence that is not actually ready, and how spatial evidence or
a learned residual are allowed to be used at all. `puresound.audio.rir.calibration`
answers each with its own reference implementation, on the module's own
terms: *"No synthetic result can promote a measured or production exit."*

### 7.1 Fail-closed measured-room fitting (M5.3)

`run_measured_campaign_fit()` in
`puresound.audio.rir.calibration.measured_runner` is the M5.3 reference
runner. Before it fits anything, it re-runs `audit_measurement_campaign()`
(§3-4) and raises `CampaignNotReadyError` — carrying the full audit payload
as `.audit` — whenever the campaign is not ready. There is no code path
that lets a fit proceed on an incomplete or malformed campaign; this is the
"fail-closed" property the milestone name refers to. Given a ready
campaign, the runner uses `deterministic_position_assignments()` (§4) to
split each `train` room into fit and holdout positions. Per train room, it
first profiles the discrete M4 mixing-time topology
(`fit_m4_parameter_profile()`, §7.2) and then refines grouped per-boundary
reflection gains from that starting point (`fit_grouped_path_gains()`,
§7.2); each room's fit is scored separately on its own fit positions and its
own held-out positions, and the *median* of every train room's fitted
parameters becomes the single `population_parameters_for_unseen_rooms` used
to score entire held-out validation and test rooms, which never contributed
to any fit. The runner's own exit block always reports
`"production_enabled": false` — passing its checks means the M5.3 mechanics
ran correctly on the supplied evidence, not that production use is
authorized.

### 7.2 Grouped material/path identifiability (M5.2d)

Before a measured-room fit, `puresound.audio.rir.calibration.inverse_m4` and
`inverse_m5` establish how PathEvent/M4 parameters map onto measured data,
and which of them can actually be told apart locally.
`inverse_m4.fit_m4_parameter_profile()` treats the FDN mixing time as a
discrete *outer* profile — changing it changes the prime-delay topology, not
just a continuous coefficient — and fits coherent-path gain plus per-octave
RT60 as a bounded, continuous *inner* solve (`scipy.optimize.least_squares`)
at each candidate mixing time. "Converged" is decided not by the optimizer's
own success flag but by restarting the solve from its own answer and
checking whether the restart can still lower the cost by more than a 0.1%
relative tolerance (`M4_PROFILE_CONVERGENCE_POLICY`) — a deliberate response
to an objective that is locally rough enough for the trust region to
collapse before `scipy`'s own tolerances trip, so `success=False` does not
automatically mean "not yet converged" and `success=True` does not
automatically mean "actually at a minimum."

`inverse_m5.analyze_local_identifiability()` takes a Jacobian and a
priority-ordered list of parameter names, normalizes each column, and
greedily accepts parameters in priority order — rejecting a parameter and
recording *why* whenever it has (a) zero local sensitivity
(`zero_local_sensitivity`), (b) an absolute correlation at or above 0.995
with an already-accepted parameter (`redundant_with:<name>`), or (c) would
push the accepted subset's condition number above 100
(`subset_condition_exceeds:100`). `fit_grouped_path_gains()` applies this to
a per-boundary-group reflection-gain vector — one adjustment per named
surface group (for example, one per shoebox wall) rather than one global
scalar — starting from the M4 profile fit above. This is what "grouped
material/path identifiability" refers to: not a single rank number, but an
explicit, named accepted/rejected partition of which physical groups the
data can actually support independently.

### 7.3 Synchronized spatial calibration (M5.4)

`inverse_m5.select_spatial_calibration_candidate()` is a discrete profile
selector, not a continuous optimizer. Given a mapping of candidate
renderings — for example different scattering, directivity, or late-field
configurations — and one synchronized multi-receiver measured reference, it
scores every candidate with the full §5 calibration loss, using a
spatial-aware weighting that raises `spatial_coherence` to 2.0 and keeps
`causality` at 10.0 while easing the waveform/decay/octave terms, and
returns the candidate with the lowest total. It requires at least two
receivers and raises if the measured reference does not actually yield an
evaluable spatial-coherence term (§5.5): §5.5's `evaluable=false` escape
hatch is a legitimate loss-report state everywhere else in this contract,
but it is a hard error here, specifically because M5.4 exists only to make
use of synchronized evidence.

### 7.4 Constrained residual (M5.5)

`fit_causal_decay_residual()` in
`puresound.audio.rir.calibration.residual` learns **one shared correction**
on top of the fitted physical renderer, not a per-room free-form fit. For
each training position, it takes the direct-relative difference between the
measured target and the physical (M4/grouped) rendering, normalizes it by
that position's own physical-tail energy norm, and takes the **median**
across positions — a robust central tendency, so one anomalous position
cannot dominate the shared correction. That raw template is then constrained
twice before it is accepted:

- `_constrain_decay()` caps its block-wise RMS decay rate so the residual
  can never sustain reverberation beyond an explicit `maximum_rt60_s`
  (0.8 s by default);
- its total energy is then capped to at most
  `maximum_residual_to_physical_energy_ratio` (0.25 by default) of the
  physical tail's own energy.

`CausalDecayResidualModel.residual_for()` applies the fitted template
starting exactly at each new observation's own direct sample, so the
correction stays causal and is rescaled per position even though its shape
is shared. `evaluate_residual_ablation()` reports physical-only,
residual-only, and combined results through the same §5 loss oracle side by
side, so a residual's benefit is never reported without also showing what
the physical renderer alone, and the residual alone, would have scored.

## 8. Reproducing M5.1-M5.6

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py
# schema round-trip, campaign audit, and the ESS/pose/environment contract (§2-4)

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_synthetic_recovery.py
# M5.2: can the lightweight approximate renderer's own parameters be
# recovered from clean synthetic targets

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_robust_recovery.py
# M5.2b: recovery under noise, known gain/latency error, and unknown
# early/late model mismatch

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_m4_parameter_mapping.py
# M5.2c: the profile fit from §7.2, against the actual M4 PathEvent/FDN coupling

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py
# M5.2d: analyze_local_identifiability() and fit_grouped_path_gains() from §7.2

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py
# M5.3: the fail-closed runner from §7.1

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py
# M5.4: select_spatial_calibration_candidate() from §7.3

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py
# M5.5: fit_causal_decay_residual() and its ablation from §7.4

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py
# M5.6: aggregates every gate above into the dual exit reported below

.venv/bin/pytest -q \
  test/test_rir_measurement_campaign.py \
  test/test_rir_calibration.py \
  test/test_rir_inverse_calibration.py \
  test/test_rir_m4_inverse_calibration.py \
  test/test_rir_m5_pipeline.py \
  test/test_rir_measured_calibration.py \
  test/test_rir_constrained_residual.py \
  test/test_m5_measurement_contract_validator.py \
  test/test_m5_synthetic_recovery_validator.py \
  test/test_m5_robust_recovery_validator.py \
  test/test_m5_m4_parameter_mapping_validator.py \
  test/test_m5_completion_validators.py
```

Expected output:

```text
M5.1 implementation exit: PASS
controlled measurement readiness: OPEN
M5.2 synthetic recovery: PASS
M5.2b robust synthetic recovery: PASS
M5.2c actual M4 parameter mapping: PASS
M5.2d grouped material/path identifiability: PASS
M5.3 runner implementation: PASS
M5.3 measured inverse fit: BLOCKED ON CONTROLLED CAMPAIGN
M5.4 synchronized spatial calibration implementation: PASS
M5.5 constrained residual implementation: PASS
M5 implementation exit: PASS
M5 empirical/production exit: OPEN
```

The two are not contradictory: the former means the code, schema, and loss
are ready; the latter means a campaign meeting the contract in §2-4 has not
yet been acquired, so M5.3's measured-room fit cannot yet be claimed
complete.
