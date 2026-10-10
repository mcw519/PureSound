# RIR measurement campaigns and calibration

繁體中文版本：[rir_measurement_campaign.zh-TW.md](rir_measurement_campaign.zh-TW.md)

`puresound.audio.rir.calibration` fits the physical parameters of the RIR
generator to measured rooms. Everything here is built to avoid confusing room
behaviour with transducer response, geometry error, clock delay or data
leakage: the campaign contract says what a measurement must retain, the audit
refuses incomplete campaigns, the splits keep rooms disjoint, and the loss
reports every term separately.

| Module | Contents |
|---|---|
| `measured_campaign.py` | campaign schema `puresound.rir_measurement_campaign.v1`, `audit_measurement_campaign` |
| `measured_runner.py` | `deterministic_position_assignments`, `run_measured_campaign_fit` |
| `loss.py` | `analyze_rir_calibration_loss`, policy `puresound.rir_calibration_loss.v1` |
| `inverse_m4.py`, `inverse_m5.py` | the parameter-profile and grouped-path-gain inverse fits, local identifiability |
| `synthetic_recovery.py` | parameter recovery on synthetic fixtures with controlled perturbations |
| `residual.py` | the constrained causal residual |

## Campaign contract

`RIRMeasurementCampaign` holds `campaign_id`, `rooms`, `transducers`,
`records`, `room_splits` (every room assigned exactly once to `train`,
`validation` or `test`) and `provenance`. Every JSON mapping must be strict
(no NaN/Inf).

**Assets.** A `MeasurementAsset` is a campaign-relative path (no absolute
path, no `..`), a 64-hex SHA-256, a media type and, for audio, sample rate,
channel count and frame count together.

**Captures.** A `SweepCapture` is an exponential sine sweep (the only accepted
method) with start/end frequency (end below Nyquist), duration, fade,
silence, playback level, and these retained assets, all at one sample rate:

- at least two raw sweep recordings, so repeatability can be estimated and
  unstable takes rejected, not merely averaged;
- the inverse filter used for deconvolution;
- a background-noise recording;
- the deconvolved RIR (the calibrated, latency-corrected linear RIR the fit
  consumes);
- `latency_correction_samples` and `deconvolution_config`.

**Rooms.** A `MeasuredRoom` has a stable `room_id` (the same physical room is
never renamed across sessions or positions), a `room_type`, a documented
`coordinate_frame`, `geometry_uncertainty_m`, provenance, and either
`dimensions_m` or a retained `mesh_asset`.

**Transducers.** A `CalibratedTransducer` (`kind` source or receiver) keeps
manufacturer, model, serial number, reference axis, calibration date, a
retained calibration response and provenance separately, so transducer
response is never folded into room parameters.

**Records.** A `MeasuredRIRRecord` is one source pose observed by one
receiver array: measurement, room, session and source ids, timestamp,
source and receiver poses, the capture, temperature, relative humidity and
pressure (they set sound speed and air absorption), and
`synchronized_receivers`. A `MeasurementPose` has position, yaw/pitch/roll
and a one-sigma uncertainty for each; distance alone identifies neither
reflection geometry nor directivity. Channels are spatially comparable only
when they are sample-synchronized observations of one excitation; channels
that hold different source positions may feed independent mono metrics but
never coherence or IACC.

## Audit

`audit_measurement_campaign(campaign, root, *, verify_hashes=True,
minimum_records_per_room=12)` checks that no path is claimed with two hashes,
every asset exists and matches its hash, all three splits have rooms, every
room reaches the minimum number of records and of distinct
source/receiver configurations, every capture has repeated raw sweeps plus
noise and inverse-filter assets, and at least one record is a synchronized
multi-receiver capture. Repeated takes of one pose improve uncertainty
estimates but do not count as distinct configurations. The fit refuses to
run (`CampaignNotReadyError`) unless every check passes
(`ready_for_m5_inverse_calibration`).

## Splits

Rooms are disjoint across splits: `validation` and `test` rooms are excluded
from fitting, and `test` rooms also from model selection. Within each train
room, `deterministic_position_assignments(campaign, holdout_fraction=0.25)`
sorts records by SHA-256 of `campaign_id:room_id:measurement_id` and marks the
first `round(0.25 n)` (at least one, never all) as `position_holdout`, the rest
as `position_fit`; records in other rooms are labelled `room_validation` or
`room_test`. Re-running the pipeline therefore reproduces the same split
(policy `puresound.m5_position_split.sha256.v1`). A train room needs at least
two records.

## Calibration objective

`analyze_rir_calibration_loss(measured, synthetic, sample_rate, *, ...)` is a
NumPy/SciPy reference metric, not an autograd loss:

$$L = \sum_i w_i L_i$$

| Term | Checks | Default weight |
|---|---|---|
| `multiresolution_stft` | early structure and spectral detail (FFT sizes 256/512/1024) | 1 |
| `energy_decay` | decay-curve shape after direct-arrival alignment | 1 |
| `arrival_timing` | direct-arrival timing | 1 |
| `octave_acoustics` | bandwise decay and level (octaves 125–4000 Hz) | 1 |
| `spatial_coherence` | late complex coherence of every receiver pair | 1 |
| `causality` | synthetic energy before the physical first arrival | 10 |
| `decay_regularization` | growing or unstable tails | 1 |

The report (`CalibrationLossReport`) keeps each term, its diagnostics and the
weights. With fewer than two channels the spatial term contributes 0 to the
total but is reported `evaluable: false` with a reason; read `evaluable`
rather than treating that 0 as a successful spatial match.

## Fitting

`run_measured_campaign_fit(campaign, root, *, minimum_records_per_room=12,
position_holdout_fraction=0.25, mixing_time_candidates_s=(0.020, 0.024,
0.032), max_order=4, maximum_evaluations=60)`:

1. audits the campaign and freezes the position assignments;
2. for every record and receiver, builds a shoebox path-event model from
   `dimensions_m` (image order `max_order`, edges and corners excluded; mesh
   rooms are not supported by this runner);
3. per train room, fits an `M4InverseParameters` profile (mixing time,
   coherent reflection gain, octave RT60 targets at the valid 500–4000 Hz
   centres) over the mixing-time candidates, then refines six per-boundary
   reflection adjustments with `fit_grouped_path_gains`;
4. evaluates the fit and holdout positions of each train room;
5. evaluates `validation` and `test` rooms with the median of the train-room
   parameters, as a model for an unseen room would be used.

The report records the audit, the split, per-room fits, population
parameters and aggregates, and passes only if the splits stayed disjoint,
every train room has fit and holdout positions, the profile and grouped fits
converged, all holdout groups were evaluated and every loss is finite. It
always reports `production_enabled: false`.

Parameter groups correspond to observable causes (materials, directivity,
timing, late field). If a campaign cannot identify two groups separately,
freeze one or collect better measurements; `analyze_local_identifiability`
and the synthetic-recovery module check this on fixtures before real data is
fitted. Fit spatial parameters only from synchronized arrays.

A residual correction (`fit_causal_decay_residual`, policy
`puresound.m5_constrained_residual.v1`) is applied only after the physical fit
is stable. It learns one shared direct-relative template from training
positions, starts at the direct arrival, is limited to a residual-to-physical
energy ratio (default 0.25) and a maximum RT60 (default 0.8 s), and
`evaluate_residual_ablation` compares results with and without it. It must
not hide invalid geometry, a failed audit
or room leakage.

## Running the tools

```bash
python egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py \
  --campaign campaign.json --asset-root /path/to/assets --output-report report.json
```

Options: `--minimum-records-per-room`, `--position-holdout-fraction`,
`--mixing-time-ms`, `--max-order`, `--maximum-evaluations`. The exit code is 0
only when the report passes. Start from
`egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`,
a schema template with placeholder hashes, not measurement evidence. The
`validate_m5_*.py` scripts in `egs/rir_generation/phases/m5_calibration/scripts/`
check the individual steps (measurement contract, synthetic and robust
recovery, the inverse mapping through the M4 renderer, grouped
identifiability, spatial candidate profiling, constrained residual, the runner
on a synthetic campaign, and the aggregate exit check); each takes `--help`. Generated reports are local
experiment output unless a release evidence bundle includes them.

Tests: `test/rir/test_rir_measurement_campaign.py`, `test_rir_calibration.py`,
`test_rir_inverse_calibration.py`.

## Interpretation limits

Passing the schema, audit and synthetic-recovery checks shows that the
pipeline is implemented consistently. It does not show real-room
generalization, which still needs controlled measurements, room-disjoint
evaluation, and listening or downstream-task results appropriate to the
release.
