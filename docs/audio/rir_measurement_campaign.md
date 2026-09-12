# RIR measurement and calibration

繁體中文版本：[rir_measurement_campaign.zh-TW.md](rir_measurement_campaign.zh-TW.md)

This document defines the measured-room data accepted by inverse calibration.
The goal is to fit room behavior without confusing it with transducer
response, geometry error, clock delay, or data leakage.

Schema: `puresound.rir_measurement_campaign.v1`

Loss policy: `puresound.rir_calibration_loss.v1`

## Required measurement assets

Each source/receiver configuration uses at least two exponential sine sweep
(ESS) captures. Retain:

- raw sweep recordings;
- the inverse filter used for deconvolution;
- a background-noise recording;
- deconvolution, harmonic-separation, and latency settings;
- the resulting RIR.

All assets for a capture use one sample rate. Repeated captures are required
to estimate repeatability and reject unstable measurements, not merely to
produce an average.

## Room record

Every measured room needs:

- a stable `room_id` and room type;
- a documented right-handed coordinate system;
- geometry uncertainty;
- dimensions or a retained mesh with SHA-256;
- source and receiver poses;
- environment values;
- retained asset hashes.

Do not rename the same physical room for different measurement sessions or
positions.

### Transducers and poses

Source and receiver records keep manufacturer, model, serial number,
reference axis, calibration date, calibration response, and provenance
separately.

A pose contains position, yaw/pitch/roll, and uncertainty for both. Distance
alone cannot identify reflection geometry or directivity.

### Environment and synchronization

Store temperature, relative humidity, and pressure. These affect sound speed
and air absorption.

Multi-channel spatial measurements must be sample-synchronized and represent
one excitation observed by several receivers. Channels containing different
source positions may be used for independent mono metrics, but not for
coherence or IACC.

### Asset identity

Asset paths are campaign-relative and cannot contain `..`. Each retained file
has a SHA-256. The campaign audit verifies file existence, hashes, and
conflicting claims for the same path.

## Data splits

Splits are room-disjoint:

- `train`: calibration positions plus deterministic position holdouts;
- `validation`: the complete room is excluded from fitting;
- `test`: the complete room is excluded from fitting and model selection.

`deterministic_position_assignments()` derives the position holdout from
campaign, room, and measurement IDs. Re-running the pipeline therefore keeps
the same split.

A train room needs enough distinct positions and orientations to identify the
selected parameters. Repeated takes of one pose improve uncertainty estimates
but do not count as distinct configurations.

## Calibration objective

The reference report combines complementary terms:

```text
L = w_stft L_stft
  + w_edc L_edc
  + w_arr L_arr
  + w_oct L_oct
  + w_sp L_sp
  + w_causal R_causal
  + w_decay R_decay
```

| Term | What it checks |
|---|---|
| Multiresolution STFT | Early structure and spectral detail |
| Energy-decay curve | Decay shape after direct-arrival alignment |
| Arrival timing | Direct and early-reflection timing |
| Octave acoustics | Bandwise decay and level |
| Spatial coherence | Synchronized receiver relationships |
| Causality regularization | Energy before physical arrival |
| Decay regularization | Implausible or unstable tails |

`analyze_rir_calibration_loss()` reports every term and diagnostic. It is a
NumPy/SciPy reference metric, not an autograd loss.

Spatial terms are `not_applicable` for non-synchronized data; they must not
be replaced with zero and counted as a successful spatial comparison.

## Calibration workflow

1. Audit the campaign contract and retained assets.
2. Freeze room and position assignments.
3. Fit only the enabled parameter groups.
4. Check parameter identifiability and recovery on synthetic fixtures.
5. Evaluate train-position holdouts.
6. Evaluate complete validation and test rooms.
7. Fit spatial parameters only from synchronized arrays.
8. Apply a constrained residual model only after the physical fit is stable.

Parameter groups should correspond to observable causes such as material,
directivity, timing, and late-field behavior. If two groups cannot be
separately identified by the campaign, freeze one or collect better
measurements.

A residual correction must remain bounded, causal, and traceable. It must not
hide invalid geometry, failed asset audit, or room leakage.

## Running the tools

Start from the template:

```text
egs/rir_generation/phases/m5_calibration/config/
  m5_measurement_campaign_template.json
```

Fit a campaign:

```bash
python egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py --help
```

Useful validation entry points include:

```bash
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py
```

Generated reports are local experiment output unless a release evidence
bundle explicitly includes them.

## Interpretation limits

Passing schema and synthetic-recovery checks proves that the pipeline is
implemented consistently. It does not prove real-room generalization.
Production evidence still requires controlled measurements, room-disjoint
evaluation, and listening or downstream-task results appropriate to the
release.
