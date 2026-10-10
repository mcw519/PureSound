# Material-derived low-frequency modal damping

繁體中文版本：[modal_damping.zh-TW.md](modal_damping.zh-TW.md)

`puresound.audio.rir.render.low_frequency.modal_damping` gives every
rectangular room mode its own decay rate, computed from the frequency-dependent
absorption of the six walls instead of one room-wide RT60. The pytARD and
analytic low-frequency backends use it when material damping is enabled. The
loss law is not validated against measured rooms, so this path is
infrastructure for experiments, not a production default.

## Selecting it

`egs/rir_generation/generate_hybrid_rir.py --low-backend` accepts
`analytic-material`, `pytard-material` and `pytard-cupy-material`. These set
`material_modal_damping=True` on `AnalyticModalLowFrequencyBackend`,
`GpuARDPytARDBackend` or `GpuARDPytARDCuPyBackend`, and require
`--scene-version v1` (a `RoomSceneV2` with surface materials). The plain
`analytic`, `pytard` and `pytard-cupy` backends keep the shared RT60 decay
envelope, so both paths stay independently comparable.

## Boundary participation

The rigid-wall pressure eigenfunctions of an $L_x \times L_y \times L_z$ room
are

$$\phi_n(x,y,z) = \cos\frac{n_x\pi x}{L_x}\cos\frac{n_y\pi y}{L_y}\cos\frac{n_z\pi z}{L_z}.$$

Along one axis the volume norm is $I = L$ for index zero and $I = L/2$
otherwise. `material_modal_decay_rates(scene, nx, ny, nz, omega_rad_s,
sound_speed, loss_scale=1.0)` interpolates the six effective boundary
absorption spectra (log-frequency, held constant beyond the end bands) at each
mode frequency and computes

$$P_n = \frac{\alpha_\text{west} + \alpha_\text{east}}{I_x} + \frac{\alpha_\text{south} + \alpha_\text{north}}{I_y} + \frac{\alpha_\text{floor} + \alpha_\text{ceiling}}{I_z}$$

$$\gamma_n = \frac{s_\text{loss}\,c\,P_n}{8},\qquad \zeta_n = \frac{\gamma_n}{\omega_n},\qquad \mathrm{RT60}_n = \frac{\ln 1000}{\gamma_n},\qquad Q_n = \frac{\omega_n}{2\gamma_n}.$$

$cP_n/4$ is the first-order Sabine energy-loss rate; modal amplitude decays at
half of it, $\gamma_n$ (1/s). A mode with a nonzero index along an axis has
twice the participation of that wall pair, so changing one surface affects
modes differently, which a single RT60 cannot express. $\gamma_n$ is capped
at $0.95\,\omega_n$ to keep every mode underdamped, and near-DC modes get no
damping.

$s_\text{loss}$ (`--material-modal-loss-scale`, default 1.0) is the
uncalibrated physical prior. It exists for controlled diagnostics, is
serialized with every result, and the CLI rejects a non-default value unless
a `*-material` backend is selected. It is not an accepted physical constant.

## Analytic probe: source/receiver coupling

`AnalyticModalLowFrequencyBackend` (defaults `num_modes_per_axis=5`,
`max_modes=256`, `physical_mode_coupling=True`) excites and observes each mode
with its eigenfunction value at the actual endpoints:

$$A_n \propto (2-\delta_{n_x0})(2-\delta_{n_y0})(2-\delta_{n_z0})\;\phi_n(\text{source})\,\phi_n(\text{receiver})\;\frac{f_1}{f_n}$$

The factors in parentheses are the inverse cosine-mode volume norms. A source
or receiver on a modal node suppresses that mode, and swapping source and
receiver leaves the response unchanged. Each mode starts at zero phase at the
geometric direct-arrival time, so the finite expansion has no pre-arrival
tail; a $1/\max(d, 0.1)$ direct impulse is added at that time.

The probe enumerates every index triplet from 0 to `num_modes_per_axis` whose
rigid frequency lies in `[low_fmin_hz, low_fmax_hz]`. The 256-mode cap is a
safety limit above the 215 non-DC triplets of a 0–5 index range, so no mode
in the band is dropped by the cap.

## Exact damped recurrence in pytARD

The pytARD solve diagonalizes the room into independent modes. With material
damping, `solve_modal_ard` integrates each mode

$$\ddot q_n + 2\gamma_n\dot q_n + \omega_n^2 q_n = f_n$$

with its exact sampled poles. For time step $\Delta t$:

$$r = e^{-\gamma_n\Delta t},\quad \omega_d = \sqrt{\omega_n^2 - \gamma_n^2},\quad a_1 = 2r\cos(\omega_d\Delta t),\quad a_2 = -r^2,\quad b = \frac{1 - a_1 - a_2}{\omega_n^2}$$

$$q[k+1] = a_1 q[k] + a_2 q[k-1] + b f[k].$$

At $\gamma_n = 0$ this is exactly the undamped pytARD recurrence. The material
path does not apply `apply_rt60_decay_envelope`, and its metadata records
`global_rt60_envelope_applied: false`.

## Metadata

`material_modal_damping_metadata(scene, config, max_mode_index=16,
max_modes=128, loss_scale=1.0)` records, under model
`surface_participation_sabine`: each mode's indices, frequency, amplitude
decay rate, predicted RT60 and Q; the min/median/max RT60 and Q across modes;
the loss scale; and the absence of a global envelope.

## Tests

- `test/rir/test_rir_scene_v2.py`: damping depends on surface and mode
  (a west-wall change affects an x-axial mode twice as much as a mode with
  zero x index); the loss scale changes decay rate and Q reciprocally; the
  material backends drop the global RT60 envelope; the exact pytARD recurrence
  decays faster for absorbing boundaries.
- `test/rir/test_hybrid_rir.py`: the analytic probe obeys modal nodes,
  causality and source/receiver reciprocity.
- `test/rir/test_fdtd_reference.py`: the independent FDTD reference
  recovers the first axial modes' frequency and boundary Q, and the peak/Q
  estimator recovers a damped sinusoid ([modal validation](modal_validation.md)).

## Limits

- The boundary loss is a first-order absorption/participation model: no
  reflection phase, no angle dependence, no mode coupling from non-shoebox
  geometry.
- Complex impedance phase is handled by the separate impedance modal backend
  and the FDTD reference ([complex impedance](impedance_measurements.md)),
  not by this recurrence.
- A scalar loss scale does not correct the shape of the damping distribution;
  making this backend a default needs installed-material impedance evidence
  and downstream tests.
