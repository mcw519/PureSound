"""pytARD low-frequency wave backends (CPU and CuPy).

Wraps the vendored pytARD solver: an exact sampled modal recurrence, the
Green-delta excitation, level calibration and the optional broadband RT60
envelope.  Moved out of ``puresound.audio.rir.render.hybrid`` in R2 of
``RIR_EXP_LOG.md``.

pytARD and CuPy are imported lazily so this module stays importable without
them.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.arrays import coerce_rir_array, pad_or_trim
from puresound.audio.rir.render.low_frequency.modal_damping import (
    material_modal_decay_rates,
)
from puresound.audio.rir.scene.geometry import clip_position_to_room
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2


PYTARD_EXCITATION_POLICY = "puresound.pytard.green_delta.v1"


class _DiscardingList:
    def append(self, _value: Any) -> None:
        return None


@contextmanager
def _nullcontext():
    yield


def _default_pytard_root() -> Path:
    """Locate the vendored pytARD checkout under the ``puresound`` package.

    Anchored on the package rather than on ``__file__`` parent counting: this
    module moved from ``puresound/audio/`` to
    ``puresound/audio/rir/render/low_frequency/`` in R2, and a relative
    ``parents[n]`` walk silently pointed at the wrong directory.
    """

    import puresound

    return Path(puresound.__file__).resolve().parent / "third_party" / "pytARD"


@contextmanager
def _pytard_import_path(root: Path):
    root_str = str(root.resolve())
    inserted = root_str not in sys.path
    if inserted:
        sys.path.insert(0, root_str)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass


def _import_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "GpuARDPytARDCuPyBackend requires CuPy. Install a CUDA-matched "
            "package, for example `uv sync --extra hybrid-rir-gpu` or "
            "`uv pip install cupy-cuda12x`."
        ) from exc
    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            "CuPy is installed, but CUDA device initialization failed. Check "
            "the NVIDIA driver, CUDA runtime, and the installed cupy package."
        ) from exc
    return cp


@contextmanager
def _pytard_cupy_dct_patch():
    try:
        from cupyx.scipy.fft import dctn as cupy_dctn
        from cupyx.scipy.fft import idctn as cupy_idctn
    except ImportError as exc:
        raise ImportError(
            "GpuARDPytARDCuPyBackend requires cupyx.scipy.fft.dctn/idctn."
        ) from exc

    import cupy as cp
    import pytARD_3D.partition as partition_module

    original_dctn = partition_module.dctn
    original_idctn = partition_module.idctn

    def _dctn_gpu(x, *args, **kwargs):
        return cupy_dctn(cp.asarray(x), *args, **kwargs)

    def _idctn_gpu(x, *args, **kwargs):
        return cupy_idctn(cp.asarray(x), *args, **kwargs)

    partition_module.dctn = _dctn_gpu
    partition_module.idctn = _idctn_gpu
    try:
        yield
    finally:
        partition_module.dctn = original_dctn
        partition_module.idctn = original_idctn


def _move_pytard_partition_to_cupy(partition: Any, cp: Any) -> None:
    for attr in (
        "impulses",
        "pressure_field",
        "new_forces",
        "omega_i",
        "M_previous",
        "M_current",
    ):
        if hasattr(partition, attr):
            value = getattr(partition, attr)
            if value is not None:
                setattr(partition, attr, cp.asarray(value))


def _resample_ratio(target_fs: int, source_fs: int) -> tuple[int, int]:
    gcd = math.gcd(int(target_fs), int(source_fs))
    return int(target_fs // gcd), int(source_fs // gcd)


def apply_rt60_decay_envelope(
    signal: np.ndarray,
    sample_rate: int,
    rt60: float,
    origin_idx: int = 0,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or sample_rate <= 0 or rt60 <= 0:
        return signal
    origin = int(max(0, min(int(origin_idx), signal.shape[-1] - 1)))
    elapsed = np.maximum(
        np.arange(signal.shape[-1], dtype=np.float64) - float(origin),
        0.0,
    ) / float(sample_rate)
    envelope = 10.0 ** (-3.0 * elapsed / max(float(rt60), 1e-3))
    return signal * envelope


def calibrate_pytard_signal(
    signal: np.ndarray,
    distance_m: float,
    target_peak: float,
    sample_rate: int,
    rt60: float,
    apply_decay: bool,
    sound_speed: float = 343.0,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(signal)))
    if peak <= 1e-20:
        return signal
    distance_gain = 1.0 / max(float(distance_m), 0.1)
    calibrated = signal / peak * float(target_peak) * distance_gain
    if apply_decay:
        # The lossless ARD room has rigid walls and never decays, so impose the
        # RT60 envelope from the physical direct-path arrival (distance / c).
        # Keying the onset to the signal peak is wrong here: a lossless modal
        # field has no sharp direct path and its peak lands mid-ring.
        direct_idx = int(
            round(
                max(float(distance_m), 0.0)
                / max(float(sound_speed), 1e-6)
                * float(sample_rate)
            )
        )
        calibrated = apply_rt60_decay_envelope(
            calibrated,
            sample_rate=int(sample_rate),
            rt60=float(rt60),
            origin_idx=direct_idx,
        )
    return calibrated


def pytard_green_delta_excitation(
    sample_count: int,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return an uncoloured causal source for the discrete Green function."""

    count = int(sample_count)
    value = float(amplitude)
    if count < 1:
        raise ValueError("pytARD excitation requires at least one sample")
    if not np.isfinite(value):
        raise ValueError("pytARD excitation amplitude must be finite")
    impulse = np.zeros(count, dtype=np.float64)
    impulse[0] = value
    return impulse


def solve_modal_ard(
    sim_param: Any,
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    source_positions: np.ndarray,
    impulse: np.ndarray,
    xp: Any,
    cp: Any,
    material_scene: Optional[RoomSceneV2] = None,
    material_modal_loss_scale: float = 1.0,
    field_capture: Optional[dict[str, Any]] = None,
) -> list[np.ndarray]:
    """Exact modal ARD solve specialized to single-voxel sources and one mic.

    The vendored pytARD interior update integrates each room mode as an
    independent undamped oscillator, so it is mathematically identical to its
    per-step 3D DCT/IDCT loop. Two facts let us drop both transforms from the
    hot loop entirely:

    * The excitation is a single voxel scaled over time and the DCT is linear,
      so the forward DCT collapses to ``impulse[t] * dctn(unit_voxel)`` with the
      spatial basis computed once per source.
    * Only one microphone voxel is read back, so the inverse DCT at that point is
      a fixed dot product against a separable basis instead of a full 3D IDCT.

    What remains is an elementwise modal recurrence, batched across all sources
    and run on ``xp`` (CuPy when available, otherwise NumPy). The result is
    numerically equivalent to pytARD's loop while removing the FFT cost that
    dominated generation time.
    """
    from scipy.fft import dctn, idct, idctn

    c = float(sim_param.c)
    dt = float(sim_param.delta_t)
    n_samples = int(sim_param.number_of_samples)
    h = c / (
        float(sim_param.spatial_samples_per_wave_length)
        * float(sim_param.max_simulation_frequency)
    )
    dim = np.asarray(room_dim, dtype=np.float64).reshape(3)
    div_x = int(dim[0] / h)
    div_y = int(dim[1] / h)
    div_z = int(dim[2] / h)
    if min(div_x, div_y, div_z) < 1:
        raise ValueError(
            f"pytARD grid collapsed to {(div_z, div_y, div_x)}; raise "
            "--pytard-low-sample-rate or --pytard-spatial-samples-per-wavelength."
        )

    # Modal angular frequencies omega_i[z, y, x], matching AirPartition3D
    # preprocessing exactly (including the 1e-8 guard on the DC mode).
    zz, yy, xx = np.ogrid[0:div_z, 0:div_y, 0:div_x]
    omega = c * np.sqrt(
        (np.pi ** 2)
        * (
            (xx ** 2) / (dim[0] ** 2)
            + (yy ** 2) / (dim[1] ** 2)
            + (zz ** 2) / (dim[2] ** 2)
        )
    )
    omega[0, 0, 0] = 1e-8
    if material_scene is None:
        cos_k = np.cos(omega * dt)
        recurrence_current = 2.0 * cos_k
        recurrence_previous = -np.ones_like(omega)
        coef = (2.0 / (omega ** 2)) * (1.0 - cos_k)
    else:
        gamma = material_modal_decay_rates(
            material_scene,
            nx=xx,
            ny=yy,
            nz=zz,
            omega_rad_s=omega,
            sound_speed=c,
            loss_scale=material_modal_loss_scale,
        )
        damped_omega = np.sqrt(np.maximum(omega**2 - gamma**2, 0.0))
        pole_radius = np.exp(-gamma * dt)
        damped_cosine = np.cos(damped_omega * dt)
        recurrence_current = 2.0 * pole_radius * damped_cosine
        recurrence_previous = -(pole_radius**2)
        coef = (
            1.0 - recurrence_current - recurrence_previous
        ) / np.maximum(omega**2, 1e-16)
    recurrence_current[0, 0, 0] = 2.0
    recurrence_previous[0, 0, 0] = -1.0

    def _voxel(pos: np.ndarray) -> tuple[int, int, int]:
        return (
            min(int(div_z * (pos[2] / dim[2])), div_z - 1),
            min(int(div_y * (pos[1] / dim[1])), div_y - 1),
            min(int(div_x * (pos[0] / dim[0])), div_x - 1),
        )

    # Separable inverse-DCT basis so the mic readout is a dot product <M, basis>.
    mic_voxel = _voxel(np.asarray(mic_pos, dtype=np.float64).reshape(3))
    basis_z = idct(np.eye(div_z), type=2, axis=0)[mic_voxel[0]]
    basis_y = idct(np.eye(div_y), type=2, axis=0)[mic_voxel[1]]
    basis_x = idct(np.eye(div_x), type=2, axis=0)[mic_voxel[2]]
    basis = (
        basis_z[:, None, None] * basis_y[None, :, None] * basis_x[None, None, :]
    )

    num_sources = int(source_positions.shape[0])
    capture = field_capture if isinstance(field_capture, dict) else None
    if capture is not None:
        source_index = int(capture.get("source_index", 0))
        if not 0 <= source_index < num_sources:
            raise ValueError(
                f"pressure-field source_index={source_index} is outside "
                f"the {num_sources} generated source channels"
            )
        if capture.get("z_index") is None:
            z_m = float(capture.get("z_m", float(mic_pos[2])))
            z_index = min(max(int(div_z * z_m / dim[2]), 0), div_z - 1)
        else:
            z_index = min(max(int(capture["z_index"]), 0), div_z - 1)
        stride = max(1, int(capture.get("stride", 1)))
        max_frames = max(1, int(capture.get("max_frames", 240)))
        stride = max(stride, int(math.ceil(n_samples / max_frames)))
        capture["source_index"] = source_index
        capture["z_index"] = z_index
        capture["z_m"] = float((z_index + 0.5) * dim[2] / div_z)
        capture["stride"] = stride
        capture["grid_shape_zyx"] = [int(div_z), int(div_y), int(div_x)]
        capture["slice_shape_yx"] = [int(div_y), int(div_x)]
        capture["dt_s"] = float(dt)
        capture["frames"] = []
        capture["times_s"] = []
        basis_z_d = xp.asarray(
            idct(np.eye(div_z), type=2, axis=0)[z_index], dtype=xp.float64
        )
    else:
        source_index = 0
        stride = 0
        max_frames = 0
        basis_z_d = None

    # The forcing fed to the (former) forward DCT lags the impulse by one step,
    # mirroring how pytARD primes new_forces in preprocessing.
    impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)
    v_in = np.zeros(n_samples, dtype=np.float64)
    if impulse.size:
        v_in[0] = impulse[0]
        if n_samples > 1:
            copy_n = min(n_samples - 1, impulse.size)
            v_in[1 : 1 + copy_n] = impulse[:copy_n]

    gain = np.zeros((num_sources, div_z, div_y, div_x), dtype=np.float64)
    dc_drive = np.zeros((num_sources, n_samples), dtype=np.float64)
    for s in range(num_sources):
        sz, sy, sx = _voxel(source_positions[s])
        unit_voxel = np.zeros((div_z, div_y, div_x), dtype=np.float64)
        unit_voxel[sz, sy, sx] = 1.0
        forces_basis = dctn(unit_voxel, type=2, s=[div_z, div_y, div_x])
        gain[s] = coef * forces_basis
        if (sz, sy, sx) == (0, 0, 0):
            dc_drive[s, : min(n_samples, impulse.size)] = impulse[:n_samples]

    recurrence_current_d = xp.asarray(recurrence_current)
    recurrence_previous_d = xp.asarray(recurrence_previous)
    basis_d = xp.asarray(basis)
    gain_d = xp.asarray(gain)
    dc_d = xp.asarray(dc_drive)
    dt2 = dt * dt

    m_prev = xp.zeros((num_sources, div_z, div_y, div_x), dtype=xp.float64)
    m_cur = xp.zeros_like(m_prev)
    signal = xp.zeros((num_sources, n_samples), dtype=xp.float64)

    for t in range(n_samples):
        force_field = float(v_in[t]) * gain_d
        # DC mode (k=0) follows the exact-impulse recurrence used by pytARD.
        force_field[:, 0, 0, 0] = (
            2.0 * m_cur[:, 0, 0, 0] - m_prev[:, 0, 0, 0] + dt2 * dc_d[:, t]
        )
        m_next = (
            recurrence_current_d * m_cur
            + recurrence_previous_d * m_prev
            + force_field
        )
        signal[:, t] = (m_next * basis_d).sum(axis=(1, 2, 3))
        if (
            capture is not None
            and t % stride == 0
            and len(capture["frames"]) < max_frames
        ):
            # A fixed-z slice of the inverse 3-D DCT can be obtained by
            # collapsing the z modes with the corresponding basis vector and
            # applying a 2-D inverse DCT over y/x.  This avoids materializing
            # the full 3-D pressure volume for every animation frame.
            plane_modes = (
                m_next[source_index] * basis_z_d[:, None, None]
            ).sum(axis=0)
            plane_modes_host = (
                cp.asnumpy(plane_modes) if cp is not None else np.asarray(plane_modes)
            )
            pressure_plane = idctn(
                plane_modes_host,
                type=2,
                s=[div_y, div_x],
            )
            capture["frames"].append(
                np.asarray(pressure_plane, dtype=np.float32)
            )
            capture["times_s"].append(float((t + 1) * dt))
        m_prev = m_cur
        m_cur = m_next

    signal_host = cp.asnumpy(signal) if cp is not None else np.asarray(signal)
    if capture is not None:
        capture["frames"] = (
            np.stack(capture["frames"], axis=0)
            if capture["frames"]
            else np.zeros((0, div_y, div_x), dtype=np.float32)
        )
        capture["times_s"] = np.asarray(capture["times_s"], dtype=np.float64)
    return [signal_host[s] for s in range(num_sources)]


@dataclass
class PytARDWaveBackend:
    """Adapter for a pytARD-style low-frequency wave backend.

    If ``simulate_fn`` is provided, it is called directly.  Otherwise this class
    tries a small set of conventional module/function names and raises a clear
    error if no pytARD-compatible function is available.
    """

    simulate_fn: Optional[Callable[..., np.ndarray]] = None

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        fn = self.simulate_fn or self._discover_simulate_fn()
        rir = fn(
            room_dim=np.asarray(scene.room_dim, dtype=np.float64),
            mic_pos=np.asarray(scene.mic_pos, dtype=np.float64),
            source_pos=np.asarray(scene.source_pos, dtype=np.float64),
            obstacles=scene.obstacles,
            rt60=float(scene.rt60),
            fs=int(config.sample_rate),
            duration=float(config.duration),
            fmin=float(config.low_fmin_hz),
            fmax=float(config.low_fmax_hz),
            sound_speed=float(config.sound_speed),
        )
        return coerce_rir_array(rir, config.num_sources, config.num_samples)

    @staticmethod
    def _discover_simulate_fn() -> Callable[..., np.ndarray]:
        import importlib

        errors: list[str] = []
        for module_name in ("pytard", "pyTARD"):
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                errors.append(f"{module_name}: {exc}")
                continue
            for attr in ("simulate_rir", "simulate", "generate_rir"):
                fn = getattr(module, attr, None)
                if callable(fn):
                    return fn
            errors.append(f"{module_name}: no simulate_rir/simulate/generate_rir")
        detail = "; ".join(errors) if errors else "module not found"
        raise ImportError(
            "No pytARD-compatible backend was found. Install pytARD or pass "
            "PytARDWaveBackend(simulate_fn=...) with a function returning "
            "[num_sources, samples] low-frequency RIRs. "
            f"Discovery detail: {detail}"
        )


@dataclass
class GpuARDPytARDBackend:
    """Backend for the vendored ``gpuard/pytARD`` implementation.

    The upstream project is script-oriented and imports modules from its repo
    root (for example ``pytARD_3D`` and ``common``).  This adapter adds the
    vendored path during simulation, builds one 3D air partition per source,
    reads the microphone signal in memory, and resamples it to the requested
    output sample rate when needed.
    """

    third_party_root: Optional[Path] = None
    low_sample_rate: int = 16000
    spatial_samples_per_wave_length: int = 2
    amplitude: float = 1.0
    filter_order: int = 41
    calibrate_output: bool = True
    calibration_peak: float = 0.05
    apply_rt60_decay: bool = True
    rt60_decay_scale: float = 1.0
    material_modal_damping: bool = False
    material_modal_loss_scale: float = 1.0
    verbose: bool = False
    visualize: bool = False
    disable_notifications: bool = True
    last_excitation_metadata: Optional[dict[str, Any]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        return self._simulate_with_pytard(scene, config, use_cupy=False)

    def simulate_with_pressure_field(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
        field_capture: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the low solver and capture a 2-D pressure-field slice.

        ``field_capture`` is deliberately opt-in because a field animation is
        a diagnostic artifact, not part of a normal RIR bank.  The capture is
        performed inside the same modal recurrence used for the microphone
        signal, so the resulting frames are not a separate illustrative FDTD
        simulation.  Only one z-slice is reconstructed and copied to host
        memory at the requested stride.
        """
        self._simulate_with_pytard(
            scene,
            config,
            use_cupy=False,
            field_capture=field_capture,
        )
        return field_capture.get("low_rir"), field_capture

    def _simulate_with_pytard(
        self,
        scene: HybridRIRScene,
        config: HybridRIRConfig,
        use_cupy: bool,
        field_capture: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        root = self.third_party_root or _default_pytard_root()
        if not root.exists():
            raise ImportError(
                f"Vendored pytARD was not found at {root}. "
                "Clone https://github.com/gpuard/pytARD.git into "
                "puresound/third_party/pytARD."
            )

        cp = _import_cupy() if use_cupy else None
        if self.material_modal_damping and not isinstance(scene, RoomSceneV2):
            raise ValueError(
                "material_modal_damping requires a RoomSceneV2 with boundary materials"
            )
        low_fs = int(self.low_sample_rate)
        spp = max(1, int(self.spatial_samples_per_wave_length))
        crossover = float(config.crossover_hz)
        nyq = low_fs / 2.0 - 1.0
        # The modal grid must resolve a little above the crossover so the low
        # band actually carries energy where it meets the high band. The grid
        # cutoff is bounded by Nyquist and the ARD CFL stability limit
        # (CFL = 3 * spp * fmax / Fs <= sqrt(1/3)).
        cfl_fmax = 0.95 * math.sqrt(1.0 / 3.0) * low_fs / (3.0 * spp)
        sim_fmax = min(max(float(config.low_fmax_hz), crossover * 1.2), nyq, cfl_fmax)
        if sim_fmax <= 0:
            raise ValueError(
                f"low_sample_rate={low_fs} with spatial_samples_per_wave_length="
                f"{spp} cannot resolve the crossover at {crossover} Hz."
            )
        xp = cp if cp is not None else np

        with _pytard_import_path(root):
            from common.parameters import SimulationParameters

            room_dim = np.asarray(scene.room_dim, dtype=np.float64)
            mic_pos = clip_position_to_room(scene.mic_pos, room_dim)
            srcs = np.asarray(scene.source_pos, dtype=np.float64)
            clipped_srcs = np.asarray(
                [clip_position_to_room(src, room_dim) for src in srcs],
                dtype=np.float64,
            )
            sim_param = SimulationParameters(
                max_simulation_frequency=sim_fmax,
                T=float(config.duration),
                spatial_samples_per_wave_length=int(self.spatial_samples_per_wave_length),
                c=int(round(float(config.sound_speed))),
                Fs=low_fs,
                verbose=bool(self.verbose),
                visualize=bool(self.visualize),
            )
            # Solve the discrete Green's function directly.  pytARD's upstream
            # ``Unit`` helper appends a delayed negative copy of a 41-tap FIR.
            # Treating that bipolar probe as an RIR imprints (1-z^-41) comb
            # zeros at 390.24 Hz intervals for 16 kHz audio.  The modal grid
            # already limits the represented frequencies to ``sim_fmax`` and
            # the hybrid crossover supplies the output band limit, so a causal
            # unit sample is the correct uncoloured source here.
            impulse = pytard_green_delta_excitation(
                int(sim_param.number_of_samples),
                amplitude=float(self.amplitude),
            )
            self.last_excitation_metadata = {
                "policy": PYTARD_EXCITATION_POLICY,
                "nonzero_sample_count": 1,
                "source_sample": 0,
                "amplitude": float(self.amplitude),
                "solver_bandlimit_hz": float(sim_fmax),
                "legacy_bipolar_fir_used": False,
            }

        mic_signals = solve_modal_ard(
            sim_param=sim_param,
            room_dim=room_dim,
            mic_pos=mic_pos,
            source_positions=clipped_srcs,
            impulse=impulse,
            xp=xp,
            cp=cp,
            material_scene=(scene if self.material_modal_damping else None),
            material_modal_loss_scale=float(self.material_modal_loss_scale),
            field_capture=field_capture,
        )

        low_rirs: list[np.ndarray] = []
        for source_idx, source in enumerate(clipped_srcs):
            signal = mic_signals[source_idx]
            if self.calibrate_output:
                distance = float(np.linalg.norm(source - mic_pos))
                signal = calibrate_pytard_signal(
                    signal,
                    distance,
                    target_peak=float(self.calibration_peak),
                    sample_rate=low_fs,
                    rt60=float(scene.rt60) * float(self.rt60_decay_scale),
                    apply_decay=bool(
                        self.apply_rt60_decay and not self.material_modal_damping
                    ),
                    sound_speed=float(config.sound_speed),
                )
            low_rirs.append(signal)

        low = pad_or_trim(low_rirs, int(round(config.duration * low_fs)))
        if low_fs != int(config.sample_rate):
            up, down = _resample_ratio(int(config.sample_rate), low_fs)
            low = np.asarray([resample_poly(channel, up, down) for channel in low])
        low = coerce_rir_array(low, config.num_sources, config.num_samples)
        if field_capture is not None:
            field_capture["low_rir"] = low
        return low


@dataclass
class GpuARDPytARDCuPyBackend(GpuARDPytARDBackend):
    """CuPy-accelerated backend for vendored ``gpuard/pytARD``.

    This keeps the third-party source untouched and patches pytARD's 3D DCT/IDCT
    calls to CuPy only while this backend is running.
    """

    def simulate_with_pressure_field(
        self,
        scene: HybridRIRScene | RoomSceneV2,
        config: HybridRIRConfig,
        field_capture: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """GPU variant of :meth:`GpuARDPytARDBackend.simulate_with_pressure_field`."""
        self._simulate_with_pytard(
            scene,
            config,
            use_cupy=True,
            field_capture=field_capture,
        )
        return field_capture.get("low_rir"), field_capture

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        return self._simulate_with_pytard(scene, config, use_cupy=True)


__all__ = [
    "PYTARD_EXCITATION_POLICY",
    "GpuARDPytARDBackend",
    "GpuARDPytARDCuPyBackend",
    "PytARDWaveBackend",
    "apply_rt60_decay_envelope",
    "calibrate_pytard_signal",
    "pytard_green_delta_excitation",
    "solve_modal_ard",
]
