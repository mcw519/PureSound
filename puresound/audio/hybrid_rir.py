"""Hybrid low-frequency wave and high-frequency geometric RIR generation.

The module is intentionally split into small, testable pieces.  Scene
sampling, obstacle geometry, crossover filtering, and file writing work without
optional simulators installed.  Real generation uses a low-frequency backend
such as pytARD and a high-frequency Pyroomacoustics backend.
"""

from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import warnings

import numpy as np
import torch
import torchaudio
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


class RIRBackend(Protocol):
    """Backend protocol returning RIRs with shape ``[num_sources, samples]``."""

    def simulate(
        self,
        scene: "HybridRIRScene",
        config: "HybridRIRConfig",
    ) -> np.ndarray:
        ...


@dataclass
class HybridRIRConfig:
    sample_rate: int = 48000
    duration: float = 1.5
    crossover_hz: float = 1000.0
    low_fmin_hz: float = 20.0
    low_fmax_hz: float = 1000.0
    sound_speed: float = 343.0
    room_dim_range: tuple[tuple[float, float], ...] = (
        (3.5, 8.0),
        (3.5, 7.0),
        (2.4, 3.6),
    )
    rt60_range: tuple[float, float] = (0.25, 0.8)
    mic_margin: float = 0.45
    source_margin: float = 0.35
    mic_height_range: tuple[float, float] = (0.65, 1.35)
    speech_source_height_range: tuple[float, float] = (1.1, 1.8)
    near_distance_range: tuple[float, float] = (0.35, 0.95)
    far_distance_range: tuple[float, float] = (2.05, 5.5)
    num_near_sources: int = 2
    num_far_sources: int = 3
    num_obstacles_range: tuple[int, int] = (1, 6)
    obstacle_density_per_m2: tuple[float, float] = (0.08, 0.16)
    max_obstacle_floor_coverage: float = 0.28
    obstacle_clearance: float = 0.25
    obstacle_obstacle_clearance: float = 0.12
    obstacle_margin: float = 0.4
    obstacle_height_range: tuple[float, float] = (0.35, 1.8)
    obstacle_radius_range: tuple[float, float] = (0.25, 0.9)
    normalize_peak: float = 0.98
    match_crossover_energy: bool = True
    crossover_match_band_hz: tuple[float, float] = (700.0, 1300.0)
    crossover_match_target_db: float = 0.0
    crossover_match_gain_range: tuple[float, float] = (1e-4, 2.0)

    @property
    def num_sources(self) -> int:
        return int(self.num_near_sources + self.num_far_sources)

    @property
    def num_samples(self) -> int:
        return int(round(float(self.sample_rate) * float(self.duration)))


@dataclass
class PolygonObstacle:
    footprint: list[list[float]]
    z_min: float
    z_max: float
    material: str
    absorption: float
    scattering: float

    def contains_xy(self, point: ArrayLike) -> bool:
        xy = np.asarray(point, dtype=np.float64)[:2]
        return _point_in_polygon(xy, np.asarray(self.footprint, dtype=np.float64))

    @property
    def center(self) -> np.ndarray:
        pts = np.asarray(self.footprint, dtype=np.float64)
        return np.asarray(
            [float(pts[:, 0].mean()), float(pts[:, 1].mean()), self.z_max * 0.5],
            dtype=np.float64,
        )


@dataclass
class HybridRIRScene:
    room_dim: list[float]
    rt60: float
    mic_pos: list[float]
    source_pos: list[list[float]]
    source_labels: list[str]
    obstacles: list[PolygonObstacle] = field(default_factory=list)

    def source_distances(self) -> list[float]:
        mic = np.asarray(self.mic_pos, dtype=np.float64)
        src = np.asarray(self.source_pos, dtype=np.float64)
        return np.linalg.norm(src - mic[None, :], axis=1).astype(float).tolist()

    def source_horizontal_distances(self) -> list[float]:
        mic = np.asarray(self.mic_pos, dtype=np.float64)[:2]
        src = np.asarray(self.source_pos, dtype=np.float64)[:, :2]
        return np.linalg.norm(src - mic[None, :], axis=1).astype(float).tolist()

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_distances"] = self.source_distances()
        data["source_horizontal_distances"] = self.source_horizontal_distances()
        data["obstacle_floor_coverage_ratio"] = _obstacle_floor_coverage(
            self.obstacles, np.asarray(self.room_dim, dtype=np.float64)
        )
        data["channel_map"] = [
            {
                "channel": idx,
                "label": label,
                "source_pos": self.source_pos[idx],
                "distance_m": data["source_distances"][idx],
                "horizontal_distance_m": data["source_horizontal_distances"][idx],
            }
            for idx, label in enumerate(self.source_labels)
        ]
        return data


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
        return _coerce_rir_array(rir, config.num_sources, config.num_samples)

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
    verbose: bool = False
    visualize: bool = False
    disable_notifications: bool = True

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        return self._simulate_with_pytard(scene, config, use_cupy=False)

    def _simulate_with_pytard(
        self,
        scene: HybridRIRScene,
        config: HybridRIRConfig,
        use_cupy: bool,
    ) -> np.ndarray:
        root = self.third_party_root or _default_pytard_root()
        if not root.exists():
            raise ImportError(
                f"Vendored pytARD was not found at {root}. "
                "Clone https://github.com/gpuard/pytARD.git into "
                "puresound/third_party/pytARD."
            )

        cp = _import_cupy() if use_cupy else None
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
        # pytARD's Unit impulse halves the requested cutoff internally
        # (firwin uses (cutoff / 2) * 0.95), so request 2x to actually excite up
        # to the crossover. Keep the excited band within the grid's resolvable
        # range to avoid injecting energy that would alias.
        excite_top = min(crossover, sim_fmax)
        unit_cutoff = min(2.0 * excite_top / 0.95, nyq)

        xp = cp if cp is not None else np

        with _pytard_import_path(root):
            from common.impulse import Unit
            from common.parameters import SimulationParameters

            room_dim = np.asarray(scene.room_dim, dtype=np.float64)
            mic_pos = _clip_position_to_room(scene.mic_pos, room_dim)
            srcs = np.asarray(scene.source_pos, dtype=np.float64)
            clipped_srcs = np.asarray(
                [_clip_position_to_room(src, room_dim) for src in srcs],
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
            filter_order = min(
                int(self.filter_order),
                max(3, int(sim_param.number_of_samples) // 2),
            )
            # The unit impulse depends only on the simulation parameters, not the
            # source location, so every source shares one broadband excitation.
            impulse = np.asarray(
                Unit(
                    sim_param,
                    mic_pos.reshape(3),
                    float(self.amplitude),
                    int(round(unit_cutoff)),
                    filter_order=filter_order,
                ).get(),
                dtype=np.float64,
            )

        mic_signals = _solve_modal_ard(
            sim_param=sim_param,
            room_dim=room_dim,
            mic_pos=mic_pos,
            source_positions=clipped_srcs,
            impulse=impulse,
            xp=xp,
            cp=cp,
        )

        low_rirs: list[np.ndarray] = []
        for source_idx, source in enumerate(clipped_srcs):
            signal = mic_signals[source_idx]
            if self.calibrate_output:
                distance = float(np.linalg.norm(source - mic_pos))
                signal = _calibrate_pytard_signal(
                    signal,
                    distance,
                    target_peak=float(self.calibration_peak),
                    sample_rate=low_fs,
                    rt60=float(scene.rt60) * float(self.rt60_decay_scale),
                    apply_decay=bool(self.apply_rt60_decay),
                    sound_speed=float(config.sound_speed),
                )
            low_rirs.append(signal)

        low = _pad_or_trim(low_rirs, int(round(config.duration * low_fs)))
        if low_fs != int(config.sample_rate):
            up, down = _resample_ratio(int(config.sample_rate), low_fs)
            low = np.asarray([resample_poly(channel, up, down) for channel in low])
        return _coerce_rir_array(low, config.num_sources, config.num_samples)


@dataclass
class GpuARDPytARDCuPyBackend(GpuARDPytARDBackend):
    """CuPy-accelerated backend for vendored ``gpuard/pytARD``.

    This keeps the third-party source untouched and patches pytARD's 3D DCT/IDCT
    calls to CuPy only while this backend is running.
    """

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        return self._simulate_with_pytard(scene, config, use_cupy=True)


def _solve_modal_ard(
    sim_param: Any,
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    source_positions: np.ndarray,
    impulse: np.ndarray,
    xp: Any,
    cp: Any,
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
    from scipy.fft import dctn, idct

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
    cos_k = np.cos(omega * dt)
    coef = (2.0 / (omega ** 2)) * (1.0 - cos_k)

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

    # The forcing fed to the (former) forward DCT lags the impulse by one step,
    # mirroring how pytARD primes new_forces in preprocessing.
    impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)
    v_in = np.zeros(n_samples, dtype=np.float64)
    if impulse.size:
        v_in[0] = impulse[0]
        if n_samples > 1:
            copy_n = min(n_samples - 1, impulse.size)
            v_in[1 : 1 + copy_n] = impulse[:copy_n]

    num_sources = int(source_positions.shape[0])
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

    cos_d = xp.asarray(cos_k)
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
        m_next = 2.0 * m_cur * cos_d - m_prev + force_field
        signal[:, t] = (m_next * basis_d).sum(axis=(1, 2, 3))
        m_prev = m_cur
        m_cur = m_next

    signal_host = cp.asnumpy(signal) if cp is not None else np.asarray(signal)
    return [signal_host[s] for s in range(num_sources)]


@dataclass
class AnalyticModalLowFrequencyBackend:
    """Small deterministic fallback for tests and smoke runs.

    This is not a replacement for pytARD.  It creates low-frequency direct
    paths and damped axial/tangential room modes so the full pipeline can be
    exercised before connecting a wave solver.
    """

    num_modes_per_axis: int = 5

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        fs = int(config.sample_rate)
        n = int(config.num_samples)
        t = np.arange(n, dtype=np.float64) / float(fs)
        room = np.asarray(scene.room_dim, dtype=np.float64)
        mic = np.asarray(scene.mic_pos, dtype=np.float64)
        srcs = np.asarray(scene.source_pos, dtype=np.float64)
        out = np.zeros((srcs.shape[0], n), dtype=np.float64)
        tau = max(float(scene.rt60) / 6.91, 1e-3)

        mode_freqs: list[float] = []
        for nx in range(self.num_modes_per_axis + 1):
            for ny in range(self.num_modes_per_axis + 1):
                for nz in range(self.num_modes_per_axis + 1):
                    if nx == ny == nz == 0:
                        continue
                    f = config.sound_speed * 0.5 * math.sqrt(
                        (nx / room[0]) ** 2
                        + (ny / room[1]) ** 2
                        + (nz / room[2]) ** 2
                    )
                    if config.low_fmin_hz <= f <= config.low_fmax_hz:
                        mode_freqs.append(float(f))
        mode_freqs = sorted(mode_freqs)[:64]

        for idx, src in enumerate(srcs):
            distance = float(np.linalg.norm(src - mic))
            direct = int(round(distance / float(config.sound_speed) * fs))
            if direct < n:
                out[idx, direct] += 1.0 / max(distance, 0.1)
            phase_seed = float(np.dot(src + mic, np.array([0.37, 0.61, 0.83])))
            for mode_idx, freq in enumerate(mode_freqs):
                phase = phase_seed * (mode_idx + 1)
                amp = 0.015 / math.sqrt(mode_idx + 1)
                wave = np.sin(2.0 * math.pi * freq * t + phase) * np.exp(-t / tau)
                out[idx] += amp * wave
        return out.astype(np.float32)


@dataclass
class PyroomacousticsHighFrequencyBackend:
    # ``absorption`` is only used as a fallback when the requested RT60 cannot be
    # satisfied by Sabine's formula for the given room (e.g. RT60 too short).
    absorption: float = 0.35
    max_order: int = 12
    ray_tracing: bool = True
    air_absorption: bool = True
    n_rays: int = 20000
    receiver_radius: float = 0.08

    def _absorption_and_max_order(self, scene: HybridRIRScene, pra) -> tuple[float, int]:
        """Derive wall absorption and ISM order from the requested scene RT60.

        The geometric backend models the full broadband room response, so its
        reverberation time must follow ``scene.rt60`` instead of a constant
        absorption. Image sources cover the early part and ray tracing fills the
        diffuse tail, so the ISM order is capped to keep generation tractable.
        """
        try:
            e_absorption, needed_order = pra.inverse_sabine(
                float(scene.rt60), list(scene.room_dim)
            )
        except Exception:
            # Scene rt60s are clamped to the Sabine-feasible minimum at sampling
            # time (_min_feasible_rt60), so this is a safety net for externally
            # constructed scenes. It changes the realized reverberation away
            # from scene.rt60 -- say so instead of diverging silently.
            warnings.warn(
                f"inverse_sabine failed for rt60={scene.rt60:.3f}s in room "
                f"{np.round(scene.room_dim, 2).tolist()}; falling back to "
                f"absorption={self.absorption} (metadata rt60 no longer matches)",
                RuntimeWarning,
                stacklevel=2,
            )
            return float(self.absorption), int(self.max_order)
        e_absorption = float(np.clip(e_absorption, 1e-3, 1.0))
        order = int(min(int(needed_order), int(self.max_order)))
        return e_absorption, max(0, order)

    def simulate(self, scene: HybridRIRScene, config: HybridRIRConfig) -> np.ndarray:
        try:
            import pyroomacoustics as pra
        except ImportError as exc:
            raise ImportError(
                "Pyroomacoustics is required for high-frequency RIR generation. "
                "Install it with `pip install pyroomacoustics` or pass a custom "
                "high-frequency backend."
            ) from exc

        e_absorption, max_order = self._absorption_and_max_order(scene, pra)
        room_kwargs = {
            "fs": int(config.sample_rate),
            "max_order": int(max_order),
        }
        if hasattr(pra, "Material"):
            room_kwargs["materials"] = pra.Material(e_absorption)
        else:
            room_kwargs["absorption"] = e_absorption
        try:
            room = pra.ShoeBox(
                scene.room_dim,
                air_absorption=bool(self.air_absorption),
                **room_kwargs,
            )
        except TypeError:
            room = pra.ShoeBox(scene.room_dim, **room_kwargs)
            if self.air_absorption and hasattr(room, "set_air_absorption"):
                room.set_air_absorption()
        if self.ray_tracing and hasattr(room, "set_ray_tracing"):
            try:
                room.set_ray_tracing(
                    receiver_radius=float(self.receiver_radius),
                    n_rays=int(self.n_rays),
                    energy_thres=1e-7,
                )
            except TypeError:
                room.set_ray_tracing(
                    receiver_radius=float(self.receiver_radius),
                    n_rays=int(self.n_rays),
                )

        for src in scene.source_pos:
            room.add_source(np.asarray(src, dtype=np.float64))
        room.add_microphone_array(np.asarray(scene.mic_pos, dtype=np.float64).reshape(3, 1))
        room.compute_rir()

        rirs = []
        for source_idx in range(config.num_sources):
            raw = np.asarray(room.rir[0][source_idx], dtype=np.float64)
            rirs.append(raw)
        rir = _pad_or_trim(rirs, config.num_samples)
        rir = _align_high_band_direct(rir, scene, config)
        return apply_obstacle_high_frequency_effects(rir, scene, config)


def _min_feasible_rt60(room_dim: np.ndarray, sound_speed: float = 343.0) -> float:
    """Shortest RT60 Sabine allows for this room (absorption capped at 0.99).

    Below this value ``pra.inverse_sabine`` needs an absorption coefficient
    above 1 and raises; the high band would silently fall back to a fixed
    absorption while the metadata kept the impossible request. Clamping at
    sampling time keeps the recorded rt60 equal to the realized one for both
    bands (the low band imposes whatever envelope it is told).
    """
    lx, ly, lz = (float(v) for v in room_dim)
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + lx * lz + ly * lz)
    sabine_coeff = 24.0 * np.log(10.0) / sound_speed
    return float(sabine_coeff * volume / (surface * 0.99))


def sample_hybrid_rir_scene(
    config: HybridRIRConfig,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> HybridRIRScene:
    rng = np.random.default_rng(seed) if rng is None else rng
    room_dim = np.asarray(
        [rng.uniform(low, high) for low, high in config.room_dim_range],
        dtype=np.float64,
    )
    rt60 = max(
        float(rng.uniform(*config.rt60_range)),
        _min_feasible_rt60(room_dim, config.sound_speed),
    )
    mic_pos = _sample_point(
        room_dim,
        config.mic_margin,
        rng,
        height_range=config.mic_height_range,
    )

    source_pos: list[np.ndarray] = []
    labels: list[str] = []
    for idx in range(config.num_near_sources):
        source_pos.append(
            _sample_source_in_horizontal_shell(
                room_dim,
                mic_pos,
                config.near_distance_range[0],
                config.near_distance_range[1],
                config.source_margin,
                config.speech_source_height_range,
                rng,
            )
        )
        labels.append(f"near_{idx}")
    for idx in range(config.num_far_sources):
        max_far = min(
            float(config.far_distance_range[1]),
            _max_room_horizontal_distance_from_point(
                room_dim, mic_pos, config.source_margin
            ),
        )
        min_far = min(float(config.far_distance_range[0]), max_far)
        source_pos.append(
            _sample_source_in_horizontal_shell(
                room_dim,
                mic_pos,
                min_far,
                max_far,
                config.source_margin,
                config.speech_source_height_range,
                rng,
            )
        )
        labels.append(f"far_{idx}")

    obstacles = sample_polygon_obstacles(
        room_dim=room_dim,
        protected_points=[mic_pos, *source_pos],
        config=config,
        rng=rng,
    )
    return HybridRIRScene(
        room_dim=room_dim.astype(float).tolist(),
        rt60=rt60,
        mic_pos=mic_pos.astype(float).tolist(),
        source_pos=[src.astype(float).tolist() for src in source_pos],
        source_labels=labels,
        obstacles=obstacles,
    )


def sample_polygon_obstacles(
    room_dim: np.ndarray,
    protected_points: list[np.ndarray],
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> list[PolygonObstacle]:
    materials = _obstacle_material_profiles(room_dim)
    material_names = list(materials.keys())
    material_weights = np.asarray(
        [materials[name]["sample_weight"] for name in material_names], dtype=np.float64
    )
    material_weights = material_weights / material_weights.sum()
    num_obstacles = _sample_obstacle_count(room_dim, config, rng)
    obstacles: list[PolygonObstacle] = []
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    used_area = 0.0
    for _ in range(num_obstacles):
        for _attempt in range(128):
            material = str(rng.choice(material_names, p=material_weights))
            profile = materials[material]
            n_vertices = int(rng.integers(4, 8))
            radius = float(rng.uniform(*profile["radius_range"]))
            center = _sample_obstacle_center(
                room_dim=room_dim,
                radius=radius,
                margin=float(config.obstacle_margin),
                placement=str(profile["placement"]),
                rng=rng,
            )
            angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=n_vertices))
            radii = radius * rng.uniform(0.55, 1.0, size=n_vertices)
            footprint = np.column_stack(
                [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]
            )
            if not _polygon_inside_room(footprint, room_dim, config.obstacle_margin * 0.5):
                continue
            if any(
                _distance_point_to_polygon(point[:2], footprint)
                < float(config.obstacle_clearance)
                or _point_in_polygon(point[:2], footprint)
                for point in protected_points
            ):
                continue
            if _obstacle_conflicts(
                footprint,
                obstacles,
                clearance=float(config.obstacle_obstacle_clearance),
            ):
                continue
            footprint_area = _polygon_area(footprint)
            if (
                used_area + footprint_area
                > floor_area * float(config.max_obstacle_floor_coverage)
            ):
                continue
            z_min, z_max = _sample_obstacle_height(room_dim, profile, config, rng)
            obstacles.append(
                PolygonObstacle(
                    footprint=footprint.astype(float).tolist(),
                    z_min=float(z_min),
                    z_max=float(z_max),
                    material=material,
                    absorption=float(profile["absorption"]),
                    scattering=float(profile["scattering"]),
                )
            )
            used_area += footprint_area
            break
    return obstacles


def generate_hybrid_rir(
    config: HybridRIRConfig,
    scene: Optional[HybridRIRScene] = None,
    low_backend: Optional[RIRBackend] = None,
    high_backend: Optional[RIRBackend] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scene = scene or sample_hybrid_rir_scene(config=config, seed=seed)
    low_backend = low_backend or GpuARDPytARDBackend()
    high_backend = high_backend or PyroomacousticsHighFrequencyBackend()

    low = low_backend.simulate(scene, config)
    high = high_backend.simulate(scene, config)
    rir = hybrid_crossover(low, high, config)
    metadata = {
        "config": _config_metadata(config),
        "scene": scene.to_metadata(),
        "obstacle_effects": obstacle_effects_metadata(scene, config),
        "bands": {
            "low": {
                "backend": low_backend.__class__.__name__,
                "frequency_hz": [config.low_fmin_hz, config.low_fmax_hz],
            },
            "high": {
                "backend": high_backend.__class__.__name__,
                "frequency_hz": [config.crossover_hz, config.sample_rate / 2.0],
            },
        },
    }
    return torch.as_tensor(rir, dtype=torch.float32), metadata


def write_hybrid_rir_dataset_item(
    output_dir: str | Path,
    sample_id: str,
    rir: torch.Tensor,
    metadata: dict[str, Any],
    sample_rate: int,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    wav_path = sample_dir / "rir_5ch.wav"
    json_path = sample_dir / "metadata.json"

    wav = rir.detach().cpu()
    if wav.ndim != 2 or wav.shape[0] != 5:
        raise ValueError(f"Expected RIR tensor [5, samples], got {tuple(wav.shape)}")
    torchaudio.save(str(wav_path), wav, int(sample_rate), encoding="PCM_F")
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return wav_path, json_path


def hybrid_crossover(
    rir_low: np.ndarray,
    rir_high: np.ndarray,
    config: HybridRIRConfig,
) -> np.ndarray:
    low = _coerce_rir_array(rir_low, config.num_sources, config.num_samples)
    high = _coerce_rir_array(rir_high, config.num_sources, config.num_samples)
    # Linkwitz-Riley (4th order = Butterworth applied twice), filtered causally
    # so the RIR stays causal (no pre-ringing before the direct path). The LR
    # low- and high-pass share an identical phase response, so the two
    # independently simulated bands stay time-aligned and sum to a flat
    # magnitude across the crossover.
    sos_lp = butter(
        2,
        float(config.crossover_hz),
        btype="lowpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    sos_hp = butter(
        2,
        float(config.crossover_hz),
        btype="highpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    low_band = sosfilt(sos_lp, sosfilt(sos_lp, low, axis=-1), axis=-1)
    high_band = sosfilt(sos_hp, sosfilt(sos_hp, high, axis=-1), axis=-1)
    if config.match_crossover_energy:
        low_band = _match_low_band_to_high_band(low_band, high_band, config)
    out = low_band + high_band
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9 and config.normalize_peak > 0:
        out = out / peak * float(config.normalize_peak)
    return out.astype(np.float32)


def _calibrate_pytard_signal(
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
        calibrated = _apply_rt60_decay_envelope(
            calibrated,
            sample_rate=int(sample_rate),
            rt60=float(rt60),
            origin_idx=direct_idx,
        )
    return calibrated


def _apply_rt60_decay_envelope(
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


def _match_low_band_to_high_band(
    low_band: np.ndarray,
    high_band: np.ndarray,
    config: HybridRIRConfig,
) -> np.ndarray:
    nyquist = float(config.sample_rate) / 2.0
    lo_hz, hi_hz = config.crossover_match_band_hz
    lo_hz = max(20.0, min(float(lo_hz), nyquist * 0.95))
    hi_hz = max(lo_hz + 1.0, min(float(hi_hz), nyquist * 0.99))
    if hi_hz <= lo_hz:
        return low_band

    sos_bp = butter(
        2,
        [lo_hz, hi_hz],
        btype="bandpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    filt = sosfiltfilt if low_band.shape[-1] > 64 else sosfilt
    low_ref = filt(sos_bp, low_band, axis=-1)
    high_ref = filt(sos_bp, high_band, axis=-1)
    low_rms = np.sqrt(np.mean(low_ref**2, axis=-1, keepdims=True))
    high_rms = np.sqrt(np.mean(high_ref**2, axis=-1, keepdims=True))
    target_ratio = 10.0 ** (float(config.crossover_match_target_db) / 20.0)
    raw_gain = high_rms * target_ratio / np.maximum(low_rms, 1e-12)
    gain_min, gain_max = config.crossover_match_gain_range
    gain = np.clip(raw_gain, float(gain_min), float(gain_max))
    return low_band * gain


def apply_obstacle_high_frequency_effects(
    rir: np.ndarray,
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> np.ndarray:
    """Approximate furniture occlusion/scattering on high-frequency RIRs.

    Pyroomacoustics is used for the room response.  Internal furniture geometry
    is applied here with deterministic geometric attenuation and scatter taps so
    the behavior is stable across Pyroomacoustics versions.
    """

    out = np.asarray(rir, dtype=np.float64).copy()
    for event in _obstacle_high_frequency_events(scene, config):
        source_idx = int(event["source_index"])
        out[source_idx] *= float(event["attenuation"])
        scatter_idx = int(event["scatter_index"])
        if 0 <= scatter_idx < out.shape[-1]:
            out[source_idx, scatter_idx] += float(event["scatter_amplitude"])
    return out.astype(np.float32)


def obstacle_effects_metadata(
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> dict[str, Any]:
    return {
        "obstacle_model": "high_frequency_post_occlusion_scatter",
        "low_band_obstacle_model": "none",
        "obstacle_count": len(scene.obstacles),
        "floor_coverage_ratio": _obstacle_floor_coverage(
            scene.obstacles, np.asarray(scene.room_dim, dtype=np.float64)
        ),
        "events": _obstacle_high_frequency_events(scene, config),
    }


def _obstacle_high_frequency_events(
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    mic = np.asarray(scene.mic_pos, dtype=np.float64)
    srcs = np.asarray(scene.source_pos, dtype=np.float64)
    fs = int(config.sample_rate)
    for source_idx, src in enumerate(srcs):
        for obstacle_idx, obstacle in enumerate(scene.obstacles):
            footprint = np.asarray(obstacle.footprint, dtype=np.float64)
            interval = _segment_polygon_crossing_interval(src[:2], mic[:2], footprint)
            if interval is None:
                continue
            # The 2D footprint is crossed; only treat it as an occluder if the
            # source-to-mic line actually passes through the obstacle's vertical
            # extent. A low table does not occlude a path at head height.
            t_mid = 0.5 * (interval[0] + interval[1])
            z_line = float(src[2] + t_mid * (mic[2] - src[2]))
            if not (float(obstacle.z_min) <= z_line <= float(obstacle.z_max)):
                continue
            attenuation = max(
                0.2,
                1.0 - 0.5 * float(obstacle.absorption) - 0.3 * float(obstacle.scattering),
            )
            scatter_point = obstacle.center
            scatter_distance = float(
                np.linalg.norm(src - scatter_point) + np.linalg.norm(scatter_point - mic)
            )
            scatter_idx = int(round(scatter_distance / config.sound_speed * fs))
            direct_distance = max(float(np.linalg.norm(src - mic)), 0.1)
            amp = (
                float(obstacle.scattering)
                * (1.0 - float(obstacle.absorption))
                * direct_distance
                / max(scatter_distance, 0.1)
            )
            events.append(
                {
                    "source_index": int(source_idx),
                    "source_label": (
                        scene.source_labels[source_idx]
                        if source_idx < len(scene.source_labels)
                        else str(source_idx)
                    ),
                    "obstacle_index": int(obstacle_idx),
                    "material": obstacle.material,
                    "z_line_m": z_line,
                    "attenuation": float(attenuation),
                    "attenuation_db": float(20.0 * math.log10(max(attenuation, 1e-12))),
                    "scatter_distance_m": scatter_distance,
                    "scatter_index": int(scatter_idx),
                    "scatter_delay_s": float(scatter_idx / max(fs, 1)),
                    "scatter_amplitude": float(0.08 * amp),
                }
            )
    return events


def _coerce_rir_array(rir: Any, num_sources: int, num_samples: int) -> np.ndarray:
    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 2 and arr.shape[0] != num_sources and arr.shape[1] == num_sources:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[0] != num_sources:
        raise ValueError(
            f"Expected RIR array [{num_sources}, samples], got shape {arr.shape}"
        )
    return _pad_or_trim([arr[idx] for idx in range(arr.shape[0])], num_samples)


def _pad_or_trim(rirs: list[np.ndarray], num_samples: int) -> np.ndarray:
    out = np.zeros((len(rirs), num_samples), dtype=np.float64)
    for idx, rir in enumerate(rirs):
        flat = np.asarray(rir, dtype=np.float64).reshape(-1)
        n = min(flat.shape[0], num_samples)
        out[idx, :n] = flat[:n]
    return out


def _config_metadata(config: HybridRIRConfig) -> dict[str, Any]:
    data = asdict(config)
    data["num_samples"] = config.num_samples
    data["num_sources"] = config.num_sources
    return data


def _default_pytard_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "pytARD"


class _DiscardingList:
    def append(self, _value: Any) -> None:
        return None


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


@contextmanager
def _nullcontext():
    yield


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


def _clip_position_to_room(point: ArrayLike, room_dim: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    eps = 1e-4
    return np.minimum(np.maximum(point, eps), room_dim - eps)


def _resample_ratio(target_fs: int, source_fs: int) -> tuple[int, int]:
    gcd = math.gcd(int(target_fs), int(source_fs))
    return int(target_fs // gcd), int(source_fs // gcd)


def _sample_point(
    room_dim: np.ndarray,
    margin: float,
    rng: np.random.Generator,
    height_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    point = rng.uniform(lower, upper)
    if height_range is not None:
        point[2] = _sample_height(room_dim, margin, height_range, rng)
    return point


def _sample_height(
    room_dim: np.ndarray,
    margin: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    lower = float(max(float(margin), min(height_range)))
    upper = float(min(float(room_dim[2]) - float(margin), max(height_range)))
    if upper <= lower:
        return 0.5 * (lower + upper)
    return float(rng.uniform(lower, upper))


def _sample_source_in_horizontal_shell(
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    min_dist: float,
    max_dist: float,
    margin: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Place a source whose 3D mic distance falls in [min_dist, max_dist].

    The distance constrained here is the same quantity written to the bank's
    ``channel_map["distance_m"]`` and consumed by training labels and the
    real-bank d0 split, so near/far membership is decided on the true
    source-receiver distance. (Historically only the floor projection was
    constrained, which let a "near" source at 0.35 m horizontal sit > 1 m away
    in 3D once the height offset was counted.)

    Per attempt: draw the target 3D distance, then a height whose vertical
    offset does not exceed it, then derive the horizontal radius. When room
    margins or the height ranges make the shell unreachable, fall back to the
    closest achievable placement toward the farthest corner (mirroring the
    documented room-size clamping of the far shell).
    """
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    z_lo = max(float(margin), float(min(height_range)))
    z_hi = max(z_lo, min(float(room_dim[2]) - float(margin), float(max(height_range))))
    mic_z = float(mic_pos[2])
    for _ in range(512):
        target = float(rng.uniform(min_dist, max_dist))
        # height compatible with the target 3D distance
        zc_lo = max(z_lo, mic_z - target)
        zc_hi = min(z_hi, mic_z + target)
        if zc_lo > zc_hi:
            continue
        z = float(rng.uniform(zc_lo, zc_hi))
        dz = z - mic_z
        radius = float(np.sqrt(max(target * target - dz * dz, 0.0)))
        direction = rng.normal(size=2)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        point = np.asarray(mic_pos, dtype=np.float64).copy()
        point[:2] = mic_pos[:2] + direction / norm * radius
        point[2] = z
        if np.all(point >= lower) and np.all(point <= upper):
            return point

    # Fallback: closest achievable 3D distance, walking toward the farthest corner.
    z = float(np.clip(mic_z, z_lo, z_hi))
    dz = z - mic_z
    corners = np.array(
        [
            [x, y]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
        ],
        dtype=np.float64,
    )
    spans = np.linalg.norm(corners - mic_pos[None, :2], axis=1)
    corner_xy = corners[int(np.argmax(spans))]
    span = max(float(spans.max()), 1e-9)
    target = float(np.clip(rng.uniform(min_dist, max_dist), abs(dz), None))
    radius = float(np.clip(np.sqrt(max(target * target - dz * dz, 0.0)), 0.0, span))
    point = np.asarray(mic_pos, dtype=np.float64).copy()
    point[:2] = mic_pos[:2] + (corner_xy - mic_pos[:2]) / span * radius
    point[2] = z
    return point


def _sample_source_in_shell(
    room_dim: np.ndarray,
    mic_pos: np.ndarray,
    min_dist: float,
    max_dist: float,
    margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return _sample_source_in_horizontal_shell(
        room_dim,
        mic_pos,
        min_dist,
        max_dist,
        margin,
        (float(margin), float(room_dim[2]) - float(margin)),
        rng,
    )


def _max_room_distance_from_point(
    room_dim: np.ndarray,
    point: np.ndarray,
    margin: float,
) -> float:
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(corners - point[None, :], axis=1).max())


def _max_room_horizontal_distance_from_point(
    room_dim: np.ndarray,
    point: np.ndarray,
    margin: float,
) -> float:
    lower = np.full(2, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim[:2] - float(margin), lower + 0.01)
    corners = np.array(
        [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])],
        dtype=np.float64,
    )
    return float(np.linalg.norm(corners - point[None, :2], axis=1).max())


def _sample_obstacle_count(
    room_dim: np.ndarray,
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> int:
    n_min, n_max = config.num_obstacles_range
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    density = float(rng.uniform(*config.obstacle_density_per_m2))
    target = int(round(floor_area * density))
    return int(np.clip(target, int(n_min), int(n_max)))


def _obstacle_material_profiles(room_dim: np.ndarray) -> dict[str, dict[str, Any]]:
    room_height = float(room_dim[2])
    return {
        "table": {
            "absorption": 0.18,
            "scattering": 0.30,
            "height_range": (0.65, 0.90),
            "radius_range": (0.25, 0.75),
            "placement": "free",
            "sample_weight": 1.0,
        },
        "sofa": {
            "absorption": 0.65,
            "scattering": 0.65,
            "height_range": (0.60, 1.10),
            "radius_range": (0.45, 1.05),
            "placement": "wall",
            "sample_weight": 0.7,
        },
        "chair": {
            "absorption": 0.35,
            "scattering": 0.45,
            "height_range": (0.45, 1.10),
            "radius_range": (0.20, 0.45),
            "placement": "free",
            "sample_weight": 1.2,
        },
        "curtain": {
            "absorption": 0.75,
            "scattering": 0.55,
            "height_range": (max(1.8, room_height * 0.75), room_height),
            "radius_range": (0.20, 0.55),
            "placement": "wall",
            "sample_weight": 0.45,
        },
        "cabinet": {
            "absorption": 0.25,
            "scattering": 0.50,
            "height_range": (0.80, min(2.0, room_height)),
            "radius_range": (0.35, 0.85),
            "placement": "wall",
            "sample_weight": 0.65,
        },
    }


def _sample_obstacle_center(
    room_dim: np.ndarray,
    radius: float,
    margin: float,
    placement: str,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = np.asarray([margin + radius, margin + radius], dtype=np.float64)
    upper = np.maximum(room_dim[:2] - margin - radius, lower + 0.01)
    center = rng.uniform(lower, upper)
    if placement == "wall":
        axis = int(rng.integers(0, 2))
        side = int(rng.integers(0, 2))
        center[axis] = lower[axis] if side == 0 else upper[axis]
    return center


def _sample_obstacle_height(
    room_dim: np.ndarray,
    profile: dict[str, Any],
    config: HybridRIRConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    lo = max(float(profile["height_range"][0]), float(config.obstacle_height_range[0]))
    hi = min(
        float(profile["height_range"][1]),
        float(config.obstacle_height_range[1]),
        float(room_dim[2]),
    )
    if hi <= lo:
        hi = max(lo, min(float(room_dim[2]), lo + 0.01))
    return 0.0, float(rng.uniform(lo, hi))


def _obstacle_floor_coverage(
    obstacles: list[PolygonObstacle],
    room_dim: np.ndarray,
) -> float:
    floor_area = max(float(room_dim[0] * room_dim[1]), 1e-6)
    area = sum(
        _polygon_area(np.asarray(obstacle.footprint, dtype=np.float64))
        for obstacle in obstacles
    )
    return float(area / floor_area)


def _obstacle_conflicts(
    footprint: np.ndarray,
    obstacles: list[PolygonObstacle],
    clearance: float,
) -> bool:
    for obstacle in obstacles:
        existing = np.asarray(obstacle.footprint, dtype=np.float64)
        if _polygons_overlap(footprint, existing):
            return True
        if _polygon_distance(footprint, existing) < float(clearance):
            return True
    return False


def _polygon_inside_room(
    polygon: np.ndarray,
    room_dim: np.ndarray,
    margin: float,
) -> bool:
    return bool(
        np.all(polygon[:, 0] >= margin)
        and np.all(polygon[:, 1] >= margin)
        and np.all(polygon[:, 0] <= room_dim[0] - margin)
        and np.all(polygon[:, 1] <= room_dim[1] - margin)
    )


def _point_in_polygon(point: ArrayLike, polygon: np.ndarray) -> bool:
    x, y = np.asarray(point, dtype=np.float64)[:2]
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _distance_point_to_polygon(point: ArrayLike, polygon: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64)[:2]
    distances = [
        _distance_point_to_segment(p, polygon[idx], polygon[(idx + 1) % polygon.shape[0]])
        for idx in range(polygon.shape[0])
    ]
    return float(min(distances))


def _polygon_area(polygon: np.ndarray) -> float:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if any(_point_in_polygon(point, b) for point in a):
        return True
    if any(_point_in_polygon(point, a) for point in b):
        return True
    for idx in range(a.shape[0]):
        a0 = a[idx]
        a1 = a[(idx + 1) % a.shape[0]]
        for jdx in range(b.shape[0]):
            if _segments_intersect(a0, a1, b[jdx], b[(jdx + 1) % b.shape[0]]):
                return True
    return False


def _polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    distances = []
    for idx in range(a.shape[0]):
        a0 = a[idx]
        a1 = a[(idx + 1) % a.shape[0]]
        distances.extend(_distance_point_to_segment(point, a0, a1) for point in b)
    for idx in range(b.shape[0]):
        b0 = b[idx]
        b1 = b[(idx + 1) % b.shape[0]]
        distances.extend(_distance_point_to_segment(point, b0, b1) for point in a)
    return float(min(distances)) if distances else 0.0


def _distance_point_to_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * seg)))


def _align_high_band_direct(
    rir: np.ndarray,
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> np.ndarray:
    """Shift the geometric RIR so the direct path lands at ``distance / c``.

    Pyroomacoustics offsets every RIR by a constant fractional-delay length, so
    its direct path arrives later than the true geometric time. The low-frequency
    wave band carries the physical propagation delay from injection, so removing
    this constant offset keeps the two bands time-aligned at the direct path.
    """
    rir = np.asarray(rir, dtype=np.float64)
    mic = np.asarray(scene.mic_pos, dtype=np.float64)
    srcs = np.asarray(scene.source_pos, dtype=np.float64)
    fs = int(config.sample_rate)
    c = max(float(config.sound_speed), 1e-6)
    n = rir.shape[-1]
    shifts: list[int] = []
    for idx in range(min(rir.shape[0], srcs.shape[0])):
        distance = float(np.linalg.norm(srcs[idx] - mic))
        expected = int(round(distance / c * fs))
        lo = max(0, expected - 16)
        hi = min(n, expected + 160)
        if hi <= lo:
            continue
        channel = np.abs(rir[idx])
        peak = float(channel.max())
        if peak <= 0.0:
            continue
        seg = channel[lo:hi]
        threshold = 0.3 * peak
        crossing = int(np.argmax(seg >= threshold))
        if seg[crossing] < threshold:
            continue
        shifts.append((lo + crossing) - expected)
    if not shifts:
        return rir
    shift = int(round(float(np.median(shifts))))
    if shift <= 0:
        return rir
    out = np.zeros_like(rir)
    out[:, : n - shift] = rir[:, shift:]
    return out


def _segment_segment_t(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> Optional[float]:
    """Parameter ``t`` in ``[0, 1]`` where segment ``a->b`` crosses ``c->d``."""
    r = b - a
    s = d - c
    rxs = float(r[0] * s[1] - r[1] * s[0])
    if abs(rxs) < 1e-12:
        return None
    qp = c - a
    t = float(qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = float(qp[0] * r[1] - qp[1] * r[0]) / rxs
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return float(min(1.0, max(0.0, t)))
    return None


def _segment_polygon_crossing_interval(
    start: ArrayLike,
    end: ArrayLike,
    polygon: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Range of ``t`` along ``start->end`` that lies inside ``polygon`` (XY)."""
    a = np.asarray(start, dtype=np.float64)[:2]
    b = np.asarray(end, dtype=np.float64)[:2]
    ts: list[float] = []
    if _point_in_polygon(a, polygon):
        ts.append(0.0)
    if _point_in_polygon(b, polygon):
        ts.append(1.0)
    for idx in range(polygon.shape[0]):
        t = _segment_segment_t(a, b, polygon[idx], polygon[(idx + 1) % polygon.shape[0]])
        if t is not None:
            ts.append(t)
    if not ts:
        return None
    return (min(ts), max(ts))


def _segment_intersects_polygon(
    start: ArrayLike,
    end: ArrayLike,
    polygon: np.ndarray,
) -> bool:
    start = np.asarray(start, dtype=np.float64)[:2]
    end = np.asarray(end, dtype=np.float64)[:2]
    if _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon):
        return True
    for idx in range(polygon.shape[0]):
        if _segments_intersect(start, end, polygon[idx], polygon[(idx + 1) % polygon.shape[0]]):
            return True
    return False


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-12
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False
