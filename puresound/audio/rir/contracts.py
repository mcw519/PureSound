"""Frozen R0 contracts for the RIR domain.

This module is the bottom layer of ``puresound.audio.rir``.  It declares the
data conventions that every RIR producer and consumer already follows, so that
later migration stages can move code without silently changing behaviour.

Nothing here invents a new convention.  Every constant below was read off the
current implementation:

- backends return ``np.ndarray`` shaped ``[num_sources, samples]``
  (``hybrid_rir.RIRBackend``);
- the whole DSP path stays in NumPy, and there is exactly one NumPy -> Torch
  conversion, at the end of ``generate_hybrid_rir``
  (``torch.as_tensor(rir, dtype=torch.float32)``);
- dataset items are written by ``torchaudio.save(..., encoding="PCM_F")``,
  i.e. 32-bit float WAV.

Import rules
------------

This module may import only the standard library and NumPy.  It must not
import ``torch``, ``torchaudio``, Pyroomacoustics, any renderer, any bank
module, or touch the filesystem.  ``test/test_rir_r0_import_boundaries.py``
enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


RIR_CONTRACT_VERSION = "puresound.rir.contracts.v1"

#: Axis order of every in-memory RIR array in this codebase.
RIR_AXIS_ORDER: tuple[str, str] = ("channel", "sample")

#: dtype used throughout the NumPy physics/DSP layer.
RIR_COMPUTE_DTYPE = np.float64

#: dtype of the delivered tensor and of the on-disk WAV payload.
RIR_DELIVERY_DTYPE = np.float32

#: libsndfile subtype written by ``torchaudio.save(..., encoding="PCM_F")``.
RIR_WAV_SUBTYPE = "FLOAT"

#: Top-level keys every generated dataset-item metadata JSON carries.
RIR_METADATA_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "bands",
        "config",
        "crossover",
        "output_calibration",
        "rir_filename",
        "metadata_filename",
        "rir_index",
        "room_id",
        "room_index",
        "sample_id",
        "scene",
    }
)

#: Keys every ``scene.channel_map`` entry carries.
RIR_METADATA_CHANNEL_MAP_KEYS: frozenset[str] = frozenset(
    {
        "channel",
        "label",
        "source_pos",
        "distance_m",
        "horizontal_distance_m",
    }
)


class BackendBand(str, Enum):
    """Frequency region a backend is responsible for."""

    LOW = "low"
    HIGH = "high"
    FULL = "full"


@dataclass(frozen=True)
class RIRArray:
    """A rendered RIR together with its layout and rate.

    The payload is always ``[channels, samples]``.  Construction validates the
    layout so that a transposed array cannot travel silently through the
    pipeline.

    Parameters
    ----------
    samples:
        Real-valued array shaped ``[channels, samples]``.
    sample_rate:
        Sampling rate in Hz; must be positive.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        array = np.asarray(self.samples)
        if array.ndim != 2:
            raise ValueError(
                f"RIR must be 2-D {RIR_AXIS_ORDER}, got ndim={array.ndim}"
            )
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"RIR must be non-empty, got shape={array.shape}")
        if np.iscomplexobj(array):
            raise ValueError("RIR must be real-valued")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        object.__setattr__(self, "samples", array)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def num_channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.num_samples) / float(self.sample_rate)

    def is_finite(self) -> bool:
        """True when no channel contains NaN or Inf."""

        return bool(np.isfinite(self.samples).all())

    def as_compute(self) -> "RIRArray":
        """Return this RIR in the NumPy compute dtype."""

        return RIRArray(
            self.samples.astype(RIR_COMPUTE_DTYPE, copy=False), self.sample_rate
        )

    def as_delivery(self) -> "RIRArray":
        """Return this RIR in the delivered/on-disk dtype."""

        return RIRArray(
            self.samples.astype(RIR_DELIVERY_DTYPE, copy=False), self.sample_rate
        )

    def first_nonzero_sample(self, channel: int, tolerance: float = 0.0) -> int | None:
        """Index of the first sample exceeding ``tolerance``, or ``None``."""

        magnitude = np.abs(self.samples[int(channel)])
        hits = np.flatnonzero(magnitude > float(tolerance))
        return int(hits[0]) if hits.size else None

    def violates_causality(
        self,
        first_physical_samples: Sequence[int],
        *,
        tolerance: float = 0.0,
    ) -> tuple[int, ...]:
        """Return channels carrying energy before their physical arrival.

        ``first_physical_samples[c]`` is ``floor(distance / c * fs)`` for
        channel ``c`` — the project-wide causality boundary, where the arrival
        sample itself is preserved and everything strictly before it must be
        zero.
        """

        if len(first_physical_samples) != self.num_channels:
            raise ValueError(
                "first_physical_samples must have one entry per channel: "
                f"got {len(first_physical_samples)} for {self.num_channels} channels"
            )
        offenders: list[int] = []
        for channel, boundary in enumerate(first_physical_samples):
            edge = max(0, int(boundary))
            if edge == 0:
                continue
            window = np.abs(self.samples[channel, :edge])
            if window.size and float(window.max()) > float(tolerance):
                offenders.append(channel)
        return tuple(offenders)


@dataclass(frozen=True)
class RenderContext:
    """The rendering parameters a backend needs, decoupled from the CLI config.

    ``HybridRIRConfig`` carries scene-sampling knobs as well as rendering
    knobs.  A backend only ever needs the latter; this context is the subset,
    so backends can be constructed and tested without the sampler.
    """

    sample_rate: int
    num_samples: int
    sound_speed: float
    crossover_hz: float
    low_fmin_hz: float
    low_fmax_hz: float

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if int(self.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if float(self.sound_speed) <= 0.0:
            raise ValueError("sound_speed must be positive")
        nyquist = 0.5 * float(self.sample_rate)
        if not 0.0 < float(self.crossover_hz) < nyquist:
            raise ValueError(
                f"crossover_hz must lie in (0, {nyquist}), got {self.crossover_hz}"
            )
        if not 0.0 <= float(self.low_fmin_hz) < float(self.low_fmax_hz):
            raise ValueError("require 0 <= low_fmin_hz < low_fmax_hz")

    @property
    def duration_s(self) -> float:
        return float(self.num_samples) / float(self.sample_rate)

    @property
    def nyquist_hz(self) -> float:
        return 0.5 * float(self.sample_rate)

    @classmethod
    def from_config(cls, config: Any) -> "RenderContext":
        """Build a context from any object exposing the HybridRIRConfig fields."""

        return cls(
            sample_rate=int(config.sample_rate),
            num_samples=int(config.num_samples),
            sound_speed=float(config.sound_speed),
            crossover_hz=float(config.crossover_hz),
            low_fmin_hz=float(config.low_fmin_hz),
            low_fmax_hz=float(config.low_fmax_hz),
        )

    def first_physical_sample(self, distance_m: float) -> int:
        """``floor(distance / c * fs)`` — the causality boundary for a source."""

        if float(distance_m) < 0.0:
            raise ValueError("distance_m must be non-negative")
        return int(
            np.floor(float(distance_m) / float(self.sound_speed) * float(self.sample_rate))
        )


@dataclass(frozen=True)
class BackendCapabilities:
    """What a renderer backend can promise.

    ``deterministic_for_fixed_seed`` is deliberately explicit.  A backend that
    draws from a random stream the generator does not seed cannot promise
    byte-identical output for a fixed seed, and a bank built from it is not
    reproducible no matter how carefully the manifest is hashed.  Declaring the
    property here makes it testable instead of assumed.
    """

    backend_id: str
    band: BackendBand
    deterministic_for_fixed_seed: bool
    source_convention: str
    optional_dependencies: tuple[str, ...] = ()
    supports_scene_v2: bool = True
    notes: str = ""
    #: Free-form flags for capabilities that are not yet contract-level.
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.backend_id).strip():
            raise ValueError("backend_id must be a non-empty string")
        if not isinstance(self.band, BackendBand):
            object.__setattr__(self, "band", BackendBand(self.band))
        if not str(self.source_convention).strip():
            raise ValueError("source_convention must be a non-empty string")
        object.__setattr__(
            self, "optional_dependencies", tuple(self.optional_dependencies)
        )

    def missing_dependencies(self) -> tuple[str, ...]:
        """Optional dependencies that cannot currently be imported."""

        import importlib.util

        missing: list[str] = []
        for name in self.optional_dependencies:
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.append(name)
        return tuple(missing)

    @property
    def is_available(self) -> bool:
        return not self.missing_dependencies()


@dataclass
class HybridRIRConfig:
    """Configuration for the hybrid generator.

    This lives at layer 0 rather than beside the renderer because scene
    sampling consumes it, and layer 2 (``scene``) may not import layer 3
    (``render``).  It is plain data — no NumPy, no backend, no filesystem —
    so the dependency stays harmless.

    The fields mix two concerns that a later stage may want to separate:
    scene-sampling knobs (room/obstacle/distance ranges, margins) and
    rendering knobs (rates, crossover, level policy).  ``RenderContext``
    already isolates the rendering subset a backend actually needs.

    Not frozen: existing code relies on ``dataclasses.replace`` and on
    mutating instances.
    """

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
    obstacle_occlusion_recovery_ms: float = 80.0
    tail_fade_ms: float = 20.0
    normalize_peak: float = 0.98
    output_mode: str = "peak_normalized"
    calibrated_reference_source_spl_db: float = 94.0
    record_realized_metrics: bool = False
    match_crossover_energy: bool = True
    crossover_match_band_hz: Optional[tuple[float, float]] = None
    crossover_match_target_db: float = 0.0
    #: Bounds on the low-band crossover match gain.
    #:
    #: The upper bound was 2.0 and that was below the median requirement: the
    #: pytARD low band is peak-normalized (``calibrate_pytard_signal`` divides
    #: out the solver's own amplitude and re-imposes 1/r by hand), so this match
    #: is the only thing that gives it a level relative to the high band.
    #: Measured over 24 scenes x 5 sources on both high backends, the required
    #: gain runs 0.76-5.27 with a median of 2.0-2.4, so a 2.0 ceiling clipped
    #: 53-68% of channels and left their low band under-level by an unrecorded
    #: amount.  8.0 covers the observed maximum with headroom and still bounds a
    #: degenerate near-silent low band.  Saturation is now reported in the
    #: crossover metadata rather than being silent.
    crossover_match_gain_range: tuple[float, float] = (1e-4, 8.0)
    preserve_source_convention_at_crossover: bool = True

    @property
    def num_sources(self) -> int:
        return int(self.num_near_sources + self.num_far_sources)

    @property
    def num_samples(self) -> int:
        return int(round(float(self.sample_rate) * float(self.duration)))


def resolve_sound_speed(metadata: Mapping[str, Any]) -> float:
    """Return the sound speed that defines this item's causality boundary.

    The generator clips each low-band channel before
    ``floor(distance / c * fs)`` using the *scene environment* sound speed,
    which is derived from temperature and humidity — not the
    ``HybridRIRConfig.sound_speed`` default that also appears in the metadata.

    The two differ in practice.  A real generated item measured during the R0
    freeze had ``config.sound_speed = 343.0`` but
    ``scene.environment.sound_speed_m_s = 344.3619``; for a 2.3172 m source
    that is boundary 107 versus 108, so a consumer using the config value
    reads the genuine direct arrival as pre-arrival energy.

    Precedence is therefore ``scene.environment.sound_speed_m_s`` first,
    ``config.sound_speed`` only as a fallback.

    Raises
    ------
    ValueError
        If neither field supplies a positive, finite sound speed.
    """

    scene = metadata.get("scene")
    if isinstance(scene, Mapping):
        environment = scene.get("environment")
        if isinstance(environment, Mapping):
            candidate = environment.get("sound_speed_m_s")
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                value = float(candidate)
                if np.isfinite(value) and value > 0.0:
                    return value

    config = metadata.get("config")
    if isinstance(config, Mapping):
        candidate = config.get("sound_speed")
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            value = float(candidate)
            if np.isfinite(value) and value > 0.0:
                return value

    raise ValueError(
        "metadata supplies no positive sound speed in "
        "scene.environment.sound_speed_m_s or config.sound_speed"
    )


def validate_rir_metadata(
    metadata: Mapping[str, Any],
    *,
    required_keys: Iterable[str] = RIR_METADATA_REQUIRED_KEYS,
) -> tuple[str, ...]:
    """Return the contract violations in a dataset-item metadata mapping.

    An empty tuple means the mapping satisfies the frozen R0 contract.  This
    reports rather than raises so callers can aggregate over a whole bank.
    """

    problems: list[str] = []
    missing = sorted(set(required_keys) - set(metadata))
    if missing:
        problems.append(f"missing top-level keys: {missing}")

    scene = metadata.get("scene")
    if not isinstance(scene, Mapping):
        problems.append("scene must be a mapping")
        return tuple(problems)

    channel_map = scene.get("channel_map")
    if not isinstance(channel_map, Sequence) or isinstance(channel_map, (str, bytes)):
        problems.append("scene.channel_map must be a sequence")
        return tuple(problems)

    seen_channels: set[int] = set()
    for position, entry in enumerate(channel_map):
        if not isinstance(entry, Mapping):
            problems.append(f"channel_map[{position}] must be a mapping")
            continue
        entry_missing = sorted(RIR_METADATA_CHANNEL_MAP_KEYS - set(entry))
        if entry_missing:
            problems.append(f"channel_map[{position}] missing keys: {entry_missing}")
            continue
        channel = entry["channel"]
        if not isinstance(channel, int) or isinstance(channel, bool):
            problems.append(f"channel_map[{position}].channel must be an int")
            continue
        if channel in seen_channels:
            problems.append(f"channel_map has duplicate channel index {channel}")
        seen_channels.add(channel)
        distance = entry["distance_m"]
        if not isinstance(distance, (int, float)) or isinstance(distance, bool):
            problems.append(f"channel_map[{position}].distance_m must be numeric")
        elif not np.isfinite(float(distance)) or float(distance) < 0.0:
            problems.append(
                f"channel_map[{position}].distance_m must be finite and non-negative"
            )
    return tuple(problems)


__all__ = [
    "RIR_AXIS_ORDER",
    "RIR_COMPUTE_DTYPE",
    "RIR_CONTRACT_VERSION",
    "RIR_DELIVERY_DTYPE",
    "RIR_METADATA_CHANNEL_MAP_KEYS",
    "RIR_METADATA_REQUIRED_KEYS",
    "RIR_WAV_SUBTYPE",
    "BackendBand",
    "BackendCapabilities",
    "HybridRIRConfig",
    "RenderContext",
    "RIRArray",
    "resolve_sound_speed",
    "validate_rir_metadata",
]
