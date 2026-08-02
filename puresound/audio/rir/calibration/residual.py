"""Causal, decay-constrained shared residual reference model for M5.5."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from puresound.audio.rir_calibration import (
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
)


M5_CONSTRAINED_RESIDUAL_POLICY = "puresound.m5_constrained_residual.v1"


@dataclass(frozen=True)
class CausalDecayResidualModel:
    """Direct-relative residual template normalized by physical-tail norm."""

    sample_rate: int
    normalized_template: np.ndarray
    decay_start_sample: int
    decay_block_samples: int
    maximum_rt60_s: float
    maximum_residual_to_physical_energy_ratio: float
    training_observation_count: int

    def __post_init__(self) -> None:
        template = np.asarray(self.normalized_template, dtype=np.float64).reshape(-1)
        if (
            self.sample_rate <= 0
            or template.size < 16
            or not np.all(np.isfinite(template))
            or not 0 <= self.decay_start_sample < template.size
            or self.decay_block_samples < 1
            or not math.isfinite(self.maximum_rt60_s)
            or self.maximum_rt60_s <= 0.0
            or not math.isfinite(self.maximum_residual_to_physical_energy_ratio)
            or not 0.0 < self.maximum_residual_to_physical_energy_ratio <= 1.0
            or self.training_observation_count < 1
        ):
            raise ValueError("constrained residual model configuration is invalid")
        object.__setattr__(self, "normalized_template", template)

    def residual_for(self, physical_rir: Any, direct_sample: int) -> np.ndarray:
        physical = np.asarray(physical_rir, dtype=np.float64).reshape(-1)
        direct = int(direct_sample)
        if (
            physical.size < 16
            or not np.all(np.isfinite(physical))
            or not 0 <= direct < physical.size
        ):
            raise ValueError("physical RIR/direct sample is invalid")
        output = np.zeros_like(physical)
        available = min(self.normalized_template.size, physical.size - direct)
        scale = max(
            float(np.linalg.norm(physical[direct:])),
            np.finfo(np.float64).tiny,
        )
        output[direct : direct + available] = (
            scale * self.normalized_template[:available]
        )
        return output

    def apply(self, physical_rir: Any, direct_sample: int) -> np.ndarray:
        physical = np.asarray(physical_rir, dtype=np.float64).reshape(-1)
        return physical + self.residual_for(physical, direct_sample)

    def to_dict(self) -> dict[str, Any]:
        digest = hashlib.sha256(
            self.normalized_template.astype("<f8", copy=False).tobytes()
        ).hexdigest()
        return {
            "policy": M5_CONSTRAINED_RESIDUAL_POLICY,
            "sample_rate": int(self.sample_rate),
            "template_sample_count": int(self.normalized_template.size),
            "template_sha256_float64_le": digest,
            "decay_start_sample": int(self.decay_start_sample),
            "decay_start_ms": float(
                self.decay_start_sample / self.sample_rate * 1000.0
            ),
            "decay_block_samples": int(self.decay_block_samples),
            "maximum_rt60_s": float(self.maximum_rt60_s),
            "maximum_residual_to_physical_energy_ratio": float(
                self.maximum_residual_to_physical_energy_ratio
            ),
            "realized_normalized_template_energy": float(
                np.dot(self.normalized_template, self.normalized_template)
            ),
            "training_observation_count": int(self.training_observation_count),
            "direct_relative_causal": True,
        }


@dataclass(frozen=True)
class ConstrainedResidualFit:
    model: CausalDecayResidualModel
    raw_normalized_template_energy: float
    constrained_normalized_template_energy: float
    decay_blocks_scaled: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": M5_CONSTRAINED_RESIDUAL_POLICY,
            "model": self.model.to_dict(),
            "raw_normalized_template_energy": float(
                self.raw_normalized_template_energy
            ),
            "constrained_normalized_template_energy": float(
                self.constrained_normalized_template_energy
            ),
            "decay_blocks_scaled": int(self.decay_blocks_scaled),
        }


def _constrain_decay(
    template: np.ndarray,
    sample_rate: int,
    decay_start: int,
    block: int,
    maximum_rt60_s: float,
) -> tuple[np.ndarray, int]:
    result = template.copy()
    previous_rms: float | None = None
    scaled = 0
    decay_per_block = 10.0 ** (-3.0 * (block / sample_rate) / maximum_rt60_s)
    for start in range(decay_start, result.size, block):
        stop = min(result.size, start + block)
        values = result[start:stop]
        rms = math.sqrt(float(np.mean(np.square(values))))
        if previous_rms is None:
            previous_rms = rms
            continue
        ceiling = previous_rms * decay_per_block
        if rms > ceiling and rms > np.finfo(np.float64).tiny:
            values *= ceiling / rms
            rms = ceiling
            scaled += 1
        previous_rms = rms
    return result, scaled


def fit_causal_decay_residual(
    targets: Sequence[Any],
    physical_rirs: Sequence[Any],
    direct_samples: Sequence[int],
    sample_rate: int,
    *,
    decay_start_ms: float = 50.0,
    decay_block_ms: float = 10.0,
    maximum_rt60_s: float = 0.8,
    maximum_residual_to_physical_energy_ratio: float = 0.25,
) -> ConstrainedResidualFit:
    """Robustly learn one shared systematic residual from training positions."""

    if (
        not targets
        or len(targets) != len(physical_rirs)
        or len(targets) != len(direct_samples)
        or sample_rate <= 0
    ):
        raise ValueError("residual training sequences must have equal non-zero size")
    pairs = []
    tail_lengths = []
    for target_value, physical_value, direct_value in zip(
        targets,
        physical_rirs,
        direct_samples,
    ):
        target = np.asarray(target_value, dtype=np.float64).reshape(-1)
        physical = np.asarray(physical_value, dtype=np.float64).reshape(-1)
        direct = int(direct_value)
        if (
            target.shape != physical.shape
            or target.size < 16
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(physical))
            or not 0 <= direct < target.size
        ):
            raise ValueError("residual training RIR/direct sample is invalid")
        scale = max(
            float(np.linalg.norm(physical[direct:])),
            np.finfo(np.float64).tiny,
        )
        pairs.append((target - physical)[direct:] / scale)
        tail_lengths.append(target.size - direct)
    length = min(tail_lengths)
    stacked = np.vstack([value[:length] for value in pairs])
    raw = np.median(stacked, axis=0)
    raw_energy = float(np.dot(raw, raw))
    if (
        not math.isfinite(decay_start_ms)
        or decay_start_ms < 0.0
        or not math.isfinite(decay_block_ms)
        or decay_block_ms <= 0.0
        or not math.isfinite(maximum_rt60_s)
        or maximum_rt60_s <= 0.0
        or not math.isfinite(maximum_residual_to_physical_energy_ratio)
        or not 0.0 < maximum_residual_to_physical_energy_ratio <= 1.0
    ):
        raise ValueError("residual constraint values are invalid")
    decay_start = min(
        length - 1,
        int(round(decay_start_ms * 1e-3 * sample_rate)),
    )
    block = max(1, int(round(decay_block_ms * 1e-3 * sample_rate)))
    constrained, scaled_blocks = _constrain_decay(
        raw,
        sample_rate,
        decay_start,
        block,
        maximum_rt60_s,
    )
    constrained_energy = float(np.dot(constrained, constrained))
    if constrained_energy > maximum_residual_to_physical_energy_ratio:
        constrained *= math.sqrt(
            maximum_residual_to_physical_energy_ratio / constrained_energy
        )
        constrained_energy = float(np.dot(constrained, constrained))
    model = CausalDecayResidualModel(
        sample_rate=int(sample_rate),
        normalized_template=constrained,
        decay_start_sample=decay_start,
        decay_block_samples=block,
        maximum_rt60_s=float(maximum_rt60_s),
        maximum_residual_to_physical_energy_ratio=float(
            maximum_residual_to_physical_energy_ratio
        ),
        training_observation_count=len(targets),
    )
    return ConstrainedResidualFit(
        model=model,
        raw_normalized_template_energy=raw_energy,
        constrained_normalized_template_energy=constrained_energy,
        decay_blocks_scaled=scaled_blocks,
    )


def evaluate_residual_ablation(
    targets: Sequence[Any],
    physical_rirs: Sequence[Any],
    direct_samples: Sequence[int],
    sample_rate: int,
    model: CausalDecayResidualModel,
    *,
    physical_first_samples: Sequence[int] | None = None,
    octave_centers_hz: Sequence[float] = (500.0, 1000.0, 2000.0),
) -> dict[str, Any]:
    """Evaluate physical-only, residual-only, and combined with one oracle."""

    targets_array = np.vstack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in targets]
    )
    physical_array = np.vstack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in physical_rirs]
    )
    if targets_array.shape != physical_array.shape:
        raise ValueError("residual ablation target/physical shapes must match")
    residual_array = np.vstack(
        [
            model.residual_for(physical, direct)
            for physical, direct in zip(physical_array, direct_samples)
        ]
    )
    combined_array = physical_array + residual_array
    weights = CalibrationLossWeights(spatial_coherence=0.0)

    def analyze(candidate: np.ndarray) -> dict[str, Any]:
        return analyze_rir_calibration_loss(
            targets_array,
            candidate,
            sample_rate,
            measured_direct_samples=direct_samples,
            synthetic_direct_samples=direct_samples,
            physical_first_samples=physical_first_samples,
            weights=weights,
            fft_sizes=(256, 512),
            octave_centers_hz=octave_centers_hz,
        ).to_dict()

    return {
        "policy": M5_CONSTRAINED_RESIDUAL_POLICY,
        "physical_only": analyze(physical_array),
        "residual_only": analyze(residual_array),
        "combined": analyze(combined_array),
        "residual_rirs": residual_array,
        "combined_rirs": combined_array,
    }


__all__ = [
    "M5_CONSTRAINED_RESIDUAL_POLICY",
    "CausalDecayResidualModel",
    "ConstrainedResidualFit",
    "evaluate_residual_ablation",
    "fit_causal_decay_residual",
]
