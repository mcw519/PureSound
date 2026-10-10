"""Fixed-pole calibration for separable impedance-room modal residues.

The nonlinear impedance eigenproblem determines the pole locations and complex
spatial eigenfunctions.  This module deliberately keeps those quantities fixed
while fitting one transferable complex residue scale and a smooth frequency
power law against independent reference responses.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from puresound.audio.rir.physics.wave.source_convention import (
    FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
    PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM,
)


IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION = (
    "puresound.impedance_modal_residue_calibration.v2"
)
LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION = (
    "puresound.impedance_modal_residue_calibration.v1"
)
IMPEDANCE_MODAL_RESIDUE_REPORT_SCHEMA_VERSION = (
    "puresound.impedance_modal_residue_calibration_report.v1"
)


def _encoded_complex(value: complex) -> dict[str, float]:
    return {
        "real": float(complex(value).real),
        "imag": float(complex(value).imag),
    }


def _decoded_complex(value: Any) -> complex:
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError("complex value must contain exactly real and imag")
    return complex(float(value["real"]), float(value["imag"]))


@dataclass(frozen=True)
class ImpedanceModalResidueCalibration:
    """Portable complex residue correction for one impedance boundary model."""

    reference_id: str
    boundary_reference_id: str
    complex_scale: complex
    frequency_exponent: float
    reference_frequency_hz: float
    valid_frequency_range_hz: tuple[float, float]
    source: dict[str, Any]
    applicability: dict[str, Any]
    fit_summary: dict[str, Any]
    fitted_source_convention: str = FDTD_PRESSURE_CELL_SOURCE_CONVENTION
    rendered_rir_convention: str = FREE_FIELD_1_OVER_R_RIR_CONVENTION
    residue_transform: str = PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM

    def __post_init__(self) -> None:
        if not self.reference_id or not self.boundary_reference_id:
            raise ValueError("residue and boundary reference ids are required")
        if not (
            math.isfinite(complex(self.complex_scale).real)
            and math.isfinite(complex(self.complex_scale).imag)
            and abs(complex(self.complex_scale)) > 0.0
        ):
            raise ValueError("complex residue scale must be finite and non-zero")
        if not math.isfinite(float(self.frequency_exponent)):
            raise ValueError("frequency exponent must be finite")
        if (
            not math.isfinite(float(self.reference_frequency_hz))
            or float(self.reference_frequency_hz) <= 0.0
        ):
            raise ValueError("reference frequency must be finite and positive")
        valid_minimum, valid_maximum = (
            float(value) for value in self.valid_frequency_range_hz
        )
        if not 0.0 < valid_minimum < valid_maximum:
            raise ValueError("valid frequency range must be positive and ordered")
        if not isinstance(self.source, dict) or not self.source.get(
            "evidence_tier"
        ):
            raise ValueError("residue calibration source.evidence_tier is required")
        if not isinstance(self.applicability, dict) or not self.applicability.get(
            "scope"
        ):
            raise ValueError("residue calibration applicability.scope is required")
        if not isinstance(self.fit_summary, dict):
            raise ValueError("residue calibration fit_summary must be an object")
        if (
            self.fitted_source_convention
            != FDTD_PRESSURE_CELL_SOURCE_CONVENTION
        ):
            raise ValueError("unsupported fitted modal-residue source convention")
        if self.rendered_rir_convention != FREE_FIELD_1_OVER_R_RIR_CONVENTION:
            raise ValueError("unsupported rendered RIR source convention")
        if (
            self.residue_transform
            != PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM
        ):
            raise ValueError("unsupported modal-residue source transform")

    def frequency_weight(self, frequency_hz: float) -> float:
        frequency = float(frequency_hz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("mode frequency must be finite and positive")
        return float(
            (float(self.reference_frequency_hz) / frequency)
            ** float(self.frequency_exponent)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
            "reference_id": self.reference_id,
            "boundary_reference_id": self.boundary_reference_id,
            "complex_scale": _encoded_complex(self.complex_scale),
            "frequency_exponent": float(self.frequency_exponent),
            "reference_frequency_hz": float(self.reference_frequency_hz),
            "valid_frequency_range_hz": [
                float(value) for value in self.valid_frequency_range_hz
            ],
            "source": dict(self.source),
            "applicability": dict(self.applicability),
            "fit_summary": dict(self.fit_summary),
            "fitted_source_convention": self.fitted_source_convention,
            "rendered_rir_convention": self.rendered_rir_convention,
            "residue_transform": self.residue_transform,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.metadata(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "ImpedanceModalResidueCalibration":
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = metadata.get("schema_version")
        if schema_version not in {
            IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
            LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported impedance modal residue schema")
        return cls(
            reference_id=str(metadata.get("reference_id", "")),
            boundary_reference_id=str(
                metadata.get("boundary_reference_id", "")
            ),
            complex_scale=_decoded_complex(metadata.get("complex_scale")),
            frequency_exponent=float(metadata["frequency_exponent"]),
            reference_frequency_hz=float(metadata["reference_frequency_hz"]),
            valid_frequency_range_hz=tuple(
                float(value)
                for value in metadata["valid_frequency_range_hz"]
            ),
            source=dict(metadata.get("source", {})),
            applicability=dict(metadata.get("applicability", {})),
            fit_summary=dict(metadata.get("fit_summary", {})),
            fitted_source_convention=str(
                metadata.get(
                    "fitted_source_convention",
                    FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
                )
            ),
            rendered_rir_convention=str(
                metadata.get(
                    "rendered_rir_convention",
                    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
                )
            ),
            residue_transform=str(
                metadata.get(
                    "residue_transform",
                    PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM,
                )
            ),
        )


@dataclass(frozen=True)
class FixedPoleResidueCase:
    """Reference response plus per-mode quadrature bases for one position pair."""

    case_id: str
    split: str
    mode_frequencies_hz: np.ndarray
    real_mode_basis: np.ndarray
    imaginary_mode_basis: np.ndarray
    target: np.ndarray
    analysis_mask: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.mode_frequencies_hz, dtype=np.float64)
        real_basis = np.asarray(self.real_mode_basis, dtype=np.float64)
        imaginary_basis = np.asarray(
            self.imaginary_mode_basis,
            dtype=np.float64,
        )
        target = np.asarray(self.target, dtype=np.float64)
        mask = np.asarray(self.analysis_mask, dtype=bool)
        if not self.case_id:
            raise ValueError("residue case id is required")
        if self.split not in {
            "train",
            "holdout",
            "position_holdout",
            "room_holdout",
            "grid_holdout",
            "boundary_holdout",
        }:
            raise ValueError("unsupported fixed-pole residue case split")
        if frequencies.ndim != 1 or frequencies.size < 1:
            raise ValueError("residue case requires at least one mode")
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("mode frequencies must be finite and positive")
        if (
            real_basis.shape != imaginary_basis.shape
            or real_basis.ndim != 2
            or real_basis.shape[0] != frequencies.size
        ):
            raise ValueError("mode bases must have shape [modes, samples]")
        if target.ndim != 1 or real_basis.shape[1] != target.size:
            raise ValueError("target and mode basis sample counts must match")
        if mask.shape != target.shape or int(np.count_nonzero(mask)) < 3:
            raise ValueError("analysis mask must select at least three samples")
        if not (
            np.all(np.isfinite(real_basis))
            and np.all(np.isfinite(imaginary_basis))
            and np.all(np.isfinite(target))
        ):
            raise ValueError("residue case arrays must be finite")
        if float(np.sqrt(np.mean(target[mask] ** 2))) <= 1e-15:
            raise ValueError("residue case target is silent in analysis window")

    def design(self, exponent: float, reference_frequency_hz: float) -> np.ndarray:
        weights = (
            float(reference_frequency_hz)
            / np.asarray(self.mode_frequencies_hz, dtype=np.float64)
        ) ** float(exponent)
        return np.column_stack(
            (
                weights @ np.asarray(self.real_mode_basis, dtype=np.float64),
                weights
                @ np.asarray(self.imaginary_mode_basis, dtype=np.float64),
            )
        )


@dataclass(frozen=True)
class ModalResidueFitResult:
    calibration: ImpedanceModalResidueCalibration
    report: dict[str, Any]


def _case_metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_norm = float(np.linalg.norm(target))
    prediction_norm = float(np.linalg.norm(prediction))
    nrmse = float(
        np.linalg.norm(target - prediction) / max(target_norm, 1e-15)
    )
    if np.std(target) <= 1e-15 or np.std(prediction) <= 1e-15:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "nrmse": nrmse,
        "correlation": correlation,
        "energy_ratio": float(
            prediction_norm**2 / max(target_norm**2, 1e-30)
        ),
    }


def _aggregate_metrics(
    metrics: Iterable[dict[str, float]],
) -> dict[str, float]:
    values = tuple(metrics)
    if not values:
        return {
            "case_count": 0,
            "mean_nrmse": float("nan"),
            "mean_correlation": float("nan"),
            "mean_energy_ratio": float("nan"),
        }
    return {
        "case_count": len(values),
        "mean_nrmse": float(np.mean([value["nrmse"] for value in values])),
        "mean_correlation": float(
            np.mean([value["correlation"] for value in values])
        ),
        "mean_energy_ratio": float(
            np.mean([value["energy_ratio"] for value in values])
        ),
    }


def fit_fixed_pole_modal_residues(
    cases: Iterable[FixedPoleResidueCase],
    *,
    reference_id: str,
    boundary_reference_id: str,
    valid_frequency_range_hz: tuple[float, float],
    reference_frequency_hz: float = 100.0,
    exponent_candidates: Iterable[float] = tuple(np.linspace(-0.5, 2.0, 51)),
    acceptance_splits: Iterable[str] = ("holdout",),
    minimum_mean_holdout_correlation: float = 0.9,
    maximum_mean_holdout_nrmse: float = 0.35,
    source: dict[str, Any] | None = None,
    applicability: dict[str, Any] | None = None,
) -> ModalResidueFitResult:
    """Fit a complex global scale and frequency exponent with fixed poles.

    Every training case is normalized by its analysis-window RMS before the
    joint least-squares solve.  This prevents a loud source/receiver pair from
    dominating quieter positions.  Holdout cases never influence the fit.
    """
    reference_cases = tuple(cases)
    training = tuple(case for case in reference_cases if case.split == "train")
    if not training:
        raise ValueError("fixed-pole residue fitting requires training cases")
    evaluation_splits = tuple(
        sorted(
            {
                case.split
                for case in reference_cases
                if case.split != "train"
            }
        )
    )
    requested_acceptance_splits = tuple(str(value) for value in acceptance_splits)
    if not requested_acceptance_splits or any(
        value not in evaluation_splits
        for value in requested_acceptance_splits
    ):
        raise ValueError(
            "every acceptance split must exist as a non-training case split"
        )
    if (
        not -1.0 <= float(minimum_mean_holdout_correlation) <= 1.0
        or not math.isfinite(float(maximum_mean_holdout_nrmse))
        or float(maximum_mean_holdout_nrmse) <= 0.0
    ):
        raise ValueError("holdout acceptance thresholds are invalid")
    candidates = tuple(float(value) for value in exponent_candidates)
    if not candidates or any(not math.isfinite(value) for value in candidates):
        raise ValueError("exponent candidates must be finite and non-empty")

    def balanced_arrays(
        exponent: float,
        selected_cases: tuple[FixedPoleResidueCase, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        designs = []
        targets = []
        for case in selected_cases:
            mask = np.asarray(case.analysis_mask, dtype=bool)
            target = np.asarray(case.target, dtype=np.float64)[mask]
            scale = float(np.sqrt(np.mean(target**2)))
            designs.append(
                case.design(exponent, reference_frequency_hz)[mask] / scale
            )
            targets.append(target / scale)
        return np.vstack(designs), np.concatenate(targets)

    best: tuple[float, float, np.ndarray] | None = None
    exponent_search = []
    for exponent in candidates:
        design, target = balanced_arrays(exponent, training)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        nrmse = float(
            np.linalg.norm(target - design @ coefficients)
            / max(np.linalg.norm(target), 1e-15)
        )
        exponent_search.append(
            {
                "frequency_exponent": exponent,
                "balanced_train_nrmse": nrmse,
            }
        )
        if best is None or nrmse < best[0]:
            best = (nrmse, exponent, coefficients)
    assert best is not None
    balanced_nrmse, exponent, coefficients = best
    # design columns are Re(z), Im(z), while the renderer uses Re(C*z).
    complex_scale = complex(
        float(coefficients[0]),
        -float(coefficients[1]),
    )

    baseline_design, baseline_target = balanced_arrays(1.0, training)
    baseline_gain = float(
        np.linalg.lstsq(
            baseline_design[:, 1:2],
            baseline_target,
            rcond=None,
        )[0][0]
    )

    case_reports: list[dict[str, Any]] = []
    for case in reference_cases:
        mask = np.asarray(case.analysis_mask, dtype=bool)
        target = np.asarray(case.target, dtype=np.float64)[mask]
        calibrated_design = case.design(
            exponent,
            reference_frequency_hz,
        )[mask]
        calibrated = calibrated_design @ coefficients
        baseline = (
            case.design(1.0, reference_frequency_hz)[mask, 1]
            * baseline_gain
        )
        case_reports.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "mode_count": int(
                    np.asarray(case.mode_frequencies_hz).size
                ),
                "calibrated": _case_metric(target, calibrated),
                "baseline_legacy_1_over_f_sine": _case_metric(
                    target,
                    baseline,
                ),
                "metadata": dict(case.metadata),
            }
        )

    aggregate: dict[str, Any] = {}
    for split in ("train", *evaluation_splits):
        reports = [value for value in case_reports if value["split"] == split]
        aggregate[split] = {
            "calibrated": _aggregate_metrics(
                value["calibrated"] for value in reports
            ),
            "baseline_legacy_1_over_f_sine": _aggregate_metrics(
                value["baseline_legacy_1_over_f_sine"] for value in reports
            ),
        }
    acceptance_by_split = {}
    for split in requested_acceptance_splits:
        calibrated_metrics = aggregate[split]["calibrated"]
        baseline_metrics = aggregate[split][
            "baseline_legacy_1_over_f_sine"
        ]
        acceptance_by_split[split] = {
            "accepted": bool(
                calibrated_metrics["mean_correlation"]
                >= float(minimum_mean_holdout_correlation)
                and calibrated_metrics["mean_nrmse"]
                <= float(maximum_mean_holdout_nrmse)
                and calibrated_metrics["mean_nrmse"]
                < baseline_metrics["mean_nrmse"]
            ),
            "calibrated_mean_correlation": calibrated_metrics[
                "mean_correlation"
            ],
            "calibrated_mean_nrmse": calibrated_metrics["mean_nrmse"],
            "baseline_mean_nrmse": baseline_metrics["mean_nrmse"],
        }
    acceptance = {
        "minimum_mean_holdout_correlation": float(
            minimum_mean_holdout_correlation
        ),
        "maximum_mean_holdout_nrmse": float(maximum_mean_holdout_nrmse),
        "requires_improvement_over_baseline": True,
        "acceptance_splits": list(requested_acceptance_splits),
        "by_split": acceptance_by_split,
        "accepted": bool(
            all(
                value["accepted"]
                for value in acceptance_by_split.values()
            )
        ),
    }
    fit_summary = {
        "balanced_train_nrmse": balanced_nrmse,
        "aggregate": aggregate,
        "acceptance": acceptance,
    }
    calibration = ImpedanceModalResidueCalibration(
        reference_id=reference_id,
        boundary_reference_id=boundary_reference_id,
        complex_scale=complex_scale,
        frequency_exponent=exponent,
        reference_frequency_hz=float(reference_frequency_hz),
        valid_frequency_range_hz=valid_frequency_range_hz,
        source=source
        or {
            "evidence_tier": "independent_numerical_reference",
            "reference_solver": "staggered_grid_fdtd",
        },
        applicability=applicability
        or {
            "scope": "controlled_rectangular_room_validation",
            "production_material_mapping_enabled": False,
        },
        fit_summary=fit_summary,
    )
    report = {
        "schema_version": IMPEDANCE_MODAL_RESIDUE_REPORT_SCHEMA_VERSION,
        "calibration": calibration.metadata(),
        "fit_configuration": {
            "reference_frequency_hz": float(reference_frequency_hz),
            "valid_frequency_range_hz": [
                float(value) for value in valid_frequency_range_hz
            ],
            "exponent_candidates": list(candidates),
            "case_balancing": "unit_analysis_window_rms_per_case",
            "pole_policy": "fixed_no_refit",
            "baseline": "fitted_gain_times_1_over_f_times_imaginary_quadrature",
        },
        "exponent_search": exponent_search,
        "cases": case_reports,
        "aggregate": aggregate,
        "acceptance": acceptance,
    }
    return ModalResidueFitResult(calibration=calibration, report=report)


__all__ = [
    "FixedPoleResidueCase",
    "IMPEDANCE_MODAL_RESIDUE_REPORT_SCHEMA_VERSION",
    "IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION",
    "LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION",
    "ImpedanceModalResidueCalibration",
    "ModalResidueFitResult",
    "fit_fixed_pole_modal_residues",
]
