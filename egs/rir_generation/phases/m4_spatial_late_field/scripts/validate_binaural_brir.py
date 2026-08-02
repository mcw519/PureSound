#!/usr/bin/env python
"""Validate M4.6 receiver directivity and optional HRTF-decoder BRIR API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.binaural_renderer import (  # noqa: E402
    AmbisonicBinauralDecoder,
    analytic_first_order_binaural_decoder,
    render_ambisonic_brir,
)
from puresound.audio.rir_metrics import analyze_binaural_iacc  # noqa: E402


DEFAULT_SPATIAL_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json"
)
DEFAULT_AMBISONIC_RIR = (
    REPO_ROOT
    / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_spatial_ambisonic_rir_4ch.wav"
)
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_binaural_brir_report.json"
)
DEFAULT_OUTPUT_BRIR = (
    REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav"
)


def _external_decoder_fixture(sample_rate: int) -> AmbisonicBinauralDecoder:
    """Return a multi-tap fixture exercising the HRTF-derived decoder contract."""

    firs = np.zeros((2, 4, 3), dtype=np.float64)
    firs[0, 0] = [0.60, 0.05, -0.01]
    firs[0, 1] = [0.35, 0.04, 0.00]
    firs[0, 3] = [0.10, 0.00, 0.00]
    firs[1, 0] = [0.60, 0.05, -0.01]
    firs[1, 1] = [-0.35, -0.04, 0.00]
    firs[1, 3] = [0.10, 0.00, 0.00]
    return AmbisonicBinauralDecoder(
        sample_rate=sample_rate,
        firs=firs,
        reference_id="m4-contract-fixture-not-measured-hrtf",
        decoder_kind="hrtf_derived_contract_fixture_not_measured",
        provenance={
            "measurement": False,
            "purpose": "exercise external FIR injection contract",
            "production_use": False,
        },
    )


def build_report(
    ambisonic: np.ndarray,
    sample_rate: int,
    spatial_report: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Decode a complete FOA RIR and return implementation-exit evidence."""

    analytic_decoder = analytic_first_order_binaural_decoder(sample_rate)
    first = render_ambisonic_brir(
        ambisonic,
        sample_rate,
        analytic_decoder,
    )
    repeated = render_ambisonic_brir(
        ambisonic,
        sample_rate,
        analytic_decoder,
    )
    external_decoder = _external_decoder_fixture(sample_rate)
    external = render_ambisonic_brir(
        ambisonic,
        sample_rate,
        external_decoder,
    )
    direct_sample = int(
        spatial_report["renderer"]["ambisonic_direct_sample"]
    )
    iacc = analyze_binaural_iacc(
        first.brir[0],
        first.brir[1],
        sample_rate,
        direct_index=direct_sample,
        early_end_ms=80.0,
        centers_hz=(500.0, 1000.0, 2000.0, 4000.0),
    )
    first_input_nonzero = np.flatnonzero(
        np.any(np.abs(ambisonic) > 1e-14, axis=0)
    )
    causal_prefix = int(first_input_nonzero[0]) if first_input_nonzero.size else 0
    checks = {
        "m4_5_input_report_passed": bool(
            spatial_report.get("exit", {}).get("passed", False)
        ),
        "input_is_acn_sn3d_first_order": bool(
            spatial_report["renderer"]["late_field"]["ambisonic"]
            ["channel_labels"]
            == ["W", "Y", "Z", "X"]
        ),
        "analytic_output_shape_is_two_by_samples": bool(
            first.brir.shape == (2, ambisonic.shape[1])
        ),
        "external_fir_output_includes_filter_tail": bool(
            external.brir.shape == (2, ambisonic.shape[1] + 2)
        ),
        "same_decoder_exact_determinism": bool(
            np.array_equal(first.brir, repeated.brir)
        ),
        "finite_output": bool(
            np.all(np.isfinite(first.brir))
            and np.all(np.isfinite(external.brir))
        ),
        "causal_prefix_preserved": bool(
            np.all(first.brir[:, :causal_prefix] == 0.0)
            and np.all(external.brir[:, :causal_prefix] == 0.0)
        ),
        "left_and_right_are_distinct": bool(
            not np.array_equal(first.brir[0], first.brir[1])
        ),
        "external_decoder_reference_and_provenance_retained": bool(
            external.metadata["decoder"]["reference_id"]
            == "m4-contract-fixture-not-measured-hrtf"
            and external.metadata["decoder"]["provenance"]["measurement"]
            is False
        ),
        "analytic_decoder_not_mislabeled_as_measured_hrtf": bool(
            analytic_decoder.decoder_kind
            == "analytic_demonstration_not_hrtf"
            and analytic_decoder.provenance[
                "production_hrtf_replacement_required"
            ]
            is True
        ),
        "iacc_analysis_valid": bool(
            iacc["broadband"]["early"]["valid"]
            and iacc["broadband"]["late"]["valid"]
        ),
    }
    passed = bool(all(checks.values()))
    report = {
        "schema_version": "puresound.m4_binaural_brir_validation.v1",
        "stage": "M4.6",
        "evidence_scope": "decoder_contract_and_analytic_render_validation",
        "configuration": {
            "sample_rate": int(sample_rate),
            "input_shape": list(ambisonic.shape),
            "input_spatial_report_schema": spatial_report["schema_version"],
        },
        "analytic_decoder": analytic_decoder.to_dict(
            include_coefficients=True
        ),
        "external_decoder_contract_fixture": external_decoder.to_dict(
            include_coefficients=True
        ),
        "analytic_render": first.metadata,
        "external_decoder_render": external.metadata,
        "iacc_diagnostic_not_a_perceptual_gate": iacc,
        "checks": checks,
        "exit": {
            "passed": passed,
            "implementation_complete": passed,
            "measured_hrtf_bundled": False,
            "production_binaural_calibration_complete": False,
            "production_default_enabled": False,
        },
        "limitations": [
            "No measured HRTF dataset is bundled because dataset identity and licensing must be explicit.",
            "The generated audition BRIR uses a headless analytic decoder and is not a perceptual HRTF reference.",
            "The HRTF-derived FIR injection API is complete, but production use still requires measured licensed decoder coefficients and listening validation.",
        ],
    }
    return report, first.brir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-report",
        type=Path,
        default=DEFAULT_SPATIAL_REPORT,
    )
    parser.add_argument(
        "--ambisonic-rir",
        type=Path,
        default=DEFAULT_AMBISONIC_RIR,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
    )
    parser.add_argument(
        "--output-brir",
        type=Path,
        default=DEFAULT_OUTPUT_BRIR,
    )
    args = parser.parse_args()
    try:
        spatial_report = json.loads(
            args.spatial_report.read_text(encoding="utf-8")
        )
        audio, sample_rate = sf.read(
            args.ambisonic_rir,
            always_2d=True,
            dtype="float64",
        )
        if audio.shape[1] != 4:
            raise ValueError("Ambisonic WAV must contain four channels")
        report, brir = build_report(
            np.asarray(audio.T, dtype=np.float64),
            int(sample_rate),
            spatial_report,
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    for key, value in report["checks"].items():
        print(f"check\t{key}\t{'PASS' if value else 'FAIL'}")
    iacc = report["iacc_diagnostic_not_a_perceptual_gate"]
    print(f"iacc\tearly_l4\t{iacc['iacc_e4']:.6f}")
    print(f"iacc\tlate_l4\t{iacc['iacc_l4']:.6f}")
    print(f"# M4.6 binaural BRIR: {'PASS' if report['exit']['passed'] else 'FAIL'}")

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    args.output_brir.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_brir, brir.T, sample_rate, subtype="FLOAT")
    print(f"# wrote {args.output_report}")
    print(f"# wrote {args.output_brir}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
