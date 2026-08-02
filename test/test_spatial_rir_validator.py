import json
import sys

import soundfile as sf

from egs.rir_generation import render_spatial_rir
from egs.rir_generation.phases.m4_spatial_late_field.scripts import (
    build_m4_listening_comparison,
    validate_m4_exit,
    validate_spatial_rir,
)
import numpy as np


def test_m4_5_validator_builds_array_foa_and_spatial_report():
    report, artifacts = validate_spatial_rir.build_report(
        sample_rate=8000,
        duration_s=0.6,
        max_order=1,
        scene_seed=61,
        spatial_seed=62,
        spacing_m=0.08,
        plane_wave_count=32,
        delay_line_count=8,
        nperseg=32,
        maximum_coherence_rmse=1.0,
        maximum_band_coherence_rmse=1.0,
        maximum_imaginary_rms=1.0,
        maximum_late_iacc_l4=1.0,
    )

    assert report["stage"] == "M4.5"
    assert report["exit"]["passed"] is True
    assert all(report["structural_checks"].values())
    assert all(report["spatial_checks"].values())
    assert artifacts["m4_spatial_receiver_rir_2ch"].shape == (2, 4800)
    assert artifacts["m4_spatial_ambisonic_rir_4ch"].shape == (4, 4800)
    json.dumps(report, allow_nan=False)


def test_spatial_cli_writes_scene_array_foa_and_demo_brir(tmp_path, monkeypatch):
    output_dir = tmp_path / "spatial"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_spatial_rir.py",
            "--sample-rate",
            "8000",
            "--duration",
            "0.35",
            "--max-order",
            "1",
            "--delay-lines",
            "8",
            "--plane-waves",
            "16",
            "--seed",
            "63",
            "--binaural-spacing-m",
            "0.08",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert render_spatial_rir.main() == 0

    metadata = json.loads((output_dir / "metadata.json").read_text())
    receivers, receiver_rate = sf.read(
        output_dir / "receiver_rir.wav",
        always_2d=True,
    )
    ambisonic, ambisonic_rate = sf.read(
        output_dir / "ambisonic_acn_sn3d_rir.wav",
        always_2d=True,
    )
    brir, brir_rate = sf.read(
        output_dir / "binaural_brir.wav",
        always_2d=True,
    )

    assert metadata["scope"] == "explicit_opt_in_m4_spatial_renderer"
    assert metadata["production_default_changed"] is False
    assert receiver_rate == ambisonic_rate == brir_rate == 8000
    assert receivers.shape == (2800, 2)
    assert ambisonic.shape == (2800, 4)
    assert brir.shape == (2800, 2)


def test_m4_exit_separates_implementation_from_empirical_evidence(tmp_path):
    reports = {
        "M4.1": {"milestone": "M4.1"},
        "M4.2": {"milestone": "M4.2", "spatial_contract": {}},
        "M4.3": {"milestone": "M4.3", "exit": {"passed": True}},
        "M4.4": {"milestone": "M4.4", "exit": {"passed": True}},
        "M4.5": {
            "stage": "M4.5",
            "renderer": {"production_default_changed": False},
            "exit": {
                "passed": True,
                "measured_multi_receiver_validation_complete": False,
            },
        },
        "M4.6": {
            "stage": "M4.6",
            "exit": {"passed": True, "measured_hrtf_bundled": False},
        },
    }
    artifacts = {}
    for name in ("array", "foa", "brir"):
        path = tmp_path / f"{name}.wav"
        path.touch()
        artifacts[name] = path

    report = validate_m4_exit.build_report(
        reports,
        artifact_paths=artifacts,
    )

    assert report["implementation_exit"]["passed"] is True
    assert report["empirical_exit"]["passed"] is False
    assert report["production_enablement"]["ready"] is False
    json.dumps(report, allow_nan=False)


def test_listening_comparison_uses_one_gain_and_preserves_early_identity():
    dry = np.asarray([1.0, -0.5, 0.25])
    m3 = np.zeros((8, 2))
    m3[1] = [1.0, 0.9]
    m3[5] = [0.2, 0.2]
    m4 = m3.copy()
    m4[5] = [0.4, -0.1]
    brir = m4 * 0.5
    report = {
        "renderer": {
            "receiver_coupling": {
                "channels": [
                    {"transition_start_sample": 3},
                    {"transition_start_sample": 3},
                ]
            }
        }
    }

    outputs, manifest = build_m4_listening_comparison.build_comparison(
        dry,
        m3,
        m4,
        brir,
        8000,
        report,
        target_peak_dbfs=-1.0,
        silence_s=0.1,
    )

    assert manifest["rir_early_identity"]["exact"] is True
    assert manifest["mastering"]["per_file_loudness_normalization"] is False
    assert 0.0 < manifest["mastering"]["common_master_gain"] <= 1.0
    assert np.array_equal(
        outputs["m4_minus_m3_difference"],
        outputs["m4_spatial_array"] - outputs["m3_coherent_array"],
    )
    assert outputs["ab_m3_then_m4"].shape[1] == 2
