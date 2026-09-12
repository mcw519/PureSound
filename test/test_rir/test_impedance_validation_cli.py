import json
import subprocess
import sys
from pathlib import Path


def test_impedance_validation_cli_writes_accepted_report(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    measurement_dir = (
        repo_root
        / "egs"
        / "rir_generation"
        / "phases"
        / "m2_impedance"
        / "measurements"
        / "zenodo_15195587"
    )
    output_path = tmp_path / "validation.json"

    subprocess.run(
        [
            sys.executable,
            str(
                repo_root
                / "egs"
                / "rir_generation"
                / "phases"
                / "m2_impedance"
                / "scripts"
                / "validate_impedance_measurement.py"
            ),
            "--csv",
            str(measurement_dir / "nasa_gfit_noflow_130db_kt.csv"),
            "--metadata",
            str(measurement_dir / "nasa_gfit_noflow_130db_kt.json"),
            "--comparison-csv",
            str(measurement_dir / "ufsc_noflow_130db_kt.csv"),
            "--comparison-metadata",
            str(measurement_dir / "ufsc_noflow_130db_kt.json"),
            "--output",
            str(output_path),
            "--dense-points",
            "512",
        ],
        check=True,
        cwd=repo_root,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["fit"]["held_out"]["maximum_complex_reflection_error"] < 0.06
    assert report["dense_passivity"]["passed"] is True
    assert (
        report["cross_measurement"]["rms_complex_cayley_difference"] > 0.11
    )
    assert report["one_dimensional_modal_diagnostic"]["q_ratio"] > 2.0
