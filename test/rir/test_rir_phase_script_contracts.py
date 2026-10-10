import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RIR_ROOT = REPO_ROOT / "egs" / "rir_generation"
REPO_ROOT_PATTERN = re.compile(
    r"^REPO_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]", re.MULTILINE
)
SCOPE_GLOBS = (
    "*.py",
    "tools/*/*.py",
    "phases/m4_spatial_late_field/scripts/*.py",
    "phases/m5_calibration/scripts/*.py",
    "phases/m6_bank/scripts/*.py",
)
M5_ARTIFACT_WRITERS = (
    "validate_m5_synthetic_recovery",
    "validate_m5_robust_recovery",
    "validate_m5_m4_parameter_mapping",
    "validate_m5_group_identifiability",
    "validate_m5_spatial_calibration",
    "validate_m5_constrained_residual",
)


def _scripts_with_repo_root():
    found = []
    for pattern in SCOPE_GLOBS:
        for path in sorted(RIR_ROOT.glob(pattern)):
            match = REPO_ROOT_PATTERN.search(path.read_text(encoding="utf-8"))
            if match:
                found.append((path, int(match.group(1))))
    return found


@pytest.mark.parametrize(
    "path,depth",
    _scripts_with_repo_root(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_script_repo_root_resolves_to_the_repository(path, depth):
    root = path.resolve().parents[depth]
    assert (root / "puresound").is_dir(), f"{path.name}: parents[{depth}] is {root}"


@pytest.mark.parametrize("module_name", M5_ARTIFACT_WRITERS)
def test_m5_validator_artifacts_can_be_written_outside_the_repository(
    module_name, tmp_path
):
    module = importlib.import_module(
        f"egs.rir_generation.phases.m5_calibration.scripts.{module_name}"
    )
    inside = module.REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/x"
    assert not str(tmp_path).startswith(str(module.REPO_ROOT))

    paths = module._write_artifacts(tmp_path, {"probe": np.zeros((2, 16))}, 8000)

    assert paths == {"probe": str(next(tmp_path.glob("*.wav")))}
    assert module._display_path(inside / "a.wav") == str(
        Path("egs/rir_generation/exp/rir_realism/m5/x/a.wav")
    )


def _fake_arm(high_backend, qc_passed):
    items = {}
    for index, split in enumerate(("train", "validation", "test")):
        key = f"space-{index}|item-{index}"
        items[key] = {
            "item": SimpleNamespace(
                item_id=f"item-{index}",
                split=split,
                qc_status="pass" if index < qc_passed else "fail",
            ),
            "metadata": {
                "m6": {
                    "scene_sha256": "s",
                    "split": split,
                    "generation_seed": index,
                    "sample_rate": 16000,
                    "channel_count": 5,
                    "frame_count": 100,
                },
                "bands": {
                    "low": {"backend": "pytard"},
                    "high": {"backend": high_backend},
                },
            },
        }
    return {"root": None, "manifest": None, "items": items}


def _run_pilot_pair(monkeypatch, tmp_path, arms):
    from egs.rir_generation.phases.m6_bank.scripts import validate_m6_pilot_pair

    for name in arms:
        bank = tmp_path / f"{name}_bank"
        bank.mkdir()
        (bank / "rir_bank_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        validate_m6_pilot_pair,
        "_load_arm",
        lambda root: arms[root.name.removesuffix("_bank")],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_m6_pilot_pair.py",
            "--pilot-root",
            str(tmp_path),
            "--arm-a",
            "arm_a",
            "--arm-b",
            "arm_b",
            "--skip-release-audit",
        ],
    )
    status = validate_m6_pilot_pair.main()
    report = json.loads((tmp_path / "pilot_pair_report.json").read_text())
    return status, report


def test_pilot_pair_exit_status_follows_the_pairing_checks_not_qc_yield(
    monkeypatch, tmp_path, capsys
):
    # Same high backend on both arms: the pairing is broken, whatever QC passed.
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    broken = {
        "arm_a": _fake_arm("path-events-m4", qc_passed=3),
        "arm_b": _fake_arm("path-events-m4", qc_passed=3),
    }
    status, report = _run_pilot_pair(monkeypatch, broken_root, broken)
    assert report["pairing_holds"] is False
    assert status == 1
    assert "PAIRING BROKEN" in capsys.readouterr().out

    # A matched pair stays a pass even when the last arm quarantined everything.
    held_root = tmp_path / "held"
    held_root.mkdir()
    held = {
        "arm_a": _fake_arm("pyroomacoustics", qc_passed=3),
        "arm_b": _fake_arm("path-events-m4", qc_passed=0),
    }
    status, report = _run_pilot_pair(monkeypatch, held_root, held)
    assert report["pairing_holds"] is True
    assert status == 0
    assert "PAIRING HOLDS" in capsys.readouterr().out


@pytest.mark.parametrize("passed,expected_status", [(True, 0), (False, 1)])
def test_multiband_fdn_exit_status_follows_the_gate(
    passed, expected_status, monkeypatch, tmp_path, capsys
):
    from egs.rir_generation.phases.m4_spatial_late_field.scripts import (
        validate_multiband_fdn,
    )

    band = {
        "qualified_high_band_gate": True,
        "target_t20_s": 0.5,
        "rendered_t20_s": None,
        "t20_relative_error": None,
        "rendered_mixing_time_s": None,
        "rendered_late_median_normalized_density": None,
    }
    report = {
        "structural_checks": {"finite_output": True},
        "band_checks": {"1000": band},
        "exit": {"passed": passed},
    }
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        validate_multiband_fdn,
        "build_report",
        lambda *args, **kwargs: (report, np.zeros(8, dtype=np.float32)),
    )
    monkeypatch.setattr(
        sys, "argv", ["validate_multiband_fdn.py", "--target-report", str(target)]
    )

    assert validate_multiband_fdn.main() == expected_status
    assert "n/a" in capsys.readouterr().out


def test_coupling_validator_reports_a_failed_band_instead_of_crashing(
    monkeypatch, tmp_path, capsys
):
    from egs.rir_generation.phases.m4_spatial_late_field.scripts import (
        validate_path_event_fdn_coupling,
    )

    band = {
        "m4_t20_relative_error": None,
        "m3_median_mixing_time_s": None,
        "m4_median_mixing_time_s": None,
        "m3_median_late_density": None,
        "m4_median_late_density": None,
        "t20_passed": False,
        "mixing_time_passed": False,
        "late_density_tolerance_passed": False,
        "late_density_improved_over_m3": False,
    }
    report = {
        "structural_checks": {"finite_output": True},
        "band_checks": {"1000": band},
        "full_hybrid": {"checks": {"exact_before_transition": True}},
        "exit": {"passed": False},
    }
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        validate_path_event_fdn_coupling,
        "build_report",
        lambda *args, **kwargs: (report, {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_path_event_fdn_coupling.py", "--target-report", str(target)],
    )

    assert validate_path_event_fdn_coupling.main() == 1
    assert "M4.4 coupling: FAIL" in capsys.readouterr().out
