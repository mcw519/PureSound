import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "egs"
    / "rir_generation"
    / "calibrate_impedance_residue_protocol.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "calibrate_impedance_residue_protocol",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m2_9_protocol_has_true_room_and_grid_holdouts():
    module = _load_script()

    by_room = {
        room.room_id: room for room in module.M2_9_CALIBRATION_SETS
    }
    unseen = by_room["room_c_unseen"]
    unseen_splits = [position.split for position in unseen.positions]
    training_rooms = [
        room
        for room in module.M2_9_CALIBRATION_SETS
        if any(position.split == "train" for position in room.positions)
    ]

    assert len(training_rooms) == 2
    assert "train" not in unseen_splits
    assert unseen_splits.count("room_holdout") == 2
    assert unseen_splits.count("grid_holdout") == 1
    assert module.ACCEPTANCE_SPLITS == (
        "position_holdout",
        "room_holdout",
        "grid_holdout",
    )


def test_boundary_case_parser_is_strict():
    module = _load_script()

    assert module._parse_boundary_case("thin=/tmp/thin.json") == (
        "thin",
        Path("/tmp/thin.json"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_boundary_case("missing_separator")
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_boundary_case("Bad Name=/tmp/model.json")
