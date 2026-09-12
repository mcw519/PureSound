import numpy as np
import pytest

from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
    _complex_metrics,
    _image_source_transfer,
    _shoebox_image_geometry,
)
from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)


def test_shoebox_image_geometry_has_direct_and_six_first_order_paths():
    geometry = _shoebox_image_geometry(
        (4.0, 3.0, 2.5),
        (1.0, 1.2, 1.1),
        (2.5, 1.8, 1.4),
        max_order=1,
    )
    orders = geometry["orders_xyz"]

    assert orders.shape == (3, 7)
    assert np.count_nonzero(np.sum(np.abs(orders), axis=0) == 0) == 1
    assert np.count_nonzero(np.sum(np.abs(orders), axis=0) == 1) == 6
    assert np.all(geometry["distances_m"] > 0.0)
    assert np.all(geometry["incidence_cosines_xyz"] >= 0.0)
    assert np.all(geometry["incidence_cosines_xyz"] <= 1.0)


def test_rigid_shoebox_reflection_policies_are_identical():
    geometry = _shoebox_image_geometry(
        (4.0, 3.0, 2.5),
        (1.0, 1.2, 1.1),
        (2.5, 1.8, 1.4),
        max_order=2,
    )
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    frequencies_hz = np.asarray([80.0, 160.0, 240.0])

    responses = [
        _image_source_transfer(
            geometry,
            frequencies_hz,
            rigid,
            sound_speed_m_s=343.0,
            reflection_policy=policy,
        )
        for policy in (
            "magnitude_only_normal",
            "complex_normal",
            "complex_angle",
        )
    ]

    assert responses[1] == pytest.approx(responses[0])
    assert responses[2] == pytest.approx(responses[0])


def test_complex_metrics_report_identity():
    reference = np.asarray([1.0 + 0.5j, -0.2 + 0.7j, 0.4 - 0.1j])

    metrics = _complex_metrics(reference, reference)

    assert metrics["complex_nrmse"] == pytest.approx(0.0)
    assert metrics["complex_correlation"] == pytest.approx(1.0)
    assert metrics["transfer_energy_ratio"] == pytest.approx(1.0)
