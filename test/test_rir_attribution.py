import numpy as np
import pytest

from egs.rir_generation.phases.m3_wave_path.scripts.validate_direct_early_later_attribution import (
    _coherent_error_attribution,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _time_domain_probe,
)
from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir_attribution import (
    complementary_early_late_masks,
    decompose_direct_early_later,
    reconstruction_error,
)
from puresound.audio.rir_path_events import (
    generate_shoebox_path_events,
    partition_path_events_by_arrival,
)


def test_early_late_masks_are_smooth_and_exactly_complementary():
    early, later = complementary_early_late_masks(
        num_samples=1000,
        sample_rate_hz=1000.0,
        split_center_s=0.5,
        transition_width_s=0.1,
    )

    assert early[:451] == pytest.approx(1.0)
    assert early[550:] == pytest.approx(0.0)
    assert np.all(np.diff(early) <= 0.0)
    assert early + later == pytest.approx(np.ones(1000))


def test_direct_early_later_decomposition_reconstructs_full_output():
    time_s = np.arange(2000, dtype=np.float64) / 2000.0
    direct = np.exp(-((time_s - 0.03) / 0.004) ** 2)
    full = (
        direct
        + 0.5 * np.exp(-((time_s - 0.06) / 0.01) ** 2)
        + 0.2 * np.exp(-((time_s - 0.2) / 0.03) ** 2)
    )

    components = decompose_direct_early_later(
        full,
        direct,
        sample_rate_hz=2000.0,
        split_center_s=0.10,
        transition_width_s=0.02,
    )
    error = reconstruction_error(components)

    assert error["maximum_absolute_error"] <= 1e-15
    assert error["nrmse"] <= 1e-15
    assert components["early_cumulative"] == pytest.approx(
        components["direct"] + components["early_reflections"]
    )


def test_path_arrival_partition_is_complete_and_uses_direct_relative_time():
    event_set = generate_shoebox_path_events(
        dimensions_m=(4.0, 3.0, 2.5),
        source_position_m=(1.0, 1.1, 0.9),
        receiver_position_m=(2.8, 2.0, 1.6),
        sound_speed_m_s=343.0,
        max_order=4,
    )
    buckets = partition_path_events_by_arrival(
        event_set,
        early_window_s=0.015,
    )
    direct_delay = buckets["direct"][0].delay_s

    assert sum(len(events) for events in buckets.values()) == len(
        event_set.events
    )
    assert all(
        event.delay_s <= direct_delay + 0.015
        for event in buckets["early_reflections"]
    )
    assert all(
        event.delay_s > direct_delay + 0.015
        for event in buckets["later_reflections"]
    )


def test_coherent_error_attribution_includes_the_interference_cross_term():
    reference = {
        "early_reflections": np.asarray([1.0 + 0.0j, 0.0 + 1.0j]),
        "later_reflections": np.asarray([0.5 + 0.0j, 0.0 - 0.5j]),
        "full": np.asarray([1.5 + 0.0j, 0.0 + 0.5j]),
    }
    early_error = np.asarray([0.2 + 0.1j, -0.1 + 0.0j])
    later_error = np.asarray([-0.05 + 0.0j, 0.0 + 0.08j])
    candidate = {
        "early_reflections": (
            reference["early_reflections"] + early_error
        ),
        "later_reflections": (
            reference["later_reflections"] + later_error
        ),
        "full": reference["full"] + early_error + later_error,
    }

    attribution = _coherent_error_attribution(reference, candidate)

    assert attribution["term_sum"] == pytest.approx(1.0)
    assert attribution["error_reconstruction_nrmse"] <= 1e-14
    assert attribution["early_later_cross_term"] != pytest.approx(0.0)


def test_1d_time_domain_probe_matches_the_discrete_boundary_equation():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )

    report = _time_domain_probe(
        model,
        scheme="face_time_extrapolated",
        frequencies_hz=np.asarray([180.0, 240.0, 300.0]),
        sample_rate_hz=30000.0,
        grid_spacing_m=0.06,
    )
    metrics = report["measured_vs_predicted_discrete_cross_ratio"]

    assert metrics["maximum_complex_error"] < 1e-3
    assert metrics["maximum_phase_error_deg"] < 0.01


def test_fourth_order_1d_closure_matches_the_harmonic_candidate():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.08,
        normalized_admittance_relaxation=0.6,
        relaxation_frequency_hz=120.0,
    )

    report = _time_domain_probe(
        model,
        scheme="face_quadratic_time_quadratic",
        frequencies_hz=np.asarray([180.0, 240.0, 300.0]),
        sample_rate_hz=30000.0,
        grid_spacing_m=0.06,
        spatial_derivative_order=4,
        near_wall_closure="third_order_one_sided",
    )
    metrics = report["measured_vs_predicted_discrete_cross_ratio"]

    assert report["geometry"]["spatial_derivative_order"] == 4
    assert report["geometry"]["near_wall_closure"] == (
        "third_order_one_sided"
    )
    assert report["geometry"]["one_dimensional_cfl_number"] < 1.0
    assert metrics["maximum_complex_error"] < 1e-3
    assert metrics["maximum_phase_error_deg"] < 0.01
