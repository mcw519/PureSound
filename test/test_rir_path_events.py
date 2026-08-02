from dataclasses import replace

import numpy as np
import pytest

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
    digital_locally_reacting_reflection_filter,
)
from puresound.audio.rir.path_events import (
    ComplexPathGainSpectrum,
    PathEventSet,
    apply_scene_object_visibility,
    causal_fractional_delay_kernel,
    directivity_pressure_gain,
    generate_scene_shoebox_path_events,
    generate_shoebox_path_events,
    locally_reacting_reflection_coefficient,
    render_path_events,
    segment_intersects_scene_object,
)
from puresound.audio.rir.scene.schema import MaterialSpectrum, Pose, SceneObject


DIMENSIONS = np.asarray([5.0, 4.0, 3.0])
SOURCE = np.asarray([1.1, 1.3, 1.2])
RECEIVER = np.asarray([3.8, 2.9, 1.6])
SOUND_SPEED = 343.0
BOUNDARY_AXIS_PLANE_NORMAL = {
    "west": (0, 0.0, np.asarray([-1.0, 0.0, 0.0])),
    "east": (0, 5.0, np.asarray([1.0, 0.0, 0.0])),
    "south": (1, 0.0, np.asarray([0.0, -1.0, 0.0])),
    "north": (1, 4.0, np.asarray([0.0, 1.0, 0.0])),
    "floor": (2, 0.0, np.asarray([0.0, 0.0, -1.0])),
    "ceiling": (2, 3.0, np.asarray([0.0, 0.0, 1.0])),
}


def _events(**kwargs):
    return generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=SOURCE,
        receiver_position_m=RECEIVER,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="unit-room",
        source_id="talker",
        receiver_id="mic",
        **kwargs,
    )


def test_complex_gain_and_event_set_round_trip_without_loss():
    spectrum = ComplexPathGainSpectrum(
        frequencies_hz=[100.0, 200.0],
        real=[0.5, 0.25],
        imag=[-0.1, -0.2],
        provenance="unit test",
    )

    assert spectrum.at(150.0) == pytest.approx(0.375 - 0.15j)
    assert spectrum.at(20.0) == pytest.approx(0.5 - 0.1j)

    event_set = _events()
    restored = PathEventSet.from_json(event_set.to_json())

    assert restored.to_dict() == event_set.to_dict()
    assert restored.events[0].gain_spectrum.quantity == (
        "pressure_gain_excluding_propagation_delay"
    )


def test_exact_shoebox_direct_and_first_order_geometry():
    event_set = _events()

    assert len(event_set.events) == 7
    direct = event_set.events[0]
    direct_distance = float(np.linalg.norm(RECEIVER - SOURCE))
    direct_direction = (RECEIVER - SOURCE) / direct_distance
    assert direct.path_type == "direct"
    assert direct.distance_m == pytest.approx(direct_distance, abs=1e-14)
    assert direct.delay_s == pytest.approx(direct_distance / SOUND_SPEED)
    assert direct.departure_direction_unit == pytest.approx(direct_direction)
    assert direct.arrival_direction_unit == pytest.approx(direct_direction)
    assert direct.gain_spectrum.constant_real_value() == pytest.approx(
        1.0 / direct_distance
    )

    for event in event_set.events[1:]:
        boundary = event.surface_ids[0]
        axis, plane, normal = BOUNDARY_AXIS_PLANE_NORMAL[boundary]
        point = np.asarray(event.interaction_points_m[0])
        departure = np.asarray(event.departure_direction_unit)
        arrival = np.asarray(event.arrival_direction_unit)
        image = SOURCE.copy()
        image[axis] = 2.0 * plane - image[axis]
        expected_distance = float(np.linalg.norm(RECEIVER - image))
        assert point[axis] == pytest.approx(plane, abs=1e-14)
        assert np.all(point >= -1e-14)
        assert np.all(point <= DIMENSIONS + 1e-14)
        assert event.distance_m == pytest.approx(expected_distance, abs=1e-14)
        assert event.distance_m == pytest.approx(
            np.linalg.norm(point - SOURCE) + np.linalg.norm(RECEIVER - point),
            abs=1e-14,
        )
        incident_cosine = abs(float(np.dot(departure, normal)))
        outgoing_cosine = abs(float(np.dot(arrival, normal)))
        assert event.incidence_cosines[0] == pytest.approx(
            incident_cosine, abs=1e-14
        )
        assert outgoing_cosine == pytest.approx(incident_cosine, abs=1e-14)


def test_source_receiver_swap_is_reciprocal():
    forward = _events()
    reverse = generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=RECEIVER,
        receiver_position_m=SOURCE,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="unit-room",
        source_id="mic",
        receiver_id="talker",
    )
    forward_by_path = {
        event.surface_ids[0] if event.surface_ids else "direct": event
        for event in forward.events
    }
    reverse_by_path = {
        event.surface_ids[0] if event.surface_ids else "direct": event
        for event in reverse.events
    }

    assert forward_by_path.keys() == reverse_by_path.keys()
    for path_id, event in forward_by_path.items():
        reciprocal = reverse_by_path[path_id]
        assert reciprocal.distance_m == pytest.approx(event.distance_m, abs=1e-14)
        assert reciprocal.delay_s == pytest.approx(event.delay_s, abs=1e-16)
        assert reciprocal.departure_direction_unit == pytest.approx(
            -np.asarray(event.arrival_direction_unit), abs=1e-14
        )
        assert reciprocal.arrival_direction_unit == pytest.approx(
            -np.asarray(event.departure_direction_unit), abs=1e-14
        )
        assert np.asarray(reciprocal.interaction_points_m) == pytest.approx(
            np.asarray(event.interaction_points_m), abs=1e-14
        )
        assert reciprocal.gain_spectrum.values == pytest.approx(
            event.gain_spectrum.values, abs=1e-14
        )


def test_first_order_directivity_patterns_use_real_pressure_gain():
    front = [1.0, 0.0, 0.0]
    side = [0.0, 1.0, 0.0]
    rear = [-1.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]

    assert directivity_pressure_gain("omnidirectional", orientation, rear) == 1.0
    assert directivity_pressure_gain("cardioid", orientation, front) == pytest.approx(1.0)
    assert directivity_pressure_gain("cardioid", orientation, side) == pytest.approx(0.5)
    assert directivity_pressure_gain("cardioid", orientation, rear) == pytest.approx(0.0)
    assert directivity_pressure_gain("hypercardioid", orientation, rear) == pytest.approx(
        -0.5
    )
    assert directivity_pressure_gain("figure_eight", orientation, rear) == pytest.approx(
        -1.0
    )


def test_scene_path_events_select_receiver_and_apply_receiver_directivity():
    from puresound.audio.rir.contracts import HybridRIRConfig
    from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene

    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.25,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(config, seed=46, room_type="office")
    original = scene.receivers[0]
    second = replace(
        original,
        transducer_id="receiver-1",
        directivity_id="cardioid",
        pose=replace(
            original.pose,
            position_m=(np.asarray(original.pose.position_m) + [0.05, 0.0, 0.0]).tolist(),
            orientation_ypr_deg=[0.0, 0.0, 0.0],
        ),
        channel_index=1,
    )
    array_scene = replace(scene, receivers=[original, second])

    event_set = generate_scene_shoebox_path_events(
        array_scene,
        source_index=0,
        receiver_index=1,
    )
    direct = next(event for event in event_set.events if event.path_type == "direct")
    arrival_from_source = -np.asarray(direct.arrival_direction_unit)
    expected = directivity_pressure_gain(
        "cardioid",
        second.pose.orientation_ypr_deg,
        arrival_from_source,
    )

    assert event_set.receiver_id == "receiver-1"
    assert direct.receiver_position_m == pytest.approx(second.pose.position_m)
    assert direct.receiver_directivity_gain == pytest.approx(expected)
    assert event_set.metadata["receiver_directivity_id"] == "cardioid"


def test_angle_aware_complex_boundary_gain_is_stored_without_delay_phase():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    frequencies = [80.0, 160.0, 240.0]
    event_set = _events(
        boundary_admittance_models={"west": model},
        reflection_frequencies_hz=frequencies,
    )
    event = next(item for item in event_set.events if item.surface_ids == ["west"])
    cosine = event.incidence_cosines[0]

    expected = np.asarray(
        [
            locally_reacting_reflection_coefficient(
                model.normalized_admittance(frequency),
                cosine,
            )
            / event.distance_m
            for frequency in frequencies
        ]
    )
    assert event.gain_spectrum.values == pytest.approx(expected)
    assert np.all(np.abs(event.gain_spectrum.values * event.distance_m) <= 1.0)
    assert "propagation_delay" in event.gain_spectrum.quantity


@pytest.mark.parametrize("sample_rate_hz", [8000.0, 16000.0, 48000.0])
def test_causal_fractional_delay_has_bounded_low_band_error(sample_rate_hz):
    frequencies = np.linspace(60.0, 1000.0, 200)
    omega = 2.0 * np.pi * frequencies / sample_rate_hz

    for fraction in (0.05, 0.25, 0.5, 0.75, 0.95):
        delay = 10.0 + fraction
        start, kernel = causal_fractional_delay_kernel(delay, order=3)
        response = np.exp(-1j * omega * start) * (
            np.exp(-1j * omega[:, None] * np.arange(kernel.size)) @ kernel
        )
        ideal = np.exp(-1j * omega * delay)
        ratio = response / ideal
        magnitude_error_db = np.max(
            np.abs(20.0 * np.log10(np.maximum(np.abs(ratio), 1e-15)))
        )
        phase_error_deg = np.max(np.abs(np.angle(ratio, deg=True)))

        assert start == 10
        assert np.sum(kernel) == pytest.approx(1.0, abs=1e-14)
        assert magnitude_error_db <= 0.11
        assert phase_error_deg <= 0.60


def test_renderer_is_zero_before_physical_floor_sample_and_preserves_dc_gain():
    event_set = _events(max_order=0)
    event = event_set.events[0]
    sample_rate = 16000.0
    first_sample = int(np.floor(event.delay_s * sample_rate))

    rir = render_path_events(
        event_set,
        sample_rate_hz=sample_rate,
        num_samples=2048,
    )

    assert np.count_nonzero(rir[:first_sample]) == 0
    assert np.sum(rir) == pytest.approx(1.0 / event.distance_m, abs=1e-14)
    assert np.count_nonzero(rir[first_sample : first_sample + 4]) == 4


def test_renderer_refuses_unrealized_complex_boundary_spectrum():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    event_set = _events(
        boundary_admittance_models={"west": model},
        reflection_frequencies_hz=[80.0, 160.0, 240.0],
    )
    west = next(item for item in event_set.events if item.surface_ids == ["west"])
    only_west = replace(event_set, events=[west])

    with pytest.raises(ValueError, match="causal boundary filter"):
        render_path_events(
            only_west,
            sample_rate_hz=16000.0,
            num_samples=2048,
        )


def test_renderer_realizes_complex_boundary_as_causal_angle_filter():
    sample_rate_hz = 8000.0
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    frequencies = np.asarray([80.0, 160.0, 240.0])
    event_set = _events(
        boundary_admittance_models={"west": model},
        reflection_frequencies_hz=frequencies,
    )
    west = next(item for item in event_set.events if item.surface_ids == ["west"])
    only_west = replace(event_set, events=[west])
    rir = render_path_events(
        only_west,
        sample_rate_hz=sample_rate_hz,
        num_samples=4096,
        surface_admittance_models={"west": model},
    )
    first_sample = int(np.floor(west.delay_s * sample_rate_hz))

    assert np.count_nonzero(rir[:first_sample]) == 0
    time_s = np.arange(rir.size, dtype=np.float64) / sample_rate_hz
    actual = np.asarray(
        [
            np.sum(rir * np.exp(-2j * np.pi * frequency * time_s))
            for frequency in frequencies
        ]
    )
    start, kernel = causal_fractional_delay_kernel(
        west.delay_s * sample_rate_hz,
        order=3,
    )
    boundary_filter = digital_locally_reacting_reflection_filter(
        model,
        west.incidence_cosines[0],
        sample_rate_hz,
    )
    expected = []
    for frequency in frequencies:
        delay = np.exp(-2j * np.pi * frequency / sample_rate_hz)
        fractional_response = delay**start * sum(
            value * delay**index for index, value in enumerate(kernel)
        )
        expected.append(
            fractional_response
            * boundary_filter.frequency_response(float(frequency))
            / west.distance_m
        )
    assert actual == pytest.approx(np.asarray(expected), abs=1e-10)


def test_filtered_path_renderer_preserves_reciprocity_and_model_identity():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    frequencies = [80.0, 160.0, 240.0]
    forward = _events(
        boundary_admittance_models={"west": model},
        reflection_frequencies_hz=frequencies,
    )
    reverse = generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=RECEIVER,
        receiver_position_m=SOURCE,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="unit-room",
        source_id="mic",
        receiver_id="talker",
        boundary_admittance_models={"west": model},
        reflection_frequencies_hz=frequencies,
    )
    forward_west = replace(
        forward,
        events=[
            event for event in forward.events if event.surface_ids == ["west"]
        ],
    )
    reverse_west = replace(
        reverse,
        events=[
            event for event in reverse.events if event.surface_ids == ["west"]
        ],
    )
    forward_rir = render_path_events(
        forward_west,
        sample_rate_hz=16000.0,
        num_samples=4096,
        surface_admittance_models={"west": model},
    )
    reverse_rir = render_path_events(
        reverse_west,
        sample_rate_hz=16000.0,
        num_samples=4096,
        surface_admittance_models={"west": model},
    )

    assert reverse_rir == pytest.approx(forward_rir, abs=1e-13)

    wrong_model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.2,
        normalized_admittance_relaxation=0.1,
        relaxation_frequency_hz=300.0,
    )
    with pytest.raises(ValueError, match="does not match"):
        render_path_events(
            forward_west,
            sample_rate_hz=16000.0,
            num_samples=4096,
            surface_admittance_models={"west": wrong_model},
        )


def test_nearby_source_motion_changes_each_path_continuously():
    step = 1e-3
    first = _events()
    moved_source = SOURCE + np.asarray([step, 0.0, 0.0])
    second = generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=moved_source,
        receiver_position_m=RECEIVER,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="unit-room",
        source_id="talker",
        receiver_id="mic",
    )

    assert [event.event_id for event in second.events] == [
        event.event_id for event in first.events
    ]
    for original, moved in zip(first.events, second.events):
        assert abs(moved.distance_m - original.distance_m) <= step + 1e-14
        assert abs(moved.delay_s - original.delay_s) <= (
            step / SOUND_SPEED + 1e-16
        )


def test_room_scene_v2_adapter_preserves_channel_and_surface_identity():
    from puresound.audio.rir.contracts import HybridRIRConfig
    from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene

    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(
        config,
        seed=31,
        room_type="office",
    )
    cardioid_events = generate_scene_shoebox_path_events(
        scene,
        source_index=2,
    )
    assert cardioid_events.metadata["source_directivity_model"] == (
        "first_order_cardioid_pressure_gain"
    )
    assert all(
        0.0 <= event.source_directivity_gain <= 1.0
        for event in cardioid_events.events
    )

    omni_scene = replace(
        scene,
        sources=[
            (
                replace(source, directivity_id="omnidirectional")
                if index == 2
                else source
            )
            for index, source in enumerate(scene.sources)
        ],
    )
    event_set = generate_scene_shoebox_path_events(
        omni_scene,
        source_index=2,
    )

    assert event_set.scene_id == omni_scene.scene_id
    assert event_set.source_id == omni_scene.sources[2].transducer_id
    assert event_set.receiver_id == omni_scene.receivers[0].transducer_id
    assert {event.surface_ids[0] for event in event_set.events[1:]} == {
        surface.surface_id for surface in omni_scene.surfaces
    }


@pytest.mark.parametrize("max_order,expected_count", [(0, 1), (1, 7), (2, 25), (4, 129)])
def test_higher_order_shoebox_enumerates_the_integer_image_lattice(
    max_order,
    expected_count,
):
    event_set = _events(max_order=max_order)

    excluded_count = len(
        event_set.metadata["excluded_edge_or_corner_image_orders"]
    )
    assert len(event_set.events) + excluded_count == expected_count
    assert event_set.metadata["enumerated_image_count"] == expected_count
    assert max(
        sum(abs(value) for value in event.image_order_xyz)
        for event in event_set.events
    ) == max_order
    assert all(
        len(event.surface_ids)
        == sum(abs(value) for value in event.image_order_xyz)
        for event in event_set.events
    )


def test_higher_order_path_records_chronological_surfaces_and_exact_distance():
    event_set = _events(max_order=4)

    by_order = {
        tuple(event.image_order_xyz): event for event in event_set.events
    }
    assert by_order[(2, 0, 0)].surface_ids == ["west", "east"]
    assert by_order[(-2, 0, 0)].surface_ids == ["east", "west"]
    for image_order, event in by_order.items():
        order = np.asarray(image_order)
        translation = np.floor_divide(order + 1, 2)
        parity = np.where(order % 2 == 0, 1.0, -1.0)
        image = 2.0 * translation * DIMENSIONS + parity * SOURCE
        expected_distance = float(np.linalg.norm(RECEIVER - image))
        vertices = [
            SOURCE,
            *[
                np.asarray(point)
                for point in event.interaction_points_m
            ],
            RECEIVER,
        ]
        folded_distance = sum(
            np.linalg.norm(second - first)
            for first, second in zip(vertices, vertices[1:])
        )

        assert event.distance_m == pytest.approx(expected_distance, abs=1e-12)
        assert folded_distance == pytest.approx(expected_distance, abs=1e-12)


def test_higher_order_source_receiver_swap_reverses_the_ordered_path():
    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    models = {boundary: model for boundary in BOUNDARY_AXIS_PLANE_NORMAL}
    forward = _events(
        max_order=3,
        boundary_admittance_models=models,
        reflection_frequencies_hz=[80.0, 160.0, 240.0],
    )
    reverse = generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=RECEIVER,
        receiver_position_m=SOURCE,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="unit-room",
        source_id="mic",
        receiver_id="talker",
        max_order=3,
        boundary_admittance_models=models,
        reflection_frequencies_hz=[80.0, 160.0, 240.0],
    )
    reverse_by_order = {
        tuple(event.image_order_xyz): event for event in reverse.events
    }

    for event in forward.events:
        reciprocal = reverse_by_order[
            tuple(
                value if abs(value) % 2 else -value
                for value in event.image_order_xyz
            )
        ]
        assert reciprocal.distance_m == pytest.approx(event.distance_m, abs=1e-12)
        assert reciprocal.surface_ids == list(reversed(event.surface_ids))
        assert np.asarray(reciprocal.interaction_points_m) == pytest.approx(
            np.asarray(list(reversed(event.interaction_points_m))),
            abs=1e-12,
        )
        assert reciprocal.departure_direction_unit == pytest.approx(
            -np.asarray(event.arrival_direction_unit), abs=1e-12
        )
        assert reciprocal.arrival_direction_unit == pytest.approx(
            -np.asarray(event.departure_direction_unit), abs=1e-12
        )
        assert reciprocal.gain_spectrum.values == pytest.approx(
            event.gain_spectrum.values, abs=1e-12
        )


def test_higher_order_event_transfer_matches_existing_analytic_images():
    from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
        _image_source_transfer,
        _shoebox_image_geometry,
    )

    model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    models = {boundary: model for boundary in BOUNDARY_AXIS_PLANE_NORMAL}
    frequencies = np.asarray([80.0, 120.0, 160.0, 200.0, 240.0])
    generic_source = np.asarray([1.13, 1.27, 1.19])
    generic_receiver = np.asarray([3.74, 2.83, 1.67])
    event_set = generate_shoebox_path_events(
        dimensions_m=DIMENSIONS,
        source_position_m=generic_source,
        receiver_position_m=generic_receiver,
        sound_speed_m_s=SOUND_SPEED,
        scene_id="generic-room",
        source_id="talker",
        receiver_id="mic",
        max_order=4,
        boundary_admittance_models=models,
        reflection_frequencies_hz=frequencies,
    )
    event_transfer = sum(
        event.gain_spectrum.values
        * np.exp(-2j * np.pi * frequencies * event.delay_s)
        for event in event_set.events
    )
    geometry = _shoebox_image_geometry(
        tuple(DIMENSIONS),
        tuple(generic_source),
        tuple(generic_receiver),
        max_order=4,
    )
    analytic = _image_source_transfer(
        geometry,
        frequencies,
        model,
        sound_speed_m_s=SOUND_SPEED,
        reflection_policy="complex_angle",
    )

    assert event_transfer == pytest.approx(analytic, abs=2e-13)


def test_edge_corner_policy_separates_physical_exclusion_from_legacy_diagnostic():
    physical = _events(max_order=4)
    diagnostic = _events(
        max_order=4,
        edge_corner_policy="sequential_face_product_diagnostic",
    )

    assert len(physical.events) == 128
    assert physical.metadata["excluded_edge_or_corner_image_orders"] == [
        [0, -2, -2]
    ]
    assert len(diagnostic.events) == 129
    assert diagnostic.metadata["excluded_edge_or_corner_image_orders"] == []
    assert diagnostic.metadata[
        "included_diagnostic_edge_or_corner_image_orders"
    ] == [[0, -2, -2]]
    corner_event = next(
        event
        for event in diagnostic.events
        if event.image_order_xyz == [0, -2, -2]
    )
    repeated_groups = [
        group
        for index, group in enumerate(corner_event.interaction_group_ids[1:])
        if group == corner_event.interaction_group_ids[index]
    ]
    assert repeated_groups


def _blocking_object() -> SceneObject:
    return SceneObject(
        object_id="cabinet",
        family="cabinet",
        footprint=[
            [2.0, 1.5],
            [3.0, 1.5],
            [3.0, 2.5],
            [2.0, 2.5],
        ],
        z_min=0.0,
        z_max=2.0,
        material_id="wood",
        absorption=0.2,
        scattering=0.3,
    )


def test_vertical_prism_visibility_blocks_direct_and_serializes_provenance():
    event_set = generate_shoebox_path_events(
        dimensions_m=(5.0, 4.0, 3.0),
        source_position_m=(1.0, 2.0, 1.0),
        receiver_position_m=(4.0, 2.0, 1.0),
        sound_speed_m_s=SOUND_SPEED,
        max_order=1,
    )

    resolved = apply_scene_object_visibility(
        event_set,
        [_blocking_object()],
    )
    direct = next(
        event for event in resolved.events if event.path_type == "direct"
    )
    restored = PathEventSet.from_json(resolved.to_json())

    assert direct.visible is False
    assert resolved.metadata["blocked_event_count"] >= 1
    assert resolved.metadata["blocked_event_occluder_ids"][
        direct.event_id
    ] == ["cabinet"]
    assert restored.to_dict() == resolved.to_dict()


def test_vertical_prism_visibility_is_reciprocal_and_height_aware():
    scene_object = _blocking_object()
    assert segment_intersects_scene_object(
        (1.0, 2.0, 1.0),
        (4.0, 2.0, 1.0),
        scene_object,
    )
    assert segment_intersects_scene_object(
        (4.0, 2.0, 1.0),
        (1.0, 2.0, 1.0),
        scene_object,
    )
    assert not segment_intersects_scene_object(
        (1.0, 2.0, 2.5),
        (4.0, 2.0, 2.5),
        scene_object,
    )


def test_opt_in_scene_interactions_add_transmission_diffraction_and_scattering():
    from puresound.audio.rir.contracts import HybridRIRConfig
    from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene

    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (3.0, 3.0)),
        num_near_sources=1,
        num_far_sources=0,
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(
        config,
        seed=9,
        room_type="office",
    )
    object_material_id = scene.surfaces[0].material_id
    object_material = scene.materials[object_material_id]
    centers = object_material.transmission.center_frequencies_hz
    interaction_material = replace(
        object_material,
        scattering=MaterialSpectrum.constant(0.4, centers),
        transmission=MaterialSpectrum.constant(0.25, centers),
    )
    scene = replace(
        scene,
        scene_id="interaction-room",
        sources=[
            replace(
                scene.sources[0],
                pose=Pose([1.0, 2.0, 1.0]),
                directivity_id="omnidirectional",
            )
        ],
        receivers=[
            replace(
                scene.receivers[0],
                pose=Pose([4.0, 2.0, 1.0]),
                directivity_id="omnidirectional",
            )
        ],
        materials={
            **scene.materials,
            object_material_id: interaction_material,
        },
        objects=[
            replace(
                _blocking_object(),
                material_id=object_material_id,
                transmission=0.25,
            )
        ],
    )

    event_set = generate_scene_shoebox_path_events(
        scene,
        max_order=1,
        include_scene_interactions=True,
    )
    path_types = [event.path_type for event in event_set.events]
    transmission = next(
        event for event in event_set.events
        if event.path_type == "transmission"
    )
    rir = render_path_events(
        event_set,
        sample_rate_hz=8000,
        num_samples=800,
    )

    assert next(
        event for event in event_set.events
        if event.path_type == "direct"
    ).visible is False
    assert transmission.gain_spectrum.constant_real_value() == (
        pytest.approx(0.5 / 3.0)
    )
    assert transmission.energy_partition_fraction == pytest.approx(0.25)
    assert path_types.count("diffraction") == 2
    assert path_types.count("scattering") > 0
    assert event_set.metadata["transmission_event_count"] == 1
    assert event_set.metadata["diffraction_event_count"] == 2
    assert np.all(np.isfinite(rir))
    assert np.any(rir != 0.0)
    assert PathEventSet.from_json(event_set.to_json()).to_dict() == (
        event_set.to_dict()
    )
