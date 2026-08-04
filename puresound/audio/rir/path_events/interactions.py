"""Transmission, edge diffraction, and controlled diffuse scattering.

Turns a specular path set into one that also carries object transmission,
bounded knife-edge detours, and an exact ``(1 - s)`` / ``s / N`` energy split.
Moved out of ``puresound.audio.rir.path_events`` in R3 of
``RIR_EXP_LOG.md``; kept apart from ``geometry`` because the
interaction model is 500+ lines on its own.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from puresound.audio.rir.path_events.geometry import (
    _BOUNDARY_GEOMETRY,
    _path_event_vertices,
    segment_intersects_scene_object,
    segment_scene_object_intersection_interval,
)
from puresound.audio.rir.path_events.schema import (
    ComplexPathGainSpectrum,
    NormalizedAdmittanceModel,
    PathEvent,
    PathEventSet,
    locally_reacting_reflection_coefficient,
)
from puresound.audio.rir.scene.schema import RoomSceneV2, SceneObject


def _gain_spectrum(
    *,
    distance_m: float,
    incidence_cosines: Iterable[float],
    frequencies_hz: list[float],
    admittance_models: Iterable[NormalizedAdmittanceModel | None],
) -> ComplexPathGainSpectrum:
    cosines = [float(value) for value in incidence_cosines]
    models = list(admittance_models)
    if len(cosines) != len(models):
        raise ValueError("incidence cosines and admittance models must match")
    if not cosines:
        return ComplexPathGainSpectrum.constant(
            1.0 / float(distance_m),
            provenance=(
                "free_field_1_over_r; propagation delay is stored separately"
            ),
        )
    if all(model is None for model in models):
        return ComplexPathGainSpectrum.constant(
            1.0 / float(distance_m),
            provenance=(
                "free_field_1_over_r_times_rigid_specular_reflection_reference"
            ),
        )
    reflections = []
    for frequency in frequencies_hz:
        reflection = complex(1.0, 0.0)
        for incidence_cosine, admittance_model in zip(cosines, models):
            if admittance_model is not None:
                reflection *= locally_reacting_reflection_coefficient(
                    admittance_model.normalized_admittance(frequency),
                    incidence_cosine,
                )
        reflections.append(reflection)
    gains = [reflection / float(distance_m) for reflection in reflections]
    return ComplexPathGainSpectrum(
        frequencies_hz=frequencies_hz,
        real=[float(value.real) for value in gains],
        imag=[float(value.imag) for value in gains],
        provenance=(
            "free_field_1_over_r_times_ordered_angle_aware_locally_reacting_"
            "complex_pressure_reflections"
        ),
    )


def _scaled_gain_spectrum(
    spectrum: ComplexPathGainSpectrum,
    scale: float,
    *,
    provenance_suffix: str,
) -> ComplexPathGainSpectrum:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("path gain scale must be finite and non-negative")
    return ComplexPathGainSpectrum(
        frequencies_hz=list(spectrum.frequencies_hz),
        real=[value * item for item in spectrum.real],
        imag=[value * item for item in spectrum.imag],
        provenance=f"{spectrum.provenance}; {provenance_suffix}",
        quantity=spectrum.quantity,
        interpolation=spectrum.interpolation,
    )


def augment_scene_path_events_with_interactions(
    event_set: PathEventSet,
    scene: RoomSceneV2,
    *,
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    scattering_reference_hz: float = 1000.0,
    scattering_branch_count: int = 4,
    diffraction_reference_hz: float = 1000.0,
    maximum_diffraction_edges_per_object: int = 2,
    include_transmission: bool = True,
    include_diffraction: bool = True,
    include_scattering: bool = True,
) -> PathEventSet:
    """Add scoped M3 furniture transmission, edge diffraction, and scattering.

    Transmission follows the blocked direct segment through all intersected
    vertical prisms. Diffraction uses the two shortest visible vertical-edge
    detours per blocker with a bounded reference-frequency knife-edge gain.
    First-order wall scattering partitions each parent event's energy between
    its specular path and deterministic nearby surface samples. Higher-order
    paths apply every encountered wall's specular retention; their removed
    diffuse share is delegated to the FDN late field to avoid exponential path
    branching.
    """
    if event_set.scene_id != scene.scene_id:
        raise ValueError("event set and scene IDs must match")
    branch_count = int(scattering_branch_count)
    maximum_edges = int(maximum_diffraction_edges_per_object)
    if branch_count != scattering_branch_count or branch_count < 1:
        raise ValueError("scattering branch count must be a positive integer")
    if maximum_edges != maximum_diffraction_edges_per_object or maximum_edges < 1:
        raise ValueError(
            "maximum diffraction edge count must be a positive integer"
        )
    if (
        not math.isfinite(float(scattering_reference_hz))
        or float(scattering_reference_hz) <= 0.0
        or not math.isfinite(float(diffraction_reference_hz))
        or float(diffraction_reference_hz) <= 0.0
    ):
        raise ValueError("interaction reference frequencies must be positive")
    models = dict(boundary_admittance_models or {})
    surface_by_id = {
        surface.surface_id: surface for surface in scene.surfaces
    }
    events: list[PathEvent] = []
    scattering_events: list[PathEvent] = []
    higher_order_scattering_loss_events = 0
    for event in event_set.events:
        if (
            include_scattering
            and event.visible
            and event.path_type == "specular_reflection"
            and len(event.surface_ids) > 1
            and all(surface_id in surface_by_id for surface_id in event.surface_ids)
        ):
            specular_energy_retention = float(
                np.prod(
                    [
                        1.0
                        - float(
                            scene.materials[
                                surface_by_id[surface_id].material_id
                            ].scattering.at(scattering_reference_hz)
                        )
                        for surface_id in event.surface_ids
                    ]
                )
            )
            if specular_energy_retention < 1.0:
                higher_order_scattering_loss_events += 1
                events.append(
                    replace(
                        event,
                        gain_spectrum=_scaled_gain_spectrum(
                            event.gain_spectrum,
                            math.sqrt(max(0.0, specular_energy_retention)),
                            provenance_suffix=(
                                "multi-order specular retention; diffuse share "
                                "delegated to the FDN late field"
                            ),
                        ),
                        energy_partition_fraction=(
                            event.energy_partition_fraction
                            * specular_energy_retention
                        ),
                        scattering_model=(
                            "puresound.multi_order_specular_retention_fdn_diffuse.v1"
                        ),
                    )
                )
                continue
        if (
            include_scattering
            and event.visible
            and event.path_type == "specular_reflection"
            and len(event.surface_ids) == 1
            and event.surface_ids[0] in surface_by_id
        ):
            surface = surface_by_id[event.surface_ids[0]]
            material = scene.materials[surface.material_id]
            scattering = float(
                material.scattering.at(scattering_reference_hz)
            )
            if scattering > 0.0:
                specular_scale = math.sqrt(max(0.0, 1.0 - scattering))
                events.append(
                    replace(
                        event,
                        gain_spectrum=_scaled_gain_spectrum(
                            event.gain_spectrum,
                            specular_scale,
                            provenance_suffix=(
                                "specular energy partition "
                                f"(1-s)={1.0 - scattering:.9g}"
                            ),
                        ),
                        energy_partition_fraction=(
                            event.energy_partition_fraction
                            * (1.0 - scattering)
                        ),
                    )
                )
                original_point = np.asarray(
                    event.interaction_points_m[0],
                    dtype=np.float64,
                )
                vertices = np.asarray(surface.vertices_m, dtype=np.float64)
                axis, _upper, normal_values = _BOUNDARY_GEOMETRY[
                    surface.boundary
                ]
                tangential_axes = [
                    value for value in range(3) if value != axis
                ]
                spans = np.ptp(vertices[:, tangential_axes], axis=0)
                radius = 0.12 * float(np.min(spans))
                normal = np.asarray(normal_values, dtype=np.float64)
                for branch in range(branch_count):
                    angle = (
                        2.0 * math.pi * (branch + 0.5) / branch_count
                    )
                    point = original_point.copy()
                    point[tangential_axes[0]] += radius * math.cos(angle)
                    point[tangential_axes[1]] += radius * math.sin(angle)
                    for tangent_axis in tangential_axes:
                        lower = float(np.min(vertices[:, tangent_axis]))
                        upper = float(np.max(vertices[:, tangent_axis]))
                        epsilon = min(1e-6, 0.1 * (upper - lower))
                        point[tangent_axis] = float(
                            np.clip(
                                point[tangent_axis],
                                lower + epsilon,
                                upper - epsilon,
                            )
                        )
                    first = point - np.asarray(
                        event.source_position_m,
                        dtype=np.float64,
                    )
                    second = np.asarray(
                        event.receiver_position_m,
                        dtype=np.float64,
                    ) - point
                    first_length = float(np.linalg.norm(first))
                    second_length = float(np.linalg.norm(second))
                    distance = first_length + second_length
                    departure = first / first_length
                    arrival = second / second_length
                    cosine = abs(float(np.dot(departure, normal)))
                    partition = scattering / branch_count
                    base_gain = _gain_spectrum(
                        distance_m=distance,
                        incidence_cosines=[cosine],
                        frequencies_hz=list(
                            event.gain_spectrum.frequencies_hz
                        ),
                        admittance_models=[
                            models.get(event.surface_ids[0])
                        ],
                    )
                    candidate = PathEvent(
                        event_id=(
                            f"{event.event_id}:scatter:{branch:02d}"
                        ),
                        source_id=event.source_id,
                        receiver_id=event.receiver_id,
                        path_type="scattering",
                        source_position_m=list(event.source_position_m),
                        receiver_position_m=list(
                            event.receiver_position_m
                        ),
                        distance_m=distance,
                        delay_s=distance / event.sound_speed_m_s,
                        sound_speed_m_s=event.sound_speed_m_s,
                        departure_direction_unit=departure.tolist(),
                        arrival_direction_unit=arrival.tolist(),
                        surface_ids=list(event.surface_ids),
                        interaction_types=["scattering"],
                        interaction_group_ids=[0],
                        interaction_points_m=[point.tolist()],
                        incidence_cosines=[cosine],
                        gain_spectrum=_scaled_gain_spectrum(
                            base_gain,
                            math.sqrt(partition),
                            provenance_suffix=(
                                "deterministic diffuse energy partition "
                                f"s/N={partition:.9g}"
                            ),
                        ),
                        source_directivity_id=(
                            event.source_directivity_id
                        ),
                        receiver_directivity_id=(
                            event.receiver_directivity_id
                        ),
                        source_directivity_gain=(
                            event.source_directivity_gain
                        ),
                        receiver_directivity_gain=(
                            event.receiver_directivity_gain
                        ),
                        energy_partition_fraction=partition,
                        visible=True,
                        diffraction_model="none",
                        scattering_model=(
                            "puresound.deterministic_surface_partition.v1"
                        ),
                    )
                    candidate_vertices = _path_event_vertices(candidate)
                    blocked = any(
                        segment_intersects_scene_object(
                            candidate_vertices[index],
                            candidate_vertices[index + 1],
                            scene_object,
                        )
                        for scene_object in scene.objects
                        for index in range(
                            len(candidate_vertices) - 1
                        )
                    )
                    scattering_events.append(
                        replace(candidate, visible=not blocked)
                    )
                continue
        events.append(event)
    events.extend(scattering_events)

    direct = next(
        (event for event in event_set.events if event.path_type == "direct"),
        None,
    )
    transmission_events: list[PathEvent] = []
    diffraction_events: list[PathEvent] = []
    if direct is not None and not direct.visible:
        source = np.asarray(direct.source_position_m, dtype=np.float64)
        receiver = np.asarray(
            direct.receiver_position_m,
            dtype=np.float64,
        )
        direct_vector = receiver - source
        intersections = []
        for scene_object in scene.objects:
            interval = segment_scene_object_intersection_interval(
                source,
                receiver,
                scene_object,
            )
            if interval is not None:
                intersections.append((interval, scene_object))
        intersections.sort(key=lambda value: value[0][0])
        if include_transmission and intersections:
            energy_transmission = float(
                np.prod(
                    [
                        scene_object.transmission
                        for _interval, scene_object in intersections
                    ]
                )
            )
            if energy_transmission > 0.0:
                points = []
                surface_ids = []
                for interval, scene_object in intersections:
                    points.extend(
                        [
                            (
                                source + interval[0] * direct_vector
                            ).tolist(),
                            (
                                source + interval[1] * direct_vector
                            ).tolist(),
                        ]
                    )
                    surface_ids.extend(
                        [
                            f"{scene_object.object_id}:entry",
                            f"{scene_object.object_id}:exit",
                        ]
                    )
                pressure_transmission = math.sqrt(energy_transmission)
                transmission_events.append(
                    PathEvent(
                        event_id=(
                            f"{direct.event_id}:object_transmission"
                        ),
                        source_id=direct.source_id,
                        receiver_id=direct.receiver_id,
                        path_type="transmission",
                        source_position_m=list(direct.source_position_m),
                        receiver_position_m=list(
                            direct.receiver_position_m
                        ),
                        distance_m=direct.distance_m,
                        delay_s=direct.delay_s,
                        sound_speed_m_s=direct.sound_speed_m_s,
                        departure_direction_unit=list(
                            direct.departure_direction_unit
                        ),
                        arrival_direction_unit=list(
                            direct.arrival_direction_unit
                        ),
                        surface_ids=surface_ids,
                        interaction_types=["transmission"] * len(points),
                        interaction_group_ids=list(range(len(points))),
                        interaction_points_m=points,
                        incidence_cosines=[1.0] * len(points),
                        gain_spectrum=ComplexPathGainSpectrum.constant(
                            pressure_transmission
                            / direct.distance_m,
                            provenance=(
                                "straight_through_vertical_prisms_times_"
                                "sqrt_product_energy_transmission"
                            ),
                        ),
                        source_directivity_id=(
                            direct.source_directivity_id
                        ),
                        receiver_directivity_id=(
                            direct.receiver_directivity_id
                        ),
                        source_directivity_gain=(
                            direct.source_directivity_gain
                        ),
                        receiver_directivity_gain=(
                            direct.receiver_directivity_gain
                        ),
                        energy_partition_fraction=energy_transmission,
                        visible=True,
                        diffraction_model="none",
                        scattering_model="none",
                    )
                )
        if include_diffraction:
            wavelength = (
                direct.sound_speed_m_s / float(diffraction_reference_hz)
            )
            for _interval, scene_object in intersections:
                candidates = []
                footprint = np.asarray(
                    scene_object.footprint,
                    dtype=np.float64,
                )
                for edge_index, edge_xy in enumerate(footprint):
                    source_horizontal = float(
                        np.linalg.norm(edge_xy - source[:2])
                    )
                    receiver_horizontal = float(
                        np.linalg.norm(edge_xy - receiver[:2])
                    )
                    horizontal_sum = (
                        source_horizontal + receiver_horizontal
                    )
                    if horizontal_sum <= 1e-12:
                        continue
                    edge_z = (
                        receiver_horizontal * source[2]
                        + source_horizontal * receiver[2]
                    ) / horizontal_sum
                    edge_z = float(
                        np.clip(
                            edge_z,
                            scene_object.z_min,
                            scene_object.z_max,
                        )
                    )
                    point = np.asarray(
                        [edge_xy[0], edge_xy[1], edge_z],
                        dtype=np.float64,
                    )
                    first = point - source
                    second = receiver - point
                    first_length = float(np.linalg.norm(first))
                    second_length = float(np.linalg.norm(second))
                    distance = first_length + second_length
                    blocked_by_other = any(
                        segment_intersects_scene_object(
                            endpoint_start,
                            endpoint_stop,
                            other,
                        )
                        for other in scene.objects
                        if other.object_id != scene_object.object_id
                        for endpoint_start, endpoint_stop in (
                            (source, point),
                            (point, receiver),
                        )
                    )
                    if blocked_by_other:
                        continue
                    candidates.append(
                        (
                            distance,
                            edge_index,
                            point,
                            first / first_length,
                            second / second_length,
                        )
                    )
                candidates.sort(key=lambda value: (value[0], value[1]))
                for (
                    distance,
                    edge_index,
                    point,
                    departure,
                    arrival,
                ) in candidates[:maximum_edges]:
                    excess = max(0.0, distance - direct.distance_m)
                    fresnel_v = math.sqrt(
                        max(0.0, 2.0 * excess / wavelength)
                    )
                    available_pressure = math.sqrt(
                        max(
                            0.0,
                            1.0
                            - scene_object.absorption
                            - scene_object.transmission,
                        )
                    )
                    coefficient = (
                        0.5
                        * available_pressure
                        / math.sqrt(1.0 + fresnel_v**2)
                    )
                    diffraction_events.append(
                        PathEvent(
                            event_id=(
                                f"{direct.event_id}:diffraction:"
                                f"{scene_object.object_id}:{edge_index}"
                            ),
                            source_id=direct.source_id,
                            receiver_id=direct.receiver_id,
                            path_type="diffraction",
                            source_position_m=list(
                                direct.source_position_m
                            ),
                            receiver_position_m=list(
                                direct.receiver_position_m
                            ),
                            distance_m=distance,
                            delay_s=distance / direct.sound_speed_m_s,
                            sound_speed_m_s=direct.sound_speed_m_s,
                            departure_direction_unit=departure.tolist(),
                            arrival_direction_unit=arrival.tolist(),
                            surface_ids=[
                                f"{scene_object.object_id}:edge:"
                                f"{edge_index}"
                            ],
                            interaction_types=["diffraction"],
                            interaction_group_ids=[0],
                            interaction_points_m=[point.tolist()],
                            incidence_cosines=[1.0],
                            gain_spectrum=(
                                ComplexPathGainSpectrum.constant(
                                    coefficient / distance,
                                    provenance=(
                                        "bounded_reference_frequency_"
                                        "knife_edge_diffraction"
                                    ),
                                )
                            ),
                            source_directivity_id=(
                                direct.source_directivity_id
                            ),
                            receiver_directivity_id=(
                                direct.receiver_directivity_id
                            ),
                            source_directivity_gain=(
                                direct.source_directivity_gain
                            ),
                            receiver_directivity_gain=(
                                direct.receiver_directivity_gain
                            ),
                            visible=coefficient > 0.0,
                            diffraction_model=(
                                "puresound.bounded_fresnel_edge.v1"
                            ),
                            scattering_model="none",
                        )
                    )
    events.extend(transmission_events)
    events.extend(diffraction_events)
    metadata = dict(event_set.metadata)
    metadata.update(
        {
            "interaction_extension": (
                "puresound.scene_interactions.m3.v1"
            ),
            "transmission_event_count": len(transmission_events),
            "diffraction_event_count": len(diffraction_events),
            "scattering_event_count": len(scattering_events),
            "higher_order_scattering_loss_event_count": (
                higher_order_scattering_loss_events
            ),
            "higher_order_scattering_policy": (
                "per_interaction_specular_retention_with_diffuse_share_"
                "delegated_to_fdn"
            ),
            "scattering_reference_hz": float(
                scattering_reference_hz
            ),
            "scattering_branch_count": branch_count,
            "diffraction_reference_hz": float(
                diffraction_reference_hz
            ),
            "maximum_diffraction_edges_per_object": maximum_edges,
            "interaction_models_are_opt_in": True,
        }
    )
    return replace(
        event_set,
        events=events,
        generator=f"{event_set.generator}+scene_interactions.m3.v1",
        metadata=metadata,
    )


__all__ = [
    "augment_scene_path_events_with_interactions",
]
