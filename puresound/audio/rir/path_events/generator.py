"""Shoebox and scene path-event generation.

Enumerates the ordered image-source lattice, attaches angle-aware complex
boundary gains, and applies visibility, interactions and directivity for a
material-first scene.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np

from puresound.audio.rir.path_events.directivity import directivity_pressure_gain
from puresound.audio.rir.path_events.geometry import (
    _ordered_shoebox_image_path,
    _validate_shoebox_position,
    apply_scene_object_visibility,
)
from puresound.audio.rir.path_events.interactions import (
    _gain_spectrum,
    augment_scene_path_events_with_interactions,
)
from puresound.audio.rir.path_events.schema import (
    NormalizedAdmittanceModel,
    PathEvent,
    PathEventSet,
    _finite_float_list,
)
from puresound.audio.rir.scene.schema import RoomSceneV2, SHOEBOX_BOUNDARIES


def generate_shoebox_path_events(
    *,
    dimensions_m: Iterable[float],
    source_position_m: Iterable[float],
    receiver_position_m: Iterable[float],
    sound_speed_m_s: float,
    scene_id: str = "shoebox",
    source_id: str = "source",
    receiver_id: str = "receiver",
    surface_ids: Mapping[str, str] | None = None,
    source_directivity_id: str = "omnidirectional",
    receiver_directivity_id: str = "omnidirectional",
    source_directivity_gain: float = 1.0,
    receiver_directivity_gain: float = 1.0,
    max_order: int = 1,
    edge_corner_policy: str = "exclude",
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    reflection_frequencies_hz: Iterable[float] = (
        60.0,
        80.0,
        120.0,
        160.0,
        200.0,
        240.0,
    ),
) -> PathEventSet:
    """Generate exact ordered shoebox image-source events through ``max_order``."""
    dimensions = np.asarray(
        _finite_float_list("shoebox dimension", dimensions_m), dtype=np.float64
    )
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError("shoebox dimensions must contain three positive values")
    source = _validate_shoebox_position(
        "source position", source_position_m, dimensions
    )
    receiver = _validate_shoebox_position(
        "receiver position", receiver_position_m, dimensions
    )
    if np.array_equal(source, receiver):
        raise ValueError("source and receiver positions must differ")
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    if (
        int(max_order) != max_order
        or int(max_order) < 0
        or int(max_order) > 20
    ):
        raise ValueError("shoebox max_order must be an integer in [0, 20]")
    if edge_corner_policy not in {
        "exclude",
        "sequential_face_product_diagnostic",
    }:
        raise ValueError("unsupported shoebox edge/corner policy")
    frequencies = _finite_float_list(
        "reflection frequency", reflection_frequencies_hz
    )
    if (
        not frequencies
        or frequencies[0] < 0.0
        or any(b <= a for a, b in zip(frequencies, frequencies[1:]))
    ):
        raise ValueError(
            "reflection frequencies must be non-negative and increasing"
        )
    if surface_ids is not None and set(surface_ids) != set(SHOEBOX_BOUNDARIES):
        raise ValueError("surface_ids must define exactly six shoebox boundaries")
    resolved_surface_ids = {
        boundary: (
            str(surface_ids[boundary])
            if surface_ids is not None
            else boundary
        )
        for boundary in SHOEBOX_BOUNDARIES
    }
    admittance_models = dict(boundary_admittance_models or {})
    unknown_boundaries = set(admittance_models).difference(SHOEBOX_BOUNDARIES)
    if unknown_boundaries:
        raise ValueError(
            f"unknown shoebox admittance boundaries: {sorted(unknown_boundaries)}"
        )

    direct_vector = receiver - source
    direct_distance = float(np.linalg.norm(direct_vector))
    direct_direction = direct_vector / direct_distance
    shared = {
        "source_id": source_id,
        "receiver_id": receiver_id,
        "source_position_m": source.astype(float).tolist(),
        "receiver_position_m": receiver.astype(float).tolist(),
        "sound_speed_m_s": sound_speed,
        "source_directivity_id": source_directivity_id,
        "receiver_directivity_id": receiver_directivity_id,
        "source_directivity_gain": float(source_directivity_gain),
        "receiver_directivity_gain": float(receiver_directivity_gain),
        "visible": True,
        "diffraction_model": "none",
        "scattering_model": "specular",
    }
    events = [
        PathEvent(
            event_id=f"{source_id}->{receiver_id}:direct",
            path_type="direct",
            distance_m=direct_distance,
            delay_s=direct_distance / sound_speed,
            departure_direction_unit=direct_direction.astype(float).tolist(),
            arrival_direction_unit=direct_direction.astype(float).tolist(),
            surface_ids=[],
            interaction_types=[],
            interaction_group_ids=[],
            interaction_points_m=[],
            incidence_cosines=[],
            gain_spectrum=_gain_spectrum(
                distance_m=direct_distance,
                incidence_cosines=[],
                frequencies_hz=frequencies,
                admittance_models=[],
            ),
            image_order_xyz=[0, 0, 0],
            **shared,
        )
    ]
    excluded_degenerate_orders: list[list[int]] = []
    included_edge_or_corner_orders: list[list[int]] = []
    reflection_orders = sorted(
        (
            (nx, ny, nz)
            for nx in range(-int(max_order), int(max_order) + 1)
            for ny in range(-int(max_order), int(max_order) + 1)
            for nz in range(-int(max_order), int(max_order) + 1)
            if 0 < abs(nx) + abs(ny) + abs(nz) <= int(max_order)
        ),
        key=lambda order: (
            abs(order[0]) + abs(order[1]) + abs(order[2]),
            order,
        ),
    )
    for image_order in reflection_orders:
        path = _ordered_shoebox_image_path(
            source_position_m=source,
            receiver_position_m=receiver,
            dimensions_m=dimensions,
            image_order_xyz=image_order,
            edge_corner_policy=edge_corner_policy,
        )
        if path is None:
            excluded_degenerate_orders.append(list(image_order))
            continue
        if path["has_edge_or_corner_hit"]:
            included_edge_or_corner_orders.append(list(image_order))
        boundaries = path["boundaries"]
        distance = float(path["distance_m"])
        events.append(
            PathEvent(
                event_id=(
                    f"{source_id}->{receiver_id}:image:"
                    f"{image_order[0]:+d},{image_order[1]:+d},"
                    f"{image_order[2]:+d}"
                ),
                path_type="specular_reflection",
                distance_m=distance,
                delay_s=distance / sound_speed,
                departure_direction_unit=path[
                    "departure_direction_unit"
                ].astype(float).tolist(),
                arrival_direction_unit=path[
                    "arrival_direction_unit"
                ].astype(float).tolist(),
                surface_ids=[
                    resolved_surface_ids[boundary]
                    for boundary in boundaries
                ],
                interaction_types=["reflection"] * len(boundaries),
                interaction_group_ids=path["interaction_group_ids"],
                interaction_points_m=[
                    point.astype(float).tolist()
                    for point in path["interaction_points_m"]
                ],
                incidence_cosines=path["incidence_cosines"],
                gain_spectrum=_gain_spectrum(
                    distance_m=distance,
                    incidence_cosines=path["incidence_cosines"],
                    frequencies_hz=frequencies,
                    admittance_models=[
                        admittance_models.get(boundary)
                        for boundary in boundaries
                    ],
                ),
                image_order_xyz=list(image_order),
                **shared,
            )
        )

    return PathEventSet(
        scene_id=str(scene_id),
        source_id=str(source_id),
        receiver_id=str(receiver_id),
        events=events,
        metadata={
            "geometry": "shoebox_unfolded_image_source_ordered_specular_paths",
            "max_order": int(max_order),
            "dimensions_m": dimensions.astype(float).tolist(),
            "enumerated_image_count": 1 + len(reflection_orders),
            "excluded_edge_or_corner_image_orders": (
                excluded_degenerate_orders
            ),
            "included_diagnostic_edge_or_corner_image_orders": (
                included_edge_or_corner_orders
            ),
            "edge_corner_policy": edge_corner_policy,
            "complex_boundary_spectra_rendered_in_time_domain": False,
        },
    )


def generate_scene_shoebox_path_events(
    scene: RoomSceneV2,
    *,
    source_index: int = 0,
    receiver_index: int = 0,
    max_order: int = 1,
    edge_corner_policy: str = "exclude",
    resolve_object_visibility: bool = True,
    object_visibility_tolerance_m: float = 1e-10,
    include_scene_interactions: bool = False,
    boundary_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    reflection_frequencies_hz: Iterable[float] = (
        60.0,
        80.0,
        120.0,
        160.0,
        200.0,
        240.0,
    ),
) -> PathEventSet:
    """Adapt a :class:`RoomSceneV2` channel to exact shoebox path events."""
    index = int(source_index)
    if index != source_index or not 0 <= index < len(scene.sources):
        raise ValueError("source_index is outside the scene source list")
    receiver_list_index = int(receiver_index)
    if (
        receiver_list_index != receiver_index
        or not 0 <= receiver_list_index < len(scene.receivers)
    ):
        raise ValueError("receiver_index is outside the scene receiver list")
    source = scene.sources[index]
    receiver = scene.receivers[receiver_list_index]
    if source.directivity_id not in {
        "omnidirectional",
        "speech_cardioid",
        "cardioid",
        "hypercardioid",
        "figure_eight",
    }:
        raise NotImplementedError(
            f"unsupported source directivity: {source.directivity_id}"
        )
    if receiver.directivity_id not in {
        "omnidirectional",
        "cardioid",
        "hypercardioid",
        "figure_eight",
    }:
        raise NotImplementedError(
            f"unsupported receiver directivity: {receiver.directivity_id}"
        )
    boundary_surface_ids = {
        surface.boundary: surface.surface_id for surface in scene.surfaces
    }
    event_set = generate_shoebox_path_events(
        dimensions_m=scene.dimensions_m,
        source_position_m=source.pose.position_m,
        receiver_position_m=receiver.pose.position_m,
        sound_speed_m_s=float(scene.environment.sound_speed_m_s),
        scene_id=scene.scene_id,
        source_id=source.transducer_id,
        receiver_id=receiver.transducer_id,
        surface_ids=boundary_surface_ids,
        source_directivity_id=source.directivity_id,
        receiver_directivity_id=receiver.directivity_id,
        max_order=max_order,
        edge_corner_policy=edge_corner_policy,
        boundary_admittance_models=boundary_admittance_models,
        reflection_frequencies_hz=reflection_frequencies_hz,
    )
    if (
        source.directivity_id != "omnidirectional"
        or receiver.directivity_id != "omnidirectional"
    ):
        event_set = replace(
            event_set,
            events=[
                replace(
                    event,
                    source_directivity_gain=float(
                        directivity_pressure_gain(
                            source.directivity_id,
                            source.pose.orientation_ypr_deg,
                            event.departure_direction_unit,
                        )
                    ),
                    receiver_directivity_gain=float(
                        directivity_pressure_gain(
                            receiver.directivity_id,
                            receiver.pose.orientation_ypr_deg,
                            -np.asarray(
                                event.arrival_direction_unit,
                                dtype=np.float64,
                            ),
                        )
                    ),
                )
                for event in event_set.events
            ],
            metadata={
                **event_set.metadata,
                "source_directivity_model": (
                    "first_order_cardioid_pressure_gain"
                    if source.directivity_id == "speech_cardioid"
                    else "first_order_real_pressure_gain"
                ),
                "source_directivity_id": source.directivity_id,
                "source_orientation_ypr_deg": list(
                    source.pose.orientation_ypr_deg
                ),
                "receiver_directivity_model": (
                    "first_order_real_pressure_gain"
                ),
                "receiver_directivity_id": receiver.directivity_id,
                "receiver_orientation_ypr_deg": list(
                    receiver.pose.orientation_ypr_deg
                ),
            },
        )
    if resolve_object_visibility:
        event_set = apply_scene_object_visibility(
            event_set,
            scene.objects,
            tolerance_m=object_visibility_tolerance_m,
        )
    if include_scene_interactions:
        surface_models = {
            surface.surface_id: boundary_admittance_models[
                surface.boundary
            ]
            for surface in scene.surfaces
            if (
                boundary_admittance_models is not None
                and surface.boundary in boundary_admittance_models
            )
        }
        event_set = augment_scene_path_events_with_interactions(
            event_set,
            scene,
            boundary_admittance_models=surface_models,
        )
    return event_set


__all__ = [
    "generate_scene_shoebox_path_events",
    "generate_shoebox_path_events",
]
