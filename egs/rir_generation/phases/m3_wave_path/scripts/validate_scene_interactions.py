#!/usr/bin/env python3
"""Validate M3.13 visibility and M3.14 scoped scene interactions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from puresound.audio.rir.path_events import (
    PathEventSet,
    generate_scene_shoebox_path_events,
    render_path_events,
    segment_scene_object_intersection_interval,
)
from puresound.audio.rir.scene.schema import (
    EnvironmentConfig,
    MaterialSpectrum,
    Pose,
    RoomSceneV2,
    SceneObject,
    SceneSurface,
    SurfaceMaterial,
    TransducerConfig,
    shoebox_surface_vertices,
)


REPORT_SCHEMA_VERSION = "puresound.scene_interactions_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate vertical-prism visibility, transmission, bounded edge "
            "diffraction, and energy-partitioned deterministic scattering."
        )
    )
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _scene(
    *,
    source_position_m: tuple[float, float, float],
    receiver_position_m: tuple[float, float, float],
) -> RoomSceneV2:
    dimensions = (5.0, 4.0, 3.0)
    centers = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    material = SurfaceMaterial(
        material_id="controlled",
        name="Controlled interaction material",
        family="validation",
        absorption=MaterialSpectrum.constant(0.2, centers),
        scattering=MaterialSpectrum.constant(0.4, centers),
        transmission=MaterialSpectrum.constant(0.25, centers),
        provenance="deterministic M3 interaction validation",
    )
    vertices = shoebox_surface_vertices(dimensions)
    surfaces = [
        SceneSurface(
            surface_id=f"{boundary}_boundary",
            boundary=boundary,
            vertices_m=points,
            material_id=material.material_id,
        )
        for boundary, points in vertices.items()
    ]
    scene_object = SceneObject(
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
        material_id=material.material_id,
        absorption=0.2,
        scattering=0.4,
        transmission=0.25,
    )
    return RoomSceneV2(
        scene_id="controlled-interaction-room",
        room_type="validation",
        dimensions_m=list(dimensions),
        surfaces=surfaces,
        materials={material.material_id: material},
        environment=EnvironmentConfig(sound_speed_m_s=343.0),
        sources=[
            TransducerConfig(
                transducer_id="source",
                kind="source",
                pose=Pose(list(source_position_m)),
                power_db_spl_at_1m=80.0,
            )
        ],
        receivers=[
            TransducerConfig(
                transducer_id="receiver",
                kind="receiver",
                pose=Pose(list(receiver_position_m)),
            )
        ],
        objects=[scene_object],
    )


def _event_signature(event_set: PathEventSet) -> list[tuple[str, float, float]]:
    return sorted(
        (
            event.path_type,
            round(event.distance_m, 12),
            round(
                float(np.max(np.abs(event.gain_spectrum.values))),
                12,
            ),
        )
        for event in event_set.events
    )


def main() -> None:
    args = _parse_args()
    forward_scene = _scene(
        source_position_m=(1.0, 2.0, 1.0),
        receiver_position_m=(4.0, 2.0, 1.0),
    )
    reverse_scene = replace(
        forward_scene,
        sources=[
            replace(
                forward_scene.sources[0],
                pose=Pose([4.0, 2.0, 1.0]),
            )
        ],
        receivers=[
            replace(
                forward_scene.receivers[0],
                pose=Pose([1.0, 2.0, 1.0]),
            )
        ],
    )
    lit_scene = _scene(
        source_position_m=(1.0, 0.5, 1.0),
        receiver_position_m=(4.0, 0.5, 1.0),
    )
    forward = generate_scene_shoebox_path_events(
        forward_scene,
        max_order=1,
        include_scene_interactions=True,
    )
    reverse = generate_scene_shoebox_path_events(
        reverse_scene,
        max_order=1,
        include_scene_interactions=True,
    )
    lit = generate_scene_shoebox_path_events(
        lit_scene,
        max_order=1,
        include_scene_interactions=True,
    )
    direct = next(
        event for event in forward.events if event.path_type == "direct"
    )
    lit_direct = next(
        event for event in lit.events if event.path_type == "direct"
    )
    transmission = next(
        event
        for event in forward.events
        if event.path_type == "transmission"
    )
    diffraction = [
        event
        for event in forward.events
        if event.path_type == "diffraction"
    ]
    scattering = [
        event
        for event in forward.events
        if event.path_type == "scattering"
    ]
    energy_partition_sums = {}
    for parent in (
        event
        for event in forward.events
        if event.path_type == "specular_reflection"
    ):
        branches = [
            event
            for event in scattering
            if event.event_id.startswith(f"{parent.event_id}:scatter:")
        ]
        energy_partition_sums[parent.event_id] = float(
            parent.energy_partition_fraction
            + sum(event.energy_partition_fraction for event in branches)
        )
    interval = segment_scene_object_intersection_interval(
        direct.source_position_m,
        direct.receiver_position_m,
        forward_scene.objects[0],
    )
    rendered = render_path_events(
        forward,
        sample_rate_hz=16000,
        num_samples=2400,
    )
    restored = PathEventSet.from_json(forward.to_json())
    maximum_reciprocal_distance_error = max(
        abs(first[1] - second[1])
        for first, second in zip(
            _event_signature(forward),
            _event_signature(reverse),
        )
    )
    visibility_accepted = bool(
        not direct.visible
        and lit_direct.visible
        and interval is not None
        and np.allclose(interval, (1.0 / 3.0, 2.0 / 3.0))
        and forward.metadata["blocked_event_count"] >= 1
    )
    transmission_accepted = bool(
        transmission.visible
        and transmission.energy_partition_fraction == 0.25
        and np.isclose(
            transmission.gain_spectrum.constant_real_value(),
            0.5 / direct.distance_m,
        )
        and len(transmission.interaction_points_m) == 2
    )
    diffraction_accepted = bool(
        len(diffraction) == 2
        and all(event.visible for event in diffraction)
        and all(
            event.distance_m > direct.distance_m for event in diffraction
        )
        and all(
            event.gain_spectrum.constant_real_value()
            <= 0.5 / event.distance_m
            for event in diffraction
        )
    )
    scattering_accepted = bool(
        len(scattering) > 0
        and len(scattering)
        == 4
        * sum(
            event.energy_partition_fraction < 1.0
            for event in forward.events
            if event.path_type == "specular_reflection"
        )
        and all(
            abs(value - 1.0) <= 1e-12
            for value in energy_partition_sums.values()
        )
    )
    reciprocity_accepted = bool(
        len(forward.events) == len(reverse.events)
        and _event_signature(forward) == _event_signature(reverse)
        and maximum_reciprocal_distance_error <= 1e-12
    )
    serialization_accepted = restored.to_dict() == forward.to_dict()
    rendering_accepted = bool(
        np.all(np.isfinite(rendered)) and np.any(rendered != 0.0)
    )
    accepted = bool(
        visibility_accepted
        and transmission_accepted
        and diffraction_accepted
        and scattering_accepted
        and reciprocity_accepted
        and serialization_accepted
        and rendering_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "configuration": {
            "visibility_geometry": "closed_vertical_prisms",
            "scattering_reference_hz": 1000.0,
            "scattering_branch_count": 4,
            "diffraction_reference_hz": 1000.0,
            "maximum_diffraction_edges_per_object": 2,
        },
        "geometry": {
            "blocked_direct_visible": direct.visible,
            "lit_direct_visible": lit_direct.visible,
            "direct_object_interval": list(interval or ()),
            "blocked_event_count": forward.metadata[
                "blocked_event_count"
            ],
        },
        "interactions": {
            "transmission_event_count": forward.metadata[
                "transmission_event_count"
            ],
            "diffraction_event_count": len(diffraction),
            "scattering_event_count": len(scattering),
            "scattering_energy_partition_sums": energy_partition_sums,
            "transmission_pressure_gain": (
                transmission.gain_spectrum.constant_real_value()
            ),
        },
        "reciprocity": {
            "forward_event_count": len(forward.events),
            "reverse_event_count": len(reverse.events),
            "maximum_distance_error_m": (
                maximum_reciprocal_distance_error
            ),
        },
        "acceptance": {
            "visibility_accepted": visibility_accepted,
            "transmission_accepted": transmission_accepted,
            "diffraction_accepted": diffraction_accepted,
            "controlled_scattering_accepted": scattering_accepted,
            "reciprocity_accepted": reciprocity_accepted,
            "serialization_accepted": serialization_accepted,
            "rendering_accepted": rendering_accepted,
            "m3_scene_interactions_accepted": accepted,
            "production_default_changed": False,
        },
        "scope": {
            "general_mesh_tracer": False,
            "vertical_prism_furniture": True,
            "straight_through_energy_transmission": True,
            "reference_frequency_shadow_edge_diffraction": True,
            "first_order_deterministic_surface_scattering": True,
            "models_are_opt_in": True,
        },
        "decision": (
            "accept_opt_in_m3_scene_interactions"
            if accepted
            else "reject_m3_scene_interactions"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "acceptance": report["acceptance"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )
    if not accepted:
        raise RuntimeError("M3 scene interaction gate failed")


if __name__ == "__main__":
    main()
