#!/usr/bin/env python3
"""Evaluate the installed mesh-capable acoustic engine for M3.12."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_SCHEMA_VERSION = "puresound.mesh_engine_evaluation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-convex visibility smoke and record the M3 mesh-engine "
            "integration decision."
        )
    )
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _pyroomacoustics_smoke() -> dict[str, Any]:
    if importlib.util.find_spec("pyroomacoustics") is None:
        return {
            "installed": False,
            "smoke_completed": False,
            "reason": "optional dependency is not installed",
        }
    import pyroomacoustics as pra

    # Counter-clockwise L-shaped room: this exercises the general polygon
    # visibility engine rather than the optimized shoebox image enumeration.
    corners = np.asarray(
        [
            [0.0, 3.0, 3.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    room = pra.Room.from_corners(
        corners,
        fs=8000,
        max_order=2,
        materials=pra.Material(
            energy_absorption=0.2,
            scattering=0.1,
        ),
    )
    room.extrude(2.4)
    room.add_source([0.45, 2.30, 1.20])
    room.add_microphone_array(
        np.asarray([[2.45], [0.45], [1.20]], dtype=np.float64)
    )
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64)
    visibility = room.visibility[0]
    return {
        "installed": True,
        "version": str(pra.__version__),
        "smoke_completed": True,
        "room_dimension": int(room.dim),
        "wall_count": len(room.walls),
        "non_convex_visibility_available": visibility is not None,
        "visible_image_count": int(np.sum(visibility)),
        "rir_sample_count": int(rir.size),
        "rir_all_finite": bool(np.all(np.isfinite(rir))),
        "rir_nonzero": bool(np.any(rir != 0.0)),
    }


def main() -> None:
    args = _parse_args()
    pyroomacoustics = _pyroomacoustics_smoke()
    accepted_as_cross_check = bool(
        pyroomacoustics.get("installed")
        and pyroomacoustics.get("smoke_completed")
        and pyroomacoustics.get("non_convex_visibility_available")
        and pyroomacoustics.get("rir_all_finite")
        and pyroomacoustics.get("rir_nonzero")
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "engines": {
            "pyroomacoustics": {
                **pyroomacoustics,
                "general_3d_polygon_walls": True,
                "non_convex_outer_room_visibility": True,
                "frequency_dependent_energy_absorption": True,
                "controlled_energy_scattering": True,
                "complex_locally_reacting_pressure_filter": False,
                "surface_transmission_path_events": False,
                "edge_diffraction_path_events": False,
                "interior_solid_furniture_public_contract": False,
                "path_event_serialization_compatible": False,
            },
            "trimesh": {
                "installed": (
                    importlib.util.find_spec("trimesh") is not None
                ),
                "role": "geometry-only ray intersection candidate",
                "acoustic_path_model": False,
            },
        },
        "requirements": {
            "shoebox_boundary_meshes": True,
            "interior_furniture_occlusion": True,
            "ordered_path_interaction_points": True,
            "complex_causal_material_reflection": True,
            "surface_transmission": True,
            "edge_diffraction": True,
            "deterministic_serialized_path_events": True,
        },
        "acceptance": {
            "existing_engine_smoke_accepted": accepted_as_cross_check,
            "existing_engine_accepted_as_production_path_event_backend": False,
            "evaluation_completed_before_new_geometry_code": True,
        },
        "decision": (
            "use_pyroomacoustics_as_visibility_cross_check_and_keep_"
            "puresound_path_events_authoritative"
            if accepted_as_cross_check
            else "keep_puresound_path_events_authoritative"
        ),
        "rationale": [
            (
                "Pyroomacoustics already supplies arbitrary 3D polygon walls "
                "and non-convex room visibility, so it is a useful independent "
                "cross-check."
            ),
            (
                "Its public acoustic contract does not expose the ordered "
                "complex reflection, furniture transmission, diffraction, "
                "and deterministic PathEvent serialization required here."
            ),
            (
                "The scoped implementation therefore adds exact vertical-"
                "prism segment visibility to existing inspectable PathEvents "
                "instead of creating another general-purpose ray tracer."
            ),
        ],
        "next_action": (
            "implement reciprocal source/reflection/receiver segment tests "
            "against serialized SceneObject vertical prisms"
        ),
    }
    if not accepted_as_cross_check:
        raise RuntimeError("installed mesh-engine smoke did not pass")
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
                "decision": report["decision"],
                "acceptance": report["acceptance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
