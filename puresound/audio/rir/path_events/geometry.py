"""Visibility, segment intersection, and shoebox image-source geometry.

Pure geometry over scene objects and rooms; no gain spectra and no rendering.
Moved out of ``puresound.audio.rir.path_events`` in R3 of
``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from puresound.audio.rir.path_events.schema import (
    PathEvent,
    PathEventSet,
    _vector3,
)
from puresound.audio.rir.scene.schema import SHOEBOX_BOUNDARIES, SceneObject


OBJECT_VISIBILITY_POLICY = (
    "puresound.closed_vertical_prism_segment_visibility.v1"
)


_BOUNDARY_GEOMETRY: dict[str, tuple[int, bool, tuple[float, float, float]]] = {
    "west": (0, False, (-1.0, 0.0, 0.0)),
    "east": (0, True, (1.0, 0.0, 0.0)),
    "south": (1, False, (0.0, -1.0, 0.0)),
    "north": (1, True, (0.0, 1.0, 0.0)),
    "floor": (2, False, (0.0, 0.0, -1.0)),
    "ceiling": (2, True, (0.0, 0.0, 1.0)),
}

_BOUNDARY_BY_AXIS_AND_UPPER = {
    (axis, upper): boundary
    for boundary, (axis, upper, _normal) in _BOUNDARY_GEOMETRY.items()
}


def _point_on_segment_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    vector = end - start
    relative = point - start
    cross = float(vector[0] * relative[1] - vector[1] * relative[0])
    if abs(cross) > tolerance * max(1.0, float(np.linalg.norm(vector))):
        return False
    dot = float(np.dot(relative, vector))
    return bool(
        dot >= -tolerance
        and dot <= float(np.dot(vector, vector)) + tolerance
    )


def _segments_intersect_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    def orientation(
        start: np.ndarray,
        end: np.ndarray,
        point: np.ndarray,
    ) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if (
        ((o1 > tolerance and o2 < -tolerance) or (
            o1 < -tolerance and o2 > tolerance
        ))
        and ((o3 > tolerance and o4 < -tolerance) or (
            o3 < -tolerance and o4 > tolerance
        ))
    ):
        return True
    return any(
        abs(orientation_value) <= tolerance
        and _point_on_segment_2d(
            point,
            segment_start,
            segment_end,
            tolerance=tolerance,
        )
        for orientation_value, point, segment_start, segment_end in (
            (o1, second_start, first_start, first_end),
            (o2, second_end, first_start, first_end),
            (o3, first_start, second_start, second_end),
            (o4, first_end, second_start, second_end),
        )
    )


def segment_intersects_scene_object(
    start_m: Iterable[float],
    end_m: Iterable[float],
    scene_object: SceneObject,
    *,
    tolerance_m: float = 1e-10,
) -> bool:
    """Return whether a closed segment touches a vertical object prism."""
    start = np.asarray(_vector3("segment start", start_m), dtype=np.float64)
    end = np.asarray(_vector3("segment end", end_m), dtype=np.float64)
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("visibility tolerance must be finite and non-negative")
    if float(np.linalg.norm(end - start)) <= tolerance:
        raise ValueError("visibility segment must have nonzero length")

    delta_z = float(end[2] - start[2])
    if abs(delta_z) <= tolerance:
        if (
            start[2] < scene_object.z_min - tolerance
            or start[2] > scene_object.z_max + tolerance
        ):
            return False
        parameter_min = 0.0
        parameter_max = 1.0
    else:
        first = (scene_object.z_min - float(start[2])) / delta_z
        second = (scene_object.z_max - float(start[2])) / delta_z
        parameter_min = max(0.0, min(first, second))
        parameter_max = min(1.0, max(first, second))
        if parameter_min > parameter_max + tolerance:
            return False

    clipped_start = (
        start[:2] + parameter_min * (end[:2] - start[:2])
    )
    clipped_end = (
        start[:2] + parameter_max * (end[:2] - start[:2])
    )
    polygon = np.asarray(scene_object.footprint, dtype=np.float64)
    if (
        scene_object.contains_xy(clipped_start)
        or scene_object.contains_xy(clipped_end)
    ):
        return True
    return any(
        _segments_intersect_2d(
            clipped_start,
            clipped_end,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
            tolerance=tolerance,
        )
        for index in range(len(polygon))
    )


def segment_scene_object_intersection_interval(
    start_m: Iterable[float],
    end_m: Iterable[float],
    scene_object: SceneObject,
    *,
    tolerance_m: float = 1e-10,
) -> tuple[float, float] | None:
    """Return the first/last segment parameters lying inside a prism."""
    start = np.asarray(_vector3("segment start", start_m), dtype=np.float64)
    end = np.asarray(_vector3("segment end", end_m), dtype=np.float64)
    delta = end - start
    tolerance = float(tolerance_m)
    if float(np.linalg.norm(delta)) <= tolerance:
        raise ValueError("intersection segment must have nonzero length")
    candidates = [0.0, 1.0]
    if abs(float(delta[2])) > tolerance:
        for height in (scene_object.z_min, scene_object.z_max):
            parameter = (height - float(start[2])) / float(delta[2])
            if -tolerance <= parameter <= 1.0 + tolerance:
                candidates.append(float(np.clip(parameter, 0.0, 1.0)))

    def cross2(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    polygon = np.asarray(scene_object.footprint, dtype=np.float64)
    ray_xy = delta[:2]
    ray_length_squared = float(np.dot(ray_xy, ray_xy))
    for index in range(len(polygon)):
        edge_start = polygon[index]
        edge_vector = polygon[(index + 1) % len(polygon)] - edge_start
        offset = edge_start - start[:2]
        denominator = cross2(ray_xy, edge_vector)
        if abs(denominator) > tolerance:
            parameter = cross2(offset, edge_vector) / denominator
            edge_parameter = cross2(offset, ray_xy) / denominator
            if (
                -tolerance <= parameter <= 1.0 + tolerance
                and -tolerance <= edge_parameter <= 1.0 + tolerance
            ):
                candidates.append(
                    float(np.clip(parameter, 0.0, 1.0))
                )
        elif (
            ray_length_squared > tolerance**2
            and abs(cross2(offset, ray_xy)) <= tolerance
        ):
            for vertex in (
                edge_start,
                polygon[(index + 1) % len(polygon)],
            ):
                parameter = float(
                    np.dot(vertex - start[:2], ray_xy)
                    / ray_length_squared
                )
                if -tolerance <= parameter <= 1.0 + tolerance:
                    candidates.append(
                        float(np.clip(parameter, 0.0, 1.0))
                    )
    candidates = sorted(candidates)
    unique = []
    for value in candidates:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    inside_intervals = []
    for lower, upper in zip(unique, unique[1:]):
        if upper - lower <= tolerance:
            continue
        midpoint = start + 0.5 * (lower + upper) * delta
        if scene_object.contains_point(midpoint):
            inside_intervals.append((lower, upper))
    if not inside_intervals:
        return None
    return (
        float(inside_intervals[0][0]),
        float(inside_intervals[-1][1]),
    )


def _path_event_vertices(event: PathEvent) -> list[np.ndarray]:
    vertices = [
        np.asarray(event.source_position_m, dtype=np.float64),
    ]
    previous_group: int | None = None
    for group, point in zip(
        event.interaction_group_ids,
        event.interaction_points_m,
    ):
        if group != previous_group:
            vertices.append(np.asarray(point, dtype=np.float64))
            previous_group = group
    vertices.append(
        np.asarray(event.receiver_position_m, dtype=np.float64)
    )
    return vertices


def _object_bounds(
    scene_object: SceneObject,
    tolerance: float,
) -> tuple[float, float, float, float, float, float]:
    """Axis-aligned bounds of one prism, grown by the visibility tolerance."""
    footprint = np.asarray(scene_object.footprint, dtype=np.float64)
    return (
        float(scene_object.z_min) - tolerance,
        float(scene_object.z_max) + tolerance,
        float(footprint[:, 0].min()) - tolerance,
        float(footprint[:, 0].max()) + tolerance,
        float(footprint[:, 1].min()) - tolerance,
        float(footprint[:, 1].max()) + tolerance,
    )


def _segment_misses_bounds(
    start: np.ndarray,
    end: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> bool:
    """Conservatively reject a segment that cannot reach a prism's bounds.

    A true answer means the exact test would have returned ``False``, so this
    only ever skips work.  Safe on both axes: the slab test is the same one the
    exact test opens with, and the XY test uses the *unclipped* segment, whose
    bounding box contains the z-clipped segment the exact test goes on to use.

    Worth having because almost nothing intersects: on a sampled office scene,
    526,304 segment-object tests produced 3,303 hits, and this rejects 94.7% of
    them before any polygon arithmetic.
    """
    z_min, z_max, x_min, x_max, y_min, y_max = bounds
    if max(start[2], end[2]) < z_min or min(start[2], end[2]) > z_max:
        return True
    if max(start[0], end[0]) < x_min or min(start[0], end[0]) > x_max:
        return True
    return bool(max(start[1], end[1]) < y_min or min(start[1], end[1]) > y_max)


def apply_scene_object_visibility(
    event_set: PathEventSet,
    scene_objects: Iterable[SceneObject],
    *,
    tolerance_m: float = 1e-10,
) -> PathEventSet:
    """Mark paths blocked by any serialized vertical-prism scene object."""
    objects = list(scene_objects)
    tolerance = float(tolerance_m)
    bounds_by_object = [_object_bounds(obj, tolerance) for obj in objects]
    blocked_by_event: dict[str, list[str]] = {}
    resolved_events = []
    for event in event_set.events:
        vertices = _path_event_vertices(event)
        segments = [
            (vertices[index], vertices[index + 1])
            for index in range(len(vertices) - 1)
        ]
        blocked = []
        for scene_object, bounds in zip(objects, bounds_by_object):
            if any(
                not _segment_misses_bounds(start, end, bounds)
                and segment_intersects_scene_object(
                    start,
                    end,
                    scene_object,
                    tolerance_m=tolerance_m,
                )
                for start, end in segments
            ):
                blocked.append(scene_object.object_id)
        if blocked:
            blocked_by_event[event.event_id] = blocked
        resolved_events.append(
            replace(
                event,
                visible=bool(event.visible and not blocked),
            )
        )
    metadata = dict(event_set.metadata)
    metadata.update(
        {
            "object_visibility_policy": OBJECT_VISIBILITY_POLICY,
            "object_visibility_tolerance_m": float(tolerance_m),
            "object_count": len(objects),
            "blocked_event_count": len(blocked_by_event),
            "visible_event_count": sum(
                event.visible for event in resolved_events
            ),
            "blocked_event_occluder_ids": blocked_by_event,
        }
    )
    return replace(
        event_set,
        events=resolved_events,
        generator=(
            f"{event_set.generator}+vertical_prism_visibility.v1"
        ),
        metadata=metadata,
    )


def _validate_shoebox_position(
    name: str,
    position_m: Iterable[float],
    dimensions_m: np.ndarray,
) -> np.ndarray:
    position = np.asarray(_vector3(name, position_m), dtype=np.float64)
    if np.any(position <= 0.0) or np.any(position >= dimensions_m):
        raise ValueError(f"{name} must lie strictly inside the shoebox")
    return position


def _fold_shoebox_position(
    unfolded_position_m: np.ndarray,
    dimensions_m: np.ndarray,
) -> np.ndarray:
    period = 2.0 * dimensions_m
    wrapped = np.mod(unfolded_position_m, period)
    return np.where(wrapped <= dimensions_m, wrapped, period - wrapped)


def _shoebox_image_position(
    source_position_m: np.ndarray,
    dimensions_m: np.ndarray,
    image_order_xyz: tuple[int, int, int],
) -> np.ndarray:
    order = np.asarray(image_order_xyz, dtype=np.int64)
    translation = np.floor_divide(order + 1, 2)
    parity = np.where(order % 2 == 0, 1.0, -1.0)
    return 2.0 * translation * dimensions_m + parity * source_position_m


def _ordered_shoebox_image_path(
    *,
    source_position_m: np.ndarray,
    receiver_position_m: np.ndarray,
    dimensions_m: np.ndarray,
    image_order_xyz: tuple[int, int, int],
    edge_corner_policy: str,
    tie_tolerance: float = 1e-10,
) -> dict[str, Any] | None:
    """Fold one unfolded image ray into ordered physical reflection points.

    A simultaneous crossing of two or three unfolded room planes is an
    edge/corner hit. The physical ``exclude`` policy returns ``None`` because a
    sequence of locally reacting face reflections is not defined there. The
    ``sequential_face_product_diagnostic`` policy preserves the old analytic
    image-source approximation by grouping coincident face interactions at one
    point; it is traceable but is not a production corner model.
    """
    if edge_corner_policy not in {
        "exclude",
        "sequential_face_product_diagnostic",
    }:
        raise ValueError("unsupported shoebox edge/corner policy")
    image = _shoebox_image_position(
        source_position_m,
        dimensions_m,
        image_order_xyz,
    )
    unfolded_vector = receiver_position_m - image
    distance = float(np.linalg.norm(unfolded_vector))
    crossings: list[tuple[float, int, int]] = []
    for axis, (image_value, receiver_value, dimension) in enumerate(
        zip(image, receiver_position_m, dimensions_m)
    ):
        lower = min(float(image_value), float(receiver_value))
        upper = max(float(image_value), float(receiver_value))
        first_plane = math.floor(lower / float(dimension)) + 1
        last_plane = math.ceil(upper / float(dimension)) - 1
        for plane_index in range(first_plane, last_plane + 1):
            plane = float(plane_index) * float(dimension)
            parameter = (
                (plane - float(image_value))
                / (float(receiver_value) - float(image_value))
            )
            if 0.0 < parameter < 1.0:
                crossings.append((float(parameter), axis, plane_index))
    expected_order = sum(abs(value) for value in image_order_xyz)
    if len(crossings) != expected_order:
        raise RuntimeError(
            "unfolded shoebox crossing count does not match image order"
        )
    crossings.sort(key=lambda item: item[0])
    crossing_groups: list[list[tuple[float, int, int]]] = []
    for crossing in crossings:
        if (
            crossing_groups
            and abs(crossing[0] - crossing_groups[-1][0][0])
            <= tie_tolerance
        ):
            crossing_groups[-1].append(crossing)
        else:
            crossing_groups.append([crossing])
    has_edge_or_corner_hit = any(
        len(group) > 1 for group in crossing_groups
    )
    if has_edge_or_corner_hit and edge_corner_policy == "exclude":
        return None

    boundaries = []
    points = []
    interaction_group_ids = []
    unique_points = []
    crossing_axes = []
    for group_index, group in enumerate(crossing_groups):
        parameter = float(group[0][0])
        unfolded_point = image + parameter * unfolded_vector
        common_point = _fold_shoebox_position(
            unfolded_point,
            dimensions_m,
        )
        group.sort(key=lambda item: item[1])
        for _parameter, axis, plane_index in group:
            point = common_point.copy()
            upper = bool(plane_index % 2)
            boundary = _BOUNDARY_BY_AXIS_AND_UPPER[(axis, upper)]
            point[axis] = float(dimensions_m[axis]) if upper else 0.0
            common_point[axis] = point[axis]
            boundaries.append(boundary)
            points.append(point)
            crossing_axes.append(axis)
            interaction_group_ids.append(group_index)
        unique_points.append(common_point)
    physical_vertices = [
        source_position_m,
        *unique_points,
        receiver_position_m,
    ]
    segments = [
        physical_vertices[index + 1] - physical_vertices[index]
        for index in range(len(physical_vertices) - 1)
    ]
    segment_lengths = [float(np.linalg.norm(segment)) for segment in segments]
    if any(length <= 0.0 for length in segment_lengths):
        raise RuntimeError("shoebox path contains a zero-length segment")
    folded_distance = float(sum(segment_lengths))
    if not math.isclose(
        folded_distance,
        distance,
        rel_tol=1e-11,
        abs_tol=1e-11,
    ):
        raise RuntimeError("folded and unfolded shoebox distances disagree")
    directions = [
        segment / length for segment, length in zip(segments, segment_lengths)
    ]
    incidence_cosines = [
        abs(float(unfolded_vector[axis])) / distance
        for axis in crossing_axes
    ]
    return {
        "image_position_m": image,
        "distance_m": distance,
        "boundaries": boundaries,
        "interaction_points_m": points,
        "interaction_group_ids": interaction_group_ids,
        "incidence_cosines": incidence_cosines,
        "departure_direction_unit": directions[0],
        "arrival_direction_unit": directions[-1],
        "has_edge_or_corner_hit": has_edge_or_corner_hit,
    }


__all__ = [
    "OBJECT_VISIBILITY_POLICY",
    "apply_scene_object_visibility",
    "segment_intersects_scene_object",
    "segment_scene_object_intersection_interval",
]
