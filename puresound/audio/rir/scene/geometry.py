"""Pure polygon and shoebox geometry helpers.

Layer 2 of the RIR package.  These functions take and return plain NumPy
arrays: no scene objects, no configuration, no randomness.  ``puresound.audio.rir.render.hybrid`` still re-exports them under their
original private names.
"""

from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def point_in_polygon(point: ArrayLike, polygon: np.ndarray) -> bool:
    x, y = np.asarray(point, dtype=np.float64)[:2]
    inside = False
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon: np.ndarray) -> float:
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-12
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if any(point_in_polygon(point, b) for point in a):
        return True
    if any(point_in_polygon(point, a) for point in b):
        return True
    for idx in range(a.shape[0]):
        a0 = a[idx]
        a1 = a[(idx + 1) % a.shape[0]]
        for jdx in range(b.shape[0]):
            if segments_intersect(a0, a1, b[jdx], b[(jdx + 1) % b.shape[0]]):
                return True
    return False


def distance_point_to_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * seg)))


def distance_point_to_polygon(point: ArrayLike, polygon: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64)[:2]
    distances = [
        distance_point_to_segment(p, polygon[idx], polygon[(idx + 1) % polygon.shape[0]])
        for idx in range(polygon.shape[0])
    ]
    return float(min(distances))


def max_room_horizontal_distance_from_point(
    room_dim: np.ndarray,
    point: np.ndarray,
    margin: float,
) -> float:
    lower = np.full(2, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim[:2] - float(margin), lower + 0.01)
    corners = np.array(
        [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])],
        dtype=np.float64,
    )
    return float(np.linalg.norm(corners - point[None, :2], axis=1).max())


def max_room_distance_from_point(
    room_dim: np.ndarray,
    point: np.ndarray,
    margin: float,
) -> float:
    lower = np.full(3, float(margin), dtype=np.float64)
    upper = np.maximum(room_dim - float(margin), lower + 0.01)
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(corners - point[None, :], axis=1).max())


def clip_position_to_room(point: ArrayLike, room_dim: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    eps = 1e-4
    return np.minimum(np.maximum(point, eps), room_dim - eps)

def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    distances = []
    for idx in range(a.shape[0]):
        a0 = a[idx]
        a1 = a[(idx + 1) % a.shape[0]]
        distances.extend(distance_point_to_segment(point, a0, a1) for point in b)
    for idx in range(b.shape[0]):
        b0 = b[idx]
        b1 = b[(idx + 1) % b.shape[0]]
        distances.extend(distance_point_to_segment(point, b0, b1) for point in a)
    return float(min(distances)) if distances else 0.0


__all__ = [
    "clip_position_to_room",
    "distance_point_to_polygon",
    "distance_point_to_segment",
    "max_room_distance_from_point",
    "max_room_horizontal_distance_from_point",
    "point_in_polygon",
    "polygon_area",
    "polygon_distance",
    "polygons_overlap",
    "segments_intersect",
]
