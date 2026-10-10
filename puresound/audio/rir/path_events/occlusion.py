"""Continuous, frequency-dependent obstacle insertion loss for path legs.

Hard visibility switches a path off the moment its segment touches an
object, which removes the direct sound completely behind a small screen and
makes the switch audible for a moving source.  Here each object is treated
as an opaque rectangular screen seen from the path leg, and the field behind
it follows the Fresnel-Kirchhoff result for a rectangle via Babinet's
principle (Born & Wolf, *Principles of Optics*, 7th ed., §8.7-8.9;
Pierce, *Acoustics*, ch. 9):

    U / U0 = 1 - (1 - t) * [F(a2) - F(a1)] * [F(b2) - F(b1)] / (2j)

where ``F(v) = C(v) + j S(v)`` is the Fresnel integral and ``[a1, a2] x [b1,
b2]`` the screen's extent in Fresnel units ``v = h * sqrt(2 (d1 + d2) /
(lambda d1 d2))`` for an edge at transverse offset ``h``, ``d1`` / ``d2``
along the leg from either end.  The gain is 0.5 (-6 dB) on the shadow
boundary, falls with frequency and depth into the shadow, and is continuous
across the boundary, so a source walking behind a screen fades rather than
switching.  ``t`` is the object's amplitude transmission,
``sqrt(SceneObject.transmission)``.  The gain at each third-octave centre is
the energy of ``U / U0`` averaged over the octave around it (nine log-spaced
frequencies): the coherent sum over the screen's three edges has sharp
interference nulls that would otherwise become notches sliding through the
spectrum as the source moves, which real edges and speech bandwidth smear.

Approximations: the screen is the bounding box of the object's corners in
Fresnel units (a thin screen; thick objects attenuate somewhat more than
this), an object standing on the floor or reaching the ceiling cannot be
passed below or above, several objects multiply, and reflections from the
object's faces are not modelled.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import fresnel

from puresound.audio.rir.path_events.geometry import _path_event_vertices
from puresound.audio.rir.path_events.schema import PathBandGain, PathEventSet
from puresound.audio.rir.scene.schema import RoomSceneV2, SceneObject

FRESNEL_KIRCHHOFF_OCCLUSION_POLICY = "puresound.fresnel_kirchhoff_screen.v1"
#: Third-octave centres; the insertion loss is smooth on this grid.
OCCLUSION_BANDS_HZ = tuple(
    float(f)
    for f in (125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
              2000, 2500, 3150, 4000, 5000, 6300, 8000)
)
_POINTS_PER_BAND = 9
_NEGLIGIBLE_DB = 0.05
_SURFACE_TOLERANCE_M = 1e-3


def screen_insertion_gain(
    start: np.ndarray,
    stop: np.ndarray,
    scene_object: SceneObject,
    *,
    room_height_m: float,
    sound_speed_m_s: float,
    frequencies_hz: tuple[float, ...] = OCCLUSION_BANDS_HZ,
) -> np.ndarray:
    """Complex field ratio ``U / U0`` for one leg past one object."""
    return _screen_field(
        np.asarray(start, dtype=np.float64)[None],
        np.asarray(stop, dtype=np.float64)[None],
        scene_object,
        room_height_m,
        sound_speed_m_s,
        np.asarray(frequencies_hz, dtype=np.float64),
    )[0]


def apply_fresnel_kirchhoff_occlusion(
    event_set: PathEventSet,
    scene: RoomSceneV2,
    *,
    frequencies_hz: tuple[float, ...] = OCCLUSION_BANDS_HZ,
) -> PathEventSet:
    """Replace hard object visibility with a band insertion loss per path.

    Every leg of every path is tested against every object; paths stay
    visible and carry the product of their legs' band gains as a
    :class:`PathBandGain`.  Gains within 0.05 dB of unity are not attached.
    """
    objects = list(scene.objects)
    if not objects:
        return event_set
    centres = np.asarray(frequencies_hz, dtype=np.float64)
    spread = 2.0 ** np.linspace(-0.5, 0.5, _POINTS_PER_BAND)
    fine = (centres[:, None] * spread[None, :]).ravel()
    starts, stops, first_leg = [], [], []
    for event in event_set.events:
        vertices = _path_event_vertices(event)
        first_leg.append(len(starts))
        starts.extend(vertices[:-1])
        stops.extend(vertices[1:])
    starts, stops = np.asarray(starts), np.asarray(stops)
    power = np.ones((len(starts), fine.size))
    for scene_object in objects:
        power *= np.abs(
            _screen_field(
                starts,
                stops,
                scene_object,
                float(scene.dimensions_m[2]),
                float(scene.environment.sound_speed_m_s),
                fine,
            )
        ) ** 2
    per_event = np.multiply.reduceat(power, first_leg, axis=0)
    gains = np.sqrt(per_event.reshape(len(first_leg), centres.size, -1).mean(axis=2))
    loud = np.max(np.abs(20.0 * np.log10(np.maximum(gains, 1e-12))), axis=1)
    events = []
    for event, gain, level in zip(event_set.events, gains, loud):
        band = event.band_gain
        if level > _NEGLIGIBLE_DB:
            band = PathBandGain(
                frequencies_hz=list(frequencies_hz),
                magnitude=gain.tolist(),
                provenance="fresnel_kirchhoff_screen",
            ).times(event.band_gain)
        events.append(replace(event, visible=True, band_gain=band))
    metadata = {
        **event_set.metadata,
        "object_occlusion_policy": FRESNEL_KIRCHHOFF_OCCLUSION_POLICY,
        "object_count": len(objects),
    }
    return replace(event_set, events=events, metadata=metadata)


def _screen_field(starts, stops, scene_object, room_height_m, sound_speed, frequencies):
    """``U / U0`` of ``(legs, frequencies)`` for legs ``starts -> stops``."""
    legs = stops - starts
    length = np.linalg.norm(legs, axis=1)
    field = np.ones((len(legs), frequencies.size), dtype=np.complex128)
    axis = legs / np.maximum(length, 1e-12)[:, None]
    footprint = np.asarray(scene_object.footprint, dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x, y in footprint for z in (scene_object.z_min, scene_object.z_max)]
    )
    depth = np.einsum("lcd,ld->lc", corners[None] - starts[:, None], axis)
    near = (length > 1e-9) & ~(
        np.all(depth <= 0.0, axis=1) | np.all(depth >= length[:, None], axis=1)
    )
    if not np.any(near):
        return field
    axis, length, start = axis[near], length[near, None], starts[near]
    depth = np.clip(depth[near], 1e-3 * length, (1.0 - 1e-3) * length)
    vertical = np.array([0.0, 0.0, 1.0])
    upright = np.abs(axis @ vertical) < 0.99
    reference = np.where(upright[:, None], vertical, np.array([1.0, 0.0, 0.0]))
    across = np.cross(axis, reference)
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    up = np.cross(across, axis)
    up *= np.where(up @ vertical < 0.0, -1.0, 1.0)[:, None]
    transverse = corners[None] - start[:, None] - depth[..., None] * axis[:, None]
    offsets = np.stack(
        [np.einsum("lcd,ld->lc", transverse, across),
         np.einsum("lcd,ld->lc", transverse, up)],
        axis=-1,
    )
    wavelength = sound_speed / frequencies
    scale = np.sqrt(
        2.0 / wavelength[None, :, None] * (1.0 / depth + 1.0 / (length - depth))[:, None, :]
    )
    nu = offsets[:, None] * scale[..., None]
    low, high = nu.min(axis=2), nu.max(axis=2)
    if scene_object.z_min <= _SURFACE_TOLERANCE_M:
        low[upright, :, 1] = -np.inf
    if scene_object.z_max >= room_height_m - _SURFACE_TOLERANCE_M:
        high[upright, :, 1] = np.inf
    span = _fresnel_integral(high) - _fresnel_integral(low)
    aperture = span[..., 0] * span[..., 1] / 2j
    transmission = np.sqrt(float(scene_object.transmission))
    field[near] = 1.0 - (1.0 - transmission) * aperture
    return field


def _fresnel_integral(nu: np.ndarray) -> np.ndarray:
    """``C(nu) + j S(nu)``, with the limits +-(0.5 + 0.5j) at +-infinity."""
    sine, cosine = fresnel(np.where(np.isfinite(nu), nu, 0.0))
    value = cosine + 1j * sine
    return np.where(np.isfinite(nu), value, np.sign(nu) * (0.5 + 0.5j))


__all__ = [
    "FRESNEL_KIRCHHOFF_OCCLUSION_POLICY",
    "OCCLUSION_BANDS_HZ",
    "apply_fresnel_kirchhoff_occlusion",
    "screen_insertion_gain",
]
