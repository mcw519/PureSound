"""Small reproducible scenes for moving-source experiments."""

from dataclasses import replace
import numpy as np
from puresound.audio.rir.scene.dynamic import (
    DynamicSceneSpec,
    DynamicSource,
    MotionKeyframe,
)
from puresound.audio.rir.scene.materials import sample_materialized_shoebox
from puresound.audio.rir.scene.schema import (
    RoomSceneV2,
    EnvironmentConfig,
    Pose,
    TransducerConfig,
    SceneObject,
)

PRESETS = ("approach", "exchange", "turn", "occlusion", "busy")


def world_preset(name="approach", *, duration_s=10.0, seed=0, room_type="office"):
    if name not in PRESETS:
        raise ValueError("unknown world preset")
    dims = [6.0, 5.0, 3.0]
    kind, surfaces, materials, _ = sample_materialized_shoebox(
        dims, np.random.default_rng(seed), room_type=room_type
    )
    d = duration_s
    tracks = {
        "approach": [
            (0, [4.5, 2.5, 1.5], 180),
            (d / 2, [1.7, 2.5, 1.5], 180),
            (d, [4.5, 2.5, 1.5], 180),
        ],
        "exchange": [(0, [1.7, 2.5, 1.5], 180), (d, [4.5, 2.5, 1.5], 180)],
        "turn": [
            (0, [1.8, 2.5, 1.5], 180),
            (d / 2, [1.8, 2.5, 1.5], 0),
            (d, [1.8, 2.5, 1.5], 180),
        ],
        "occlusion": [(0, [4, 1.3, 1.5], 180), (d, [4, 3.7, 1.5], 180)],
    }
    tracks["busy"] = [
        (0, [3.4, 1.3, 1.5], 150),
        (d / 2, [1.8, 2.1, 1.5], 160),
        (d, [3.4, 1.3, 1.5], 150),
    ]
    a = tracks[name]
    b = (
        [(0, [4.6, 3.3, 1.5], 180), (d, [1.7, 3.1, 1.5], 180)]
        if name == "exchange"
        else [(0, [4.8, 3.8, 1.5], 180)]
    )
    defs = [
        ("speaker-a", "speaker-a", a, "target", 0),
        ("speaker-b", "speaker-b", b, "interferer", -6),
        ("noise", "fan", [(0, [3.5, 1, 0.8], 0)], "noise", -12),
    ]
    if name == "busy":
        # Two people talk by the far wall, a fan runs and a humming machine
        # is wheeled across the room behind them.
        defs = [
            defs[0],
            ("talker-c", "speaker-b", [(0, [4.7, 3.9, 1.5], 200)], "interferer", -6),
            ("talker-d", "speaker-a-2", [(0, [3.6, 4.3, 1.5], 260)], "interferer", -9),
            ("noise", "fan", [(0, [3.5, 1.0, 0.8], 0)], "noise", -15),
            ("noise-2", "hum", [(0, [5.4, 0.8, 0.4], 0), (d, [5.4, 4.2, 0.4], 0)], "noise", -18),
        ]
    sources = tuple(
        DynamicSource(
            sid,
            asset,
            tuple(MotionKeyframe(t, tuple(p), yaw) for t, p, yaw in keys),
            role,
            gain,
            repeat=True,
        )
        for sid, asset, keys, role, gain in defs
    )
    transducers = [
        TransducerConfig(
            s.source_id,
            "source",
            Pose(list(s.keyframes[0].position_m), [s.keyframes[0].yaw_deg, 0, 0]),
            "omnidirectional" if s.role == "noise" else "speech_human",
            power_db_spl_at_1m=65,
        )
        for s in sources
    ]
    objects = []
    if name == "occlusion":
        mat = next(iter(materials))
        objects = [
            SceneObject(
                "screen",
                "partition",
                [[2.6, 2.1], [3.0, 2.1], [3.0, 2.9], [2.6, 2.9]],
                0.0,
                2.0,
                mat,
                0.3,
                0.1,
            )
        ]
    room = RoomSceneV2(
        f"world-{name}",
        kind,
        dims,
        surfaces,
        materials,
        EnvironmentConfig(),
        transducers,
        [TransducerConfig("mic", "receiver", Pose([1.0, 2.5, 1.4]))],
        objects,
    )
    return DynamicSceneSpec(room, sources, duration_s=duration_s, seed=seed)


def replace_room_materials(room, room_type, seed):
    """``room`` with surfaces and materials drawn for ``room_type`` from ``seed``.

    Obstacle materials are kept: their IDs are part of the geometry.
    """
    kind, surfaces, materials, _ = sample_materialized_shoebox(
        room.dimensions_m, np.random.default_rng(seed), room_type=room_type
    )
    materials.update(
        {
            k: v
            for k, v in room.materials.items()
            if k in {o.material_id for o in room.objects}
        }
    )
    return replace(room, room_type=kind, surfaces=surfaces, materials=materials)


def replace_world_material(spec, room_type):
    return replace(spec, room=replace_room_materials(spec.room, room_type, spec.seed))
