# puresound.audio.room_simulator

繁體中文版本：[room_simulator.zh-TW.md](room_simulator.zh-TW.md)

`RoomImpulseResponseSimulator` renders shoebox-room RIRs on the fly with the
image-source method (J. B. Allen and D. A. Berkley, *Image method for
efficiently simulating small-room acoustics*, JASA 1979), through E. Habets'
`rir_generator`. It is the on-the-fly alternative to a pre-generated bank
([rir_bank.md](rir_bank.md)); recipes reach it through
`AudioEffectAugmentor.init_room_simulator` and `apply_rir`
([augmentation.md](augmentation.md)).

## Scene first, sources per call

`sample_scene()` fixes a room, a receiver and an RT60, but **no source**.
`generate()` places a new source on every call, at a distance chosen by its
`source_role`. One scene can therefore hold a near foreground talker, far
interferers and a wall-mounted media source, all heard by the same microphone
in the same room: the dataset calls `apply_rir` once per source with the same
scene and a different role.

## Configuration

```python
@dataclass
class RoomImpulseResponseSimulator:
    room_dim_range: list[list[float]]              # [[xmin,xmax],[ymin,ymax],[zmin,zmax]], m
    rt60_range: list[float]                        # [min, max], s
    source_receiver_distance_range: list[float]    # [min, max], m; fallback for any role
    foreground_distance_range: Optional[list[float]] = None
    interferer_distance_range: Optional[list[float]] = None
    media_distance_range: Optional[list[float]] = None
    receiver_margin: float = 0.4                   # min receiver-to-wall distance, m
    source_margin: float = 0.4                     # min source-to-wall distance, m
    media_wall_offset_max: float = 0.2             # media source's max extra standoff from its wall, m
    sound_speed: float = 343.0                     # m/s
    nsample: Optional[int] = None                  # RIR length; None = int(RT60 · fs)
    order: int = -1                                # reflection order; -1 = all
    hp_filter: bool = True                         # Allen–Berkley high-pass (removes the DC build-up)
```

The YAML block is `augmentation_reverb.simulator` with the same keys, plus
`used`, `source_level` and `pregenerated`, which the dataset layer reads and
`init_room_simulator` strips. The sample rate is a `generate()` argument, not a
setting.

## `sample_scene() -> {"room_dim", "receiver", "rt60"}`

Room dimensions uniformly per axis from `room_dim_range`, a receiver uniformly
inside the room at least `receiver_margin` from every wall, and RT60 uniformly
from `rt60_range`. All draws use NumPy's global generator.

## `generate(sample_rate, scene=None, source_role="source", distance_range_override=None) -> (rir, metadata)`

1. `scene` defaults to a fresh `sample_scene()`.
2. **Distance range** for the source:

   | condition | range |
   | --- | --- |
   | `distance_range_override` given | the override, exactly |
   | `source_role="foreground"` | `foreground_distance_range` |
   | `source_role="media"` | `media_distance_range`, else `interferer_distance_range` |
   | `source_role="interferer"` | `interferer_distance_range` |
   | anything else, or the role's range is `None` | `source_receiver_distance_range` |

3. **Placement.** Up to 64 draws of a point at least `source_margin` from the
   walls, accepted when its distance to the receiver lies in the range. A
   `"media"` source is snapped to a random x or y wall at
   `source_margin + U(0, media_wall_offset_max)`, because a TV or loudspeaker
   stands against a wall and its early reflections are stronger than a
   free-standing talker's at the same distance.
4. **Shell fallback.** Thin shells (e.g. `[0.3, 0.5]` m) rarely accept a
   uniform-in-room draw, so the simulator then samples the shell directly:
   uniform radius and direction, up to 256 draws, rejected only by the wall
   margins. If the shell does not intersect the room, the source is placed
   toward the farthest in-room corner at the requested radius clipped to that
   corner's distance, the closest achievable distance. The media wall snap is
   dropped in this path: the distance label outranks the placement.
5. **Render.** `rir_generator.generate(c, fs, r, s, L, reverberation_time=rt60,
   nsample, order, hp_filter)` with an omnidirectional receiver. Given an RT60,
   `rir_generator` inverts Sabine's formula into one frequency-independent
   absorption for all six walls, `α = 24·ln10·V / (c·S·RT60)`, reflection
   coefficient `β = √(1 − α)`; it raises `ValueError` when `α > 1` (an RT60 too
   short for the sampled room).

Returns `rir: Tensor[1, T]` (float32) and

```python
{"room_dim": [x, y, z], "receiver": [x, y, z], "source": [x, y, z],
 "rt60": float, "source_role": str,
 "source_receiver_distance": float,           # realised, m
 "drr_db": float}                             # compute_drr_db, 2.5 ms window
```

## Exact distances: simulator versus bank

`distance_range_override` is honoured exactly here, because the simulator
renders on demand; a pre-generated bank can only pick the stored channel
closest to the requested band. The overrides in the library are:

- `puresound/task/ns.py`, residual playback echo: the device's own loudspeaker
  at `augmentation_speech.echo_playback.distance_range` (default `[0.2, 1.0]` m).
  A true near-field echo channel needs the simulator; with a bank the echo gets
  the nearest stored channel.
- `puresound/task/session_rows.py`: the user at `user_distance_range` and
  bystanders at `bystander_distance_range`.

## Design limits

The model is an empty shoebox with uniform, frequency-independent absorption,
an omnidirectional point source and receiver, and specular reflections only. It
has no furniture, no scattering, no frequency-dependent decay and no source
directivity. Those are what the RIR generation stack adds (see
[hybrid RIR](hybrid_rir.md) and [scene schema](rir_scene_v2.md)).

## Example

```python
from puresound.audio.room_simulator import RoomImpulseResponseSimulator

sim = RoomImpulseResponseSimulator(
    room_dim_range=[[3, 8], [3, 6], [2.5, 4]],
    rt60_range=[0.15, 0.8],
    source_receiver_distance_range=[0.3, 4.0],
    foreground_distance_range=[0.3, 1.2],
    interferer_distance_range=[1.2, 4.0],
    nsample=8192,
)
scene = sim.sample_scene()
fg_rir, fg_meta = sim.generate(16000, scene=scene, source_role="foreground")
it_rir, it_meta = sim.generate(16000, scene=scene, source_role="interferer")
# same room and receiver; fg distance in [0.3, 1.2] m, interferer in [1.2, 4.0] m
```
