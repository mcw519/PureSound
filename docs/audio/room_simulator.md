# puresound.audio.room_simulator

繁體中文版本：[`room_simulator.zh-TW.md`](room_simulator.zh-TW.md)

On-the-fly shoebox RIR simulation via the image-source method
(`rir_generator`, Habets' implementation of Allen & Berkley). Wired through
[`augmentation.AudioEffectAugmentor.init_room_simulator`/`.apply_rir`](augmentation.md);
the pre-generated alternative (faster, pre-vetted, no per-item CPU cost) is
[`rir_bank.md`](rir_bank.md)'s `PreGeneratedRoomBank`/`PreGeneratedReleaseBank`.

The central design point: **`sample_scene()` fixes only a room and a
microphone position** — there is no source position yet. `generate()`
samples the source separately on *each call*, with its distance from the
receiver controlled by a `source_role` argument. This is what lets a
training pipeline draw one room and then place a foreground speaker close,
an interferer far away, and a media/TV source against a wall — all inside
the *same* room/receiver — by calling `generate()` (indirectly, through
`apply_rir`) multiple times with the same `scene` dict and a different
`source_role` each time.

## Class: `RoomImpulseResponseSimulator` (dataclass)

```python
@dataclass
class RoomImpulseResponseSimulator:
    room_dim_range: list[list[float]]                        # [[xmin,xmax],[ymin,ymax],[zmin,zmax]] meters
    rt60_range: list[float]                                  # [min, max] seconds
    source_receiver_distance_range: list[float]              # [min, max] meters — role-less fallback
    foreground_distance_range: Optional[list[float]] = None  # used when source_role="foreground"
    interferer_distance_range: Optional[list[float]] = None  # used when source_role="interferer"
    media_distance_range: Optional[list[float]] = None        # source_role="media"; else falls back to interferer_distance_range, then the base range
    receiver_margin: float = 0.4                              # min. mic-to-wall clearance, meters
    source_margin: float = 0.4                                # min. source-to-wall clearance, meters
    media_wall_offset_max: float = 0.2                        # media source's max standoff from the wall it gets snapped to
    sound_speed: float = 343.0
    nsample: Optional[int] = None                              # RIR length in samples; None = rir_generator's own default
    order: int = -1                                            # reflection order; -1 = rir_generator's default (full image method)
    hp_filter: bool = True                                      # rir_generator's built-in high-pass (models a mic capsule's low-end rolloff)
```

There is **no `sample_rate` field** — sample rate is a `generate()` call
argument, not a simulator setting, so one simulator instance (and one
sampled room) can be asked to render at whatever rate the current recipe
needs.

### `sample_scene() -> dict`

```python
{"room_dim": np.ndarray[3], "receiver": np.ndarray[3], "rt60": float}
```

Samples room x/y/z uniformly from `room_dim_range`, a receiver point
uniformly inside the room keeping `receiver_margin` from every wall, and
RT60 uniformly from `rt60_range`. **No source position** — that's
`generate()`'s job, per call, below.

### `generate(sample_rate: int, scene: Optional[dict] = None, source_role: str = "source", distance_range_override: Optional[list[float]] = None) -> Tuple[Tensor, dict]`

1. `scene` defaults to a fresh `sample_scene()` if not given.
2. Samples a source position via the private `_sample_source`, whose
   distance range is resolved by `_distance_range_for_role`:

   | `source_role` | distance range used |
   |---|---|
   | any value, with `distance_range_override` given | `distance_range_override`, exactly |
   | `"foreground"` | `foreground_distance_range` (falls through to the row below if `None`) |
   | `"media"` | `media_distance_range`, else `interferer_distance_range`, else the row below |
   | `"interferer"` | `interferer_distance_range` (falls through if `None`) |
   | anything else (including the default, `"source"`) | `source_receiver_distance_range` |

   `"media"` sources are additionally **wall-snapped**: the sampled point's
   x or y coordinate is clamped to within `[source_margin, source_margin +
   media_wall_offset_max]` of a randomly chosen room wall — modeling that a
   TV/loudspeaker sits against a wall rather than free-standing, which
   strengthens its early reflections relative to a talker at the same
   distance (per the source comment).
3. Rejection-samples up to 64 draws of a point inside the room at the
   target distance from the receiver. If that fails — typical for a *thin*
   distance shell, e.g. `[0.3, 0.5]`, where uniform-in-room rejection rarely
   lands in range — falls back to `_sample_source_in_shell`: sample
   directly on the distance shell (uniform radius × uniform direction,
   up to 256 draws) and reject only on room-margin containment. If even
   *that* shell has no intersection with the room at all, walk toward the
   farthest in-room corner and clamp the radius to the achievable span —
   trading the unreachable exact distance for the closest one the room's
   geometry actually allows, rather than silently ignoring the requested
   range and mislabeling the sample. This shell fallback also drops the
   media wall-snap: distance correctness outranks wall placement when they
   conflict.
4. Calls `rir_generator.generate(c=sound_speed, fs=sample_rate, r=receiver,
   s=source, L=room_dim, reverberation_time=rt60, nsample=nsample,
   order=order, hp_filter=hp_filter)` and transposes the result to
   `[channels, T]` — 1 channel here, since `r`/`s` are single points, not
   microphone/source arrays.

Returns `(rir: Tensor[1, T], metadata: dict)`:

```python
{
    "room_dim": [x, y, z], "receiver": [x, y, z], "source": [x, y, z],
    "rt60": float, "source_role": str,
    "source_receiver_distance": float,
    "drr_db": float,   # via impulse_response.compute_drr_db
}
```

### `distance_range_override`: exact here, only approximate in a pre-generated bank

Passing `distance_range_override` makes the simulator **rejection-sample
exactly** within that window, regardless of the configured per-role range.
Real uses in this repo:

- `puresound/task/ns.py`'s residual playback-echo source: a very close
  `[0.2, 1.0] m` "channel" simulating loudspeaker-to-mic leakage that an
  upstream AEC would leave behind, independent of whatever
  `interferer_distance_range` the recipe otherwise uses for ordinary
  interferers (comment there: *"a true near-field echo channel needs the
  on-the-fly simulator"*).
- `egs/voice_isolate/scripts/eval_domain_gap.py` forcing a tight probe
  distance (`[distance_m - tol, distance_m + tol]`) to evaluate behavior at
  one specific, controlled range.

**A pre-generated bank cannot honor this exactly** — it can only pick
whichever pre-rendered channel in its pool is *closest* to the requested
band, since it isn't rendering RIRs on demand (see
[rir_bank.md](rir_bank.md)). A true near-field override needs this
on-the-fly simulator, not the bank.

## Module-level helpers

- **`_sample_range(bounds: list[float]) -> float`** —
  `Uniform(bounds[0], bounds[1])`.
- **`_sample_room_dim(room_dim_range: list[list[float]]) -> np.ndarray[3]`**
  — one `_sample_range` draw per axis.
- **`_sample_point(room_dim: np.ndarray, margin: float) -> np.ndarray[3]`**
  — a uniform point at least `margin` from every wall (guards
  smaller-than-`2*margin` rooms by flooring the sampling range to at least
  1 cm wide). `margin` has no default — every call site passes
  `receiver_margin` or `source_margin` explicitly.

These, plus `_distance_range_for_role`/`_sample_source`/
`_sample_source_in_shell` described above, are private (leading underscore)
— documented here because they *are* `generate()`'s algorithm, not a
separate public surface.

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
fg_rir, fg_meta = sim.generate(sample_rate=16000, scene=scene, source_role="foreground")
it_rir, it_meta = sim.generate(sample_rate=16000, scene=scene, source_role="interferer")
# fg_meta["room_dim"] == it_meta["room_dim"]  (same room, same scene)
# fg_meta["source_receiver_distance"] <= 1.2, it_meta["source_receiver_distance"] >= 1.2
```

In practice this is reached indirectly through
`AudioEffectAugmentor.init_room_simulator` + `.sample_room_scene()` +
`.apply_rir(..., room_scene=scene, source_role=...)` (see
[augmentation.md](augmentation.md)) rather than instantiated directly — the
example above is the mechanics `apply_rir` wraps.
