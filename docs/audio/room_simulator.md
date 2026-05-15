# puresound.audio.room_simulator

Physics-based shoebox room impulse response (RIR) simulator using the `rir_generator` library.

## Class: `RoomImpulseResponseSimulator`

A dataclass-based simulator that samples random room geometries and generates corresponding Room Impulse Responses.

### Dataclass Fields

| Field | Type | Description |
|-------|------|-------------|
| `room_dim_range` | `List[[min, max], [min, max], [min, max]]` | Ranges for room x, y, z dimensions (meters) |
| `rt60_range` | `[min, max]` | Reverberation time T60 range (seconds) |
| `source_receiver_distance_range` | `[min, max]` | Generic source–receiver distance range (meters) |
| `foreground_distance_range` | `[min, max]` | Distance range for the foreground/target speaker |
| `interferer_distance_range` | `[min, max]` | Distance range for interference sources |
| `sample_rate` | `int` | Sample rate for generated RIR (default: `16000`) |

### Methods

#### `sample_scene() -> Dict`

Samples a random room configuration.

**Returns dictionary with:**
- `room_dim` – Sampled room dimensions `[x, y, z]`
- `rt60` – Sampled reverberation time
- `source_pos` – Sampled source position `[x, y, z]`
- `receiver_pos` – Sampled receiver/microphone position `[x, y, z]`
- `distance` – Source–receiver distance

---

#### `generate(scene: Optional[Dict] = None) -> Tuple[Tensor, Dict]`

Generates a Room Impulse Response for a given or newly sampled scene.

**Parameters:**
- `scene` – Pre-sampled scene dict from `sample_scene()`. If `None`, a new scene is sampled.

**Returns:**
- `rir` – RIR tensor `[1, T]`
- `meta` – Scene metadata dict including room dimensions, positions, and RT60

## Helper Functions

### `_sample_range(bounds: List[float, float]) -> float`

Samples a uniformly distributed value within `[bounds[0], bounds[1]]`.

---

### `_sample_room_dim(room_dim_range) -> List[float]`

Samples a 3D room dimension `[x, y, z]` from the given ranges.

---

### `_sample_point(room_dim: List[float], margin: float = 0.3) -> List[float]`

Samples a random 3D point inside the room, keeping at least `margin` meters from all walls.

## Example

```python
from puresound.audio.room_simulator import RoomImpulseResponseSimulator

sim = RoomImpulseResponseSimulator(
    room_dim_range=[[3, 8], [3, 6], [2.5, 4]],
    rt60_range=[0.1, 0.8],
    source_receiver_distance_range=[0.5, 3.0],
    foreground_distance_range=[0.3, 1.5],
    interferer_distance_range=[1.0, 4.0],
    sample_rate=16000,
)

scene = sim.sample_scene()
rir, meta = sim.generate(scene)
```
