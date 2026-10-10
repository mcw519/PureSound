# Acoustic world

[繁體中文](world.zh-TW.md)

`puresound web` → **Acoustic world** (`#/world`) places up to eight talkers and
noise sources in a room — each standing still or moving — renders what one
microphone picks up, runs an enhancement model on it and scores the result
against the reference its task calls for. A limit
map repeats the render over a grid of two parameters. The renderer is described
in [moving-source rendering](../algorithms/audio/dynamic_scene.md).

## Build a scene

Pick a **Starting point** in Settings:

| Preset | What happens |
| --- | --- |
| Walk up and away | The target talker walks to the microphone and back; another talker stands still. |
| Swap near and far | The two talkers cross the room, trading the near and far positions. |
| Turn while talking | The target talker stays put and turns away from the microphone and back. |
| Walk behind a screen | The target talker walks behind a screen that blocks the direct sound. |
| Busy room | The target talker walks past two other talkers who stand still; a fan runs and a humming machine crosses the room. |

### Sources

**+ Talker** and **+ Noise** under **Sources** add a source, up to eight; at the
limit both are disabled and say so. A new source stands 1.5 m from the
microphone at the first free bearing — inside the room, outside obstacles and at
least 0.5 m from every other source's path — facing the microphone, with its own
audio (talkers take the speech samples in turn, noise the noise samples). **×**
removes a source; the last one stays.

Each source has a role:

| Role | Plays | Radiates |
| --- | --- | --- |
| Target talker | a person the model should keep | like a talker (`speech_human`) |
| Other talker | a person the model should remove, or keep in **All speech** | like a talker |
| Noise | a noise source | evenly (`omnidirectional`) |

A scene can hold several of each, or none of a role. Sources of one role share
its colour, each further one a lighter shade, and carry their ID on the plan, in
the 3-D view and in the deck.

**Stays still** keeps a source on one spot (one keyframe); turning it off adds a
keyframe at the end of the scene on the same spot, ready to drag. Noise can
move like any other source.

Every source follows keyframes: a time, a position and the direction it faces
(yaw, and pitch under **Height and pitch**). Edit them on the floor plan — drag a
numbered keyframe or the microphone; arrow keys nudge the focused one by 5 cm
(Shift: 25 cm), Enter selects it — or in the keyframe table under **Sources**.
Double-click the floor to put the selected source there at the timeline's time:
a new keyframe, or the one already at that time moves. **Add keyframe at … s**
under a source adds one on the existing path, so nothing changes until it is
moved. Keyframes that share a spot share one handle ("1,3").

The plan draws the room to scale with a 1 m grid, the microphone, the near region
around it, obstacles, each path and where every source is at the timeline's
time. Under it, the readout gives each talker's distance, its angle off the
microphone and its near-region weight.

Other settings:

- **Length** rescales the keyframe times so each keeps its share of the scene.
- **Seed** draws the room's materials (for the chosen **Room type**) and its
  reverberant tail. **Size** and **Microphone** are in metres.
- **Reference** sets what each reference keeps of a talker's sound (below).
- **Near radius** sets the near-region reference (below).
- Each source has its audio (a sample or an upload), a gain, a start time and
  **Repeat complete clips** for talkers or **Loop the clip** for noise. Speech
  uses complete source utterances, not the speaker-verification half-clips.
  Only whole repetitions that fit after the start time are played; remaining
  time is silent, with propagation and reverberation preserved. If even one
  speech clip cannot fit, extend the scene, start earlier, or choose a shorter
  complete recording. Sample durations are shown in the selector. Noise can
  fill a partial final cycle. Loop seams get a 10 ms taper. `speaker-a-2` is a
  compatibility alias for the same full utterance as `speaker-a`, not a third
  independent voice. User uploads should themselves contain complete speech;
  the renderer does not infer sentence boundaries within recordings.

Problems are shown as you edit, against the same limits the server checks: a
keyframe outside the room, closer than 0.2 m to the microphone or inside an
obstacle, keyframe times out of order, a path faster than 2 m/s. **Render** stays disabled, with the reason on its tooltip, until they are
fixed; the server's own check follows a moment after each edit.

## Render and listen

### Optional ai-coustics comparison

Install the application extra with `uv sync --extra cpu --extra aicoustics`, or
install `aic-sdk==3.3.0` into the server's existing environment. In Settings,
enable **Compare with ai-coustics**, enter the complete SDK key from the
ai-coustics developer platform, select a 16 kHz model and its enhancement
strength. Quail Voice Focus targets a primary speaker; Quail Multi Speaker and
Rook Multi Speaker retain multiple speakers. Use a reference policy appropriate
to the task when interpreting their scores.

The existing world job renders once. Both providers receive the exact same mono
mixture and common scene gain. The native Python SDK processes complete blocks,
flushes with silence and removes its queried audio delay, preserving the original
sample count. The player offers ai-coustics output and removed-component tracks;
all reference policies, transcript comparison and downloads work on these tracks.
The score table shows SI-SDR, improvement, STOI, PESQ, processing RTF and SDK
audio delay. Setup/download time is recorded separately from processing time;
these RTF values are individual runs, not hardware-independent benchmarks.

Sweeps run both providers on each cell. **Map comparison** selects PureSound,
ai-coustics or the SI-SDR difference (ai-coustics minus PureSound). Missing
comparison scores are not treated as silence or as a failed-model threshold.
An optional provider failure preserves the local result and explains the failure.
**Retry** reruns that cell's scene and both processors; re-running a saved
comparison cell requires entering the key again.

Keys stay in tab/job memory. They are omitted from history, saved requests,
scene JSON, ZIPs, reports and application error messages. No credential is read
from an environment variable as an automatic fallback: enabling comparison and
entering a key are explicit. The SDK executes audio locally on the server; model
downloads, authorization and metering use the network. Models are cached under
`~/.cache/puresound/ai-coustics`; `PURESOUND_AIC_CACHE` selects another directory.
Each call has a fresh processor, with cached weights shared between calls.
Reports include requested/resolved model IDs, checksum, SDK version, strength,
block size and delay. This adapter is confined to `puresound.web`; training and
sampling never invoke it.

Official references: [Python SDK](https://docs.ai-coustics.com/reference/sdk/language-bindings/python),
[model catalog](https://docs.ai-coustics.com/reference/sdk/models),
[credentials](https://docs.ai-coustics.com/models/get-started/authenticate-apps),
[delay alignment](https://docs.ai-coustics.com/reference/concepts/latency).

The player opens in **Listen** with larger track cards and waveforms. Use
**Visible tracks** to check the microphone, model output, references or source
stems you want to compare. The browser remembers this combination for the player,
including after a reload. **Inspect audio** exposes spectrograms, zoom and analysis
tools without opening unchecked tracks. Clicking a card switches the audible
track at the same playback position. Unchecking the audible track switches to
another checked, loaded track; at least one track remains checked. Scores and
reference policy are in the disclosure below the player.

After rendering, the 3-D view shows volume-driven ripples in each source's
colour. Their brightness and visible range follow that source's RMS level at
the microphone, on a shared −65 to −15 dBFS display scale. They include distance,
occlusion and reverberation, and are independent of the selected listening
track or playback boost. These are slowed visual cues, not simulated physical
wavefronts. Thin glowing shells replace wire grids; a fast attack and 100 ms
release soften level changes. Playback, pause and seeking share the audio clock; emitted ripples
stay at their earlier positions as a source moves. With reduced motion enabled,
only the source's glow changes. Before rendering, no audio ripples are shown.

**Render scene** (Ctrl+Enter) runs a cancellable job; progress shows under the
page header. Pick **None · room only** as the model to render without one; the
scores then describe the microphone itself.

**Score against** chooses the reference, and switching re-scores the same render:

- **Target talkers** — every source whose role is Target talker.
- **All speech** — every talker, target or not. The default for noise-suppression
  models.
- **Near region** — every talker inside the near radius, faded out across 0.2 m at
  its edge. The default for voice-isolation models.

Choosing a model, a preset or an imported scene, or opening a render from
History, picks the default again; without a model it is Target talkers when the
scene has one, else All speech. When a reference is silent — no target talker,
nobody near — the result reports how quiet the output is instead of SI-SDR.

**Reference** (Settings › Scene) sets what the references keep of each talker,
with the training pipeline's names; changing it needs a new render:

| Reference | Keeps |
| --- | --- |
| Early · direct + 50 ms (default) | the direct sound and the first 50 ms of reflections |
| Direct · direct + 6 ms | the direct sound only |
| Full · with reverberation | the whole sound at the microphone |
| Anechoic · dry | the dry clip at the direct sound's arrival, as if there were no room |

The stats give SI-SDR of the output and of the microphone, their change, STOI and
PESQ. The deck plays the microphone, the model output, what was removed, the
references and each source alone; under it run the SI-SDR change per 1 s window
and each talker's distance, aligned with the audio. Playing moves the sources on
the plan and in the 3-D view. The word check transcribes tracks when you want WER.

**Run on this device** runs the same model in the browser (see
[On this device](web.md#on-this-device)) and adds it as a track, with its difference from the
server's output.

## Limit map

Open the **Limit map** tab, choose a parameter and comma-separated values for the
columns and the rows (at most 25 cells), and **Run sweep**. Each parameter
changes every source of its kind by the same amount, so several sources keep
their differences:

| Parameter | Changes |
| --- | --- |
| Noise level change · dB | Added to the gain of every noise source |
| Other talker level change · dB | Added to the gain of every other talker |
| Target distance change · m | Moves every keyframe of every target talker away from (+) or toward (−) the microphone |

A parameter whose kind the scene lacks (a noise change without noise) blocks the
sweep. Cells fill in as they finish and show the output's SI-SDR against the
chosen reference, with its change from the microphone. Maps made before these
parameters existed still open and show the old parameter names. Shading only compares cells within
one map. Open a cell to listen to it; retry a failed or cancelled cell. A map
needs a model.

## Files and API

**Export scene JSON** keeps the scene and references to its audio (uploads must
still exist). A render's **Scene package** carries the scene, renderer version and
settings, the source audio with SHA-256 hashes, every rendered track and the
report; **Import…** a package to reproduce the render. Scenes and packages saved
before roles existed import with the scored talker as the target and the others
as other talkers.

- `GET /api/world` — presets, room types, sample assets, limits (roles, reference
  types, source count), sweep parameters and the role each changes, and the
  reference each task defaults to.
- `POST /api/world/validate` — `{scene, assets}`; the normalized scene or the error.
- `POST /api/world/materials` — `{scene, room_type}`; the scene with materials
  re-drawn for the room type and the scene's seed.
- `POST /api/jobs` with `kind: "world_render"` or `"world_sweep"`, `scene` and
  optionally `assets`, `model_id`, `provider`; a sweep adds two
  `axes: [{parameter, values}]` and optionally `cell_indices`. Poll with
  `GET /api/jobs/{id}?summary=1` (a running sweep returns each cell's status and
  SI-SDR) and cancel as for any job; finished jobs export a ZIP with every track.

```python
from puresound.audio.rir.scene.world_presets import world_preset
from puresound.audio.rir.render.dynamic import render_dynamic_scene
scene = world_preset("approach", seed=4)
render = render_dynamic_scene(scene, assets)  # mono 16 kHz arrays keyed by asset_id
```

Tests: `test/rir/test_dynamic_scene.py`, `test/rir/test_rir_path_band_effects.py`,
`test/web/test_world.py` and `test/web/world_model.test.cjs`.
