# Web UI and inference API

Traditional Chinese: [web.zh-TW.md](web.zh-TW.md)

PureSound includes a local browser UI backed by the same Model Zoo runtime as
the command-line interface.

## Start the server

```bash
puresound web
# http://127.0.0.1:7860
```

To change the bind address:

```bash
puresound web --host 0.0.0.0 --port 8080
```

The server has no authentication. Bind to `0.0.0.0` only on a trusted network
or behind an authenticated reverse proxy.

A web page you open elsewhere can still send requests to a server on your own
machine. The server therefore refuses a write or a live session that comes from
another origin, and a server bound to a loopback address answers only to
`localhost`, IP addresses and its bind address, so a page cannot reach it under
a name of its own (DNS rebinding). To reach it by another name -- a reverse
proxy's public name -- pass `--allow-host NAME` (repeatable) or list that name in
`PURESOUND_ALLOWED_HOSTS`, comma separated (`create_server(..., allowed_hosts=[...])`
from Python).

Browsers give a page the microphone only over HTTPS or on `localhost`. To
record or go live from another machine, serve HTTPS:

```bash
puresound web --host 0.0.0.0 --https          # self-signed, made once with openssl
puresound web --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem
```

The self-signed certificate is kept beside the run history (`tls/`); the
browser warns once, then works. Live mode then uses `wss://`.

## The workbench

Every screen has the same frame. The **sidebar** groups the screens -- *Run*
(Playground, Verify, Compare), *Data* (Annotate, Pipeline), *Library* (Models,
History) -- and holds the runtime status, the language switch (EN / 中), the
theme (Auto follows the system, Light, Dark) and the version. A screen's
**page header** carries its title, one line on what it is for, its secondary
actions, *Help* (what the screen does, its steps and keys) and *Settings*, and
the one primary action at the right (*Run*, *Verify*, *Run comparison*,
*Download export*, *Build and trace*). While work runs, a state line under the
header shows what it is doing and how far along; it says *Done* or why it
failed when it ends.

Every setting of a screen lives in its **Inspector**, a column on the right.
*Settings* hides or shows it, and each screen remembers that. Below 1100 px the
Inspector opens as a drawer over the page (Esc or a click outside closes it);
below 760 px the sidebar becomes a top bar with a menu. The whole interface is
in English and Traditional Chinese: numbers, units, model and file names and
metric names stay as they are.

## Screens

| Screen | Address | What it does |
| --- | --- | --- |
| **Playground** | `#/playground` | A file, a sample, a recording from the microphone or a live session through a voice-isolation or noise-suppression model, on the server or in the browser (*Runs on*, below) -- optionally after another one (*Run first*, e.g. noise suppression before voice isolation); listen, compare and download. |
| **Verify** | `#/verify` | Whether an enrollment and a test recording are the same speaker (cosine similarity against a threshold). |
| **Compare** | `#/compare` | Several enhancement models on the same recordings, next to the unprocessed input, with optional reference metrics and DNSMOS. A row is a model and its own settings; *+* adds another setting of the same model, so a dry-blend or onset-guard sweep is one comparison. |
| **Annotate** | `#/annotate` | Mark keep / suppress spans on a recording and export them for evaluation (below). |
| **Acoustic world** | `#/world` | Talkers moving through a room, rendered as one microphone, run through a model and scored; a limit map sweeps two parameters ([world.md](world.md)). |
| **Pipeline** | `#/pipeline` | One training row built from your own audio (or the bundled samples) with a released recipe's augmentation knobs, followed stage by stage (below). |
| **Models** | `#/models` | Model metadata, the release notes and gate records the catalog lists for each model (*Details*), the gate records of several models side by side (*Compare gate records*), and the catalog integrity check. |
| **History** | `#/history` | Every playground run, speaker check, comparison and live session; *Open* puts one back on its screen. |

The address bar follows the screen, so a reload or a shared link lands on the
same one; the page opens on the last screen used, Playground on a first visit.

Playback gain and the browser limiter do not change model inputs or downloaded
files.

### Word check

Under each comparison, *Which words survived?* transcribes the tracks and
aligns them word by word: struck through = lost, underlined = added, marked =
changed. Clicking a word selects that track and loops a moment around it.

- **Recognisers:** Whisper on this server (faster-whisper; install the `asr`
  extra, `uv sync --extra asr`; `large-v3` is the default because a weak
  recogniser hides over-suppression), Azure Speech (key + region) or
  ElevenLabs Scribe (API key, model `scribe_v2` by default).
- **Keys** are typed into the page and kept in that tab's memory only. They go
  with a transcription request and nowhere else: the server does not log them,
  write them to the run history or return them, and scrubs them from error
  messages; the browser does not store them. Closing the tab forgets them.
- **With a reference transcript** (what the near talker actually said) every
  track gets an error rate against it. **Without one**, tracks are compared
  with one track's transcript (the input by default). That is a difference,
  not an error rate: a voice-isolation model is meant to remove a distant
  talker's words, and those show up as missing too.
- Chinese, Japanese and Korean are scored per character (CER); other text per
  word, normalised as `puresound.evaluation.tools.wer` normalises it.
- A track whose transcript is over 1.5× the reference is flagged as a
  recogniser loop.

Word checks are not part of the run history.

### Microphone and live mode

*Record* captures the microphone at 16 kHz with the browser's echo
cancellation, noise suppression and gain control off, so the model hears the
capture chain as it is; the recording then runs like a file. *Live* streams
the microphone through the selected model 20 ms at a time. *Listen to* only
changes what you hear while it runs -- *Nothing*, *Model output* (use
headphones), or *Raw microphone*, the unprocessed input, to switch against the
model while talking; the session is streamed and kept either way. The page shows the model's compute per
chunk, the round trip, the algorithmic latency (analysis window plus
look-ahead) and an estimate of microphone-to-ear delay. Stopping puts the
session on the comparison deck. The microphone needs a secure page: open the
UI on `localhost` or behind HTTPS.

### On this device

*Runs on* in Playground's Settings chooses where the model runs: *Server*, or
*This device* -- the browser, with ONNX Runtime Web (WebAssembly) in a worker and
the [Web SDK](../../sdk/web/README.md)'s streaming runtime, so the audio is not
uploaded. File, Record and Live all run there and open in the same comparison
deck; running a file on the server and then on this device keeps the server's
output as *Previous*, to hear the two side by side.

- Only models with a device build run there. Build them once with
  `npm ci && npm run build && npm run assets` in `sdk/web`; the other models
  stay in the list, marked *server only*.
- *Run first*, the artifact variant and the provider are server settings. A
  device run has no DNSMOS and is not kept in History; dry blend and the onset
  guard apply as on the server.
- `puresound web` makes every page cross-origin isolated (COOP `same-origin`,
  COEP `require-corp`), so WebAssembly can use up to four threads. Browsers
  isolate only secure pages: over plain HTTP to another machine it runs on one
  thread (serve `--https` for more).
- Live first measures the device: if one second of audio takes more than 0.7 s
  to process, it suggests recording instead. A session that falls a second
  behind stops and opens what it has.

Other screens run models on the device through the same module,
`window.PureSoundDevice`; its calls are in the
[Web SDK README](../../sdk/web/README.md#in-the-web-app-windowpuresounddevice).

The playground ships three enhancement samples and two speaker pairs
(LibriSpeech speech, synthetic rooms and noise; `tools/build_web_samples.py`).

*ZIP* and *Report* (on a result, a comparison and every History row) take a
run out of the workspace: every audio output with `report.json`, or one HTML
page with the settings, the scores and players with the audio embedded, for
someone without a PureSound server. A report stops embedding audio past
40 MB and says so.

*Clear* empties a screen back to its first state: the playground's input and
result, both speaker-verification recordings, or a comparison report (the
recordings and rows chosen in its Settings stay). It is disabled while a run or
a live session is going.

### Pipeline inspector

The Pipeline screen answers "what does the training data pipeline do to a row,
and what does that ask of the model?" for one row at a time.

- **Build a row.** Pick a released training recipe (`egs/*/config/*.yaml`
  under `--pipeline-root`), a seed, the rows' role (training or validation) and,
  for a recipe with a curriculum, the epoch. The audio is yours: a foreground
  clip, other talkers, noise clips and a room -- a bundled sample room with
  geometry, uploaded impulse responses (applied to the whole mixture) or none.
  Two LibriSpeech readers, six synthesised noises and four rooms from
  PureSound's own synthetic RIR bank are bundled (`tools/build_pipeline_samples.py`);
  the training corpora are never read. Blocks that need a corpus the inspector does not have (real-recording
  pools, conversation rows) are switched off and listed under *What the
  inspector changed in this recipe*.
- **Stages.** Every stage of synthesis in order, grouped by what it acts on.
  Solid stages acted on this row; dashed ones did not fire (with their
  probability), dotted ones are not in the recipe or were switched off.
- **Effective SNR.** Target energy against everything else in the mixture after
  each stage -- noise, talkers that are not target, late reverberation past an
  `early` target, transmission damage: what is left for the model to remove.
- **Model.** With a model chosen, every stage's pair is scored as if training
  stopped there: SI-SDR, STOI and PESQ of the mixture and of the model's output
  against that stage's target, or the output's level change on a row with no
  target. Pairs above full scale (before the A/D stage) are gain-staged first,
  as the converter would.
- **Stage detail.** What the stage does and what it means for the model (in the
  page's language), what it drew on this row, the pair before and after, the model's
  scores, and a deck with the mixture before and after the stage, the target,
  the other talkers, the model's output and the stage's own change.
- **Room.** The sample room in 3-D (three.js, vendored; a floor plan without
  WebGL): walls, obstacles, the microphone and every source, the ones the row
  used in their role's colour; and each impulse response's envelope, energy
  decay and target window.

One row takes a few seconds to build, plus a few seconds per stage when a model
scores it; rows are built one at a time.

### Annotate

Annotate marks labelled time ranges on a recording, for evaluation windows and
listening notes. It runs in the browser: audio is decoded there and never
uploaded.

1. Open one or more audio files or a folder (*Open audio*, the Inspector's
   *Files*, or drop them on the stage). Span files next to the audio --
   `<name>.spans.json`, `.spans.csv`, `windows.json`, Audacity labels -- are
   read in; *Import spans* reads one explicitly.
2. Drag over the waveform or spectrogram to select a range, choose a tag
   (1-9, or the Inspector's *Tag for new spans*) and press Enter.
3. Adjust spans in the lane under the waveform (drag an edge, or the body),
   retag or label them in the table.
4. *Download export*, or *Save into folder* to write `<name>.spans.json` next
   to each file (a folder opened read-write, in Chromium browsers).

| Format | Contents |
| --- | --- |
| JSON | the file's metadata and spans; a folder's files together |
| CSV | start, end, duration, tag and label (and the file, for several) |
| `windows.json` | per-file `keep` and `suppress` intervals for evaluation |
| Audacity labels | start, end and label, tab-separated, for the file on screen |

In `windows.json`, tags containing `keep`, `near` or `double` become `keep`
intervals and tags containing `sup` or `far` become `suppress`; other tags are
left out.

Every comparison deck (Playground, Compare, Pipeline) has *Annotate*, which
opens its audible track here as a new file. Annotate's keys (Space, Enter, 1-9,
S / E, Backspace, [ / ], + / − / 0, Ctrl/⌘ + O / I / S / E; *Help* lists them)
act only while the screen is on and you are not typing in a field; the
player's keys stand aside there.

### Listening and comparing

An enhancement result opens in a comparison deck: the input as the model heard
it, the output, and what was removed (input minus output), on one time axis
and one transport. Every track plays at once and only the selected one is
audible, so switching is gapless and sample-aligned. Running the same file
again keeps the last output as a fourth track, so a settings change can be
heard against it. *Match level* plays each track at the input's loudness, so
the louder one does not simply sound better.

| Key or gesture | Action |
| --- | --- |
| Space | Play / pause |
| 1-9 | Switch the audible track |
| Drag | Select a region; it loops |
| Drag a region edge | Adjust the region |
| Shift + click | Extend the region to the click |
| L | Loop on / off |
| Z | Zoom to the region / back |
| + / − / 0 | Zoom in / out in time / fit time and frequency |
| Ctrl/⌘ + wheel | Zoom time around the pointer |
| Shift + wheel | Scroll in time (or drag the bar under the ruler) |
| Alt + wheel | Zoom frequency around the pointer (or the + / − and bar at the right) |
| Esc | Clear the region and zoom |
| Left / Right | Seek 1 s (Shift: 5 s) |
| Ctrl+Enter | Run the current task |

*Export* downloads the selection of the audible track, cut from the file at
its own rate; *Annotate* opens the whole track on the Annotate screen. *View ⚙* sets the spectrogram (FFT size, frequency range up to
Nyquist, linear or logarithmic axis, floor and dynamic range, colours) and the
waveform (linear or dB amplitude, gain); the settings apply to every player on
the page and are remembered per browser. File previews use the same player
with one track.

*Δ vs input* adds two lanes that compare the selected track with the input:
its level per 20 ms frame (total energy -- speech, noise and reverb together,
so a drop is not by itself suppression of the right thing) and its spectrum
bin by bin (cool = removed, warm = added). *dB* draws waveforms on a 0 to
-60 dBFS axis so a quiet residual is visible. Hovering reads out time,
frequency and level. A model with auxiliary heads can return them with
`collect_extras`; they appear as curve lanes.


## Providers

`auto` tries CUDA, CoreML, then CPU. Explicit values are `cpu`, `cuda`,
`coreml`, and `mps`; `mps` is an alias for CoreML. Responses report the
provider that actually ran the model.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Runtime and provider status |
| GET | `/api/models` | List catalog models |
| GET | `/api/models/{model_id}` | Read one model contract |
| GET | `/api/models/{model_id}/benchmarks` | The catalog's benchmark references for a model, parsed |
| GET | `/api/validate` | Validate artifacts and ONNX graphs |
| POST | `/api/infer` | Run synchronous inference |
| POST | `/api/measure` | Compare models and metrics |
| GET | `/api/jobs` | Recent jobs (`?limit=`, default 20) |
| POST | `/api/jobs` | Start an asynchronous job |
| GET | `/api/jobs/{job_id}` | Read job status and result |
| POST | `/api/jobs/{job_id}/cancel` | Request cancellation |
| GET | `/api/runs/{run_id}/{output}` | Download generated audio |
| POST | `/api/uploads` | Store a file's raw bytes (`X-Filename` header); returns `upload_id` |
| GET | `/api/uploads/{upload_id}` | Check that an upload is still stored |
| GET (WebSocket) | `/api/live` | Stream audio through a model frame by frame |
| GET | `/api/asr` | Recognisers available for word checks, cached Whisper models |
| GET | `/api/pipeline` | Pipeline inspector: the training recipes it offers, its bundled samples, whether it is available |
| GET | `/api/jobs/{job_id}/outputs.zip` | Every audio output of a finished run, plus `report.json` |
| GET | `/api/jobs/{job_id}/report.html` | The run as one self-contained HTML page, audio embedded (`?tz=` for times) |

Asynchronous job states are `queued`, `running`, `succeeded`, `failed`,
and `cancelled`.

## Audio input

The browser sends each file once with `POST /api/uploads` (the raw bytes as
the body, the name in `X-Filename`) and refers to it afterwards:

```json
{"upload_id": "3f2a...", "filename": "speech.wav"}
```

so re-running or comparing more models costs no second transfer. Uploads are
bounded (64 files, 1 GiB) and live until the server stops; a request naming
an evicted one gets an error asking for the file again. One decoded audio input
may be at most `--max-upload-mb` MiB (default 64), and its samples, counted as
16-bit, at most 16 times that, so a compressed file cannot expand without bound.
A data URL also works:

```json
{
  "filename": "speech.wav",
  "data": "data:audio/wav;base64,..."
}
```

Local paths are disabled by default. For a trusted local client, start the
server with `--allow-local-paths` and send:

```json
{"path": "/absolute/path/input.wav"}
```

Generated files and the run history are bounded (32 runs). `puresound web`
keeps them on disk under `$XDG_CACHE_HOME/puresound/web` (else
`~/.cache/puresound/web`) so they survive a restart; `--history-dir` moves
them and `--no-history` keeps them in memory, gone when the server stops.
Only finished runs are kept.

## Enhancement outputs

An enhancement request (voice isolation or noise suppression) returns several
`output_urls`:

| Output | Content |
| --- | --- |
| `audio` | The stream as the runtime emitted it: streaming latency at the front, then the whole input and nothing past it. The same samples the CLI writes. |
| `aligned` | `audio` with the latency removed and cut to the input length. The browser plays and exports this one. |
| `input` | The input as the model heard it: mono, at the model sample rate. |
| `removed` | `input - aligned`: what the model took out. |

`alignment` reports the latency that was removed.

## Measurements

Set `"measurements": {"include_dnsmos": true}` to request DNSMOS. If the
optional scorer is unavailable, other metrics are still returned.

With a clean reference, measurements can include SI-SDR, SNR, correlation,
STOI, and PESQ. Without one, level and spectral measurements remain available.

A comparison scores the unprocessed input with the same scorers and returns
it as `baseline`, with its audio; every score in the page is shown next to
it. Rows are `candidates`, each a model and its own parameters over the shared
`parameters`:

```json
{
  "candidates": [
    {"model_id": "voice-isolate-dpcrn-curriculum-v1", "label": "release"},
    {"model_id": "voice-isolate-dpcrn-curriculum-v1", "parameters": {"dry_blend": 1.0}}
  ]
}
```

`models` (one row per model, shared parameters) still works. A candidate (and
a plain `/api/infer` request) may add `"stages": [{"model_id": ...}]`: models
that run first, each on the previous one's output cut back onto the input's
time axis. The reply's `pipeline` lists them with their latency; RTF and
latency then cover the whole chain.

Several recordings go in `inputs.clips` (`[{"audio": ..., "reference": ...}]`,
up to 50). The reply has every clip's report under `clips` and, per
candidate, an `aggregate`: the change from each clip's unprocessed input,
averaged, with a 95% bootstrap interval, the number of clips that improved,
and `resolved` -- true only when the interval excludes zero and there are at
least five clips.

RTF covers frame processing only; building the stream and the first ORT session
is reported separately as `metadata.setup_seconds`. A freshly loaded
enhancement model also runs an untimed 0.5 s warm-up. One RTF is still one
draw -- identical runs spread noticeably on a shared host -- so a comparison can
set `"measurements": {"timing_repeats": 3}` (1-5) to report the median, with
every run in `rtf_runs`. Responses include `host` (1-minute load average and
CPU count) so a number can be read against what else the machine was doing.

## Word-check jobs

`POST /api/jobs` with `"kind": "transcription"`:

```json
{
  "kind": "transcription",
  "backend": "whisper",
  "model": "large-v3",
  "language": "zh-TW",
  "credentials": {},
  "reference_text": "",
  "reference_track": "input",
  "tracks": [
    {"id": "input", "label": "Input", "source": {"url": "/api/runs/<run>/input"}},
    {"id": "output", "label": "Output", "source": {"upload_id": "<id>"}}
  ]
}
```

`backend` is `whisper`, `azure` (`credentials: {"key", "region"}`) or
`elevenlabs` (`credentials: {"key"}`). `language` is BCP-47 or empty for
auto-detection (Azure then uses `en-US`). The result has, per track, the text,
the words with times, the scoring tokens and an `alignment` of
`{"op": "hit" | "sub" | "del" | "ins", "ref", "hyp"}` against the reference,
with `counts` (`error_rate`, `del_rate`, ...). These jobs are kept in memory
only and are not listed in `/api/jobs`.

## Pipeline-trace jobs

`POST /api/jobs` with `"kind": "pipeline_trace"` builds one training row with a
recipe from `GET /api/pipeline` and traces it stage by stage:

```json
{
  "kind": "pipeline_trace",
  "recipe": "egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml",
  "role": "train",
  "seed": 1234,
  "epoch": null,
  "seconds": 6.0,
  "foreground": {"upload_id": "3f2a..."},
  "talkers": [{"sample": "reader-1272"}],
  "noises": [{"sample": "babble"}, {"upload_id": "9c1d..."}],
  "rir": {"kind": "samples", "rooms": ["room_000020_000000"]},
  "model_id": "noise-suppression-dpcrn-mamba-v2"
}
```

Every source is an upload or a bundled sample; the training corpora, RIR banks
and recording pools are never read. The recipe supplies its augmentation knobs
only: its corpus paths are pointed at a throwaway workspace holding the chosen
audio, and blocks that need a corpus the workspace cannot stand in for
(real-recording pools, conversation rows) are switched off and listed in the
report's `notes`. `rir.kind` is `samples` (bundled rooms with geometry),
`upload` (impulse responses applied to the whole mixture) or `none`.

The finished job's `result` is the trace report (`schema:
puresound.pipeline-trace/1`): every stage in synthesis order with what it drew,
the pair's levels and effective SNR after it, and -- with `model_id` -- the
model's scores on that pair, plus the impulse responses, the room geometry and
an `audio_urls` map. The audio is 32-bit float WAV, because stages before the
converter may exceed full scale. Rows are built one at a time. The recipes
offered are the training recipes in `egs/*/config/*.yaml` under
`--pipeline-root` (default: the current directory when it has an `egs/`
folder). Pipeline jobs and the audio of the last six traces are kept in memory only, apart from the run history.

## Live protocol

`/api/live` is a WebSocket. The client's first message is JSON:
`{"model_id", "variant", "provider", "parameters"}`; the server answers
`{"type": "ready", "sample_rate", "hop_length", "win_length",
"latency_samples", ...}`. Each binary message is then a little-endian
`uint32` sequence number followed by float32 samples at the model rate (at
most 16000); each is answered with the same sequence number, a float32
processing time in milliseconds, and the float32 samples the stream emitted.
The reply stream carries the model's output `latency_samples` late, as the
offline runtime's does. The server sends `{"type": "stats"}` about once per
second of audio; `{"type": "stop"}` ends the session. A session ends after
30 minutes.

## Onset guard

Voice-isolation requests may override these parameters:

- `onset_guard`
- `onset_guard_t_arm_s`
- `onset_guard_t_forget_s`
- `onset_guard_tau_dn_s`
- `onset_guard_margin_db`

The guard passes the input through until sustained speech is detected, then
releases toward the model output. It protects the beginning of an utterance at
the cost of weaker early suppression. Omit the override to use the model
manifest.

## Acoustic world

The `#/world` screen, its API and the scene package are described in
[world.md](world.md).
