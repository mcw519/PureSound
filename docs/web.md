# PureSound web workspace

The repository includes a small browser client for the Model Zoo and unified
ONNX facade. It deliberately uses the Python standard library and plain HTML,
CSS, and JavaScript so the service stays easy to audit and deploy.

Start it from the repository root:

```bash
puresound web
# open http://127.0.0.1:7860
```

The bind address and port are CLI arguments:

```bash
puresound web --ip 0.0.0.0 --port 8080
# --host is an alias of --ip
```

Binding to `0.0.0.0` exposes inference to the surrounding network. The service
does not provide authentication, so only do this on a trusted network or
behind an authenticated reverse proxy.

The browser client has three screens:

- **Model Zoo** lists the catalog's runnable models and exposes the same
  artifact, lifecycle, and preprocessing metadata as `puresound models list`.
- **Voice Isolation** uploads one audio file and calls the named `audio` input
  through `stft_frame_ort`. The output is available as a browser player and a
  WAV download; the processor remains responsible for manifest delay
  alignment and dry-blend overrides. The result also includes output level
  metrics and a reference-free DNSMOS score when the scorer is available, so a
  Playground run is already a small quality check.
- **Speaker Verification** uploads `enrollment` and `test` audio, calls the
  waveform embedding processor twice, and displays the cosine score and
  threshold verdict.
- **Measurements** accepts a probe recording, an optional clean reference, and
  one or more Voice Isolation models. It reports reproducible level/spectral
  metrics (RMS, peak, clipping, silence, zero-crossing rate, and spectral
  centroid), reference-free DNSMOS when requested, plus SI-SDR, SNR,
  correlation, STOI, and PESQ when a reference is available. The page explains
  the three-step flow and keeps Model Zoo integrity checks in a separate
  technical disclosure. Setup lives in a right-side drawer so the report keeps
  the full workspace width. Successful model outputs are latency-aligned,
  stored in the bounded in-memory run store, and shown with the same waveform,
  spectrogram, playback boost, and limiter controls used by the Playground.
  A comparison fails when no model succeeds and reports a warning when only
  some selected models fail.

Provider choices are shared by the browser, CLI, and legacy streaming runtime:
`auto` prefers CUDA, then Apple `CoreML`, then CPU; `cpu` forces CPU; `cuda`
requests NVIDIA CUDA; and `coreml` requests Apple's CoreML execution provider.
`mps` is accepted as a user-facing alias for `coreml` because ONNX Runtime does
not expose a native MPS execution provider. The `/api/health` response reports
the providers registered by the active ONNX Runtime wheel, and the Playground
disables unavailable explicit choices. A requested CUDA/CoreML provider may
still fall back to CPU if its shared libraries or driver cannot be loaded; each
inference result reports the actual provider list.

The completed Measurements report can be downloaded directly as JSON (full
request, input/reference metrics, and per-model results) or CSV (one row per
model). These downloads are generated in the browser and do not persist data
on the server.

The primary navigation can be collapsed and the browser remembers that
preference. Playground configuration uses task-specific right-side drawers, so
Voice Isolation and Speaker Verification keep the full workspace width while
their settings are closed.

Every uploaded or generated audio file uses the same inspection panel. It
provides a shared time ruler, synchronized waveform and 0–8 kHz spectrogram,
click-to-seek playback, and Wave/Both/Spec view modes. Playback boost ranges
from 0 to +48 dB and includes an **Auto** setting that targets -1.5 dBFS plus a
safety limiter. Boost is audition-only: it never changes the model input or
the downloaded WAV.

The API is intentionally small:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Runtime and catalog summary |
| GET | `/api/models?task=voice_isolation` | Catalog entries |
| GET | `/api/models/{model_id}` | One model contract |
| GET | `/api/validate` | Paths, sidecars, hashes, and graph checks |
| POST | `/api/infer` | Run a model with JSON named inputs; optionally attach output measurements |
| POST | `/api/jobs` | Start asynchronous inference or measurement and return a job id |
| GET | `/api/jobs/{job_id}` | Read progress, status, and result |
| GET | `/api/jobs?limit=20` | List recent inference and measurement jobs |
| POST | `/api/jobs/{job_id}/cancel` | Request cancellation of a queued/running job |
| POST | `/api/measure` | Measure a probe and compare selected Voice Isolation models, including optional DNSMOS |
| GET | `/api/runs/{run_id}/{output}` | Download a generated audio output |

Asynchronous jobs expose `queued`, `running`, `succeeded`, `failed`, or
`cancelled` status. Voice Isolation reports progress for every streaming frame;
Speaker Verification reports its audio loading, enrollment embedding, test
embedding, and scoring phases. Submit a measurement through the same endpoint
with `"kind": "measurement"`; its progress includes the current model and that
model's frame progress. Cancellation takes effect at the next streaming frame
or between embedding calls. ONNX Runtime does not forcibly interrupt a single
active session call.

Uploaded inputs are JSON descriptors such as:

```json
{
  "filename": "speech.wav",
  "data": "data:audio/wav;base64,..."
}
```

Local filesystem paths are disabled by default. For a trusted local client,
`puresound web --allow-local-paths` enables the compatibility form
`{"path": "/absolute/path/input.wav"}`. Generated outputs stay in a bounded
in-memory store and are discarded when the process exits.

To request reference-free quality metrics from either inference or measurement,
include `"measurements": {"include_dnsmos": true}`. The browser clients send
this flag automatically. DNSMOS is optional: if its scorer or model assets are
unavailable, the response keeps the level metrics and reports the DNSMOS error
instead of failing the run.
