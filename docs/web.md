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
  alignment and dry-blend overrides.
- **Speaker Verification** uploads `enrollment` and `test` audio, calls the
  waveform embedding processor twice, and displays the cosine score and
  threshold verdict.

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
| POST | `/api/infer` | Run a model with JSON named inputs |
| GET | `/api/runs/{run_id}/{output}` | Download a generated audio output |

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
