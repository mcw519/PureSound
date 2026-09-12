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

## Screens

- **Model Zoo:** model metadata and artifact validation.
- **Voice Isolation:** upload audio, run a model, listen, and download WAV.
- **Speaker Verification:** compare enrollment and test recordings.
- **Measurements:** compare voice-isolation outputs with optional reference
  metrics and DNSMOS.

Playback gain and the browser limiter do not change model inputs or downloaded
files.

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
| GET | `/api/validate` | Validate artifacts and ONNX graphs |
| POST | `/api/infer` | Run synchronous inference |
| POST | `/api/measure` | Compare models and metrics |
| POST | `/api/jobs` | Start an asynchronous job |
| GET | `/api/jobs/{job_id}` | Read job status and result |
| POST | `/api/jobs/{job_id}/cancel` | Request cancellation |
| GET | `/api/runs/{run_id}/{output}` | Download generated audio |

Asynchronous job states are `queued`, `running`, `succeeded`, `failed`,
and `cancelled`.

## Audio input

Browser uploads use a data URL:

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

Generated files are held in a bounded in-memory store and are removed when the
server stops.

## Measurements

Set `"measurements": {"include_dnsmos": true}` to request DNSMOS. If the
optional scorer is unavailable, other metrics are still returned.

With a clean reference, measurements can include SI-SDR, SNR, correlation,
STOI, and PESQ. Without one, level and spectral measurements remain available.

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
