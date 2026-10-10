# PureSound Web SDK

[繁體中文](README.zh-TW.md)

An independent TypeScript streaming runtime, with periodic Hann STFT, overlap-add,
explicit graph state, aligned dry blend and onset protection. ONNX Runtime Web is
pinned to 1.24.3. The web app runs it for Playground's *This device* runs (see
[the web UI guide](../../docs/usage/web.md#on-this-device)): the WASM backend runs in
a dedicated Worker, on one thread when the page is not cross-origin isolated and up
to four when it is. `puresound web` isolates every page it serves.

## Build

From this directory:

```sh
npm ci
npm run build    # src/ -> dist/
npm run assets   # -> puresound/web/static/device/
```

`npm run assets` (`tools/build_assets.py`) copies the compiled runtime, ONNX Runtime
Web and each released model -- its ONNX file and a strict JSON manifest -- into
`puresound/web/static/device/`, with `catalog.json` (the SHA-256 of every file, which
the page checks downloads against) and the third-party notices. It copies the
released models rather than exporting or training new ones. Then run
`.venv/bin/python -m puresound web` from the repository root and choose *This device*
in Playground's Settings.

## Runtime API

```ts
import { PureSoundStreamingRuntime } from '@puresound/streaming';
const runtime = await PureSoundStreamingRuntime.create(modelBytes, manifest);
const chunk = await runtime.processSamples(float32Mono16k);
const tail = await runtime.flush();
runtime.reset();
await runtime.dispose();
```

Call processing methods sequentially. Output is the input
`streaming_delay_frames * hop_length` samples late, matching the Python SDK; after
`flush` the stream holds exactly the input length plus that delay, so for file
playback drop that many samples from the front. `flush` is idempotent. `reset` starts a new independent stream; `dispose` releases the session.
A bundler resolves the default ONNX dependency; a static Worker can pass an imported
backend through `create(bytes, manifest, {backend: ort})`, as
`puresound/web/static/device/worker.js` does. Unsupported manifest processors and
postprocessing are rejected explicitly.

## In the web app: `window.PureSoundDevice`

`puresound/web/static/device/device.js` runs the runtime in a Worker and is the one
way a screen of the web app runs a model on the device; Playground uses it for File,
Record and Live. A page loads it with `<script src="/device/device.js" defer>`.

| Call | Resolves to |
| --- | --- |
| `status()` | `{ready, models, threads, isolated, reason}`: whether models can run here, the ids with a device build, the thread count, and why not (in the page's language) when they cannot |
| `decode(blob, {sampleRate = 48000})` | `{samples, sampleRate}`: audio the browser can decode, as a mono `Float32Array`; pass the file's own rate so it is not resampled twice |
| `process(samples, options)` | the whole recording through a model, offline (below) |
| `liveLink({model, dryBlend, onsetGuard})` | a live link for `PureSoundCapture.LiveSession` (`capture.js`), which owns the microphone, the monitor and the statistics |

`process(samples, {model, sampleRate = 16000, dryBlend, onsetGuard, onProgress, signal})`
resamples a mono `Float32Array` to 16 kHz in the worker, streams it through the model
and flushes the stream. The result has `input` (the 16 kHz input), `output` (the stream
with the model's look-ahead dropped, as long as `input`), `stream` (as emitted),
`removed` (`input − output`), `latencySamples`, `latencyMs`, `rtf`, `threads`, the
`dryBlend` and `onsetGuard` that applied, and `seconds`. `dryBlend` is in (0, 1];
`onsetGuard` is `false` (off), `true` or an object of knobs (`t_arm_s`, `t_forget_s`,
`tau_dn_s`, `margin_db`, ...) over the manifest's own, or left out for the manifest's
setting. `onProgress(fraction, phase)` reports `"load"` (fraction `null`) and then
`"run"`. Aborting `signal` rejects with an error whose `cancelled` is `true` and ends
the worker; the next call loads the model again. Calls take turns, the last model
used stays loaded, and input over ten minutes is refused.

```js
const run = await PureSoundDevice.process(samples, { model: "voice-isolate-dpcrn-curriculum-v1", sampleRate: 16000, signal });

const session = new PureSoundCapture.LiveSession({ onStats, onLevel, onStatus });
await session.start({ link: PureSoundDevice.liveLink({ model, dryBlend: 0.9 }), monitor: "off" });
const { input, output, removed } = await session.stop();
```

A live link measures the device when it opens: one second of audio in 20 ms chunks
must run at RTF 0.7 or less, or it refuses with a message suggesting a recording. A
session that falls one second (50 chunks) behind ends, keeping what it streamed.
`PureSoundCapture.serverLink` is the same link to the server's `/api/live`.

## Validation

From the repository root:

```sh
.venv/bin/python sdk/web/tools/generate_fixtures.py
```

Then from this directory:

```sh
npm test
npx playwright install chromium firefox webkit
npm run test:browser
npm run test:microphone
```

Fixtures compare every model the page ships a device build of to Python on short input, silence, speech,
explicit onset-guard arming/forgetting, arbitrary chunks, repeated flush and reset,
with NRMS ≤ 1e-4.

The browser and microphone suites drive Playground in *This device* mode against a
running `puresound web` with device assets built, at port 7861; `PURESOUND_WEB_URL`
changes it, `PURESOUND_BROWSERS=chromium` limits a run to one engine, and
`PURESOUND_CHROMIUM_PATH` points Playwright at a Chromium it did not install. The
browser suite checks that the page is cross-origin isolated, runs every model with a
device build on a sample, finds server-only models listed and marked, recovers from a
failed and a cancelled model download and a cancelled run, compares the device output
with the server's on the same sample (NRMS < 2e-3 after 16-bit export), and follows
the language switch. `?threads=1` in the page address forces the single-thread
baseline. The microphone suite uses a simulated capture device: it records and runs
the recording on the device, streams live -- or checks that a device too slow for
live is refused -- and checks a denied microphone; `PURESOUND_LIVE_SECONDS=600` runs
a ten-minute stream. Actual Safari still requires a macOS release check (Playwright
WebKit is a separate engine check), and actual audio devices a check of their own.

Architecture references: [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/build-web-app.html)
and [WASM flags and thread requirements](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html).

## License

Original SDK code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE)
for exclusions. PureSound's own trained models, including model zoo checkpoints
and ONNX exports, also use Apache-2.0. Dependencies and third-party models retain
their own licenses.
The files under `licenses/` contain ONNX Runtime's license and third-party notices.
When copying the SDK into another project, include LICENSE, NOTICE, and the
applicable third-party notices.
