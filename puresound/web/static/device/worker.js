/* The worker behind device.js: one released model in ONNX Runtime Web
 * (WebAssembly) and the streaming runtime of sdk/web.  Requests are handled one
 * at a time in arrival order, so a live stream's chunks and its flush, or a run
 * and the next one's settings, never interleave. */
import * as ort from "./vendor/ort/ort.wasm.min.mjs";
import { PureSoundStreamingRuntime, concat } from "./runtime/runtime.js";
import { resample } from "./runtime/audio.js";

const MODEL_RATE = 16000;
const FILE_CHUNK = 4096;
const PROGRESS_EVERY_MS = 100;

let session = null;
let runtime = null;

/* A runtime releases the session it was given when it rejects a manifest; this
 * one serves every configuration, so it is lent without that release. */
function lentSession() {
  return { run: (...args) => session.run(...args), release: async () => {} };
}

async function configure(manifest) {
  runtime = await PureSoundStreamingRuntime.create(null, manifest, { backend: ort, session: lentSession() });
}

const handlers = {
  async load({ model, threads }) {
    // Threads need SharedArrayBuffer, which only a cross-origin isolated worker has.
    ort.env.wasm.numThreads = threads > 1 && self.crossOriginIsolated ? threads : 1;
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
    session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    return { reply: { threads: ort.env.wasm.numThreads } };
  },

  /* A whole recording: resampled to the model rate, streamed through in file
   * chunks, flushed.  The reply is the stream as emitted, look-ahead included. */
  async process({ manifest, samples, sampleRate }, progress) {
    await configure(manifest);
    const input = sampleRate === MODEL_RATE ? samples : resample(samples, sampleRate, MODEL_RATE);
    const parts = [];
    const started = performance.now();
    let reported = started;
    for (let offset = 0; offset < input.length; offset += FILE_CHUNK) {
      parts.push(await runtime.processSamples(input.subarray(offset, offset + FILE_CHUNK)));
      if (performance.now() - reported >= PROGRESS_EVERY_MS) {
        reported = performance.now();
        progress((offset + FILE_CHUNK) / input.length);
      }
    }
    parts.push(await runtime.flush());
    const rtf = (performance.now() - started) / 1000 / (input.length / MODEL_RATE);
    const stream = concat(...parts);
    return { reply: { input, stream, rtf }, transfer: [input.buffer, stream.buffer] };
  },

  /* A live stream with these settings.  One second of a tone in live-sized
   * chunks measures whether this device keeps up; the stream then starts fresh. */
  async open({ manifest, chunk }) {
    await configure(manifest);
    const tone = Float32Array.from({ length: MODEL_RATE }, (_, index) => 0.05 * Math.sin((2 * Math.PI * 500 * index) / MODEL_RATE));
    const started = performance.now();
    for (let offset = 0; offset < tone.length; offset += chunk) await runtime.processSamples(tone.subarray(offset, offset + chunk));
    await runtime.flush();
    const rtf = (performance.now() - started) / 1000;
    runtime.reset();
    return { reply: { rtf } };
  },

  async chunk({ samples }) {
    const started = performance.now();
    const out = await runtime.processSamples(samples);
    return { reply: { samples: out, computeMs: performance.now() - started }, transfer: [out.buffer] };
  },

  async flush() {
    const out = await runtime.flush();
    return { reply: { samples: out }, transfer: [out.buffer] };
  },
};

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  const { id, type, ...request } = data;
  queue = queue.then(async () => {
    try {
      if (!handlers[type]) throw new Error(`Unknown request: ${type}`);
      const { reply, transfer = [] } = await handlers[type](request, (value) => postMessage({ id, type: "progress", value }));
      postMessage({ id, type: "result", ...reply }, transfer);
    } catch (error) {
      postMessage({ id, type: "error", error: String(error?.message ?? error) });
    }
  });
};
