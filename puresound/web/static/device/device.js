/* On-device inference: the released streaming models run in this browser, in
 * ONNX Runtime Web (WebAssembly) inside a dedicated Worker (worker.js), through
 * the streaming runtime of sdk/web.  Playground's "This device" runs use it, and
 * so can any screen: window.PureSoundDevice, described in sdk/web/README.md.
 * `npm run assets` in sdk/web writes the runtime, ONNX Runtime and the model
 * files next to this one.  The pure helpers are also loaded by Node
 * (test/web/device.test.cjs). */
(function (root, factory) {
  const api = factory(root);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundDevice = api;
})(typeof self !== "undefined" ? self : this, function (root) {
  "use strict";

  const BASE = "/device/";
  const MODEL_RATE = 16000;
  const MAX_THREADS = 4;
  // The longest recording a run here takes: the bound the recorder has.
  const MAX_SECONDS = 10 * 60;
  // A live stream needs headroom over real time for the audio thread and the page.
  const LIVE_RTF_LIMIT = 0.7;
  // Chunks a live stream may have waiting before it stops: one second of audio.
  const LIVE_BACKLOG = 50;

  function fill(text, vars) {
    return String(text).replace(/\{(\w+)\}/g, (match, name) => (vars && vars[name] != null ? String(vars[name]) : match));
  }
  const t = (key, vars) => (root && root.PureSoundI18n ? root.PureSoundI18n.t(key, vars) : fill(key, vars));

  // ------------------------------------------------------------ pure helpers

  /* WebAssembly threads need SharedArrayBuffer, which only a cross-origin
   * isolated page has; `forced` 1 (from ?threads=1) is the single-thread baseline. */
  function threadCount({ isolated, cores, forced = null }) {
    if (!isolated || forced === 1) return 1;
    return Math.min(MAX_THREADS, Math.max(1, Math.floor(cores) || 1));
  }

  /* A model manifest with one run's settings.  `dryBlend` is the share of the
   * model output, in (0, 1]; `onsetGuard` false turns the guard off, true or an
   * object of knobs (t_arm_s, t_forget_s, tau_dn_s, margin_db, ...) turns it on
   * over the manifest's own, and undefined leaves the manifest in charge. */
  function configuredManifest(manifest, { dryBlend, onsetGuard } = {}) {
    const out = { ...manifest, recommended_inference: { ...(manifest.recommended_inference || {}) } };
    if (dryBlend != null) {
      const value = Number(dryBlend);
      if (!(value > 0 && value <= 1)) throw new RangeError("dryBlend must be in (0, 1]");
      out.recommended_inference.dry_blend = value;
    }
    if (onsetGuard === false) delete out.onset_guard;
    else if (onsetGuard) out.onset_guard = { ...(manifest.onset_guard || {}), ...(onsetGuard === true ? {} : onsetGuard) };
    return out;
  }

  /* Samples by which the emitted stream trails its input: the graph's look-ahead. */
  function latencySamples(manifest) {
    return (manifest.streaming_delay_frames || 0) * manifest.hop_length;
  }

  /* The emitted stream on the input's time axis: the look-ahead dropped from
   * the front and the flush tail past `length` cut off. */
  function alignStream(stream, length, delay) {
    const out = new Float32Array(length);
    out.set(stream.subarray(Math.min(delay, stream.length), Math.min(stream.length, delay + length)));
    return out;
  }

  function difference(a, b) {
    const out = new Float32Array(a.length);
    for (let index = 0; index < a.length; index += 1) out[index] = a[index] - b[index];
    return out;
  }

  // ---------------------------------------------------------------- browser

  function cancelled() {
    return Object.assign(new Error(t("Inference cancelled.")), { name: "AbortError", cancelled: true });
  }

  /* Stop waiting for `promise` when `signal` aborts; the work itself goes on. */
  function abortable(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) return Promise.reject(cancelled());
    return new Promise((resolve, reject) => {
      const abort = () => reject(cancelled());
      signal.addEventListener("abort", abort, { once: true });
      promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    });
  }

  function environment() {
    const isolated = Boolean(root.crossOriginIsolated) && typeof SharedArrayBuffer !== "undefined";
    let forced = null;
    try { forced = Number(new URLSearchParams(root.location.search).get("threads")) || null; } catch { /* no location */ }
    return { isolated, threads: threadCount({ isolated, cores: root.navigator?.hardwareConcurrency, forced }) };
  }

  let catalogRequest = null;
  function loadCatalog() {
    catalogRequest = catalogRequest || (async () => {
      if (typeof WebAssembly !== "object" || typeof Worker !== "function") throw new Error(t("This browser cannot run models itself: it has no WebAssembly or Worker support."));
      const response = await fetch(`${BASE}catalog.json`);
      if (!response.ok) throw new Error(t("No device builds yet: run npm run build and npm run assets in sdk/web."));
      return response.json();
    })();
    catalogRequest.catch(() => { catalogRequest = null; });
    return catalogRequest;
  }

  /* Whether models can run here, which ones, and with how many threads; when
   * they cannot, `reason` says why in the page's language. */
  async function status() {
    try {
      const catalog = await loadCatalog();
      const models = (catalog.models || []).map((entry) => entry.id);
      return { ready: models.length > 0, models, ...environment(), reason: models.length ? null : t("No device builds yet: run npm run build and npm run assets in sdk/web.") };
    } catch (error) {
      return { ready: false, models: [], ...environment(), reason: error.message };
    }
  }

  async function sha256(bytes) {
    const digest = new Uint8Array(await root.crypto.subtle.digest("SHA-256", bytes));
    return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
  }

  /* A catalog file, checked against the catalog's hash where the page can hash
   * (WebCrypto needs a secure page): a rebuild half-copied fails here. */
  async function fetchChecked(path, expected, signal) {
    const response = await fetch(BASE + path, { signal });
    if (!response.ok) throw new Error(t("Could not download {file} ({status}).", { file: path, status: response.status }));
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (expected && root.crypto?.subtle && (await sha256(bytes)) !== expected) throw new Error(t("{file} does not match the device catalog: build the assets again.", { file: path }));
    return bytes;
  }

  /* A model's manifest and ONNX bytes, downloaded once per page.  A download
   * that fails or is aborted is forgotten, so the next run starts it again. */
  const downloads = new Map();
  function loadModel(id) {
    if (downloads.has(id)) return downloads.get(id);
    const controller = new AbortController();
    const request = (async () => {
      const catalog = await loadCatalog();
      const entry = (catalog.models || []).find((item) => item.id === id);
      if (!entry) throw new Error(t("{model} has no device build.", { model: id }));
      const manifest = JSON.parse(new TextDecoder().decode(await fetchChecked(entry.manifest, catalog.files?.[entry.manifest], controller.signal)));
      const model = await fetchChecked(entry.model, catalog.files?.[entry.model], controller.signal);
      return { manifest, model };
    })();
    const download = { request, controller, settled: false };
    downloads.set(id, download);
    request.then(() => { download.settled = true; }, () => { if (downloads.get(id) === download) downloads.delete(id); });
    return download;
  }

  /* One worker holding one model.  Work inside the worker cannot be
   * interrupted, so cancelling a call ends the worker. */
  class Engine {
    constructor(id) {
      this.id = id;
      this.calls = new Map();
      this.sequence = 0;
      this.dead = null;
    }

    async start() {
      try {
        this.download = loadModel(this.id);
        const { manifest, model } = await this.download.request;
        if (this.dead) throw this.dead;
        this.manifest = manifest;
        this.worker = new Worker(`${BASE}worker.js`, { type: "module" });
        this.worker.onmessage = ({ data }) => this.receive(data);
        this.worker.onerror = (event) => {
          event.preventDefault();
          this.terminate(new Error(event.message || t("The on-device runtime stopped.")));
        };
        const { threads } = await this.call("load", { model: model.slice(), threads: environment().threads });
        this.threads = threads;
        return this;
      } catch (error) {
        this.terminate(error);
        throw error;
      }
    }

    call(type, data = {}, { transfer = [], signal = null, onProgress = null } = {}) {
      if (this.dead) return Promise.reject(this.dead);
      if (signal?.aborted) return Promise.reject(cancelled());
      const id = ++this.sequence;
      const abort = () => this.terminate(cancelled());
      signal?.addEventListener("abort", abort, { once: true });
      return new Promise((resolve, reject) => {
        this.calls.set(id, { resolve, reject, onProgress });
        this.worker.postMessage({ id, type, ...data }, transfer);
      }).finally(() => signal?.removeEventListener("abort", abort));
    }

    receive({ id, type, ...data }) {
      const call = this.calls.get(id);
      if (!call) return;
      if (type === "progress") { call.onProgress?.(Math.min(1, data.value)); return; }
      this.calls.delete(id);
      if (type === "error") call.reject(new Error(data.error));
      else call.resolve(data);
    }

    terminate(reason = cancelled()) {
      if (this.dead) return;
      this.dead = reason;
      if (this.download && !this.download.settled) this.download.controller.abort();
      this.worker?.terminate();
      this.calls.forEach((call) => call.reject(reason));
      this.calls.clear();
    }
  }

  // The last offline run's engine stays loaded for the next run of that model.
  let warm = null;
  async function engineFor(id, signal) {
    if (!warm || warm.dead || warm.id !== id) {
      warm?.terminate();
      warm = new Engine(id);
      warm.ready = warm.start();
      warm.ready.catch(() => {});
    }
    const engine = warm;
    try {
      await abortable(engine.ready, signal);
    } catch (error) {
      engine.terminate(error);
      throw error;
    }
    return engine;
  }

  /* Decoded audio as mono samples.  The browser decodes at `sampleRate`; pass
   * the file's own rate (WAV and FLAC give it in their header) so it is not
   * resampled twice. */
  async function decode(blob, { sampleRate = 48000 } = {}) {
    const context = new OfflineAudioContext(1, 1, sampleRate);
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const samples = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let index = 0; index < data.length; index += 1) samples[index] += data[index] / buffer.numberOfChannels;
    }
    return { samples, sampleRate: buffer.sampleRate };
  }

  async function runOffline(samples, { model, sampleRate = MODEL_RATE, dryBlend, onsetGuard, onProgress = () => {}, signal = null } = {}) {
    if (!(samples instanceof Float32Array) || !samples.length) throw new TypeError("samples must be a non-empty Float32Array");
    if (samples.length > MAX_SECONDS * sampleRate) throw new Error(t("A run on this device takes at most {minutes} minutes of audio; run longer recordings on the server.", { minutes: MAX_SECONDS / 60 }));
    const started = performance.now();
    onProgress(null, "load");
    const engine = await engineFor(model, signal);
    const manifest = configuredManifest(engine.manifest, { dryBlend, onsetGuard });
    onProgress(0, "run");
    const copy = samples.slice();
    const { input, stream, rtf } = await engine.call("process", { manifest, samples: copy, sampleRate }, { transfer: [copy.buffer], signal, onProgress: (value) => onProgress(value, "run") });
    const delay = latencySamples(manifest);
    const output = alignStream(stream, input.length, delay);
    return {
      model,
      sampleRate: MODEL_RATE,
      input,
      output,
      stream,
      removed: difference(input, output),
      latencySamples: delay,
      latencyMs: (1000 * delay) / MODEL_RATE,
      windowSamples: manifest.win_length,
      hopSamples: manifest.hop_length,
      dryBlend: manifest.recommended_inference.dry_blend ?? 1,
      onsetGuard: manifest.onset_guard || null,
      rtf,
      threads: engine.threads,
      seconds: (performance.now() - started) / 1000,
    };
  }

  // Runs take turns: two screens asking at once must not share one worker's stream.
  let lane = Promise.resolve();
  function process(samples, options) {
    const run = lane.then(() => runOffline(samples, options));
    lane = run.catch(() => {});
    return run;
  }

  /* A live stream on this device for PureSoundCapture.LiveSession, which
   * owns the microphone, the monitor and the statistics. */
  function liveLink({ model, dryBlend, onsetGuard } = {}) {
    let engine = null;
    let handlers = {};
    let waiting = 0;
    let ended = false;
    let chunkMs = 20;
    const compute = [];
    const end = (message) => {
      if (ended) return;
      ended = true;
      handlers.onEnded?.(message);
    };
    return {
      runner: "device",
      get rtf() {
        if (!compute.length) return null;
        const sorted = [...compute].sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)] / chunkMs;
      },
      async open(callbacks = {}) {
        handlers = callbacks;
        engine = new Engine(model);
        await engine.start();
        const manifest = configuredManifest(engine.manifest, { dryBlend, onsetGuard });
        const chunk = root.PureSoundCapture?.CHUNK_SAMPLES || 320;
        chunkMs = (1000 * chunk) / MODEL_RATE;
        const { rtf } = await engine.call("open", { manifest, chunk });
        if (rtf > LIVE_RTF_LIMIT) throw new Error(t("This device runs the model at RTF {rtf}, too slow to keep up live (at most {limit}). Record instead, then run the recording.", { rtf: rtf.toFixed(2), limit: LIVE_RTF_LIMIT }));
        const latency = latencySamples(manifest);
        return {
          runner: "device",
          sample_rate: MODEL_RATE,
          hop_length: manifest.hop_length,
          win_length: manifest.win_length,
          latency_samples: latency,
          latency_ms: (1000 * latency) / MODEL_RATE,
          threads: engine.threads,
          rtf,
        };
      },
      send(sequence, samples) {
        if (ended || !engine || engine.dead) return;
        if (waiting >= LIVE_BACKLOG) {
          end(t("This device fell a second behind the microphone, so the session stopped. Record instead, then run the recording."));
          return;
        }
        waiting += 1;
        engine.call("chunk", { samples }).then(({ samples: out, computeMs }) => {
          waiting -= 1;
          compute.push(computeMs);
          if (compute.length > 250) compute.shift();
          handlers.onReply?.(sequence, computeMs, out);
        }, (error) => { if (!error.cancelled) end(error.message); });
      },
      /* Ends the stream: every chunk sent is answered, then the flush tail. */
      async stop() {
        if (!engine || engine.dead) return {};
        const { samples } = await engine.call("flush");
        handlers.onReply?.(null, 0, samples);
        return {};
      },
      close() {
        ended = true;
        engine?.terminate();
      },
    };
  }

  return { MODEL_RATE, status, decode, process, liveLink, threadCount, configuredManifest, alignStream };
});
