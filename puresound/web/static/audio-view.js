/* Shared browser-only audio helpers for the comparison deck (compare-deck.js):
 * the playback chain, waveform and ruler drawing, spectrogram rendering and
 * WAV encoding.  Visuals and playback boost never modify inference inputs or
 * exports. */
(() => {
  "use strict";
  const t = (key, vars) => (typeof window !== "undefined" && window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);

  const MAX_BOOST_DB = 48;
  const SAFE_PEAK_DBFS = -1.5;
  const SPEC_MAX_HZ = 8000;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const linearGain = (db) => 10 ** (db / 20);
  const levelDb = (value) => 20 * Math.log10(Math.max(value, 1e-9));
  const safetyCurve = Float32Array.from({ length: 2048 }, (_, index) => {
    const value = index / 2047 * 2 - 1;
    const magnitude = Math.abs(value);
    const knee = 0.7;
    return Math.sign(value) * (magnitude <= knee ? magnitude : knee + (1 - knee) * Math.tanh((magnitude - knee) / (1 - knee)));
  });

  /* Spectrogram settings a viewer can change; the deck keeps them per browser. */
  const DEFAULT_SPECTROGRAM = Object.freeze({
    fftSize: 1024,
    maxHz: SPEC_MAX_HZ,
    scale: "linear",
    floorDb: -92,
    rangeDb: 82,
    colormap: "puresound",
  });

  let sharedContext = null;
  // One audible player at a time: starting any player pauses the one that was
  // playing, so an input and its output never sound on top of each other.
  let audiblePlayer = null;

  function claimPlayback(player) {
    if (audiblePlayer && audiblePlayer !== player) audiblePlayer.pause();
    audiblePlayer = player;
  }

  function releasePlayback(player) {
    if (audiblePlayer === player) audiblePlayer = null;
  }

  function audioContext() {
    if (!sharedContext) sharedContext = new (window.AudioContext || window.webkitAudioContext)();
    return sharedContext;
  }

  /* Start the audio clock.  Without an output device some browsers never
   * settle the resume() promise; after two seconds the page says so instead
   * of leaving Play silently doing nothing. */
  async function resumeAudio() {
    const context = audioContext();
    if (context.state === "running") return true;
    const started = await Promise.race([
      context.resume().then(() => true, () => false),
      new Promise((resolve) => window.setTimeout(() => resolve(false), 2000)),
    ]);
    if (started && context.state === "running") return true;
    window.dispatchEvent(new CustomEvent("puresound:audio-unavailable"));
    return false;
  }

  function formatTime(seconds) {
    const millis = Math.round(Math.max(0, Number(seconds) || 0) * 1000);
    const minutes = Math.floor(millis / 60000);
    return `${minutes}:${((millis - minutes * 60000) / 1000).toFixed(3).padStart(6, "0")}`;
  }

  function formatHz(hz) {
    return hz >= 1000 ? `${(hz / 1000).toFixed(hz % 1000 ? 1 : 0)}k` : `${Math.round(hz)}`;
  }

  /* Boost -> limiter -> soft safety clip -> speakers.  Playback only. */
  function createOutputChain(context, boostDb) {
    const boost = context.createGain();
    boost.gain.value = linearGain(boostDb);
    const limiter = context.createDynamicsCompressor();
    limiter.threshold.value = -1;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.1;
    const shaper = context.createWaveShaper();
    shaper.curve = safetyCurve;
    shaper.oversample = "4x";
    boost.connect(limiter).connect(shaper).connect(context.destination);
    return {
      input: boost,
      limiter,
      setBoost(db) { boost.gain.setTargetAtTime(linearGain(db), context.currentTime, 0.02); },
      disconnect() { [boost, limiter, shaper].forEach((node) => { try { node.disconnect(); } catch { /* already detached */ } }); },
    };
  }

  function mixToMono(buffer) {
    if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
    const mono = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const values = buffer.getChannelData(channel);
      for (let index = 0; index < values.length; index += 1) mono[index] += values[index] / buffer.numberOfChannels;
    }
    return mono;
  }

  function peakOf(samples) {
    let peak = 0;
    for (let index = 0; index < samples.length; index += 1) peak = Math.max(peak, Math.abs(samples[index]));
    return peak;
  }

  function rmsOf(samples) {
    let total = 0;
    for (let index = 0; index < samples.length; index += 1) total += samples[index] * samples[index];
    return Math.sqrt(total / Math.max(1, samples.length));
  }

  function fitCanvas(canvas) {
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.floor(bounds.width * ratio));
    const height = Math.max(1, Math.floor(bounds.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { context: canvas.getContext("2d"), width, height, ratio };
  }

  /* Time ruler over [start, end] seconds, with an optional shaded region. */
  function drawRuler(canvas, start, end, { region = null } = {}) {
    const { context, width, height, ratio } = fitCanvas(canvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#11122d";
    context.fillRect(0, 0, width, height);
    const span = end - start;
    if (!(span > 0)) return;
    if (region) {
      const left = (region.start - start) / span * width;
      const right = (region.end - start) / span * width;
      context.fillStyle = "rgba(252,76,2,.28)";
      context.fillRect(left, 0, right - left, height);
    }
    const steps = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    const step = steps.find((candidate) => span / candidate <= Math.max(2, width / (75 * ratio))) || 600;
    context.font = `${9 * ratio}px ui-monospace, monospace`;
    context.textBaseline = "middle";
    for (let time = Math.ceil(start / step - 1e-9) * step; time <= end + 1e-9; time += step) {
      const x = (time - start) / span * width;
      context.strokeStyle = "rgba(255,255,255,.18)";
      context.beginPath(); context.moveTo(x, height - 7 * ratio); context.lineTo(x, height); context.stroke();
      context.fillStyle = "rgba(255,255,255,.62)";
      const label = step < 0.01 ? `${time.toFixed(4)} s` : formatTime(time).replace(/\.000$/, "");
      context.fillText(label, x + 4 * ratio, height / 2);
    }
  }

  // The dB waveform shows 0 to -60 dBFS: a residual 40 dB down is a flat line
  // on a linear axis and a visible shape on this one.
  const WAVE_DB_RANGE = 60;

  function waveAmplitude(value, scale, gain) {
    if (scale !== "db") return clamp(value * gain, -1, 1);
    const magnitude = Math.abs(value) * gain;
    const scaled = clamp((levelDb(magnitude) + WAVE_DB_RANGE) / WAVE_DB_RANGE, 0, 1);
    return Math.sign(value) * scaled;
  }

  /* Min/max waveform of samples[startSample, endSample); `gain` magnifies it. */
  function drawWave(canvas, samples, color, { startSample = 0, endSample = samples?.length || 0, scale = "linear", gain = 1 } = {}) {
    const { context, width, height, ratio } = fitCanvas(canvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#010120";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(255,255,255,.1)";
    context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
    if (!samples?.length || endSample <= startSample) return;
    const samplesPerPixel = (endSample - startSample) / width;
    context.fillStyle = color;
    context.globalAlpha = 0.9;
    if (samplesPerPixel < 1) {
      // Zoomed in past one sample per pixel: draw the samples as a line.
      context.strokeStyle = color;
      context.lineWidth = Math.max(1, ratio);
      context.beginPath();
      for (let index = Math.max(0, startSample - 1); index <= Math.min(samples.length - 1, endSample + 1); index += 1) {
        const x = (index - startSample) / samplesPerPixel;
        const y = (1 - waveAmplitude(samples[index], scale, gain)) * height / 2;
        if (index === Math.max(0, startSample - 1)) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    } else {
      for (let pixel = 0; pixel < width; pixel += 1) {
        const first = startSample + Math.floor(pixel * samplesPerPixel);
        if (first >= samples.length) break;
        const last = Math.max(first + 1, Math.min(samples.length, startSample + Math.ceil((pixel + 1) * samplesPerPixel)));
        let low = 1;
        let high = -1;
        for (let index = first; index < last; index += 1) {
          low = Math.min(low, samples[index]);
          high = Math.max(high, samples[index]);
        }
        const top = (1 - waveAmplitude(high, scale, gain)) * height / 2;
        const bottom = (1 - waveAmplitude(low, scale, gain)) * height / 2;
        context.fillRect(pixel, top, 1, Math.max(1, bottom - top));
      }
    }
    context.globalAlpha = 1;
    const labels = [scale === "db" ? `dB · ${-WAVE_DB_RANGE}…0` : "", gain !== 1 ? `×${gain}` : ""].filter(Boolean).join(" · ");
    if (labels) {
      context.fillStyle = "rgba(255,255,255,.45)";
      context.font = `${8 * ratio}px ui-monospace, monospace`;
      context.textBaseline = "top";
      context.textAlign = "right";
      context.fillText(labels, width - 6 * ratio, 4 * ratio);
      context.textAlign = "left";
    }
  }

  /* RMS level in dBFS of consecutive frames, for level traces and hover. */
  function frameLevels(samples, sampleRate, frameSeconds = 0.02) {
    const frame = Math.max(1, Math.round(sampleRate * frameSeconds));
    const count = Math.ceil((samples?.length || 0) / frame);
    const levels = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      const first = index * frame;
      const last = Math.min(samples.length, first + frame);
      let total = 0;
      for (let sample = first; sample < last; sample += 1) total += samples[sample] * samples[sample];
      levels[index] = levelDb(Math.sqrt(total / Math.max(1, last - first)));
    }
    return { levels, frameSeconds: frame / sampleRate };
  }

  /* WAV ------------------------------------------------------------------- */
  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
    text(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    text(8, "WAVE");
    text(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    text(36, "data");
    view.setUint32(40, samples.length * 2, true);
    for (let index = 0; index < samples.length; index += 1) {
      const value = clamp(samples[index], -1, 1);
      view.setInt16(44 + index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    }
    return buffer;
  }

  /* [start, end) seconds of a 16-bit PCM WAV, cut from its own samples at its
   * own rate; null when the bytes are some other format. */
  function sliceWav(bytes, startSeconds, endSeconds) {
    if (!bytes || bytes.byteLength < 44) return null;
    const view = new DataView(bytes);
    const text = (offset) => String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
    if (text(0) !== "RIFF" || text(8) !== "WAVE") return null;
    let format = null;
    for (let offset = 12; offset + 8 <= bytes.byteLength;) {
      const id = text(offset);
      const size = view.getUint32(offset + 4, true);
      if (id === "fmt ") {
        format = { audioFormat: view.getUint16(offset + 8, true), channels: view.getUint16(offset + 10, true), rate: view.getUint32(offset + 12, true), bits: view.getUint16(offset + 22, true) };
      } else if (id === "data" && format && format.audioFormat === 1 && format.bits === 16) {
        const frameBytes = 2 * format.channels;
        const frames = Math.floor(Math.min(size, bytes.byteLength - offset - 8) / frameBytes);
        const first = clamp(Math.floor(startSeconds * format.rate), 0, frames);
        const last = clamp(Math.ceil(endSeconds * format.rate), first, frames);
        const body = new Uint8Array(bytes, offset + 8 + first * frameBytes, (last - first) * frameBytes);
        const out = new ArrayBuffer(44 + body.byteLength);
        const header = new DataView(out);
        new Uint8Array(out).set(new Uint8Array(bytes, 0, 12));
        new Uint8Array(out).set(new Uint8Array([0x66, 0x6d, 0x74, 0x20]), 12);
        header.setUint32(16, 16, true);
        header.setUint16(20, 1, true);
        header.setUint16(22, format.channels, true);
        header.setUint32(24, format.rate, true);
        header.setUint32(28, format.rate * frameBytes, true);
        header.setUint16(32, frameBytes, true);
        header.setUint16(34, 16, true);
        new Uint8Array(out).set(new Uint8Array([0x64, 0x61, 0x74, 0x61]), 36);
        header.setUint32(40, body.byteLength, true);
        header.setUint32(4, 36 + body.byteLength, true);
        new Uint8Array(out, 44).set(body);
        return { bytes: out, sampleRate: format.rate };
      }
      offset += 8 + size + (size % 2);
    }
    return null;
  }

  /* Spectrogram ---------------------------------------------------------- */
  let workerSequence = 0;
  let spectrogramWorker = null;
  const spectrogramJobs = new Map();

  function requestSpectrogram(payload) {
    if (typeof Worker === "undefined") return Promise.reject(new Error("Web Workers are unavailable"));
    if (!spectrogramWorker) {
      spectrogramWorker = new Worker("/audio-worker.js");
      spectrogramWorker.onmessage = ({ data }) => {
        const job = spectrogramJobs.get(data.id);
        if (!job) return;
        if (data.type === "progress") {
          job.progress(data.value);
          return;
        }
        spectrogramJobs.delete(data.id);
        if (data.type === "complete") job.resolve(data);
        else job.reject(new Error(data.error || "Spectrogram worker failed"));
      };
      spectrogramWorker.onerror = (event) => {
        spectrogramJobs.forEach((job) => job.reject(new Error(event.message || "Spectrogram worker failed")));
        spectrogramJobs.clear();
        spectrogramWorker.terminate();
        spectrogramWorker = null;
      };
    }
    const id = `spec-${workerSequence += 1}`;
    return new Promise((resolve, reject) => {
      spectrogramJobs.set(id, { resolve, reject, progress: payload.onProgress || (() => {}) });
      const message = { ...payload, id };
      delete message.onProgress;
      spectrogramWorker.postMessage(message, message.referenceFrames ? [message.frames, message.referenceFrames] : [message.frames]);
    });
  }

  /* One analysis frame per column: `fftSize` samples centred on the column's
   * time.  The payload is columns x fftSize whatever the recording's length. */
  function columnFrames(samples, sampleRate, startSeconds, endSeconds, columns, fftSize) {
    const frames = new Float32Array(columns * fftSize);
    const span = endSeconds - startSeconds;
    for (let column = 0; column < columns; column += 1) {
      const center = Math.round((startSeconds + (column + 0.5) / columns * span) * sampleRate);
      const first = center - fftSize / 2;
      const low = Math.max(0, first);
      const high = Math.min(samples.length, first + fftSize);
      if (high > low) frames.set(samples.subarray(low, high), column * fftSize + (low - first));
    }
    return frames;
  }

  /* Frequency ticks for a window, linear or logarithmic. */
  function frequencyTicks(low, high, scale) {
    if (scale === "log") return [50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000].filter((hz) => hz > low && hz < high);
    const span = high - low;
    const step = [50, 100, 200, 250, 500, 1000, 2000, 4000, 5000].find((candidate) => span / candidate <= 6) || 10000;
    const ticks = [];
    for (let hz = Math.ceil((low + 1) / step) * step; hz < high; hz += step) ticks.push(hz);
    return ticks;
  }

  /* One spectrogram canvas.  The image is computed off the main thread for a
   * time window and frequency window and scaled on resize; it is recomputed
   * only when a window or a setting changes, or the canvas outgrows it.  Given
   * a reference, it renders the target's level relative to it instead. */
  class SpectrogramView {
    constructor(canvas, progressLabel = null, { ticks = true } = {}) {
      this.canvas = canvas;
      this.progressLabel = progressLabel;
      this.ticks = ticks;
      this.image = null;
      this.key = null;
      this.result = null;
    }

    invalidate() {
      this.image = null;
      this.key = null;
      this.result = null;
      this.deferred = null;
    }

    showProgress(text, isError = false) {
      if (!this.progressLabel) return;
      this.progressLabel.textContent = text;
      this.progressLabel.classList.toggle("is-visible", Boolean(text));
      this.progressLabel.classList.toggle("is-error", isError);
    }

    /* options: { reference, referenceKey, settings, minHz, maxHz } */
    render(samples, sampleRate, startSeconds, endSeconds, options = {}) {
      const { reference = null, referenceKey = "", settings = DEFAULT_SPECTROGRAM } = options;
      const { context, width, height, ratio } = fitCanvas(this.canvas);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#010120";
      context.fillRect(0, 0, width, height);
      if (!samples?.length || !(endSeconds > startSeconds)) return;
      const maxHz = Math.min(options.maxHz ?? settings.maxHz, sampleRate / 2);
      const minHz = clamp(options.minHz ?? 0, 0, maxHz - 10);
      const fftSize = settings.fftSize;
      const columns = clamp(Math.ceil(width / 200) * 200, 200, 1600);
      const rows = clamp(Math.ceil(height / 64) * 64, 128, 512);
      const key = [startSeconds.toFixed(5), endSeconds.toFixed(5), samples.length, sampleRate, columns, rows, minHz.toFixed(1), maxHz.toFixed(1), fftSize, settings.scale, settings.floorDb, settings.rangeDb, settings.colormap, referenceKey].join(":");
      if (this.key !== key && this.inFlight) {
        // One request per view at a time: a resize or zoom storm would
        // otherwise queue a job per frame in the shared worker.  The latest
        // wish runs next.
        this.deferred = () => this.render(samples, sampleRate, startSeconds, endSeconds, options);
      } else if (this.key !== key) {
        this.key = key;
        this.image = null;
        this.result = null;
        this.inFlight = true;
        const frames = columnFrames(samples, sampleRate, startSeconds, endSeconds, columns, fftSize);
        const referenceFrames = reference ? columnFrames(reference, sampleRate, startSeconds, endSeconds, columns, fftSize) : null;
        this.showProgress(`${t(reference ? "Computing difference" : "Computing spectrogram")} · 0%`);
        requestSpectrogram({
          frames: frames.buffer,
          referenceFrames: referenceFrames?.buffer,
          mode: reference ? "diff" : "level",
          sampleRate,
          fftSize,
          rows,
          minFrequency: minHz,
          maxFrequency: maxHz,
          scale: settings.scale,
          floorDb: settings.floorDb,
          rangeDb: settings.rangeDb,
          colormap: settings.colormap,
          onProgress: (value) => { if (this.key === key) this.showProgress(`${t(reference ? "Computing difference" : "Computing spectrogram")} · ${Math.round(value * 100)}%`); },
        }).finally(() => {
          this.inFlight = false;
          const deferred = this.deferred;
          this.deferred = null;
          if (deferred) deferred();
        }).then((result) => {
          if (this.key !== key) return;
          const offscreen = document.createElement("canvas");
          offscreen.width = result.columns;
          offscreen.height = result.rows;
          offscreen.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(result.pixels), result.columns, result.rows), 0, 0);
          this.image = offscreen;
          this.result = { columns: result.columns, rows: result.rows, minFrequency: result.minFrequency, maxFrequency: result.maxFrequency, scale: result.scale, mode: result.mode, levels: new Float32Array(result.levels) };
          this.window = { start: startSeconds, end: endSeconds };
          this.showProgress("");
          this.render(samples, sampleRate, startSeconds, endSeconds, options);
        }).catch((error) => {
          if (this.key !== key) return;
          this.showProgress(`${t("Spectrogram unavailable")} · ${error.message}`, true);
        });
      }
      if (this.image) {
        context.imageSmoothingEnabled = true;
        context.drawImage(this.image, 0, 0, width, height);
      }
      if (this.ticks) this.drawFrequencyTicks(context, width, height, ratio, minHz, maxHz, settings.scale);
    }

    drawFrequencyTicks(context, width, height, ratio, low, high, scale) {
      context.font = `${8 * ratio}px ui-monospace, monospace`;
      context.textBaseline = "middle";
      context.textAlign = "right";
      const position = (hz) => scale === "log"
        ? (Math.log(Math.max(hz, 30)) - Math.log(Math.max(low, 30))) / (Math.log(high) - Math.log(Math.max(low, 30)))
        : (hz - low) / (high - low);
      frequencyTicks(low, high, scale).forEach((hz) => {
        const y = (1 - position(hz)) * height;
        if (y < 6 * ratio || y > height - 6 * ratio) return;
        context.strokeStyle = "rgba(255,255,255,.14)";
        context.beginPath(); context.moveTo(width - 5 * ratio, y); context.lineTo(width, y); context.stroke();
        context.fillStyle = "rgba(255,255,255,.55)";
        context.fillText(formatHz(hz), width - 7 * ratio, y);
      });
      context.textAlign = "left";
    }

    /* {seconds, hz, db} under a point given as fractions of the canvas box. */
    valueAt(xFraction, yFraction) {
      const result = this.result;
      if (!result || !this.window) return null;
      const column = clamp(Math.floor(xFraction * result.columns), 0, result.columns - 1);
      const row = clamp(Math.floor(yFraction * result.rows), 0, result.rows - 1);
      const position = 1 - yFraction;
      const low = result.minFrequency;
      const high = result.maxFrequency;
      const hz = result.scale === "log"
        ? Math.exp(Math.log(Math.max(low, 30)) + position * (Math.log(high) - Math.log(Math.max(low, 30))))
        : low + position * (high - low);
      return {
        seconds: this.window.start + xFraction * (this.window.end - this.window.start),
        hz,
        db: result.levels[row * result.columns + column],
        mode: result.mode,
      };
    }
  }

  window.PureSoundAudio = {
    MAX_BOOST_DB,
    SAFE_PEAK_DBFS,
    SPEC_MAX_HZ,
    DEFAULT_SPECTROGRAM,
    clamp,
    linearGain,
    levelDb,
    audioContext,
    resumeAudio,
    claimPlayback,
    releasePlayback,
    formatTime,
    formatHz,
    createOutputChain,
    mixToMono,
    peakOf,
    rmsOf,
    fitCanvas,
    drawRuler,
    drawWave,
    frameLevels,
    encodeWav,
    sliceWav,
    requestSpectrogram,
    SpectrogramView,
  };
})();
