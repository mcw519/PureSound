/* Reusable browser-only audio inspection and audition component.
 * Visuals and playback boost never modify inference inputs or exports. */
(() => {
  "use strict";

  const MAX_BOOST_DB = 48;
  const SAFE_PEAK_DBFS = -1.5;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const linearGain = (db) => 10 ** (db / 20);
  const levelDb = (value) => 20 * Math.log10(Math.max(value, 1e-9));
  const safetyCurve = Float32Array.from({ length: 2048 }, (_, index) => {
    const value = index / 2047 * 2 - 1;
    const magnitude = Math.abs(value);
    const knee = 0.7;
    return Math.sign(value) * (magnitude <= knee ? magnitude : knee + (1 - knee) * Math.tanh((magnitude - knee) / (1 - knee)));
  });
  let sharedContext = null;

  function audioContext() {
    if (!sharedContext) sharedContext = new (window.AudioContext || window.webkitAudioContext)();
    return sharedContext;
  }

  function formatTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    return `${minutes}:${(value - minutes * 60).toFixed(3).padStart(6, "0")}`;
  }

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
      spectrogramWorker.postMessage(message, [message.samples]);
    });
  }

  class AudioPanel {
    constructor(root) {
      this.root = root;
      this.buffer = null;
      this.samples = null;
      this.source = null;
      this.gainNode = null;
      this.limiter = null;
      this.safetyShaper = null;
      this.offset = 0;
      this.startedAt = 0;
      this.playing = false;
      this.boostDb = 0;
      this.peak = 0;
      this.animation = null;
      this.spectrogram = null;
      this.pendingSpectrogram = null;
      this.spectrogramGeneration = 0;
      this.waveColor = root.dataset.waveColor || "#c8f6f9";
      this.renderShell();
      this.bindEvents();
      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(this.stage);
    }

    renderShell() {
      this.root.innerHTML = `
        <div class="audio-tools">
          <div class="audio-transport">
            <button class="audio-tool-button audio-play" type="button" aria-label="Play audio" disabled>▶</button>
            <button class="audio-tool-button audio-stop" type="button" aria-label="Stop audio" disabled>■</button>
            <span class="audio-clock"><span data-clock-current>0:00.000</span><span> / </span><span data-clock-duration>0:00.000</span></span>
          </div>
          <div class="audio-view-switch" role="group" aria-label="Audio view">
            <button type="button" data-view="wave">Wave</button><button class="is-active" type="button" data-view="both">Both</button><button type="button" data-view="spec">Spec</button>
          </div>
          <div class="audio-boost" title="Playback-only gain with a safety limiter; inference and exports are unchanged">
            <label>Boost <input type="range" min="0" max="48" step="0.5" value="0" data-boost /></label>
            <output data-boost-value>0.0 dB</output>
            <button class="audio-auto" type="button" disabled>Auto</button>
            <span class="audio-limiter" data-limiter>LIM</span>
          </div>
        </div>
        <div class="audio-stage is-both">
          <div class="audio-ruler"><canvas aria-hidden="true"></canvas></div>
          <div class="audio-pane audio-wave-pane"><span>WAVEFORM</span><canvas aria-label="Audio waveform"></canvas></div>
          <div class="audio-pane audio-spec-pane"><span>SPECTROGRAM · 0–8 KHZ</span><canvas aria-label="Audio spectrogram"></canvas></div>
          <div class="audio-playhead" aria-hidden="true"></div>
          <div class="audio-empty">Decoding audio…</div>
          <div class="audio-progress" data-audio-progress aria-live="polite"></div>
        </div>
        <p class="audio-audition-note">Boost affects browser audition only — model input and exported WAV stay unchanged.</p>`;
      this.stage = this.root.querySelector(".audio-stage");
      this.ruler = this.root.querySelector(".audio-ruler canvas");
      this.waveform = this.root.querySelector(".audio-wave-pane canvas");
      this.spectrum = this.root.querySelector(".audio-spec-pane canvas");
      this.playhead = this.root.querySelector(".audio-playhead");
      this.playButton = this.root.querySelector(".audio-play");
      this.stopButton = this.root.querySelector(".audio-stop");
      this.currentClock = this.root.querySelector("[data-clock-current]");
      this.durationClock = this.root.querySelector("[data-clock-duration]");
      this.boost = this.root.querySelector("[data-boost]");
      this.boostValue = this.root.querySelector("[data-boost-value]");
      this.autoButton = this.root.querySelector(".audio-auto");
      this.limiterBadge = this.root.querySelector("[data-limiter]");
      this.progressLabel = this.root.querySelector("[data-audio-progress]");
    }

    bindEvents() {
      this.playButton.addEventListener("click", () => this.playing ? this.pause() : this.play());
      this.stopButton.addEventListener("click", () => this.stop());
      this.boost.addEventListener("input", () => this.setBoost(Number(this.boost.value)));
      this.autoButton.addEventListener("click", () => this.setBoost(clamp(SAFE_PEAK_DBFS - levelDb(this.peak), 0, MAX_BOOST_DB)));
      this.root.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => this.setView(button.dataset.view)));
      [this.waveform, this.spectrum, this.ruler].forEach((canvas) => canvas.addEventListener("pointerdown", (event) => this.seek(event)));
    }

    async loadFile(file) {
      return this.loadArrayBuffer(await file.arrayBuffer());
    }

    async loadUrl(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not load audio (${response.status})`);
      return this.loadArrayBuffer(await response.arrayBuffer());
    }

    async loadArrayBuffer(bytes) {
      this.stop();
      this.buffer = null;
      this.samples = null;
      this.spectrogram = null;
      this.pendingSpectrogram = null;
      this.spectrogramGeneration += 1;
      this.root.classList.remove("is-ready");
      this.playButton.disabled = true;
      this.stopButton.disabled = true;
      this.autoButton.disabled = true;
      this.durationClock.textContent = "0:00.000";
      this.progressLabel.classList.remove("is-visible", "is-error");
      this.progressLabel.textContent = "";
      this.draw();
      this.root.classList.add("is-loading");
      let decoded;
      try {
        decoded = await audioContext().decodeAudioData(bytes.slice(0));
      } finally {
        this.root.classList.remove("is-loading");
      }
      this.buffer = decoded;
      this.samples = this.mixToMono(this.buffer);
      this.peak = this.peakAcrossChannels(this.buffer);
      this.offset = 0;
      this.spectrogram = null;
      this.spectrogramGeneration += 1;
      this.root.classList.add("is-ready");
      this.playButton.disabled = false;
      this.stopButton.disabled = false;
      this.autoButton.disabled = false;
      this.durationClock.textContent = formatTime(this.buffer.duration);
      this.updatePosition();
      this.updateLimiter();
      this.draw();
      return this.buffer;
    }

    mixToMono(buffer) {
      if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
      const mono = new Float32Array(buffer.length);
      for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const values = buffer.getChannelData(channel);
        for (let index = 0; index < values.length; index += 1) mono[index] += values[index] / buffer.numberOfChannels;
      }
      return mono;
    }

    peakAcrossChannels(buffer) {
      let peak = 0;
      for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const values = buffer.getChannelData(channel);
        for (let index = 0; index < values.length; index += 1) peak = Math.max(peak, Math.abs(values[index]));
      }
      return peak;
    }

    async play() {
      if (!this.buffer) return;
      const context = audioContext();
      await context.resume();
      if (this.offset >= this.buffer.duration - 0.005) this.offset = 0;
      this.disconnectSource();
      this.source = context.createBufferSource();
      this.source.buffer = this.buffer;
      this.gainNode = context.createGain();
      this.gainNode.gain.value = linearGain(this.boostDb);
      this.limiter = context.createDynamicsCompressor();
      this.limiter.threshold.value = -1;
      this.limiter.knee.value = 0;
      this.limiter.ratio.value = 20;
      this.limiter.attack.value = 0.003;
      this.limiter.release.value = 0.1;
      this.safetyShaper = context.createWaveShaper();
      this.safetyShaper.curve = safetyCurve;
      this.safetyShaper.oversample = "4x";
      this.source.connect(this.gainNode).connect(this.limiter).connect(this.safetyShaper).connect(context.destination);
      this.source.start(0, this.offset);
      this.startedAt = context.currentTime;
      this.source.onended = () => {
        if (!this.playing) return;
        this.offset = this.buffer.duration;
        this.finishPlayback();
      };
      this.playing = true;
      this.playButton.textContent = "❚❚";
      this.playButton.setAttribute("aria-label", "Pause audio");
      this.tick();
    }

    pause() {
      if (!this.playing) return;
      this.offset = this.currentTime();
      this.finishPlayback();
    }

    stop() {
      this.offset = 0;
      this.finishPlayback();
      this.updatePosition();
    }

    finishPlayback() {
      this.playing = false;
      this.disconnectSource();
      cancelAnimationFrame(this.animation);
      this.playButton.textContent = "▶";
      this.playButton.setAttribute("aria-label", "Play audio");
      this.updatePosition();
      this.updateLimiter();
    }

    disconnectSource() {
      if (!this.source) return;
      this.source.onended = null;
      try { this.source.stop(); } catch { /* source may already have ended */ }
      this.source.disconnect();
      this.source = null;
    }

    currentTime() {
      if (!this.playing) return this.offset;
      return clamp(this.offset + audioContext().currentTime - this.startedAt, 0, this.buffer?.duration || 0);
    }

    tick() {
      if (!this.playing) return;
      this.updatePosition();
      this.updateLimiter();
      this.animation = requestAnimationFrame(() => this.tick());
    }

    updatePosition() {
      const current = this.currentTime();
      const fraction = this.buffer?.duration ? current / this.buffer.duration : 0;
      this.currentClock.textContent = formatTime(current);
      this.playhead.style.left = `${fraction * 100}%`;
    }

    seek(event) {
      if (!this.buffer) return;
      const bounds = event.currentTarget.getBoundingClientRect();
      const fraction = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
      const resume = this.playing;
      this.offset = fraction * this.buffer.duration;
      this.finishPlayback();
      if (resume) this.play();
    }

    setBoost(value) {
      this.boostDb = clamp(value, 0, MAX_BOOST_DB);
      this.boost.value = this.boostDb;
      this.boostValue.textContent = `${this.boostDb.toFixed(1)} dB`;
      if (this.gainNode) this.gainNode.gain.setTargetAtTime(linearGain(this.boostDb), audioContext().currentTime, 0.02);
      this.updateLimiter();
    }

    updateLimiter() {
      const reduction = this.limiter && this.playing ? Math.max(0, -this.limiter.reduction) : 0;
      const willLimit = Boolean(this.buffer) && levelDb(this.peak) + this.boostDb > -1;
      this.limiterBadge.classList.toggle("is-active", reduction > 0.4);
      this.limiterBadge.classList.toggle("is-warning", reduction <= 0.4 && willLimit);
      this.limiterBadge.textContent = reduction > 0.4 ? `LIM ${reduction.toFixed(1)}` : "LIM";
    }

    setView(view) {
      this.stage.classList.remove("is-wave", "is-both", "is-spec");
      this.stage.classList.add(`is-${view}`);
      this.root.querySelectorAll("[data-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
      requestAnimationFrame(() => this.draw());
    }

    fitCanvas(canvas) {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.floor(bounds.width * ratio));
      const height = Math.max(1, Math.floor(bounds.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        this.spectrogram = null;
        this.pendingSpectrogram = null;
        this.spectrogramGeneration += 1;
      }
      return { context: canvas.getContext("2d"), width, height, ratio };
    }

    draw() {
      this.drawRuler();
      this.drawWaveform();
      this.drawSpectrogram();
    }

    drawRuler() {
      const { context, width, height, ratio } = this.fitCanvas(this.ruler);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#11122d";
      context.fillRect(0, 0, width, height);
      if (!this.buffer) return;
      const duration = this.buffer.duration;
      const steps = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120];
      const step = steps.find((candidate) => duration / candidate <= Math.max(2, width / (75 * ratio))) || 300;
      context.font = `${9 * ratio}px ui-monospace, monospace`;
      context.textBaseline = "middle";
      for (let time = 0; time <= duration; time += step) {
        const x = time / duration * width;
        context.strokeStyle = "rgba(255,255,255,.18)";
        context.beginPath(); context.moveTo(x, height - 7 * ratio); context.lineTo(x, height); context.stroke();
        context.fillStyle = "rgba(255,255,255,.62)";
        context.fillText(formatTime(time).replace(/\.000$/, ""), x + 4 * ratio, height / 2);
      }
    }

    drawWaveform() {
      const { context, width, height } = this.fitCanvas(this.waveform);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#010120";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "rgba(255,255,255,.1)";
      context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
      if (!this.samples?.length) return;
      const samplesPerPixel = this.samples.length / width;
      context.fillStyle = this.waveColor;
      context.globalAlpha = 0.9;
      for (let pixel = 0; pixel < width; pixel += 1) {
        const start = Math.floor(pixel * samplesPerPixel);
        const end = Math.max(start + 1, Math.min(this.samples.length, Math.ceil((pixel + 1) * samplesPerPixel)));
        let low = 1;
        let high = -1;
        for (let index = start; index < end; index += 1) {
          low = Math.min(low, this.samples[index]);
          high = Math.max(high, this.samples[index]);
        }
        const top = (1 - high) * height / 2;
        const bottom = (1 - low) * height / 2;
        context.fillRect(pixel, top, 1, Math.max(1, bottom - top));
      }
      context.globalAlpha = 1;
    }

    drawSpectrogram() {
      if (this.stage.classList.contains("is-wave")) return;
      const { context, width, height } = this.fitCanvas(this.spectrum);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#010120";
      context.fillRect(0, 0, width, height);
      if (!this.samples?.length || !this.buffer) return;
      if (!this.spectrogram && !this.pendingSpectrogram) {
        const generation = this.spectrogramGeneration;
        const compact = this.compactSamples();
        this.progressLabel.textContent = "Computing spectrogram · 0%";
        this.progressLabel.classList.add("is-visible");
        const request = requestSpectrogram({
          samples: compact.samples.buffer,
          sampleRate: compact.sampleRate,
          width,
          height,
          onProgress: (value) => {
            if (generation === this.spectrogramGeneration) this.progressLabel.textContent = `Computing spectrogram · ${Math.round(value * 100)}%`;
          },
        });
        this.pendingSpectrogram = request;
        request.then((result) => {
          if (this.pendingSpectrogram !== request) return;
          this.pendingSpectrogram = null;
          if (generation !== this.spectrogramGeneration) return;
          const offscreen = document.createElement("canvas");
          offscreen.width = result.columns;
          offscreen.height = result.rows;
          offscreen.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(result.pixels), result.columns, result.rows), 0, 0);
          this.spectrogram = offscreen;
          this.progressLabel.textContent = "";
          this.progressLabel.classList.remove("is-visible");
          this.drawSpectrogram();
        }).catch((error) => {
          if (this.pendingSpectrogram !== request) return;
          this.pendingSpectrogram = null;
          this.progressLabel.textContent = `Spectrogram unavailable · ${error.message}`;
          this.progressLabel.classList.add("is-visible", "is-error");
        });
      }
      if (this.spectrogram) {
        context.imageSmoothingEnabled = true;
        context.drawImage(this.spectrogram, 0, 0, width, height);
      }
    }

    compactSamples() {
      const maximum = 4_000_000;
      if (this.samples.length <= maximum) return { samples: this.samples.slice(), sampleRate: this.buffer.sampleRate };
      const stride = Math.ceil(this.samples.length / maximum);
      const compact = new Float32Array(Math.ceil(this.samples.length / stride));
      for (let index = 0; index < compact.length; index += 1) {
        const start = index * stride;
        const end = Math.min(this.samples.length, start + stride);
        let total = 0;
        for (let sample = start; sample < end; sample += 1) total += this.samples[sample];
        compact[index] = total / (end - start);
      }
      return { samples: compact, sampleRate: this.buffer.sampleRate / stride };
    }
  }

  window.PureSoundAudioPanel = AudioPanel;
})();
