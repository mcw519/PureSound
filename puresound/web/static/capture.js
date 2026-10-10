/* Microphone capture for the playground: recording a clip to process, and a
 * live session that streams the microphone through a model and back.
 * Capture happens at the model rate (16 kHz) in an AudioWorklet, with the
 * browser's own echo cancellation, noise suppression and gain control off:
 * the model has to hear the capture chain as it really is. */
(() => {
  "use strict";

  const MODEL_RATE = 16000;
  const CHUNK_SAMPLES = 320; // 20 ms at the model rate
  const MAX_RECORD_SECONDS = 10 * 60;
  const t = (key, vars) => (window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);

  function captureAvailability() {
    if (!window.isSecureContext) return t("The microphone needs a secure page: open the playground on localhost or over HTTPS.");
    if (!navigator.mediaDevices?.getUserMedia) return t("This browser does not expose a microphone.");
    if (typeof AudioWorkletNode === "undefined") return t("This browser has no AudioWorklet support.");
    return "";
  }

  /* Mono float samples -> 16-bit PCM WAV bytes. */
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
      const value = Math.max(-1, Math.min(1, samples[index]));
      view.setInt16(44 + index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    }
    return buffer;
  }

  function concatenate(chunks) {
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    chunks.forEach((chunk) => { out.set(chunk, offset); offset += chunk.length; });
    return out;
  }

  class MicCapture {
    async start({ onChunk = () => {}, jitterSeconds = 0.04, deviceId = "" } = {}) {
      const problem = captureAvailability();
      if (problem) throw new Error(problem);
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
      });
      try {
        this.context = new AudioContext({ latencyHint: "interactive" });
        await this.context.audioWorklet.addModule("/capture-worklet.js");
        this.source = this.context.createMediaStreamSource(this.stream);
        this.node = new AudioWorkletNode(this.context, "puresound-io", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
          processorOptions: { modelRate: MODEL_RATE, chunkSamples: CHUNK_SAMPLES, jitterSeconds },
        });
        this.node.port.onmessage = ({ data }) => { if (data.type === "chunk") onChunk(data); };
        this.source.connect(this.node);
        this.node.connect(this.context.destination);
        await this.context.resume();
      } catch (error) {
        // The microphone was already granted: release it rather than leave it open.
        await this.stop().catch(() => {});
        throw error;
      }
      const track = this.stream.getAudioTracks()[0];
      this.device = track?.label || "microphone";
      this.settings = track?.getSettings?.() || {};
      return this;
    }

    setMonitor(mode) {
      this.node?.port.postMessage({ type: "monitor", mode });
    }

    play(samples) {
      this.node?.port.postMessage({ type: "play", samples }, [samples.buffer]);
    }

    /* Output-side delay the browser reports for this device, in seconds. */
    deviceLatency() {
      return (this.context?.baseLatency || 0) + (this.context?.outputLatency || 0);
    }

    async stop() {
      try { this.source?.disconnect(); this.node?.disconnect(); } catch { /* already detached */ }
      this.stream?.getTracks().forEach((track) => track.stop());
      if (this.context && this.context.state !== "closed") await this.context.close();
      this.node = null;
    }
  }

  /* Record the microphone to a WAV file at the model rate. */
  class Recorder {
    constructor({ onLevel = () => {}, onTime = () => {} } = {}) {
      this.onLevel = onLevel;
      this.onTime = onTime;
      this.chunks = [];
    }

    async start({ deviceId = "" } = {}) {
      this.chunks = [];
      this.finished = null;
      this.capture = new MicCapture();
      await this.capture.start({
        deviceId,
        onChunk: ({ samples, rms }) => {
          this.chunks.push(samples);
          const seconds = this.chunks.length * CHUNK_SAMPLES / MODEL_RATE;
          this.onLevel(rms);
          this.onTime(seconds);
          if (seconds >= MAX_RECORD_SECONDS) this.stop();
        },
      });
      return this.capture.device;
    }

    /* The recording as a WAV file, or null when nothing was captured.  The
     * length bound ends the recording by itself; stopping again then gives
     * the same file. */
    stop() {
      this.finished ??= this.finish();
      return this.finished;
    }

    async finish() {
      if (!this.capture) return null;
      const capture = this.capture;
      this.capture = null;
      await capture.stop();
      const samples = concatenate(this.chunks);
      if (!samples.length) return null;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      return new File([encodeWav(samples, MODEL_RATE)], `recording-${stamp}.wav`, { type: "audio/wav" });
    }
  }

  /* The model end of a live session on the server: a WebSocket to
   * /api/live (the protocol is in docs/usage/web.md).  A link opens with the
   * session's callbacks and returns the stream's contract; send() is answered
   * through onReply(sequence, processingMs, samples), in order; stop() ends
   * the stream once every reply is in.  PureSoundDevice.liveLink is the other
   * link, for a model running in this browser. */
  function serverLink({ modelId, variant, provider, parameters }) {
    let socket = null;
    let ready = null;
    let stats = null;
    let jobId = null;
    return {
      runner: "server",
      get rtf() { return stats?.rtf ?? null; },
      open({ onReply, onEnded }) {
        const scheme = location.protocol === "https:" ? "wss" : "ws";
        socket = new WebSocket(`${scheme}://${location.host}/api/live`);
        socket.binaryType = "arraybuffer";
        return new Promise((resolve, reject) => {
          socket.onopen = () => socket.send(JSON.stringify({ model_id: modelId, variant, provider, parameters }));
          socket.onerror = () => reject(new Error(t("The live connection failed.")));
          socket.onclose = (event) => { if (!ready) reject(new Error(event.reason || t("The server closed the live connection."))); else onEnded(); };
          socket.onmessage = (event) => {
            if (typeof event.data !== "string") {
              const view = new DataView(event.data);
              onReply(view.getUint32(0, true), view.getFloat32(4, true), new Float32Array(event.data.slice(8)));
              return;
            }
            const message = JSON.parse(event.data);
            if (message.type === "ready") { ready = message; resolve(message); }
            else if (message.type === "error") { reject(new Error(message.message)); onEnded(message.message); }
            else if (message.type === "stats") stats = message;
            else if (message.type === "stopped") { stats = { ...stats, ...message }; jobId = message.job_id || null; }
          };
        });
      },
      send(sequence, samples) {
        if (socket?.readyState !== WebSocket.OPEN) return;
        const message = new ArrayBuffer(4 + samples.length * 4);
        new DataView(message).setUint32(0, sequence, true);
        new Float32Array(message, 4).set(samples);
        socket.send(message);
      },
      async stop() {
        if (socket?.readyState !== WebSocket.OPEN) return { jobId };
        socket.send(JSON.stringify({ type: "stop" }));
        // Wait for "stopped" (it names the run kept in the history), briefly.
        for (let waited = 0; waited < 1500 && !jobId; waited += 50) await new Promise((resolve) => window.setTimeout(resolve, 50));
        return { jobId };
      },
      close() {
        if (socket && socket.readyState <= WebSocket.OPEN) socket.close(1000);
      },
    };
  }

  /* The microphone through a model and back, over a link to wherever the
   * model runs.  Everything sent and received is kept (up to a bound) so the
   * session can be put on the comparison deck afterwards. */
  class LiveSession {
    constructor({ onStatus = () => {}, onStats = () => {}, onLevel = () => {} } = {}) {
      this.onStatus = onStatus;
      this.onStats = onStats;
      this.onLevel = onLevel;
    }

    async start({ link, monitor = "off", deviceId = "" }) {
      this.link = link;
      this.sent = [];
      this.received = [];
      this.sequence = 0;
      this.sentAt = new Map();
      this.roundTrips = [];
      this.processing = [];
      this.ready = null;
      this.stopped = false;
      this.error = null;
      this.capture = new MicCapture();
      this.ready = await link.open({
        onReply: (sequence, processingMs, samples) => this.receive(sequence, processingMs, samples),
        onEnded: (error) => this.end(error),
      });
      await this.capture.start({ onChunk: (data) => this.send(data), deviceId });
      this.capture.setMonitor(monitor);
      this.startedAt = performance.now();
      this.timer = window.setInterval(() => this.reportStats(), 500);
      return this.ready;
    }

    send({ samples, rms, buffered, underruns }) {
      this.onLevel(rms);
      this.jitter = { buffered, underruns };
      if (!this.ready || this.stopped) return;
      if (this.sent.length * CHUNK_SAMPLES >= MAX_RECORD_SECONDS * MODEL_RATE) { this.end(); return; }
      const sequence = this.sequence;
      this.sequence += 1;
      this.sent.push(samples);
      this.sentAt.set(sequence, performance.now());
      this.link.send(sequence, samples);
    }

    /* A reply; a null sequence is the stream's flush tail. */
    receive(sequence, processingMs, samples) {
      const sentAt = this.sentAt.get(sequence);
      if (sentAt != null) {
        this.roundTrips.push(performance.now() - sentAt);
        this.sentAt.delete(sequence);
        this.processing.push(processingMs);
      }
      if (this.roundTrips.length > 250) this.roundTrips.shift();
      if (this.processing.length > 250) this.processing.shift();
      this.received.push(samples);
      if (samples.length) this.capture?.play(samples.slice());
    }

    reportStats() {
      const median = (values) => {
        if (!values.length) return null;
        const sorted = [...values].sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      };
      const hopMs = this.ready ? 1000 * this.ready.hop_length / this.ready.sample_rate : 10;
      const windowMs = this.ready ? 1000 * this.ready.win_length / this.ready.sample_rate : 32;
      this.onStats({
        runner: this.link.runner,
        seconds: this.sent.length * CHUNK_SAMPLES / MODEL_RATE,
        roundTripMs: median(this.roundTrips),
        processingMs: median(this.processing),
        chunkMs: 1000 * CHUNK_SAMPLES / MODEL_RATE,
        lookAheadMs: this.ready?.latency_ms ?? 0,
        windowMs,
        hopMs,
        jitterMs: 1000 * (this.jitter?.buffered || 0),
        underruns: this.jitter?.underruns || 0,
        deviceMs: 1000 * (this.capture?.deviceLatency() || 0),
        rtf: this.link.rtf,
        backlog: this.sentAt.size,
      });
    }

    /* The model end or the length bound ended the session; the owner stops it. */
    end(error = null) {
      if (this.stopped) return;
      if (error) this.error = error;
      this.onStatus({ error, ended: true });
    }

    async stop() {
      if (this.stopped) return this.result();
      this.stopped = true;
      window.clearInterval(this.timer);
      try {
        const { jobId } = (await this.link?.stop()) || {};
        this.jobId = jobId || null;
      } catch (error) {
        this.error = this.error || error.message;
      }
      this.link?.close();
      await this.capture?.stop();
      return this.result();
    }

    /* The session on one time axis: the output stream carries the model's
     * look-ahead in front, as the offline runtime's does, so dropping it lines
     * each output sample up with the input sample it came from. */
    result() {
      const input = concatenate(this.sent || []);
      const output = concatenate(this.received || []);
      const latency = this.ready?.latency_samples || 0;
      const aligned = output.slice(Math.min(latency, output.length));
      const length = Math.min(input.length, aligned.length);
      const removed = new Float32Array(length);
      for (let index = 0; index < length; index += 1) removed[index] = input[index] - aligned[index];
      return {
        sampleRate: MODEL_RATE,
        seconds: length / MODEL_RATE,
        input: input.slice(0, length),
        output: aligned.slice(0, length),
        removed,
        ready: this.ready,
        runner: this.link?.runner,
        error: this.error,
        jobId: this.jobId || null,
      };
    }
  }

  /* Microphones the browser will name; names appear only once the page has
   * been given the microphone at least once. */
  async function listMicrophones() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === "audioinput" && device.deviceId && device.deviceId !== "default");
  }

  window.PureSoundCapture = { MODEL_RATE, CHUNK_SAMPLES, captureAvailability, encodeWav, listMicrophones, Recorder, LiveSession, serverLink };
})();
