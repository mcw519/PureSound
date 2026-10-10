/* Audio-thread half of microphone capture and live monitoring.
 *
 * The page's AudioContext runs at the device rate; models run at 16 kHz.  The
 * processor resamples the microphone down to the model rate and posts it in
 * fixed chunks, and resamples what comes back up into a jitter buffer that
 * feeds the output.  The monitor can play the model's output, the raw
 * microphone (the live A/B), or nothing.  Browser echo cancellation, noise
 * suppression and gain control are switched off by the caller, so the model
 * hears the capture chain as it is. */

/* Windowed-sinc resampler for a stream: any ratio, state kept across calls. */
class StreamResampler {
  constructor(inputRate, outputRate, halfTaps = 16) {
    this.step = inputRate / outputRate;
    this.cutoff = Math.min(1, outputRate / inputRate) * 0.94;
    this.half = halfTaps;
    this.buffer = new Float32Array(0);
    this.position = 0;
  }

  kernel(distance) {
    const x = distance * this.cutoff;
    const sinc = Math.abs(x) < 1e-6 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const window = 0.5 + 0.5 * Math.cos(Math.PI * distance / (this.half + 1));
    return this.cutoff * sinc * window;
  }

  push(input) {
    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer);
    merged.set(input, this.buffer.length);
    this.buffer = merged;
    const out = [];
    while (Math.floor(this.position) + this.half < this.buffer.length) {
      const center = this.position;
      const base = Math.floor(center);
      let total = 0;
      for (let tap = -this.half + 1; tap <= this.half; tap += 1) {
        const index = base + tap;
        if (index < 0 || index >= this.buffer.length) continue;
        total += this.buffer[index] * this.kernel(center - index);
      }
      out.push(total);
      this.position += this.step;
    }
    const drop = Math.max(0, Math.floor(this.position) - this.half);
    if (drop > 0) {
      this.buffer = this.buffer.slice(drop);
      this.position -= drop;
    }
    return Float32Array.from(out);
  }
}

class PureSoundIoProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { modelRate = 16000, chunkSamples = 320, jitterSeconds = 0.04 } = options.processorOptions || {};
    this.down = new StreamResampler(sampleRate, modelRate);
    this.up = new StreamResampler(modelRate, sampleRate);
    this.chunkSamples = chunkSamples;
    this.pending = new Float32Array(0);
    this.monitor = "off";
    // Playback jitter buffer at the device rate: playback starts once it holds
    // `jitterSeconds`, and underruns (no model audio in time) play silence.
    this.ring = new Float32Array(Math.ceil(sampleRate * 4));
    this.readIndex = 0;
    this.writeIndex = 0;
    this.buffered = 0;
    this.primed = false;
    this.jitterSamples = Math.ceil(sampleRate * jitterSeconds);
    this.underruns = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "monitor") this.monitor = data.mode;
      else if (data.type === "play") this.enqueue(this.up.push(data.samples));
      else if (data.type === "reset") { this.readIndex = 0; this.writeIndex = 0; this.buffered = 0; this.primed = false; }
    };
  }

  enqueue(samples) {
    for (let index = 0; index < samples.length; index += 1) {
      if (this.buffered >= this.ring.length) break;
      this.ring[this.writeIndex] = samples[index];
      this.writeIndex = (this.writeIndex + 1) % this.ring.length;
      this.buffered += 1;
    }
    if (this.buffered >= this.jitterSamples) this.primed = true;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (input) {
      const resampled = this.down.push(input);
      const merged = new Float32Array(this.pending.length + resampled.length);
      merged.set(this.pending);
      merged.set(resampled, this.pending.length);
      let offset = 0;
      while (merged.length - offset >= this.chunkSamples) {
        const chunk = merged.slice(offset, offset + this.chunkSamples);
        let energy = 0;
        for (let index = 0; index < chunk.length; index += 1) energy += chunk[index] * chunk[index];
        this.port.postMessage({ type: "chunk", samples: chunk, rms: Math.sqrt(energy / chunk.length), buffered: this.buffered / sampleRate, underruns: this.underruns }, [chunk.buffer]);
        offset += this.chunkSamples;
      }
      this.pending = merged.slice(offset);
    }
    if (output) {
      if (this.monitor === "raw" && input) {
        output.set(input);
      } else if (this.monitor === "model" && this.primed) {
        for (let index = 0; index < output.length; index += 1) {
          if (this.buffered > 0) {
            output[index] = this.ring[this.readIndex];
            this.readIndex = (this.readIndex + 1) % this.ring.length;
            this.buffered -= 1;
          } else {
            output[index] = 0;
            this.underruns += 1;
            this.primed = false;
          }
        }
      } else {
        output.fill(0);
        // Not listening to the model: keep the buffer from growing stale.
        if (this.monitor !== "model") { this.readIndex = this.writeIndex; this.buffered = 0; this.primed = false; }
      }
    }
    return true;
  }
}

registerProcessor("puresound-io", PureSoundIoProcessor);
