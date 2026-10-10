import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web";

export interface Manifest {
  processor: string;
  sample_rate: number;
  fft_length: number;
  win_length: number;
  hop_length: number;
  freq_bins: number;
  output_names: string[];
  state_input_names: string[];
  state_output_names: string[];
  state_shapes: Record<string, number[]>;
  streaming_delay_frames?: number;
  onset_guard?: Record<string, number | null>;
  recommended_inference?: {
    dry_blend?: number;
    spec_floor?: number;
    mix_phase?: boolean;
  };
}
type Backend = typeof import("onnxruntime-web");
type Session = Pick<InferenceSession, "run" | "release">;
const f32 = Math.fround;
export function concat(...arrays: Float32Array[]): Float32Array {
  const out = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = ((inverse ? 2 : -2) * Math.PI) / len;
    for (let start = 0; start < n; start += len) {
      let wr = 1,
        wi = 0;
      const cr = Math.cos(angle),
        ci = Math.sin(angle);
      for (let k = 0; k < len / 2; k++) {
        const a = start + k,
          b = a + len / 2,
          br = re[b] * wr - im[b] * wi,
          bi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - br;
        im[b] = im[a] - bi;
        re[a] += br;
        im[a] += bi;
        const nw = wr * cr - wi * ci;
        wi = wi * cr + wr * ci;
        wr = nw;
      }
    }
  }
  if (inverse)
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
}
function pyround(x: number): number {
  const floor = Math.floor(x);
  return x - floor === 0.5
    ? floor % 2 === 0
      ? floor
      : floor + 1
    : Math.round(x);
}
class Guard {
  k: Record<string, number>;
  hop: number;
  floor: number = Infinity;
  i = 0;
  last = -1e9;
  held = 0;
  run = 0;
  sil = 0;
  confirmed = false;
  started = false;
  gain = 1;
  previous: Float32Array | null = null;
  init: number[] = [];
  win: [number, number][] = [];
  nInit: number;
  nFloor: number;
  nHold: number;
  nMin: number;
  nArm: number;
  nForget: number;
  rise: number;
  up: number;
  down: number;
  constructor(knobs: Record<string, number | null>, hop: number, sr: number) {
    this.k = {
      t_arm_s: 1,
      t_forget_s: 5,
      tau_up_s: 0.05,
      tau_dn_s: 2,
      margin_db: 8,
      floor_win_s: 2,
      floor_rise_db_per_s: 3,
      init_s: 0.2,
      hangover_s: 0.2,
      min_run_s: 0.1,
      snap: 0.001,
    };
    for (const key of Object.keys(this.k))
      if (key in knobs)
        this.k[key] = knobs[key] === null ? Infinity : Number(knobs[key]);
    const k = this.k,
      fps = sr / hop;
    this.hop = hop;
    this.nInit = Math.max(1, pyround(k.init_s * fps));
    this.nFloor = Math.max(1, pyround(k.floor_win_s * fps));
    this.nHold = Math.floor(k.hangover_s * fps) + 1;
    this.nMin = Math.max(1, Math.floor(k.min_run_s * fps));
    this.nArm = Math.max(1, pyround(k.t_arm_s * fps));
    this.nForget = Math.max(1, pyround(k.t_forget_s * fps));
    this.rise = k.floor_rise_db_per_s / fps;
    this.up = 1 - Math.exp(-1 / (k.tau_up_s * fps));
    this.down = 1 - Math.exp(-1 / (k.tau_dn_s * fps));
    if (this.nInit + 2 > this.nMin + this.nArm)
      throw Error("Onset guard priming exceeds arming window");
  }
  advance(i: number, e: number): number {
    while (this.win.length && this.win[this.win.length - 1][1] >= e)
      this.win.pop();
    this.win.push([i, e]);
    if (this.win[0][0] <= i - this.nFloor) this.win.shift();
    const c = this.win[0][1];
    this.floor = c < this.floor ? c : Math.min(c, this.floor + this.rise);
    if (e > this.floor + this.k.margin_db) this.last = i;
    const held = i - this.last <= this.nHold - 1;
    this.held = held ? this.held + 1 : 0;
    if (this.held >= this.nMin) {
      this.run++;
      this.sil = 0;
      if (!this.confirmed && this.run >= this.nArm) this.confirmed = true;
    } else {
      this.run = 0;
      this.sil++;
      if (this.confirmed && this.sil >= this.nForget) this.confirmed = false;
    }
    const target = this.confirmed ? 0 : 1;
    if (!this.started) {
      this.gain = target;
      this.started = true;
    } else if (Math.abs(this.gain - target) <= this.k.snap) this.gain = target;
    else {
      const a = target > this.gain ? this.up : this.down;
      this.gain = target + (this.gain - target) * (1 - a);
      if (Math.abs(this.gain - target) <= this.k.snap) this.gain = target;
    }
    this.i = i + 1;
    return this.gain;
  }
  step(chunk: Float32Array): number {
    if (!this.previous) {
      this.previous = chunk;
      return 1;
    }
    let power = 0;
    for (const x of this.previous) power += x * x;
    for (const x of chunk) power += x * x;
    this.previous = chunk;
    const e = 10 * Math.log10(power / (2 * this.hop) + 1e-12),
      i = this.i;
    if (i < this.nInit - 1) {
      this.init.push(e);
      this.i++;
      return 1;
    }
    if (i === this.nInit - 1) {
      this.init.push(e);
      this.floor = Math.min(...this.init);
      for (let j = 0; j < this.init.length; j++) this.advance(j, this.init[j]);
      this.init = [];
      return this.gain;
    }
    return this.advance(i, e);
  }
}

export class PureSoundStreamingRuntime {
  readonly manifest: Manifest;
  readonly sampleRate: number;
  private backend: Backend;
  private session: Session;
  private window: Float32Array;
  private states: Record<string, OrtTensor> = {};
  private input: Float32Array = new Float32Array(0);
  private ola: Float32Array;
  private norm: Float32Array;
  private dry: Float32Array = new Float32Array(0);
  private dryStart = 0;
  private emitted = 0;
  private guard: Guard | null = null;
  private guardBuffer: Float32Array = new Float32Array(0);
  private guardGains: number[] = [];
  private gainStart = 0;
  private hops = 0;
  private received = 0;
  private ended = false;
  private disposed = false;
  private busy = false;
  private constructor(backend: Backend, session: Session, m: Manifest) {
    this.backend = backend;
    this.session = session;
    this.manifest = m;
    this.sampleRate = m.sample_rate;
    if (
      m.processor !== "stft_frame_ort" ||
      m.sample_rate !== 16000 ||
      m.fft_length < 2 ||
      m.fft_length & (m.fft_length - 1) ||
      m.win_length > m.fft_length ||
      m.hop_length < 1 ||
      m.win_length < m.hop_length ||
      m.freq_bins !== m.fft_length / 2 + 1 ||
      m.state_input_names.length !== m.state_output_names.length
    )
      throw Error("Unsupported streaming manifest");
    const p = m.recommended_inference ?? {},
      blend = p.dry_blend ?? 1;
    if (
      p.spec_floor ||
      p.mix_phase ||
      !Number.isFinite(blend) ||
      blend <= 0 ||
      blend > 1
    )
      throw Error("Unsupported postprocessing");
    this.window = Float32Array.from(
      { length: m.win_length },
      (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / m.win_length),
    );
    this.ola = new Float32Array(m.win_length);
    this.norm = new Float32Array(m.win_length);
    this.reset();
  }
  static async create(
    model: string | Uint8Array,
    manifest: Manifest,
    options: { backend?: Backend; session?: Session } = {},
  ): Promise<PureSoundStreamingRuntime> {
    const backend = options.backend ?? (await import("onnxruntime-web"));
    const session =
      options.session ??
      (await (typeof model === "string"
        ? backend.InferenceSession.create(model, {
            executionProviders: ["wasm"],
          })
        : backend.InferenceSession.create(model, {
            executionProviders: ["wasm"],
          })));
    try {
      return new PureSoundStreamingRuntime(backend, session, manifest);
    } catch (error) {
      await session.release();
      throw error;
    }
  }
  reset(): void {
    if (this.busy || this.disposed) throw Error("Runtime is busy or disposed");
    for (const name of this.manifest.state_input_names) {
      const dims = this.manifest.state_shapes[name];
      if (!dims || dims.some((v) => !Number.isInteger(v) || v < 1))
        throw Error("Invalid state shape");
      this.states[name] = new this.backend.Tensor(
        "float32",
        new Float32Array(dims.reduce((a, b) => a * b, 1)),
        dims,
      );
    }
    this.input = new Float32Array(0);
    this.ola.fill(0);
    this.norm.fill(0);
    this.dry = new Float32Array(0);
    this.dryStart = 0;
    this.emitted = 0;
    this.guard = this.manifest.onset_guard
      ? new Guard(
          this.manifest.onset_guard,
          this.manifest.hop_length,
          this.sampleRate,
        )
      : null;
    if (
      this.guard &&
      (this.manifest.streaming_delay_frames ?? 0) +
        Math.floor(this.manifest.win_length / this.manifest.hop_length) -
        2 <
        0
    )
      throw Error("Insufficient guard lookahead");
    this.guardBuffer = new Float32Array(0);
    this.guardGains = [];
    this.gainStart = 0;
    this.hops = 0;
    this.ended = false;
    this.received = 0;
  }
  private feedGuard(samples: Float32Array): void {
    if (!this.guard) return;
    this.guardBuffer = concat(this.guardBuffer, samples);
    const hop = this.manifest.hop_length;
    while (this.guardBuffer.length >= hop) {
      const gain = this.guard.step(this.guardBuffer.slice(0, hop));
      this.guardBuffer = this.guardBuffer.slice(hop);
      if (this.hops++) this.guardGains.push(gain);
    }
  }
  private blend(output: Float32Array): Float32Array {
    const m = this.manifest,
      blend = m.recommended_inference?.dry_blend ?? 1;
    const start = this.emitted - (m.streaming_delay_frames ?? 0) * m.hop_length;
    this.emitted += output.length;
    if (blend === 1 && !this.guard) return output;
    const lo = Math.max(start, this.dryStart),
      hi = Math.min(start + output.length, this.dryStart + this.dry.length);
    for (let i = lo; i < hi; i++) {
      const at = i - start,
        raw = this.dry[i - this.dryStart];
      if (blend < 1)
        output[at] = Math.max(
          -1,
          Math.min(
            1,
            f32(f32(f32(blend) * output[at]) + f32(f32(1 - blend) * raw)),
          ),
        );
      if (this.guard) {
        const frame = Math.floor(i / m.hop_length) - this.gainStart;
        if (frame < 0 || frame >= this.guardGains.length)
          throw Error("Missing onset gain");
        const g = f32(this.guardGains[frame]);
        output[at] = f32(f32(g * raw) + f32(f32(1 - g) * output[at]));
      }
    }
    for (let i = 0; i < output.length; i++)
      output[i] = Math.max(-1, Math.min(1, output[i]));
    const drop = Math.max(
      0,
      Math.min(
        this.dry.length,
        this.emitted -
          (m.streaming_delay_frames ?? 0) * m.hop_length -
          this.dryStart,
      ),
    );
    this.dry = this.dry.slice(drop);
    this.dryStart += drop;
    const old = Math.floor(
      (this.emitted - (m.streaming_delay_frames ?? 0) * m.hop_length) /
        m.hop_length,
    );
    const gains = Math.max(
      0,
      Math.min(this.guardGains.length, old - this.gainStart),
    );
    this.guardGains.splice(0, gains);
    this.gainStart += gains;
    return output;
  }
  private async frame(
    frame: Float32Array,
    keep = this.manifest.hop_length,
  ): Promise<Float32Array> {
    const m = this.manifest,
      n = m.fft_length,
      re = new Float64Array(n),
      im = new Float64Array(n);
    for (let i = 0; i < m.win_length; i++)
      re[i] = f32(frame[i] * this.window[i]);
    fft(re, im);
    const data = new Float32Array(m.freq_bins * 2);
    for (let i = 0; i < m.freq_bins; i++) {
      data[2 * i] = re[i];
      data[2 * i + 1] = im[i];
    }
    const outputs = await this.session.run(
      {
        noisy_frame: new this.backend.Tensor("float32", data, [
          1,
          m.freq_bins,
          2,
        ]),
        ...this.states,
      },
      m.output_names,
    );
    for (let i = 0; i < m.state_input_names.length; i++)
      this.states[m.state_input_names[i]] = outputs[m.state_output_names[i]];
    const enhanced = outputs[m.output_names[0]].data as Float32Array;
    for (let i = 0; i < m.freq_bins; i++) {
      re[i] = enhanced[2 * i];
      im[i] = enhanced[2 * i + 1];
    }
    for (let i = m.freq_bins; i < n; i++) {
      re[i] = re[n - i];
      im[i] = -im[n - i];
    }
    im[0] = 0;
    im[n / 2] = 0;
    fft(re, im, true);
    for (let i = 0; i < m.win_length; i++) {
      this.ola[i] = f32(this.ola[i] + f32(f32(re[i]) * this.window[i]));
      this.norm[i] = f32(this.norm[i] + f32(this.window[i] * this.window[i]));
    }
    const out = Float32Array.from(
      this.ola.subarray(0, keep),
      (v, i) => v / Math.max(this.norm[i], 1e-8),
    );
    this.ola.copyWithin(0, m.hop_length);
    this.ola.fill(0, m.win_length - m.hop_length);
    this.norm.copyWithin(0, m.hop_length);
    this.norm.fill(0, m.win_length - m.hop_length);
    return this.blend(out);
  }
  async processSamples(samples: Float32Array): Promise<Float32Array> {
    if (this.busy || this.ended || this.disposed)
      throw Error(
        "Reset before processing a finished stream; calls must be sequential",
      );
    if (samples.some((v) => !Number.isFinite(v)))
      throw Error("Audio contains nonfinite values");
    this.busy = true;
    try {
      if (
        (this.manifest.recommended_inference?.dry_blend ?? 1) < 1 ||
        this.guard
      )
        this.dry = concat(this.dry, samples);
      this.feedGuard(samples);
      this.received += samples.length;
      this.input = concat(this.input, samples);
      const out: Float32Array[] = [];
      while (this.input.length >= this.manifest.win_length) {
        out.push(
          await this.frame(this.input.subarray(0, this.manifest.win_length)),
        );
        this.input = this.input.slice(this.manifest.hop_length);
      }
      return concat(...out);
    } finally {
      this.busy = false;
    }
  }
  async flush(): Promise<Float32Array> {
    if (this.disposed || this.busy) throw Error("Runtime is busy or disposed");
    if (this.ended) return new Float32Array(0);
    this.busy = true;
    this.ended = true;
    try {
      const m = this.manifest,
        out: Float32Array[] = [];
      if (this.guard) {
        const pad =
          (m.hop_length - (this.guardBuffer.length % m.hop_length)) %
          m.hop_length;
        this.feedGuard(new Float32Array(pad + m.hop_length));
      }
      // Owed: every output sample that carries input, i.e. received plus the
      // graph's look-ahead. Zero frames drain that look-ahead; a hop is complete
      // once the frame starting in it has run. The overlap-add remainder maps
      // to padding and holds only part of its window sum, so it is dropped.
      const end = this.received
        ? this.received + (m.streaming_delay_frames ?? 0) * m.hop_length
        : 0;
      while (this.emitted < end) {
        const frame = new Float32Array(m.win_length);
        frame.set(this.input.subarray(0, m.win_length));
        out.push(
          await this.frame(frame, Math.min(m.hop_length, end - this.emitted)),
        );
        this.input = this.input.slice(m.hop_length);
      }
      this.input = new Float32Array(0);
      this.ola.fill(0);
      this.norm.fill(0);
      return concat(...out);
    } finally {
      this.busy = false;
    }
  }
  async dispose(): Promise<void> {
    if (this.busy) throw Error("Runtime is busy");
    if (!this.disposed) {
      this.disposed = true;
      this.dry = new Float32Array(0);
      this.input = new Float32Array(0);
      this.states = {};
      await this.session.release();
    }
  }
}
