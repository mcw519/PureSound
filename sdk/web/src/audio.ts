/** Strict WAV I/O and antialiasing resampling; no browser decoder dependency. */
export function decodeWav(buffer: ArrayBuffer): {
  samples: Float32Array;
  sampleRate: number;
} {
  const v = new DataView(buffer),
    text = (at: number, n: number) =>
      String.fromCharCode(...new Uint8Array(buffer, at, n));
  if (buffer.byteLength < 44 || text(0, 4) !== "RIFF" || text(8, 4) !== "WAVE")
    throw Error("Expected a RIFF WAV file");
  if (v.getUint32(4, true) + 8 > buffer.byteLength)
    throw Error("Truncated WAV");
  let format = 0,
    channels = 0,
    rate = 0,
    bits = 0,
    align = 0,
    data = -1,
    size = 0;
  for (let pos = 12; pos + 8 <= buffer.byteLength; ) {
    const tag = text(pos, 4),
      len = v.getUint32(pos + 4, true),
      start = pos + 8;
    if (start + len > buffer.byteLength) throw Error("Truncated WAV chunk");
    if (tag === "fmt ") {
      if (len < 16) throw Error("Invalid WAV format");
      format = v.getUint16(start, true);
      channels = v.getUint16(start + 2, true);
      rate = v.getUint32(start + 4, true);
      align = v.getUint16(start + 12, true);
      bits = v.getUint16(start + 14, true);
    }
    if (tag === "data") {
      data = start;
      size = len;
    }
    pos = start + len + (len & 1);
  }
  if (
    data < 0 ||
    ![1, 2].includes(channels) ||
    ![16000, 44100, 48000].includes(rate) ||
    !((format === 1 && bits === 16) || (format === 3 && bits === 32)) ||
    align !== (channels * bits) / 8 ||
    size % align
  )
    throw Error("Use PCM16 or Float32 WAV, mono/stereo, at 16/44.1/48 kHz");
  const count = size / align;
  if (count < 1 || count > rate * 120)
    throw Error("Audio must contain at most two minutes");
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) {
      const at = data + i * align + (ch * bits) / 8;
      sum +=
        format === 1 ? v.getInt16(at, true) / 32768 : v.getFloat32(at, true);
    }
    samples[i] = sum / channels;
    if (!Number.isFinite(samples[i])) throw Error("Nonfinite audio");
  }
  return { samples, sampleRate: rate };
}
export function resample(
  input: Float32Array,
  rate: number,
  target = 16000,
): Float32Array {
  if (rate === target) return input.slice();
  const ratio = rate / target,
    cutoff = Math.min(1, target / rate) * 0.94,
    half = 32;
  const out = new Float32Array(Math.round(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const center = i * ratio,
      base = Math.floor(center);
    let sum = 0;
    for (let tap = -half + 1; tap <= half; tap++) {
      const index = base + tap;
      if (index < 0 || index >= input.length) continue;
      const distance = center - index,
        x = distance * cutoff;
      const sinc =
        Math.abs(x) < 1e-6 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      sum +=
        input[index] *
        cutoff *
        sinc *
        (0.5 + 0.5 * Math.cos((Math.PI * distance) / (half + 1)));
    }
    out[i] = sum;
  }
  return out;
}
export function encodeWav(samples: Float32Array, sr = 16000): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 4),
    v = new DataView(bytes.buffer);
  const text = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) bytes[at + i] = s.charCodeAt(i);
  };
  text(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 4, true);
  text(8, "WAVE");
  text(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 3, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 4, true);
  v.setUint16(32, 4, true);
  v.setUint16(34, 32, true);
  text(36, "data");
  v.setUint32(40, samples.length * 4, true);
  for (let i = 0; i < samples.length; i++)
    v.setFloat32(44 + 4 * i, samples[i], true);
  return bytes;
}
