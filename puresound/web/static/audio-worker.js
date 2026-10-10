/* Off-main-thread spectrogram renderer.
 *
 * The page sends one analysis frame per image column (`fftSize` samples
 * centred on that column's time), so the work and the transfer are bounded by
 * the image size, never by the recording's length, and a ten-minute file is
 * analysed at its real bandwidth.
 *
 * Two modes.  The default renders one signal's magnitude; ``mode: "diff"``
 * renders a target against a reference, bin by bin, in dB -- where the model
 * took energy out (cool) or put it in (warm).  Both return the dB value behind
 * every pixel so the page can read it back on hover.  The frequency axis is a
 * window [minFrequency, maxFrequency], linear or logarithmic. */
(() => {
  "use strict";

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  // Below this in both signals a bin is silence, and a ratio of two silences
  // is noise: the diff view leaves it dark rather than coloured.
  const DIFF_FLOOR_DB = -78;
  const DIFF_RANGE_DB = 30;
  const LOG_MIN_HZ = 30;
  const windows = new Map();

  const COLORMAPS = {
    puresound: [[1, 1, 32], [45, 42, 112], [32, 126, 138], [239, 44, 193], [252, 244, 220]],
    magma: [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]],
    viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    gray: [[0, 0, 0], [255, 255, 255]],
  };

  function hann(size) {
    if (!windows.has(size)) windows.set(size, Float32Array.from({ length: size }, (_, index) => 0.5 - 0.5 * Math.cos(2 * Math.PI * index / (size - 1))));
    return windows.get(size);
  }

  function fft(real, imaginary) {
    const size = real.length;
    for (let index = 1, swap = 0; index < size; index += 1) {
      let bit = size >> 1;
      for (; swap & bit; bit >>= 1) swap ^= bit;
      swap ^= bit;
      if (index < swap) {
        [real[index], real[swap]] = [real[swap], real[index]];
        [imaginary[index], imaginary[swap]] = [imaginary[swap], imaginary[index]];
      }
    }
    for (let length = 2; length <= size; length <<= 1) {
      const angle = -2 * Math.PI / length;
      const stepReal = Math.cos(angle);
      const stepImaginary = Math.sin(angle);
      for (let start = 0; start < size; start += length) {
        let phaseReal = 1;
        let phaseImaginary = 0;
        for (let offset = 0; offset < length / 2; offset += 1) {
          const upperReal = real[start + offset];
          const upperImaginary = imaginary[start + offset];
          const lower = start + offset + length / 2;
          const lowerReal = real[lower] * phaseReal - imaginary[lower] * phaseImaginary;
          const lowerImaginary = real[lower] * phaseImaginary + imaginary[lower] * phaseReal;
          real[start + offset] = upperReal + lowerReal;
          imaginary[start + offset] = upperImaginary + lowerImaginary;
          real[lower] = upperReal - lowerReal;
          imaginary[lower] = upperImaginary - lowerImaginary;
          const nextReal = phaseReal * stepReal - phaseImaginary * stepImaginary;
          phaseImaginary = phaseReal * stepImaginary + phaseImaginary * stepReal;
          phaseReal = nextReal;
        }
      }
    }
  }

  function ramp(stops, value) {
    const position = clamp(value, 0, 1) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(position));
    const fraction = position - index;
    return stops[index].map((channel, offset) => channel + (stops[index + 1][offset] - channel) * fraction);
  }

  /* Negative (energy removed) runs dark -> cyan, positive (added) dark -> orange. */
  function divergingRamp(value) {
    const base = [14, 14, 44];
    const cool = [80, 200, 255];
    const warm = [255, 120, 40];
    const amount = clamp(Math.abs(value), 0, 1) ** 0.8;
    const target = value < 0 ? cool : warm;
    return base.map((channel, offset) => channel + (target[offset] - channel) * amount);
  }

  /* Fractional FFT bin for each image row, top row = highest frequency. */
  function rowBins(rows, sampleRate, fftSize, low, high, scale) {
    const bins = new Float32Array(rows);
    const binHz = sampleRate / fftSize;
    const logLow = Math.log(Math.max(low, LOG_MIN_HZ));
    const logHigh = Math.log(Math.max(high, LOG_MIN_HZ + 1));
    for (let y = 0; y < rows; y += 1) {
      const position = 1 - (y + 0.5) / rows;
      const hz = scale === "log" ? Math.exp(logLow + position * (logHigh - logLow)) : low + position * (high - low);
      bins[y] = clamp(hz / binHz, 0, fftSize / 2);
    }
    return bins;
  }

  /* dB magnitude per row of one frame, interpolated between FFT bins. */
  function frameLevels(frames, column, fftSize, bins, real, imaginary, magnitudes, out) {
    const window = hann(fftSize);
    const offset = column * fftSize;
    for (let index = 0; index < fftSize; index += 1) {
      real[index] = frames[offset + index] * window[index];
      imaginary[index] = 0;
    }
    fft(real, imaginary);
    for (let bin = 0; bin <= fftSize / 2; bin += 1) magnitudes[bin] = Math.hypot(real[bin], imaginary[bin]) / (fftSize / 4);
    for (let y = 0; y < bins.length; y += 1) {
      const position = bins[y];
      const lower = Math.floor(position);
      const upper = Math.min(fftSize / 2, lower + 1);
      const fraction = position - lower;
      const magnitude = magnitudes[lower] * (1 - fraction) + magnitudes[upper] * fraction;
      out[y] = 20 * Math.log10(magnitude + 1e-9);
    }
  }

  self.onmessage = ({ data }) => {
    const { id, sampleRate } = data;
    const fftSize = data.fftSize || 1024;
    const frames = new Float32Array(data.frames);
    const referenceFrames = data.mode === "diff" ? new Float32Array(data.referenceFrames) : null;
    const columns = Math.round(frames.length / fftSize);
    const rows = Math.round(clamp(data.rows ?? 256, 64, 1024));
    const high = Math.min(data.maxFrequency || 8000, sampleRate / 2);
    const low = clamp(data.minFrequency || 0, 0, high - 1);
    const scale = data.scale === "log" ? "log" : "linear";
    const floorDb = data.floorDb ?? -92;
    const rangeDb = Math.max(10, data.rangeDb ?? 82);
    const stops = COLORMAPS[data.colormap] || COLORMAPS.puresound;
    const bins = rowBins(rows, sampleRate, fftSize, low, high, scale);
    const image = new Uint8ClampedArray(columns * rows * 4);
    const levels = new Float32Array(columns * rows);
    const real = new Float32Array(fftSize);
    const imaginary = new Float32Array(fftSize);
    const magnitudes = new Float32Array(fftSize / 2 + 1);
    const target = new Float32Array(rows);
    const base = new Float32Array(rows);
    for (let x = 0; x < columns; x += 1) {
      frameLevels(frames, x, fftSize, bins, real, imaginary, magnitudes, target);
      if (referenceFrames) frameLevels(referenceFrames, x, fftSize, bins, real, imaginary, magnitudes, base);
      for (let y = 0; y < rows; y += 1) {
        let value;
        let color;
        if (referenceFrames) {
          const silent = Math.max(target[y], base[y]) < DIFF_FLOOR_DB;
          value = silent ? 0 : clamp(target[y], DIFF_FLOOR_DB - 20, 20) - clamp(base[y], DIFF_FLOOR_DB - 20, 20);
          color = silent ? [8, 8, 30] : divergingRamp(value / DIFF_RANGE_DB);
        } else {
          value = target[y];
          color = ramp(stops, (value - floorDb) / rangeDb);
        }
        levels[y * columns + x] = value;
        const offset = (y * columns + x) * 4;
        image[offset] = color[0];
        image[offset + 1] = color[1];
        image[offset + 2] = color[2];
        image[offset + 3] = 255;
      }
      if (x % 40 === 0 || x === columns - 1) self.postMessage({ id, type: "progress", value: (x + 1) / columns });
    }
    self.postMessage({ id, type: "complete", columns, rows, minFrequency: low, maxFrequency: high, scale, mode: referenceFrames ? "diff" : "level", pixels: image.buffer, levels: levels.buffer }, [image.buffer, levels.buffer]);
  };
})();
