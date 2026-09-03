/* Off-main-thread spectrogram renderer. Messages contain a compact mono view
 * of the decoded audio, so large files never block transport or navigation. */
(() => {
  "use strict";

  const FFT_SIZE = 1024;
  const hann = Float32Array.from({ length: FFT_SIZE }, (_, index) => 0.5 - 0.5 * Math.cos(2 * Math.PI * index / (FFT_SIZE - 1)));
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

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

  function colorRamp(value) {
    const stops = [[1, 1, 32], [45, 42, 112], [32, 126, 138], [239, 44, 193], [252, 244, 220]];
    const position = clamp(value, 0, 1) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(position));
    const fraction = position - index;
    return stops[index].map((channel, offset) => channel + (stops[index + 1][offset] - channel) * fraction);
  }

  self.onmessage = ({ data }) => {
    const { id, samples: rawSamples, sampleRate, width, height } = data;
    const samples = new Float32Array(rawSamples);
    const columns = Math.min(1000, Math.max(160, Math.floor(width)));
    const rows = Math.min(280, Math.max(90, Math.floor(height)));
    const image = new Uint8ClampedArray(columns * rows * 4);
    const real = new Float32Array(FFT_SIZE);
    const imaginary = new Float32Array(FFT_SIZE);
    const maxFrequency = Math.min(8000, sampleRate / 2);
    const maxBin = Math.max(8, Math.min(FFT_SIZE / 2, Math.floor(maxFrequency / (sampleRate / FFT_SIZE))));
    const sampleStep = samples.length / columns;
    for (let x = 0; x < columns; x += 1) {
      const center = Math.floor(x * sampleStep);
      const start = center - FFT_SIZE / 2;
      for (let index = 0; index < FFT_SIZE; index += 1) {
        const sampleIndex = start + index;
        real[index] = (sampleIndex >= 0 && sampleIndex < samples.length ? samples[sampleIndex] : 0) * hann[index];
        imaginary[index] = 0;
      }
      fft(real, imaginary);
      for (let y = 0; y < rows; y += 1) {
        const bin = Math.min(maxBin, Math.floor((1 - y / rows) * maxBin));
        const magnitude = Math.hypot(real[bin], imaginary[bin]) / (FFT_SIZE / 4);
        const intensity = (20 * Math.log10(magnitude + 1e-9) + 92) / 82;
        const [red, green, blue] = colorRamp(intensity);
        const offset = (y * columns + x) * 4;
        image[offset] = red;
        image[offset + 1] = green;
        image[offset + 2] = blue;
        image[offset + 3] = 255;
      }
      if (x % 20 === 0 || x === columns - 1) self.postMessage({ id, type: "progress", value: (x + 1) / columns });
    }
    self.postMessage({ id, type: "complete", columns, rows, pixels: image.buffer }, [image.buffer]);
  };
})();
