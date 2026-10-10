import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { PureSoundStreamingRuntime, concat, fft } from "../dist/runtime.js";
import { decodeWav, encodeWav, resample } from "../dist/audio.js";
const read = (path) => {
  const b = fs.readFileSync(path);
  return new Float32Array(
    b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
  );
};
test("WAV roundtrip, bounds and antialiasing", () => {
  const x = Float32Array.from(
      { length: 1601 },
      (_, i) => Math.sin(i * 0.07) * 0.1,
    ),
    wav = encodeWav(x);
  assert.deepEqual(decodeWav(wav.buffer).samples, x);
  assert.throws(() => decodeWav(wav.buffer.slice(0, 50)), /Truncated/);
  const high = Float32Array.from({ length: 48000 }, (_, i) =>
    Math.sin((2 * Math.PI * 12000 * i) / 48000),
  );
  const down = resample(high, 48000).slice(100, -100);
  assert.ok(
    Math.sqrt(down.reduce((s, x) => s + x * x, 0) / down.length) < 0.01,
  );
});
test("FFT inverse", () => {
  const re = Float64Array.from({ length: 512 }, (_, i) => Math.sin(i)),
    expected = re.slice(),
    im = new Float64Array(512);
  fft(re, im);
  fft(re, im, true);
  assert.ok(re.every((v, i) => Math.abs(v - expected[i]) < 1e-10));
});
const path =
  process.env.PURESOUND_WEB_FIXTURES ?? "/tmp/puresound-web-fixtures";
for (const fixture of JSON.parse(
  fs.readFileSync(path + "/wav-cases.json", "utf8"),
))
  test(`WAV format ${fixture.file}`, () => {
    const bytes = fs.readFileSync(fixture.file),
      decoded = decodeWav(
        bytes.buffer.slice(
          bytes.byteOffset,
          bytes.byteOffset + bytes.byteLength,
        ),
      );
    assert.equal(decoded.sampleRate, fixture.rate);
    assert.deepEqual(decoded.samples, read(fixture.reference));
    assert.equal(
      resample(decoded.samples, fixture.rate).length,
      Math.round((decoded.samples.length * 16000) / fixture.rate),
    );
  });
const cases = JSON.parse(fs.readFileSync(path + "/cases.json", "utf8"));
for (const fixture of cases)
  test(`WASM matches Python: ${fixture.name}`, async () => {
    const stem = fixture.stem,
      manifest = JSON.parse(
        fs
          .readFileSync(fixture.manifest ?? stem + ".json", "utf8")
          .replaceAll("-Infinity", "null"),
      );
    console.log("Running", fixture.name);
    const runtime = await PureSoundStreamingRuntime.create(
      new Uint8Array(fs.readFileSync(stem + ".onnx")),
      manifest,
    );
    const input = read(fixture.input),
      expected = read(fixture.output),
      parts = [];
    assert.equal((await runtime.flush()).length, 0);
    runtime.reset();
    for (let i = 0; i < input.length; i += 317)
      parts.push(await runtime.processSamples(input.subarray(i, i + 317)));
    parts.push(await runtime.flush());
    const actual = concat(...parts);
    assert.equal(actual.length, expected.length);
    let error = 0,
      energy = 0;
    for (let i = 0; i < actual.length; i++) {
      error += (actual[i] - expected[i]) ** 2;
      energy += expected[i] ** 2;
    }
    const nrms = Math.sqrt(error / Math.max(energy, 1e-12));
    console.log(fixture.name, "NRMS", nrms);
    assert.ok(nrms <= 1e-4, `NRMS ${nrms}`);
    assert.equal((await runtime.flush()).length, 0);
    runtime.reset();
    const again = concat(
      await runtime.processSamples(input),
      await runtime.flush(),
    );
    assert.deepEqual(again, actual);
    await runtime.dispose();
    await assert.rejects(runtime.processSamples(input), /disposed|finished/);
  });
