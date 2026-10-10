// The comparison deck's audio helpers (audio-view.js) that need no browser,
// run by test_web_js.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const window = {};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../puresound/web/static/audio-view.js"), "utf8"), { window, Float32Array, Math, Number, String, Map, Promise });
const A = window.PureSoundAudio;

test("a time that rounds up to the next minute reads as that minute", () => {
  assert.equal(A.formatTime(0), "0:00.000");
  assert.equal(A.formatTime(12.3456), "0:12.346");
  assert.equal(A.formatTime(59.9996), "1:00.000");
  assert.equal(A.formatTime(61.001), "1:01.001");
  assert.equal(A.formatTime(-3), "0:00.000");
  assert.equal(A.formatTime(NaN), "0:00.000");
});

test("a slice of a 16-bit WAV is a WAV of that span at the file's own rate", () => {
  const rate = 16000;
  const samples = Float32Array.from({ length: rate * 2 }, (_, index) => Math.sin(index / 40));
  const slice = A.sliceWav(A.encodeWav(samples, rate), 0.5, 1.25);
  assert.equal(slice.sampleRate, rate);
  const view = new DataView(slice.bytes);
  assert.equal(view.getUint32(40, true), 0.75 * rate * 2);
  assert.equal(view.getUint32(4, true), 36 + 0.75 * rate * 2);
  assert.equal(slice.bytes.byteLength, 44 + 0.75 * rate * 2);
  assert.equal(A.sliceWav(new ArrayBuffer(10), 0, 1), null);
});
