// Microphone capture (capture.js) against a stand-in Web Audio and microphone,
// run by test_web_js.py.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const SOURCE = fs.readFileSync(path.join(__dirname, "../../puresound/web/static/capture.js"), "utf8");

/* A capture module over a fake microphone; `failWorklet` makes the audio
 * worklet module fail to load after the microphone was granted. */
function load({ failWorklet = false } = {}) {
  const probe = { tracksStopped: 0, contextsClosed: 0, node: null };
  class FakeNode {
    constructor() { this.port = { postMessage() {} }; probe.node = this; }
    connect() {}
    disconnect() {}
  }
  class FakeContext {
    constructor() {
      this.state = "running";
      this.destination = {};
      this.audioWorklet = { addModule: async () => { if (failWorklet) throw new Error("module failed"); } };
    }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    async resume() {}
    async close() { this.state = "closed"; probe.contextsClosed += 1; }
  }
  const track = { label: "mic", getSettings: () => ({}), stop() { probe.tracksStopped += 1; } };
  const window = { isSecureContext: true, setTimeout, clearTimeout, setInterval, clearInterval };
  const context = {
    window,
    AudioContext: FakeContext,
    AudioWorkletNode: FakeNode,
    File: class { constructor(parts, name) { this.parts = parts; this.name = name; } },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [track], getAudioTracks: () => [track] }) } },
    performance, ArrayBuffer, DataView, Float32Array, Date, Math, Promise, Array, Set, Map, JSON, WebSocket: { OPEN: 1 },
  };
  vm.runInNewContext(SOURCE, context);
  return { capture: window.PureSoundCapture, probe };
}

// The bound counts chunks, so a one-sample chunk keeps the file small.
const chunk = () => ({ data: { type: "chunk", samples: new Float32Array(1).fill(0.1), rms: 0.1 } });

test("a recording that reaches the length bound is still handed over when it is stopped", async () => {
  const { capture, probe } = load();
  const recorder = new capture.Recorder();
  await recorder.start();
  // Ten minutes of 20 ms chunks: the recorder ends itself.
  for (let index = 0; index < 30000; index += 1) probe.node.port.onmessage(chunk());
  const file = await recorder.stop();
  assert.ok(file, "stopping after the bound returned no file");
  assert.match(file.name, /^recording-.*\.wav$/);
  assert.equal(await recorder.stop(), file);
});

test("stopping a recording that captured nothing gives no file", async () => {
  const { capture } = load();
  const recorder = new capture.Recorder();
  await recorder.start();
  assert.equal(await recorder.stop(), null);
});

test("a capture that fails after the microphone was granted releases the microphone", async () => {
  const { capture, probe } = load({ failWorklet: true });
  const recorder = new capture.Recorder();
  await assert.rejects(recorder.start(), /module failed/);
  assert.equal(probe.tracksStopped, 1);
  assert.equal(probe.contextsClosed, 1);
});
