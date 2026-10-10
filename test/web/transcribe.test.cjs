// The word-check panel's job handling (transcribe.js), run by test_web_js.py:
// polling a job the server no longer knows, and a result for audio the deck
// no longer holds.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const SOURCE = fs.readFileSync(path.join(__dirname, "../../puresound/web/static/transcribe.js"), "utf8");

function load(fetch) {
  const window = { setTimeout: (callback) => setTimeout(callback, 0) };
  vm.runInNewContext(SOURCE, { window, fetch, Set, Math, JSON, Promise, Error, String, Array, Object, Number });
  return window.PureSoundTranscribe;
}

const reply = (status, body) => ({ ok: status < 400, status, json: async () => body });

/* A panel with only what run() and poll() touch. */
function panel(Panel, fetch) {
  const element = (extra = {}) => ({ textContent: "", disabled: false, hidden: false, classList: { toggle() {} }, ...extra });
  const self = Object.create(Panel.prototype);
  Object.assign(self, {
    generation: 0,
    backend: { value: "whisper" },
    whisperModel: { value: "base" },
    language: { value: "" },
    reference: { value: "" },
    referenceTrack: { value: "input" },
    trackList: { querySelectorAll: () => [{ value: "input" }] },
    runButton: element(),
    cancelButton: element(),
    status: element(),
    deck: { track: () => ({ id: "input", label: "Input", url: "/api/runs/r/input" }) },
    rendered: 0,
    renderResult() { this.rendered += 1; },
  });
  return self;
}

test("polling stops with the server's message when the job is gone", async () => {
  let polls = 0;
  const Panel = load(async () => {
    if ((polls += 1) > 20) throw new Error("kept polling a job the server does not know");
    return reply(404, { error: { message: "'job not found: j'" } });
  });
  await assert.rejects(panel(Panel).poll("j"), { message: "'job not found: j'" });
  assert.equal(polls, 1);
});

test("a result for audio the deck has since replaced is not shown", async () => {
  const jobs = { POST: reply(200, { job_id: "j" }), GET: reply(200, { status: "succeeded", result: { elapsed_seconds: 1, model: "m", tracks: [] } }) };
  const Panel = load(async (url, options = {}) => jobs[options.method || "GET"]);
  const word = panel(Panel);
  const running = word.run();
  word.generation += 1; // what refresh() does when the deck's tracks change
  await running;
  assert.equal(word.rendered, 0);
  assert.equal(word.status.textContent, "");
});

test("a result for the audio that was sent is shown", async () => {
  const jobs = { POST: reply(200, { job_id: "j" }), GET: reply(200, { status: "succeeded", result: { elapsed_seconds: 1, model: "m", tracks: [] } }) };
  const Panel = load(async (url, options = {}) => jobs[options.method || "GET"]);
  const word = panel(Panel);
  await word.run();
  assert.equal(word.rendered, 1);
});
