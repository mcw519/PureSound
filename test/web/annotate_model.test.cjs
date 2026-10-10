// The Annotate screen's parsing and export, pinned byte for byte against the
// standalone annotator it replaced (fixtures recorded from that tool's own
// functions).  Run by test_web_js.py with `node --test`.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const M = require("../../puresound/web/static/annotate-model.js");

const FIXTURES = path.join(__dirname, "fixtures", "annotate");
const reference = JSON.parse(fs.readFileSync(path.join(FIXTURES, "reference.json"), "utf8"));
const fixture = (name) => fs.readFileSync(path.join(FIXTURES, `${name}.txt`), "utf8");

const spans = (list) => list.map(([a, b, tag, label], i) => ({ id: i + 1, a, b, tag, label }));
const SETS = {
  single: {
    folderName: "",
    current: 0,
    files: [{ name: "near-and-far.wav", path: "near-and-far.wav", sr: 16000, duration: 6.0213, spans: spans([
      [0.41234, 1.9, "near / keep", ""],
      [2.0005, 3.25, "far / suppress", "door \"slam\""],
      [3.3, 4.0, "double-talk", "a, b, c"],
      [4.1, 4.45, "exclude", "cough"],
      [4.5, 5.99987, "laughter, loud", ""],
    ]) }],
  },
  folder: {
    folderName: "session-a",
    current: 2,
    files: [
      { name: "clip_01.wav", path: "session-a/clip_01.wav", sr: 48000, duration: 2.5, spans: spans([[0.1, 0.9, "near / keep", "hello"], [1.0, 2.2, "far / suppress", ""]]) },
      { name: "clip_02.flac", path: "session-a/sub/clip_02.flac", sr: 0, duration: 0, spans: [] },
      { name: "take 3.wav", path: "session-a/take 3.wav", sr: 16000, duration: 9.87654, spans: spans([[0, 1.23456, "Keep this", "=cmd"], [2, 3, "suppressed far", "tab\there"], [5.5, 6.25, "noise", "no role"]]) },
    ],
  },
  empty: { folderName: "", current: 0, files: [{ name: "silence.wav", path: "silence.wav", sr: 16000, duration: 1, spans: [] }] },
};
const FORMATS = ["json", "csv", "windows", "audacity"];

for (const [setName, set] of Object.entries(SETS)) {
  for (const format of FORMATS) {
    test(`${setName} · ${format}: the export matches the old tool byte for byte`, () => {
      const options = { current: set.files[set.current], folderName: set.folderName };
      assert.equal(M.buildExport(format, set.files, options), fixture(`${setName}.${format}`));
      assert.equal(M.exportName(format, { files: set.files, ...options }), reference[`${setName}.${format}`].name);
    });
    test(`${setName} · ${format}: reading the export back gives what the old tool read`, () => {
      const text = fixture(`${setName}.${format}`);
      assert.deepEqual(M.parseSpans(text, reference[`${setName}.${format}`].name, "near / keep"), reference[`${setName}.${format}`].parsed);
    });
  }
}

for (const key of ["audacity", "csvNoFile", "csvWithFile", "bareArray", "windows", "junk", "emptyText", "badJson"]) {
  test(`reads ${key} the way the old tool did`, () => {
    const inputs = {
      audacity: ["0.5\t1.25\tspeech", "2\t3\t", "x\ty\tbad"].join("\n"),
      csvNoFile: ['start_s,end_s,duration_s,tag,label', '1,2,1,"far / suppress","quote ""x"""', '3,4,1,,'].join("\n"),
      csvWithFile: ['file,start_s,end_s,tag', '"a b.wav",0,1,near', 'c.wav,1,2,'].join("\r\n"),
      bareArray: JSON.stringify([{ start_s: 1, end_s: 2, tag: "t" }, { a: 3, b: 4 }, { start: 5, end: 6, label: "l" }]),
      windows: JSON.stringify({ _comment: "x", "clip.wav": { role: "r", keep: [[0, 1]], suppress: [[2, 3]] }, bad: [1], other: { keep: [] } }),
      junk: "hello world",
      emptyText: "   ",
      badJson: "{nope",
    };
    assert.deepEqual(M.parseSpans(inputs[key], "input.wav", "near / keep"), reference[`read.${key}`].parsed);
  });
}

test("CSV fields split on commas outside quotes", () => {
  assert.deepEqual(['a,"b,c",d', '"x ""y""",,z', ""].map(M.splitCsv), reference.splitCsv);
});

test("a tag's role in windows.json", () => {
  assert.equal(M.windowsRole("near / keep"), "keep");
  assert.equal(M.windowsRole("Double-talk"), "keep");
  assert.equal(M.windowsRole("far / suppress"), "suppress");
  assert.equal(M.windowsRole("exclude"), null);
});

test("keyboard shortcuts stand aside while typing, and belong to Annotate on its screen", () => {
  const target = (tagName, extra = {}) => ({ tagName, isContentEditable: false, ...extra });
  for (const tag of ["INPUT", "TEXTAREA", "SELECT"]) assert.equal(M.shortcutTarget({ target: target(tag) }, "annotate"), null);
  assert.equal(M.shortcutTarget({ target: target("DIV", { isContentEditable: true }) }, "annotate"), null);
  assert.equal(M.shortcutTarget({ target: target("CANVAS") }, "annotate"), "annotate");
  assert.equal(M.shortcutTarget({ target: target("BUTTON") }, "annotate"), "annotate");
  assert.equal(M.shortcutTarget({ target: target("CANVAS") }, "playground"), "deck");
  assert.equal(M.shortcutTarget({ target: null }, "annotate"), "annotate");
});
