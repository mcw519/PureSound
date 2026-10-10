// Pure logic behind the Pipeline screen, run by test_web_js.py with `node --test`.
const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../../puresound/web/static/pipeline-model.js");

const catalog = {
  groups: [{ id: "source", color: "#111", title: { en: "Source", zh: "來源" } }, { id: "noise", color: "#222", title: { en: "Noise", zh: "噪音" } }, { id: "emit", color: "#333", title: { en: "Pair", zh: "訓練對" } }],
  stages: {
    "source.load": { group: "source", title: { en: "Load", zh: "載入" } },
    "noise.recorded": { group: "noise", title: { en: "Noise", zh: "噪音" } },
    "noise.white": { group: "noise", title: { en: "White", zh: "白" } },
    "noise.floor": { group: "noise", title: { en: "Floor", zh: "底噪" } },
    "chain.adc": { group: "emit", title: { en: "ADC", zh: "ADC" } },
    "row.emit": { group: "emit", title: { en: "Pair", zh: "訓練對" } },
  },
};

const metrics = (esnr, state = "finite") => ({ esnr_db: esnr, esnr_state: state, noisy_rms_dbfs: -20 });
const report = {
  notes: [],
  stages: [
    { id: "source.load", group: "source", block: null, fired: true, recipe: { configured: true, enabled: true, prob: null }, metrics: metrics(null, "identical"), model: { input: { si_sdr_db: 100 }, output: { si_sdr_db: 25 }, level_change_db: -0.5 } },
    { id: "noise.recorded", group: "noise", block: "augmentation_noise_args", fired: true, recipe: { configured: true, enabled: true, prob: 0.9 }, metrics: metrics(5.0), model: { input: { si_sdr_db: 5.1 }, output: { si_sdr_db: 12.0 }, level_change_db: -3 } },
    { id: "noise.white", group: "noise", block: "augmentation_noise_args", fired: false, recipe: { configured: true, enabled: true, prob: 0.25 }, metrics: null, model: null },
    { id: "noise.floor", group: "noise", block: "augmentation_noise_args.absolute_floor", fired: false, recipe: { configured: false, enabled: false, prob: null }, metrics: null, model: null },
    { id: "chain.adc", group: "emit", block: null, fired: false, recipe: { configured: true, enabled: true, prob: null }, metrics: null, model: null },
    { id: "row.emit", group: "emit", block: null, fired: true, recipe: { configured: true, enabled: true, prob: null }, metrics: metrics(2.0), model: { input: { si_sdr_db: 2.0 }, output: { si_sdr_db: 9.5 }, level_change_db: -4, same_input_as: null } },
  ],
};

test("rows carry catalog titles, groups and a state that says why a stage did not act", () => {
  const rows = M.stageRows(report, catalog, "zh");
  assert.deepEqual(rows.map((row) => row.state), ["fired", "fired", "skipped", "absent", "skipped", "fired"]);
  assert.equal(rows[0].title, "載入");
  assert.equal(rows[1].groupInfo.color, "#222");
  assert.match(M.stateLabel(rows[2]), /did not fire.*p = 0\.25/);
  assert.match(M.stateLabel(rows[4]), /not needed/);
  const withNote = M.stageRows({ ...report, notes: [{ subject: "augmentation_noise", reason: "no noise clip was chosen" }], stages: report.stages.map((stage) => ({ ...stage, recipe: stage.id === "noise.white" ? { configured: false, enabled: false, prob: null } : stage.recipe })) }, catalog, "en");
  assert.equal(withNote[2].state, "off-here");
  assert.equal(M.stateLabel({ state: "zero" }), "probability 0 for these rows");
});

test("the effective-SNR series skips silent stages and measures each change from the last finite value", () => {
  const series = M.esnrSeries(M.stageRows(report, catalog, "en"));
  assert.deepEqual(series.map((point) => [point.id, point.value, point.delta, point.state]), [
    ["source.load", null, null, "identical"],
    ["noise.recorded", 5.0, null, "finite"],
    ["row.emit", 2.0, -3.0, "finite"],
  ]);
  assert.equal(M.biggestDrop(series), "row.emit");
});

test("the model series pairs input and output per fired stage", () => {
  const series = M.modelSeries(M.stageRows(report, catalog, "en"), "si_sdr_db");
  assert.deepEqual(series.map((point) => [point.id, point.input, point.output, point.change]), [
    ["source.load", 100, 25, -75],
    ["noise.recorded", 5.1, 12.0, 6.9],
    ["row.emit", 2.0, 9.5, 7.5],
  ]);
  assert.deepEqual(M.modelSeries(M.stageRows(report, catalog, "en"), "level_change_db").map((point) => point.output), [-0.5, -3, -4]);
});

test("chart domains ignore missing values, clamp outliers and keep a readable span", () => {
  assert.deepEqual(M.chartDomain([null, 5, 2]), [-4, 11]);
  assert.deepEqual(M.chartDomain([100, 0]), [-3, 60]);
  assert.deepEqual(M.chartDomain([]), [-30, 60]);
  assert.deepEqual(M.chartDomain([0.5, 0.52], { floor: 0, ceil: 1, pad: 0.05, minSpan: 0.2 }), [0.41, 0.61]);
});

test("numbers read as the page shows them", () => {
  assert.equal(M.formatNumber(null), "—");
  assert.equal(M.formatNumber(-3.14159, { unit: " dB" }), "−3.1 dB");
  assert.equal(M.formatNumber(100, { unit: " dB", cap: 100 }), "≥ 100 dB");
  assert.equal(M.formatSigned(2.25), "+2.3");
  assert.equal(M.formatSigned(null), "—");
});

test("stage parameters flatten to readable rows without paths, empty values or internals", () => {
  const rows = M.flattenParams({
    plan: { class: "RowPlan", target_absent: false, speed_factor: 1.05, session: null },
    room_scene: { room_id: "r1", wav_path: "/tmp/x.wav", rt60: 0.4123, near: "<list>", split: null, used_channels: [1], union_member_name: "samples" },
    snr_db: 3.21987,
    codec: null,
  });
  assert.deepEqual(rows, [["plan", "RowPlan"], ["plan.target_absent", "no"], ["plan.speed_factor", "1.05"], ["room_scene.room_id", "r1"], ["room_scene.rt60", "0.412"], ["snr_db", "3.22"]]);
});

test("a room projects to a top view with the used sources marked", () => {
  const view = M.topView({ room_dim: [5, 4, 3], receiver: [2, 2, 1], sources: [{ label: "near_0", position: [2.5, 2, 1.2], distance_m: 0.54, roles: ["foreground"] }], obstacles: [{ footprint: [[1, 1], [1.5, 1], [1.5, 1.5]], z_max: 1.2 }] });
  assert.deepEqual([view.width, view.depth, view.receiver.x, view.receiver.y], [5, 4, 2, 2]);
  assert.deepEqual(view.sources[0], { x: 2.5, y: 2, z: 1.2, label: "near_0", roles: ["foreground"], distance: 0.54 });
  assert.equal(view.obstacles[0].points.length, 3);
  assert.equal(M.topView(null), null);
});

test("the previous fired stage is the reference a stage is compared with", () => {
  const rows = M.stageRows(report, catalog, "en");
  assert.equal(M.previousFired(rows, "row.emit").id, "noise.recorded");
  assert.equal(M.previousFired(rows, "source.load"), null);
});

test("a stage the inspector switched off says why, in the note's own words", () => {
  const rows = M.stageRows({ ...report, notes: [{ subject: "augmentation_noise", reason: "no noise clip was chosen" }], stages: report.stages.map((stage) => ({ ...stage, recipe: stage.id === "noise.white" ? { configured: false, enabled: false, prob: null } : stage.recipe })) }, catalog, "en");
  assert.equal(rows[2].state, "off-here");
  assert.equal(M.stateLabel(rows[2]), "off in the inspector: no noise clip was chosen");
});

test("roles are named as the page draws them: noise coloured through the room is noise", () => {
  const rirs = [
    { stage: "foreground.channel", role: "foreground", metadata: { label: "near_0" } },
    { stage: "interferers.sample", role: "interferer", metadata: { label: "far_0" } },
    { stage: "noise.recorded", role: "interferer", metadata: { label: "far_0" } },
    { stage: "noise.recorded", role: "interferer", metadata: { label: "far_2" } },
  ];
  assert.deepEqual(rirs.map(M.displayRole), ["foreground", "interferer", "noise", "noise"]);
  const room = M.displayRoom({ room: { room_dim: [5, 4, 3], receiver: [2, 2, 1], sources: [{ label: "near_0", roles: ["foreground"] }, { label: "far_0", roles: ["interferer"] }, { label: "far_2", roles: ["interferer"] }, { label: "far_1", roles: [] }] }, rirs });
  assert.deepEqual(room.sources.map((source) => source.roles), [["foreground"], ["interferer", "noise"], ["noise"], []]);
  assert.deepEqual(M.stageRoles({ rirs }, "noise.recorded"), ["noise"]);
  assert.equal(M.displayRoom({ room: null, rirs }), null);
});

test("a float WAV keeps samples above full scale", () => {
  const bytes = M.encodeFloatWav(new Float32Array([0, 1.5, -2.25]), 16000);
  const view = new DataView(bytes);
  const text = (offset) => String.fromCharCode(...new Uint8Array(bytes, offset, 4));
  assert.equal(text(0), "RIFF");
  assert.equal(text(8), "WAVE");
  assert.equal(view.getUint16(20, true), 3);   // IEEE float
  assert.equal(view.getUint16(34, true), 32);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(text(36), "data");
  assert.deepEqual([view.getFloat32(44, true), view.getFloat32(48, true), view.getFloat32(52, true)], [0, 1.5, -2.25]);
});
