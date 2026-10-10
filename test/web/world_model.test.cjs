// Pure logic behind the Acoustic world screen, run by test_web_js.py with `node --test`.
const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("../../puresound/web/static/world-model.js");

const LIMITS = { speed_m_s: 2, min_mic_distance_m: 0.2, sources: 8 };

const transducer = (id, position, directivity) => ({ transducer_id: id, kind: "source", pose: { position_m: position, orientation_ypr_deg: [0, 0, 0] }, directivity_id: directivity, power_db_spl_at_1m: 65 });

function scene() {
  return {
    duration_s: 10,
    near_radius_m: 1,
    room: {
      dimensions_m: [6, 5, 3],
      receivers: [{ pose: { position_m: [1, 2.5, 1.4] } }],
      sources: [transducer("a", [4.5, 2.5, 1.5], "speech_human"), transducer("b", [4.8, 3.8, 1.5], "speech_human"), transducer("n", [3.5, 1, 0.8], "omnidirectional")],
      surfaces: [{ vertices_m: [[6, 5, 3], [0, 0, 0]] }],
      objects: [{ footprint: [[2.6, 2.1], [3, 2.1], [3, 2.9], [2.6, 2.9]], z_min: 0, z_max: 2 }],
    },
    sources: [
      { source_id: "a", asset_id: "a", role: "target", gain_db: 0, start_s: 0, repeat: true, keyframes: [
        { time_s: 0, position_m: [4.5, 2.5, 1.5], yaw_deg: 170, pitch_deg: 0 },
        { time_s: 5, position_m: [1.7, 2.5, 1.5], yaw_deg: -170, pitch_deg: 0 },
      ] },
      { source_id: "b", asset_id: "b", role: "interferer", gain_db: -6, start_s: 0, repeat: true, keyframes: [{ time_s: 0, position_m: [4.8, 3.8, 1.5], yaw_deg: 180, pitch_deg: 0 }] },
      { source_id: "n", asset_id: "n", role: "noise", gain_db: -12, start_s: 9.95, repeat: true, keyframes: [{ time_s: 0, position_m: [3.5, 1, 0.8], yaw_deg: 0, pitch_deg: 0 }] },
    ],
  };
}

test("poses interpolate linearly and turn the short way round", () => {
  const source = scene().sources[0];
  const half = M.pose(source, 2.5);
  assert.deepEqual(half.position_m, [3.1, 2.5, 1.5]);
  assert.equal(half.yaw_deg, 180);
  assert.deepEqual(M.pose(source, 9).position_m, [1.7, 2.5, 1.5]);
  assert.equal(M.angleDelta(350, 10), 20);
  assert.equal(M.angleDelta(10, 190), 180);
});

test("near weight follows the renderer's raised cosine", () => {
  assert.equal(M.nearWeight(0.9, 1), 1);
  assert.ok(Math.abs(M.nearWeight(1, 1) - 0.5) < 1e-12);
  assert.equal(M.nearWeight(1.1, 1), 0);
  const state = M.sourceState(scene(), scene().sources[2], 0);
  assert.equal(state.near_weight, 0); // noise never counts as a talker
});

test("roles separate the target, the other talker and noise", () => {
  const s = scene();
  assert.deepEqual(s.sources.map((source) => M.sourceRole(s, source)), ["target", "interferer", "noise"]);
});

test("keyframe issues point at the keyframe that breaks a rule", () => {
  const s = scene();
  assert.deepEqual(M.keyframeIssues(s, LIMITS), []);
  s.sources[0].keyframes[1].time_s = 1; // 2.8 m in 1 s
  s.sources[0].keyframes.push({ time_s: 0.5, position_m: [1.1, 2.5, 1.4], yaw_deg: 0, pitch_deg: 0 });
  s.sources[1].keyframes[0].position_m = [2.8, 2.5, 1];
  s.sources[2].keyframes.push({ time_s: 2, position_m: [3.6, 1, 0.8], yaw_deg: 0, pitch_deg: 0 }); // noise may move
  const kinds = M.keyframeIssues(s, LIMITS).map((issue) => `${issue.source}:${issue.key}:${issue.kind}`);
  assert.deepEqual(kinds.sort(), ["0:1:speed", "0:2:mic", "0:2:order", "1:0:obstacle"]);
  const speed = M.keyframeIssues(s, LIMITS).find((issue) => issue.kind === "speed");
  assert.ok(Math.abs(speed.speed - 2.8) < 1e-9);
});

test("duration changes keep each keyframe's share of the scene", () => {
  const s = M.withDuration(scene(), 5);
  assert.equal(s.duration_s, 5);
  assert.equal(s.sources[0].keyframes[1].time_s, 2.5);
  assert.equal(s.sources[2].start_s, 4.9); // start stays inside the scene
  assert.equal(scene().sources[0].keyframes[1].time_s, 5); // input untouched
});

test("room size changes move the shoebox surfaces with it", () => {
  const s = M.withRoomSize(scene(), 0, 9);
  assert.equal(s.room.dimensions_m[0], 9);
  assert.deepEqual(s.room.surfaces[0].vertices_m[0], [9, 5, 3]);
});

test("a new keyframe sits on the path, so it changes nothing until moved", () => {
  const before = scene();
  const { scene: after, index } = M.withKeyframe(before, 0, 2.5);
  assert.equal(index, 1);
  assert.equal(after.sources[0].keyframes.length, 3);
  for (const time of [1, 2.5, 4]) assert.deepEqual(M.pose(after.sources[0], time).position_m.map((v) => +v.toFixed(6)), M.pose(before.sources[0], time).position_m.map((v) => +v.toFixed(6)));
  const again = M.withKeyframe(after, 0, 2.5);
  assert.equal(again.scene.sources[0].keyframes.length, 3); // the same instant twice is one keyframe
  const end = M.withKeyframe(before, 0, 8);
  assert.equal(end.index, 2);
  assert.deepEqual(end.scene.sources[0].keyframes[2].position_m, [1.7, 2.5, 1.5]);
});

test("the first keyframe anchors time zero and cannot be removed", () => {
  assert.equal(M.withoutKeyframe(scene(), 0, 0).sources[0].keyframes.length, 2);
  assert.equal(M.withoutKeyframe(scene(), 0, 1).sources[0].keyframes.length, 1);
});

test("sweep values parse commas, spaces and the typographic minus", () => {
  assert.deepEqual(M.parseValues("−24, -12 0", 25), { values: [-24, -12, 0], error: null });
  assert.deepEqual(M.parseValues(" ", 25).error, { kind: "empty" });
  assert.deepEqual(M.parseValues("1, x", 25).error, { kind: "number", value: "x" });
  assert.deepEqual(M.parseValues("1 2 3", 2).error, { kind: "count", value: 2 });
});

test("the plan keeps metres square and inverts exactly", () => {
  const T = M.planTransform([10, 3, 3], 640, 440, 34);
  assert.equal(T.scale, (640 - 68) / 10);
  assert.ok(Math.abs((T.x(1) - T.x(0)) - (T.y(0) - T.y(1))) < 1e-9);
  assert.deepEqual(T.toMetres(T.x(2.5), T.y(1.25)).map((v) => +v.toFixed(9)), [2.5, 1.25]);
});

test("map cells, shading and re-run merges", () => {
  const axes = [{ values: [1, 2, 3] }, { values: [10, 20] }];
  assert.deepEqual(M.cellPosition(3, axes), { column: 1, row: 1 });
  assert.equal(M.shade(5, [0, 5, 10]), 0.5);
  assert.equal(M.shade(3, [3, 3]), 0.5);
  assert.equal(M.shade(null, [1, 2]), null);
  const previous = { axes, cells: [{ index: 0, status: "failed" }, { index: 1, status: "succeeded" }] };
  const update = { axes, request: { cell_indices: [0] }, cells: [{ index: 0, status: "succeeded" }] };
  assert.deepEqual(M.mergeCells(previous, update).cells.map((cell) => cell.status), ["succeeded", "succeeded"]);
  const fresh = { axes, request: {}, cells: [{ index: 0, status: "queued" }] };
  assert.equal(M.mergeCells(previous, fresh), fresh); // a new sweep replaces the map
});


test("a new talker lands 1.5 m from the mic, clear of walls, obstacles and others", () => {
  const { scene: next, index, sample } = M.addSource(scene(), "talker", LIMITS);
  const source = next.sources[index];
  assert.equal(source.source_id, "talker-1");
  assert.equal(source.asset_id, "talker-1");
  assert.equal(source.role, "interferer"); // the scene already has a target
  assert.equal(source.keyframes.length, 1);
  const p = source.keyframes[0].position_m;
  assert.ok(Math.abs(M.distance(p, [1, 2.5, p[2]]) - 1.5) < 1e-3); // positions are rounded to 1 mm
  assert.ok(p[0] > 0.1 && p[1] > 0.1 && p[0] < 5.9 && p[1] < 4.9);
  for (const other of scene().sources) assert.ok(M.distance(p, other.keyframes[0].position_m) >= 0.5);
  assert.ok(!next.room.objects.some((object) => M.insideObject(p, object)));
  const t = next.room.sources.find((item) => item.transducer_id === "talker-1");
  assert.equal(t.directivity_id, "speech_human");
  assert.deepEqual(t.pose.position_m, p);
  assert.ok(typeof sample === "string");
});

test("the first talker of a scene without a target becomes the target", () => {
  const s = M.withRole(scene(), 0, "interferer");
  const { scene: next, index } = M.addSource(s, "talker", LIMITS);
  assert.equal(next.sources[index].role, "target");
});

test("noise is added omnidirectional and the limit stops a ninth source", () => {
  let s = scene();
  const added = M.addSource(s, "noise", LIMITS);
  assert.equal(added.scene.sources[added.index].role, "noise");
  assert.equal(added.scene.room.sources.at(-1).directivity_id, "omnidirectional");
  for (let i = 0; i < 5; i++) s = M.addSource(s, i % 2 ? "noise" : "talker", LIMITS).scene;
  assert.equal(s.sources.length, 8);
  assert.equal(M.addSource(s, "talker", LIMITS), null);
});

test("removing keeps at least one source and drops its transducer", () => {
  const s = M.removeSource(scene(), 1);
  assert.deepEqual(s.sources.map((source) => source.source_id), ["a", "n"]);
  assert.deepEqual(s.room.sources.map((t) => t.transducer_id), ["a", "n"]);
  let one = s;
  one = M.removeSource(one, 0);
  assert.equal(M.removeSource(one, 0).sources.length, 1);
});

test("a role change switches the directivity between talker and noise", () => {
  const s = M.withRole(scene(), 0, "noise");
  assert.equal(s.sources[0].role, "noise");
  assert.equal(s.room.sources[0].directivity_id, "omnidirectional");
  assert.equal(M.withRole(s, 0, "interferer").room.sources[0].directivity_id, "speech_human");
});

test("stays still keeps one keyframe; moving again adds one at the end", () => {
  const still = M.withStill(scene(), 0, true);
  assert.equal(still.sources[0].keyframes.length, 1);
  assert.ok(M.isStill(still.sources[0]));
  const moving = M.withStill(still, 0, false);
  assert.deepEqual(moving.sources[0].keyframes.map((key) => key.time_s), [0, 10]);
  assert.deepEqual(moving.sources[0].keyframes[1].position_m, moving.sources[0].keyframes[0].position_m);
  assert.ok(!M.isStill(scene().sources[0]));
});

test("the default reference follows the model's task", () => {
  const defaults = { noise_suppression: "speech", voice_isolation: "near" };
  assert.equal(M.defaultPolicy("noise_suppression", scene(), defaults), "speech");
  assert.equal(M.defaultPolicy("voice_isolation", scene(), defaults), "near");
  assert.equal(M.defaultPolicy(null, scene(), defaults), "target");
  assert.equal(M.defaultPolicy(null, M.withRole(scene(), 0, "interferer"), defaults), "speech");
});

test("sources of one role step through shades of its colour", () => {
  let s = scene();
  s = M.addSource(s, "noise", LIMITS).scene;
  assert.equal(M.roleShade(s, 0), 0);
  assert.equal(M.roleShade(s, 2), 0);
  assert.equal(M.roleShade(s, 3), 1);
  assert.equal(M.shadeColor("#000000", 0), "#000000");
  assert.equal(M.shadeColor("#000000", 1), "#737373");
});

test("new sources keep 0.5 m on the floor from every other source's path, whatever their height", () => {
  const flat = (p) => [p[0], p[1]];
  const toSegment = (p, a, b) => {
    const d = [b[0] - a[0], b[1] - a[1]];
    const length = d[0] ** 2 + d[1] ** 2;
    const u = length ? Math.max(0, Math.min(1, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / length)) : 0;
    return Math.hypot(p[0] - a[0] - u * d[0], p[1] - a[1] - u * d[1]);
  };
  let s = scene();
  for (let i = 0; i < 5; i++) s = M.addSource(s, i % 2 ? "noise" : "talker", LIMITS).scene;
  for (const added of s.sources.slice(3)) {
    const p = flat(added.keyframes[0].position_m);
    for (const other of s.sources) {
      if (other === added) continue;
      const keys = other.keyframes.map((key) => flat(key.position_m));
      const segments = keys.length > 1 ? keys.slice(1).map((b, i) => [keys[i], b]) : [[keys[0], keys[0]]];
      for (const [a, b] of segments) assert.ok(toSegment(p, a, b) >= 0.5 - 1e-9, `${added.source_id} is ${toSegment(p, a, b).toFixed(2)} m from ${other.source_id}`);
    }
  }
});

test('volume ripples preserve relative levels, silence and audio-clock seeking', () => {
  const trace = (db) => ({ levels: new Float32Array(100).fill(db), frameSeconds: 0.02 });
  assert.equal(M.volumeAt(null, 1), 0);
  assert.equal(M.volumeAt(trace(-120), 1), 0);
  assert.equal(M.volumeAt(trace(-15), 3), 0);
  assert.equal(M.volumeAt(trace(-15), -1), 0);
  assert(M.volumeAt(trace(-25), 1) > M.volumeAt(trace(-45), 1));
  assert.deepEqual(M.volumeRipples(trace(-120), 1), []);
  const loud = M.volumeRipples(trace(-25), 1);
  const quiet = M.volumeRipples(trace(-45), 1);
  assert(loud.length <= 6 && quiet.length <= 6);
  assert(Math.max(...loud.map(x => x.radius)) > Math.max(...quiet.map(x => x.radius)));
  assert(loud[0].opacity > quiet[0].opacity);
  M.volumeRipples(trace(-25), 1.7);
  assert.deepEqual(M.volumeRipples(trace(-25), 1), loud);
  assert.deepEqual(M.volumeRipples(trace(-25), 4), []);
  assert(loud.every(x => x.birth <= 1 && x.opacity > 0));
});

test('a short syllable between fronts is visible and then fades', () => {
  const levels = new Float32Array(100).fill(-120);
  levels[4] = -20;
  const trace = {levels, frameSeconds: 0.02};
  assert(M.volumeRipples(trace, 0.3).length > 0);
  assert.deepEqual(M.volumeRipples(trace, 1.4), []);
});
