/* Acoustic world screen, pure logic: poses along keyframed paths, scene
 * edits that keep a scene consistent, the checks the editor shows before the
 * server validates, sweep axes and the plan's coordinate transform.  No DOM
 * here, so Node can test it (test/web/world_model.test.cjs); the browser
 * loads it as a plain script. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundWorldModel = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  // One shared dBFS scale, never normalized per source: distance, source gain
  // and silence remain visible. Traces are RMS levels from the rendered stems.
  function volumeAt(trace, time) {
    if (!trace || time < 0 || time >= trace.levels.length * trace.frameSeconds) return 0;
    const index = Math.floor(time / trace.frameSeconds);
    let level = 0;
    // Fast attack, 100 ms release; look back only, so seeking stays exact.
    const history = Math.ceil(0.3 / trace.frameSeconds);
    for (let j = Math.max(0, index - history); j <= index; j += 1) {
      const db = trace.levels[j];
      if (Number.isFinite(db)) level = Math.max(level, clamp((db + 65) / 50, 0, 1) * Math.exp(-(index - j) * trace.frameSeconds / 0.1));
    }
    return level;
  }

  // Reconstruct a bounded set of ripples from the audio clock. Seeking,
  // pausing, looping and frame rate cannot change their ages or origins.
  function volumeRipples(trace, time) {
    const period = 0.28;
    const frames = [];
    for (let i = 0; i < 4; i += 1) {
      const birth = (Math.floor(time / period) - i) * period;
      if (birth < 0 || !trace) continue;
      let level = 0;
      // Keep brief syllables between successive fronts visible.
      for (let t = Math.max(0, birth - period); t <= birth + 1e-9; t += trace.frameSeconds) level = Math.max(level, volumeAt(trace, t));
      const age = time - birth;
      const lifetime = 0.3 + 0.7 * level;
      if (level <= 0 || age >= lifetime) continue;
      frames.push({ birth, radius: 0.08 + age * 1.4, opacity: 0.65 * level * (1 - age / lifetime) ** 1.5 });
    }
    return frames;
  }

  /* Signed shortest turn from angle a to b, in degrees. */
  function angleDelta(a, b) {
    const raw = b - a;
    const delta = ((((raw + 180) % 360) + 360) % 360) - 180;
    return delta === -180 && raw > 0 ? 180 : delta;
  }

  /* Position and orientation of a source at `time`, linear between keyframes
   * and held outside them, turning the short way round — the renderer's rule. */
  function pose(source, time) {
    const keys = source.keyframes;
    let before = keys[0];
    let after = keys[keys.length - 1];
    for (let i = 0; i < keys.length; i++) {
      if (keys[i].time_s <= time) before = keys[i];
      if (keys[i].time_s >= time) { after = keys[i]; break; }
    }
    const span = after.time_s - before.time_s;
    const u = span > 0 ? clamp((time - before.time_s) / span, 0, 1) : 0;
    return {
      position_m: before.position_m.map((v, i) => v + (after.position_m[i] - v) * u),
      yaw_deg: before.yaw_deg + angleDelta(before.yaw_deg, after.yaw_deg) * u,
      pitch_deg: (before.pitch_deg || 0) + angleDelta(before.pitch_deg || 0, after.pitch_deg || 0) * u,
    };
  }

  /* The renderer's near-region weight: a raised cosine across 0.2 m. */
  function nearWeight(metres, radius) {
    const x = clamp((metres - (radius - 0.1)) / 0.2, 0, 1);
    return 0.5 * (1 + Math.cos(Math.PI * x));
  }

  function micPosition(scene) {
    return scene.room.receivers[0].pose.position_m;
  }

  /* What a source is doing at `time`, for the readout under the plan. */
  function sourceState(scene, source, time) {
    const state = pose(source, time);
    const metres = distance(state.position_m, micPosition(scene));
    return {
      ...state,
      distance_m: metres,
      near_weight: isTalker(source) ? nearWeight(metres, scene.near_radius_m) : 0,
    };
  }

  /* A source's role: "target", "interferer" (another talker) or "noise". */
  function sourceRole(scene, source) {
    return source.role;
  }

  function isTalker(source) {
    return source.role === "target" || source.role === "interferer";
  }

  function insidePolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i];
      const [xj, yj] = polygon[j];
      if (yi > point[1] !== yj > point[1] && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function insideObject(point, object) {
    return point[2] >= object.z_min && point[2] <= object.z_max && insidePolygon(point, object.footprint);
  }

  /* Problems the editor can see without the server, as { source, key, kind }
   * so the table and plan can mark the exact keyframe (a "speed" issue marks
   * the segment that ends there).  The server's validation stays
   * authoritative; these only point at where to look.  Kinds: outside, mic,
   * obstacle, start, order, range, speed. */
  function keyframeIssues(scene, limits) {
    const issues = [];
    const dims = scene.room.dimensions_m;
    const mic = micPosition(scene);
    scene.sources.forEach((source, s) => {
      const keys = source.keyframes;
      keys.forEach((key, k) => {
        const add = (kind, extra = {}) => issues.push({ source: s, key: k, kind, ...extra });
        const p = key.position_m;
        if (p.some((v, i) => v <= 0 || v >= dims[i])) add("outside");
        else if (distance(p, mic) < limits.min_mic_distance_m - 1e-9) add("mic");
        else if (scene.room.objects.some((object) => insideObject(p, object))) add("obstacle");
        if (k === 0 && key.time_s !== 0) add("start");
        if (key.time_s < 0 || key.time_s > scene.duration_s) add("range");
        if (k === 0) return;
        const previous = keys[k - 1];
        const dt = key.time_s - previous.time_s;
        const moved = distance(p, previous.position_m);
        if (dt <= 0) add("order");
        else if (moved / dt > limits.speed_m_s + 1e-9) add("speed", { speed: moved / dt });
      });
    });
    return issues;
  }

  /* Speed of the segment that ends at keyframe k, in m/s (null for the first). */
  function segmentSpeed(source, k) {
    if (k === 0) return null;
    const a = source.keyframes[k - 1];
    const b = source.keyframes[k];
    const dt = b.time_s - a.time_s;
    return dt > 0 ? distance(a.position_m, b.position_m) / dt : Infinity;
  }

  /* Change the duration; keyframe times keep their share of the scene and
   * start times stay inside it. */
  function withDuration(scene, seconds) {
    const next = structuredClone(scene);
    const scale = seconds / scene.duration_s;
    next.duration_s = seconds;
    for (const source of next.sources) {
      for (const key of source.keyframes) key.time_s = Math.round(key.time_s * scale * 1000) / 1000;
      source.start_s = Math.min(source.start_s, Math.max(0, seconds - 0.1));
    }
    return next;
  }

  /* Change one room dimension; the shoebox surfaces follow. */
  function withRoomSize(scene, axis, metres) {
    const next = structuredClone(scene);
    const old = next.room.dimensions_m[axis];
    next.room.dimensions_m[axis] = metres;
    for (const surface of next.room.surfaces) {
      for (const vertex of surface.vertices_m) vertex[axis] = (vertex[axis] / old) * metres;
    }
    return next;
  }

  /* A keyframe at `time` on the path itself, so adding one changes nothing
   * until it is moved; past the last keyframe it extends the path in place. */
  function withKeyframe(scene, s, time) {
    const next = structuredClone(scene);
    const source = next.sources[s];
    const keys = source.keyframes;
    const at = clamp(Math.round(time * 100) / 100, 0, next.duration_s);
    if (keys.some((key) => Math.abs(key.time_s - at) < 0.005)) return { scene: next, index: keys.findIndex((key) => Math.abs(key.time_s - at) < 0.005) };
    const state = pose(source, at);
    const key = { time_s: at, position_m: state.position_m.map((v) => Math.round(v * 1000) / 1000), yaw_deg: Math.round(state.yaw_deg * 10) / 10, pitch_deg: Math.round(state.pitch_deg * 10) / 10 };
    const index = keys.findIndex((existing) => existing.time_s > at);
    if (index < 0) keys.push(key);
    else keys.splice(index, 0, key);
    return { scene: next, index: index < 0 ? keys.length - 1 : index };
  }

  /* Remove keyframe k; the first keyframe anchors time zero and stays. */
  function withoutKeyframe(scene, s, k) {
    const next = structuredClone(scene);
    const keys = next.sources[s].keyframes;
    if (k > 0 && keys.length > 1) keys.splice(k, 1);
    return next;
  }

  const SPEECH_SAMPLES = ["speaker-a", "speaker-b", "speaker-a-2"];
  const NOISE_SAMPLES = ["fan", "hum", "babble", "clatter", "pink", "rumble"];
  const DIRECTIVITY = { target: "speech_human", interferer: "speech_human", noise: "omnidirectional" };

  /* Add a talker or a noise source, standing still 1.5 m from the
   * microphone at the first free bearing (inside the walls, outside
   * obstacles, at least 0.5 m on the floor from every other source's path),
   * facing the microphone.  A talker is the target when the scene has none.  Returns
   * { scene, index, sample } with a suggested sample for its audio, or null
   * at the source limit. */
  function addSource(scene, kind, limits) {
    if (scene.sources.length >= limits.sources) return null;
    const next = structuredClone(scene);
    const prefix = kind === "noise" ? "noise" : "talker";
    const taken = new Set(next.sources.map((source) => source.source_id));
    let n = 1;
    while (taken.has(`${prefix}-${n}`)) n += 1;
    const id = `${prefix}-${n}`;
    const role = kind === "noise" ? "noise" : next.sources.some((source) => source.role === "target") ? "interferer" : "target";
    const height = kind === "noise" ? 0.8 : 1.5;
    const position = freeSpot(next, height);
    const mic = micPosition(next);
    const yaw = kind === "noise" ? 0 : Math.round((Math.atan2(mic[1] - position[1], mic[0] - position[0]) * 180) / Math.PI);
    const same = next.sources.filter((source) => (kind === "noise") === (source.role === "noise")).length;
    const pool = kind === "noise" ? NOISE_SAMPLES : SPEECH_SAMPLES;
    next.sources.push({
      source_id: id,
      asset_id: id,
      keyframes: [{ time_s: 0, position_m: position, yaw_deg: yaw, pitch_deg: 0 }],
      role,
      gain_db: role === "target" ? 0 : role === "interferer" ? -6 : -12,
      start_s: 0,
      repeat: true,
    });
    const template = next.room.sources[0];
    next.room.sources.push({
      ...structuredClone(template),
      transducer_id: id,
      pose: { ...structuredClone(template.pose), position_m: [...position], orientation_ypr_deg: [yaw, 0, 0] },
      directivity_id: DIRECTIVITY[role],
    });
    return { scene: next, index: next.sources.length - 1, sample: pool[same % pool.length] };
  }

  function freeSpot(scene, height) {
    const mic = micPosition(scene);
    const dims = scene.room.dimensions_m;
    const paths = scene.sources.flatMap((source) => {
      const keys = source.keyframes.map((key) => key.position_m);
      return keys.length > 1 ? keys.slice(1).map((b, i) => [keys[i], b]) : [[keys[0], keys[0]]];
    });
    for (const radius of [1.5, 1.0, 2.0, 2.5, 0.6]) {
      for (let step = 0; step < 24; step++) {
        const angle = (step * Math.PI) / 12;
        const point = [mic[0] + radius * Math.cos(angle), mic[1] + radius * Math.sin(angle), height].map((v) => Math.round(v * 1000) / 1000);
        const inside = point[0] > 0.1 && point[1] > 0.1 && point[0] < dims[0] - 0.1 && point[1] < dims[1] - 0.1;
        if (!inside || scene.room.objects.some((object) => insideObject(point, object))) continue;
        if (paths.every(([a, b]) => floorDistanceToSegment(point, a, b) >= 0.5)) return point;
      }
    }
    return [dims[0] / 2, dims[1] / 2, height];
  }

  function floorDistanceToSegment(point, a, b) {
    const d = [b[0] - a[0], b[1] - a[1]];
    const length = d[0] ** 2 + d[1] ** 2;
    const u = length ? Math.max(0, Math.min(1, ((point[0] - a[0]) * d[0] + (point[1] - a[1]) * d[1]) / length)) : 0;
    return Math.hypot(point[0] - a[0] - u * d[0], point[1] - a[1] - u * d[1]);
  }

  /* Remove source `index` and its transducer; a scene keeps one source. */
  function removeSource(scene, index) {
    if (scene.sources.length <= 1) return scene;
    const next = structuredClone(scene);
    const [removed] = next.sources.splice(index, 1);
    next.room.sources = next.room.sources.filter((t) => t.transducer_id !== removed.source_id);
    return next;
  }

  /* Change a source's role; talkers radiate like a person, noise evenly. */
  function withRole(scene, index, role) {
    const next = structuredClone(scene);
    const source = next.sources[index];
    source.role = role;
    const transducer = next.room.sources.find((t) => t.transducer_id === source.source_id);
    if (transducer) transducer.directivity_id = DIRECTIVITY[role];
    return next;
  }

  function isStill(source) {
    return source.keyframes.length === 1;
  }

  /* Still keeps only the first keyframe; moving again adds a keyframe at the
   * end of the scene on the same spot, ready to be dragged. */
  function withStill(scene, index, still) {
    const next = structuredClone(scene);
    const source = next.sources[index];
    if (still) source.keyframes = source.keyframes.slice(0, 1);
    else if (source.keyframes.length === 1) source.keyframes.push({ ...structuredClone(source.keyframes[0]), time_s: next.duration_s });
    return next;
  }

  /* The reference a model is scored against unless the user picks one. */
  function defaultPolicy(task, scene, defaults) {
    if (defaults && defaults[task]) return defaults[task];
    return scene.sources.some((source) => source.role === "target") ? "target" : "speech";
  }

  /* 0..1 position of a source among the sources sharing its role. */
  function roleShade(scene, index) {
    const role = scene.sources[index].role;
    const peers = scene.sources.map((source, i) => (source.role === role ? i : -1)).filter((i) => i >= 0);
    return peers.length > 1 ? peers.indexOf(index) / (peers.length - 1) : 0;
  }

  /* A role colour lightened by `shade` (0 keeps it), up to 45 % toward white. */
  function shadeColor(hex, shade) {
    const mix = 0.45 * shade;
    const channels = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    return `#${channels.map((c) => Math.round(c + (255 - c) * mix).toString(16).padStart(2, "0")).join("")}`;
  }

  /* "−24, −12, 0" -> numbers.  `error` is null, or { kind, value } with kind
   * "empty", "number" (value: the entry that is not one) or "count". */
  function parseValues(text, maximum) {
    const parts = String(text).split(/[,\s]+/).map((part) => part.replace("\u2212", "-")).filter(Boolean);
    if (!parts.length) return { values: [], error: { kind: "empty" } };
    const values = [];
    for (const part of parts) {
      const value = Number(part);
      if (!Number.isFinite(value)) return { values: [], error: { kind: "number", value: part } };
      values.push(value);
    }
    return { values, error: values.length > maximum ? { kind: "count", value: maximum } : null };
  }

  /* Uniform plan scale: metres -> SVG units, y pointing up, with `pad` units
   * around the room.  Returns the transform and its inverse. */
  function planTransform(dims, width, height, pad) {
    const scale = Math.min((width - 2 * pad) / dims[0], (height - 2 * pad) / dims[1]);
    const left = (width - dims[0] * scale) / 2;
    const top = (height - dims[1] * scale) / 2;
    return {
      scale,
      x: (metres) => left + metres * scale,
      y: (metres) => top + (dims[1] - metres) * scale,
      toMetres: (x, y) => [(x - left) / scale, dims[1] - (y - top) / scale],
      box: { left, top, width: dims[0] * scale, height: dims[1] * scale },
    };
  }

  /* Position of a cell in the sweep grid: axis 0 runs across, axis 1 down. */
  function cellPosition(index, axes) {
    const rows = axes[1].values.length;
    return { column: Math.floor(index / rows), row: index % rows };
  }

  /* 0..1 position of `value` inside the finite values of a map, for shading;
   * a map with one distinct value sits in the middle. */
  function shade(value, values) {
    const finite = values.filter(Number.isFinite);
    if (!Number.isFinite(value) || !finite.length) return null;
    const low = Math.min(...finite);
    const high = Math.max(...finite);
    return high > low ? (value - low) / (high - low) : 0.5;
  }

  /* Merge a re-run of some cells into the map they came from. */
  function mergeCells(previous, update) {
    if (!previous || !update.request?.cell_indices) return update;
    if (JSON.stringify(previous.axes) !== JSON.stringify(update.axes)) return update;
    const replaced = new Map(update.cells.map((cell) => [cell.index, cell]));
    return { ...previous, cells: previous.cells.map((cell) => replaced.get(cell.index) || cell) };
  }

  return {
    volumeAt,
    volumeRipples,
    angleDelta,
    cellPosition,
    distance,
    addSource,
    defaultPolicy,
    insideObject,
    isStill,
    isTalker,
    keyframeIssues,
    mergeCells,
    micPosition,
    nearWeight,
    parseValues,
    planTransform,
    pose,
    removeSource,
    roleShade,
    segmentSpeed,
    shade,
    shadeColor,
    sourceRole,
    sourceState,
    withDuration,
    withKeyframe,
    withRole,
    withRoomSize,
    withStill,
    withoutKeyframe,
  };
});
