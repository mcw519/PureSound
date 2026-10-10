/* Pipeline screen, pure logic: how a trace report becomes rows, series, axis
 * ranges and readable values.  No DOM here, so Node can test it
 * (test/web/pipeline_model.test.cjs); the browser loads it as a plain script. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundPipelineModel = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Keys a reader has no use for: local paths, hashes, QC bookkeeping and the
  // bank's own sampling state.
  const HIDDEN_PARAM = /(^|_)(path|sha256)$|^qc_|^used_channels$|^union_member_name$/;
  // A container the recorder could only name by its type ("<list>", "<tensor [1, 96000]>").
  const PLACEHOLDER = /^<[^>]+>$/;
  const STATE_LABELS = {
    fired: "applied on this row",
    absent: "not in this recipe",
    zero: "probability 0 for these rows",
    "off-here": "off in the inspector",
  };

  function text(entry, lang) {
    return (entry && (entry[lang] || entry.en)) || "";
  }

  const round = (value, places = 9) => Math.round(value * 10 ** places) / 10 ** places;
  const finite = (value) => typeof value === "number" && Number.isFinite(value);

  /* The recipe key a stage's block belongs to: "augmentation_noise_args.absolute_floor" -> "augmentation_noise". */
  function blockKey(block) {
    return block ? String(block).split(".")[0].replace(/_args$/, "") : null;
  }

  function stageState(stage, notes) {
    if (stage.fired) return "fired";
    const recipe = stage.recipe || {};
    const key = blockKey(stage.block);
    if (!recipe.configured && key && (notes || []).some((note) => note.subject === key)) return "off-here";
    if (!recipe.configured) return "absent";
    if (recipe.enabled && recipe.prob === 0) return "zero";
    return "skipped";
  }

  function stateLabel(row) {
    if (row.state === "off-here" && row.offReason) return `${STATE_LABELS["off-here"]}: ${row.offReason}`;
    if (row.state !== "skipped") return STATE_LABELS[row.state] || row.state;
    if (!row.block) return "not needed on this row";
    const prob = row.recipe && row.recipe.prob;
    return finite(prob) ? `did not fire on this row · p = ${formatParamNumber(prob)}` : "did not fire on this row";
  }

  function stageRows(report, catalog, lang) {
    const groups = new Map((catalog.groups || []).map((group) => [group.id, group]));
    return (report.stages || []).map((stage, index) => {
      const entry = (catalog.stages || {})[stage.id] || {};
      const group = groups.get(stage.group) || { id: stage.group, color: "#6f7078", title: { en: stage.group } };
      const state = stageState(stage, report.notes);
      const key = blockKey(stage.block);
      const note = state === "off-here" ? (report.notes || []).find((item) => item.subject === key) : null;
      return {
        ...stage,
        index,
        title: text(entry.title, lang) || stage.id,
        entry,
        groupInfo: { id: group.id, color: group.color, title: text(group.title, lang) },
        state,
        offReason: note ? note.reason : null,
      };
    });
  }

  function previousFired(rows, id) {
    let previous = null;
    for (const row of rows) {
      if (row.id === id) return previous;
      if (row.fired) previous = row;
    }
    return null;
  }

  function esnrSeries(rows) {
    let last = null;
    return rows.filter((row) => row.fired && row.metrics).map((row) => {
      const value = finite(row.metrics.esnr_db) ? row.metrics.esnr_db : null;
      const delta = value !== null && last !== null ? round(value - last) : null;
      if (value !== null) last = value;
      return { id: row.id, index: row.index, value, delta, state: row.metrics.esnr_state };
    });
  }

  function biggestDrop(series) {
    let best = null;
    series.forEach((point) => {
      if (finite(point.delta) && point.delta < 0 && (best === null || point.delta < best.delta)) best = point;
    });
    return best ? best.id : null;
  }

  function modelSeries(rows, metric) {
    return rows.filter((row) => row.fired && row.model && !row.model.skipped).map((row) => {
      const model = row.model;
      if (metric === "level_change_db") {
        const value = finite(model.level_change_db) ? model.level_change_db : null;
        return { id: row.id, index: row.index, input: 0, output: value, change: value, reused: model.same_input_as || null };
      }
      const input = finite((model.input || {})[metric]) ? model.input[metric] : null;
      const output = finite((model.output || {})[metric]) ? model.output[metric] : null;
      const change = input !== null && output !== null ? round(output - input) : null;
      return { id: row.id, index: row.index, input, output, change, reused: model.same_input_as || null };
    });
  }

  /* An axis range over the finite values, clamped to [floor, ceil], padded,
   * and never narrower than minSpan so a flat series still reads as flat. */
  function chartDomain(values, { floor = -30, ceil = 60, pad = 3, minSpan = 15 } = {}) {
    const kept = (values || []).filter(finite).map((value) => Math.min(Math.max(value, floor), ceil));
    if (!kept.length) return [floor, ceil];
    let low = Math.min(...kept) - pad;
    let high = Math.max(...kept) + pad;
    if (high - low < minSpan) {
      const centre = (low + high) / 2;
      low = centre - minSpan / 2;
      high = centre + minSpan / 2;
    }
    return [round(Math.max(low, floor), 6), round(Math.min(high, ceil), 6)];
  }

  function minus(value, digits) {
    return (value < 0 ? "−" : "") + Math.abs(value).toFixed(digits);
  }

  function formatNumber(value, { digits = 1, unit = "", cap = null } = {}) {
    if (!finite(value)) return "—";
    if (cap !== null && value >= cap) return `≥ ${cap}${unit}`;
    return minus(value, digits) + unit;
  }

  function formatSigned(value, { digits = 1, unit = "" } = {}) {
    if (!finite(value)) return "—";
    const rounded = Number(value.toFixed(digits));
    return (rounded > 0 ? "+" : "") + minus(rounded, digits) + unit;
  }

  function formatParamNumber(value) {
    if (Number.isInteger(value) || Math.abs(value) >= 100) return String(Math.round(value));
    return String(Number(value.toPrecision(3)));
  }

  function formatValue(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "number") return finite(value) ? formatParamNumber(value) : "—";
    if (Array.isArray(value)) return value.map(formatValue).join(", ");
    return String(value);
  }

  /* [label, value] rows for a stage's drawn parameters; nested objects are
   * flattened with dots, a dataclass shows its class, and paths, internals,
   * empty values and type placeholders are left out. */
  function flattenParams(params, prefix = "") {
    const rows = [];
    Object.entries(params || {}).forEach(([key, value]) => {
      if (HIDDEN_PARAM.test(key)) return;
      const name = prefix ? `${prefix}.${key}` : key;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        if (value.class) rows.push([name, String(value.class)]);
        const { class: _ignored, ...rest } = value;
        rows.push(...flattenParams(rest, name));
      } else if (value !== null && value !== undefined && !(typeof value === "string" && PLACEHOLDER.test(value))) {
        rows.push([name, formatValue(value)]);
      }
    });
    return rows;
  }

  /* The floor plan of a bank room: metres, x across and y into the room. */
  function topView(room) {
    if (!room || !Array.isArray(room.room_dim) || !Array.isArray(room.receiver)) return null;
    const point = (position) => ({ x: position[0], y: position[1], z: position[2] });
    return {
      width: room.room_dim[0],
      depth: room.room_dim[1],
      height: room.room_dim[2],
      receiver: point(room.receiver),
      sources: (room.sources || []).filter((source) => Array.isArray(source.position)).map((source) => ({
        ...point(source.position),
        label: source.label,
        roles: source.roles || [],
        distance: source.distance_m,
      })),
      obstacles: (room.obstacles || []).map((obstacle) => ({
        points: (obstacle.footprint || []).map(([x, y]) => ({ x, y })),
        height: obstacle.z_max,
        material: obstacle.material || null,
      })),
    };
  }

  /* The role a response played, as the page names it: a noise clip coloured
   * through the room is served as an "interferer" channel, but it is noise. */
  function displayRole(rir) {
    return rir.stage === "noise.recorded" ? "noise" : rir.role;
  }

  /* The report's room with each source's roles in display terms. */
  function displayRoom(report) {
    if (!report.room) return null;
    const roles = new Map();
    (report.rirs || []).forEach((rir) => {
      const label = rir.metadata && rir.metadata.label;
      if (!label) return;
      const list = roles.get(label) || [];
      const role = displayRole(rir);
      if (!list.includes(role)) list.push(role);
      roles.set(label, list);
    });
    return { ...report.room, sources: (report.room.sources || []).map((source) => ({ ...source, roles: roles.get(source.label) || [] })) };
  }

  /* Roles whose impulse responses were applied in a stage (to light them up in the room). */
  function stageRoles(report, id) {
    return [...new Set((report.rirs || []).filter((rir) => rir.stage === id).map(displayRole))];
  }

  /* Mono 32-bit float WAV: a stage's own change can exceed full scale, and a
   * 16-bit file would clip exactly what the lane is there to show. */
  function encodeFloatWav(samples, sampleRate) {
    const bytes = new ArrayBuffer(44 + samples.length * 4);
    const view = new DataView(bytes);
    const ascii = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
    ascii(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 4, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 3, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 4, true);
    view.setUint16(32, 4, true);
    view.setUint16(34, 32, true);
    ascii(36, "data");
    view.setUint32(40, samples.length * 4, true);
    for (let index = 0; index < samples.length; index += 1) view.setFloat32(44 + index * 4, samples[index], true);
    return bytes;
  }

  return {
    text,
    blockKey,
    stageState,
    stateLabel,
    stageRows,
    previousFired,
    esnrSeries,
    biggestDrop,
    modelSeries,
    chartDomain,
    formatNumber,
    formatSigned,
    formatValue,
    flattenParams,
    topView,
    displayRole,
    displayRoom,
    stageRoles,
    encodeFloatWav,
  };
});
