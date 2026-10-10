/* Annotate screen, pure logic: reading span files (the tool's own JSON and
 * CSV, windows.json, Audacity labels) and writing the four export formats.
 * The output is byte for byte what the standalone annotator wrote, because
 * evaluation reads its windows.json (test/web/annotate_model.test.cjs pins
 * that).  No DOM here; the browser loads it as a plain script. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PureSoundAnnotateModel = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /* "take 3.spans.json" -> "take 3": what a span file and its audio share. */
  const base = (name) => String(name).replace(/\.[^.]+$/, "").replace(/\.spans$/i, "");
  const r3 = (n) => +(+n).toFixed(3);

  function splitCsv(line) {
    const out = [];
    let cur = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (quoted) {
        if (c === '"') {
          if (line[i + 1] === '"') { cur += '"'; i++; } else quoted = false;
        } else cur += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { out.push(cur); cur = ""; } else cur += c;
    }
    out.push(cur);
    return out;
  }

  function normSpan(row, defaultTag) {
    return { a: +(row.start_s ?? row.a ?? row.start ?? 0), b: +(row.end_s ?? row.b ?? row.end ?? 0), tag: row.tag || defaultTag, label: row.label || "" };
  }

  /* {base name: [{a, b, tag, label}]} from a span file, or null when the
   * text is none of the formats. */
  function parseSpans(text, fileName, defaultTag) {
    const trimmed = text.trim();
    if (!trimmed) return null;
    if (trimmed[0] === "{" || trimmed[0] === "[") {
      let json;
      try { json = JSON.parse(trimmed); } catch (error) { return null; }
      // One file, as this tool writes it.
      if (json && Array.isArray(json.spans)) return { [base(json.file || fileName)]: json.spans.map((row) => normSpan(row, defaultTag)) };
      // A whole folder.
      if (json && Array.isArray(json.files)) {
        const out = {};
        json.files.forEach((file) => { out[base(file.file || file.path || "")] = (file.spans || []).map((row) => normSpan(row, defaultTag)); });
        return out;
      }
      if (Array.isArray(json)) return { [base(fileName)]: json.map((row) => normSpan(row, defaultTag)) };
      // windows.json: {base: {keep: [[a, b]], suppress: [[a, b]]}}
      const out = {};
      let hit = false;
      for (const key in json) {
        const value = json[key];
        if (key[0] === "_" || !value || typeof value !== "object" || Array.isArray(value)) continue;
        const rows = [];
        (value.keep || []).forEach((pair) => rows.push({ a: +pair[0], b: +pair[1], tag: "near / keep", label: "" }));
        (value.suppress || []).forEach((pair) => rows.push({ a: +pair[0], b: +pair[1], tag: "far / suppress", label: "" }));
        if (rows.length) { out[base(key)] = rows; hit = true; }
      }
      return hit ? out : null;
    }
    const lines = trimmed.split(/\r?\n/).filter((line) => line.trim());
    // Audacity labels: start<TAB>end<TAB>label
    if (lines[0].includes("\t") && !/start_s/i.test(lines[0])) {
      const rows = lines.map((line) => {
        const cells = line.split("\t");
        return { a: +cells[0], b: +cells[1], tag: defaultTag, label: (cells[2] || "").trim() };
      }).filter((row) => isFinite(row.a) && isFinite(row.b));
      return rows.length ? { [base(fileName)]: rows } : null;
    }
    // CSV, with or without a leading file column.
    const head = splitCsv(lines[0]).map((cell) => cell.trim().toLowerCase());
    if (!head.includes("start_s")) return null;
    const column = (name) => head.indexOf(name);
    const out = {};
    for (let i = 1; i < lines.length; i++) {
      const cells = splitCsv(lines[i]);
      if (!cells.length) continue;
      const key = column("file") >= 0 ? base(cells[column("file")] || fileName) : base(fileName);
      (out[key] = out[key] || []).push({
        a: +cells[column("start_s")],
        b: +cells[column("end_s")],
        tag: (column("tag") >= 0 ? cells[column("tag")] : "") || defaultTag,
        label: (column("label") >= 0 ? cells[column("label")] : "") || "",
      });
    }
    for (const key in out) out[key] = out[key].filter((row) => isFinite(row.a) && isFinite(row.b));
    return out;
  }

  /* keep: near speech that must survive; suppress: a voice to attenuate. */
  function windowsRole(tag) {
    const name = String(tag).toLowerCase();
    if (name.includes("keep") || name.includes("near") || name.includes("double")) return "keep";
    if (name.includes("sup") || name.includes("far")) return "suppress";
    return null;
  }

  const spanRows = (file) => file.spans.map((span) => ({ start_s: r3(span.a), end_s: r3(span.b), duration_s: r3(span.b - span.a), tag: span.tag, label: span.label || null }));
  const singleFileJson = (file) => JSON.stringify({ file: file.name, sample_rate: file.sr || null, duration_s: file.duration ? r3(file.duration) : null, spans: spanRows(file) }, null, 2);

  /* The export of `files` in `format` ("json" | "csv" | "windows" | "audacity").
   * Audacity labels are per file: `current` is the file on screen. */
  function buildExport(format, files, { current = null, folderName = "" } = {}) {
    const list = files.filter((file) => file.spans.length);
    if (!list.length) return "(no spans yet)";
    const many = files.length > 1;
    if (format === "json") {
      if (!many) return singleFileJson(list[0]);
      return JSON.stringify({
        folder: folderName || null,
        annotated_files: list.length,
        spans_total: list.reduce((total, file) => total + file.spans.length, 0),
        generated_by: "Span",
        files: list.map((file) => ({ file: file.name, path: file.path, sample_rate: file.sr || null, duration_s: file.duration ? r3(file.duration) : null, spans: spanRows(file) })),
      }, null, 2);
    }
    if (format === "csv") {
      const head = (many ? "file," : "") + "start_s,end_s,duration_s,tag,label";
      const quote = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
      return head + "\n" + list.flatMap((file) => file.spans.map((span) =>
        (many ? quote(file.name) + "," : "") + `${r3(span.a)},${r3(span.b)},${r3(span.b - span.a)},${quote(span.tag)},${quote(span.label || "")}`)).join("\n");
    }
    if (format === "audacity") {
      if (!current || !current.spans.length) return "(the current file has no spans — Audacity labels are per file)";
      return current.spans.map((span) => `${r3(span.a)}\t${r3(span.b)}\t${span.label || span.tag}`).join("\n");
    }
    // windows.json: one entry per file, keep / suppress from the tag name.
    const out = { _comment: "Spans annotated in Span. keep = near speech that must survive (preservation ~0 dB); suppress = >1 m voice that must be attenuated. Tags matching neither are omitted." };
    list.forEach((file) => {
      const keep = [];
      const suppress = [];
      file.spans.forEach((span) => {
        const role = windowsRole(span.tag);
        if (role === "keep") keep.push([r3(span.a), r3(span.b)]);
        else if (role === "suppress") suppress.push([r3(span.a), r3(span.b)]);
      });
      if (!keep.length && !suppress.length) return;
      const clip = { role: "annotated in Span" };
      if (keep.length) clip.keep = keep;
      if (suppress.length) clip.suppress = suppress;
      out[base(file.name)] = clip;
    });
    return JSON.stringify(out, null, 2);
  }

  function exportName(format, { files = [], current = null, folderName = "" } = {}) {
    const extension = format === "csv" ? "csv" : format === "audacity" ? "txt" : "json";
    const name = current?.name || "";
    if (format === "windows") return "windows.json";
    if (format === "audacity") return `${base(name || "audio")}.txt`;
    if (files.length > 1) return folderName ? `${folderName}.spans.${extension}` : `spans.${extension}`;
    return `${base(name || "audio")}.spans.${extension}`;
  }

  /* Who a key press is for: nobody while typing in a field, the annotator on
   * its own screen, the player anywhere else. */
  function shortcutTarget(event, screen) {
    const target = event && event.target;
    if (target) {
      const tag = String(target.tagName || "").toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable) return null;
    }
    return screen === "annotate" ? "annotate" : "deck";
  }

  return { base, splitCsv, normSpan, parseSpans, windowsRole, buildExport, exportName, singleFileJson, shortcutTarget, AUDIO_RE: /\.(wav|wave|flac|mp3|ogg|oga|opus|m4a|mp4|aac|aif|aiff|aifc|caf|webm)$/i, SPAN_FILE_RE: /\.(json|csv|txt|tsv)$/i };
});
