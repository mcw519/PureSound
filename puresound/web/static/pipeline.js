/* Pipeline screen: build one training row from chosen audio with a released
 * recipe's knobs, then read it stage by stage -- what each stage drew, the
 * effective SNR it left the model, how a released model scores the pair at
 * that point, and the room the row was rendered in.  The numbers come from a
 * "pipeline_trace" job; everything here is presentation.  Pure logic lives in
 * pipeline-model.js; the 3-D room in pipeline-room.js (optional). */
(() => {
  "use strict";

  const M = window.PureSoundPipelineModel;
  const A = window.PureSoundAudio;
  const FORM_KEY = "puresound.pipeline.form";
  const POLL_MS = 400;
  const SVG = "http://www.w3.org/2000/svg";
  const ROLE_COLORS = { foreground: "#fc4c02", interferer: "#ef2cc1", media: "#8f8cff", echo: "#e0a800", noise: "#b0782b", source: "#6fd3db" };
  const ROLE_LABELS = { foreground: "foreground", interferer: "interferer", media: "media source", echo: "residual echo", noise: "noise, through the room", source: "other" };
  const roleLabel = (role) => (ROLE_LABELS[role] ? (window.PureSoundI18n ? window.PureSoundI18n.t(ROLE_LABELS[role]) : ROLE_LABELS[role]) : role);
  const METRICS = [
    { key: "si_sdr_db", label: "SI-SDR", unit: " dB", domain: { floor: -30, ceil: 60, pad: 3, minSpan: 15 }, cap: 100 },
    { key: "stoi", label: "STOI", unit: "", digits: 2, domain: { floor: 0, ceil: 1, pad: 0.05, minSpan: 0.2 } },
    { key: "pesq_wb", label: "PESQ", unit: "", digits: 2, domain: { floor: 1, ceil: 4.64, pad: 0.1, minSpan: 0.6 } },
    { key: "level_change_db", label: "Level Δ", unit: " dB", domain: { floor: -60, ceil: 12, pad: 2, minSpan: 10 } },
  ];

  const state = {
    loaded: false,
    loading: null,
    overview: null,
    catalog: null,
    models: [],
    lang: "en",
    files: { foreground: [], talkers: [], noises: [], rirs: [] },
    report: null,
    rows: [],
    selected: null,
    metric: "si_sdr_db",
    jobId: null,
    running: false,
    deck: null,
    room: null,
    roomMode: "3d",
    preview: null,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  const app = () => window.PureSoundApp;
  const shell = () => window.PureSoundShell;
  const t = (key, vars) => (window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);
  // A sentence with markup in it: the words are translated and escaped, the
  // values (already markup) go in afterwards.
  const html = (key, vars = {}) => esc(t(key)).replace(/\{(\w+)\}/g, (match, name) => (name in vars ? String(vars[name]) : match));
  const stateText = (row) => (row.state === "off-here" && row.offReason ? `${t("off in the inspector")}: ${row.offReason}` : row.state === "skipped" && row.block && Number.isFinite(row.recipe?.prob) ? t("did not fire on this row · p = {p}", { p: M.formatValue(row.recipe.prob) }) : t(M.stateLabel(row)));
  // Stage explanations follow the page's language ("zh" entries in pipeline-stages.json).
  const pageLang = () => (window.PureSoundI18n?.lang === "zh-TW" ? "zh" : "en");
  const toast = (message, isError = false) => app()?.showToast(message, isError);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const store = {
    get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be disabled */ } },
  };

  // ----------------------------------------------------------------- loading

  async function show() {
    if (state.loaded) { state.deck?.draw?.(); state.room?.resize?.(); return; }
    if (!state.loading) state.loading = load().catch((error) => { state.loading = null; renderUnavailable(error.message); });
    await state.loading;
  }

  async function load() {
    const api = app().api;
    const [overview, catalog, models] = await Promise.all([
      api("/api/pipeline"),
      fetch("/pipeline-stages.json").then((response) => response.json()),
      api("/api/models").then((payload) => payload.models || []).catch(() => []),
    ]);
    state.overview = overview;
    state.catalog = catalog;
    state.models = models.filter((model) => model.runnable !== false && ["voice_isolation", "noise_suppression"].includes(model.task));
    state.lang = pageLang();
    state.loaded = true;
    if (!overview.available) { renderUnavailable(overview.reason); return; }
    renderSetup();
    applyLang();
  }

  function renderUnavailable(reason) {
    const box = $("#pipe-unavailable");
    box.hidden = false;
    box.innerHTML = `<span class="note-mark">i</span><div><strong>${esc(t("The pipeline inspector is not available on this server."))}</strong><p>${esc(reason || t("Unknown reason."))}</p><p>${html("Start {command} from a PureSound checkout, or pass {option}.", { command: "<code>puresound web</code>", option: "<code>--pipeline-root /path/to/PureSound</code>" })}</p></div>`;
    $("#pipe-layout").hidden = true;
  }

  // ------------------------------------------------------------------- setup

  function renderSetup() {
    const { recipes, samples } = state.overview;
    const saved = store.get(FORM_KEY, {});
    const byTask = { noise_suppression: [], voice_isolation: [] };
    recipes.forEach((recipe) => (byTask[recipe.task] || (byTask[recipe.task] = [])).push(recipe));
    const labels = { noise_suppression: t("Noise suppression"), voice_isolation: t("Voice isolation") };
    $("#pipe-recipe").innerHTML = Object.entries(byTask).filter(([, items]) => items.length).map(([task, items]) => `
      <optgroup label="${esc(labels[task] || task)}">${items.map((recipe) => `<option value="${esc(recipe.id)}">${esc(recipe.name)}${recipe.curriculum ? ` · ${esc(t("curriculum"))}` : ""}</option>`).join("")}</optgroup>`).join("");
    const preferred = recipes.find((recipe) => recipe.id === saved.recipe) || recipes.find((recipe) => /_s1$/.test(recipe.name)) || recipes.find((recipe) => recipe.name.startsWith("train_")) || recipes[0];
    if (preferred) $("#pipe-recipe").value = preferred.id;
    if (Number.isInteger(saved.seed)) $("#pipe-seed").value = saved.seed;

    $("#pipe-foreground-sample").innerHTML = samples.talkers.map((talker) => `<option value="${esc(talker.id)}">${esc(talker.title)} — ${esc(talker.description)}</option>`).join("");
    $("#pipe-talkers").innerHTML = samples.talkers.map((talker, index) => sampleCheck("talker", talker, index === samples.talkers.length - 1, talker.files[0])).join("");
    $("#pipe-noises").innerHTML = samples.noises.map((noise) => sampleCheck("noise", noise, true, noise.file)).join("");
    $("#pipe-rooms").innerHTML = samples.rooms.map((room) => sampleCheck("room", { ...room, description: t("{sources} sources · {obstacles} obstacles", { sources: room.sources, obstacles: room.obstacles }) }, true, null)).join("");
    syncForegroundTalkers();
    onRecipeChange();
    renderModelOptions();
    renderFiles();
  }

  function sampleCheck(kind, item, checked, previewFile) {
    const preview = previewFile ? `<button class="pipe-play" type="button" data-preview="/samples/pipeline/${esc(previewFile)}" title="${esc(t("Listen"))}" aria-label="${esc(t("Listen to {name}", { name: item.title }))}">▶</button>` : "";
    return `<label class="pipe-check" title="${esc(item.description || "")}"><input type="checkbox" data-kind="${kind}" value="${esc(item.id)}"${checked ? " checked" : ""} /><span>${esc(item.title)}</span>${preview}</label>`;
  }

  function currentRecipe() {
    return state.overview.recipes.find((recipe) => recipe.id === $("#pipe-recipe").value) || null;
  }

  function onRecipeChange() {
    const recipe = currentRecipe();
    if (!recipe) return;
    const seconds = Number(recipe.row_seconds);
    if (Number.isFinite(seconds)) $("#pipe-seconds").value = Math.min(20, Math.max(1, seconds));
    renderRecipeHint(recipe);
    syncEpoch();
    renderModelOptions();
  }

  function renderRecipeHint(recipe) {
    const taskLabel = t(recipe.task === "voice_isolation" ? "Voice isolation: other talkers are suppressed." : "Noise suppression: every voice is kept, only noise is removed.");
    $("#pipe-recipe-hint").textContent = `${taskLabel}${recipe.curriculum ? ` ${t("Its knobs move with the epoch.")}` : ""}`;
  }

  function syncEpoch() {
    const recipe = currentRecipe();
    $("#pipe-epoch-field").hidden = !(recipe?.curriculum && $("#pipe-role").value === "train");
  }

  function renderModelOptions() {
    const recipe = currentRecipe();
    const models = state.models.filter((model) => !recipe || model.task === recipe.task);
    const fallback = models.find((model) => model.roles?.includes("default")) || models[0];
    const current = $("#pipe-model").value;
    $("#pipe-model").innerHTML = `<option value="">${esc(t("None — report the stages only"))}</option>` + models.map((model) => `<option value="${esc(model.id)}">${esc(model.display_name || model.id)}${model.roles?.includes("default") ? ` · ${esc(t("default"))}` : ""}</option>`).join("");
    $("#pipe-model").value = models.some((model) => model.id === current) ? current : (fallback?.id || "");
  }

  /* The sample chosen as the foreground cannot also be another talker. */
  function syncForegroundTalkers() {
    const foreground = $("#pipe-foreground-sample").value;
    const usingUpload = state.files.foreground.length > 0;
    $$('#pipe-talkers input[type="checkbox"]').forEach((input) => {
      const clash = !usingUpload && input.value === foreground;
      input.disabled = clash;
      if (clash) input.checked = false;
      input.closest(".pipe-check").classList.toggle("is-disabled", clash);
    });
    $("#pipe-foreground-sample").disabled = usingUpload;
  }

  function renderFiles() {
    $$("[data-files]").forEach((host) => {
      const kind = host.dataset.files;
      host.innerHTML = state.files[kind].map((file, index) => `<span class="pipe-file-chip"><span>${esc(file.name)}</span><button type="button" data-remove-file="${kind}" data-index="${index}" aria-label="Remove ${esc(file.name)}">×</button></span>`).join("");
    });
  }

  function addFiles(kind, fileList, { single = false } = {}) {
    const files = [...(fileList || [])];
    if (!files.length) return;
    state.files[kind] = single ? files.slice(0, 1) : [...state.files[kind], ...files];
    renderFiles();
    syncForegroundTalkers();
  }

  function rirKind() {
    return $('input[name="pipe-rir"]:checked')?.value || "samples";
  }

  function checked(kind) {
    return $$(`input[data-kind="${kind}"]:checked`).map((input) => input.value);
  }

  async function buildPayload() {
    const upload = (file) => app().uploadDescriptor(file).then((descriptor) => ({ upload_id: descriptor.upload_id }));
    const recipe = currentRecipe();
    if (!recipe) throw new Error(t("Choose a training recipe."));
    const seed = Number($("#pipe-seed").value);
    if (!Number.isInteger(seed) || seed < 0) throw new Error(t("The seed must be a whole number, 0 or more."));
    const seconds = Number($("#pipe-seconds").value);
    if (!(seconds >= 1 && seconds <= 20)) throw new Error(t("The row length must be between 1 and 20 s."));
    const foreground = state.files.foreground.length ? await upload(state.files.foreground[0]) : { sample: $("#pipe-foreground-sample").value };
    const talkers = [...checked("talker").map((id) => ({ sample: id })), ...(await Promise.all(state.files.talkers.map(upload)))];
    const noises = [...checked("noise").map((id) => ({ sample: id })), ...(await Promise.all(state.files.noises.map(upload)))];
    const kind = rirKind();
    let rir = { kind: "none" };
    if (kind === "samples") rir = { kind, rooms: checked("room") };
    if (kind === "upload") rir = { kind, files: await Promise.all(state.files.rirs.map(upload)) };
    if (kind === "samples" && !rir.rooms.length) throw new Error(t("Pick at least one sample room, or choose None."));
    if (kind === "upload" && !rir.files.length) throw new Error(t("Add an impulse response, or choose another room option."));
    const payload = {
      kind: "pipeline_trace",
      recipe: recipe.id,
      role: $("#pipe-role").value,
      seed,
      seconds,
      foreground,
      talkers,
      noises,
      rir,
      model_id: $("#pipe-model").value || null,
      provider: "cpu",
    };
    if (!$("#pipe-epoch-field").hidden) payload.epoch = Math.max(0, Math.round(Number($("#pipe-epoch").value) || 0));
    store.set(FORM_KEY, { recipe: recipe.id, seed });
    return payload;
  }

  // --------------------------------------------------------------------- run

  function setRunning(running) {
    state.running = running;
    $("#pipe-run").classList.toggle("is-loading", running);
    $("#pipe-run").disabled = running;
    $("#pipe-cancel").hidden = !running;
  }

  function setProgress(value, phase) {
    const label = String(phase || t("Working")).replace(/:/g, " · ").replace(/_/g, " ");
    shell()?.setState("pipeline", { state: "run", label, progress: Math.max(0, Math.min(1, value || 0)) });
  }

  async function run() {
    if (state.running || !state.loaded || !state.overview?.available) return;
    const note = $("#pipe-note");
    note.className = "form-note";
    note.textContent = "";
    setRunning(true);
    setProgress(0.01, t("Uploading"));
    try {
      const payload = await buildPayload();
      let job = await app().api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      state.jobId = job.job_id;
      while (job.status === "queued" || job.status === "running") {
        setProgress(job.progress, job.phase);
        await sleep(POLL_MS);
        job = await app().api(`/api/jobs/${encodeURIComponent(job.job_id)}`);
      }
      if (job.status === "cancelled") { shell()?.setState("pipeline", { state: "idle" }); note.textContent = t("Cancelled."); return; }
      if (job.status !== "succeeded") throw new Error(job.error || t("The trace failed."));
      renderReport(job.result);
      shell()?.closeInspector();
      const seconds = job.result.timing?.total_seconds;
      shell()?.setState("pipeline", { state: "ok", label: Number.isFinite(seconds) ? t("Row built in {seconds} s.", { seconds: seconds.toFixed(1) }) : t("Row built.") });
    } catch (error) {
      shell()?.setState("pipeline", { state: "err", label: error.message });
      note.className = "form-note is-error";
      note.textContent = error.message;
    } finally {
      state.jobId = null;
      setRunning(false);
    }
  }

  async function cancel() {
    if (!state.jobId) return;
    try { await app().api(`/api/jobs/${encodeURIComponent(state.jobId)}/cancel`, { method: "POST", body: "{}" }); } catch { /* the poll reports the outcome */ }
  }

  // ------------------------------------------------------------------ report

  function renderReport(report) {
    state.report = report;
    state.rows = M.stageRows(report, state.catalog, state.lang);
    const emitted = state.rows.find((row) => row.id === "row.emit");
    state.metric = emitted?.metrics?.esnr_state === "no_target" ? "level_change_db" : (state.metric === "level_change_db" ? "si_sdr_db" : state.metric);
    const series = M.esnrSeries(state.rows);
    const firstChange = state.rows.find((row) => row.fired && row.id !== "source.load" && row.id !== "row.plan" && row.changed);
    state.selected = M.biggestDrop(series) || firstChange?.id || "row.emit";
    $("#pipe-empty").hidden = true;
    $("#pipe-result").hidden = false;
    renderSummary();
    renderRail();
    renderMetricButtons();
    renderCharts();
    renderRoom();
    select(state.selected, { scroll: false });
  }

  function renderSummary() {
    const report = state.report;
    const fired = state.rows.filter((row) => row.fired).length;
    const model = report.model;
    const recipe = state.overview.recipes.find((item) => item.id === report.recipe);
    const chips = [
      [t("Task"), t(report.task === "voice_isolation" ? "Voice isolation" : "Noise suppression")],
      [t("Recipe"), recipe?.name || report.recipe],
      [t("Rows"), t(report.role === "validation" ? "validation" : "training")],
      [t("Seed"), report.seed],
      ...(report.curriculum ? [[t("Epoch"), report.epoch ?? 0]] : []),
      [t("Length"), `${Number(report.row_seconds).toFixed(2)} s`],
      [t("Stages"), t("{fired} of {total} acted", { fired, total: state.rows.length })],
      [t("Other talkers"), report.talkers],
      [t("Model"), model ? (model.skipped ? `${model.display_name || model.id} · ${t("not run")}` : model.display_name || model.id) : t("none")],
    ];
    const notes = report.notes || [];
    $("#pipe-summary").innerHTML = `
      <p class="pipe-story">${storyLine()}</p>
      <div class="pipe-summary-chips">${chips.map(([label, value]) => `<div class="pipe-chip"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}</div>
      ${model?.skipped ? `<p class="pipe-warning">${esc(model.skipped)}</p>` : ""}
      ${notes.length ? `<details class="pipe-notes"${notes.length <= 3 ? " open" : ""}><summary>${esc(t("What the inspector changed in this recipe"))} <span>${notes.length}</span></summary><ul>${notes.map((note) => `<li><code>${esc(note.subject)}</code> ${esc(note.reason)}</li>`).join("")}</ul></details>` : `<p class="pipe-quiet">${esc(t("The recipe ran as written; only its corpus paths point at your audio."))}</p>`}`;
  }

  /* One sentence to start reading from: where the row ends up, which stage cost
   * the most, and what the model makes of the training pair. */
  function storyLine() {
    const series = M.esnrSeries(state.rows);
    const emitted = state.rows.find((row) => row.id === "row.emit");
    const parts = [];
    const final = emitted?.metrics;
    if (final?.esnr_state === "no_target") parts.push(html("This row has {none}: the right output is silence.", { none: `<strong>${esc(t("no target"))}</strong>` }));
    else if (Number.isFinite(final?.esnr_db)) parts.push(html("The training pair leaves the model {snr} of effective SNR to work from.", { snr: `<strong>${esc(M.formatNumber(final.esnr_db, { unit: " dB" }))}</strong>` }));
    const worst = M.biggestDrop(series);
    if (worst) {
      const row = state.rows.find((item) => item.id === worst);
      const point = series.find((item) => item.id === worst);
      parts.push(html("{stage} cost the most ({delta}).", { stage: `<strong>${esc(row.title)}</strong>`, delta: esc(M.formatSigned(point.delta, { unit: " dB" })) }));
    }
    const model = emitted?.model;
    const name = state.report.model?.display_name || state.report.model?.id;
    if (model && final?.esnr_state === "no_target" && Number.isFinite(model.level_change_db)) parts.push(html("{model} lowers it by {level}.", { model: esc(name), level: `<strong>${esc(M.formatNumber(-model.level_change_db, { unit: " dB" }))}</strong>` }));
    else if (model && Number.isFinite(model.input?.si_sdr_db) && Number.isFinite(model.output?.si_sdr_db)) parts.push(html("{model} takes its SI-SDR from {before} to {after}.", { model: esc(name), before: esc(M.formatNumber(model.input.si_sdr_db, { unit: " dB" })), after: `<strong>${esc(M.formatNumber(model.output.si_sdr_db, { unit: " dB" }))}</strong>` }));
    return parts.join(" ");
  }

  function renderRail() {
    const rail = $("#pipe-rail");
    const series = M.esnrSeries(state.rows);
    const deltas = new Map(series.map((point) => [point.id, point.delta]));
    // The first stage that makes the pair differ has no change to report
    // (it starts from "the same signal"), so it shows where it lands instead.
    const landing = new Map(series.filter((point, index) => index > 0 && point.value !== null && series[index - 1].state === "identical").map((point) => [point.id, point.value]));
    let group = null;
    const parts = [];
    state.rows.forEach((row) => {
      if (row.group !== group) {
        if (group !== null) parts.push("</div></div>");
        group = row.group;
        parts.push(`<div class="pipe-group" style="--group:${esc(row.groupInfo.color)}"><span class="pipe-group-title">${esc(row.groupInfo.title)}</span><div class="pipe-group-stages">`);
      }
      const change = deltas.get(row.id);
      let delta = Number.isFinite(change) && Math.abs(change) >= 0.3 ? `<span class="pipe-stage-delta ${change < 0 ? "is-down" : "is-up"}" title="${esc(t("effective SNR change"))}">${esc(M.formatSigned(change))}</span>` : "";
      if (landing.has(row.id)) delta = `<span class="pipe-stage-delta is-down" title="${esc(t("effective SNR from here on (the pair was the same signal before)"))}">→ ${esc(M.formatNumber(landing.get(row.id)))}</span>`;
      parts.push(`<button class="pipe-stage is-${row.state}" type="button" role="tab" data-stage="${esc(row.id)}" aria-selected="false" title="${esc(stateText(row))}"><span class="pipe-stage-index">${String(row.index + 1).padStart(2, "0")}</span><span class="pipe-stage-title">${esc(row.title)}</span>${delta}</button>`);
    });
    if (group !== null) parts.push("</div></div>");
    rail.innerHTML = parts.join("");
  }

  function renderMetricButtons() {
    $("#pipe-metric").innerHTML = METRICS.map((metric) => `<button type="button" data-metric="${metric.key}" aria-pressed="${metric.key === state.metric}">${esc(metric.label)}</button>`).join("");
  }

  // ------------------------------------------------------------------ charts

  /* An SVG element; ``children`` is markup, so text goes through text() or tip(). */
  function svg(tag, attributes = {}, children = "") {
    const attrs = Object.entries(attributes).filter(([, value]) => value !== null && value !== undefined).map(([name, value]) => `${name}="${esc(value)}"`).join(" ");
    return `<${tag} ${attrs}>${children}</${tag}>`;
  }

  const text = (attributes, value) => svg("text", attributes, esc(value));
  const tip = (value) => svg("title", {}, esc(value));

  function ticks([low, high], count = 5) {
    const raw = (high - low) / count;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((candidate) => candidate >= raw) || raw;
    const values = [];
    for (let value = Math.ceil(low / step) * step; value <= high + 1e-9; value += step) values.push(Number(value.toFixed(6)));
    return values;
  }

  /* The stages both charts share an x axis over: every stage that acted. */
  function chartStages() {
    return state.rows.filter((row) => row.fired && row.metrics);
  }

  function chartFrame(host, domain, { height = 230, unit = "", digits = 0, labels = true } = {}) {
    const stages = chartStages();
    const width = Math.max(280, Math.round(host.clientWidth || 760));
    const margin = { left: labels ? 112 : 64, right: 24, top: 26, bottom: labels ? 92 : 16 };
    const inner = { width: width - margin.left - margin.right, height: height - margin.top - margin.bottom };
    const position = new Map(stages.map((row, index) => [row.id, index]));
    const x = (id) => margin.left + (stages.length > 1 ? (position.get(id) / (stages.length - 1)) * inner.width : inner.width / 2);
    const y = (value) => margin.top + ((domain[1] - Math.min(Math.max(value, domain[0]), domain[1])) / (domain[1] - domain[0] || 1)) * inner.height;
    const grid = ticks(domain).map((value) => svg("g", { class: "pipe-grid" }, svg("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: value === 0 ? "is-zero" : null }) + text({ x: margin.left - 8, y: y(value) + 3, "text-anchor": "end" }, `${M.formatNumber(value, { digits })}${unit}`))).join("");
    const columns = stages.map((row) => svg("line", { x1: x(row.id), x2: x(row.id), y1: margin.top, y2: height - margin.bottom, class: row.id === state.selected ? "pipe-guide" : "pipe-column" })).join("");
    const axis = labels ? stages.map((row) => svg("g", { class: `pipe-xlabel${row.id === state.selected ? " is-selected" : ""}`, transform: `translate(${x(row.id)},${height - margin.bottom + 14}) rotate(-35)` }, text({ "text-anchor": "end" }, `${String(row.index + 1).padStart(2, "0")} ${row.title.length > 22 ? `${row.title.slice(0, 21)}…` : row.title}`))).join("") : "";
    const hit = (row, caption) => svg("rect", { x: x(row.id) - 16, y: margin.top - 8, width: 32, height: inner.height + 16, class: "pipe-hit" }) + tip(caption);
    return { x, y, margin, width, height, grid: grid + columns, axis, hit, top: margin.top, bottom: height - margin.bottom };
  }

  function linePath(entries, x, y) {
    let path = "";
    let open = false;
    entries.forEach(([id, value]) => {
      if (!Number.isFinite(value)) { open = false; return; }
      path += `${open ? "L" : "M"}${x(id).toFixed(1)},${y(value).toFixed(1)}`;
      open = true;
    });
    return path;
  }

  function renderCharts() {
    renderEsnrChart();
    renderModelChart();
  }

  function hasModelChart() {
    return Boolean(state.report?.model && !state.report.model.skipped);
  }

  function renderEsnrChart() {
    const host = $("#pipe-esnr-chart");
    const points = M.esnrSeries(state.rows);
    if (!points.length) { host.innerHTML = `<p class="pipe-quiet">${esc(t("No stage acted on this row."))}</p>`; return; }
    const domain = M.chartDomain(points.map((point) => point.value));
    const frame = chartFrame(host, domain, { unit: " dB", labels: !hasModelChart(), height: hasModelChart() ? 200 : 260 });
    const { x, y } = frame;
    const marks = points.map((point, index) => {
      const row = state.rows.find((item) => item.id === point.id);
      const selected = point.id === state.selected;
      let mark;
      let caption;
      if (point.value === null) {
        const same = point.state === "identical";
        const cy = same ? frame.top + 2 : frame.bottom - 2;
        // One caption per run of equal marks, or neighbours overprint each other.
        const first = index === 0 || points[index - 1].value !== null || points[index - 1].state !== point.state;
        mark = svg("circle", { cx: x(point.id), cy, r: selected ? 7 : 5.5, class: "pipe-hollow", stroke: row.groupInfo.color }) + (first ? text({ x: x(point.id) - 6, y: same ? cy - 11 : cy + 17, "text-anchor": "start", class: "pipe-cap" }, same ? `∞ ${t("same signal")}` : t("no target")) : "");
        caption = `${row.title}: ${t(same ? "the mixture is still the target" : "no target on this row")}`;
      } else {
        const delta = Number.isFinite(point.delta) && Math.abs(point.delta) >= 0.3 ? text({ x: x(point.id) + 8, y: y(point.value) + (point.delta < 0 ? 17 : -10), class: `pipe-delta ${point.delta < 0 ? "is-down" : "is-up"}` }, `Δ ${M.formatSigned(point.delta, { unit: " dB" })}`) : "";
        const value = selected ? text({ x: x(point.id) - 9, y: y(point.value) - 10, "text-anchor": "end", class: "pipe-value" }, M.formatNumber(point.value, { unit: " dB" })) : "";
        mark = svg("circle", { cx: x(point.id), cy: y(point.value), r: selected ? 7 : 5, fill: row.groupInfo.color, class: "pipe-dot" }) + delta + value;
        caption = `${row.title}: ${t("effective SNR {value}", { value: M.formatNumber(point.value, { unit: " dB" }) })}${Number.isFinite(point.delta) ? ` (${M.formatSigned(point.delta, { unit: " dB" })})` : ""}`;
      }
      return svg("g", { class: `pipe-point${selected ? " is-selected" : ""}`, "data-stage": point.id, tabindex: 0 }, mark + frame.hit(row, caption));
    }).join("");
    const line = svg("path", { d: linePath(points.map((point) => [point.id, point.value]), x, y), class: "pipe-line" });
    host.innerHTML = `<svg class="pipe-svg" width="${frame.width}" height="${frame.height}" viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="${esc(t("Effective SNR after each stage"))}">${frame.grid}${line}${marks}${frame.axis}</svg>`;
  }

  function renderModelChart() {
    const host = $("#pipe-model-chart");
    const report = state.report;
    $("#pipe-metric").hidden = !hasModelChart();
    if (!report.model) { host.innerHTML = `<p class="pipe-quiet">${html("Choose a model under {field} to see how it does on the pair at each stage.", { field: `<strong>${esc(t("Score every stage with"))}</strong>` })}</p>`; return; }
    if (report.model.skipped) { host.innerHTML = `<p class="pipe-quiet">${esc(report.model.skipped)}</p>`; return; }
    const metric = METRICS.find((item) => item.key === state.metric) || METRICS[0];
    const points = M.modelSeries(state.rows, metric.key);
    const capped = (value) => Boolean(metric.cap) && Number.isFinite(value) && value >= metric.cap;
    const plotted = (value) => (capped(value) ? null : value);
    if (!points.some((point) => Number.isFinite(point.output))) {
      const absent = state.rows.find((row) => row.id === "row.emit")?.metrics?.esnr_state === "no_target";
      host.innerHTML = `<p class="pipe-quiet">${esc(t(absent ? "{metric} is undefined on this row: there is no target. Level Δ shows how far the model lowers the mixture." : "{metric} is undefined on this row.", { metric: metric.label }))}</p>`;
      return;
    }
    const level = metric.key === "level_change_db";
    const values = level ? points.map((point) => point.output) : [...points.map((point) => plotted(point.input)), ...points.map((point) => plotted(point.output))];
    const domain = M.chartDomain(values, metric.domain);
    const frame = chartFrame(host, domain, { unit: metric.unit, digits: metric.digits ? 1 : 0, height: 270 });
    const { x, y } = frame;
    const format = (value) => M.formatNumber(value, { digits: metric.digits || 1, unit: metric.unit, cap: metric.cap || null });
    const marks = points.map((point) => {
      const row = state.rows.find((item) => item.id === point.id);
      const selected = point.id === state.selected;
      let body = "";
      if (!level && Number.isFinite(point.input)) {
        if (capped(point.input)) body += text({ x: x(point.id), y: frame.top - 8, "text-anchor": "middle", class: "pipe-cap" }, "∞");
        else body += svg("circle", { cx: x(point.id), cy: y(point.input), r: 4, class: "pipe-input-dot" });
        if (Number.isFinite(point.output) && !capped(point.input)) body = svg("line", { x1: x(point.id), x2: x(point.id), y1: y(point.input), y2: y(point.output), class: `pipe-gain ${point.output >= point.input ? "is-up" : "is-down"}` }) + body;
      }
      if (Number.isFinite(point.output)) {
        body += svg("circle", { cx: x(point.id), cy: y(plotted(point.output) ?? domain[1]), r: selected ? 7 : 5, class: "pipe-output-dot" });
        if (selected) body += text({ x: x(point.id) - 9, y: y(plotted(point.output) ?? domain[1]) - 10, "text-anchor": "end", class: "pipe-value" }, level ? M.formatSigned(point.output, { unit: " dB" }) : format(point.output));
      }
      const caption = level
        ? `${row.title}: ${t("the model's output is {level} against its input", { level: M.formatSigned(point.output, { unit: " dB" }) })}`
        : `${row.title}: ${t("the pair {input} → the model {output}", { input: format(point.input), output: format(point.output) })}${point.reused ? ` · ${t("same input as {stage}", { stage: state.rows.find((item) => item.id === point.reused)?.title || point.reused })}` : ""}`;
      return svg("g", { class: `pipe-point${selected ? " is-selected" : ""}`, "data-stage": point.id, tabindex: 0 }, body + frame.hit(row, caption));
    }).join("");
    const lines = level
      ? svg("path", { d: linePath(points.map((point) => [point.id, point.output]), x, y), class: "pipe-line is-output" })
      : svg("path", { d: linePath(points.map((point) => [point.id, plotted(point.input)]), x, y), class: "pipe-line is-input" }) + svg("path", { d: linePath(points.map((point) => [point.id, plotted(point.output)]), x, y), class: "pipe-line is-output" });
    const legend = level
      ? `<div class="pipe-legend"><span><i class="is-output"></i>${esc(t("the model's output level against its input"))}</span></div>`
      : `<div class="pipe-legend"><span><i class="is-input"></i>${esc(t("the pair as it stands (mixture vs target)"))}</span><span><i class="is-output"></i>${esc(t("model output vs target"))}</span><span><i class="is-gain"></i>${esc(t("what the model adds"))}</span></div>`;
    host.innerHTML = `${legend}<svg class="pipe-svg" width="${frame.width}" height="${frame.height}" viewBox="0 0 ${frame.width} ${frame.height}" role="img" aria-label="${esc(t("Model score at each stage"))}">${frame.grid}${lines}${marks}${frame.axis}</svg>`;
  }

  // ------------------------------------------------------------------ detail

  function select(id, { scroll = true } = {}) {
    const row = state.rows.find((item) => item.id === id);
    if (!row) return;
    state.selected = id;
    $$("#pipe-rail .pipe-stage").forEach((button) => {
      const on = button.dataset.stage === id;
      button.classList.toggle("is-selected", on);
      button.setAttribute("aria-selected", on ? "true" : "false");
    });
    renderCharts();
    renderDetail(row);
    loadDeck(row);
    highlightRoom(row);
    if (scroll) $("#pipe-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function step(direction) {
    const fired = state.rows.filter((row) => row.fired);
    const index = fired.findIndex((row) => row.id === state.selected);
    const next = fired[Math.min(fired.length - 1, Math.max(0, (index < 0 ? 0 : index) + direction))];
    if (next) {
      select(next.id, { scroll: false });
      $(`#pipe-rail [data-stage="${CSS.escape(next.id)}"]`)?.focus();
    }
  }

  function renderDetail(row) {
    const entry = row.entry || {};
    const lang = state.lang;
    const recipe = row.recipe || {};
    const recipeText = !row.block ? t("always runs") : !recipe.configured ? t("not in this recipe") : `${t("on")}${Number.isFinite(recipe.prob) ? ` · p = ${M.formatValue(recipe.prob)}` : ""}`;
    $("#pipe-detail-head").innerHTML = `
      <div><p class="eyebrow" style="color:${esc(row.groupInfo.color)}">${esc(row.groupInfo.title)} · ${esc(t("stage {n} of {total}", { n: String(row.index + 1).padStart(2, "0"), total: state.rows.length }))}</p><h2>${esc(row.title)}</h2></div>
      <div class="pipe-detail-state"><span class="pipe-badge is-${row.state}">${esc(stateText(row))}</span><span class="pipe-quiet">${esc(t("Recipe: {state}", { state: recipeText }))}</span></div>
      <div class="pipe-detail-nav"><button class="button button-icon" type="button" data-step="-1" aria-label="${esc(t("Previous stage"))}">‹</button><button class="button button-icon" type="button" data-step="1" aria-label="${esc(t("Next stage"))}">›</button></div>`;
    $("#pipe-explain").innerHTML = `
      <p class="eyebrow">${esc(t("What it does"))}</p><p>${esc(M.text(entry.what, lang))}</p>
      <p class="eyebrow">${esc(t("What it means for the model"))}</p><p>${esc(M.text(entry.model, lang))}</p>
      ${entry.doc ? `<p class="pipe-doc">${esc(t("More"))}: <code>${esc(entry.doc)}</code></p>` : ""}`;
    $("#pipe-tables").innerHTML = row.fired ? [paramsTable(row), pairTable(row), modelTable(row)].join("") : `<p class="pipe-quiet">${esc(stateText(row))}. ${esc(t("Nothing to measure: the pair passes through unchanged."))}</p>`;
    $("#pipe-listen-hint")?.remove();
  }

  function paramsTable(row) {
    const rows = M.flattenParams(row.params);
    const body = rows.length ? rows.map(([name, value]) => `<tr><th>${esc(name)}</th><td>${esc(value)}</td></tr>`).join("") : `<tr><td colspan="2" class="pipe-quiet">${esc(t("Nothing drawn: this stage has no random choice."))}</td></tr>`;
    return `<div class="pipe-table-block"><p class="eyebrow">${esc(t("Drawn on this row"))}</p><table class="pipe-table">${body}</table></div>`;
  }

  function pairTable(row) {
    const before = M.previousFired(state.rows, row.id)?.metrics || null;
    const after = row.metrics || {};
    const lines = [
      [t("Mixture RMS"), "noisy_rms_dbfs", " dBFS"],
      [t("Mixture peak"), "noisy_peak_dbfs", " dBFS"],
      [t("Target RMS"), "target_rms_dbfs", " dBFS"],
      [t("Everything else"), "residual_rms_dbfs", " dBFS"],
      [t("Effective SNR"), "esnr_db", " dB"],
      [t("Input SI-SDR"), "input_si_sdr_db", " dB"],
    ];
    const describe = (metrics, key, unit) => {
      if (!metrics) return "—";
      if (key === "esnr_db" && metrics.esnr_state === "identical") return `∞ (${t("same signal")})`;
      if ((key === "esnr_db" || key === "input_si_sdr_db") && metrics.esnr_state === "no_target") return t("no target");
      return M.formatNumber(metrics[key], { unit });
    };
    const body = lines.map(([label, key, unit]) => {
      const change = before && Number.isFinite(before[key]) && Number.isFinite(after[key]) ? after[key] - before[key] : null;
      const cls = key === "esnr_db" && Number.isFinite(change) ? (change < -0.05 ? "is-down" : change > 0.05 ? "is-up" : "") : "";
      return `<tr><th>${esc(label)}</th><td>${esc(describe(before, key, unit))}</td><td>${esc(describe(after, key, unit))}</td><td class="${cls}">${esc(M.formatSigned(change))}</td></tr>`;
    }).join("");
    const peak = after.noisy_peak_dbfs;
    const hot = Number.isFinite(peak) && peak > 0 ? `<p class="pipe-quiet">${esc(t("The mixture peaks above full scale here: before the converter a level is sound pressure, not samples. The A/D stage brings the pair back under 1.0."))}</p>` : "";
    return `<div class="pipe-table-block"><p class="eyebrow">${esc(t("The pair · before → after"))}</p><table class="pipe-table is-numeric"><thead><tr><th></th><th>${esc(t("Before"))}</th><th>${esc(t("After"))}</th><th>${esc(t("Change"))}</th></tr></thead><tbody>${body}</tbody></table>${hot}</div>`;
  }

  function modelTable(row) {
    const report = state.report;
    if (!report.model || report.model.skipped) return "";
    const model = row.model;
    if (!model) return `<div class="pipe-table-block"><p class="eyebrow">${esc(t("Model"))}</p><p class="pipe-quiet">${esc(t("Not scored."))}</p></div>`;
    const lines = [["SI-SDR", "si_sdr_db", " dB", 1, 100], ["STOI", "stoi", "", 2, null], ["PESQ", "pesq_wb", "", 2, null]];
    const body = lines.map(([label, key, unit, digits, cap]) => {
      const input = model.input?.[key];
      const output = model.output?.[key];
      const change = Number.isFinite(input) && Number.isFinite(output) && !(cap && input >= cap) ? output - input : null;
      return `<tr><th>${label}</th><td>${esc(M.formatNumber(input, { digits, unit, cap }))}</td><td>${esc(M.formatNumber(output, { digits, unit, cap }))}</td><td class="${change > 0 ? "is-up" : change < 0 ? "is-down" : ""}">${esc(M.formatSigned(change, { digits }))}</td></tr>`;
    }).join("");
    const notes = [];
    notes.push(esc(t("Output level {level} against its input.", { level: M.formatSigned(model.level_change_db, { unit: " dB" }) })));
    if (Number.isFinite(model.gain) && model.gain < 1) notes.push(esc(t("Scored on the pair divided by {factor}, as the converter would deliver it.", { factor: M.formatValue(1 / model.gain) })));
    if (model.same_input_as) notes.push(esc(t("Same input as {stage}: scored once.", { stage: state.rows.find((item) => item.id === model.same_input_as)?.title || model.same_input_as })));
    return `<div class="pipe-table-block"><p class="eyebrow">${esc(t("Model"))} · ${esc(report.model.display_name || report.model.id)}</p><table class="pipe-table is-numeric"><thead><tr><th></th><th>${esc(t("Input"))}</th><th>${esc(t("Output"))}</th><th>${esc(t("Change"))}</th></tr></thead><tbody>${body}</tbody></table><p class="pipe-quiet">${notes.join(" ")}</p></div>`;
  }

  // -------------------------------------------------------------------- deck

  function audioUrl(name) {
    return name ? state.report.audio_urls?.[name] || null : null;
  }

  async function loadDeck(row) {
    if (!state.deck) state.deck = new window.PureSoundCompareDeck($("#pipe-deck"), { emptyText: "This stage did not act on the row: there is nothing new to hear." });
    const deck = state.deck;
    if (!row.fired || !row.audio) { deck.setTracks([]); deck.setCurves([]); return; }
    const previous = M.previousFired(state.rows, row.id);
    const tracks = [];
    if (previous?.audio) tracks.push({ id: "before", label: "Before", color: "#c8f6f9", hint: t("the mixture after {stage}", { stage: previous.title }), levelReference: true, url: audioUrl(previous.audio.noisy) });
    tracks.push({ id: "after", label: previous ? "After" : "Mixture", color: "#bdbbff", hint: t("the mixture after {stage}", { stage: row.title }), levelReference: !previous, url: audioUrl(row.audio.noisy) });
    tracks.push({ id: "target", label: "Target", color: "#7ee0a1", hint: "what the loss compares the output with", url: audioUrl(row.audio.target) });
    if (row.audio.interferers) tracks.push({ id: "interferers", label: "Interferers", color: "#ef2cc1", hint: "the other talkers' bus at this point", url: audioUrl(row.audio.interferers) });
    if (row.model?.audio) tracks.push({ id: "model", label: "Model output", color: "#fc4c02", hint: t("{model} on this stage's mixture", { model: state.report.model?.display_name || t("model") }), url: audioUrl(row.model.audio) });
    if (previous?.audio) tracks.push({ id: "delta", label: "Stage change", color: "#ffd166", hint: "after − before: what this stage added or removed", diagnostic: true, url: null });
    const token = Symbol(row.id);
    state.deckToken = token;
    deck.setTracks(tracks);
    const frames = row.metrics?.frame_esnr_db || [];
    deck.setCurves(frames.length ? [{ label: "Effective SNR", hint: "per 20 ms: target against everything else, −30 to 60 dB", values: frames.map((value) => (value === null ? NaN : value)), hopSeconds: row.metrics.frame_hop_seconds || 0.02, range: [-30, 60], color: "#ffd166" }] : []);
    await Promise.allSettled(tracks.filter((track) => track.url).map((track) => deck.loadTrack(track.id, track.url)));
    if (state.deckToken !== token) return;
    const before = deck.track("before");
    const after = deck.track("after");
    if (before?.samples && after?.samples && deck.track("delta")) {
      if (before.samples.length === after.samples.length) {
        const diff = new Float32Array(after.samples.length);
        for (let index = 0; index < diff.length; index += 1) diff[index] = after.samples[index] - before.samples[index];
        await deck.loadTrack("delta", M.encodeFloatWav(diff, after.buffer.sampleRate)).catch(() => null);
      } else {
        const lane = deck.track("delta");
        if (lane?.meta) lane.meta.textContent = t("lengths differ (this stage changes the timing)");
      }
    }
    if (state.deckToken === token) deck.select("after");
  }

  // -------------------------------------------------------------------- room

  /* The line over the room: its size and reverberation, or why there is none. */
  function renderRoomHelp() {
    const room = M.displayRoom(state.report);
    $("#pipe-room-help").textContent = room
      ? `${t("Room {id}", { id: room.room_id })} · ${room.room_dim.map((value) => value.toFixed(1)).join(" × ")} m · RT60 ${M.formatNumber(room.rt60, { digits: 2, unit: " s" })}. ${t("The microphone is the device; coloured sources are the ones this row used, lit when the selected stage uses them.")}`
      : t("Uploaded impulse responses carry no geometry, so only the responses are drawn.");
    return room;
  }

  function renderRoom() {
    const report = state.report;
    const panel = $("#pipe-room-panel");
    const rirs = report.rirs || [];
    panel.hidden = !report.room && !rirs.length;
    if (panel.hidden) return;
    const room = renderRoomHelp();
    const view = $("#pipe-room-view");
    view.hidden = !room;
    if (room && state.roomMode === "3d") loadRoomView();
    const Room = window.PureSoundRoomView;
    const canThreeD = Boolean(Room && Room.supported());
    $("#pipe-room-mode").hidden = !room || !webgl();
    if (room) {
      if (canThreeD && state.roomMode === "3d") {
        if (!state.room) state.room = new Room(view);
        state.room.show(room, { colors: ROLE_COLORS, labels: ROLE_LABELS });
      } else {
        state.room?.dispose?.();
        state.room = null;
        view.innerHTML = topViewSvg(M.topView(room));
      }
    }
    $$("#pipe-room-mode [data-room-mode]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.roomMode === (canThreeD ? state.roomMode : "plan") ? "true" : "false"));
    renderRirs([]);
  }

  /* three.js is fetched the first time a room is drawn, not with the page; the
   * module announces itself with "puresound:room-view-ready" and the room is
   * redrawn then. Until it arrives (or without WebGL) the floor plan stands in. */
  function loadRoomView() {
    if (window.PureSoundRoomView || state.roomLoading || !webgl()) return;
    state.roomLoading = import("/pipeline-room.js").catch(() => { state.roomMode = "plan"; });
  }

  function webgl() {
    if (state.webgl === undefined) {
      try {
        const canvas = document.createElement("canvas");
        state.webgl = Boolean(window.WebGLRenderingContext && (canvas.getContext("webgl2") || canvas.getContext("webgl")));
      } catch {
        state.webgl = false;
      }
    }
    return state.webgl;
  }

  function highlightRoom(row) {
    const rirs = state.report.rirs || [];
    const roles = row ? [...new Set(rirs.filter((rir) => rir.stage === row.id).map(M.displayRole))] : [];
    if (state.room) state.room.highlight(roles);
    else $$("#pipe-room-view [data-role]").forEach((node) => node.classList.toggle("is-dim", roles.length > 0 && !roles.includes(node.dataset.role)));
    renderRirs(row ? rirs.filter((rir) => rir.stage === row.id).map((rir) => rir.index) : []);
  }

  function topViewSvg(view) {
    if (!view) return "";
    const width = 420;
    const scale = width / view.width;
    const height = view.depth * scale;
    const px = (point) => `${(point.x * scale).toFixed(1)},${((view.depth - point.y) * scale).toFixed(1)}`;
    const obstacles = view.obstacles.map((obstacle) => svg("polygon", { points: obstacle.points.map(px).join(" "), class: "pipe-obstacle" }, tip(t("{material}, {height} high", { material: obstacle.material || t("obstacle"), height: M.formatNumber(obstacle.height, { digits: 1, unit: " m" }) })))).join("");
    const receiver = view.receiver;
    const sources = view.sources.map((source) => {
      const role = source.roles[0] || null;
      const [sx, sy] = px(source).split(",").map(Number);
      const [rx, ry] = px(receiver).split(",").map(Number);
      const line = role ? svg("line", { x1: sx, y1: sy, x2: rx, y2: ry, class: "pipe-ray", stroke: ROLE_COLORS[role] }) + text({ x: (sx + rx) / 2 + 4, y: (sy + ry) / 2 - 4, class: "pipe-ray-label" }, `${M.formatNumber(source.distance, { digits: 2 })} m`) : "";
      return svg("g", { "data-role": role || "unused" }, line + svg("circle", { cx: sx, cy: sy, r: role ? 7 : 4, class: role ? "pipe-source" : "pipe-source is-unused", fill: role ? ROLE_COLORS[role] : "none" }) + text({ x: sx + 9, y: sy + 4, class: "pipe-source-label" }, source.label) + tip(`${source.label} · ${role ? source.roles.map(roleLabel).join(", ") : t("not used on this row")} · ${t("{distance} m from the microphone", { distance: M.formatNumber(source.distance, { digits: 2 }) })}`));
    }).join("");
    const mic = svg("g", { class: "pipe-mic" }, svg("circle", { cx: px(receiver).split(",")[0], cy: px(receiver).split(",")[1], r: 8 }) + text({ x: Number(px(receiver).split(",")[0]) + 11, y: Number(px(receiver).split(",")[1]) + 4 }, "mic"));
    return `<svg class="pipe-svg pipe-topview" viewBox="-12 -12 ${width + 24} ${height + 24}" role="img" aria-label="${esc(t("Room seen from above"))}">${svg("rect", { x: 0, y: 0, width, height, class: "pipe-room-outline" })}${obstacles}${sources}${mic}</svg>${roleLegend(view)}`;
  }

  function roleLegend(view) {
    const used = new Set(view.sources.flatMap((source) => source.roles));
    const items = Object.keys(ROLE_COLORS).filter((role) => used.has(role));
    return `<div class="pipe-legend">${items.map((role) => `<span><i style="background:${ROLE_COLORS[role]}"></i>${esc(roleLabel(role))}</span>`).join("")}<span><i class="is-unused"></i>${esc(t("not used on this row"))}</span><span><i class="is-mic"></i>${esc(t("microphone"))}</span></div>`;
  }

  function renderRirs(highlighted) {
    const rirs = state.report.rirs || [];
    const host = $("#pipe-rirs");
    if (!rirs.length) { host.innerHTML = `<p class="pipe-quiet">${esc(t("No impulse response on this row."))}</p>`; return; }
    host.innerHTML = rirs.map((rir) => rirCard(rir, highlighted.includes(rir.index))).join("");
  }

  function rirCard(rir, highlighted) {
    const meta = rir.metadata || {};
    const summary = rir.summary;
    const facts = [
      meta.label ? t("channel {label}", { label: meta.label }) : null,
      Number.isFinite(meta.source_receiver_distance) ? `${meta.source_receiver_distance.toFixed(2)} m` : null,
      Number.isFinite(meta.drr_db) ? `DRR ${M.formatNumber(meta.drr_db, { unit: " dB" })}` : null,
      Number.isFinite(meta.rt60) ? `RT60 ${meta.rt60.toFixed(2)} s` : null,
      summary && Number.isFinite(summary.t20_s) ? `T20 ${summary.t20_s.toFixed(2)} s` : null,
    ].filter(Boolean);
    const stage = state.rows.find((row) => row.id === rir.stage);
    const role = M.displayRole(rir);
    return `<div class="pipe-rir${highlighted ? " is-highlighted" : ""}"><div class="pipe-rir-head"><i style="background:${ROLE_COLORS[role] || "#999"}"></i><strong>${esc(roleLabel(role))}</strong><span>${esc(facts.join(" · "))}</span><small>${esc(stage ? stage.title : "")} · ${esc((rir.modes || []).join(" + "))}</small></div>${summary ? rirPlot(summary, rir.modes || []) : `<p class="pipe-quiet">${esc(t("A folder impulse response: its samples were not kept."))}</p>`}</div>`;
  }

  function rirPlot(summary, modes) {
    const width = 420;
    const height = 120;
    const span = Math.min(summary.length_ms, Math.max(200, (summary.t20_s || 0.4) * 1000 * 1.2));
    const floor = -80;
    const x = (ms) => (ms / span) * width;
    const y = (db) => ((0 - Math.max(db, floor)) / -floor) * height;
    const bins = summary.envelope_db.slice(0, Math.ceil(span / summary.bin_ms));
    const envelope = `M0,${height} ` + bins.map((db, index) => `L${x(index * summary.bin_ms).toFixed(1)},${y(db).toFixed(1)}`).join(" ") + ` L${x(bins.length * summary.bin_ms).toFixed(1)},${height} Z`;
    const edc = summary.edc_db.slice(0, Math.ceil((span - summary.peak_ms) / summary.bin_ms)).map((db, index) => `${index ? "L" : "M"}${x(summary.peak_ms + index * summary.bin_ms).toFixed(1)},${y(db).toFixed(1)}`).join(" ");
    const windows = [];
    if (modes.includes("early")) windows.push(svg("rect", { x: x(summary.peak_ms), y: 0, width: Math.max(1, x(summary.early_end_ms) - x(summary.peak_ms)), height, class: "pipe-window" }, tip("early target window: direct peak + 50 ms")) + text({ x: x(summary.early_end_ms) + 3, y: 11, class: "pipe-window-label" }, "early target ends"));
    if (modes.includes("direct")) windows.push(svg("line", { x1: x(summary.direct_end_ms), x2: x(summary.direct_end_ms), y1: 0, y2: height, class: "pipe-window-line" }));
    const grid = [-20, -40, -60].map((db) => svg("line", { x1: 0, x2: width, y1: y(db), y2: y(db), class: "pipe-grid-line" }) + text({ x: width - 2, y: y(db) - 2, "text-anchor": "end", class: "pipe-axis" }, `${db} dB`)).join("");
    const axis = [0, span / 2, span].map((ms) => text({ x: Math.min(width - 2, x(ms) + 2), y: height + 12, class: "pipe-axis", "text-anchor": ms === span ? "end" : "start" }, `${Math.round(ms)} ms`)).join("");
    return `<svg class="pipe-svg pipe-rir-plot" viewBox="0 0 ${width} ${height + 16}" role="img" aria-label="Impulse response envelope and energy decay">${windows.join("")}${grid}${svg("path", { d: envelope, class: "pipe-envelope" })}${svg("path", { d: edc, class: "pipe-edc" })}${axis}</svg><div class="pipe-legend is-small"><span><i class="is-envelope"></i>peak envelope</span><span><i class="is-edc"></i>energy decay</span></div>`;
  }

  // -------------------------------------------------------------- language

  function applyLang() {
    state.lang = pageLang();
    if (state.overview?.available) {
      const recipe = currentRecipe();
      if (recipe) renderRecipeHint(recipe);
      renderModelOptions();
    }
    if (state.report) {
      const selected = state.selected;
      state.rows = M.stageRows(state.report, state.catalog, state.lang);
      renderSummary();
      renderRoomHelp();
      renderRail();
      renderCharts();
      const row = state.rows.find((item) => item.id === selected);
      if (row) {
        $$("#pipe-rail .pipe-stage").forEach((button) => button.classList.toggle("is-selected", button.dataset.stage === selected));
        renderDetail(row);
      }
    }
  }

  // ------------------------------------------------------------------ events

  function playPreview(button) {
    const url = button.dataset.preview;
    if (state.preview && state.preview.dataset === url) {
      state.preview.audio.pause();
      state.preview.button.textContent = "▶";
      state.preview = null;
      return;
    }
    if (state.preview) { state.preview.audio.pause(); state.preview.button.textContent = "▶"; }
    const audio = new Audio(url);
    audio.addEventListener("ended", () => { button.textContent = "▶"; state.preview = null; });
    audio.play().catch(() => toast(t("Could not play the sample."), true));
    button.textContent = "■";
    state.preview = { audio, button, dataset: url };
  }

  function wire() {
    $("#pipe-setup").addEventListener("submit", (event) => { event.preventDefault(); run(); });
    $("#pipe-cancel").addEventListener("click", cancel);
    $("#pipe-recipe").addEventListener("change", onRecipeChange);
    $("#pipe-role").addEventListener("change", syncEpoch);
    $("#pipe-foreground-sample").addEventListener("change", syncForegroundTalkers);
    $("#pipe-seed-roll").addEventListener("click", () => { $("#pipe-seed").value = Math.floor(Math.random() * 100000); });
    $("#pipe-foreground-file").addEventListener("change", (event) => { addFiles("foreground", event.target.files, { single: true }); event.target.value = ""; });
    $("#pipe-talker-files").addEventListener("change", (event) => { addFiles("talkers", event.target.files); event.target.value = ""; });
    $("#pipe-noise-files").addEventListener("change", (event) => { addFiles("noises", event.target.files); event.target.value = ""; });
    $("#pipe-rir-files").addEventListener("change", (event) => { addFiles("rirs", event.target.files); event.target.value = ""; });
    $$('input[name="pipe-rir"]').forEach((input) => input.addEventListener("change", () => {
      $("#pipe-rooms").hidden = rirKind() !== "samples";
      $("#pipe-rir-upload").hidden = rirKind() !== "upload";
    }));
    $("#pipe-setup").addEventListener("click", (event) => {
      const remove = event.target.closest("[data-remove-file]");
      if (remove) {
        state.files[remove.dataset.removeFile].splice(Number(remove.dataset.index), 1);
        renderFiles();
        syncForegroundTalkers();
        return;
      }
      const preview = event.target.closest("[data-preview]");
      if (preview) { event.preventDefault(); playPreview(preview); }
    });
    window.addEventListener("puresound:lang", () => { if (state.loaded) applyLang(); });
    $("#pipe-rail").addEventListener("click", (event) => { const button = event.target.closest("[data-stage]"); if (button) select(button.dataset.stage, { scroll: false }); });
    $("#pipe-rail").addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); step(1); }
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); step(-1); }
    });
    ["#pipe-esnr-chart", "#pipe-model-chart"].forEach((selector) => {
      $(selector).addEventListener("click", (event) => { const point = event.target.closest("[data-stage]"); if (point) select(point.dataset.stage, { scroll: false }); });
      $(selector).addEventListener("keydown", (event) => { if (event.key === "Enter") { const point = event.target.closest("[data-stage]"); if (point) select(point.dataset.stage, { scroll: false }); } });
    });
    $("#pipe-metric").addEventListener("click", (event) => {
      const button = event.target.closest("[data-metric]");
      if (!button) return;
      state.metric = button.dataset.metric;
      $$("#pipe-metric [data-metric]").forEach((item) => item.setAttribute("aria-pressed", item === button ? "true" : "false"));
      renderModelChart();
    });
    $("#pipe-detail").addEventListener("click", (event) => { const button = event.target.closest("[data-step]"); if (button) step(Number(button.dataset.step)); });
    $("#pipe-room-mode").addEventListener("click", (event) => {
      const button = event.target.closest("[data-room-mode]");
      if (!button || !state.report) return;
      state.roomMode = button.dataset.roomMode === "plan" ? "plan" : "3d";
      renderRoom();
      highlightRoom(state.rows.find((row) => row.id === state.selected));
    });
    let resizeFrame = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => { if (state.report && !$("#pipe-result").hidden) renderCharts(); });
    }).observe($("#pipe-esnr-chart"));
    window.addEventListener("puresound:room-view-ready", () => { if (state.report) { renderRoom(); highlightRoom(state.rows.find((row) => row.id === state.selected)); } });
  }

  wire();

  window.PureSoundPipeline = {
    show,
    run,
    deck: () => state.deck,
  };
})();
