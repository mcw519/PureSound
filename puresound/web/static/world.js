/* Acoustic world screen: edit a keyframed scene on the room's floor plan,
 * render it as a cancellable server job, listen against two references and
 * sweep two parameters into a limit map.  Pure logic lives in world-model.js;
 * the renderer is puresound/audio/rir/render/dynamic.py. */
(() => {
  "use strict";

  const M = window.PureSoundWorldModel;
  const t = (key, vars) => window.PureSoundI18n.t(key, vars);
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const api = (...args) => window.PureSoundApp.api(...args);
  const shell = () => window.PureSoundShell;
  const fixed = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
  const round = (value, places = 3) => Math.round(value * 10 ** places) / 10 ** places;

  // Source colours follow the Pipeline room view; track colours follow the decks.
  const ROLE_COLORS = { target: "#fc4c02", interferer: "#ef2cc1", noise: "#b0782b" };
  // Texts are functions so each reads the current language when shown.
  const ROLE_NAMES = { target: () => t("Target talker"), interferer: () => t("Other talker"), noise: () => t("Noise") };
  const TRACK_COLORS = { input: "#c8f6f9", aligned: "#bdbbff", device: "#8f8cff", aicoustics: "#67e8c8", removed: "#ff9a62", target: "#7ee0a1", speech: "#8fd3ff", near: "#b5e48c", windows: "#ffd166" };
  const ENHANCEMENT_TASKS = ["voice_isolation", "noise_suppression"];
  const PRESETS = {
    approach: () => [t("Walk up and away"), t("The target talker walks up to the microphone and back while another talker stands still.")],
    exchange: () => [t("Swap near and far"), t("The two talkers cross the room, trading the near and far positions.")],
    turn: () => [t("Turn while talking"), t("The target talker stays put and turns away from the microphone and back.")],
    occlusion: () => [t("Walk behind a screen"), t("The target talker walks behind a screen that blocks the direct sound.")],
    busy: () => [t("Busy room"), t("The target talker walks past two other talkers who stand still; a fan runs and a humming machine crosses the room.")],
  };
  const MATERIALS = { classroom: () => t("Classroom"), living_room: () => t("Living room"), meeting_room: () => t("Meeting room"), office: () => t("Office") };
  // Sweep parameters change every source of one role by the same amount.
  const PARAMETERS = {
    noise_gain_change_db: () => [t("Noise level change · dB"), t("Added to the gain of every noise source.")],
    interferer_gain_change_db: () => [t("Other talker level change · dB"), t("Added to the gain of every other talker.")],
    target_distance_change_m: () => [t("Target distance change · m"), t("Moves every keyframe of every target talker away from (+) or toward (−) the microphone.")],
  };
  const ABSENT = {
    noise: () => t("The limit map changes the noise sources, but the scene has none."),
    interferer: () => t("The limit map changes the other talkers, but the scene has none."),
    target: () => t("The limit map moves the target talkers, but the scene has none."),
  };
  // What a reference keeps of each talker's sound, with training's names.
  const REFERENCES = {
    early: () => [t("Early · direct + 50 ms"), t("the direct sound and the first 50 ms of reflections")],
    direct: () => [t("Direct · direct + 6 ms"), t("the direct sound only")],
    full: () => [t("Full · with reverberation"), t("the whole sound at the microphone, reverberation included")],
    anechoic: () => [t("Anechoic · dry"), t("the dry clip, as if there were no room")],
  };
  const POLICIES = {
    target: (v) => [t("Target talkers"), t("Every target talker: {reference}.", v)],
    speech: (v) => [t("All speech"), t("Every talker, target or not: {reference}.", v)],
    near: (v) => [t("Near region"), t("Every talker inside {radius} m of the microphone, faded across 0.2 m at the edge: {reference}. Silent while nobody is inside.", v)],
  };
  const SILENT = {
    target: () => t("The scene has no target talker, so there is nothing to keep: what counts is how quiet the output is."),
    speech: () => t("The scene has no talker, so there is nothing to keep: what counts is how quiet the output is."),
    near: () => t("Nobody is inside the near region, so there is nothing to keep: what counts is how quiet the output is."),
  };
  const PHASES = {
    preparing: () => t("Preparing"),
    geometry: () => t("Tracing paths"),
    reflections: () => t("Filtering reflections"),
    reverberation: () => t("Adding reverberation"),
    inference: () => t("Running the model"),
    aicoustics_model: () => t("Loading ai-coustics model"),
    aicoustics_inference: () => t("Running ai-coustics"),
    measurements: () => t("Scoring"),
    sweep: () => t("Sweeping"),
    complete: () => t("Done"),
  };
  const CELL_STATES = { queued: () => t("queued"), running: () => t("rendering…"), failed: () => t("failed"), cancelled: () => t("cancelled") };
  const ISSUES = {
    outside: (v) => t("Keyframe {n} of {source} is outside the room.", v),
    mic: (v) => t("Keyframe {n} of {source} is closer than {limit} m to the microphone.", v),
    obstacle: (v) => t("Keyframe {n} of {source} is inside an obstacle.", v),
    start: (v) => t("The first keyframe of {source} must be at 0 s.", v),
    order: (v) => t("Keyframe {n} of {source} must come later than keyframe {previous}.", v),
    range: (v) => t("Keyframe {n} of {source} lies outside the scene's {duration} s.", v),
    speed: (v) => t("{source} moves at {speed} m/s into keyframe {n}; the limit is {limit} m/s.", v),
    clip: (v) => t("{source} needs {seconds} s for its complete clip. Increase the scene length to at least {end} s, start earlier, or choose a shorter recording.", v),
  };
  // Keyframe columns: field, header, step, shown only with "Height and pitch".
  const KEY_FIELDS = [
    ["time", () => t("t · s"), 0.1, false],
    ["x", () => "X · m", 0.1, false],
    ["y", () => "Y · m", 0.1, false],
    ["z", () => "Z · m", 0.1, true],
    ["yaw", () => t("Yaw · °"), 5, false],
    ["pitch", () => t("Pitch · °"), 5, true],
  ];
  // The plan's drawing size follows its box, so text keeps its size on phones.
  const PLAN = { minWidth: 340, maxWidth: 820, aspect: 0.69, pad: 34 };

  const state = {
    overview: null,
    limits: null,
    models: [],
    model: "",
    preset: "approach",
    scene: null,
    assets: {},
    uploads: {},
    selected: { source: 0, key: 0 },
    time: 0,
    tab: "scene",
    view: "plan",
    policy: "target",
    heights: false,
    issues: [],
    serverError: null,
    validation: 0,
    validateTimer: null,
    transform: null,
    drag: null,
    job: null,
    report: null,
    cell: null,
    deck: null,
    transcriber: null,
    deviceTrack: null,
    deviceNote: null,
    deviceRun: null,
    materials: 0,
    device: null,
    room: null,
    roomShown: false,
    map: null,
    axes: [{ parameter: "noise_gain_change_db", text: "-12, -6, 0, 6" }, { parameter: "interferer_gain_change_db", text: "-6, 0, 6" }],
    wasPlaying: false,
    aicoustics: { enabled: false, key: "", model_id: "quail-vf-2.2-l-16khz", enhancement_level: 1, autoShow: false },
    mapProvider: "puresound",
  };

  /* Entry points ------------------------------------------------------------ */
  let starting = null;
  function show() {
    if (state.scene) {
      drawStage();
      return Promise.resolve();
    }
    starting ??= initialize().finally(() => { starting = null; });
    return starting;
  }

  async function initialize() {
    try {
      const [overview, catalog] = await Promise.all([api("/api/world"), api("/api/models")]);
      state.overview = overview;
      state.limits = overview.limits;
      state.models = catalog.models.filter((model) => model.runnable !== false && ENHANCEMENT_TASKS.includes(model.task));
      state.model = state.models.find((model) => model.roles?.includes("default"))?.id || "";
      loadPreset("approach");
      applyDefaultPolicy();
      wire();
      renderSettings();
      renderPolicies();
      renderMap();
      refresh();
      requestAnimationFrame(tick);
    } catch (error) {
      $("#world-layout").hidden = true;
      const box = $("#world-unavailable");
      box.hidden = false;
      box.textContent = t("The acoustic world is unavailable: {reason}", { reason: error.message });
    }
  }

  function loadPreset(name) {
    state.preset = name;
    state.scene = structuredClone(state.overview.presets[name]);
    state.materials += 1; // a material draw in flight was for the scene just replaced
    state.assets = {};
    state.uploads = {};
    state.selected = { source: 0, key: 0 };
    state.time = 0;
    invalidateResult();
  }

  /* Events ------------------------------------------------------------------ */
  function wire() {
    $("#world-run").addEventListener("click", run);
    $("#world-cancel").addEventListener("click", cancel);
    $("#world-empty").addEventListener("click", run);
    $$("[data-world-tab]").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.worldTab)));
    $$("[data-world-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.worldView)));
    $("#world-timeline").addEventListener("input", (event) => seek(Number(event.target.value), { deck: true }));
    $("#world-play").addEventListener("click", () => state.deck?.toggle());
    $("#world-device").addEventListener("click", runOnDevice);
    $$("[data-world-policy]").forEach((group) => group.addEventListener("click", (event) => {
      const button = event.target.closest("[data-policy]");
      if (button) setPolicy(button.dataset.policy);
    }));
    $("#world-map").addEventListener("click", onMapClick);
    $("#world-map-provider").addEventListener("change", (event) => { state.mapProvider = event.target.value; renderMap(); });
    $("#world-cell-banner").addEventListener("click", (event) => { if (event.target.closest("[data-action=back-to-map]")) setTab("map"); });
    const settings = $("#world-settings");
    settings.addEventListener("change", onSettingChange);
    settings.addEventListener("input", onSettingInput);
    settings.addEventListener("click", onSettingClick);
    const plan = $("#world-plan");
    plan.addEventListener("pointerdown", onPlanPointerDown);
    plan.addEventListener("pointermove", onPlanPointerMove);
    plan.addEventListener("pointerup", onPlanPointerUp);
    plan.addEventListener("pointercancel", onPlanPointerUp);
    plan.addEventListener("dblclick", onPlanDoubleClick);
    plan.addEventListener("keydown", onPlanKey);
    let width = 0;
    new ResizeObserver(() => {
      if (!state.scene || Math.abs(plan.clientWidth - width) < 8) return;
      width = plan.clientWidth;
      drawPlan();
      drawNow();
    }).observe(plan);
  }

  function setTab(tab) {
    state.tab = tab;
    $$("[data-world-tab]").forEach((button) => {
      const on = button.dataset.worldTab === tab;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-selected", String(on));
    });
    $("#world-scene").hidden = tab !== "scene";
    $("#world-map-view").hidden = tab !== "map";
    $$(".world-sweep-only").forEach((node) => { node.hidden = tab !== "map"; });
    updateRunButton();
    if (tab === "scene") drawStage();
  }

  function setView(view) {
    state.view = view;
    $$("[data-world-view]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.worldView === view)));
    // An <svg> has no hidden property; the attribute is what CSS hides.
    $("#world-plan").toggleAttribute("hidden", view !== "plan");
    $("#world-3d").hidden = view !== "3d";
    $("#world-stage-title").textContent = t(view === "plan" ? "Room · seen from above" : "Room · 3-D");
    $("#world-stage-help").textContent = t(view === "plan" ? "Drag a numbered keyframe or the microphone. Double-click the floor to put the selected source there at the timeline's time." : "Drag to turn, scroll to zoom, right-drag to pan. Paths are edited on the floor plan.");
    if (view === "3d") showRoom3d();
  }

  /* The reference the chosen model's task calls for. */
  function applyDefaultPolicy() {
    const task = state.models.find((model) => model.id === state.model)?.task;
    setPolicy(M.defaultPolicy(task, state.scene, state.overview.policy_defaults));
  }

  function setPolicy(policy) {
    state.policy = policy;
    renderPolicies();
    if (state.report) {
      renderScores();
      state.deck?.setCurves(curves(state.report));
    }
    renderMap();
  }

  /* Any edit to the scene: the last render no longer describes it. */
  function changed({ settings = false, room = false } = {}) {
    invalidateResult();
    if (settings) renderSettings();
    if (room) state.roomShown = false;
    refresh();
  }

  function refresh() {
    state.time = Math.min(state.time, state.scene.duration_s);
    validate();
    drawStage();
    updateRunButton();
  }

  function invalidateResult() {
    state.deck?.pause();
    state.deviceRun?.abort();
    state.report = null;
    state.cell = null;
    state.deviceTrack = null;
    state.deviceNote = null;
    $("#world-results").hidden = true;
    $("#world-empty").hidden = false;
    $("#world-play").disabled = true;
  }

  /* A source's colour: its role's hue, lighter for each further source of the role. */
  function roleColor(scene, s) {
    return M.shadeColor(ROLE_COLORS[scene.sources[s].role], M.roleShade(scene, s));
  }

  /* Settings ---------------------------------------------------------------- */
  function renderSettings() {
    const scene = state.scene;
    const limits = state.limits;
    const room = scene.room;
    const dims = room.dimensions_m;
    const mic = M.micPosition(scene);
    const presetText = PRESETS[state.preset]?.();
    const presets = Object.keys(state.overview.presets).map((name) => `<option value="${esc(name)}"${name === state.preset ? " selected" : ""}>${esc(PRESETS[name]?.()[0] || name)}</option>`);
    if (!state.preset) presets.unshift(`<option value="" selected disabled>${esc(t("Imported scene"))}</option>`);
    $("#world-settings").innerHTML = `<fieldset class="world-settings-form" id="world-settings-form">
      <section class="inspector-section">
        <p class="eyebrow">${esc(t("Scene"))}</p>
        <div class="field"><label for="world-preset">${esc(t("Starting point"))}</label>
          <select class="control" id="world-preset" data-inspector-focus>${presets.join("")}</select>
          <small>${esc(presetText ? presetText[1] : t("Loaded from a file."))}</small></div>
        <div class="world-grid-2">
          ${numberField("world-duration", t("Length · s"), scene.duration_s, { min: 1, max: limits.duration_s, step: 0.5, attrs: 'data-scene="duration"' })}
          <label class="field"><span class="field-name">${esc(t("Seed"))}</span><span class="world-seed"><input class="control" id="world-seed" type="number" min="0" max="4294967295" step="1" value="${esc(scene.seed)}" data-scene="seed" /><button class="button button-icon" type="button" data-action="roll-seed" title="${esc(t("Another seed"))}" aria-label="${esc(t("Another seed"))}">⚄</button></span></label>
        </div>
        <small class="field-hint">${esc(t("The seed draws the room's materials and its reverberant tail."))}</small>
        <div class="field"><label for="world-reference">${esc(t("Reference"))}</label>
          <select class="control" id="world-reference" data-scene="reference">${limits.reference_rir.map((name) => `<option value="${esc(name)}"${name === scene.reference_rir ? " selected" : ""}>${esc(REFERENCES[name]?.()[0] || name)}</option>`).join("")}</select>
          <small>${esc(t("What the scores compare the output with: {description}.", { description: REFERENCES[scene.reference_rir]?.()[1] || scene.reference_rir }))}</small></div>
      </section>
      <section class="inspector-section">
        <p class="eyebrow">${esc(t("Room"))}</p>
        <div class="field"><label for="world-material">${esc(t("Room type"))}</label>
          <select class="control" id="world-material" data-scene="material">${state.overview.materials.map((name) => `<option value="${esc(name)}"${name === room.room_type ? " selected" : ""}>${esc(MATERIALS[name]?.() || name)}</option>`).join("")}</select>
          <small>${esc(t("Wall, floor and ceiling materials typical of this room."))}</small></div>
        <p class="world-subhead">${esc(t("Size · m"))}</p>
        <div class="world-grid-3">${[t("Width"), t("Depth"), t("Height")].map((name, axis) => numberField(`world-dim-${axis}`, name, dims[axis], { min: 1.5, max: 30, step: 0.1, attrs: `data-scene="dim" data-axis="${axis}"` })).join("")}</div>
        <p class="world-subhead">${esc(t("Microphone · m"))}</p>
        <div class="world-grid-3">${["X", "Y", "Z"].map((name, axis) => numberField(`world-mic-${axis}`, name, mic[axis], { min: 0.05, max: dims[axis] - 0.05, step: 0.1, attrs: `data-scene="mic" data-axis="${axis}"` })).join("")}</div>
        ${numberField("world-radius", t("Near radius · m"), scene.near_radius_m, { min: limits.near_radius_m[0], max: limits.near_radius_m[1], step: 0.1, attrs: 'data-scene="radius"', hint: t("Talkers closer than this count for the near-region reference.") })}
      </section>
      <section class="inspector-section">
        <p class="eyebrow">${esc(t("Model"))}</p>
        <div class="field"><label for="world-model">${esc(t("Enhancement model"))}</label>
          <select class="control" id="world-model" data-scene="model"><option value="">${esc(t("None · room only"))}</option>${state.models.map((model) => `<option value="${esc(model.id)}"${model.id === state.model ? " selected" : ""}>${esc(model.display_name || model.id)}</option>`).join("")}</select>
          <small>${esc(t("Runs on the server after the room is rendered. Without one, scores describe the microphone."))}</small></div>
      </section>
      ${aicousticsSettings()}
      <section class="inspector-section">
        <div class="world-section-head"><p class="eyebrow">${esc(t("Sources"))}</p>
          <label class="world-inline-toggle"><input type="checkbox" data-action="toggle-heights"${state.heights ? " checked" : ""} /> ${esc(t("Height and pitch"))}</label></div>
        ${scene.sources.map((source, s) => sourceFieldset(source, s)).join("")}
        ${addSourceRow()}
      </section>
      <section class="inspector-section world-sweep-only"${state.tab === "map" ? "" : " hidden"}>
        <p class="eyebrow">${esc(t("Limit map"))}</p>
        ${state.axes.map((axis, a) => axisFields(axis, a)).join("")}
        <p class="field-hint" id="world-sweep-summary"></p>
      </section>
      <section class="inspector-section">
        <p class="eyebrow">${esc(t("Scene file"))}</p>
        <div class="world-file-actions">
          <label class="button button-secondary button-small world-import"><input type="file" accept=".json,.zip" data-action="import" />${esc(t("Import…"))}</label>
          <button class="button button-secondary button-small" type="button" data-action="export">${esc(t("Export scene JSON"))}</button>
        </div>
        <small class="field-hint">${esc(t("A scene package (.zip) from a render brings its audio along."))}</small>
      </section>
    </fieldset>`;
    $("#world-settings-form").disabled = Boolean(state.job);
    renderSweepSummary();
    markIssues();
  }

  function numberField(id, label, value, { min, max, step, attrs = "", hint = "" }) {
    return `<label class="field"><span class="field-name">${esc(label)}</span><input class="control" id="${esc(id)}" type="number" value="${esc(round(value))}" min="${esc(min)}" max="${esc(max)}" step="${esc(step)}" ${attrs} />${hint ? `<small>${esc(hint)}</small>` : ""}</label>`;
  }

  function aicousticsSettings() {
    const config = state.aicoustics;
    const cap = state.overview.aicoustics;
    return `<section class="inspector-section world-aicoustics">
      <p class="eyebrow">ai-coustics</p>
      <label class="toggle-row"><input type="checkbox" data-aicoustics="enabled"${config.enabled ? " checked" : ""}${cap?.available ? "" : " disabled"} /><span class="toggle-ui"></span><span>${esc(t("Compare with ai-coustics"))}</span></label>
      ${cap?.available ? "" : `<small class="field-hint">${esc(t("Install aic-sdk==3.3.0 on the server to enable comparison."))}</small>`}
      <div${config.enabled ? "" : " hidden"}>
        <label class="field"><span>${esc(t("ai-coustics SDK key"))}</span><input class="control" type="password" data-aicoustics="key" value="${esc(config.key)}" autocomplete="off" spellcheck="false" /></label>
        <label class="field"><span>${esc(t("ai-coustics model"))}</span><select class="control" data-aicoustics="model">${(cap?.models || []).map((model) => `<option value="${esc(model.id)}"${model.id === config.model_id ? " selected" : ""}>${esc(model.label)}</option>`).join("")}</select></label>
        <small class="field-hint">${esc(t("Voice Focus isolates a primary speaker. Multi Speaker keeps speech from multiple speakers."))}</small>
        <label class="field"><span>${esc(t("Enhancement strength"))} · <output data-aicoustics-level>${fixed(config.enhancement_level, 2)}</output></span><input class="range" type="range" min="0" max="1" step="0.05" value="${config.enhancement_level}" data-aicoustics="level" /></label>
        <small class="field-hint">${esc(t("Uses the same scene input on this server. SDK licensing and usage reporting require network access. The key stays in this tab and the running job; it is excluded from saved scenes, history and exports."))}</small>
        <button type="button" class="link-button" data-aicoustics="forget">${esc(t("Forget ai-coustics key"))}</button>
      </div></section>`;
  }

  function sourceFieldset(source, s) {
    const scene = state.scene;
    const still = M.isStill(source);
    const upload = state.uploads[source.asset_id];
    const reference = state.assets[source.asset_id];
    const chosen = reference?.upload_id ? "upload" : reference?.sample || source.asset_id;
    const samples = state.overview.assets.map((asset) => `<option value="${esc(asset.id)}"${chosen === asset.id ? " selected" : ""}>${esc(asset.id)} · ${fixed(asset.duration_s)} s</option>`).join("");
    const uploadOption = upload ? `<option value="upload"${chosen === "upload" ? " selected" : ""}>${esc(t("Uploaded: {name}", { name: upload }))}</option>` : "";
    const fields = KEY_FIELDS.filter(([, , , extra]) => state.heights || !extra);
    const speeds = source.keyframes.map((_, k) => M.segmentSpeed(source, k)).filter((value) => value !== null);
    const top = speeds.length ? Math.max(...speeds) : 0;
    const roles = state.limits.roles.map((role) => `<button type="button" role="radio" aria-checked="${role === source.role}" data-action="set-role" data-source="${s}" data-role="${role}">${esc(ROLE_NAMES[role]())}</button>`).join("");
    const foot = still
      ? `<span class="field-hint">${esc(t("Stays at one spot; drag it on the plan to move it."))}</span>`
      : `<button class="button button-secondary button-small" type="button" data-action="add-key" data-source="${s}">${esc(t("Add keyframe at {time} s", { time: fixed(state.time) }))}</button><span class="field-hint">${esc(t("Top speed {speed} m/s", { speed: fixed(top, 1) }))}</span>`;
    return `<fieldset class="world-source" style="--role:${roleColor(scene, s)}">
      <legend><i></i>${esc(source.source_id)}</legend>
      <div class="world-source-head">
        <div class="segmented" role="radiogroup" aria-label="${esc(t("Role of {source}", { source: source.source_id }))}">${roles}</div>
        <button class="button button-ghost button-icon button-small" type="button" data-action="remove-source" data-source="${s}" aria-label="${esc(t("Remove {source}", { source: source.source_id }))}" title="${esc(scene.sources.length > 1 ? t("Remove {source}", { source: source.source_id }) : t("A scene keeps at least one source."))}"${scene.sources.length > 1 ? "" : " disabled"}>×</button>
      </div>
      <div class="field"><label for="world-asset-${s}">${esc(t("Audio"))}</label>
        <select class="control" id="world-asset-${s}" data-source="${s}" data-field="asset">${samples}${uploadOption}</select>
        <label class="pipe-file"><input type="file" accept="audio/*,.wav,.flac" data-source="${s}" data-field="upload" /><span>${esc(t("Upload a clip instead…"))}</span></label></div>
      <div class="world-grid-2">
        ${numberField(`world-gain-${s}`, t("Gain · dB"), source.gain_db, { min: state.limits.gain_db[0], max: state.limits.gain_db[1], step: 1, attrs: `data-source="${s}" data-field="gain"` })}
        ${numberField(`world-start-${s}`, t("Starts at · s"), source.start_s, { min: 0, max: scene.duration_s, step: 0.1, attrs: `data-source="${s}" data-field="start"` })}
      </div>
      <label class="toggle-row"><input type="checkbox" data-source="${s}" data-field="repeat"${source.repeat ? " checked" : ""} /><span class="toggle-ui"></span><span>${esc(t(source.role === "noise" ? "Loop the clip" : "Repeat complete clips"))}</span></label>
      ${source.role === "noise" ? "" : `<small class="field-hint">${esc(t("Speech plays whole clips only. Repeats stop before one would overrun the scene; the remaining time is quiet, with reverberation preserved."))}</small>`}
      <label class="toggle-row"><input type="checkbox" data-source="${s}" data-field="still"${still ? " checked" : ""} /><span class="toggle-ui"></span><span>${esc(t("Stays still"))}</span></label>
      <div class="world-keys-wrap"><table class="world-keys"><thead><tr><th scope="col">#</th>${fields.map(([, label]) => `<th scope="col" class="num">${esc(label())}</th>`).join("")}<th scope="col"><span class="visually-hidden">${esc(t("Delete"))}</span></th></tr></thead>
        <tbody>${source.keyframes.map((key, k) => keyRow(source, s, key, k, fields)).join("")}</tbody></table></div>
      <div class="world-key-foot">${foot}</div>
    </fieldset>`;
  }

  function addSourceRow() {
    const count = state.scene.sources.length;
    const limit = state.limits.sources;
    const full = count >= limit;
    const title = full ? ` disabled title="${esc(t("A scene holds at most {count} sources.", { count: limit }))}"` : "";
    return `<div class="world-source-add">
      <button class="button button-secondary button-small" type="button" data-action="add-source" data-kind="talker"${title}>${esc(t("+ Talker"))}</button>
      <button class="button button-secondary button-small" type="button" data-action="add-source" data-kind="noise"${title}>${esc(t("+ Noise"))}</button>
      <span class="field-hint">${esc(t("{count} of {limit} sources", { count, limit }))}</span>
    </div>`;
  }

  function keyRow(source, s, key, k, fields) {
    const values = { time: key.time_s, x: key.position_m[0], y: key.position_m[1], z: key.position_m[2], yaw: key.yaw_deg, pitch: key.pitch_deg || 0 };
    const selected = state.selected.source === s && state.selected.key === k;
    // Time zero anchors every path.
    const locked = (field) => field === "time" && k === 0;
    return `<tr data-row="${s}:${k}" class="${selected ? "is-selected" : ""}">
      <td><button class="world-key-index" type="button" data-action="select-key" data-key="${s}:${k}" aria-label="${esc(t("Select keyframe {n}", { n: k + 1 }))}">${k + 1}</button></td>
      ${fields.map(([field, label, step]) => `<td><input class="control world-cell-input" type="number" step="${step}" value="${esc(round(values[field]))}" data-key="${s}:${k}" data-field="${field}" aria-label="${esc(t("Keyframe {n}, {field}", { n: k + 1, field: label() }))}"${locked(field) ? " readonly" : ""} /></td>`).join("")}
      <td><button class="button button-ghost button-icon button-small" type="button" data-action="delete-key" data-key="${s}:${k}" aria-label="${esc(t("Delete keyframe {n}", { n: k + 1 }))}"${k === 0 ? " disabled" : ""}>×</button></td>
    </tr>`;
  }

  function axisFields(axis, a) {
    const parsed = M.parseValues(axis.text, state.limits.sweep_cells);
    const other = state.axes[1 - a].parameter;
    return `<div class="world-axis">
      <div class="field"><label for="world-axis-${a}">${esc(t(a ? "Rows" : "Columns"))}</label>
        <select class="control" id="world-axis-${a}" data-axis-index="${a}" data-field="parameter">${state.overview.sweep_parameters.map((name) => `<option value="${esc(name)}"${name === axis.parameter ? " selected" : ""}${name === other ? " disabled" : ""}>${esc(parameterName(name))}</option>`).join("")}</select>
        <small>${esc(PARAMETERS[axis.parameter]?.()[1] || "")}</small></div>
      <div class="field"><label for="world-values-${a}">${esc(t("Values"))}</label>
        <input class="control" id="world-values-${a}" value="${esc(axis.text)}" data-axis-index="${a}" data-field="values" spellcheck="false" autocomplete="off" />
        <small class="${parsed.error ? "is-error" : ""}" data-axis-note="${a}">${esc(axisNote(parsed))}</small></div>
    </div>`;
  }

  function axisNote(parsed) {
    if (!parsed.error) return t("{count} values: {values}", { count: parsed.values.length, values: parsed.values.join(", ") });
    if (parsed.error.kind === "empty") return t("Enter at least one value, separated by commas.");
    if (parsed.error.kind === "number") return t("“{value}” is not a number.", { value: parsed.error.value });
    return t("At most {count} values.", { count: parsed.error.value });
  }

  function sweepAxes() {
    return state.axes.map((axis) => ({ parameter: axis.parameter, ...M.parseValues(axis.text, state.limits.sweep_cells) }));
  }

  function renderSweepSummary() {
    const node = $("#world-sweep-summary");
    if (!node) return;
    const axes = sweepAxes();
    const cells = axes[0].values.length * axes[1].values.length;
    const over = cells > state.limits.sweep_cells;
    const absent = absentKind();
    node.classList.toggle("is-error", over || Boolean(absent));
    node.textContent = absent ? ABSENT[absent]() : axes.some((axis) => axis.error) ? "" : t(over ? "{cells} cells; the limit is {limit}." : "{cells} cells, one render of the scene each.", { cells, limit: state.limits.sweep_cells });
  }

  /* The role a sweep axis changes when the scene has no source of it. */
  function absentKind() {
    const roles = state.axes.map((axis) => state.overview.sweep_roles[axis.parameter]);
    return roles.find((role) => !state.scene.sources.some((source) => source.role === role));
  }

  function onSettingInput(event) {
    if (event.target.dataset.aicoustics === "key") {
      state.aicoustics.key = event.target.value.trim();
      updateRunButton();
      return;
    }
    if (event.target.dataset.aicoustics === "level") {
      state.aicoustics.enhancement_level = Number(event.target.value);
      $("[data-aicoustics-level]").textContent = fixed(state.aicoustics.enhancement_level, 2);
      invalidateResult();
      updateRunButton();
      return;
    }
    const target = event.target;
    if (target.dataset.field !== "values") return;
    const a = Number(target.dataset.axisIndex);
    state.axes[a].text = target.value;
    const parsed = M.parseValues(target.value, state.limits.sweep_cells);
    const note = $(`[data-axis-note="${a}"]`);
    note.textContent = axisNote(parsed);
    note.classList.toggle("is-error", Boolean(parsed.error));
    renderSweepSummary();
    updateRunButton();
  }

  async function onSettingChange(event) {
    const target = event.target;
    if (target.dataset.aicoustics) {
      const field = target.dataset.aicoustics;
      if (field === "enabled") { state.aicoustics.enabled = target.checked; state.aicoustics.autoShow = target.checked; }
      if (field === "model") state.aicoustics.model_id = target.value;
      if (field === "key") { state.aicoustics.key = target.value.trim(); return updateRunButton(); }
      if (field === "level") state.aicoustics.enhancement_level = Number(target.value);
      invalidateResult(); renderSettings(); updateRunButton(); return;
    }
    const scene = state.scene;
    const value = Number(target.value);
    const kind = target.dataset.scene;
    try {
      if (target.id === "world-preset") {
        loadPreset(target.value);
        applyDefaultPolicy();
        changed({ settings: true, room: true });
      } else if (kind === "duration") {
        if (!(value > 0)) return renderSettings();
        state.scene = M.withDuration(scene, Math.min(value, state.limits.duration_s));
        changed({ settings: true });
      } else if (kind === "seed") {
        await redrawMaterials(scene.room.room_type, Math.max(0, Math.min(4294967295, Math.round(value) || 0)));
      } else if (kind === "material") {
        await redrawMaterials(target.value, scene.seed);
      } else if (kind === "dim") {
        if (!(value > 0)) return renderSettings();
        state.scene = M.withRoomSize(scene, Number(target.dataset.axis), value);
        changed({ room: true });
      } else if (kind === "mic") {
        M.micPosition(scene)[Number(target.dataset.axis)] = value;
        changed({ room: true });
      } else if (kind === "reference") {
        scene.reference_rir = target.value;
        changed({ settings: true });
        renderPolicies();
      } else if (kind === "radius") {
        scene.near_radius_m = value;
        changed();
        renderPolicies();
      } else if (kind === "model") {
        state.model = target.value;
        invalidateResult();
        applyDefaultPolicy();
        updateRunButton();
      } else if (target.dataset.source !== undefined) {
        await onSourceChange(target, Number(target.dataset.source));
      } else if (target.dataset.key) {
        onKeyChange(target);
      } else if (target.dataset.field === "parameter") {
        state.axes[Number(target.dataset.axisIndex)].parameter = target.value;
        renderSettings();
        updateRunButton();
      } else if (target.dataset.action === "toggle-heights") {
        state.heights = target.checked;
        renderSettings();
      } else if (target.dataset.action === "import") {
        await importScene(target.files[0]);
      }
    } catch (error) {
      shell().toast(t(error.message), true);
    }
  }

  async function onSourceChange(target, s) {
    const source = state.scene.sources[s];
    const field = target.dataset.field;
    if (field === "gain") source.gain_db = Number(target.value);
    else if (field === "start") source.start_s = Math.max(0, Number(target.value));
    else if (field === "repeat") source.repeat = target.checked;
    else if (field === "still") {
      state.scene = M.withStill(state.scene, s, target.checked);
      state.selected = { source: s, key: 0 };
      return changed({ settings: true, room: true });
    } else if (field === "asset") {
      if (target.value === "upload") return;
      state.assets[source.asset_id] = { sample: target.value };
    } else if (field === "upload") {
      const file = target.files[0];
      if (!file) return;
      state.assets[source.asset_id] = await window.PureSoundApp.uploadDescriptor(file);
      state.uploads[source.asset_id] = file.name;
      return changed({ settings: true });
    }
    changed();
  }

  function onKeyChange(target) {
    const [s, k] = target.dataset.key.split(":").map(Number);
    const key = state.scene.sources[s].keyframes[k];
    const value = Number(target.value);
    if (!Number.isFinite(value)) return renderSettings();
    const field = target.dataset.field;
    if (field === "time") key.time_s = value;
    else if (field === "yaw") key.yaw_deg = value;
    else if (field === "pitch") key.pitch_deg = value;
    else key.position_m["xyz".indexOf(field)] = value;
    state.selected = { source: s, key: k };
    changed({ settings: field === "time", room: true });
  }

  function onSettingClick(event) {
    if (event.target.closest('[data-aicoustics="forget"]')) {
      state.aicoustics.key = ""; state.aicoustics.enabled = false;
      renderSettings(); updateRunButton(); return;
    }
    const button = event.target.closest("button[data-action]");
    if (!button || state.job) return;
    const action = button.dataset.action;
    if (action === "roll-seed") {
      redrawMaterials(state.scene.room.room_type, Math.floor(Math.random() * 2 ** 31));
    } else if (action === "select-key") {
      const [s, k] = button.dataset.key.split(":").map(Number);
      select(s, k, { seek: true });
    } else if (action === "delete-key") {
      const [s, k] = button.dataset.key.split(":").map(Number);
      state.scene = M.withoutKeyframe(state.scene, s, k);
      state.selected = { source: s, key: Math.max(0, k - 1) };
      changed({ settings: true, room: true });
    } else if (action === "add-key") {
      addKeyframe(Number(button.dataset.source), state.time);
    } else if (action === "add-source") {
      addSource(button.dataset.kind);
    } else if (action === "remove-source") {
      removeSource(Number(button.dataset.source));
    } else if (action === "set-role") {
      state.scene = M.withRole(state.scene, Number(button.dataset.source), button.dataset.role);
      changed({ settings: true, room: true });
    } else if (action === "export") {
      download(new Blob([JSON.stringify({ scene: state.scene, assets: state.assets }, null, 2)], { type: "application/json" }), "scene.json");
    }
  }

  function addSource(kind) {
    const added = M.addSource(state.scene, kind, state.limits);
    if (!added) return;
    state.scene = added.scene;
    state.assets[added.scene.sources[added.index].asset_id] = { sample: added.sample };
    state.selected = { source: added.index, key: 0 };
    changed({ settings: true, room: true });
    $(`#world-asset-${added.index}`)?.closest("fieldset")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function removeSource(s) {
    const asset = state.scene.sources[s].asset_id;
    state.scene = M.removeSource(state.scene, s);
    if (!state.scene.sources.some((source) => source.asset_id === asset)) {
      delete state.assets[asset];
      delete state.uploads[asset];
    }
    const selected = state.selected.source;
    state.selected = selected === s ? { source: Math.min(s, state.scene.sources.length - 1), key: 0 } : { ...state.selected, source: selected > s ? selected - 1 : selected };
    changed({ settings: true, room: true });
  }

  /* A keyframe at `time`; a source that stays still moves to `position`. */
  function addKeyframe(s, time, position = null) {
    if (M.isStill(state.scene.sources[s])) {
      if (!position) return;
      moveTarget({ source: s, key: 0 }, position);
      return changed({ settings: true, room: true });
    }
    if (state.scene.sources[s].keyframes.length >= state.limits.keyframes) {
      shell().toast(t("A source can have at most {count} keyframes.", { count: state.limits.keyframes }), true);
      return;
    }
    const { scene, index } = M.withKeyframe(state.scene, s, time);
    if (position) scene.sources[s].keyframes[index].position_m.splice(0, 2, ...position);
    state.scene = scene;
    state.selected = { source: s, key: index };
    changed({ settings: true, room: true });
  }

  /* New materials for a room type and seed.  Only the room changes, so edits
   * made while the request runs are kept; a superseded answer is dropped and
   * a failure puts the panel back. */
  async function redrawMaterials(roomType, seed) {
    const ticket = ++state.materials;
    try {
      const { scene } = await api("/api/world/materials", { method: "POST", body: JSON.stringify({ scene: requestScene(), room_type: roomType, seed }) });
      if (ticket !== state.materials) return;
      Object.assign(state.scene.room, { room_type: scene.room.room_type, surfaces: scene.room.surfaces, materials: scene.room.materials });
      state.scene.seed = scene.seed;
      changed({ settings: true, room: true });
    } catch (error) {
      if (ticket !== state.materials) return;
      renderSettings();
      shell().toast(t(error.message), true);
    }
  }

  /* Selection and time ------------------------------------------------------ */
  function select(s, k, { seek: jump = false } = {}) {
    state.selected = { source: s, key: k };
    $$(".world-keys tr.is-selected").forEach((row) => row.classList.remove("is-selected"));
    $(`.world-keys tr[data-row="${s}:${k}"]`)?.classList.add("is-selected");
    drawPlan();
    drawTicks();
    if (jump) seek(state.scene.sources[s].keyframes[k].time_s, { deck: true });
    else drawNow();
  }

  function seek(time, { deck = false } = {}) {
    state.time = Math.max(0, Math.min(state.scene.duration_s, time));
    if (deck && state.report && state.deck && !state.deck.playing) state.deck.seekTo(state.time);
    drawNow();
    $$("button[data-action=add-key]").forEach((button) => { button.textContent = t("Add keyframe at {time} s", { time: fixed(state.time) }); });
  }

  function tick() {
    const playing = Boolean(state.deck?.playing && state.report);
    const clock = state.report && state.deck ? state.deck.currentTime() : null;
    if (clock !== null && shell().current() === "world" && (playing || clock !== state.audioTime)) {
      state.audioTime = clock;
      state.time = Math.min(state.scene.duration_s, clock);
      drawNow();
    }
    if (playing !== state.wasPlaying) {
      state.wasPlaying = playing;
      const play = $("#world-play");
      play.textContent = playing ? "❚❚" : "▶";
      play.setAttribute("aria-label", t(playing ? "Pause" : "Play"));
    }
    requestAnimationFrame(tick);
  }

  /* Validation -------------------------------------------------------------- */
  function requestScene() {
    const scene = structuredClone(state.scene);
    // The room's transducer poses record where each source starts.
    for (const source of scene.sources) {
      const transducer = scene.room.sources.find((item) => item.transducer_id === source.source_id);
      transducer.pose.position_m = [...source.keyframes[0].position_m];
    }
    return scene;
  }

  /* The editor's own checks at once, the server's a moment later. */
  function validate() {
    state.issues = M.keyframeIssues(state.scene, state.limits);
    state.scene.sources.forEach((source, index) => {
      if (source.role === "noise") return;
      const ref = state.assets[source.asset_id];
      const asset = ref?.upload_id ? null : state.overview.assets.find((item) => item.id === (ref?.sample || source.asset_id));
      if (asset && Math.round(asset.duration_s * 16000) > Math.round(state.scene.duration_s * 16000) - Math.round(source.start_s * 16000)) {
        state.issues.push({ kind: "clip", source: index, seconds: asset.duration_s, end: source.start_s + asset.duration_s });
      }
    });
    markIssues();
    clearTimeout(state.validateTimer);
    const ticket = ++state.validation;
    state.validateTimer = setTimeout(async () => {
      let error = null;
      try {
        await api("/api/world/validate", { method: "POST", body: JSON.stringify({ scene: requestScene(), assets: state.assets }) });
      } catch (failure) {
        error = failure.message;
      }
      if (ticket !== state.validation) return;
      state.serverError = error;
      markIssues();
      updateRunButton();
    }, 250);
  }

  function issueText(issue) {
    return ISSUES[issue.kind]({
      n: issue.key + 1,
      previous: issue.key,
      source: state.scene.sources[issue.source].source_id,
      limit: issue.kind === "speed" ? state.limits.speed_m_s : state.limits.min_mic_distance_m,
      speed: fixed(issue.speed, 1),
      duration: state.scene.duration_s,
      seconds: fixed(issue.seconds, 3),
      end: fixed(Math.ceil(issue.end * 10) / 10, 1),
    });
  }

  function markIssues() {
    const list = $("#world-issues");
    const messages = [...new Set(state.issues.map(issueText))];
    // The server's check repeats the editor's; it adds a line only when the
    // editor found nothing.
    if (state.serverError && !state.issues.length) messages.push(t(state.serverError));
    list.hidden = !messages.length;
    list.innerHTML = messages.map((message) => `<li>${esc(message)}</li>`).join("");
    $$(".world-keys tr.is-invalid").forEach((row) => row.classList.remove("is-invalid"));
    for (const issue of state.issues) $(`.world-keys tr[data-row="${issue.source}:${issue.key}"]`)?.classList.add("is-invalid");
  }

  function blocker() {
    if (state.aicoustics.enabled && !state.aicoustics.key) return t("Enter the complete ai-coustics SDK key to compare.");
    if (state.issues.length || state.serverError) return t("Fix the scene first: {problem}", { problem: state.issues.length ? issueText(state.issues[0]) : t(state.serverError) });
    if (state.tab !== "map") return "";
    const axes = sweepAxes();
    if (axes.some((axis) => axis.error)) return t("Check the limit map's values.");
    const absent = absentKind();
    if (absent) return ABSENT[absent]();
    if (axes[0].values.length * axes[1].values.length > state.limits.sweep_cells) return t("The limit map has more than {limit} cells.", { limit: state.limits.sweep_cells });
    if (!state.model) return t("Pick a model: a limit map scores a model.");
    return "";
  }

  function updateRunButton() {
    const button = $("#world-run");
    if (!state.scene) return;
    const axes = sweepAxes();
    const cells = axes[0].values.length * axes[1].values.length;
    $("#world-run-label").textContent = state.tab === "map" ? t("Run sweep · {cells} cells", { cells }) : t("Render scene");
    const reason = blocker();
    button.disabled = Boolean(reason) && !state.job;
    button.title = reason || t(state.tab === "map" ? "Run sweep (Ctrl+Enter)" : "Render (Ctrl+Enter)");
  }

  /* The stage: floor plan, 3-D view, timeline and readout ------------------- */
  function drawStage() {
    if (!state.scene) return;
    drawPlan();
    drawTicks();
    drawNow();
    if (state.view === "3d") showRoom3d();
  }

  function drawPlan() {
    const scene = state.scene;
    const dims = scene.room.dimensions_m;
    const svg = $("#world-plan");
    const width = Math.round(Math.min(PLAN.maxWidth, Math.max(PLAN.minWidth, svg.clientWidth || 640)));
    const height = Math.round(width * PLAN.aspect);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const T = M.planTransform(dims, width, height, PLAN.pad);
    state.transform = T;
    const mic = M.micPosition(scene);
    const step = Math.max(dims[0], dims[1]) > 14 ? 2 : 1;
    const grid = [];
    for (let x = 0; x <= dims[0] + 1e-9; x += step) grid.push(`<line class="world-grid-line" x1="${T.x(x)}" y1="${T.y(0)}" x2="${T.x(x)}" y2="${T.y(dims[1])}"/><text class="world-axis-label" x="${T.x(x)}" y="${T.y(0) + 15}" text-anchor="middle">${x}</text>`);
    for (let y = 0; y <= dims[1] + 1e-9; y += step) grid.push(`<line class="world-grid-line" x1="${T.x(0)}" y1="${T.y(y)}" x2="${T.x(dims[0])}" y2="${T.y(y)}"/><text class="world-axis-label" x="${T.x(0) - 7}" y="${T.y(y) + 4}" text-anchor="end">${y}</text>`);
    const obstacles = scene.room.objects.map((object) => {
      const points = object.footprint.map(([x, y]) => `${T.x(x)},${T.y(y)}`).join(" ");
      const cx = object.footprint.reduce((sum, p) => sum + p[0], 0) / object.footprint.length;
      const cy = object.footprint.reduce((sum, p) => sum + p[1], 0) / object.footprint.length;
      return `<polygon class="world-obstacle" points="${points}"><title>${esc(t("{object}, {height} m high", { object: object.object_id, height: object.z_max }))}</title></polygon><text class="world-obstacle-label" x="${T.x(cx)}" y="${T.y(cy) + 4}" text-anchor="middle">${esc(object.object_id)}</text>`;
    });
    const fast = new Set(state.issues.filter((issue) => issue.kind === "speed").map((issue) => `${issue.source}:${issue.key}`));
    const paths = scene.sources.map((source, s) => {
      const color = roleColor(scene, s);
      const keys = source.keyframes;
      const segments = keys.slice(1).map((key, i) => {
        const a = keys[i].position_m;
        const b = key.position_m;
        return `<line class="world-path${fast.has(`${s}:${i + 1}`) ? " is-invalid" : ""}" x1="${T.x(a[0])}" y1="${T.y(a[1])}" x2="${T.x(b[0])}" y2="${T.y(b[1])}" stroke="${color}"/>`;
      }).join("");
      // Keyframes on one spot share a handle (a path that returns); the
      // handle moves the selected one, or the earliest.
      const spots = new Map();
      keys.forEach((key, k) => {
        const spot = key.position_m.slice(0, 2).map((v) => v.toFixed(2)).join(",");
        spots.set(spot, [...(spots.get(spot) || []), k]);
      });
      const handles = [...spots.values()].map((members) => {
        const k = members.includes(state.selected.key) && state.selected.source === s ? state.selected.key : members[0];
        const key = keys[k];
        const selected = state.selected.source === s && members.includes(state.selected.key);
        const issue = state.issues.some((item) => item.source === s && members.includes(item.key));
        const label = members.map((member) => member + 1).join(",");
        return `<g class="world-key${selected ? " is-selected" : ""}${issue ? " is-invalid" : ""}" data-key="${s}:${k}" tabindex="0" role="button" aria-label="${esc(t("{source}, keyframe {n} at {time} s", { source: source.source_id, n: label, time: fixed(key.time_s) }))}" transform="translate(${T.x(key.position_m[0])} ${T.y(key.position_m[1])})"><rect x="${-5 - 3.5 * label.length}" y="-9" width="${10 + 7 * label.length}" height="18" rx="9" fill="${color}"/><text y="3.5" text-anchor="middle">${label}</text></g>`;
      }).join("");
      return { segments, handles };
    });
    const radius = scene.near_radius_m * T.scale;
    $("#world-plan").innerHTML = `
      <rect class="world-room" x="${T.box.left}" y="${T.box.top}" width="${T.box.width}" height="${T.box.height}"/>
      <g>${grid.join("")}</g>
      <circle class="world-near" cx="${T.x(mic[0])}" cy="${T.y(mic[1])}" r="${radius}"/>
      <text class="world-near-label" x="${T.x(mic[0])}" y="${T.y(mic[1]) - radius - 5}" text-anchor="middle">${esc(t("near region · {radius} m", { radius: scene.near_radius_m }))}</text>
      <g>${obstacles.join("")}</g>
      <g>${paths.map((path) => path.segments).join("")}</g>
      <g id="world-plan-now"></g>
      <g>${paths.map((path) => path.handles).join("")}</g>
      <g class="world-mic" data-mic tabindex="0" role="button" aria-label="${esc(t("Microphone at {x}, {y} m", { x: fixed(mic[0], 1), y: fixed(mic[1], 1) }))}" transform="translate(${T.x(mic[0])} ${T.y(mic[1])})"><circle r="7"/><circle class="world-mic-core" r="2.5"/><text x="11" y="4">${esc(t("Mic"))}</text></g>
      <text class="world-axis-title" x="${T.box.left}" y="${T.box.top + T.box.height + 30}">X · m</text>
      <text class="world-axis-title" transform="translate(${T.box.left - 24} ${T.box.top + T.box.height / 2}) rotate(-90)" text-anchor="middle">Y · m</text>
      <g class="world-scale" transform="translate(${T.box.left + T.box.width - T.scale} ${T.box.top + T.box.height + 26})"><line x1="0" x2="${T.scale}" y1="0" y2="0"/><line x1="0" x2="0" y1="-4" y2="4"/><line x1="${T.scale}" x2="${T.scale}" y1="-4" y2="4"/><text x="${T.scale / 2}" y="-6" text-anchor="middle">1 m</text></g>`;
  }

  function drawTicks() {
    const scene = state.scene;
    $("#world-timeline").max = scene.duration_s;
    $("#world-ticks").innerHTML = scene.sources.flatMap((source, s) => (M.isStill(source) ? [] : source.keyframes.map((key, k) => {
      const selected = state.selected.source === s && state.selected.key === k;
      return `<i class="${selected ? "is-selected" : ""}" style="left:${(key.time_s / scene.duration_s) * 100}%;background:${roleColor(scene, s)}"></i>`;
    }))).join("");
  }

  /* What moves with time: positions on the plan, the clock and the readout. */
  function drawNow() {
    const scene = state.scene;
    const T = state.transform;
    if (!scene || !T) return;
    $("#world-timeline").value = state.time;
    $("#world-clock").textContent = `${fixed(state.time)} / ${fixed(scene.duration_s)} s`;
    const states = scene.sources.map((source) => M.sourceState(scene, source, state.time));
    const mic = M.micPosition(scene);
    $("#world-plan-now").innerHTML = scene.sources.map((source, s) => {
      const now = states[s];
      const yaw = (now.yaw_deg * Math.PI) / 180;
      const color = roleColor(scene, s);
      const x = T.x(now.position_m[0]);
      const y = T.y(now.position_m[1]);
      // Labels near the right wall read leftward so they stay inside the plan.
      const leftward = x > T.box.left + T.box.width - 130;
      return `<g class="world-now"><line x1="${x}" y1="${y}" x2="${x + 24 * Math.cos(yaw)}" y2="${y - 24 * Math.sin(yaw)}" stroke="${color}"/><circle cx="${x}" cy="${y}" r="6" fill="${color}"/><text x="${leftward ? x - 12 : x + 12}" y="${y - 12}" text-anchor="${leftward ? "end" : "start"}">${esc(source.source_id)} · ${fixed(now.distance_m)} m</text></g>`;
    }).join("");
    $("#world-readout").innerHTML = scene.sources.map((source, s) => {
      const now = states[s];
      const facts = [`${fixed(now.distance_m)} m`];
      if (M.isTalker(source)) {
        facts.push(t("{angle}° off the mic", { angle: Math.round(offAxis(now, mic)) }));
        facts.push(t("near weight {weight}", { weight: fixed(now.near_weight) }));
      }
      return `<span class="world-readout-item"><i style="background:${roleColor(scene, s)}"></i><strong>${esc(source.source_id)}</strong>${facts.map((fact) => `<span>${esc(fact)}</span>`).join("")}</span>`;
    }).join("");
    if (state.view === "3d" && state.roomShown) {
      state.room.updateSourcePositions(states.map((now) => now.position_m), states);
      drawVolumeWaves();
    }
  }

  function drawVolumeWaves() {
    const clock = state.report && state.deck ? state.deck.currentTime() : state.time;
    state.room.updateAudioWaves(state.scene.sources.map((source) => {
      const track = state.report && state.deck?.tracks.find((item) => item.id === `source-${source.source_id}`);
      const trace = track ? state.deck.levelTrace(track) : null;
      return {
        level: M.volumeAt(trace, clock),
        fronts: M.volumeRipples(trace, clock).map((front) => ({ ...front, position: M.pose(source, Math.min(front.birth, state.scene.duration_s)).position_m })),
      };
    }));
  }

  /* Angle between where a talker faces and the direction to the microphone. */
  function offAxis(now, mic) {
    const yaw = (now.yaw_deg * Math.PI) / 180;
    const pitch = ((now.pitch_deg || 0) * Math.PI) / 180;
    const facing = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
    const toward = mic.map((v, i) => v - now.position_m[i]);
    const length = Math.hypot(...toward) || 1;
    const cosine = facing.reduce((sum, v, i) => sum + (v * toward[i]) / length, 0);
    return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
  }

  async function showRoom3d() {
    if (!state.room) {
      try {
        await import("/pipeline-room.js");
        // View and language changes can both wait on the same import. Reuse
        // the first instance instead of replacing its populated canvas.
        if (!state.room) state.room = new window.PureSoundRoomView($("#world-3d"));
      } catch {
        $("#world-3d").textContent = t("The 3-D view needs WebGL, which this browser does not offer.");
        return;
      }
    }
    if (state.roomShown) return drawNow();
    const scene = state.scene;
    state.room.show({
      room_dim: scene.room.dimensions_m,
      receiver: M.micPosition(scene),
      obstacles: scene.room.objects,
      sources: scene.sources.map((source, s) => {
        const now = M.sourceState(scene, source, state.time);
        return { label: source.source_id, position: now.position_m, roles: [M.sourceRole(scene, source)], color: roleColor(scene, s), dynamic: true, yaw_deg: now.yaw_deg, pitch_deg: now.pitch_deg, distance_m: now.distance_m };
      }),
    }, {
      colors: ROLE_COLORS,
      labels: { target: ROLE_NAMES.target(), interferer: ROLE_NAMES.interferer(), noise: ROLE_NAMES.noise(), mic: t("Microphone"), controls: `${t("drag to turn · scroll to zoom · right-drag to pan")} · ${t("After rendering, ripples show each source's level at the microphone; visual speed is slowed, not a physical wavefront.")}` },
      waves: false,
      soft: true,
    });
    state.roomShown = true;
    drawVolumeWaves();
  }

  /* Plan editing ------------------------------------------------------------ */
  function planPoint(event) {
    const svg = $("#world-plan");
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(svg.getScreenCTM().inverse());
    return state.transform.toMetres(local.x, local.y);
  }

  function clampToRoom([x, y]) {
    const dims = state.scene.room.dimensions_m;
    return [Math.min(dims[0] - 0.05, Math.max(0.05, x)), Math.min(dims[1] - 0.05, Math.max(0.05, y))].map((v) => Math.round(v * 100) / 100);
  }

  function dragTarget(node) {
    if (node.dataset.mic !== undefined) return { mic: true };
    const [source, key] = node.dataset.key.split(":").map(Number);
    return { source, key };
  }

  function onPlanPointerDown(event) {
    const node = event.target.closest("[data-key],[data-mic]");
    if (!node || state.job) return;
    event.preventDefault();
    const target = dragTarget(node);
    $("#world-plan").setPointerCapture(event.pointerId);
    if (!target.mic) select(target.source, target.key, { seek: true });
    state.drag = { ...target, moved: false };
  }

  function onPlanPointerMove(event) {
    if (!state.drag) return;
    moveTarget(state.drag, clampToRoom(planPoint(event)));
    state.drag.moved = true;
  }

  function moveTarget(target, [x, y]) {
    if (target.mic) {
      M.micPosition(state.scene).splice(0, 2, x, y);
    } else {
      state.scene.sources[target.source].keyframes[target.key].position_m.splice(0, 2, x, y);
    }
    drawPlan();
    drawNow();
  }

  function onPlanPointerUp() {
    const drag = state.drag;
    state.drag = null;
    if (drag?.moved) changed({ settings: true, room: true });
  }

  function onPlanDoubleClick(event) {
    if (state.job || event.target.closest("[data-key],[data-mic]")) return;
    const [x, y] = planPoint(event);
    const dims = state.scene.room.dimensions_m;
    if (x <= 0 || y <= 0 || x >= dims[0] || y >= dims[1]) return;
    addKeyframe(state.selected.source, state.time, clampToRoom([x, y]));
  }

  function onPlanKey(event) {
    const node = event.target.closest("[data-key],[data-mic]");
    if (node?.dataset.key && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      const [s, k] = node.dataset.key.split(":").map(Number);
      select(s, k, { seek: true });
      $(`#world-plan [data-key="${s}:${k}"]`)?.focus();
      return;
    }
    const steps = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    if (!node || !steps[event.key] || state.job) return;
    event.preventDefault();
    const target = dragTarget(node);
    const size = event.shiftKey ? 0.25 : 0.05;
    const from = target.mic ? M.micPosition(state.scene) : state.scene.sources[target.source].keyframes[target.key].position_m;
    moveTarget(target, clampToRoom([from[0] + steps[event.key][0] * size, from[1] + steps[event.key][1] * size]));
    changed({ settings: true, room: true });
    $(target.mic ? "#world-plan [data-mic]" : `#world-plan [data-key="${target.source}:${target.key}"]`)?.focus();
  }

  /* Running ------------------------------------------------------------------ */
  function renderRequest() {
    const comparison = state.aicoustics.enabled ? { enabled: true, api_key: state.aicoustics.key, model_id: state.aicoustics.model_id, enhancement_level: state.aicoustics.enhancement_level } : undefined;
    return { kind: "world_render", scene: requestScene(), assets: structuredClone(state.assets), model_id: state.model || null, provider: "cpu", ...(comparison ? { aicoustics: comparison } : {}) };
  }

  function run() {
    if (!state.scene || state.job || blocker()) return;
    if (state.tab === "map") start({ ...renderRequest(), kind: "world_sweep", axes: sweepAxes().map(({ parameter, values }) => ({ parameter, values })) });
    else start(renderRequest());
  }

  function retryCell(index) {
    if (state.job || !state.map?.request) return;
    const request = structuredClone(state.map.request);
    if (request.aicoustics?.enabled) {
      if (!state.aicoustics.key) return shell().toast(t("Enter the complete ai-coustics SDK key to compare."), true);
      request.aicoustics.api_key = state.aicoustics.key;
    }
    start({ ...request, cell_indices: [index] });
  }

  const isFinal = (status) => ["succeeded", "failed", "cancelled"].includes(status);

  async function start(request) {
    const sweep = request.kind === "world_sweep";
    state.deck?.pause();
    state.job = "starting";
    setBusy(true);
    shell().setState("world", { state: "run", label: t("Starting…"), progress: 0 });
    const began = performance.now();
    try {
      const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(request) });
      state.job = job.job_id;
      if (sweep && !request.cell_indices) {
        const saved = structuredClone(request);
        if (saved.aicoustics) delete saved.aicoustics.api_key;
        state.map = { axes: request.axes, cells: [], request: saved };
      }
      if (sweep) state.map.job_id = job.job_id;
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        const status = await api(`/api/jobs/${encodeURIComponent(job.job_id)}?summary=1`);
        const final = isFinal(status.status);
        if (sweep && status.result?.cells) showMap(status.result, { running: !final });
        if (!final) {
          shell().setState("world", { state: "run", label: PHASES[status.phase]?.() || t("Working"), progress: status.progress || 0 });
          continue;
        }
        const seconds = ((performance.now() - began) / 1000).toFixed(1);
        if (status.status === "failed") throw new Error(status.error || t("The render failed."));
        if (status.status === "cancelled") {
          shell().setState("world", { state: "idle" });
          shell().toast(t("Cancelled."));
        } else if (sweep) {
          shell().setState("world", { state: "ok", label: t("Swept {cells} cells in {seconds} s.", { cells: status.result.cells.length, seconds }) });
        } else {
          await showResult(status.result);
          shell().setState("world", { state: "ok", label: t("Rendered in {seconds} s.", { seconds }) });
        }
        break;
      }
    } catch (error) {
      shell().setState("world", { state: "err", label: t(error.message) });
    } finally {
      state.job = null;
      setBusy(false);
      if (state.map) {
        state.map.running = false;
        renderMap();
      }
    }
  }

  function cancel() {
    state.deviceRun?.abort();
    if (state.job && state.job !== "starting") api(`/api/jobs/${encodeURIComponent(state.job)}/cancel`, { method: "POST", body: "{}" }).catch(() => null);
  }

  function setBusy(busy) {
    $("#world-run").classList.toggle("is-loading", busy);
    $("#world-cancel").hidden = !busy;
    const form = $("#world-settings-form");
    if (form) form.disabled = busy;
    $("#world-layout").classList.toggle("is-busy", busy);
    updateRunButton();
  }

  /* Result ------------------------------------------------------------------- */
  async function showResult(report, { cell = null, defaults = false } = {}) {
    state.scene = structuredClone(report.scene);
    state.materials += 1;
    state.model = report.model?.id || "";
    // The audio behind each asset as rendered; older reports do not record it.
    if (report.assets) {
      state.assets = structuredClone(report.assets);
      state.uploads = Object.fromEntries(Object.entries(report.assets).filter(([, ref]) => ref.upload_id).map(([id, ref]) => [id, ref.filename || ref.upload_id]));
    }
    if (defaults) applyDefaultPolicy();
    state.selected = { source: 0, key: 0 };
    state.roomShown = false;
    renderSettings();
    validate();
    drawStage();
    state.report = report;
    state.cell = cell;
    state.deviceTrack = null;
    state.deviceNote = null;
    $("#world-empty").hidden = true;
    $("#world-results").hidden = false;
    $("#world-play").disabled = false;
    const pack = report.output_urls?.["scene-package"];
    $("#world-package").hidden = !pack;
    $("#world-package").href = pack || "#";
    renderResultText();
    await loadDeck();
    updateDeviceButton();
  }

  /* The result's words: title, cell banner, scores and notes. */
  function renderResultText() {
    const report = state.report;
    const cell = state.cell;
    $("#world-result-title").textContent = report.model ? report.model.display_name || report.model.id : t("Room only, no model");
    const banner = $("#world-cell-banner");
    banner.hidden = !cell;
    banner.innerHTML = cell ? `<span>${esc(t("Limit-map cell: {x} × {y}", { x: axisValue(0, cell.x), y: axisValue(1, cell.y) }))}</span><button class="link-button" type="button" data-action="back-to-map">${esc(t("Back to the map"))}</button>` : "";
    renderScores();
  }

  function axisValue(a, value) {
    return `${parameterName(state.map?.axes?.[a]?.parameter)} ${value}`;
  }

  function renderPolicies() {
    $$("[data-world-policy]").forEach((group) => {
      group.innerHTML = Object.entries(POLICIES).map(([key, text]) => `<button type="button" role="radio" data-policy="${key}" aria-checked="${key === state.policy}">${esc(text()[0])}</button>`).join("");
    });
    const scene = state.scene;
    const reference = REFERENCES[scene?.reference_rir]?.()[1] || "";
    $("#world-policy-help").textContent = POLICIES[state.policy]({ radius: scene?.near_radius_m ?? 1, reference })[1];
  }

  function renderScores() {
    const report = state.report;
    const metrics = report.metrics[state.policy];
    const modelRun = Boolean(report.model);
    const stats = [];
    const notes = [];
    if (!modelRun) notes.push(t("No model ran: the output is the microphone itself, so these scores describe the input."));
    if (!metrics) {
      notes.push(t("This render has no such reference; render the scene again to score against it."));
    } else if (metrics.output) {
      const before = metrics.input?.si_sdr_db;
      const after = metrics.output.si_sdr_db;
      const change = after - before;
      stats.push(stat(t("SI-SDR · output"), `${fixed(after)} dB`));
      if (modelRun) stats.push(stat(t("SI-SDR · microphone"), `${fixed(before)} dB`), stat(t("Change"), `${change > 0 ? "+" : ""}${fixed(change)} dB`, change > 0.5 ? "is-good" : change < -0.5 ? "is-bad" : ""));
      stats.push(stat("STOI", fixed(metrics.output.stoi, 3)), stat("PESQ", fixed(metrics.output.pesq_wb)));
    } else {
      const levels = metrics.windows.map((window) => window.residual_dbfs).filter(Number.isFinite);
      const mean = levels.length ? levels.reduce((a, b) => a + b, 0) / levels.length : NaN;
      stats.push(stat(t("Reference"), t("silent")), stat(t("Output level"), `${fixed(mean, 1)} dBFS`));
      notes.push(SILENT[state.policy]());
    }
    if (state.deviceNote) notes.push(t("This device's output differs from the server's by {difference} dB (energy of the difference against the output), at {rtf}× real time on {threads} threads.", state.deviceNote));
    $("#world-stats").innerHTML = stats.join("");
    $("#world-result-note").textContent = notes.join(" ");
    renderComparison();
  }

  function renderComparison() {
    const comparison = state.report?.comparisons?.aicoustics;
    const host = $("#world-comparison");
    host.hidden = !comparison;
    if (!comparison) return;
    if (comparison.status !== "succeeded") {
      host.innerHTML = `<p class="form-note is-error">${esc(t("ai-coustics comparison unavailable: {reason}", { reason: t(comparison.error || "Unknown error") }))}</p>`;
      return;
    }
    const rows = [
      { label: state.report.model?.display_name || t("Microphone"), metrics: state.report.metrics[state.policy], rtf: state.report.processing?.rtf, delay: null },
      { label: `ai-coustics · ${comparison.display_name}`, metrics: comparison.metrics[state.policy], rtf: comparison.rtf, delay: comparison.audio_delay_ms },
    ];
    const body = rows.map((row) => {
      const out = row.metrics?.output;
      const change = Number.isFinite(out?.si_sdr_db) && Number.isFinite(row.metrics?.input?.si_sdr_db) ? out.si_sdr_db - row.metrics.input.si_sdr_db : null;
      return `<tr><th scope="row">${esc(row.label)}</th><td>${fixed(out?.si_sdr_db)}</td><td>${fixed(change)}</td><td>${fixed(out?.stoi, 3)}</td><td>${fixed(out?.pesq_wb)}</td><td>${fixed(row.rtf, 3)}</td><td>${fixed(row.delay, 1)}</td></tr>`;
    }).join("");
    host.innerHTML = `<p class="eyebrow">${esc(t("Same input · same reference"))}</p><div class="world-comparison-scroll"><table><caption>${esc(t("Score against"))} · ${esc(POLICIES[state.policy]()[0])}</caption><thead><tr><th>${esc(t("Model"))}</th><th>SI-SDR · dB</th><th>Δ · dB</th><th>STOI</th><th>PESQ</th><th>RTF</th><th>${esc(t("SDK delay · ms"))}</th></tr></thead><tbody>${body}</tbody></table></div>
      <p class="field-hint">${esc(t("SDK delay is compensated in playback and scoring. RTF measures processing calls, excluding setup and download. Silent references have no SI-SDR."))}</p>
      ${rows[0].metrics?.output ? "" : `<p class="field-hint">${esc(t("Silent reference · output level: PureSound {local} dBFS · ai-coustics {other} dBFS", { local: fixed(state.report.output_rms_dbfs, 1), other: fixed(comparison.output_rms_dbfs, 1) }))}</p>`}`;
  }

  function stat(label, value, tone = "") {
    return `<div class="stat ${tone}"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
  }

  async function loadDeck() {
    const report = state.report;
    const scene = report.scene;
    const urls = report.output_urls;
    state.deck ??= new window.PureSoundCompareDeck($("#world-deck"), { listening: true });
    const modelRun = Boolean(report.model);
    const comparison = report.comparisons?.aicoustics;
    const tracks = [
      // Labels are English keys: the deck translates them and follows a language switch.
      { id: "input", label: "Microphone", color: TRACK_COLORS.input, hint: "what the microphone picked up", levelReference: true, url: urls.input },
      modelRun && { id: "aligned", label: "Model output", color: TRACK_COLORS.aligned, hint: report.model.display_name || report.model.id, url: urls.aligned },
      state.deviceTrack,
      modelRun && { id: "removed", label: "Removed", color: TRACK_COLORS.removed, hint: "microphone − output", diagnostic: true, url: urls.removed },
      comparison?.status === "succeeded" && { id: "aicoustics", label: "ai-coustics", color: TRACK_COLORS.aicoustics, hint: comparison.display_name, defaultVisible: true, url: urls.aicoustics },
      comparison?.status === "succeeded" && { id: "aicoustics-removed", label: "ai-coustics removed", color: TRACK_COLORS.removed, hint: "microphone − ai-coustics", diagnostic: true, url: urls["aicoustics-removed"] },
      scene.sources.some((source) => source.role === "target") && { id: "reference-target", label: "Target talkers", color: TRACK_COLORS.target, hint: "reference: every target talker", url: urls["reference-target"] },
      scene.sources.some(M.isTalker) && { id: "reference-speech", label: "All speech", color: TRACK_COLORS.speech, hint: "reference: every talker", url: urls["reference-speech"] },
      scene.sources.some(M.isTalker) && { id: "reference-near", label: "Near region", color: TRACK_COLORS.near, hint: "reference: talkers inside the near radius", url: urls["reference-near"] },
      ...scene.sources.map((source, s) => ({ id: `source-${source.source_id}`, label: source.source_id, color: roleColor(scene, s), hint: "this source alone, at the microphone", url: urls[`source-${source.source_id}`] })),
    ].filter((track) => track && (track.url || track.data));
    state.deck.setTracks(tracks);
    await Promise.allSettled(tracks.map((track) => state.deck.loadTrack(track.id, track.data || track.url)));
    if (state.report !== report) return;
    if (comparison?.status === "succeeded" && state.aicoustics.autoShow) {
      state.deck.setTrackVisible("aicoustics", true);
      state.aicoustics.autoShow = false;
    }
    const preferred = modelRun ? "aligned" : "input";
    state.deck.select(state.deck.visibleTracks.has(preferred) ? preferred : [...state.deck.visibleTracks][0]);
    state.deck.setCurves(curves(report));
    drawNow();
    if (window.PureSoundTranscribe) {
      state.transcriber ??= new window.PureSoundTranscribe($("#world-transcribe"), state.deck);
      state.transcriber.refresh();
    }
  }

  /* Lanes under the deck: the score per second and each talker's distance. */
  function curves(report) {
    const metrics = report.metrics[state.policy];
    const modelRun = Boolean(report.model);
    const list = [];
    if (metrics?.output) {
      const values = metrics.windows.map((window) => (modelRun ? window.improvement_db : window.input_si_sdr_db) ?? NaN);
      list.push(modelRun
        ? { label: t("SI-SDR change"), hint: t("per 1 s window, −12 to +12 dB"), values, hopSeconds: 0.5, offsetSeconds: 0.5, range: [-12, 12], color: TRACK_COLORS.windows }
        : { label: t("SI-SDR · microphone"), hint: t("per 1 s window, −20 to +20 dB"), values, hopSeconds: 0.5, offsetSeconds: 0.5, range: [-20, 20], color: TRACK_COLORS.windows });
    }
    const comparison = report.comparisons?.aicoustics?.metrics?.[state.policy];
    if (comparison?.output) list.push({
      label: "ai-coustics · SI-SDR change", hint: t("per 1 s window, −12 to +12 dB"),
      values: comparison.windows.map((window) => window.improvement_db ?? NaN),
      hopSeconds: 0.5, offsetSeconds: 0.5, range: [-12, 12], color: TRACK_COLORS.aicoustics,
    });
    const times = report.timeline.times_s;
    const hop = 0.05;
    const grid = Array.from({ length: Math.floor(report.scene.duration_s / hop) + 1 }, (_, i) => i * hop);
    const top = Math.ceil(Math.max(1, ...Object.values(report.timeline.sources).flatMap((source) => source.distance_m)));
    report.scene.sources.forEach((source, s) => {
      const track = report.timeline.sources[source.source_id];
      if (!track || !M.isTalker(source)) return;
      list.push({
        label: t("Distance · {source}", { source: source.source_id }),
        hint: t("metres from the microphone, 0 to {top} m", { top }),
        values: grid.map((time) => interpolate(times, track.distance_m, time)),
        hopSeconds: hop,
        range: [0, top],
        color: roleColor(report.scene, s),
      });
    });
    return list;
  }

  function interpolate(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) {
      if (xs[i] >= x) return ys[i - 1] + ((ys[i] - ys[i - 1]) * (x - xs[i - 1])) / (xs[i] - xs[i - 1] || 1);
    }
    return ys[ys.length - 1];
  }

  /* On this device: the same model in this browser, as one more track. */
  async function updateDeviceButton() {
    const button = $("#world-device");
    const model = state.report?.model?.id;
    button.hidden = !window.PureSoundDevice || !model;
    if (button.hidden) return;
    state.device ??= await window.PureSoundDevice.status().catch(() => null);
    const device = state.device;
    const usable = Boolean(device?.ready && device.models.includes(model));
    button.disabled = !usable;
    button.title = usable
      ? t("Run the same model in this browser and compare it with the server's output.")
      : device?.ready ? t("{model} has no build for this device.", { model }) : device?.reason || t("On-device inference is unavailable here.");
  }

  async function runOnDevice() {
    const report = state.report;
    const model = report?.model?.id;
    if (!model || state.deviceRun) return;
    const button = $("#world-device");
    button.classList.add("is-loading");
    state.deviceRun = new AbortController();
    $("#world-cancel").hidden = false;
    try {
      const samplesOf = async (url) => (await window.PureSoundDevice.decode(await (await fetch(url)).blob(), { sampleRate: 16000 })).samples;
      const result = await window.PureSoundDevice.process(await samplesOf(report.output_urls.input), {
        model,
        sampleRate: 16000,
        signal: state.deviceRun.signal,
        onProgress: (fraction, phase) => shell().setState("world", { state: "run", label: t(phase === "load" ? "Loading the model on this device" : "Running on this device"), progress: fraction }),
      });
      const difference = nrmsDb(result.output, await samplesOf(report.output_urls.aligned));
      if (state.report !== report) {
        shell().setState("world", { state: "idle" });
        return;
      }
      const wav = window.PureSoundPipelineModel.encodeFloatWav(result.output, 16000);
      state.deviceTrack = { id: "device", label: "This device", color: TRACK_COLORS.device, hint: "the same model in this browser", data: new Blob([wav], { type: "audio/wav" }) };
      state.deviceNote = { difference: fixed(difference, 1), rtf: fixed(result.rtf), threads: result.threads };
      await loadDeck();
      state.deck.select("device");
      renderScores();
      shell().setState("world", { state: "ok", label: t("Ran on this device.") });
    } catch (error) {
      shell().setState("world", { state: error.cancelled ? "idle" : "err", label: error.cancelled ? "" : t(error.message) });
    } finally {
      state.deviceRun = null;
      button.classList.remove("is-loading");
      $("#world-cancel").hidden = !state.job;
    }
  }

  function nrmsDb(estimate, reference) {
    const length = Math.min(estimate.length, reference.length);
    let error = 0;
    let energy = 0;
    for (let i = 0; i < length; i++) {
      error += (estimate[i] - reference[i]) ** 2;
      energy += reference[i] ** 2;
    }
    return 10 * Math.log10(Math.max(error, 1e-20) / Math.max(energy, 1e-20));
  }

  /* Limit map ------------------------------------------------------------------ */
  function showMap(result, { running }) {
    const previous = state.map?.cells?.length ? state.map : null;
    const merged = M.mergeCells(previous, result);
    state.map = {
      ...merged,
      request: state.map?.request || result.request,
      job_id: state.map?.job_id || result.job_id,
      running,
    };
    renderMap();
  }

  function cellScore(cell) {
    const full = cell.result?.metrics?.[state.policy];
    const local = full?.output?.si_sdr_db ?? cell.si_sdr_db?.[state.policy] ?? null;
    const other = cell.result?.comparisons?.aicoustics?.metrics?.[state.policy]?.output?.si_sdr_db ?? cell.aicoustics_si_sdr_db?.[state.policy] ?? null;
    if (state.mapProvider === "difference") return { output: Number.isFinite(local) && Number.isFinite(other) ? other - local : null, input: null };
    return { output: state.mapProvider === "aicoustics" ? other : local, input: full?.input?.si_sdr_db ?? null };
  }

  function parameterName(parameter) {
    return PARAMETERS[parameter]?.()[0] || parameter || "";
  }

  function renderMap() {
    const map = state.map;
    const compared = Boolean(map?.request?.aicoustics?.enabled || map?.cells?.some((cell) => cell.result?.comparisons?.aicoustics || cell.aicoustics_status));
    $("#world-map-provider-row").hidden = !compared;
    if (!compared) state.mapProvider = "puresound";
    $("#world-map-provider").value = state.mapProvider;
    const has = Boolean(map?.axes && map.cells?.length);
    $("#world-map-empty").hidden = has;
    $("#world-map").hidden = !has;
    $("#world-map-legend").hidden = !has;
    $("#world-map-title").textContent = "";
    if (!has) return;
    const [across, down] = map.axes;
    const scores = new Map(map.cells.map((cell) => [cell.index, cellScore(cell)]));
    const values = [...scores.values()].map((score) => score.output);
    const done = map.cells.filter((cell) => cell.status === "succeeded").length;
    const failed = map.cells.filter((cell) => cell.status === "failed").length;
    const rendering = map.running ? map.cells.find((cell) => cell.status === "queued")?.index : undefined;
    $("#world-map-title").textContent = `${parameterName(across.parameter)} × ${parameterName(down.parameter)} · ${t("{done} of {total} cells", { done, total: map.cells.length })}${failed ? ` · ${t("{count} failed", { count: failed })}` : ""}`;
    const grid = $("#world-map");
    grid.style.setProperty("--columns", across.values.length);
    const head = `<div class="world-map-corner"><span>${esc(parameterName(across.parameter))} →</span><span>${esc(parameterName(down.parameter))} ↓</span></div>${across.values.map((value) => `<div class="world-map-head">${esc(value)}</div>`).join("")}`;
    const rows = down.values.map((rowValue, row) => `<div class="world-map-head is-row">${esc(rowValue)}</div>${across.values.map((_, column) => {
      const index = column * down.values.length + row;
      const cell = map.cells.find((item) => item.index === index);
      if (!cell) return `<div class="world-map-cell is-missing"></div>`;
      const score = scores.get(index);
      const shade = M.shade(score.output, values);
      const status = index === rendering ? "running" : cell.status;
      const change = Number.isFinite(score.output) && Number.isFinite(score.input) ? score.output - score.input : null;
      const body = status === "succeeded"
        ? `<strong>${Number.isFinite(score.output) ? `${fixed(score.output, 1)} dB` : esc(t(state.mapProvider === "puresound" ? "silent reference" : "no comparison score"))}</strong>${change === null ? "" : `<small>Δ ${change > 0 ? "+" : ""}${fixed(change, 1)} dB</small>`}`
        : `<small>${esc(CELL_STATES[status]?.() || status)}</small>`;
      const comparisonFailed = (cell.result?.comparisons?.aicoustics?.status || cell.aicoustics_status) === "failed";
      const retry = ["failed", "cancelled"].includes(status) || comparisonFailed ? `<button class="button button-ghost button-small" type="button" data-retry="${index}">${esc(t("Retry"))}</button>` : "";
      const error = cell.error || cell.result?.comparisons?.aicoustics?.error || cell.aicoustics_error;
      const label = t("Open cell {x} × {y}", { x: axisValue(0, cell.x), y: axisValue(1, cell.y) });
      return `<div class="world-map-cell is-${esc(status)}"${shade === null ? "" : ` style="--shade:${Math.round(10 + shade * 60)}%"`}${error ? ` title="${esc(t(error))}"` : ""}><button class="world-map-open" type="button" data-cell="${index}" aria-label="${esc(label)}"${status === "succeeded" ? "" : " disabled"}>${body}</button>${retry}</div>`;
    }).join("")}`).join("");
    // Each poll redraws the map; keep the keyboard where it was.
    const focused = document.activeElement?.closest?.("#world-map [data-cell], #world-map [data-retry]");
    const focusKey = focused ? (focused.dataset.cell !== undefined ? `[data-cell="${focused.dataset.cell}"]` : `[data-retry="${focused.dataset.retry}"]`) : null;
    grid.innerHTML = head + rows;
    if (focusKey) grid.querySelector(focusKey)?.focus();
    const finite = values.filter(Number.isFinite);
    $("#world-map-legend").innerHTML = finite.length ? `<span>${esc(t("lower"))} ${fixed(Math.min(...finite), 1)} dB</span><i></i><span>${fixed(Math.max(...finite), 1)} dB ${esc(t("higher"))}</span>` : "";
  }

  async function onMapClick(event) {
    const retry = event.target.closest("[data-retry]");
    if (retry) return retryCell(Number(retry.dataset.retry));
    const open = event.target.closest("[data-cell]");
    if (!open) return;
    const index = Number(open.dataset.cell);
    let cell = state.map.cells.find((item) => item.index === index);
    if (!cell.result && state.map.job_id) {
      const full = await api(`/api/jobs/${encodeURIComponent(state.map.job_id)}`).catch(() => null);
      cell = full?.result?.cells?.find((item) => item.index === index) || cell;
    }
    if (!cell.result) return shell().toast(t("This cell opens once the sweep has finished."));
    setTab("scene");
    await showResult(cell.result, { cell });
    $("#world-results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* Files ------------------------------------------------------------------------ */
  function download(blob, name) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  }

  async function importScene(file) {
    if (!file) return;
    const assets = {};
    const uploads = {};
    let data;
    if (file.name.endsWith(".zip")) data = await readPackage(file, assets, uploads);
    else {
      data = JSON.parse(await file.text());
      // A package's scene.json points at audio files inside the package.
      if (Object.values(data.assets || {}).some((asset) => asset?.sha256)) throw new Error(t("This scene.json belongs to a scene package; import the .zip to bring its audio along."));
      Object.assign(assets, data.assets || {});
    }
    const { scene } = await api("/api/world/validate", { method: "POST", body: JSON.stringify({ scene: data.scene ?? data, assets }) });
    state.scene = scene;
    state.materials += 1;
    state.assets = assets;
    state.uploads = uploads;
    state.preset = "";
    state.selected = { source: 0, key: 0 };
    applyDefaultPolicy();
    changed({ settings: true, room: true });
    shell().toast(t("Scene imported."));
  }

  /* A scene package: scene.json plus each source's audio, checked against the
   * SHA-256 the package recorded for it. */
  async function readPackage(file, assets, uploads) {
    const limit = 64 * 1024 * 1024;
    if (file.size > limit) throw new Error(t("The scene package is too large."));
    const { unzipSync } = await import("/vendor/fflate/fflate.mjs");
    let expanded = 0;
    const files = unzipSync(new Uint8Array(await file.arrayBuffer()), {
      filter: (entry) => {
        expanded += entry.originalSize;
        if (expanded > 2 * limit) throw new Error(t("The scene package is too large."));
        return entry.name === "scene.json" || /^asset-\d+\.wav$/.test(entry.name);
      },
    });
    if (!files["scene.json"]) throw new Error(t("The package has no scene.json."));
    const data = JSON.parse(new TextDecoder().decode(files["scene.json"]));
    for (const [id, asset] of Object.entries(data.assets || {})) {
      const bytes = files[asset.file];
      if (!bytes) throw new Error(t("The package is missing {file}.", { file: asset.file }));
      const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", wavData(bytes)))].map((b) => b.toString(16).padStart(2, "0")).join("");
      if (digest !== asset.sha256) throw new Error(t("{file} does not match the package's checksum.", { file: asset.file }));
      assets[id] = await window.PureSoundApp.uploadDescriptor(new File([bytes], asset.file, { type: "audio/wav" }));
      uploads[id] = asset.file;
    }
    return data;
  }

  /* The sample bytes of a RIFF/WAVE file's data chunk. */
  function wavData(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let position = 12; position + 8 <= bytes.length;) {
      const size = view.getUint32(position + 4, true);
      if (position + 8 + size > bytes.length) break;
      if (String.fromCharCode(...bytes.subarray(position, position + 4)) === "data") return bytes.subarray(position + 8, position + 8 + size);
      position += 8 + size + (size % 2);
    }
    throw new Error(t("A package audio file is not a valid WAV file."));
  }

  /* Language and public API ------------------------------------------------------- */
  window.addEventListener("puresound:lang", () => {
    if (!state.scene) return;
    renderSettings();
    renderPolicies();
    setView(state.view);
    state.roomShown = false;
    drawStage();
    updateRunButton();
    renderMap();
    if (state.report) {
      renderResultText();
      state.deck?.setCurves(curves(state.report));
      updateDeviceButton();
    }
  });

  window.PureSoundWorld = {
    show,
    run,
    async open(result) {
      await show();
      if (result.cells) {
        state.map = null;
        showMap(result, { running: false });
        setTab("map");
      } else {
        setTab("scene");
        await showResult(result, { defaults: true });
      }
    },
  };
})();
