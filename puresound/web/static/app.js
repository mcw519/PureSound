/* PureSound web client.  It is deliberately framework-free so the UI can be
 * served by the same small standard-library process as the Model Zoo API. */

(() => {
  "use strict";

  const state = {
    models: [],
    voiceModels: [],
    svModels: [],
    files: {},
    audioPanels: {},
    activeJobs: {},
    measurementFiles: {},
    measurementReport: null,
    measureVariants: null,
    historyFilter: "all",
    uploads: new Map(),
    acceptors: {},
    samples: null,
    clipIndex: 0,
    inputMode: "file",
    live: null,
    jobs: [],
    dropzoneFiles: {},
    runner: "server",
    device: null,
    deviceRun: null,
    blobUrls: new Set(),
  };

  /* Onset guard: catalog parameter name -> control id suffix.  The switch is a
   * per-request override of the manifest, so the browser only sends what the
   * user actually chose. */
  const ONSET_GUARD_KNOBS = [
    { parameter: "onset_guard_t_arm_s", suffix: "t-arm" },
    { parameter: "onset_guard_t_forget_s", suffix: "t-forget" },
    { parameter: "onset_guard_tau_dn_s", suffix: "tau-dn" },
    { parameter: "onset_guard_margin_db", suffix: "margin" },
  ];

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  const shell = () => window.PureSoundShell;
  const t = (key, vars) => (window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);
  // Which screen shows each run's state under its header.
  const SCREEN_OF = { voice: "playground", sv: "verify", measure: "compare" };

  function showToast(message, isError = false) {
    shell().toast(message, isError);
  }

  /* Put `key` (translated) into an element.  A plain key stays marked with
   * data-i18n so a language switch redoes it; a filled-in one cannot be
   * redone, so its mark is dropped. */
  function setKey(element, key, vars) {
    if (!element) return;
    if (vars) element.removeAttribute("data-i18n");
    else element.setAttribute("data-i18n", key);
    element.textContent = t(key, vars);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }
    if (!response.ok) {
      throw new Error(payload?.error?.message || t("Request failed ({status})", { status: response.status }));
    }
    return payload;
  }

  function modelIsDefault(model) { return model.roles?.includes("default"); }
  const ENHANCEMENT_TASKS = ["voice_isolation", "noise_suppression"];
  const TASK_LABELS = { voice_isolation: "Voice isolation", noise_suppression: "Noise suppression", speaker_embedding: "Speaker verification" };
  function taskLabel(task) { return TASK_LABELS[task] ? t(TASK_LABELS[task]) : task; }

  function renderModels() {
    const search = $("#model-search").value.trim().toLowerCase();
    const task = $("#task-filter").value;
    const lifecycle = $("#lifecycle-filter").value;
    const filtered = state.models.filter((model) => {
      const haystack = [model.id, model.display_name, model.description, model.task].join(" ").toLowerCase();
      return (!search || haystack.includes(search)) && (task === "all" || model.task === task) && (lifecycle === "all" || model.lifecycle === lifecycle);
    });
    $("#model-count").textContent = t("{shown} of {total} models", { shown: filtered.length, total: state.models.length });
    const grid = $("#model-grid");
    if (!filtered.length) {
      grid.innerHTML = `<div class="empty-state">${escapeHtml(t("No models match these filters."))}</div>`;
      return;
    }
    grid.innerHTML = filtered.map((model) => {
      const artifact = model.artifacts?.find((item) => item.variant === model.default_variant) || model.artifacts?.[0];
      const symbolClass = model.task === "speaker_embedding" ? " sv" : "";
      const badges = [
        `<span class="badge badge-task${model.task === "speaker_embedding" ? " sv" : ""}">${escapeHtml(taskLabel(model.task))}</span>`,
        ...(model.roles || []).filter((role) => ["default", "candidate", "reference", "diagnostic"].includes(role)).map((role) => `<span class="badge badge-${escapeHtml(role)}">${escapeHtml(t(role))}</span>`),
      ].join("");
      return `<article class="model-card${modelIsDefault(model) ? " is-default" : ""}">
        <div class="model-card-top"><div class="model-symbol${symbolClass}">${model.task === "speaker_embedding" ? "◌" : "∿"}</div><span class="badge${model.runnable ? " badge-quiet" : ""}">${escapeHtml(model.runnable ? t(model.lifecycle) : t("Reserved"))}</span></div>
        <div style="margin-top:12px"><h3>${escapeHtml(model.display_name)}</h3><p class="model-card-subtitle">${escapeHtml(model.id)}</p></div>
        <div class="model-badges">${badges}</div>
        <p class="model-description" data-clamp>${escapeHtml(model.description || t("Catalog-backed ONNX inference model."))}</p>
        <button class="card-more" type="button" data-more hidden>${escapeHtml(t("More"))}</button>
        <div class="model-footer"><div class="model-meta"><span>${escapeHtml(t("Artifact · sample rate"))}</span><strong>${escapeHtml(artifact?.filename || "—")} · ${escapeHtml(model.sample_rate ? `${model.sample_rate / 1000} kHz` : "—")}</strong></div><button class="card-link" data-inspect="${escapeHtml(model.id)}" type="button">${escapeHtml(t("Details"))} →</button></div>
      </article>`;
    }).join("");
    $$('[data-inspect]', grid).forEach((button) => button.addEventListener("click", () => openModelDialog(button.dataset.inspect)));
    // Long descriptions fold to four lines; "More" shows the rest.
    $$(".model-card", grid).forEach((card) => {
      const more = $("[data-more]", card);
      more.addEventListener("click", () => {
        const open = card.classList.toggle("is-expanded");
        more.textContent = t(open ? "Less" : "More");
      });
    });
    requestAnimationFrame(() => $$(".model-card", grid).forEach((card) => {
      const text = $("[data-clamp]", card);
      $("[data-more]", card).hidden = text.scrollHeight <= text.clientHeight + 2;
    }));
  }

  function populateSelect(select, models, { includeVariants = false } = {}) {
    if (!select) return;
    select.innerHTML = models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.display_name)}${modelIsDefault(model) ? ` · ${escapeHtml(t("default"))}` : ""}</option>`).join("");
    // Several tasks can share one select (the enhancement tab lists every
    // audio-in/audio-out task); prefer the voice-isolation default, then any default.
    if (models.length) select.value = models.find((model) => modelIsDefault(model) && model.task === "voice_isolation")?.id || models.find(modelIsDefault)?.id || models[0].id;
    if (includeVariants) populateVoiceVariants();
  }

  function selectedModel(selectId) {
    const id = $(selectId)?.value;
    return state.models.find((model) => model.id === id) || null;
  }

  function populateVoiceVariants() {
    const model = selectedModel("#voice-model");
    const select = $("#voice-variant");
    if (!model || !select) return;
    select.innerHTML = (model.artifacts || []).map((artifact) => `<option value="${escapeHtml(artifact.variant)}">${escapeHtml(artifact.variant)}${artifact.variant === model.default_variant ? ` · ${escapeHtml(t("default"))}` : ""}</option>`).join("");
    select.value = model.default_variant || model.artifacts?.[0]?.variant || "default";
    const dry = model.recommended_inference?.dry_blend;
    if (typeof dry === "number") {
      $("#voice-dry-blend").value = dry;
      $("#voice-dry-output").value = dry.toFixed(2);
      $("#voice-dry-output").textContent = dry.toFixed(2);
    }
    $("#voice-sample-rate").textContent = model.sample_rate ? `${model.sample_rate / 1000} kHz` : "—";
    $("#voice-channels").textContent = model.channels === 1 ? t("Mono") : t("{n} channels", { n: model.channels || "—" });
    $("#voice-delay").textContent = t(model.capabilities?.realtime_streaming ? "Removed on display" : "None");
    if (!model.capabilities?.auxiliary_heads) $("#voice-collect-extras").checked = false;
    syncExtrasRow();
    applyOnsetGuardDefaults("voice", model);
    populateStageSelect();
  }

  /* Auxiliary heads come back from the server's runtime only. */
  function syncExtrasRow() {
    $("#voice-collect-extras-row").hidden = onDevice() || !selectedModel("#voice-model")?.capabilities?.auxiliary_heads;
  }

  /* "Run first": any other runnable enhancement model at the same rate. */
  function populateStageSelect() {
    const select = $("#voice-stage");
    const model = selectedModel("#voice-model");
    if (!select || !model) return;
    const current = select.value;
    const options = state.voiceModels.filter((item) => item.id !== model.id && item.sample_rate === model.sample_rate);
    select.innerHTML = `<option value="">${escapeHtml(t("Nothing — the input as it is"))}</option>` + options.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.display_name)}</option>`).join("");
    select.value = options.some((item) => item.id === current) ? current : "";
  }

  function syncOnsetGuardKnobs(prefix) {
    const toggle = $(`#${prefix}-onset-guard`);
    const knobs = $(`#${prefix}-onset-knobs`);
    if (!toggle || !knobs) return;
    knobs.disabled = !toggle.checked;
  }

  function applyOnsetGuardDefaults(prefix, model) {
    const toggle = $(`#${prefix}-onset-guard`);
    if (!toggle) return;
    const specs = model?.parameters || {};
    const guardSpec = specs.onset_guard;
    toggle.disabled = !guardSpec;
    toggle.checked = Boolean(guardSpec?.default);
    ONSET_GUARD_KNOBS.forEach(({ parameter, suffix }) => {
      const input = $(`#${prefix}-onset-${suffix}`);
      const spec = specs[parameter];
      if (!input || !spec) return;
      if (spec.minimum != null) input.min = spec.minimum;
      if (spec.maximum != null) input.max = spec.maximum;
      if (spec.default != null) input.value = spec.default;
      if (spec.description) input.title = spec.description;
    });
    syncOnsetGuardKnobs(prefix);
  }

  function onsetGuardParameters(prefix) {
    const toggle = $(`#${prefix}-onset-guard`);
    if (!toggle || toggle.disabled) return {};
    const parameters = { onset_guard: toggle.checked };
    if (!toggle.checked) return parameters;
    ONSET_GUARD_KNOBS.forEach(({ parameter, suffix }) => {
      const value = Number($(`#${prefix}-onset-${suffix}`)?.value);
      if (Number.isFinite(value)) parameters[parameter] = value;
    });
    return parameters;
  }

  function onsetGuardSummary(guard) {
    if (!guard) return t("off");
    return [
      t("on"),
      t("arm {value}", { value: measurementValue(guard.t_arm_s, 2, " s") }),
      t("forget {value}", { value: measurementValue(guard.t_forget_s, 1, " s") }),
      t("release {value}", { value: measurementValue(guard.tau_dn_s, 2, " s") }),
      t("margin {value}", { value: measurementValue(guard.margin_db, 1, " dB") }),
    ].join(" · ");
  }

  function updateRuntimeStatus(health) {
    const ok = health.status === "ok";
    $("#runtime-status").textContent = t(ok ? "Ready" : "Unavailable");
    $("#runtime-dot").className = `status-dot ${ok ? "is-ok" : "is-err"}`;
    state.runtimeMeta = t("{n} runnable", { n: health.runnable_models ?? 0 });
    $("#runtime-meta").textContent = state.runtimeMeta;
  }

  function updateProviderAvailability(health) {
    const capabilities = health?.provider_capabilities;
    if (!capabilities) return;
    const labels = { cuda: "CUDA", mps: "Apple MPS · CoreML" };
    ["#voice-provider", "#sv-provider"].forEach((selector) => {
      const select = $(selector);
      if (!select) return;
      Object.entries(labels).forEach(([value, label]) => {
        const option = select.querySelector(`option[value="${value}"]`);
        if (!option) return;
        const available = value === "mps" ? Boolean(capabilities.coreml) : Boolean(capabilities[value]);
        option.disabled = !available;
        option.textContent = available ? label : `${label} · ${t("unavailable")}`;
      });
    });
    const preferred = capabilities.cuda ? "CUDA" : capabilities.coreml ? "CoreML" : "CPU";
    $("#runtime-meta").textContent = [state.runtimeMeta, preferred].filter(Boolean).join(" · ");
  }

  async function loadCatalog() {
    try {
      const [health, catalog] = await Promise.all([api("/api/health"), api("/api/models?include_empty=1")]);
      state.models = catalog.models || [];
      state.voiceModels = state.models.filter((model) => ENHANCEMENT_TASKS.includes(model.task) && model.runnable);
      state.svModels = state.models.filter((model) => model.task === "speaker_embedding" && model.runnable);
      state.health = health;
      updateRuntimeStatus(health);
      updateProviderAvailability(health);
      renderModels();
      loadGateRecords();
      populateSelect($("#voice-model"), state.voiceModels, { includeVariants: true });
      populateSelect($("#sv-model"), state.svModels);
      populateVoiceVariants();
      renderRunner();
      applySvDefaults();
      renderMeasurementModels();
    } catch (error) {
      $("#runtime-status").textContent = t("Offline");
      $("#runtime-dot").className = "status-dot is-err";
      $("#runtime-meta").textContent = t("Start puresound web");
      $("#model-grid").innerHTML = `<div class="empty-state"><strong>${escapeHtml(t("Unable to load the catalog."))}</strong><span>${escapeHtml(error.message)}</span></div>`;
      showToast(error.message, true);
    }
  }

  function detailValue(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "object") {
      const entries = Object.entries(value);
      if (!entries.length) return "—";
      return entries.map(([key, item]) => `${key.replace(/_/g, " ")} ${typeof item === "object" ? JSON.stringify(item) : item}`).join(" · ");
    }
    return String(value);
  }

  /* Escaped text with the two inline marks the READMEs use: **bold**, `code`. */
  function inlineMarkdown(text) {
    return escapeHtml(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function openModelDialog(modelId) {
    const model = state.models.find((item) => item.id === modelId);
    if (!model) return;
    state.dialogModel = model.id;
    renderModelDialog(model);
    const dialog = $("#model-dialog");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    loadModelBenchmarks(model.id);
  }

  function renderModelDialog(model) {
    $("#dialog-title").removeAttribute("data-i18n");
    $("#dialog-title").textContent = model.display_name;
    const artifacts = (model.artifacts || []).map((artifact) => `<div class="artifact-row"><strong>${escapeHtml(artifact.variant)}</strong> · ${escapeHtml(artifact.filename)} · ${escapeHtml(t(artifact.available ? "available" : "missing"))}<br><span>processor ${escapeHtml(artifact.processor)} · sha ${escapeHtml((artifact.sha256 || "").slice(0, 12))}…</span></div>`).join("");
    const parameters = Object.entries(model.parameters || {}).map(([name, spec]) => {
      const range = spec.minimum != null || spec.maximum != null ? ` · ${spec.minimum ?? "…"}–${spec.maximum ?? "…"}` : "";
      return `<div class="detail-item"><span>${escapeHtml(name)}</span><strong>${escapeHtml(`${spec.type || ""} · ${t("default")} ${spec.default ?? "—"}${range}`)}</strong>${spec.description ? `<small>${escapeHtml(spec.description)}</small>` : ""}</div>`;
    }).join("");
    $("#dialog-body").innerHTML = `
      <div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Contract"))}</div><div class="detail-grid"><div class="detail-item"><span>${escapeHtml(t("Task"))}</span><strong>${escapeHtml(taskLabel(model.task))}</strong></div><div class="detail-item"><span>${escapeHtml(t("Lifecycle"))}</span><strong>${escapeHtml(t(model.lifecycle))}</strong></div><div class="detail-item"><span>${escapeHtml(t("Inputs"))}</span><strong>${escapeHtml((model.inputs || []).join(", ") || "—")}</strong></div><div class="detail-item"><span>${escapeHtml(t("Outputs"))}</span><strong>${escapeHtml((model.outputs || []).join(", ") || "—")}</strong></div><div class="detail-item"><span>${escapeHtml(t("Audio"))}</span><strong>${escapeHtml(model.sample_rate ? `${model.sample_rate} Hz · ${model.channels || 1} ch` : "—")}</strong></div><div class="detail-item"><span>${escapeHtml(t("Roles"))}</span><strong>${escapeHtml((model.roles || []).join(", ") || "—")}</strong></div></div></div>
      <div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Description"))}</div><p class="dialog-copy">${escapeHtml(model.description || "—")}</p></div>
      <div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Benchmarks"))}</div><div id="dialog-benchmarks" data-model="${escapeHtml(model.id)}"><span class="measure-empty">${escapeHtml(t("Loading benchmark references…"))}</span></div></div>
      <div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Artifacts"))}</div><div class="artifact-list">${artifacts || `<span>${escapeHtml(t("Reserved for a future ONNX artifact."))}</span>`}</div></div>
      <div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Pre/post-processing"))}</div><div class="detail-grid"><div class="detail-item"><span>${escapeHtml(t("Preprocessing"))}</span><strong>${escapeHtml(detailValue(model.preprocessing))}</strong></div><div class="detail-item"><span>${escapeHtml(t("Postprocessing"))}</span><strong>${escapeHtml(detailValue(model.postprocessing))}</strong></div><div class="detail-item"><span>${escapeHtml(t("Recommended"))}</span><strong>${escapeHtml(detailValue(model.recommended_inference))}</strong></div><div class="detail-item"><span>${escapeHtml(t("Source checkpoint"))}</span><strong>${escapeHtml(model.source_checkpoint || "—")}</strong></div></div></div>
      ${parameters ? `<div class="dialog-section"><div class="dialog-section-title">${escapeHtml(t("Request parameters"))}</div><div class="detail-grid">${parameters}</div></div>` : ""}`;
  }

  async function loadModelBenchmarks(modelId) {
    const target = () => { const node = $("#dialog-benchmarks"); return node?.dataset.model === modelId ? node : null; };
    try {
      const data = await api(`/api/models/${encodeURIComponent(modelId)}/benchmarks`);
      const node = target();
      if (!node) return;
      node.innerHTML = data.references?.length
        ? data.references.map(renderBenchmarkReference).join("")
        : `<span class="measure-empty">${escapeHtml(t("The catalog lists no benchmark references for this model."))}</span>`;
    } catch (error) {
      const node = target();
      if (node) node.innerHTML = `<span class="measure-empty">${escapeHtml(t("Benchmark references unavailable: {reason}", { reason: error.message }))}</span>`;
    }
  }

  function renderBenchmarkReference(reference) {
    const path = `<code>${escapeHtml(reference.path)}</code>`;
    if (reference.kind === "record") {
      const record = reference.record || {};
      const verdict = String(record.verdict || "—");
      const rows = (record.stages || []).map((stage) => {
        const difference = stage.difference;
        const change = difference && Number.isFinite(Number(difference.point))
          ? `${Number(difference.point) >= 0 ? "+" : ""}${Number(difference.point).toFixed(4)}${Number.isFinite(Number(difference.ci_low)) ? ` <small>[${Number(difference.ci_low).toFixed(4)}, ${Number(difference.ci_high).toFixed(4)}]</small>` : ""}`
          : "—";
        return `<tr><td><strong>${escapeHtml(stage.name)}</strong><small>${escapeHtml([stage.role, stage.direction?.replace(/_/g, " "), stage.n != null ? `n=${stage.n}` : ""].filter(Boolean).join(" · "))}</small></td><td>${measurementValue(stage.value, 4)}</td><td>${measurementValue(stage.baseline, 4)}</td><td>${change}</td><td><span class="job-status job-status-${stage.verdict === "pass" ? "succeeded" : stage.verdict === "fail" ? "failed" : "queued"}">${escapeHtml(stage.verdict ? t(stage.verdict) : "—")}</span></td></tr>`;
      }).join("");
      const unresolved = record.unresolved_gates?.length ? ` · ${escapeHtml(t("unresolved: {gates}", { gates: record.unresolved_gates.join(", ") }))}` : "";
      return `<div class="benchmark-ref"><p class="benchmark-ref-head">${escapeHtml(t("Gate record"))} ${path} · <b>${escapeHtml(record.tag || "")}</b> · ${escapeHtml(t("verdict"))} <span class="job-status job-status-${verdict === "pass" ? "succeeded" : "failed"}">${escapeHtml(t(verdict))}</span>${unresolved}</p><div class="measurement-table-wrap"><table class="table measurement-table benchmark-table"><thead><tr><th>${escapeHtml(t("Stage"))}</th><th class="num">${escapeHtml(t("Value"))}</th><th class="num">${escapeHtml(t("Baseline"))}</th><th class="num">Δ [95% CI]</th><th class="num">${escapeHtml(t("Verdict"))}</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
    }
    if (reference.kind === "markdown") {
      const rows = (reference.rows || []).map((row) => `<div class="benchmark-row">${row.section ? `<p class="benchmark-section">${escapeHtml(row.section)}</p>` : ""}<dl>${row.cells.map(([header, cell]) => `<dt>${escapeHtml(header)}</dt><dd>${inlineMarkdown(cell)}</dd>`).join("")}</dl></div>`).join("");
      return `<div class="benchmark-ref"><p class="benchmark-ref-head">${escapeHtml(t("Release notes"))} ${path}</p>${rows || `<span class="measure-empty">${escapeHtml(t("No table row names this checkpoint."))}</span>`}<details class="benchmark-full"><summary>${escapeHtml(t("Full document"))}</summary><pre>${escapeHtml(reference.text || "")}</pre></details></div>`;
    }
    return `<div class="benchmark-ref"><p class="benchmark-ref-head">${path} · ${escapeHtml(reference.kind === "missing" ? t("not found in this checkout") : reference.error || reference.kind)}</p></div>`;
  }

  function setBusy(button, busy) {
    button.classList.toggle("is-loading", busy);
    button.disabled = busy;
  }

  // A task can be started from its run bar and from its settings panel.
  function setRunBusy(key, busy) {
    $$(`[data-run="${key}"]`).forEach((button) => setBusy(button, busy));
  }

  function fileIdentity(file) {
    return file ? `${file.name}:${file.size}:${file.lastModified}` : "";
  }

  function shortModelName(model) {
    return String(model?.display_name || model?.id || "model").replace(/^(Voice Isolation|Noise Suppression|Speaker Verification)\s+/, "");
  }

  function selectLabel(selector) {
    const select = $(selector);
    return select?.selectedOptions?.[0]?.textContent?.split(" · ")[0] || "—";
  }

  function readDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve({ filename: file.name, data: reader.result });
      reader.onerror = () => reject(new Error(t("Could not read {name}", { name: file.name })));
      reader.readAsDataURL(file);
    });
  }

  // The last file accepted wins: a sample or a recording replaces whatever
  // the file input still holds.
  function selectedFile(inputId) {
    return state.files[inputId] || $(inputId).files?.[0] || null;
  }

  /* Send a file's bytes once and refer to it by id afterwards: re-running with
   * other settings, or comparing more models, costs no second transfer. */
  async function uploadDescriptor(file) {
    const key = fileIdentity(file);
    const cached = state.uploads.get(key);
    if (cached) {
      try {
        await api(`/api/uploads/${encodeURIComponent(cached.upload_id)}`);
        return { upload_id: cached.upload_id, filename: file.name };
      } catch {
        state.uploads.delete(key); // expired or the server restarted
      }
    }
    const response = await fetch("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.error?.message || t("Upload failed ({status})", { status: response.status }));
    state.uploads.set(key, payload);
    return { upload_id: payload.upload_id, filename: file.name };
  }

  /* The file's own sample rate from its header (WAV, FLAC).  The browser
   * decodes everything to its output rate, so the decoded buffer cannot say. */
  async function sniffSampleRate(file) {
    try {
      const bytes = new Uint8Array(await file.slice(0, 4096).arrayBuffer());
      const text = (offset, length) => String.fromCharCode(...bytes.slice(offset, offset + length));
      const view = new DataView(bytes.buffer);
      if (text(0, 4) === "RIFF" && text(8, 4) === "WAVE") {
        for (let offset = 12; offset + 8 <= bytes.length;) {
          const size = view.getUint32(offset + 4, true);
          if (text(offset, 4) === "fmt ") return view.getUint32(offset + 12, true);
          offset += 8 + size + (size % 2);
        }
      }
      if (text(0, 4) === "fLaC" && bytes.length >= 21) return (bytes[18] << 12) | (bytes[19] << 4) | (bytes[20] >> 4);
    } catch { /* unknown container */ }
    return null;
  }

  function formatSeconds(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return number < 1 ? `${Math.round(number * 1000)} ms` : `${number.toFixed(2)} s`;
  }

  function providerLabel(value) {
    const provider = String(value || "");
    if (provider.includes("CUDA")) return "CUDA";
    if (provider.includes("CoreML")) return "Apple CoreML";
    if (provider.includes("CPU")) return "CPU";
    return provider || "—";
  }

  function hostLoadLabel(host) {
    const load = Number(host?.load_average_1m);
    if (!Number.isFinite(load)) return "";
    return host?.cpu_count ? t("host load {load} / {cores} cores", { load: load.toFixed(1), cores: host.cpu_count }) : t("host load {load}", { load: load.toFixed(1) });
  }

  /* A one-track deck for looking at a file before it runs: the same zoom,
   * selection and spectrogram tools as the comparison. */
  function previewDeck(root, { compact = false } = {}) {
    const deck = new window.PureSoundCompareDeck(root, { emptyText: "Decoding audio…", analysis: false, compact });
    return {
      deck,
      async loadFile(file) {
        deck.setTracks([{ id: "file", label: file.name, color: root.dataset.waveColor || "#c8f6f9" }]);
        return deck.loadTrack("file", file);
      },
      stop() { deck.stop(); },
      clear() { deck.setTracks([]); },
    };
  }

  async function previewAudio(file, elements) {
    const { preview, name, meta, panel } = elements;
    elements.file = file;
    preview.hidden = false;
    name.textContent = file.name;
    meta.textContent = `${(file.size / 1024).toFixed(0)} KB · ${t("decoding…")}`;
    try {
      const [buffer, fileRate] = await Promise.all([panel.loadFile(file), sniffSampleRate(file)]);
      // Another file or a clear replaced this one while it decoded.
      if (!buffer || elements.file !== file) return;
      meta.textContent = [`${buffer.duration.toFixed(2)} s`, fileRate ? t("{rate} kHz file", { rate: (fileRate / 1000).toFixed(fileRate % 1000 ? 1 : 0) }) : "", `${(file.size / 1024).toFixed(0)} KB`].filter(Boolean).join(" · ");
    } catch (error) {
      if (elements.file !== file) return;
      meta.textContent = `${(file.size / 1024).toFixed(0)} KB · ${t("preview unavailable")}`;
      showToast(t("Browser preview failed: {reason}", { reason: error.message }), true);
    }
  }

  function wireFileInput(inputId, elements, onFile = () => {}) {
    const input = $(inputId);
    const dropzone = input.closest(".dropzone");
    const titleNode = dropzone.querySelector("[data-dropzone-title]");
    const hintNode = dropzone.querySelector("[data-dropzone-hint]");
    dropzone.dataset.title = titleNode?.getAttribute("data-i18n") || titleNode?.textContent || "";
    dropzone.dataset.hint = hintNode?.getAttribute("data-i18n") || hintNode?.textContent || "";
    const accept = (file) => {
      state.files[inputId] = file;
      // Once a file is chosen the drop target shrinks to a replace bar.
      dropzone.classList.add("has-file");
      const title = dropzone.querySelector("[data-dropzone-title]");
      const hint = dropzone.querySelector("[data-dropzone-hint]");
      if (title) { title.removeAttribute("data-i18n"); title.textContent = t("Replace audio"); }
      if (hint) { hint.removeAttribute("data-i18n"); hint.textContent = t("{name} · drop or click to choose another file", { name: file.name }); }
      state.dropzoneFiles[inputId] = file.name;
      elements.preview.hidden = false;
      previewAudio(file, elements);
      onFile(file);
    };
    state.acceptors[inputId] = (file) => { accept(file); updateClearButtons(); };
    input.addEventListener("change", updateClearButtons);
    dropzone.addEventListener("drop", () => window.setTimeout(updateClearButtons, 0));
    input.addEventListener("change", () => { if (input.files?.[0]) accept(input.files[0]); });
    ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("is-dragging"); }));
    ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("is-dragging"); }));
    dropzone.addEventListener("drop", (event) => { const file = event.dataTransfer.files?.[0]; if (file) { try { input.files = event.dataTransfer.files; } catch { /* keep the in-memory file for browsers that forbid assignment */ } accept(file); } });
  }

  function renderMetrics(target, metrics) {
    target.innerHTML = metrics.map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  }

  function updateJobProgress(key, job) {
    const phase = String(job.phase || job.status || "working")
      .replace(/:/g, " · ")
      .replace(/_/g, " ");
    shell().setState(SCREEN_OF[key], { state: "run", label: phase, progress: Math.max(0, Math.min(1, Number(job.progress) || 0)) });
  }

  async function runBackgroundJob(payload, key) {
    const cancelButton = $(`#${key}-cancel`);
    if (state.activeJobs[key]) throw new Error(t("An inference is already running."));
    setRunBusy(key, true);
    cancelButton.hidden = false;
    shell().setState(SCREEN_OF[key], { state: "run", label: t("Queued"), progress: 0 });
    updateClearButtons();
    try {
      const initial = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      state.activeJobs[key] = initial.job_id;
      updateClearButtons();
      updateJobProgress(key, initial);
      while (true) {
        const job = await api(`/api/jobs/${encodeURIComponent(initial.job_id)}`);
        updateJobProgress(key, job);
        if (job.status === "succeeded") {
          shell().setState(SCREEN_OF[key], { state: "ok", label: Number.isFinite(job.elapsed_seconds) ? formatSeconds(job.elapsed_seconds) : "" });
          return job.result;
        }
        if (job.status === "failed") throw new Error(job.error || t("Job failed."));
        if (job.status === "cancelled") throw Object.assign(new Error(t(job.kind === "measurement" ? "Measurement cancelled." : "Inference cancelled.")), { cancelled: true });
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
    } catch (error) {
      shell().setState(SCREEN_OF[key], error.cancelled ? { state: "idle" } : { state: "err", label: error.message });
      throw error;
    } finally {
      delete state.activeJobs[key];
      setRunBusy(key, false);
      updateClearButtons();
      cancelButton.hidden = true;
      refreshJobHistory();
    }
  }

  async function cancelInferenceJob(key) {
    const jobId = state.activeJobs[key];
    if (!jobId) return;
    const button = $(`#${key}-cancel`);
    button.disabled = true;
    try {
      await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: "{}" });
    } catch (error) {
      showToast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function jobCategory(job) {
    if (["world_render", "world_sweep"].includes(job.kind)) return "world";
    if (job.kind === "measurement") return "measurement";
    if (job.kind === "live") return "live";
    const task = job.result?.task || state.models.find((model) => model.id === job.model_id)?.task;
    return task === "speaker_embedding" ? "speaker" : "enhancement";
  }

  function timeZone() {
    try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch { return ""; }
  }

  /* Point a ZIP or report link at a finished job, or hide it. */
  function exportLink(selector, jobId, kind) {
    const link = $(selector);
    if (!link) return;
    link.hidden = !jobId;
    if (!jobId) { link.removeAttribute("href"); return; }
    link.href = kind === "zip"
      ? `/api/jobs/${encodeURIComponent(jobId)}/outputs.zip`
      : `/api/jobs/${encodeURIComponent(jobId)}/report.html?tz=${encodeURIComponent(timeZone())}`;
    link.setAttribute("download", "");
  }

  function formatWhen(epochSeconds) {
    const value = Number(epochSeconds);
    if (!Number.isFinite(value)) return "";
    const locale = window.PureSoundI18n?.lang === "zh-TW" ? "zh-TW" : [];
    return new Date(value * 1000).toLocaleString(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function renderJobHistory(jobs) {
    const target = $("#job-history");
    if (!target) return;
    const filter = state.historyFilter || "all";
    const shown = jobs.filter((job) => filter === "all" || jobCategory(job) === filter);
    $("#history-count").textContent = t("{shown} of {total} runs", { shown: shown.length, total: jobs.length });
    if (!shown.length) {
      target.innerHTML = `<div class="empty-state"><span>${escapeHtml(t(jobs.length ? "No runs match this filter." : "No runs yet. Playground runs and comparisons appear here."))}</span></div>`;
      return;
    }
    const kinds = { world: t("World"), enhancement: t("Enhancement"), speaker: t("Speaker"), measurement: t("Comparison"), live: t("Live") };
    target.innerHTML = shown.map((job) => {
      const result = job.result || {};
      const category = jobCategory(job);
      const model = state.models.find((item) => item.id === job.model_id);
      const file = result.input_files?.audio || result.input_files?.test || "";
      let title = model?.display_name || job.model_id;
      let detail = job.error || job.phase || "—";
      if (job.status === "succeeded") {
        if (category === "measurement") {
          const count = (result.clips ? result.aggregate?.length : result.models?.length) || 0;
          title = [t("Comparison"), t(count === 1 ? "{n} candidate" : "{n} candidates", { n: count }), result.clips ? t("{n} clips", { n: result.clips.length }) : ""].filter(Boolean).join(" · ");
          detail = [file, result.clips ? "" : t("{seconds} input", { seconds: measurementValue(result.input?.duration_seconds, 1, " s") }), t(result.reference ? "with reference" : "no reference"), job.elapsed_seconds == null ? "" : formatSeconds(job.elapsed_seconds)].filter(Boolean).join(" · ");
        } else if (category === "live") {
          const kept = result.live?.streamed_seconds > result.live?.kept_seconds + 0.5
            ? t("{kept} kept of {streamed}", { kept: measurementValue(result.live?.kept_seconds, 1, " s"), streamed: measurementValue(result.live.streamed_seconds, 0, " s") })
            : t("{kept} kept", { kept: measurementValue(result.live?.kept_seconds, 1, " s") });
          detail = [kept, result.rtf == null ? "" : `RTF ${Number(result.rtf).toFixed(3)}`, result.metadata?.dry_blend == null ? "" : t("blend {value}", { value: Number(result.metadata.dry_blend).toFixed(2) }), t(result.metadata?.onset_guard ? "guard on" : "guard off")].filter(Boolean).join(" · ");
        } else if (category === "speaker") {
          detail = t(result.scores?.verdict ? "similarity {score} · match at {threshold}" : "similarity {score} · no match at {threshold}", { score: measurementValue(result.scores?.cosine_similarity, 3), threshold: measurementValue(result.scores?.threshold, 2) });
        } else {
          detail = [file, result.rtf == null ? "" : `RTF ${Number(result.rtf).toFixed(3)}`, result.metadata?.dry_blend == null ? "" : t("blend {value}", { value: Number(result.metadata.dry_blend).toFixed(2) }), t(result.metadata?.onset_guard ? "guard on" : "guard off")].filter(Boolean).join(" · ");
        }
      }
      const output = result.output_urls?.aligned || result.output_urls?.audio;
      const resumable = job.status === "succeeded" || (category === "world" && job.status === "cancelled" && result.cells);
      const open = resumable ? `<button class="button button-secondary button-small" type="button" data-open-job="${escapeHtml(job.job_id)}">${escapeHtml(t("Open"))} <span aria-hidden="true">→</span></button>` : "";
      const exportable = resumable && category !== "speaker";
      const exports = exportable ? `<a href="/api/jobs/${encodeURIComponent(job.job_id)}/outputs.zip" download title="${escapeHtml(t("Every audio output, plus report.json"))}">ZIP ↓</a><a href="/api/jobs/${encodeURIComponent(job.job_id)}/report.html?tz=${encodeURIComponent(timeZone())}" download title="${escapeHtml(t("One HTML page with the audio embedded"))}">${escapeHtml(t("Report"))} ↓</a>` : "";
      return `<article class="job-history-item panel"><div class="job-history-main"><span class="job-kind job-kind-${category}">${kinds[category]}</span><div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div></div><div class="job-history-side"><span class="job-when">${escapeHtml(formatWhen(job.finished_at || job.created_at))}</span><span class="job-status job-status-${escapeHtml(job.status)}">${escapeHtml(t(job.status))}</span>${output ? `<a href="${escapeHtml(output)}" download>WAV ↓</a>` : ""}${exports}${open}</div></article>`;
    }).join("");
    $$("[data-open-job]", target).forEach((button) => button.addEventListener("click", () => openHistoryJob(button.dataset.openJob)));
  }

  /* Put a finished run back on screen: the comparison deck for an
   * enhancement, the verdict for a speaker pair, the report for a comparison. */
  async function openHistoryJob(jobId) {
    const job = state.jobs.find((item) => item.job_id === jobId);
    if (!job?.result) return;
    // A result stored without its own job id exports by the job's.
    job.result.job_id = job.result.job_id || job.job_id;
    const category = jobCategory(job);
    const model = state.models.find((item) => item.id === job.model_id) || { id: job.model_id, display_name: job.model_id };
    try {
      if (category === "world") {
        shell().show("world");
        await window.PureSoundWorld.open(job.result);
      } else if (category === "measurement") {
        shell().show("compare");
        state.clipIndex = 0;
        renderMeasurementReport(job.result);
      } else if (category === "speaker") {
        shell().show("verify");
        renderSvResult(job.result, Number(job.result.scores?.threshold));
      } else {
        shell().show("playground");
        const fileName = job.result.input_files?.audio || "history";
        await renderVoiceResult(job.result, { fileName, fileKey: `job:${job.job_id}`, model });
        const note = $("#voice-form-note");
        note.textContent = t("Showing a run from {when}. Upload a file to run again.", { when: formatWhen(job.finished_at) });
        note.className = "form-note";
      }
    } catch (error) {
      showToast(t("Could not open that run: {reason}", { reason: error.message }), true);
    }
  }

  async function refreshJobHistory() {
    try {
      const response = await api("/api/jobs?limit=20");
      state.jobs = response.jobs || [];
      renderJobHistory(state.jobs);
    } catch (error) {
      showToast(t("Could not load run history: {reason}", { reason: error.message }), true);
    }
  }

  async function runVoiceInference() {
    const file = selectedFile("#voice-audio");
    const note = $("#voice-form-note");
    if (!file) { setKey(note, "Select an audio file first."); note.className = "form-note is-error"; return; }
    if (onDevice()) { await runVoiceOnDevice(file); return; }
    shell().closeInspector();
    setRunBusy("voice", true); setKey(note, "Preparing audio and running the streaming processor…"); note.className = "form-note";
    try {
      const model = selectedModel("#voice-model");
      const input = await uploadDescriptor(file);
      const dryBlend = Number($("#voice-dry-blend").value);
      const parameters = { dry_blend: dryBlend, ...onsetGuardParameters("voice") };
      if ($("#voice-collect-extras").checked) parameters.collect_extras = true;
      const result = await runBackgroundJob(
        {
          model_id: model.id,
          variant: $("#voice-variant").value,
          provider: $("#voice-provider").value,
          inputs: { audio: input },
          parameters,
          stages: $("#voice-stage").value ? [{ model_id: $("#voice-stage").value }] : undefined,
          measurements: { include_dnsmos: true },
        },
        "voice",
      );
      await renderVoiceResult(result, { fileName: file.name, fileKey: fileIdentity(file), model });
      setKey(note, "Done. Switch tracks with 1–4 while it plays."); note.className = "form-note is-success";
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; }
    finally { setRunBusy("voice", false); }
  }

  /* The same run in this browser (device.js): decoded here, processed in a
   * worker, nothing uploaded, nothing kept in the server's history. */
  async function runVoiceOnDevice(file) {
    if (state.deviceRun) return; // Ctrl+Enter while one is running
    const note = $("#voice-form-note");
    const model = selectedModel("#voice-model");
    const controller = new AbortController();
    const progress = (label, value = null) => shell().setState("playground", { state: "run", label, progress: value });
    shell().closeInspector();
    state.deviceRun = controller;
    setRunBusy("voice", true);
    $("#voice-cancel").hidden = false;
    updateClearButtons();
    setKey(note, "Decoding the audio and running the model on this device…"); note.className = "form-note";
    progress(t("Decoding audio…"));
    try {
      const { samples, sampleRate } = await window.PureSoundDevice.decode(file, { sampleRate: (await sniffSampleRate(file)) || undefined });
      const run = await window.PureSoundDevice.process(samples, {
        model: model.id,
        sampleRate,
        dryBlend: Number($("#voice-dry-blend").value),
        onsetGuard: deviceOnsetGuard(),
        signal: controller.signal,
        onProgress: (value, phase) => progress(t(phase === "load" ? "Loading the model on this device" : "Processing on this device"), value),
      });
      shell().setState("playground", { state: "ok", label: formatSeconds(run.seconds) });
      await renderVoiceResult(deviceResult(run), { fileName: file.name, fileKey: fileIdentity(file), model });
      setKey(note, "Done. Switch tracks with 1–4 while it plays."); note.className = "form-note is-success";
    } catch (error) {
      shell().setState("playground", error.cancelled ? { state: "idle" } : { state: "err", label: error.message });
      note.textContent = error.message; note.className = "form-note is-error";
    } finally {
      state.deviceRun = null;
      setRunBusy("voice", false);
      $("#voice-cancel").hidden = true;
      updateClearButtons();
    }
  }

  /* The Inspector's onset guard as device.js takes it: false, or its knobs. */
  function deviceOnsetGuard() {
    const parameters = onsetGuardParameters("voice");
    if (!("onset_guard" in parameters)) return undefined;
    if (!parameters.onset_guard) return false;
    return Object.fromEntries(ONSET_GUARD_KNOBS.filter(({ parameter }) => parameter in parameters).map(({ parameter }) => [parameter.replace(/^onset_guard_/, ""), parameters[parameter]]));
  }

  function blobUrl(blob) {
    const url = URL.createObjectURL(blob);
    state.blobUrls.add(url);
    return url;
  }

  /* Audio made in this page that nothing on it shows any more. */
  function releaseBlobUrls(keep = []) {
    state.blobUrls.forEach((url) => {
      if (keep.includes(url)) return;
      URL.revokeObjectURL(url);
      state.blobUrls.delete(url);
    });
  }

  /* A device run in the shape of a server result, its audio as blob URLs. */
  function deviceResult(run) {
    const wav = (samples) => blobUrl(new Blob([window.PureSoundCapture.encodeWav(samples, run.sampleRate)], { type: "audio/wav" }));
    let energy = 0;
    run.output.forEach((value) => { energy += value * value; });
    return {
      runner: "device",
      threads: run.threads,
      sample_rate: run.sampleRate,
      rtf: run.rtf,
      output_urls: { input: wav(run.input), aligned: wav(run.output), audio: wav(run.stream), removed: wav(run.removed) },
      alignment: { latency_ms: run.latencyMs, sample_rate: run.sampleRate },
      metadata: { dry_blend: run.dryBlend, onset_guard: run.onsetGuard, duration_seconds: run.input.length / run.sampleRate, win_length: run.windowSamples, hop_length: run.hopSamples, latency_ms: run.latencyMs },
      measurements: { output: { rms_dbfs: 10 * Math.log10(energy / run.output.length + 1e-12) } },
    };
  }

  function deviceLabel(threads) {
    return t(threads === 1 ? "WebAssembly · 1 thread" : "WebAssembly · {n} threads", { n: threads });
  }

  async function renderVoiceResult(result, { fileName, fileKey, model }) {
    window.setTimeout(updateClearButtons, 0);
    releaseBlobUrls([state.voiceLast?.url, ...Object.values(result.output_urls || {})]);
    exportLink("#voice-export-zip", result.job_id, "zip");
    exportLink("#voice-export-report", result.job_id, "report");
    const outputUrl = result.output_urls?.aligned || result.output_urls?.audio;
    if (!outputUrl) throw new Error(t("The runtime did not return an audio output."));
    $("#voice-result").hidden = false;
    $("#voice-download").href = outputUrl;
    $("#voice-download-raw").href = result.output_urls?.audio || outputUrl;
    $("#voice-download-raw").download = "enhanced_stream.wav";
    $("#voice-download-raw").hidden = !result.output_urls?.aligned;
    $("#voice-download-removed").href = result.output_urls?.removed || "#";
    $("#voice-download-removed").hidden = !result.output_urls?.removed;
    $("#voice-result-name").textContent = `${fileName} · ${model.display_name}${result.runner === "device" ? ` · ${t("this device")}` : ""}`;
    state.voiceView = { result, model };
    renderVoiceText(result, model);
    await showVoiceComparison(result, { fileKey, model, dryBlend: result.metadata?.dry_blend, outputUrl });
  }

  function renderVoiceText(result, model) {
    const alignment = result.alignment;
    $("#voice-alignment-note").textContent = alignment
      ? t("Aligned to the input: {latency} streaming latency removed, same length as the input at {rate} kHz.", { latency: measurementValue(alignment.latency_ms, 0, " ms"), rate: alignment.sample_rate / 1000 })
      : t("As emitted by the runtime.");
    $("#voice-result-meta").textContent = `${result.sample_rate ? `${result.sample_rate / 1000} kHz` : "—"} · ${formatSeconds(result.metadata?.duration_seconds)}${hostLoadLabel(result.host) ? ` · ${hostLoadLabel(result.host)}` : ""}`;
    const outputMeasurements = result.measurements?.output || {};
    const dnsMos = result.measurements?.reference_free?.dnsmos || {};
    const pipeline = result.pipeline;
    const rate = Number(result.sample_rate) || 16000;
    const windowMs = result.metadata?.win_length ? 1000 * result.metadata.win_length / rate : null;
    const lookAheadMs = pipeline ? pipeline.latency_ms : result.metadata?.latency_ms;
    // Look-ahead is what the graph adds; a sample also waits for the analysis
    // window to fill, so input-to-output is up to window + look-ahead.
    renderMetrics($("#voice-metrics"), [
      [pipeline ? t("RTF · chain") : "RTF", (pipeline ? pipeline.rtf : result.rtf) == null ? "—" : Number(pipeline ? pipeline.rtf : result.rtf).toFixed(3)],
      [t(pipeline ? "Look-ahead · chain" : "Look-ahead"), lookAheadMs == null ? "—" : `${Number(lookAheadMs).toFixed(1)} ms`],
      [t("Input → output"), windowMs == null || lookAheadMs == null ? "—" : (() => {
        const models = pipeline ? pipeline.stages.length + 1 : 1;
        const hopMs = result.metadata?.hop_length ? 1000 * result.metadata.hop_length / rate : 0;
        const longest = windowMs * models + Number(lookAheadMs);
        return `${(longest - hopMs * models).toFixed(0)}–${longest.toFixed(0)} ms`;
      })()],
      ["DNSMOS OVR", measurementValue(dnsMos.dnsmos_ovr, 2)],
      [t("Output RMS"), measurementValue(outputMeasurements.rms_dbfs, 1, " dBFS")],
    ]);
    const chain = pipeline ? `${pipeline.stages.map((stage) => shortModelName(stage)).join(" → ")} → ${shortModelName(model)} · ` : "";
    const latency = windowMs ? ` · ${t(pipeline ? "algorithmic latency = {window} ms window per model + look-ahead" : "algorithmic latency = {window} ms window + look-ahead", { window: windowMs.toFixed(0) })}` : "";
    $("#voice-applied").textContent = `${chain}${t("Dry blend")} ${measurementValue(result.metadata?.dry_blend, 2)} · ${t("Onset guard")} ${onsetGuardSummary(result.metadata?.onset_guard)}${latency}`;
    const measurementNote = $("#voice-measurement-note");
    if (result.runner === "device") {
      measurementNote.textContent = `${t("Processed on this device with {runtime}.", { runtime: deviceLabel(result.threads) })} ${t("DNSMOS is scored on the server: run there to see it.")}`;
      measurementNote.className = "measurement-inline-note";
      return;
    }
    const dnsError = result.measurements?.reference_free?.dnsmos_error;
    const dnsAvailable = Number.isFinite(Number(dnsMos.dnsmos_ovr));
    const where = t("Output measured on {provider}.", { provider: providerLabel(result.provider) });
    if (dnsError) {
      measurementNote.textContent = `${where} ${t("DNSMOS unavailable: {reason}", { reason: dnsError })}`;
    } else if (dnsAvailable) {
      measurementNote.textContent = `${where} ${t("DNSMOS OVR is a reference-free quality score (1–5; higher is better); run a comparison to see it next to the unprocessed input's.")}`;
    } else {
      measurementNote.textContent = `${where} ${t("DNSMOS was not returned by the runtime.")}`;
    }
    measurementNote.className = `measurement-inline-note${dnsError || !dnsAvailable ? " is-warning" : ""}`;
  }

  async function showVoiceComparison(result, { fileKey, model, dryBlend, outputUrl }) {
    const urls = result.output_urls || {};
    const label = `${shortModelName(model)}${Number.isFinite(Number(dryBlend)) ? ` · ${t("blend {value}", { value: Number(dryBlend).toFixed(2) })}` : ""}${result.runner === "device" ? ` · ${t("this device")}` : ""}`;
    const previous = state.voiceLast?.fileKey === fileKey && state.voiceLast.url !== outputUrl ? state.voiceLast : null;
    const tracks = [];
    if (urls.input) tracks.push({ id: "input", label: "Input", color: "#c8f6f9", hint: "as the model hears it", levelReference: true, url: urls.input });
    tracks.push({ id: "output", label: "Output", color: "#bdbbff", hint: label, url: outputUrl });
    if (urls.removed) tracks.push({ id: "removed", label: "Removed", color: "#ff9a62", hint: "input − output", diagnostic: true, url: urls.removed });
    (result.pipeline?.stages || []).forEach((stage, index) => {
      if (urls[`stage-${index + 1}`]) tracks.splice(tracks.length - (urls.removed ? 1 : 0), 0, { id: `stage-${index + 1}`, label: t("After {model}", { model: shortModelName(stage) }), color: "#ffd166", hint: "aligned, before the next model", url: urls[`stage-${index + 1}`] });
    });
    if (previous) tracks.push({ id: "previous", label: "Previous", color: "#ef2cc1", hint: previous.label, url: previous.url });
    const deck = state.voiceDeck;
    deck.setTracks(tracks);
    deck.setCurves(extrasCurves(result.extras));
    $("#voice-input-preview").hidden = Boolean(urls.input);
    state.audioPanels.voiceInput.stop();
    const loads = await Promise.allSettled(tracks.map((track) => deck.loadTrack(track.id, track.url)));
    const failed = loads.filter((item) => item.status === "rejected");
    if (failed.length) showToast(`${t("Audio preview failed: {reason}", { reason: failed[0].reason?.message || t("decode error") })}${failed.length === tracks.length ? ` — ${t("the stored output may have expired.")}` : ""}`, true);
    deck.select("output");
    state.voiceLast = { fileKey, url: outputUrl, label };
  }

  /* VAD and other per-frame heads, as curve lanes under the deck. */
  function extrasCurves(extras) {
    const colors = ["#7ee0a1", "#ffd166", "#8ecbff", "#f7a8d8"];
    const curves = [];
    Object.entries(extras || {}).forEach(([name, extra]) => {
      (extra.series || []).forEach((values, column) => {
        curves.push({
          label: extra.series.length > 1 ? `${name} [${column}]` : name,
          hint: t("{frames} frames · {hop} ms per point", { frames: extra.frames, hop: (extra.hop_seconds * 1000).toFixed(0) }),
          values,
          hopSeconds: extra.hop_seconds,
          offsetSeconds: extra.offset_seconds || 0,
          range: extra.range,
          color: colors[curves.length % colors.length],
        });
      });
    });
    return curves;
  }

  async function runSpeakerVerification() {
    const enrollment = selectedFile("#sv-enrollment");
    const test = selectedFile("#sv-test");
    const note = $("#sv-form-note");
    if (!enrollment || !test) { setKey(note, "Select both enrollment and test audio."); note.className = "form-note is-error"; return; }
    shell().closeInspector();
    setRunBusy("sv", true); setKey(note, "Extracting both embeddings…"); note.className = "form-note";
    try {
      const model = selectedModel("#sv-model");
      const inputs = { enrollment: await uploadDescriptor(enrollment), test: await uploadDescriptor(test) };
      const threshold = Number($("#sv-threshold").value);
      const result = await runBackgroundJob({ model_id: model.id, provider: $("#sv-provider").value, inputs, parameters: { threshold } }, "sv");
      const verdict = renderSvResult(result, threshold);
      setKey(note, "Verification complete."); note.className = "form-note is-success";
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; }
    finally { setRunBusy("sv", false); }
  }

  function renderSvResult(result, threshold) {
    window.setTimeout(updateClearButtons, 0);
    state.svView = { result, threshold };
    const score = Number(result.scores?.cosine_similarity || 0);
    const verdict = Boolean(result.scores?.verdict);
    $("#sv-result").hidden = false;
    $("#sv-score").textContent = score.toFixed(2);
    $("#sv-score-gauge").style.setProperty("--score", `${Math.max(0, Math.min(1, score)) * 100}%`);
    $("#sv-verdict").textContent = t(verdict ? "Match" : "No match");
    $("#sv-verdict").style.color = verdict ? "var(--good)" : "var(--bad)";
    $("#sv-threshold-result").textContent = Number.isFinite(threshold) ? threshold.toFixed(2) : "—";
    $("#sv-verdict-note").textContent = t(verdict ? "Similarity is above the selected threshold." : "Similarity is below the selected threshold.");
    $("#sv-result-time").textContent = formatSeconds(result.elapsed_seconds);
    renderMetrics($("#sv-metrics"), [[t("Similarity"), score.toFixed(3)], [t("Threshold"), Number.isFinite(threshold) ? threshold.toFixed(2) : "—"], [t("Embedding"), `${result.metadata?.embedding_dim || "—"} d`], [t("Provider"), providerLabel(result.provider)]]);
    return verdict;
  }

  function modelSupportsGuard(model) {
    return Boolean(model?.parameters?.onset_guard);
  }

  /* The comparison's rows: one per model to begin with, and "+" adds another
   * setting of the same model, so a blend or guard sweep is one comparison. */
  function renderMeasurementModels() {
    const target = $("#measure-model-list");
    if (!target) return;
    if (!state.voiceModels.length) {
      target.innerHTML = `<span class="measure-empty">${escapeHtml(t("No runnable enhancement models are available."))}</span>`;
      $("#measure-model-count").textContent = t("{n} selected", { n: 0 });
      return;
    }
    if (!state.measureVariants) {
      state.measureVariants = state.voiceModels.map((model, index) => ({
        key: `row-${index}`,
        modelId: model.id,
        checked: (modelIsDefault(model) && model.task === "voice_isolation") || index === 0,
        blend: "",
        guard: "global",
        clone: false,
      }));
      // One switch for the whole comparison, so it takes its ranges and
      // defaults from the release default model rather than from any one row.
      applyOnsetGuardDefaults("measure", state.voiceModels.find(modelIsDefault) || state.voiceModels[0]);
    }
    state.measureVariants = state.measureVariants.filter((variant) => state.voiceModels.some((model) => model.id === variant.modelId));
    target.innerHTML = state.measureVariants.map((variant) => {
      const model = state.voiceModels.find((item) => item.id === variant.modelId);
      const releaseBlend = model.recommended_inference?.dry_blend;
      const guard = modelSupportsGuard(model);
      return `<div class="measure-variant${variant.clone ? " is-clone" : ""}" data-variant="${escapeHtml(variant.key)}">
        <label class="measure-model-option"><input type="checkbox" data-variant-check${variant.checked ? " checked" : ""}><span class="measure-check"></span><span><strong>${escapeHtml(model.display_name)}${variant.clone ? ` · ${escapeHtml(t("another setting"))}` : ""}</strong><small>${escapeHtml(model.id)} · ${escapeHtml(taskLabel(model.task))} · ${escapeHtml(t(model.lifecycle))}</small></span></label>
        <div class="measure-variant-knobs">
          <label><span>${escapeHtml(t("Blend"))}</span><input class="control" type="number" min="0.1" max="1" step="0.05" placeholder="${typeof releaseBlend === "number" ? releaseBlend.toFixed(2) : "—"}" value="${escapeHtml(variant.blend)}" data-variant-blend aria-label="${escapeHtml(t("Dry blend for this row"))}"></label>
          <label><span>${escapeHtml(t("Guard"))}</span><select class="control" data-variant-guard aria-label="${escapeHtml(t("Onset guard for this row"))}"${guard ? "" : ` disabled title="${escapeHtml(t("This model has no onset guard"))}"`}><option value="global">${escapeHtml(t("default"))}</option><option value="on">${escapeHtml(t("on"))}</option><option value="off">${escapeHtml(t("off"))}</option></select></label>
          <label class="measure-variant-before"><span>${escapeHtml(t("Run first"))}</span><select class="control" data-variant-before aria-label="${escapeHtml(t("Model to run before this one"))}"><option value="">${escapeHtml(t("nothing"))}</option>${state.voiceModels.filter((item) => item.id !== model.id && item.sample_rate === model.sample_rate).map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(shortModelName(item))}</option>`).join("")}</select></label>
          <button class="button button-icon" type="button" data-variant-add title="${escapeHtml(t("Add another setting of this model"))}" aria-label="${escapeHtml(t("Add another setting of {model}", { model: model.display_name }))}">+</button>
          ${variant.clone ? `<button class="button button-icon" type="button" data-variant-remove title="${escapeHtml(t("Remove this row"))}" aria-label="${escapeHtml(t("Remove this row"))}">×</button>` : ""}
        </div>
      </div>`;
    }).join("");
    $$(".measure-variant", target).forEach((row) => {
      const variant = state.measureVariants.find((item) => item.key === row.dataset.variant);
      const guardSelect = $("[data-variant-guard]", row);
      guardSelect.value = variant.guard;
      const beforeSelect = $("[data-variant-before]", row);
      beforeSelect.value = variant.before || "";
      beforeSelect.addEventListener("change", (event) => { variant.before = event.target.value; });
      $("[data-variant-check]", row).addEventListener("change", (event) => { variant.checked = event.target.checked; updateMeasurementModelCount(); });
      $("[data-variant-blend]", row).addEventListener("input", (event) => { variant.blend = event.target.value; });
      guardSelect.addEventListener("change", (event) => { variant.guard = event.target.value; syncMeasureGuardKnobs(); });
      $("[data-variant-add]", row).addEventListener("click", () => {
        const index = state.measureVariants.indexOf(variant);
        state.measureVariants.splice(index + 1, 0, { ...variant, key: `row-${Date.now()}`, checked: true, clone: true });
        renderMeasurementModels();
      });
      $("[data-variant-remove]", row)?.addEventListener("click", () => {
        state.measureVariants.splice(state.measureVariants.indexOf(variant), 1);
        renderMeasurementModels();
      });
    });
    updateMeasurementModelCount();
    syncMeasureGuardKnobs();
  }

  /* The guard knobs matter wherever a row will run with the guard on. */
  function syncMeasureGuardKnobs() {
    const globalOn = $("#measure-onset-guard").checked;
    const anyOn = (state.measureVariants || []).some((variant) => variant.checked && (variant.guard === "on" || (variant.guard === "global" && globalOn)));
    $("#measure-onset-knobs").disabled = !anyOn;
  }

  function guardKnobValues(prefix) {
    const values = {};
    ONSET_GUARD_KNOBS.forEach(({ parameter, suffix }) => {
      const value = Number($(`#${prefix}-onset-${suffix}`)?.value);
      if (Number.isFinite(value)) values[parameter] = value;
    });
    return values;
  }

  function measurementCandidates() {
    const globalOn = $("#measure-onset-guard").checked;
    return (state.measureVariants || []).filter((variant) => variant.checked).map((variant) => {
      const model = state.voiceModels.find((item) => item.id === variant.modelId);
      const parameters = {};
      const blend = variant.blend === "" ? null : Number(variant.blend);
      if (blend != null && Number.isFinite(blend)) parameters.dry_blend = Math.min(1, Math.max(0.1, blend));
      let guardOn = false;
      if (modelSupportsGuard(model)) {
        guardOn = variant.guard === "global" ? globalOn : variant.guard === "on";
        parameters.onset_guard = guardOn;
        if (guardOn) Object.assign(parameters, guardKnobValues("measure"));
      }
      const before = variant.before ? state.voiceModels.find((item) => item.id === variant.before) : null;
      const parts = [before ? `${shortModelName(before)} → ${shortModelName(model)}` : shortModelName(model)];
      if (parameters.dry_blend != null) parts.push(`blend ${parameters.dry_blend.toFixed(2)}`);
      if (guardOn) parts.push("guard");  // row labels go into the report file, so they stay English
      const candidate = { model_id: model.id, parameters, label: parts.join(" · ") };
      if (before) candidate.stages = [{ model_id: before.id }];
      return candidate;
    });
  }

  function updateMeasurementModelCount() {
    const selected = (state.measureVariants || []).filter((variant) => variant.checked).length;
    $("#measure-model-count").textContent = t("{n} selected", { n: selected });
    syncMeasureGuardKnobs();
  }

  function measurementValue(value, digits = 2, suffix = "") {
    return value == null || !Number.isFinite(Number(value)) ? "—" : `${Number(value).toFixed(digits)}${suffix}`;
  }

  function updateRunSummaries() {
    const voiceModel = selectedModel("#voice-model");
    const guard = $("#voice-onset-guard");
    const voiceSettings = [t("dry blend {value}", { value: Number($("#voice-dry-blend").value).toFixed(2) }), t(guard.checked && !guard.disabled ? "onset guard on" : "onset guard off"), onDevice() ? t("on this device") : selectLabel("#voice-provider")];
    $("#voice-run-summary").innerHTML = voiceModel
      ? `<strong>${escapeHtml(voiceModel.display_name)}</strong><span>${escapeHtml(voiceSettings.join(" · "))}</span>`
      : `<strong>${escapeHtml(t("No runnable enhancement model"))}</strong>`;
    const svModel = selectedModel("#sv-model");
    $("#sv-run-summary").innerHTML = svModel
      ? `<strong>${escapeHtml(svModel.display_name)}</strong><span>${escapeHtml([t("threshold {value}", { value: Number($("#sv-threshold").value).toFixed(2) }), selectLabel("#sv-provider")].join(" · "))}</span>`
      : `<strong>${escapeHtml(t("No runnable speaker model"))}</strong>`;
  }

  function applySvDefaults() {
    const model = selectedModel("#sv-model");
    const threshold = model?.recommended_inference?.threshold ?? model?.parameters?.threshold?.default;
    if (typeof threshold === "number") {
      $("#sv-threshold").value = threshold;
      $("#sv-threshold-output").textContent = threshold.toFixed(2);
    }
    updateRunSummaries();
  }

  function downloadText(filename, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function measurementCsv(report) {
    const columns = [
      "row", "model_id", "label", "display_name", "provider", "elapsed_seconds", "rtf", "rtf_runs",
      "dry_blend", "rms_dbfs", "peak_dbfs", "clipping_ratio", "silence_ratio",
      "spectral_centroid_hz", "si_sdr_db", "snr_db", "correlation", "stoi", "pesq_wb",
      "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_error", "error",
      "latency_samples", "onset_guard", "output_url",
    ];
    const cell = (value) => `"${String(value == null ? "" : value).replace(/"/g, '""')}"`;
    const line = (row, model) => {
      const output = model.output || {};
      const quality = model.quality || {};
      const dnsmos = model.reference_free?.dnsmos || {};
      return [
        row,
        model.model_id,
        model.label,
        model.display_name,
        model.provider,
        model.elapsed_seconds,
        model.rtf,
        (model.rtf_runs || []).join(" "),
        model.dry_blend,
        output.rms_dbfs,
        output.peak_dbfs,
        output.clipping_ratio,
        output.silence_ratio,
        output.spectral_centroid_hz,
        quality.si_sdr_db,
        quality.snr_db,
        quality.correlation,
        quality.stoi,
        quality.pesq_wb,
        dnsmos.dnsmos_p808,
        dnsmos.dnsmos_sig,
        dnsmos.dnsmos_bak,
        dnsmos.dnsmos_ovr,
        model.reference_free?.dnsmos_error,
        model.error,
        model.latency_samples,
        model.row === "unprocessed" ? "" : onsetGuardSummary(model.onset_guard),
        model.output_url,
      ].map(cell).join(",");
    };
    const rows = [];
    (report.clips || [report]).forEach((clip, index) => {
      const tag = (row) => (report.clips ? `${row}@${clip.input_files?.audio || `clip ${index + 1}`}` : row);
      if (clip.baseline) rows.push(line(tag("unprocessed"), { ...clip.baseline, model_id: "", row: "unprocessed" }));
      (clip.models || []).forEach((model) => rows.push(line(tag("model"), model)));
    });
    return [columns.map(cell).join(","), ...rows].join("\n") + "\n";
  }

  function exportMeasurement(format) {
    if (!state.measurementReport) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    if (format === "csv") {
      downloadText(`puresound-measurements-${stamp}.csv`, measurementCsv(state.measurementReport), "text/csv;charset=utf-8");
    } else {
      downloadText(`puresound-measurements-${stamp}.json`, JSON.stringify(state.measurementReport, null, 2), "application/json;charset=utf-8");
    }
  }

  const DECK_COLORS = ["#bdbbff", "#ef2cc1", "#7ee0a1", "#ffd166", "#ff9a62", "#8ecbff", "#f7a8d8", "#b5e48c"];

  /* A score cell: the value, its change from the unprocessed input, and a
   * mark when it is the best of the candidates on this clip. */
  function scoreCell(value, baseline, digits, suffix, best) {
    const number = Number(value);
    if (value == null || !Number.isFinite(number)) return "<td>—</td>";
    const base = Number(baseline);
    const delta = baseline != null && Number.isFinite(base) ? `<small class="score-delta ${number - base >= 0 ? "is-up" : "is-down"}">${number - base >= 0 ? "+" : ""}${(number - base).toFixed(digits)}</small>` : "";
    return `<td class="${best ? "is-best" : ""}">${number.toFixed(digits)}${suffix}${delta}</td>`;
  }

  /* One report may hold several clips: the aggregate on top, then one clip's
   * table and deck at a time, chosen by tab. */
  function renderMeasurementReport(report, { withDeck = true } = {}) {
    window.setTimeout(updateClearButtons, 0);
    exportLink("#measure-export-zip", report.job_id, "zip");
    exportLink("#measure-export-report", report.job_id, "report");
    state.measurementReport = report;
    const clips = report.clips || null;
    $("#measurement-aggregate").hidden = !clips;
    if (clips) {
      state.clipIndex = Math.min(state.clipIndex || 0, clips.length - 1);
      renderMeasurementAggregate(report);
      renderMeasurementClip(clips[state.clipIndex], report, withDeck);
    } else {
      renderMeasurementClip(report, report, withDeck);
    }
  }

  function aggregateCell(summary, digits, suffix) {
    if (!summary || summary.mean == null) return "<td>—</td>";
    const sign = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
    const interval = summary.ci_low == null ? t("one clip · no interval") : `[${sign(summary.ci_low)}, ${sign(summary.ci_high)}]`;
    const tone = !summary.resolved ? "is-unresolved" : summary.mean > 0 ? "is-up" : "is-down";
    const verdict = summary.resolved ? "" : ` · ${t(summary.n < 5 ? "too few clips to resolve" : "not resolved")}`;
    return `<td class="aggregate-cell ${tone}"><strong>${sign(summary.mean)}${suffix}</strong><small>${escapeHtml(interval)} · ${escapeHtml(t("{wins}/{n} up", { wins: summary.wins, n: summary.n }))}${escapeHtml(verdict)}</small></td>`;
  }

  function renderMeasurementAggregate(report) {
    const clips = report.clips;
    $("#measurement-aggregate-help").textContent = t("{n} clips. Each cell is the change from the unprocessed input of the same clip, averaged over clips, with a 95% bootstrap interval; “not resolved” means the interval spans zero, so this set cannot tell the difference apart from no difference. Fewer than 5 clips never resolve: an interval over so few values is far too narrow.", { n: clips.length });
    $("#measurement-aggregate-body").innerHTML = (report.aggregate || []).map((row) => `<tr><td><strong>${escapeHtml(row.label && row.label !== row.model_id ? row.label : row.model_id)}</strong>${row.clips_failed ? `<small class="table-error">${escapeHtml(t(row.clips_failed === 1 ? "{n} clip failed" : "{n} clips failed", { n: row.clips_failed }))}</small>` : ""}</td><td>${row.clips_ok}</td><td>${measurementValue(row.rtf_median, 3)}</td>${aggregateCell(row.metrics?.dnsmos_ovr, 2, "")}${aggregateCell(row.metrics?.si_sdr_db, 2, " dB")}${aggregateCell(row.metrics?.stoi, 3, "")}${aggregateCell(row.metrics?.pesq_wb, 2, "")}</tr>`).join("");
    $("#measurement-clip-tabs").innerHTML = clips.map((clip, index) => `<button type="button" role="tab" aria-selected="${index === state.clipIndex}" class="${index === state.clipIndex ? "is-active" : ""}" data-clip="${index}">${escapeHtml(clip.input_files?.audio || t("Clip {n}", { n: index + 1 }))}</button>`).join("");
    $$("[data-clip]", $("#measurement-clip-tabs")).forEach((button) => button.addEventListener("click", () => {
      state.clipIndex = Number(button.dataset.clip);
      renderMeasurementAggregate(report);
      renderMeasurementClip(clips[state.clipIndex], report);
    }));
  }

  function renderMeasurementClip(report, whole = report, withDeck = true) {
    $("#measurement-report-content").hidden = false;
    $("#measure-export-json").hidden = false;
    $("#measure-export-csv").hidden = false;
    $("#measure-export-menu").hidden = false;
    const input = report.input || {};
    const reference = report.reference;
    const baseline = report.baseline;
    const models = report.models || [];
    const dnsMosErrors = models.map((model) => model.reference_free?.dnsmos_error).filter(Boolean);
    const summary = t(reference
      ? "Reference supplied — use SI-SDR, STOI and PESQ against the unprocessed row, then check speed and clipping for deployment."
      : "No clean reference — DNSMOS gives a reference-free signal next to the unprocessed row; also compare speed, level, and clipping.");
    const failedModels = Number(report.summary?.failed || 0);
    const resultSummary = dnsMosErrors.length > 0 && dnsMosErrors.length === models.length ? `${summary} ${t("DNSMOS is unavailable in this runtime.")}` : summary;
    $("#measurement-result-summary").textContent = failedModels
      ? `${t(failedModels === 1 ? "{n} selected model failed." : "{n} selected models failed.", { n: failedModels })} ${resultSummary}`
      : resultSummary;
    $("#measurement-summary").innerHTML = [
      [t("Input RMS"), measurementValue(input.rms_dbfs, 1, " dBFS")],
      [t("Input peak"), measurementValue(input.peak_dbfs, 1, " dBFS")],
      [t("Input clipping"), measurementValue(input.clipping_ratio * 100, 2, "%")],
      [t("Reference"), reference ? `${measurementValue(reference.duration_seconds, 2, " s")} · ${measurementValue(reference.rms_dbfs, 1, " dBFS")}` : t("Not provided")],
    ].map(([label, value]) => `<div class="measurement-summary-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    $("#measurement-result-time").textContent = [t("{seconds} total", { seconds: measurementValue(whole.elapsed_seconds, 2, " s") }), hostLoadLabel(whole.host)].filter(Boolean).join(" · ");
    const repeats = whole.request?.measurements?.timing_repeats || 1;
    const clipPart = whole.clips ? `${t("Clip {n} of {total}", { n: state.clipIndex + 1, total: whole.clips.length })}: ${report.input_files?.audio || ""} · ` : "";
    $("#measurement-applied").textContent = `${clipPart}${t("Provider {provider}", { provider: whole.request?.provider || "auto" })} · ${repeats > 1 ? t("RTF median of {n} runs", { n: repeats }) : t("RTF one run")} · ${t("settings per row")}`;
    const ok = models.filter((model) => !model.error);
    const metrics = {
      dnsmos: (model) => model.reference_free?.dnsmos?.dnsmos_ovr,
      sisdr: (model) => model.quality?.si_sdr_db,
      stoi: (model) => model.quality?.stoi,
      pesq: (model) => model.quality?.pesq_wb,
    };
    const best = Object.fromEntries(Object.entries(metrics).map(([key, read]) => {
      const values = ok.map(read).map(Number).filter(Number.isFinite);
      return [key, values.length > 1 ? Math.max(...values) : null];
    }));
    const isBest = (key, model) => best[key] != null && Number(metrics[key](model)) === best[key];
    const baselineRow = baseline
      ? `<tr class="is-baseline"><td><strong>${escapeHtml(t("Unprocessed input"))}</strong><small>${escapeHtml(t("what leaving the audio alone scores"))}</small></td><td>—</td><td>${measurementValue(baseline.output?.rms_dbfs, 1, " dB")}</td><td>${measurementValue(baseline.output?.peak_dbfs, 1, " dB")}</td><td>${measurementValue(baseline.output?.clipping_ratio * 100, 2, "%")}</td>${scoreCell(metrics.dnsmos(baseline), null, 2, "", false)}${scoreCell(metrics.sisdr(baseline), null, 2, " dB", false)}${scoreCell(metrics.stoi(baseline), null, 3, "", false)}${scoreCell(metrics.pesq(baseline), null, 2, "", false)}</tr>`
      : "";
    $("#measurement-table-body").innerHTML = baselineRow + models.map((model) => {
      const output = model.output || {};
      const title = model.label && model.label !== model.model_id ? model.label : (model.display_name || model.model_id);
      if (model.error) return `<tr><td><strong>${escapeHtml(title)}</strong><small class="table-error">${escapeHtml(model.error)}</small></td><td colspan="8">—</td></tr>`;
      const runs = (model.rtf_runs || []).filter(Number.isFinite);
      const rtfSpread = runs.length > 1 ? `<small>${escapeHtml(t("{n} runs", { n: runs.length }))} · ${Math.min(...runs).toFixed(3)}–${Math.max(...runs).toFixed(3)}</small>` : "";
      const settings = [model.stages?.length ? t("after {models}", { models: model.stages.map((stage) => shortModelName(stage)).join(" → ") }) : "", t("blend {value}", { value: measurementValue(model.dry_blend, 2) }), t(model.onset_guard ? "guard on" : "guard off"), providerLabel(model.provider)].filter(Boolean).join(" · ");
      const dnsTitle = model.reference_free?.dnsmos_error ? ` title="${escapeHtml(t("DNSMOS unavailable: {reason}", { reason: model.reference_free.dnsmos_error }))}"` : "";
      return `<tr><td title="${escapeHtml(model.display_name || model.model_id)}"><strong>${escapeHtml(title)}</strong><small>${escapeHtml(settings)}</small></td><td>${measurementValue(model.rtf, 3)}${rtfSpread}</td><td>${measurementValue(output.rms_dbfs, 1, " dB")}</td><td>${measurementValue(output.peak_dbfs, 1, " dB")}</td><td>${measurementValue(output.clipping_ratio * 100, 2, "%")}</td>${scoreCell(metrics.dnsmos(model), metrics.dnsmos(baseline || {}), 2, "", isBest("dnsmos", model)).replace("<td", `<td${dnsTitle}`)}${scoreCell(metrics.sisdr(model), metrics.sisdr(baseline || {}), 2, " dB", isBest("sisdr", model))}${scoreCell(metrics.stoi(model), metrics.stoi(baseline || {}), 3, "", isBest("stoi", model))}${scoreCell(metrics.pesq(model), metrics.pesq(baseline || {}), 2, "", isBest("pesq", model))}</tr>`;
    }).join("");
    if (withDeck) renderMeasurementDeck(report);
  }

  async function renderMeasurementDeck(report) {
    const deck = state.measureDeck;
    const tracks = [];
    if (report.baseline?.output_url) tracks.push({ id: "unprocessed", label: "Unprocessed", color: "#c8f6f9", hint: "the input, at the model rate", levelReference: true, url: report.baseline.output_url });
    (report.models || []).filter((model) => model.output_url && !model.error).forEach((model, index) => {
      const title = model.label && model.label !== model.model_id ? model.label : shortModelName(model);
      tracks.push({ id: `candidate-${model.candidate_id ?? index}`, label: title, color: DECK_COLORS[index % DECK_COLORS.length], hint: `${t("blend {value}", { value: measurementValue(model.dry_blend, 2) })} · ${t(model.onset_guard ? "guard on" : "guard off")}`, url: model.output_url });
    });
    if (report.reference_url) tracks.push({ id: "reference", label: "Clean reference", color: "#b5e48c", hint: "what the output should approach", url: report.reference_url });
    deck.setTracks(tracks);
    const loads = await Promise.allSettled(tracks.map((track) => deck.loadTrack(track.id, track.url)));
    const failed = loads.filter((item) => item.status === "rejected").length;
    if (failed) showToast(t(failed === tracks.length ? "{failed} of {total} tracks could not be loaded — the stored outputs may have expired." : "{failed} of {total} tracks could not be loaded.", { failed, total: tracks.length }), true);
    const firstCandidate = tracks.find((track) => track.id.startsWith("candidate-"));
    if (firstCandidate) deck.select(firstCandidate.id);
  }

  async function runMeasurements() {
    const clips = measurementClips();
    const note = $("#measure-form-note");
    const candidates = measurementCandidates();
    const refuse = (message) => {
      note.textContent = message; note.className = "form-note is-error";
      shell().openInspector("compare", { focus: false });
    };
    if (!clips.length) { refuse(t("Choose at least one recording to compare.")); return; }
    if (!candidates.length) { refuse(t("Select at least one model to compare.")); return; }
    setRunBusy("measure", true); note.textContent = clips.length > 1 ? t("Uploading {n} recordings…", { n: clips.length }) : t("Measuring input and running selected models…"); note.className = "form-note";
    try {
      const uploaded = [];
      for (const clip of clips) {
        const entry = { audio: await uploadDescriptor(clip.file) };
        if (clip.reference) entry.reference = await uploadDescriptor(clip.reference);
        uploaded.push(entry);
      }
      setKey(note, "Measuring inputs and running selected models…");
      const report = await runBackgroundJob(
        {
          kind: "measurement",
          inputs: uploaded.length === 1 ? uploaded[0] : { clips: uploaded },
          candidates,
          provider: "auto",
          parameters: {},
          measurements: { include_dnsmos: true, timing_repeats: Number($("#measure-timing-repeats").value) || 1 },
        },
        "measure",
      );
      state.clipIndex = 0;
      renderMeasurementReport(report);
      const failed = Number(report.summary?.failed || 0);
      note.textContent = failed ? t(failed === 1 ? "Comparison complete with {n} failed run." : "Comparison complete with {n} failed runs.", { n: failed }) : t("Comparison complete.");
      note.className = failed ? "form-note is-error" : "form-note is-success";
      shell().closeInspector();
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; }
    finally { setRunBusy("measure", false); }
  }

  async function runValidation() {
    const button = $("#validate-button");
    setBusy(button, true);
    $("#validation-status").className = "validation-status";
    $("#validation-status").innerHTML = `<span class="status-icon">…</span><div><strong>${escapeHtml(t("Checking the catalog…"))}</strong><span>${escapeHtml(t("Paths, sidecars, hashes and ONNX graph contracts."))}</span></div>`;
    try {
      state.validation = await api("/api/validate");
      renderValidation(state.validation);
    } catch (error) { showToast(error.message, true); }
    finally { setBusy(button, false); }
  }

  function renderValidation(report) {
    const status = $("#validation-status");
    status.classList.add(report.ok ? "is-ok" : "is-error");
    status.innerHTML = `<span class="status-icon">${report.ok ? "✓" : "!"}</span><div><strong>${escapeHtml(t(report.ok ? "The catalog is valid" : "The check found issues"))}</strong><span>${escapeHtml(report.ok ? t("Every registered artifact passed the checks.") : t("{n} issues need attention.", { n: report.errors.length }))}</span></div>`;
    $("#validation-summary").innerHTML = `<div class="validation-item"><span>${escapeHtml(t("Logical models"))}</span><strong>${report.models}</strong></div><div class="validation-item"><span>${escapeHtml(t("ONNX artifacts"))}</span><strong>${report.artifacts}</strong></div><div class="validation-item"><span>${escapeHtml(t("Checks"))}</span><strong>${escapeHtml(t(report.ok ? "Passed" : "Review"))}</strong></div>`;
    const errors = $("#validation-errors");
    errors.hidden = report.ok;
    errors.textContent = (report.errors || []).join("\n");
  }

  function runTask(key) {
    // Ctrl+Enter reaches here while a run is going; its button is disabled then.
    if ($(`[data-run="${key}"]`)?.disabled) return;
    if (key === "voice") runVoiceInference();
    else if (key === "sv") runSpeakerVerification();
    else if (key === "measure") runMeasurements();
  }

  /* The deck keyboard shortcuts act on: the one last clicked if it is on
   * screen, else the screen's main deck. */
  function activeDeck() {
    const visible = (deck) => deck && deck.root.offsetParent !== null && deck.duration > 0;
    const focused = window.PureSoundCompareDeck.focused;
    if (visible(focused)) return focused;
    const screen = shell().current();
    if (screen === "playground") {
      if (!$("#voice-result").hidden) return state.voiceDeck;
      if (visible(state.audioPanels.voiceInput?.deck)) return state.audioPanels.voiceInput.deck;
    }
    if (screen === "compare" && state.measureDeck && !$("#measurement-report-content").hidden) return state.measureDeck;
    if (screen === "pipeline" && visible(window.PureSoundPipeline?.deck())) return window.PureSoundPipeline.deck();
    return null;
  }

  function overlayOpen() {
    return shell().overlayOpen();
  }

  /* Listening shortcuts for the deck on screen.  They stay out of the way of
   * form controls, and of a deck button that Space would click anyway. */
  function handleShortcut(event) {
    // Annotate has its own keys (annotate.js); the player's stand aside there.
    if (shell().current() === "annotate") return;
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      const screen = shell().current();
      if (screen === "playground") runTask("voice");
      else if (screen === "verify") runTask("sv");
      else if (screen === "compare") runTask("measure");
      else if (screen === "pipeline") window.PureSoundPipeline?.run();
      else if (screen === "world") window.PureSoundWorld?.run();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey || overlayOpen()) return;
    const deck = activeDeck();
    if (!deck) return;
    const control = event.target.closest?.("input, select, textarea, button, a, [contenteditable]");
    if (control && (!deck.root.contains(control) || event.key === " ")) return;
    const key = event.key;
    if (key === " ") deck.toggle();
    else if (/^[1-9]$/.test(key)) deck.selectIndex(Number(key) - 1);
    else if (key === "l" || key === "L") deck.toggleLoop();
    else if (key === "z" || key === "Z") deck.toggleZoom();
    else if (key === "+" || key === "=") deck.zoomTime(0.5, deck.zoomAnchor());
    else if (key === "-" || key === "_") deck.zoomTime(2, deck.zoomAnchor());
    else if (key === "0") deck.fitView();
    else if (key === "ArrowLeft") deck.nudge(event.shiftKey ? -5 : -1);
    else if (key === "ArrowRight") deck.nudge(event.shiftKey ? 5 : 1);
    else if (key === "Home") deck.seekTo(deck.loop && deck.region ? deck.region.start : 0);
    else if (key === "Escape" && (deck.region || deck.view)) deck.clearRegion();
    else return;
    event.preventDefault();
  }

  /* Several recordings, each paired with the reference of the same name. */
  function wireMeasurementFile(inputId, key) {
    const input = $(inputId);
    input.addEventListener("change", () => {
      const files = [...(input.files || [])];
      if (!files.length) return;
      setMeasurementFiles(key, files);
    });
  }

  function fileStem(name) {
    return String(name || "").replace(/\.[^.]+$/, "").toLowerCase();
  }

  function setMeasurementFiles(key, files) {
    state.measurementFiles[key] = files;
    const label = $(`[data-measure-name="${key}"]`);
    const meta = $(`[data-measure-meta="${key}"]`);
    label.removeAttribute("data-i18n");
    meta.removeAttribute("data-i18n");
    label.textContent = files.length === 1 ? files[0].name : `${t("{n} files", { n: files.length })} · ${files.slice(0, 3).map((file) => file.name).join(", ")}${files.length > 3 ? "…" : ""}`;
    const kilobytes = files.reduce((sum, file) => sum + file.size, 0) / 1024;
    meta.textContent = `${kilobytes.toFixed(0)} KB · ${t("ready for analysis")}`;
    const pairs = measurementClips();
    if (pairs.length && (state.measurementFiles.reference || []).length) {
      const matched = pairs.filter((pair) => pair.reference).length;
      const referenceMeta = $('[data-measure-meta="reference"]');
      referenceMeta.removeAttribute("data-i18n");
      referenceMeta.textContent = t("{matched} of {total} recordings have a matching reference", { matched, total: pairs.length });
    }
  }

  function measurementClips() {
    const recordings = state.measurementFiles.audio || [];
    const references = state.measurementFiles.reference || [];
    return recordings.map((file) => {
      let reference = references.find((item) => fileStem(item.name) === fileStem(file.name)) || null;
      if (!reference && recordings.length === 1 && references.length === 1) reference = references[0];
      return { file, reference };
    });
  }

  /* Samples -------------------------------------------------------------- */
  async function loadSamples() {
    try {
      const response = await fetch("/samples/index.json");
      if (!response.ok) return;
      state.samples = await response.json();
    } catch { return; }
    renderSampleRows();
  }

  function renderSampleRows() {
    if (!state.samples) return;
    const clips = state.samples.clips || [];
    const enhancement = clips.filter((clip) => clip.task === "enhancement");
    const voiceRow = $("#voice-samples");
    voiceRow.hidden = !enhancement.length;
    voiceRow.innerHTML = `<span>${escapeHtml(t("Try a sample"))}</span>${enhancement.map((clip) => `<button type="button" class="sample-chip" data-sample="${escapeHtml(clip.id)}" title="${escapeHtml(clip.description)}">${escapeHtml(clip.title)} <small>${clip.seconds.toFixed(1)} s</small></button>`).join("")}`;
    $$("[data-sample]", voiceRow).forEach((button) => button.addEventListener("click", async () => {
      const file = await sampleFile(button.dataset.sample);
      if (file) state.acceptors["#voice-audio"](file);
    }));
    const pairs = state.samples.speaker_pairs || [];
    const svRow = $("#sv-samples");
    svRow.hidden = !pairs.length;
    svRow.innerHTML = `<span>${escapeHtml(t("Try a pair"))}</span>${pairs.map((pair) => `<button type="button" class="sample-chip" data-pair="${escapeHtml(pair.id)}">${escapeHtml(pair.title)}</button>`).join("")}`;
    $$("[data-pair]", svRow).forEach((button) => button.addEventListener("click", async () => {
      const pair = pairs.find((item) => item.id === button.dataset.pair);
      const [enrollment, test] = await Promise.all([sampleFile(pair.enrollment), sampleFile(pair.test)]);
      if (enrollment) state.acceptors["#sv-enrollment"](enrollment);
      if (test) state.acceptors["#sv-test"](test);
    }));
    const useSamples = $("#measure-use-samples");
    useSamples.hidden = !enhancement.length;
    useSamples.textContent = t("Use the {n} sample clips", { n: enhancement.length });
  }

  /* A sample as a File, with a fixed timestamp so its upload is reused. */
  async function sampleFile(id) {
    const clip = state.samples?.clips?.find((item) => item.id === id);
    if (!clip) return null;
    try {
      const response = await fetch(`/samples/${encodeURIComponent(clip.file)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return new File([await response.blob()], clip.file, { type: "audio/wav", lastModified: 0 });
    } catch (error) {
      showToast(t("Could not load the sample: {reason}", { reason: error.message }), true);
      return null;
    }
  }

  /* Recording ------------------------------------------------------------ */
  function formatClock(seconds) {
    const tenths = Math.round(seconds * 10);
    const minutes = Math.floor(tenths / 600);
    return `${minutes}:${((tenths - minutes * 600) / 10).toFixed(1).padStart(4, "0")}`;
  }

  /* A record button that turns the microphone into a file for `inputId`. */
  function mountRecorder(host, inputId, { compact = false } = {}) {
    if (!host) return;
    const problem = window.PureSoundCapture.captureAvailability();
    host.innerHTML = `<div class="recorder${compact ? " is-compact" : ""}">
      <button class="button button-secondary${compact ? " button-small" : ""} record-button" type="button"${problem ? " disabled" : ""}><span class="record-dot" aria-hidden="true"></span><span data-record-label data-i18n="${compact ? "Record" : "Start recording"}">${escapeHtml(t(compact ? "Record" : "Start recording"))}</span></button>
      <div class="level-meter" aria-hidden="true"><span data-record-level></span></div>
      <span class="record-time" data-record-time>0:00.0</span>
      ${compact ? "" : `<label class="mic-select"><span data-i18n>Microphone</span><select class="control" data-mic-select aria-label="${escapeHtml(t("Microphone"))}"><option value="">${escapeHtml(t("System default"))}</option></select></label>`}
      ${compact ? "" : `<p class="toggle-hint" data-record-note${problem ? "" : ' data-i18n="Records at 16 kHz with the browser\'s echo cancellation, noise suppression and gain control off, so the model hears your capture chain as it is. Stop, then run on the recording."'}>${escapeHtml(problem || t("Records at 16 kHz with the browser's echo cancellation, noise suppression and gain control off, so the model hears your capture chain as it is. Stop, then run on the recording."))}</p>`}
    </div>`;
    window.PureSoundI18n?.apply(host);
    if (problem && compact) host.querySelector(".record-button").title = problem;
    const button = host.querySelector(".record-button");
    const label = host.querySelector("[data-record-label]");
    const level = host.querySelector("[data-record-level]");
    const clock = host.querySelector("[data-record-time]");
    let recorder = null;
    button.addEventListener("click", async () => {
      if (!recorder) {
        recorder = new window.PureSoundCapture.Recorder({
          onLevel: (rms) => { level.style.width = `${Math.min(100, Math.max(0, (20 * Math.log10(rms + 1e-9) + 60) / 60 * 100))}%`; },
          onTime: (seconds) => { clock.textContent = formatClock(seconds); },
        });
        try {
          const device = await recorder.start({ deviceId: selectedMicrophone() });
          refreshMicrophones();
          host.querySelector(".recorder").classList.add("is-recording");
          if (compact) setKey(label, "Stop");
          else setKey(label, "Stop · {device}", { device });
        } catch (error) {
          recorder = null;
          showToast(t("Microphone unavailable: {reason}", { reason: error.message }), true);
        }
        return;
      }
      const active = recorder;
      recorder = null;
      host.querySelector(".recorder").classList.remove("is-recording");
      setKey(label, compact ? "Record" : "Start recording");
      level.style.width = "0";
      const file = await active.stop();
      if (!file) { showToast(t("Nothing was recorded."), true); return; }
      state.acceptors[inputId](file);
      if (inputId === "#voice-audio") setInputMode("file");
      showToast(t("Recorded {name}.", { name: file.name }));
    });
  }

  /* Input modes: a file (upload, sample or recording) or a live session. */
  function setInputMode(mode) {
    if (state.live && mode !== "live") return; // stop the live session first
    state.inputMode = mode;
    $$("[data-input-mode]").forEach((button) => { const active = button.dataset.inputMode === mode; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", active ? "true" : "false"); });
    $$("[data-input-panel]").forEach((panel) => { panel.hidden = panel.dataset.inputPanel !== mode; });
    $("#voice-run-bar").hidden = mode === "live";
    // Live has its own start button; Run is for a file or a recording.
    $("#voice-run").disabled = mode === "live" || Boolean(state.activeJobs.voice || state.deviceRun);
    const titles = { file: "Upload a noisy recording", record: "Record with your microphone", live: "Live through the model" };
    setKey($("#voice-input-title"), titles[mode]);
    if (mode === "live") $("#voice-input-preview").hidden = true;
    else if (selectedFile("#voice-audio")) $("#voice-input-preview").hidden = !$("#voice-result").hidden && Boolean(state.voiceDeck.track("input"));
  }

  /* Live ----------------------------------------------------------------- */
  function renderLiveStats(stats) {
    const rows = [
      [t("Streamed"), formatClock(stats.seconds)],
      [t("Model RTF"), stats.rtf == null ? "—" : stats.rtf.toFixed(3)],
      [t("Per 20 ms chunk"), stats.processingMs == null ? "—" : t("{ms} ms compute", { ms: stats.processingMs.toFixed(1) })],
      [t("Round trip"), stats.roundTripMs == null ? "—" : `${stats.roundTripMs.toFixed(1)} ms`],
      [t("Algorithmic"), t("{low}–{high} ms (window {window} + look-ahead {ahead})", { low: (stats.windowMs - stats.hopMs + stats.lookAheadMs).toFixed(0), high: (stats.windowMs + stats.lookAheadMs).toFixed(0), window: stats.windowMs.toFixed(0), ahead: stats.lookAheadMs.toFixed(0) })],
      [t("Mic → ear, estimated"), stats.roundTripMs == null ? "—" : `≈ ${(stats.chunkMs + stats.roundTripMs + stats.windowMs + stats.lookAheadMs + stats.jitterMs + stats.deviceMs).toFixed(0)} ms`],
      [t("Jitter buffer"), t("{ms} ms · {n} underrun samples", { ms: stats.jitterMs.toFixed(0), n: stats.underruns })],
    ];
    if (stats.backlog > 5) rows.push([t("Backlog"), t(stats.runner === "device" ? "{n} chunks waiting — this device is falling behind" : "{n} chunks waiting — the server is falling behind", { n: stats.backlog })]);
    $("#live-stats").innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  }

  async function toggleLive() {
    const button = $("#live-toggle");
    const note = $("#live-note");
    if (state.live) {
      const session = state.live;
      state.live = null;
      updateClearButtons();
      button.classList.remove("is-recording");
      setKey($("[data-live-label]", button), "Start live");
      const result = await session.stop();
      $("#live-level").style.width = "0";
      if (result.error) note.textContent = result.error;
      else if (result.seconds > 0.5) setKey(note, "Streamed {seconds} s. The session is below: switch Input / Output / Removed.", { seconds: result.seconds.toFixed(1) });
      else setKey(note, "The session was too short to compare.");
      note.className = result.error ? "form-note is-error" : "form-note";
      if (result.seconds > 0.5) await showLiveComparison(result);
      return;
    }
    const model = selectedModel("#voice-model");
    if (!model) return;
    const problem = window.PureSoundCapture.captureAvailability();
    if (problem) { note.textContent = problem; note.className = "form-note is-error"; return; }
    const session = new window.PureSoundCapture.LiveSession({
      onStats: renderLiveStats,
      onLevel: (rms) => { $("#live-level").style.width = `${Math.min(100, Math.max(0, (20 * Math.log10(rms + 1e-9) + 60) / 60 * 100))}%`; },
      onStatus: (status) => {
        if (status.error) { note.textContent = status.error; note.className = "form-note is-error"; }
        if (status.ended && state.live === session) toggleLive();
      },
    });
    const device = onDevice();
    const dryBlend = Number($("#voice-dry-blend").value);
    const link = device
      ? window.PureSoundDevice.liveLink({ model: model.id, dryBlend, onsetGuard: deviceOnsetGuard() })
      : window.PureSoundCapture.serverLink({ modelId: model.id, variant: $("#voice-variant").value, provider: $("#voice-provider").value, parameters: { dry_blend: dryBlend, ...onsetGuardParameters("voice") } });
    button.disabled = true;
    setKey(note, device ? "Loading the model on this device and measuring its speed…" : "Connecting and loading the model — the first session after the server starts takes a few seconds…");
    note.className = "form-note";
    try {
      const ready = await session.start({ link, monitor: $(".segmented [data-monitor].is-active")?.dataset.monitor || "off", deviceId: selectedMicrophone() });
      refreshMicrophones();
      state.live = session;
      updateClearButtons();
      button.classList.add("is-recording");
      setKey($("[data-live-label]", button), "Stop");
      const where = device ? deviceLabel(ready.threads) : providerLabel(ready.provider);
      note.textContent = `${t("Live on {model}", { model: model.display_name })} · ${where} · ${session.capture?.device || t("microphone")}${!device && $("#voice-stage").value ? ` · ${t("the “Run first” chain is not used live")}` : ""}.`;
    } catch (error) {
      await session.stop().catch(() => {});
      note.textContent = error.message;
      note.className = "form-note is-error";
    } finally {
      button.disabled = false;
    }
  }

  async function showLiveComparison(result) {
    window.setTimeout(updateClearButtons, 0);
    const { encodeWav } = window.PureSoundCapture;
    const blob = (samples) => new Blob([encodeWav(samples, result.sampleRate)], { type: "audio/wav" });
    const model = selectedModel("#voice-model");
    const onDevice = result.runner === "device";
    const tracks = [
      { id: "input", label: "Input", color: "#c8f6f9", hint: "the microphone, as streamed", levelReference: true, data: blob(result.input) },
      { id: "output", label: "Output", color: "#bdbbff", hint: `${shortModelName(model)} · ${t("live")}${onDevice ? ` · ${t("this device")}` : ""}`, data: blob(result.output) },
      { id: "removed", label: "Removed", color: "#ff9a62", hint: "input − output", diagnostic: true, data: blob(result.removed) },
    ];
    const deck = state.voiceDeck;
    $("#voice-result").hidden = false;
    state.voiceView = null;
    $("#voice-result-name").textContent = `${t("Live session")} · ${model.display_name}${onDevice ? ` · ${t("this device")}` : ""}`;
    // The device stream is flushed at the end, so its output covers all of the input.
    $("#voice-alignment-note").textContent = t(onDevice ? "Aligned: the stream's {latency} look-ahead removed." : "Aligned: the stream's {latency} look-ahead removed; the last window of input is not included.", { latency: measurementValue(result.ready?.latency_ms, 0, " ms") });
    $("#voice-result-meta").textContent = `${result.sampleRate / 1000} kHz · ${formatSeconds(result.seconds)}`;
    releaseBlobUrls();
    const downloads = { "#voice-download": tracks[1], "#voice-download-removed": tracks[2] };
    Object.entries(downloads).forEach(([selector, track]) => { const link = $(selector); link.href = blobUrl(track.data); link.hidden = false; });
    $("#voice-download-raw").href = blobUrl(tracks[0].data);
    $("#voice-download-raw").download = "live_input.wav";
    $("#voice-download-raw").hidden = false;
    $("#voice-metrics").innerHTML = "";
    $("#voice-applied").textContent = `${t("Live session")} · ${t(onDevice ? "processed on this device; run the recording as a file for RTF." : "measured in the browser; run the recording as a file for DNSMOS and RTF.")}`;
    exportLink("#voice-export-zip", result.jobId, "zip");
    exportLink("#voice-export-report", result.jobId, "report");
    $("#voice-measurement-note").textContent = "";
    deck.setTracks(tracks);
    deck.setCurves([]);
    await Promise.allSettled(tracks.map((track) => deck.loadTrack(track.id, track.data)));
    deck.select("output");
    state.voiceLast = null;
  }

  /* Clearing -------------------------------------------------------------- */
  /* Each workspace can be emptied back to its first state.  Not while a job
   * or a live session is running: its result would land on a cleared page. */
  function updateClearButtons() {
    const voiceHas = Boolean(selectedFile("#voice-audio")) || !$("#voice-result").hidden;
    const svHas = Boolean(selectedFile("#sv-enrollment") || selectedFile("#sv-test")) || !$("#sv-result").hidden;
    const states = {
      voice: [voiceHas, Boolean(state.activeJobs.voice || state.live || state.deviceRun)],
      sv: [svHas, Boolean(state.activeJobs.sv)],
      measure: [Boolean(state.measurementReport), Boolean(state.activeJobs.measure)],
    };
    $$("[data-clear]").forEach((button) => {
      const [has, busy] = states[button.dataset.clear] || [false, false];
      button.hidden = !has;
      button.disabled = busy;
      if (!button.dataset.titleIdle) button.dataset.titleIdle = button.getAttribute("data-i18n-src-title") || button.title;
      button.title = t(busy ? "Wait for the run to finish, or cancel it" : button.dataset.titleIdle);
    });
  }

  function resetFileInput(inputId) {
    delete state.files[inputId];
    const input = $(inputId);
    input.value = "";
    const dropzone = input.closest(".dropzone");
    dropzone.classList.remove("has-file", "is-dragging");
    const title = dropzone.querySelector("[data-dropzone-title]");
    const hint = dropzone.querySelector("[data-dropzone-hint]");
    if (title) setKey(title, dropzone.dataset.title);
    if (hint) setKey(hint, dropzone.dataset.hint);
    delete state.dropzoneFiles[inputId];
  }

  function resetDownload(selector) {
    const link = $(selector);
    if (link.href.startsWith("blob:")) URL.revokeObjectURL(link.href);
    link.href = "#";
  }

  function clearVoiceWorkspace() {
    if (state.activeJobs.voice || state.live || state.deviceRun) return;
    state.voiceDeck.setTracks([]);
    state.voiceDeck.setCurves([]);
    state.audioPanels.voiceInput.clear();
    resetFileInput("#voice-audio");
    $("#voice-input-preview").hidden = true;
    $("#voice-result").hidden = true;
    ["#voice-download", "#voice-download-raw", "#voice-download-removed"].forEach(resetDownload);
    releaseBlobUrls();
    exportLink("#voice-export-zip", null);
    exportLink("#voice-export-report", null);
    $("#voice-metrics").innerHTML = "";
    state.voiceLast = null;
    const note = $("#voice-form-note");
    setKey(note, "Select an audio file to begin.");
    note.className = "form-note";
    $("#live-note").textContent = "";
    $("#live-stats").innerHTML = `<div><dt>${escapeHtml(t("Status"))}</dt><dd>${escapeHtml(t("Idle"))}</dd></div>`;
    updateClearButtons();
  }

  function clearSvWorkspace() {
    if (state.activeJobs.sv) return;
    [["#sv-enrollment", state.audioPanels.svEnrollment, "#sv-enrollment-preview"], ["#sv-test", state.audioPanels.svTest, "#sv-test-preview"]].forEach(([inputId, panel, preview]) => {
      panel.clear();
      resetFileInput(inputId);
      $(preview).hidden = true;
    });
    $("#sv-result").hidden = true;
    $("#sv-metrics").innerHTML = "";
    const note = $("#sv-form-note");
    setKey(note, "Select enrollment and test audio.");
    note.className = "form-note";
    updateClearButtons();
  }

  /* The report only: what Configure holds (recordings, rows) stays, so the
   * same comparison can run again. */
  function clearMeasurementReport() {
    if (state.activeJobs.measure) return;
    state.measurementReport = null;
    state.clipIndex = 0;
    state.measureDeck.setTracks([]);
    $("#measurement-report-content").hidden = true;
    $("#measurement-aggregate").hidden = true;
    $("#measure-export-json").hidden = true;
    $("#measure-export-csv").hidden = true;
    $("#measure-export-menu").hidden = true;
    exportLink("#measure-export-zip", null);
    exportLink("#measure-export-report", null);
    $("#measurement-summary").innerHTML = state.measurementEmptyHtml;
    window.PureSoundI18n?.apply($("#measurement-summary"));
    $("#measurement-result-summary").textContent = "";
    $("#measurement-result-time").textContent = "";
    updateClearButtons();
  }

  /* What the live session plays while it streams.  It never changes what is
   * sent to the model or kept for the comparison afterwards. */
  const MONITOR_EXPLANATIONS = {
    off: "Nothing plays. The model still processes everything; after you stop, listen to the input and the output side by side below.",
    model: "You hear the model's output as you speak, a little over a tenth of a second late. Use headphones — through speakers the output feeds back into the microphone.",
    raw: "You hear your microphone straight through, unprocessed. Switch between this and Model output while talking to hear what the model takes out: a live A/B.",
  };

  function setMonitor(mode) {
    $$("[data-monitor]").forEach((item) => {
      const active = item.dataset.monitor === mode;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-checked", active ? "true" : "false");
    });
    state.monitor = mode;
    setKey($("#monitor-explain"), MONITOR_EXPLANATIONS[mode] || "");
    state.live?.capture?.setMonitor(mode);
  }

  /* Microphone choice ----------------------------------------------------- */
  const MIC_KEY = "puresound.microphone";

  function selectedMicrophone() {
    try { return localStorage.getItem(MIC_KEY) || ""; } catch { return ""; }
  }

  async function refreshMicrophones() {
    const devices = await window.PureSoundCapture.listMicrophones().catch(() => []);
    const saved = selectedMicrophone();
    const named = devices.some((device) => device.label);
    $$("[data-mic-select]").forEach((select) => {
      select.innerHTML = `<option value="">${escapeHtml(t("System default"))}</option>${devices.map((device, index) => `<option value="${escapeHtml(device.deviceId)}">${escapeHtml(device.label || t("Microphone {n}", { n: index + 1 }))}</option>`).join("")}`;
      select.value = devices.some((device) => device.deviceId === saved) ? saved : "";
      select.title = named ? "" : t("Names appear after the page has used the microphone once");
    });
  }

  function chooseMicrophone(deviceId) {
    try { localStorage.setItem(MIC_KEY, deviceId); } catch { /* storage may be disabled */ }
    $$("[data-mic-select]").forEach((select) => { select.value = deviceId; });
  }

  /* Runs on: the server, or this device ------------------------------------ */
  /* On this device the model runs in the browser (device/device.js) for File,
   * Record and Live alike.  Only models with a device build can; the others
   * stay in the list, marked, so it is clear why they cannot be picked. */
  const RUNNER_KEY = "puresound.runner";

  async function loadDeviceStatus() {
    try { state.runner = localStorage.getItem(RUNNER_KEY) === "device" ? "device" : "server"; } catch { /* storage may be disabled */ }
    state.device = await window.PureSoundDevice.status();
    renderRunner();
  }

  function deviceModels() {
    return state.device?.ready ? state.voiceModels.filter((model) => state.device.models.includes(model.id)) : [];
  }

  /* The choice is kept even while no model can run here; it applies once one can. */
  function onDevice() {
    return state.runner === "device" && deviceModels().length > 0;
  }

  function setRunner(runner) {
    if (runner === (onDevice() ? "device" : "server")) return;
    if (state.live) { showToast(t("Stop the live session first."), true); return; }
    if (state.activeJobs.voice || state.deviceRun) { showToast(t("Wait for the run to finish, or cancel it"), true); return; }
    state.runner = runner;
    try { localStorage.setItem(RUNNER_KEY, runner); } catch { /* storage may be disabled */ }
    renderRunner();
  }

  function renderRunner() {
    const usable = deviceModels();
    const device = onDevice();
    $$("[data-runner]").forEach((button) => {
      const active = button.dataset.runner === (device ? "device" : "server");
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-checked", active ? "true" : "false");
      if (button.dataset.runner === "device") button.disabled = !usable.length;
    });
    const hint = $("#voice-runner-hint");
    if (state.device?.reason) { hint.removeAttribute("data-i18n"); hint.textContent = state.device.reason; }
    else if (!usable.length) setKey(hint, state.device ? "None of these models has a device build." : "Checking whether this browser can run models…");
    else if (!device) clearText(hint);
    else if (state.device.threads > 1) setKey(hint, "In this browser with WebAssembly, {n} threads; the audio is not uploaded.", { n: state.device.threads });
    else setKey(hint, state.device.isolated ? "In this browser with WebAssembly, one thread; the audio is not uploaded." : "In this browser with WebAssembly, one thread: threads need localhost or HTTPS. The audio is not uploaded.");
    hint.hidden = !hint.textContent;
    labelVoiceModels();
    const select = $("#voice-model");
    if (device && !usable.some((model) => model.id === select.value)) {
      select.value = (usable.find((model) => modelIsDefault(model) && model.task === "voice_isolation") || usable.find(modelIsDefault) || usable[0]).id;
      populateVoiceVariants();
    }
    const modelHint = $("#voice-model-hint");
    if (device && usable.length < state.voiceModels.length) setKey(modelHint, "{n} of {total} models have a device build; the others are marked server only.", { n: usable.length, total: state.voiceModels.length });
    else clearText(modelHint);
    modelHint.hidden = !modelHint.textContent;
    $("#voice-stage").disabled = device;
    setKey($("#voice-stage-hint"), device ? "Server runs only." : "Another enhancement model ahead of this one; file and recording runs only.");
    $("#voice-variant-field").hidden = device;
    $("#voice-provider-field").hidden = device;
    syncExtrasRow();
    updateRunSummaries();
  }

  /* Empty, and unmarked so a language switch does not fill it again. */
  function clearText(element) {
    element.removeAttribute("data-i18n");
    element.textContent = "";
  }

  function labelVoiceModels() {
    const usable = deviceModels();
    $$("#voice-model option").forEach((option) => {
      const model = state.voiceModels.find((item) => item.id === option.value);
      if (!model) return;
      const serverOnly = onDevice() && !usable.includes(model);
      option.disabled = serverOnly;
      option.textContent = `${model.display_name}${modelIsDefault(model) ? ` · ${t("default")}` : ""}${serverOnly ? ` · ${t("server only")}` : ""}`;
    });
  }

  /* Gate records side by side ------------------------------------------- */
  /* Every model's benchmark references, fetched once; the ones with a gate
   * record (JSON) can be compared stage by stage. */
  async function loadGateRecords() {
    const entries = await Promise.all(state.models.map(async (model) => {
      try {
        const data = await api(`/api/models/${encodeURIComponent(model.id)}/benchmarks`);
        const record = data.references?.find((reference) => reference.kind === "record")?.record;
        return record ? [model.id, record] : null;
      } catch { return null; }
    }));
    state.gateRecords = Object.fromEntries(entries.filter(Boolean));
    $("#compare-records").hidden = Object.keys(state.gateRecords).length < 2;
  }

  function openGateComparison() {
    const ids = Object.keys(state.gateRecords || {});
    if (ids.length < 2) return;
    state.gateSelection = state.gateSelection?.filter((id) => ids.includes(id)).length >= 2 ? state.gateSelection : ids;
    setKey($("#dialog-title"), "Gate records side by side");
    renderGateComparison();
    const dialog = $("#model-dialog");
    dialog.classList.add("is-wide");
    if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
  }

  function renderGateComparison() {
    const ids = Object.keys(state.gateRecords);
    const chosen = state.gateSelection;
    const modelName = (id) => shortModelName(state.models.find((model) => model.id === id) || { id });
    const stages = [];
    chosen.forEach((id) => (state.gateRecords[id].stages || []).forEach((stage) => { if (!stages.includes(stage.name)) stages.push(stage.name); }));
    const find = (id, name) => (state.gateRecords[id].stages || []).find((stage) => stage.name === name);
    let group = "";
    const rows = stages.map((name) => {
      const [set] = name.includes(".") ? name.split(".") : ["general"];
      const cells = chosen.map((id) => find(id, name));
      const direction = cells.find(Boolean)?.direction || "";
      const values = cells.map((stage) => Number(stage?.value)).filter(Number.isFinite);
      const best = values.length > 1 ? (direction === "lower_is_better" ? Math.min(...values) : Math.max(...values)) : null;
      const baselines = [...new Set(cells.filter(Boolean).map((stage) => stage.baseline).filter((value) => value != null).map((value) => Number(value).toFixed(4)))];
      const heading = set !== group ? `<tr class="gate-group"><td colspan="${chosen.length + 2}">${escapeHtml(set.replace(/_/g, " "))}</td></tr>` : "";
      group = set;
      const modelCells = cells.map((stage) => {
        if (!stage) return `<td class="gate-missing">${escapeHtml(t("not measured"))}</td>`;
        const difference = stage.difference;
        const change = difference && Number.isFinite(Number(difference.point))
          ? `<small>${Number(difference.point) >= 0 ? "+" : ""}${Number(difference.point).toFixed(4)}${Number.isFinite(Number(difference.ci_low)) ? ` [${Number(difference.ci_low).toFixed(4)}, ${Number(difference.ci_high).toFixed(4)}]` : ""}</small>`
          : "";
        const tone = stage.verdict === "pass" ? "succeeded" : stage.verdict === "fail" ? "failed" : "queued";
        return `<td class="${best != null && Number(stage.value) === best ? "is-best" : ""}">${measurementValue(stage.value, 4)}${change}<span class="job-status job-status-${tone}">${escapeHtml(stage.verdict ? t(stage.verdict) : "—")}</span></td>`;
      }).join("");
      return `${heading}<tr><td><strong>${escapeHtml(name.includes(".") ? name.split(".").slice(1).join(".") : name)}</strong><small>${escapeHtml([cells.find(Boolean)?.role, direction.replace(/_/g, " ")].filter(Boolean).join(" · "))}</small></td><td>${baselines.length === 1 ? baselines[0] : baselines.length ? escapeHtml(t("differs")) : "—"}</td>${modelCells}</tr>`;
    }).join("");
    const pickers = ids.map((id) => `<label class="gate-pick"><input type="checkbox" value="${escapeHtml(id)}"${chosen.includes(id) ? " checked" : ""}><span>${escapeHtml(modelName(id))}</span><small>${escapeHtml(state.gateRecords[id].tag || "")} · ${escapeHtml(t("verdict"))} ${escapeHtml(state.gateRecords[id].verdict ? t(state.gateRecords[id].verdict) : "—")}</small></label>`).join("");
    $("#dialog-body").innerHTML = `
      <div class="dialog-section"><p class="dialog-copy">${escapeHtml(t("Each model's released gate record, stage by stage: the value, its change from the record's baseline with the 95% interval, and the stage's verdict. The best value of a row is green. Records made at different times can use different sets or baselines; a “differs” baseline means the rows are not directly comparable."))}</p><div class="gate-picks">${pickers}</div></div>
      <div class="dialog-section"><div class="measurement-table-wrap"><table class="table measurement-table gate-table"><thead><tr><th>${escapeHtml(t("Stage"))}</th><th class="num">${escapeHtml(t("Baseline"))}</th>${chosen.map((id) => `<th>${escapeHtml(modelName(id))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
    $$(".gate-pick input", $("#dialog-body")).forEach((input) => input.addEventListener("change", () => {
      const picked = $$(".gate-pick input:checked", $("#dialog-body")).map((item) => item.value);
      if (picked.length < 2) { input.checked = true; showToast(t("Keep at least two models to compare."), true); return; }
      state.gateSelection = picked;
      renderGateComparison();
    }));
  }

  function wireEvents() {
    state.audioPanels = {
      voiceInput: previewDeck($("#voice-input-audio")),
      svEnrollment: previewDeck($("#sv-enrollment-audio"), { compact: true }),
      svTest: previewDeck($("#sv-test-audio"), { compact: true }),
    };
    state.voiceDeck = new window.PureSoundCompareDeck($("#voice-deck"), { emptyText: "Run to compare the input and the output." });
    state.measureDeck = new window.PureSoundCompareDeck($("#measurement-deck"), { emptyText: "Run a comparison to listen to every output." });
    state.voiceTranscribe = new window.PureSoundTranscribe($("#voice-transcribe"), state.voiceDeck);
    state.measureTranscribe = new window.PureSoundTranscribe($("#measurement-transcribe"), state.measureDeck);
    $$("[data-run]").forEach((button) => button.addEventListener("click", () => runTask(button.dataset.run)));
    $("#model-search").addEventListener("input", renderModels);
    $("#task-filter").addEventListener("change", renderModels);
    $("#lifecycle-filter").addEventListener("change", renderModels);
    $("#refresh-button").addEventListener("click", loadCatalog);
    window.addEventListener("puresound:audio-unavailable", () => showToast(t("The browser could not start audio playback: check that an output device is connected and allowed for this page."), true));
    $("#dialog-close").addEventListener("click", () => $("#model-dialog").close?.());
    $("#model-dialog").addEventListener("close", () => $("#model-dialog").classList.remove("is-wide"));
    $("#compare-records").addEventListener("click", openGateComparison);
    $("#voice-model").addEventListener("change", () => { populateVoiceVariants(); updateRunSummaries(); });
    $("#voice-dry-blend").addEventListener("input", (event) => { $("#voice-dry-output").textContent = Number(event.target.value).toFixed(2); updateRunSummaries(); });
    $("#voice-onset-guard").addEventListener("change", () => { syncOnsetGuardKnobs("voice"); updateRunSummaries(); });
    $("#voice-provider").addEventListener("change", updateRunSummaries);
    $("#measure-onset-guard").addEventListener("change", syncMeasureGuardKnobs);
    $("#sv-model").addEventListener("change", applySvDefaults);
    $("#sv-provider").addEventListener("change", updateRunSummaries);
    $("#sv-threshold").addEventListener("input", (event) => { $("#sv-threshold-output").textContent = Number(event.target.value).toFixed(2); updateRunSummaries(); });
    $("#voice-cancel").addEventListener("click", () => (state.deviceRun ? state.deviceRun.abort() : cancelInferenceJob("voice")));
    $$("[data-runner]").forEach((button) => button.addEventListener("click", () => setRunner(button.dataset.runner)));
    $("#sv-cancel").addEventListener("click", () => cancelInferenceJob("sv"));
    $("#validate-button").addEventListener("click", runValidation);
    $("#measure-cancel").addEventListener("click", () => cancelInferenceJob("measure"));
    $("#measure-export-json").addEventListener("click", () => exportMeasurement("json"));
    $("#measure-export-csv").addEventListener("click", () => exportMeasurement("csv"));
    state.measurementEmptyHtml = $("#measurement-summary").innerHTML;
    $$("[data-clear]").forEach((button) => button.addEventListener("click", () => {
      if (button.dataset.clear === "voice") clearVoiceWorkspace();
      else if (button.dataset.clear === "sv") clearSvWorkspace();
      else if (button.dataset.clear === "measure") clearMeasurementReport();
    }));
    setMonitor("off");
    document.addEventListener("keydown", handleShortcut);
    $("#history-refresh").addEventListener("click", refreshJobHistory);
    $$("[data-history-filter]").forEach((button) => button.addEventListener("click", () => {
      state.historyFilter = button.dataset.historyFilter;
      $$("[data-history-filter]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderJobHistory(state.jobs);
    }));
    wireMeasurementFile("#measure-audio", "audio");
    wireMeasurementFile("#measure-reference", "reference");
    const readyNote = (selector, key, vars) => { const note = $(selector); setKey(note, key, vars); note.className = "form-note"; };
    const svReady = () => readyNote("#sv-form-note", selectedFile("#sv-enrollment") && selectedFile("#sv-test") ? "Both recordings ready. Verify to compare them." : "Select the other recording to compare.");
    wireFileInput("#voice-audio", { preview: $("#voice-input-preview"), name: $("#voice-file-name"), meta: $("#voice-file-meta"), panel: state.audioPanels.voiceInput }, (file) => {
      // A new file makes the comparison below stale.
      if (state.voiceLast?.fileKey !== fileIdentity(file)) { state.voiceDeck.stop(); $("#voice-result").hidden = true; }
      readyNote("#voice-form-note", "{name} is ready. Run to process it.", { name: file.name });
    });
    wireFileInput("#sv-enrollment", { preview: $("#sv-enrollment-preview"), name: $("#sv-enrollment-name"), meta: $("#sv-enrollment-meta"), panel: state.audioPanels.svEnrollment }, svReady);
    wireFileInput("#sv-test", { preview: $("#sv-test-preview"), name: $("#sv-test-name"), meta: $("#sv-test-meta"), panel: state.audioPanels.svTest }, svReady);
    mountRecorder($("#voice-recorder"), "#voice-audio");
    $$("[data-recorder-for]").forEach((host) => mountRecorder(host, `#${host.dataset.recorderFor}`, { compact: true }));
    $$("[data-input-mode]").forEach((button) => button.addEventListener("click", () => {
      if (state.live && button.dataset.inputMode !== "live") { showToast(t("Stop the live session first."), true); return; }
      setInputMode(button.dataset.inputMode);
    }));
    $("#live-toggle").addEventListener("click", toggleLive);
    document.addEventListener("change", (event) => { if (event.target.matches?.("[data-mic-select]")) chooseMicrophone(event.target.value); });
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshMicrophones);
    refreshMicrophones();
    $$("[data-monitor]").forEach((button) => button.addEventListener("click", () => setMonitor(button.dataset.monitor)));
    $("#measure-use-samples").addEventListener("click", async () => {
      const clips = (state.samples?.clips || []).filter((clip) => clip.task === "enhancement");
      const files = (await Promise.all(clips.map((clip) => sampleFile(clip.id)))).filter(Boolean);
      if (files.length) setMeasurementFiles("audio", files);
    });
    window.addEventListener("puresound:lang", relabel);
    shell().onShow("history", refreshJobHistory);
    shell().onShow("world", () => window.PureSoundWorld?.show());
    shell().onShow("pipeline", () => window.PureSoundPipeline?.show());
    refreshJobHistory();
    loadSamples();
    loadDeviceStatus();
  }

  /* A language switch: markup marked with data-i18n is redone by i18n.js;
   * what this file composed from values is rendered again here, without
   * reloading any audio. */
  function relabel() {
    if (state.health) { updateRuntimeStatus(state.health); updateProviderAvailability(state.health); }
    renderModels();
    $$("#sv-model option").forEach((option) => {
      const model = state.svModels.find((item) => item.id === option.value);
      if (model) option.textContent = `${model.display_name}${modelIsDefault(model) ? ` · ${t("default")}` : ""}`;
    });
    const variant = $("#voice-variant").value;
    const stage = $("#voice-stage").value;
    const model = selectedModel("#voice-model");
    if (model) {
      $$("#voice-variant option").forEach((option) => { option.textContent = `${option.value}${option.value === model.default_variant ? ` · ${t("default")}` : ""}`; });
      $("#voice-channels").textContent = model.channels === 1 ? t("Mono") : t("{n} channels", { n: model.channels || "—" });
      $("#voice-delay").textContent = t(model.capabilities?.realtime_streaming ? "Removed on display" : "None");
      populateStageSelect();
      $("#voice-variant").value = variant;
      $("#voice-stage").value = stage;
    }
    updateRunSummaries();
    // Why this browser cannot run models came from device.js in the old language.
    if (state.device && !state.device.ready) loadDeviceStatus();
    else renderRunner();
    renderMeasurementModels();
    renderSampleRows();
    renderJobHistory(state.jobs);
    Object.entries(state.dropzoneFiles).forEach(([inputId, name]) => {
      const dropzone = $(inputId)?.closest(".dropzone");
      const title = dropzone?.querySelector("[data-dropzone-title]");
      const hint = dropzone?.querySelector("[data-dropzone-hint]");
      if (title) title.textContent = t("Replace audio");
      if (hint) hint.textContent = t("{name} · drop or click to choose another file", { name });
    });
    if (state.voiceView) renderVoiceText(state.voiceView.result, state.voiceView.model);
    if (state.svView) renderSvResult(state.svView.result, state.svView.threshold);
    if (state.measurementReport) renderMeasurementReport(state.measurementReport, { withDeck: false });
    const dialog = $("#model-dialog");
    if (dialog.open && dialog.classList.contains("is-wide") && state.gateSelection) renderGateComparison();
    else if (dialog.open && state.dialogModel) {
      const model = state.models.find((item) => item.id === state.dialogModel);
      if (model) { renderModelDialog(model); loadModelBenchmarks(model.id); }
    }
    if (state.validation) renderValidation(state.validation);
  }

  /* What the pipeline screen (pipeline.js) borrows from this one. */
  window.PureSoundApp = { api, uploadDescriptor, showToast, t };

  wireEvents();
  loadCatalog();
})();
