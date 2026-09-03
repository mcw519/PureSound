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
    toastTimer: null,
  };

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

  function showToast(message, isError = false) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.classList.toggle("is-error", isError);
    toast.classList.add("is-visible");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 4200);
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
      throw new Error(payload?.error?.message || `Request failed (${response.status})`);
    }
    return payload;
  }

  function modelIsDefault(model) { return model.roles?.includes("default"); }
  function taskLabel(task) { return task === "speaker_embedding" ? "Speaker verification" : "Voice isolation"; }
  function lifecycleClass(value) { return `badge-${String(value || "").toLowerCase()}`; }

  function renderModels() {
    const search = $("#model-search").value.trim().toLowerCase();
    const task = $("#task-filter").value;
    const lifecycle = $("#lifecycle-filter").value;
    const filtered = state.models.filter((model) => {
      const haystack = [model.id, model.display_name, model.description, model.task].join(" ").toLowerCase();
      return (!search || haystack.includes(search)) && (task === "all" || model.task === task) && (lifecycle === "all" || model.lifecycle === lifecycle);
    });
    $("#model-count").textContent = `${filtered.length} of ${state.models.length} models`;
    const grid = $("#model-grid");
    if (!filtered.length) {
      grid.innerHTML = '<div class="empty-state">No models match these filters.</div>';
      return;
    }
    grid.innerHTML = filtered.map((model) => {
      const artifact = model.artifacts?.find((item) => item.variant === model.default_variant) || model.artifacts?.[0];
      const symbolClass = model.task === "speaker_embedding" ? " sv" : "";
      const badges = [
        `<span class="badge badge-task${model.task === "speaker_embedding" ? " sv" : ""}">${escapeHtml(taskLabel(model.task))}</span>`,
        ...(model.roles || []).filter((role) => ["default", "candidate", "reference", "diagnostic"].includes(role)).map((role) => `<span class="badge ${lifecycleClass(model.lifecycle)}">${escapeHtml(role)}</span>`),
      ].join("");
      return `<article class="model-card${modelIsDefault(model) ? " is-default" : ""}">
        <div class="model-card-top"><div class="model-symbol${symbolClass}">${model.task === "speaker_embedding" ? "◌" : "∿"}</div><span class="badge">${model.runnable ? "Ready" : "Reserved"}</span></div>
        <div style="margin-top:15px"><h3>${escapeHtml(model.display_name)}</h3><p class="model-card-subtitle">${escapeHtml(model.id)}</p></div>
        <div class="model-badges">${badges}</div>
        <p class="model-description">${escapeHtml(model.description || "Catalog-backed ONNX inference model.")}</p>
        <div class="model-footer"><div class="model-meta"><span>Artifact / sample rate</span><strong>${escapeHtml(artifact?.filename || "—")} · ${escapeHtml(model.sample_rate ? `${model.sample_rate / 1000} kHz` : "—")}</strong></div><button class="card-link" data-inspect="${escapeHtml(model.id)}" type="button">Inspect →</button></div>
      </article>`;
    }).join("");
    $$('[data-inspect]', grid).forEach((button) => button.addEventListener("click", () => openModelDialog(button.dataset.inspect)));
  }

  function populateSelect(select, models, { includeVariants = false } = {}) {
    if (!select) return;
    select.innerHTML = models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.display_name)}${modelIsDefault(model) ? " · default" : ""}</option>`).join("");
    if (models.length) select.value = models.find(modelIsDefault)?.id || models[0].id;
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
    select.innerHTML = (model.artifacts || []).map((artifact) => `<option value="${escapeHtml(artifact.variant)}">${escapeHtml(artifact.variant)}${artifact.variant === model.default_variant ? " · default" : ""}</option>`).join("");
    select.value = model.default_variant || model.artifacts?.[0]?.variant || "default";
    const dry = model.recommended_inference?.dry_blend;
    if (typeof dry === "number") {
      $("#voice-dry-blend").value = dry;
      $("#voice-dry-output").value = dry.toFixed(2);
      $("#voice-dry-output").textContent = dry.toFixed(2);
    }
    $("#voice-sample-rate").textContent = model.sample_rate ? `${model.sample_rate / 1000} kHz` : "—";
    $("#voice-channels").textContent = model.channels === 1 ? "Mono" : `${model.channels || "—"} channels`;
    $("#voice-delay").textContent = model.postprocessing?.streaming_delay_alignment ? "Aligned" : "Manifest";
    const hasHeads = Boolean(model.capabilities?.auxiliary_heads);
    $("#voice-collect-extras").disabled = !hasHeads;
    if (!hasHeads) $("#voice-collect-extras").checked = false;
  }

  function updateRuntimeStatus(health) {
    $("#runtime-status").textContent = health.status === "ok" ? "Ready" : "Unavailable";
    $("#runtime-meta").textContent = `${health.runnable_models ?? 0} runnable · schema ${health.schema_version ?? "—"}`;
    $("#stat-models").textContent = health.models ?? "—";
  }

  async function loadCatalog() {
    try {
      const [health, catalog] = await Promise.all([api("/api/health"), api("/api/models?include_empty=1")]);
      state.models = catalog.models || [];
      state.voiceModels = state.models.filter((model) => model.task === "voice_isolation" && model.runnable);
      state.svModels = state.models.filter((model) => model.task === "speaker_embedding" && model.runnable);
      updateRuntimeStatus(health);
      const artifacts = state.models.reduce((total, model) => total + (model.artifacts?.length || 0), 0);
      $("#stat-artifacts").textContent = artifacts;
      $("#stat-default-voice").textContent = state.voiceModels.find(modelIsDefault)?.display_name?.replace("Voice Isolation ", "") || "—";
      renderModels();
      populateSelect($("#voice-model"), state.voiceModels, { includeVariants: true });
      populateSelect($("#sv-model"), state.svModels);
      populateVoiceVariants();
      showToast(`Loaded ${state.models.length} catalog models.`);
    } catch (error) {
      $("#runtime-status").textContent = "Offline";
      $("#runtime-meta").textContent = "Start puresound web";
      $("#model-grid").innerHTML = `<div class="empty-state">Unable to load the Model Zoo.<br><small>${escapeHtml(error.message)}</small></div>`;
      showToast(error.message, true);
    }
  }

  function openModelDialog(modelId) {
    const model = state.models.find((item) => item.id === modelId);
    if (!model) return;
    $("#dialog-title").textContent = model.display_name;
    const artifacts = (model.artifacts || []).map((artifact) => `<div class="artifact-row"><strong>${escapeHtml(artifact.variant)}</strong> · ${escapeHtml(artifact.filename)} · ${artifact.available ? "available" : "missing"}<br><span>processor ${escapeHtml(artifact.processor)} · sha ${escapeHtml((artifact.sha256 || "").slice(0, 12))}…</span></div>`).join("");
    $("#dialog-body").innerHTML = `<div class="dialog-section"><div class="dialog-section-title">Contract</div><div class="detail-grid"><div class="detail-item"><span>Task</span><strong>${escapeHtml(taskLabel(model.task))}</strong></div><div class="detail-item"><span>Lifecycle</span><strong>${escapeHtml(model.lifecycle)}</strong></div><div class="detail-item"><span>Inputs</span><strong>${escapeHtml((model.inputs || []).join(", ") || "—")}</strong></div><div class="detail-item"><span>Outputs</span><strong>${escapeHtml((model.outputs || []).join(", ") || "—")}</strong></div><div class="detail-item"><span>Audio</span><strong>${escapeHtml(model.sample_rate ? `${model.sample_rate} Hz · ${model.channels || 1} ch` : "—")}</strong></div><div class="detail-item"><span>Roles</span><strong>${escapeHtml((model.roles || []).join(", ") || "—")}</strong></div></div></div><div class="dialog-section"><div class="dialog-section-title">Artifacts</div><div class="artifact-list">${artifacts || "<span>Reserved for a future ONNX artifact.</span>"}</div></div><div class="dialog-section"><div class="dialog-section-title">Pre/post-processing</div><div class="detail-grid"><div class="detail-item"><span>Preprocessing</span><strong>${escapeHtml(JSON.stringify(model.preprocessing || {}))}</strong></div><div class="detail-item"><span>Recommended</span><strong>${escapeHtml(JSON.stringify(model.recommended_inference || {}))}</strong></div></div></div>`;
    const dialog = $("#model-dialog");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  function setBusy(button, busy) {
    button.classList.toggle("is-loading", busy);
    button.disabled = busy;
  }

  function readDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve({ filename: file.name, data: reader.result });
      reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
      reader.readAsDataURL(file);
    });
  }

  function selectedFile(inputId) {
    return $(inputId).files?.[0] || state.files[inputId] || null;
  }

  function formatSeconds(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return number < 1 ? `${Math.round(number * 1000)} ms` : `${number.toFixed(2)} s`;
  }

  function providerLabel(value) {
    const provider = String(value || "");
    if (provider.includes("CUDA")) return "CUDA";
    if (provider.includes("CPU")) return "CPU";
    return provider || "—";
  }

  async function previewAudio(file, elements) {
    const { preview, name, meta, panel } = elements;
    preview.hidden = false;
    name.textContent = file.name;
    meta.textContent = `${(file.size / 1024).toFixed(0)} KB · decoding…`;
    try {
      const buffer = await panel.loadFile(file);
      meta.textContent = `${buffer.duration.toFixed(2)} s · ${(buffer.sampleRate / 1000).toFixed(1)} kHz`;
    } catch (error) {
      meta.textContent = `${(file.size / 1024).toFixed(0)} KB · preview unavailable`;
      showToast(`Browser preview failed: ${error.message}`, true);
    }
  }

  function wireFileInput(inputId, elements) {
    const input = $(inputId);
    const dropzone = input.closest(".dropzone");
    input.addEventListener("change", () => { if (input.files?.[0]) { state.files[inputId] = input.files[0]; previewAudio(input.files[0], elements); } });
    ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("is-dragging"); }));
    ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("is-dragging"); }));
    dropzone.addEventListener("drop", (event) => { const file = event.dataTransfer.files?.[0]; if (file) { state.files[inputId] = file; try { input.files = event.dataTransfer.files; } catch { /* keep the in-memory file for browsers that forbid assignment */ } previewAudio(file, elements); } });
  }

  function renderMetrics(target, metrics) {
    target.innerHTML = metrics.map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  }

  async function runVoiceInference() {
    const file = selectedFile("#voice-audio");
    const note = $("#voice-form-note");
    if (!file) { note.textContent = "Select an audio file first."; note.className = "form-note is-error"; return; }
    const button = $("#voice-run");
    setBusy(button, true); note.textContent = "Preparing audio and running the streaming processor…"; note.className = "form-note";
    try {
      const model = selectedModel("#voice-model");
      const input = await readDataUrl(file);
      const parameters = { dry_blend: Number($("#voice-dry-blend").value) };
      if ($("#voice-collect-extras").checked) parameters.collect_extras = true;
      const result = await api("/api/infer", { method: "POST", body: JSON.stringify({ model_id: model.id, variant: $("#voice-variant").value, provider: $("#voice-provider").value, inputs: { audio: input }, parameters }) });
      const outputUrl = result.output_urls?.audio;
      if (!outputUrl) throw new Error("The runtime did not return an audio output.");
      $("#voice-result").hidden = false;
      $("#voice-download").href = outputUrl;
      $("#voice-result-meta").textContent = `${result.sample_rate ? `${result.sample_rate / 1000} kHz` : "—"} · ${formatSeconds(result.metadata?.duration_seconds)}`;
      renderMetrics($("#voice-metrics"), [["RTF", result.rtf == null ? "—" : Number(result.rtf).toFixed(3)], ["Latency", result.metadata?.latency_ms == null ? "—" : `${Number(result.metadata.latency_ms).toFixed(1)} ms`], ["Output", `${result.outputs?.audio?.shape?.[0] || "—"} samples`], ["Provider", providerLabel(result.provider)]]);
      try { await state.audioPanels.voiceOutput.loadUrl(outputUrl); } catch (error) { showToast(`Output preview failed: ${error.message}`, true); }
      note.textContent = "Inference complete. Streaming delay alignment was kept by the processor."; note.className = "form-note is-success";
      showToast(`Finished ${model.display_name}.`);
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; showToast(error.message, true); }
    finally { setBusy(button, false); }
  }

  async function runSpeakerVerification() {
    const enrollment = selectedFile("#sv-enrollment");
    const test = selectedFile("#sv-test");
    const note = $("#sv-form-note");
    if (!enrollment || !test) { note.textContent = "Select both enrollment and test audio."; note.className = "form-note is-error"; return; }
    const button = $("#sv-run");
    setBusy(button, true); note.textContent = "Extracting both embeddings…"; note.className = "form-note";
    try {
      const model = selectedModel("#sv-model");
      const inputs = { enrollment: await readDataUrl(enrollment), test: await readDataUrl(test) };
      const threshold = Number($("#sv-threshold").value);
      const result = await api("/api/infer", { method: "POST", body: JSON.stringify({ model_id: model.id, provider: $("#sv-provider").value, inputs, parameters: { threshold } }) });
      const score = Number(result.scores?.cosine_similarity || 0);
      const verdict = Boolean(result.scores?.verdict);
      $("#sv-result").hidden = false;
      $("#sv-score").textContent = score.toFixed(2);
      $("#sv-score-gauge").style.setProperty("--score", `${Math.max(0, Math.min(1, score)) * 100}%`);
      $("#sv-verdict").textContent = verdict ? "MATCH" : "NO MATCH";
      $("#sv-verdict").style.color = verdict ? "var(--green)" : "var(--danger)";
      $("#sv-threshold-result").textContent = threshold.toFixed(2);
      $("#sv-verdict-note").textContent = verdict ? "Similarity is above the selected threshold." : "Similarity is below the selected threshold.";
      $("#sv-result-time").textContent = formatSeconds(result.elapsed_seconds);
      renderMetrics($("#sv-metrics"), [["Similarity", score.toFixed(3)], ["Threshold", threshold.toFixed(2)], ["Embedding", `${result.metadata?.embedding_dim || "—"} d`], ["Provider", providerLabel(result.provider)]]);
      note.textContent = "Verification complete."; note.className = "form-note is-success";
      showToast(verdict ? "Speaker match." : "No speaker match.");
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; showToast(error.message, true); }
    finally { setBusy(button, false); }
  }

  async function runValidation() {
    const button = $("#validate-button");
    setBusy(button, true);
    $("#validation-status").className = "validation-status";
    $("#validation-status").innerHTML = '<span class="status-icon">…</span><div><strong>Validating catalog…</strong><span>Checking paths, sidecars, hashes and ONNX graph contracts.</span></div>';
    try {
      const report = await api("/api/validate");
      const status = $("#validation-status");
      status.classList.add(report.ok ? "is-ok" : "is-error");
      status.innerHTML = `<span class="status-icon">${report.ok ? "✓" : "!"}</span><div><strong>${report.ok ? "Catalog is valid" : "Validation found issues"}</strong><span>${report.ok ? "Every registered artifact passed the requested checks." : `${report.errors.length} issue(s) need attention.`}</span></div>`;
      $("#validation-summary").innerHTML = `<div class="validation-item"><span>Logical models</span><strong>${report.models}</strong></div><div class="validation-item"><span>ONNX artifacts</span><strong>${report.artifacts}</strong></div><div class="validation-item"><span>Checks</span><strong>${report.ok ? "Passed" : "Review"}</strong></div>`;
      const errors = $("#validation-errors");
      errors.hidden = report.ok;
      errors.textContent = (report.errors || []).join("\n");
      showToast(report.ok ? "Model Zoo validation passed." : "Model Zoo validation needs attention.", !report.ok);
    } catch (error) { showToast(error.message, true); }
    finally { setBusy(button, false); }
  }

  function switchScreen(screen) {
    $$('[data-screen-panel]').forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.screenPanel === screen));
    $$(".nav-link").forEach((link) => link.classList.toggle("is-active", link.dataset.screen === screen));
    const labels = { zoo: "Model Zoo / Overview", playground: "Playground / Inference", measurements: "Measurements / Validation" };
    $("#topbar-context").textContent = labels[screen] || "PureSound";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function wireEvents() {
    state.audioPanels = {
      voiceInput: new window.PureSoundAudioPanel($("#voice-input-audio")),
      voiceOutput: new window.PureSoundAudioPanel($("#voice-output-audio")),
      svEnrollment: new window.PureSoundAudioPanel($("#sv-enrollment-audio")),
      svTest: new window.PureSoundAudioPanel($("#sv-test-audio")),
    };
    $$(".nav-link").forEach((link) => link.addEventListener("click", () => switchScreen(link.dataset.screen)));
    $$('[data-screen-target]').forEach((button) => button.addEventListener("click", () => switchScreen(button.dataset.screenTarget)));
    $("#model-search").addEventListener("input", renderModels);
    $("#task-filter").addEventListener("change", renderModels);
    $("#lifecycle-filter").addEventListener("change", renderModels);
    $("#refresh-button").addEventListener("click", loadCatalog);
    $("#dialog-close").addEventListener("click", () => $("#model-dialog").close?.());
    $("#voice-model").addEventListener("change", populateVoiceVariants);
    $("#voice-dry-blend").addEventListener("input", (event) => { $("#voice-dry-output").textContent = Number(event.target.value).toFixed(2); });
    $("#sv-threshold").addEventListener("input", (event) => { $("#sv-threshold-output").textContent = Number(event.target.value).toFixed(2); });
    $("#voice-run").addEventListener("click", runVoiceInference);
    $("#sv-run").addEventListener("click", runSpeakerVerification);
    $("#validate-button").addEventListener("click", runValidation);
    $$(".workspace-tab").forEach((tab) => tab.addEventListener("click", () => {
      const workspace = tab.dataset.workspace;
      $$(".workspace-tab").forEach((item) => { const active = item === tab; item.classList.toggle("is-active", active); item.setAttribute("aria-selected", active ? "true" : "false"); });
      $$("[data-workspace-panel]").forEach((panel) => { panel.hidden = panel.dataset.workspacePanel !== workspace; });
    }));
    wireFileInput("#voice-audio", { preview: $("#voice-input-preview"), name: $("#voice-file-name"), meta: $("#voice-file-meta"), panel: state.audioPanels.voiceInput });
    wireFileInput("#sv-enrollment", { preview: $("#sv-enrollment-preview"), name: $("#sv-enrollment-name"), meta: $("#sv-enrollment-meta"), panel: state.audioPanels.svEnrollment });
    wireFileInput("#sv-test", { preview: $("#sv-test-preview"), name: $("#sv-test-name"), meta: $("#sv-test-meta"), panel: state.audioPanels.svTest });
  }

  wireEvents();
  loadCatalog();
})();
