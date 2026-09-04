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
    measurementAudioPanels: [],
    jobs: [],
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
        option.textContent = available ? label : `${label} · unavailable`;
      });
    });
    const preferred = capabilities.cuda ? "CUDA" : capabilities.coreml ? "Apple CoreML" : "CPU";
    const chipLabel = $("#provider-chip-label");
    if (chipLabel) chipLabel.textContent = `Provider auto · ${preferred}`;
    const runtimeStat = $("#stat-runtime");
    if (runtimeStat) runtimeStat.textContent = preferred;
  }

  async function loadCatalog() {
    try {
      const [health, catalog] = await Promise.all([api("/api/health"), api("/api/models?include_empty=1")]);
      state.models = catalog.models || [];
      state.voiceModels = state.models.filter((model) => model.task === "voice_isolation" && model.runnable);
      state.svModels = state.models.filter((model) => model.task === "speaker_embedding" && model.runnable);
      updateRuntimeStatus(health);
      updateProviderAvailability(health);
      const artifacts = state.models.reduce((total, model) => total + (model.artifacts?.length || 0), 0);
      $("#stat-artifacts").textContent = artifacts;
      $("#stat-default-voice").textContent = state.voiceModels.find(modelIsDefault)?.display_name?.replace("Voice Isolation ", "") || "—";
      renderModels();
      populateSelect($("#voice-model"), state.voiceModels, { includeVariants: true });
      populateSelect($("#sv-model"), state.svModels);
      populateVoiceVariants();
      renderMeasurementModels();
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
    if (provider.includes("CoreML")) return "Apple CoreML";
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

  function updateJobProgress(key, job) {
    const progress = $(`#${key}-job-progress`);
    if (!progress) return;
    const amount = Math.round(Math.max(0, Math.min(1, Number(job.progress) || 0)) * 100);
    progress.querySelector(".job-progress-bar span").style.width = `${amount}%`;
    const phase = String(job.phase || job.status || "working")
      .replace(/:/g, " · ")
      .replace(/_/g, " ");
    progress.querySelector("[data-job-phase]").textContent = phase;
    progress.querySelector("[data-job-percent]").textContent = `${amount}%`;
    progress.hidden = false;
  }

  async function runBackgroundJob(payload, key) {
    const runButton = $(`#${key}-run`);
    const cancelButton = $(`#${key}-cancel`);
    const progress = $(`#${key}-job-progress`);
    if (state.activeJobs[key]) throw new Error("An inference is already running.");
    setBusy(runButton, true);
    cancelButton.hidden = false;
    progress.hidden = false;
    try {
      const initial = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
      state.activeJobs[key] = initial.job_id;
      updateJobProgress(key, initial);
      while (true) {
        const job = await api(`/api/jobs/${encodeURIComponent(initial.job_id)}`);
        updateJobProgress(key, job);
        if (job.status === "succeeded") return job.result;
        if (job.status === "failed") throw new Error(job.error || "Job failed.");
        if (job.status === "cancelled") throw new Error(job.kind === "measurement" ? "Measurement cancelled." : "Inference cancelled.");
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
    } finally {
      delete state.activeJobs[key];
      setBusy(runButton, false);
      cancelButton.hidden = true;
      window.setTimeout(() => { if (!state.activeJobs[key]) progress.hidden = true; }, 900);
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
    } finally {
      button.disabled = false;
    }
  }

  function renderJobHistory(jobs) {
    const target = $("#job-history");
    if (!target) return;
    if (!jobs.length) {
      target.innerHTML = '<span class="measure-empty">No asynchronous runs yet.</span>';
      return;
    }
    target.innerHTML = jobs.map((job) => {
      const result = job.result || {};
      const score = result.scores?.cosine_similarity;
      const measurement = job.kind === "measurement";
      const detail = job.status === "succeeded"
        ? (measurement
          ? `${result.models?.length || 0} models · ${job.elapsed_seconds == null ? "—" : formatSeconds(job.elapsed_seconds)}`
          : `${result.rtf == null ? "—" : `RTF ${Number(result.rtf).toFixed(3)}`} · ${job.elapsed_seconds == null ? "—" : formatSeconds(job.elapsed_seconds)}`)
        : (job.error || job.phase || "—");
      const output = result.output_urls?.audio ? `<a href="${escapeHtml(result.output_urls.audio)}" target="_blank" rel="noreferrer">Output ↗</a>` : "";
      const title = measurement ? "Measurement comparison" : job.model_id;
      return `<article class="job-history-item"><div class="job-history-main"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div><div class="job-history-side"><span class="job-status job-status-${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>${score == null ? "" : `<span>${Number(score).toFixed(3)}</span>`}${output}</div></article>`;
    }).join("");
  }

  async function refreshJobHistory() {
    try {
      const response = await api("/api/jobs?limit=20");
      state.jobs = response.jobs || [];
      renderJobHistory(state.jobs);
    } catch (error) {
      showToast(`Could not load run history: ${error.message}`, true);
    }
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
      const result = await runBackgroundJob(
        {
          model_id: model.id,
          variant: $("#voice-variant").value,
          provider: $("#voice-provider").value,
          inputs: { audio: input },
          parameters,
          measurements: { include_dnsmos: true },
        },
        "voice",
      );
      const outputUrl = result.output_urls?.audio;
      if (!outputUrl) throw new Error("The runtime did not return an audio output.");
      $("#voice-result").hidden = false;
      $("#voice-download").href = outputUrl;
      $("#voice-result-meta").textContent = `${result.sample_rate ? `${result.sample_rate / 1000} kHz` : "—"} · ${formatSeconds(result.metadata?.duration_seconds)}`;
      const outputMeasurements = result.measurements?.output || {};
      const dnsMos = result.measurements?.reference_free?.dnsmos || {};
      renderMetrics($("#voice-metrics"), [
        ["RTF", result.rtf == null ? "—" : Number(result.rtf).toFixed(3)],
        ["Latency", result.metadata?.latency_ms == null ? "—" : `${Number(result.metadata.latency_ms).toFixed(1)} ms`],
        ["DNSMOS OVR", measurementValue(dnsMos.dnsmos_ovr, 2)],
        ["Output RMS", measurementValue(outputMeasurements.rms_dbfs, 1, " dBFS")],
      ]);
      const measurementNote = $("#voice-measurement-note");
      const dnsError = result.measurements?.reference_free?.dnsmos_error;
      const dnsAvailable = Number.isFinite(Number(dnsMos.dnsmos_ovr));
      if (dnsError) {
        measurementNote.textContent = `Output measured on ${providerLabel(result.provider)}. DNSMOS unavailable: ${dnsError}`;
      } else if (dnsAvailable) {
        measurementNote.textContent = `Output measured on ${providerLabel(result.provider)}. DNSMOS OVR is a reference-free quality score (1–5; higher is better).`;
      } else {
        measurementNote.textContent = `Output measured on ${providerLabel(result.provider)}. DNSMOS was not returned by the runtime.`;
      }
      measurementNote.className = `measurement-inline-note${dnsError || !dnsAvailable ? " is-warning" : ""}`;
      try { await state.audioPanels.voiceOutput.loadUrl(outputUrl); } catch (error) { showToast(`Output preview failed: ${error.message}`, true); }
      note.textContent = "Inference complete. Streaming delay alignment was kept by the processor."; note.className = "form-note is-success";
      setPlaygroundDrawer(null);
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
      const result = await runBackgroundJob({ model_id: model.id, provider: $("#sv-provider").value, inputs, parameters: { threshold } }, "sv");
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
      setPlaygroundDrawer(null);
      showToast(verdict ? "Speaker match." : "No speaker match.");
    } catch (error) { note.textContent = error.message; note.className = "form-note is-error"; showToast(error.message, true); }
    finally { setBusy(button, false); }
  }

  function renderMeasurementModels() {
    const target = $("#measure-model-list");
    if (!target) return;
    if (!state.voiceModels.length) {
      target.innerHTML = '<span class="measure-empty">No runnable Voice Isolation models are available.</span>';
      $("#measure-model-count").textContent = "0 selected";
      return;
    }
    target.innerHTML = state.voiceModels.map((model, index) => {
      const checked = modelIsDefault(model) || index === 0;
      return `<label class="measure-model-option"><input type="checkbox" value="${escapeHtml(model.id)}"${checked ? " checked" : ""}><span class="measure-check"></span><span><strong>${escapeHtml(model.display_name)}</strong><small>${escapeHtml(model.id)} · ${escapeHtml(model.lifecycle)}</small></span></label>`;
    }).join("");
    target.querySelectorAll("input").forEach((input) => input.addEventListener("change", updateMeasurementModelCount));
    updateMeasurementModelCount();
  }

  function updateMeasurementModelCount() {
    const selected = $$("#measure-model-list input:checked").length;
    $("#measure-model-count").textContent = `${selected} selected`;
  }

  function measurementValue(value, digits = 2, suffix = "") {
    return value == null || !Number.isFinite(Number(value)) ? "—" : `${Number(value).toFixed(digits)}${suffix}`;
  }

  function setMeasurementDrawer(open) {
    const drawer = $("#measurement-drawer");
    const backdrop = $("#measurement-drawer-backdrop");
    $("#measure-configure").setAttribute("aria-expanded", open ? "true" : "false");
    drawer.setAttribute("aria-hidden", open ? "false" : "true");
    document.body.classList.toggle("measurement-drawer-open", open);
    if (open) {
      backdrop.hidden = false;
      requestAnimationFrame(() => {
        drawer.classList.add("is-open");
        backdrop.classList.add("is-open");
        $("#measurement-drawer-close").focus();
      });
      return;
    }
    drawer.classList.remove("is-open");
    backdrop.classList.remove("is-open");
    window.setTimeout(() => { if (!drawer.classList.contains("is-open")) backdrop.hidden = true; }, 240);
  }

  function activePlaygroundKey() {
    return $(".workspace-tab.is-active")?.dataset.workspace || "voice";
  }

  function setPlaygroundDrawer(key) {
    const backdrop = $("#playground-drawer-backdrop");
    const openButton = $("#playground-settings-open");
    const drawer = key ? $(`#playground-drawer-${key}`) : null;
    $$(".playground-drawer").forEach((item) => {
      const open = item === drawer;
      item.classList.toggle("is-open", open);
      item.setAttribute("aria-hidden", open ? "false" : "true");
    });
    openButton.setAttribute("aria-expanded", drawer ? "true" : "false");
    openButton.setAttribute("aria-controls", `playground-drawer-${key || activePlaygroundKey()}`);
    document.body.classList.toggle("playground-drawer-open", Boolean(drawer));
    if (drawer) {
      backdrop.hidden = false;
      requestAnimationFrame(() => {
        backdrop.classList.add("is-open");
        drawer.querySelector("[data-playground-settings-close]")?.focus();
      });
      return;
    }
    backdrop.classList.remove("is-open");
    window.setTimeout(() => {
      if (!$(".playground-drawer.is-open")) backdrop.hidden = true;
    }, 240);
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
      "model_id", "display_name", "provider", "elapsed_seconds", "rtf",
      "rms_dbfs", "peak_dbfs", "clipping_ratio", "silence_ratio",
      "spectral_centroid_hz", "si_sdr_db", "snr_db", "correlation", "stoi", "pesq_wb",
      "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_error", "error",
      "latency_samples", "output_url",
    ];
    const cell = (value) => `"${String(value == null ? "" : value).replace(/"/g, '""')}"`;
    const rows = (report.models || []).map((model) => {
      const output = model.output || {};
      const quality = model.quality || {};
      return [
        model.model_id,
        model.display_name,
        model.provider,
        model.elapsed_seconds,
        model.rtf,
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
        model.reference_free?.dnsmos?.dnsmos_p808,
        model.reference_free?.dnsmos?.dnsmos_sig,
        model.reference_free?.dnsmos?.dnsmos_bak,
        model.reference_free?.dnsmos?.dnsmos_ovr,
        model.reference_free?.dnsmos_error,
        model.error,
        model.latency_samples,
        model.output_url,
      ].map(cell).join(",");
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

  function renderMeasurementReport(report) {
    state.measurementReport = report;
    $("#measurement-report-content").hidden = false;
    $("#measure-export-json").hidden = false;
    $("#measure-export-csv").hidden = false;
    const input = report.input || {};
    const reference = report.reference;
    const dnsMosErrors = (report.models || [])
      .map((model) => model.reference_free?.dnsmos_error)
      .filter(Boolean);
    const summary = reference
      ? "Reference supplied — use SI-SDR and STOI for quality, then check speed and clipping for deployment."
      : "No clean reference — DNSMOS provides a reference-free quality signal; also compare speed, level, and clipping.";
    const modelCount = (report.models || []).length;
    const failedModels = Number(report.summary?.failed || 0);
    const resultSummary = dnsMosErrors.length > 0 && dnsMosErrors.length === modelCount
      ? `${summary} DNSMOS is unavailable in this runtime.`
      : summary;
    $("#measurement-result-summary").textContent = failedModels
      ? `${failedModels} selected model${failedModels === 1 ? "" : "s"} failed. ${resultSummary}`
      : resultSummary;
    $("#measurement-summary").innerHTML = [
      ["Input RMS", measurementValue(input.rms_dbfs, 1, " dBFS")],
      ["Input peak", measurementValue(input.peak_dbfs, 1, " dBFS")],
      ["Input clipping", measurementValue(input.clipping_ratio * 100, 2, "%")],
      ["Reference", reference ? `${measurementValue(reference.duration_seconds, 2, " s")} · ${measurementValue(reference.rms_dbfs, 1, " dBFS")}` : "Not provided"],
    ].map(([label, value]) => `<div class="measurement-summary-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    $("#measurement-result-time").textContent = `${measurementValue(report.elapsed_seconds, 2, " s")} total`;
    $("#measurement-table-body").innerHTML = (report.models || []).map((model) => {
      const output = model.output || {};
      const quality = model.quality || {};
      const referenceFree = model.reference_free || {};
      const dnsMos = referenceFree.dnsmos || {};
      const dnsMosTitle = referenceFree.dnsmos_error
        ? `DNSMOS unavailable: ${referenceFree.dnsmos_error}`
        : "DNSMOS overall quality, 1–5; higher is better";
      if (model.error) return `<tr><td><strong>${escapeHtml(model.model_id)}</strong><small class="table-error">${escapeHtml(model.error)}</small></td><td colspan="7">—</td></tr>`;
      return `<tr><td><strong>${escapeHtml(model.display_name || model.model_id)}</strong><small>${escapeHtml(providerLabel(model.provider))}</small></td><td>${measurementValue(model.rtf, 3)}</td><td>${measurementValue(output.rms_dbfs, 1, " dB")}</td><td>${measurementValue(output.peak_dbfs, 1, " dB")}</td><td>${measurementValue(output.clipping_ratio * 100, 2, "%")}</td><td title="${escapeHtml(dnsMosTitle)}">${measurementValue(dnsMos.dnsmos_ovr, 2)}</td><td>${measurementValue(quality.si_sdr_db, 2, " dB")}</td><td>${measurementValue(quality.stoi, 3)}</td></tr>`;
    }).join("");
    state.measurementAudioPanels.forEach((panel) => panel.stop());
    state.measurementAudioPanels = [];
    const playable = (report.models || []).filter((model) => model.output_url && !model.error);
    $("#measurement-audio-results").innerHTML = playable.map((model, index) => `
      <section class="measurement-audio-card">
        <div class="measurement-audio-card-head"><strong>${escapeHtml(model.display_name || model.model_id)}</strong><span>${escapeHtml(providerLabel(model.provider))}</span></div>
        <div class="audio-inspector is-compact" id="measurement-audio-${index}" data-wave-color="#c8f6f9"></div>
      </section>`).join("");
    playable.forEach((model, index) => {
      const panel = new window.PureSoundAudioPanel($(`#measurement-audio-${index}`));
      state.measurementAudioPanels.push(panel);
      panel.loadUrl(model.output_url).catch((error) => showToast(error.message, true));
    });
  }

  async function runMeasurements() {
    const file = state.measurementFiles.audio;
    const reference = state.measurementFiles.reference;
    const note = $("#measure-form-note");
    const models = $$("#measure-model-list input:checked").map((input) => input.value);
    if (!file) { note.textContent = "Choose the recording to compare first."; note.className = "form-note is-error"; return; }
    if (!models.length) { note.textContent = "Select at least one model to compare."; note.className = "form-note is-error"; return; }
    const button = $("#measure-run");
    setBusy(button, true); note.textContent = "Measuring input and running selected models…"; note.className = "form-note";
    try {
      const inputs = { audio: await readDataUrl(file) };
      if (reference) inputs.reference = await readDataUrl(reference);
      const report = await runBackgroundJob(
        {
          kind: "measurement",
          inputs,
          models,
          provider: "auto",
          measurements: { include_dnsmos: true },
        },
        "measure",
      );
      renderMeasurementReport(report);
      const failed = Number(report.summary?.failed || 0);
      note.textContent = failed ? `Comparison complete with ${failed} failed model${failed === 1 ? "" : "s"}.` : "Comparison complete.";
      note.className = failed ? "form-note is-error" : "form-note is-success";
      setMeasurementDrawer(false);
      showToast(failed ? "Comparison completed with warnings." : "Audio measurements complete.", failed > 0);
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
    setPlaygroundDrawer(null);
    setMeasurementDrawer(false);
    $$('[data-screen-panel]').forEach((panel) => panel.classList.toggle("is-visible", panel.dataset.screenPanel === screen));
    $$(".nav-link").forEach((link) => link.classList.toggle("is-active", link.dataset.screen === screen));
    const labels = { zoo: "Model Zoo / Overview", playground: "Playground / Inference", measurements: "Measurements / Validation" };
    $("#topbar-context").textContent = labels[screen] || "PureSound";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setSidebarCollapsed(collapsed) {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    const button = $("#sidebar-toggle");
    button.setAttribute("aria-expanded", collapsed ? "false" : "true");
    button.setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
    button.textContent = collapsed ? "›" : "‹";
    try { localStorage.setItem("puresound.sidebar-collapsed", collapsed ? "1" : "0"); } catch { /* storage may be disabled */ }
  }

  function restoreLayout() {
    let sidebar = false;
    try {
      sidebar = localStorage.getItem("puresound.sidebar-collapsed") === "1";
    } catch { /* storage may be disabled */ }
    setSidebarCollapsed(sidebar);
  }

  function wireMeasurementFile(inputId, key) {
    const input = $(inputId);
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      if (!file) return;
      state.measurementFiles[key] = file;
      $(`[data-measure-name="${key}"]`).textContent = file.name;
      $(`[data-measure-meta="${key}"]`).textContent = `${(file.size / 1024).toFixed(0)} KB · ready for analysis`;
    });
  }

  function wireEvents() {
    state.audioPanels = {
      voiceInput: new window.PureSoundAudioPanel($("#voice-input-audio")),
      voiceOutput: new window.PureSoundAudioPanel($("#voice-output-audio")),
      svEnrollment: new window.PureSoundAudioPanel($("#sv-enrollment-audio")),
      svTest: new window.PureSoundAudioPanel($("#sv-test-audio")),
    };
    $("#sidebar-toggle").addEventListener("click", () => setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed")));
    $("#playground-settings-open").addEventListener("click", () => setPlaygroundDrawer(activePlaygroundKey()));
    $$("[data-playground-settings-close]").forEach((button) => button.addEventListener("click", () => setPlaygroundDrawer(null)));
    $("#playground-drawer-backdrop").addEventListener("click", () => setPlaygroundDrawer(null));
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
    $("#voice-cancel").addEventListener("click", () => cancelInferenceJob("voice"));
    $("#sv-run").addEventListener("click", runSpeakerVerification);
    $("#sv-cancel").addEventListener("click", () => cancelInferenceJob("sv"));
    $("#validate-button").addEventListener("click", runValidation);
    $("#measure-run").addEventListener("click", runMeasurements);
    $("#measure-cancel").addEventListener("click", () => cancelInferenceJob("measure"));
    $("#measure-export-json").addEventListener("click", () => exportMeasurement("json"));
    $("#measure-export-csv").addEventListener("click", () => exportMeasurement("csv"));
    $("#measure-configure").addEventListener("click", () => setMeasurementDrawer(true));
    $("#measurement-empty-configure").addEventListener("click", () => setMeasurementDrawer(true));
    $("#measurement-drawer-close").addEventListener("click", () => setMeasurementDrawer(false));
    $("#measurement-drawer-backdrop").addEventListener("click", () => setMeasurementDrawer(false));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && $("#measurement-drawer").classList.contains("is-open")) setMeasurementDrawer(false);
      if (event.key === "Escape" && $(".playground-drawer.is-open")) setPlaygroundDrawer(null);
    });
    $("#history-refresh").addEventListener("click", refreshJobHistory);
    wireMeasurementFile("#measure-audio", "audio");
    wireMeasurementFile("#measure-reference", "reference");
    $$(".workspace-tab").forEach((tab) => tab.addEventListener("click", () => {
      const workspace = tab.dataset.workspace;
      setPlaygroundDrawer(null);
      $$(".workspace-tab").forEach((item) => { const active = item === tab; item.classList.toggle("is-active", active); item.setAttribute("aria-selected", active ? "true" : "false"); });
      $$("[data-workspace-panel]").forEach((panel) => { panel.hidden = panel.dataset.workspacePanel !== workspace; });
      $("#playground-settings-open").setAttribute("aria-controls", `playground-drawer-${workspace}`);
    }));
    wireFileInput("#voice-audio", { preview: $("#voice-input-preview"), name: $("#voice-file-name"), meta: $("#voice-file-meta"), panel: state.audioPanels.voiceInput });
    wireFileInput("#sv-enrollment", { preview: $("#sv-enrollment-preview"), name: $("#sv-enrollment-name"), meta: $("#sv-enrollment-meta"), panel: state.audioPanels.svEnrollment });
    wireFileInput("#sv-test", { preview: $("#sv-test-preview"), name: $("#sv-test-name"), meta: $("#sv-test-meta"), panel: state.audioPanels.svTest });
    restoreLayout();
    refreshJobHistory();
  }

  wireEvents();
  loadCatalog();
})();
