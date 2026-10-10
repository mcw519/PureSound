/* Word check under a comparison deck: transcribe the deck's tracks with a
 * recogniser and show, word by word, what each one lost, added or changed.
 *
 * Keys for the cloud recognisers live in this module's memory only -- not in
 * localStorage, not in the page's URL, not on the server.  They are sent with
 * a transcription request and are gone when the tab closes. */
(() => {
  "use strict";

  // Shared by every panel on the page, so a key is typed once per tab.
  const credentials = { azure: { key: "", region: "" }, elevenlabs: { key: "" } };
  let capabilities = null;
  const t = (key, vars) => (window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);

  const LANGUAGES = [
    ["", "Auto-detect"],  // translated where the options are drawn
    ["en-US", "English (US)"],
    ["en-GB", "English (UK)"],
    ["zh-TW", "中文（台灣）"],
    ["zh-CN", "中文（中国）"],
    ["ja-JP", "日本語"],
    ["ko-KR", "한국어"],
  ];
  const WHISPER_MODELS = ["large-v3", "distil-large-v3", "medium", "small", "base", "tiny"];
  const CJK = /^[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]$/;

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  }

  async function loadCapabilities() {
    if (capabilities) return capabilities;
    try {
      const response = await fetch("/api/asr");
      capabilities = response.ok ? await response.json() : { whisper: { cached: [], default: "large-v3" }, elevenlabs: { default: "scribe_v2" } };
    } catch {
      capabilities = { whisper: { cached: [], default: "large-v3" }, elevenlabs: { default: "scribe_v2" } };
    }
    return capabilities;
  }

  async function postJson(path, body) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.error?.message || t("Request failed ({status})", { status: response.status }));
    return payload;
  }

  class TranscribePanel {
    constructor(host, deck) {
      this.host = host;
      this.deck = deck;
      this.jobId = null;
      this.result = null;
      this.generation = 0;
      this.render();
      deck.root.addEventListener("deck:tracks", () => this.refresh());
      window.addEventListener("puresound:asr-credentials", () => this.syncKeyFields());
      loadCapabilities().then(() => this.fillModels());
      window.addEventListener("puresound:lang", () => {
        this.fillModels();
        this.renderTrackList();
        if (this.result) this.renderResult(this.result);
      });
      this.refresh();
    }

    render() {
      this.host.classList.add("transcribe");
      this.host.innerHTML = `
        <details class="transcribe-box">
          <summary><span class="eyebrow" data-i18n>Words · transcribe</span><strong data-i18n>Which words survived?</strong><small data-i18n>Transcribe the tracks and see, word by word, what each one lost, added or changed.</small></summary>
          <div class="transcribe-form">
            <label><span data-i18n>Recogniser</span><select class="control" data-asr-backend><option value="whisper" data-i18n>Whisper · runs on this server</option><option value="azure">Azure Speech</option><option value="elevenlabs">ElevenLabs Scribe</option></select></label>
            <label data-asr-for="whisper"><span data-i18n>Model</span><select class="control" data-asr-whisper-model></select></label>
            <label data-asr-for="elevenlabs" hidden><span data-i18n>Model</span><input class="control" data-asr-eleven-model spellcheck="false" autocomplete="off"></label>
            <label><span data-i18n>Language</span><select class="control" data-asr-language>${LANGUAGES.map(([value, label]) => `<option value="${value}" data-i18n>${escapeHtml(label)}</option>`).join("")}</select></label>
            <label data-asr-for="azure" hidden><span data-i18n>Azure key</span><input class="control" type="password" data-asr-azure-key autocomplete="off" spellcheck="false" placeholder="Speech resource key" data-i18n-attr="placeholder"></label>
            <label data-asr-for="azure" hidden><span data-i18n>Region</span><input class="control" data-asr-azure-region autocomplete="off" spellcheck="false" placeholder="e.g. eastus" data-i18n-attr="placeholder"></label>
            <label data-asr-for="elevenlabs" hidden><span data-i18n>ElevenLabs key</span><input class="control" type="password" data-asr-eleven-key autocomplete="off" spellcheck="false" placeholder="xi-api-key"></label>
            <p class="transcribe-key-note" data-asr-key-note hidden><span data-i18n>Keys stay in this tab's memory and go only with a transcription request — neither the server nor the browser stores them, and they are gone when the tab closes.</span> <button type="button" class="link-button" data-asr-forget data-i18n>Forget keys</button></p>
            <label class="transcribe-reference"><span><span data-i18n>Reference transcript</span> <em data-i18n>optional</em></span><textarea class="control" rows="2" data-asr-reference placeholder="What the near talker actually said. With it, every track gets an error rate against it; without it, tracks are compared with the reference track's transcript." data-i18n-attr="placeholder"></textarea></label>
            <label data-asr-reference-track-row><span data-i18n>Compare against</span><select class="control" data-asr-reference-track></select></label>
            <div class="transcribe-tracks" data-asr-tracks></div>
            <div class="transcribe-actions"><button class="button button-secondary" type="button" data-asr-run><span data-i18n>Transcribe</span> <span aria-hidden="true">→</span></button><button class="button button-secondary" type="button" data-asr-cancel hidden><span data-i18n>Cancel</span></button><span class="transcribe-status" data-asr-status></span></div>
          </div>
          <div class="transcribe-results" data-asr-results></div>
        </details>`;
      window.PureSoundI18n?.apply(this.host);
      const $ = (selector) => this.host.querySelector(selector);
      this.backend = $("[data-asr-backend]");
      this.whisperModel = $("[data-asr-whisper-model]");
      this.elevenModel = $("[data-asr-eleven-model]");
      this.language = $("[data-asr-language]");
      this.azureKey = $("[data-asr-azure-key]");
      this.azureRegion = $("[data-asr-azure-region]");
      this.elevenKey = $("[data-asr-eleven-key]");
      this.keyNote = $("[data-asr-key-note]");
      this.reference = $("[data-asr-reference]");
      this.referenceTrack = $("[data-asr-reference-track]");
      this.referenceTrackRow = $("[data-asr-reference-track-row]");
      this.trackList = $("[data-asr-tracks]");
      this.runButton = $("[data-asr-run]");
      this.cancelButton = $("[data-asr-cancel]");
      this.status = $("[data-asr-status]");
      this.results = $("[data-asr-results]");
      this.backend.addEventListener("change", () => this.syncBackend());
      this.azureKey.addEventListener("input", () => this.storeKeys());
      this.azureRegion.addEventListener("input", () => this.storeKeys());
      this.elevenKey.addEventListener("input", () => this.storeKeys());
      this.reference.addEventListener("input", () => this.syncReference());
      $("[data-asr-forget]").addEventListener("click", () => {
        credentials.azure = { key: "", region: credentials.azure.region };
        credentials.elevenlabs = { key: "" };
        window.dispatchEvent(new CustomEvent("puresound:asr-credentials"));
      });
      this.runButton.addEventListener("click", () => this.run());
      this.cancelButton.addEventListener("click", () => this.cancel());
      this.results.addEventListener("click", (event) => this.seekToken(event));
      this.syncBackend();
      this.syncKeyFields();
    }

    fillModels() {
      const cached = new Set(capabilities?.whisper?.cached || []);
      const names = [...new Set([...WHISPER_MODELS, ...cached])];
      const chosen = this.whisperModel.value;
      this.whisperModel.innerHTML = names.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}${name === (capabilities?.whisper?.default || "large-v3") ? ` · ${escapeHtml(t("recommended"))}` : ""}${cached.has(name) ? "" : ` · ${escapeHtml(t("downloads on first use"))}`}</option>`).join("");
      this.whisperModel.value = names.includes(chosen) ? chosen : capabilities?.whisper?.default || "large-v3";
      if (!this.elevenModel.value) this.elevenModel.value = capabilities?.elevenlabs?.default || "scribe_v2";
    }

    storeKeys() {
      credentials.azure = { key: this.azureKey.value.trim(), region: this.azureRegion.value.trim() };
      credentials.elevenlabs = { key: this.elevenKey.value.trim() };
      window.dispatchEvent(new CustomEvent("puresound:asr-credentials"));
    }

    syncKeyFields() {
      if (this.azureKey.value !== credentials.azure.key) this.azureKey.value = credentials.azure.key;
      if (this.azureRegion.value !== credentials.azure.region) this.azureRegion.value = credentials.azure.region;
      if (this.elevenKey.value !== credentials.elevenlabs.key) this.elevenKey.value = credentials.elevenlabs.key;
    }

    syncBackend() {
      const backend = this.backend.value;
      this.host.querySelectorAll("[data-asr-for]").forEach((row) => { row.hidden = row.dataset.asrFor !== backend; });
      this.keyNote.hidden = backend === "whisper";
    }

    syncReference() {
      this.referenceTrackRow.hidden = Boolean(this.reference.value.trim());
    }

    /* The deck's tracks changed: offer them, and drop results about old audio. */
    refresh() {
      this.generation += 1;
      const tracks = this.deck.tracks.filter((track) => track.url || track.data);
      this.host.hidden = !tracks.length;
      this.renderTrackList({ keep: false });
      this.results.innerHTML = "";
      this.result = null;
      this.status.textContent = "";
      this.syncReference();
    }

    /* The tracks to transcribe and the one to compare against; `keep` holds
     * the current choices (a language switch redraws the labels only). */
    renderTrackList({ keep = true } = {}) {
      const tracks = this.deck.tracks.filter((track) => track.url || track.data);
      const was = keep ? new Set([...this.trackList.querySelectorAll("input:checked")].map((input) => input.value)) : null;
      const wasReference = keep ? this.referenceTrack.value : null;
      const checked = (track) => (was ? was.has(track.id) : !track.diagnostic);
      this.trackList.innerHTML = `<span>${escapeHtml(t("Tracks"))}</span>${tracks.map((track) => `<label class="transcribe-track"><input type="checkbox" value="${escapeHtml(track.id)}"${checked(track) ? " checked" : ""}><i style="background:${escapeHtml(track.color)}"></i>${escapeHtml(t(track.label))}</label>`).join("")}`;
      const reference = tracks.find((track) => track.id === wasReference) || tracks.find((track) => track.levelReference) || tracks[0];
      this.referenceTrack.innerHTML = tracks.map((track) => `<option value="${escapeHtml(track.id)}"${track === reference ? " selected" : ""}>${escapeHtml(t("{track}’s transcript", { track: t(track.label) }))}</option>`).join("");
    }

    async trackSource(track) {
      if (track.url && track.url.startsWith("/api/runs/")) return { url: track.url };
      if (track.uploadId) return { upload_id: track.uploadId };
      const blob = track.data || (track.url ? await (await fetch(track.url)).blob() : null);
      if (!blob) throw new Error(t("{track} has no audio to send", { track: track.label }));
      const response = await fetch("/api/uploads", { method: "POST", headers: { "Content-Type": blob.type || "audio/wav", "X-Filename": encodeURIComponent(`${track.id}.wav`) }, body: blob });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload?.error?.message || t("Upload failed ({status})", { status: response.status }));
      track.uploadId = payload.upload_id;
      return { upload_id: payload.upload_id };
    }

    async run() {
      const backend = this.backend.value;
      const generation = this.generation;
      const chosen = [...this.trackList.querySelectorAll("input:checked")].map((input) => this.deck.track(input.value)).filter(Boolean);
      const referenceText = this.reference.value.trim();
      const referenceTrack = this.referenceTrack.value;
      if (!chosen.length) { this.say(t("Choose at least one track."), true); return; }
      if (!referenceText && !chosen.some((track) => track.id === referenceTrack)) {
        const track = this.deck.track(referenceTrack);
        if (track) chosen.unshift(track);
      }
      if (backend === "azure" && (!credentials.azure.key || !credentials.azure.region)) { this.say(t("Azure needs a key and a region."), true); return; }
      if (backend === "elevenlabs" && !credentials.elevenlabs.key) { this.say(t("ElevenLabs needs an API key."), true); return; }
      this.runButton.disabled = true;
      this.cancelButton.hidden = false;
      this.say(t(backend === "whisper" ? "Transcribing on the server — a large model takes a few seconds per track, longer the first time it loads…" : "Sending to the recogniser…"));
      try {
        const tracks = [];
        for (const track of chosen) tracks.push({ id: track.id, label: track.label, source: await this.trackSource(track) });
        const job = await postJson("/api/jobs", {
          kind: "transcription",
          backend,
          model: backend === "whisper" ? this.whisperModel.value : backend === "elevenlabs" ? this.elevenModel.value.trim() : "",
          language: this.language.value,
          credentials: backend === "whisper" ? {} : { ...credentials[backend] },
          reference_text: referenceText,
          reference_track: referenceTrack,
          tracks,
        });
        this.jobId = job.job_id;
        const result = await this.poll(job.job_id);
        if (generation !== this.generation) { this.status.textContent = ""; return; } // the deck holds other audio now
        this.result = result;
        this.renderResult(result);
        this.say(`${t("Done in {seconds} s", { seconds: result.elapsed_seconds.toFixed(1) })} · ${result.model}${result.language ? ` · ${result.language}` : ""}.`);
      } catch (error) {
        this.say(error.message, true);
      } finally {
        this.jobId = null;
        this.runButton.disabled = false;
        this.cancelButton.hidden = true;
      }
    }

    async poll(jobId) {
      for (;;) {
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        const job = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(job?.error?.message || t("Request failed ({status})", { status: response.status }));
        if (job.status === "succeeded") return job.result;
        if (job.status === "failed") throw new Error(job.error || t("Transcription failed."));
        if (job.status === "cancelled") throw new Error(t("Transcription cancelled."));
        const phase = String(job.phase || "").replace(/_/g, " ").replace(/:/g, " · ");
        this.status.textContent = `${Math.round((job.progress || 0) * 100)}% · ${phase}`;
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    }

    async cancel() {
      if (!this.jobId) return;
      await fetch(`/api/jobs/${encodeURIComponent(this.jobId)}/cancel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    }

    say(text, isError = false) {
      this.status.textContent = text;
      this.status.classList.toggle("is-error", isError);
    }

    /* Rendering ------------------------------------------------------------ */
    renderResult(result) {
      const reference = result.reference;
      const unit = result.unit === "character" ? "CER" : "WER";
      const e = (key, vars) => escapeHtml(t(key, vars));
      const trackLabel = result.tracks.find((track) => track.id === reference.track_id)?.label;
      const intro = reference.kind === "text"
        ? `${e("Scored against your reference transcript: an error rate ({unit}).", { unit })} <del>${e("Struck through")}</del> = ${e("lost")}, <ins>${e("underlined")}</ins> = ${e("added")}, <mark>${e("marked")}</mark> = ${e("changed")}.`
        : `${trackLabel ? e("No reference transcript, so each track is compared with {track}’s transcript.", { track: trackLabel }) : e("No reference transcript, so each track is compared with the reference track’s transcript.")} <b>${e("This is a difference, not an error rate")}</b>: ${e("a voice-isolation model is meant to remove a distant talker's words, and those show up as missing too. Paste what the near talker said for a real {unit}.", { unit })}`;
      const cards = result.tracks.map((track) => this.trackCard(track, result)).join("");
      const notes = [...new Set(result.tracks.flatMap((track) => track.notes || []))];
      this.results.innerHTML = `<p class="transcribe-intro">${intro}${notes.length ? ` <em>${escapeHtml(notes.join(" · "))}</em>` : ""}</p>${reference.kind === "text" ? `<div class="transcribe-card is-reference"><div class="transcribe-card-head"><strong>${escapeHtml(t("Reference"))}</strong><small>${escapeHtml(t("your text"))} · ${escapeHtml(t(result.unit === "character" ? "{n} characters" : "{n} words", { n: reference.tokens.length }))}</small></div><p class="transcribe-text">${this.joinTokens(reference.tokens.map((token) => ({ token, className: "", time: null })))}</p></div>` : ""}${cards}<p class="transcribe-hint">${escapeHtml(t("Click a word to hear it: the player selects that track and loops a moment around it."))}</p>`;
    }

    trackCard(track, result) {
      const deckTrack = this.deck.track(track.id);
      const color = deckTrack?.color || "#999";
      if (track.is_reference) {
        const items = track.tokens.map((token, index) => ({ token, className: "", time: track.times[index] }));
        return `<div class="transcribe-card is-reference" data-card="${escapeHtml(track.id)}"><div class="transcribe-card-head"><i style="background:${escapeHtml(color)}"></i><strong>${escapeHtml(t(track.label))}</strong><small>${escapeHtml(t("reference for the others"))} · ${escapeHtml(t(result.unit === "character" ? "{n} characters" : "{n} words", { n: track.tokens.length }))}${track.language ? ` · ${escapeHtml(track.language)}` : ""}</small></div><p class="transcribe-text">${this.joinTokens(items, track.id) || `<em>${escapeHtml(t("(nothing recognised)"))}</em>`}</p></div>`;
      }
      const counts = track.counts;
      const unit = result.unit === "character" ? "CER" : "WER";
      const referenceTimes = result.reference.times;
      const items = this.alignedItems(track, result.reference.tokens, referenceTimes);
      const stats = result.mode === "reference"
        ? `<span class="transcribe-rate">${unit} ${(counts.error_rate * 100).toFixed(1)}%</span><span class="is-del">${escapeHtml(t("lost {n}", { n: counts.del }))}</span><span class="is-ins">${escapeHtml(t("added {n}", { n: counts.ins }))}</span><span class="is-sub">${escapeHtml(t("changed {n}", { n: counts.sub }))}</span><span>${escapeHtml(t("of {n}", { n: counts.ref_tokens }))}</span>`
        : `<span class="transcribe-rate">${escapeHtml(t("differs {rate}%", { rate: (counts.error_rate * 100).toFixed(1) }))}</span><span class="is-del">${escapeHtml(t("missing {n}", { n: counts.del }))}</span><span class="is-ins">${escapeHtml(t("added {n}", { n: counts.ins }))}</span><span class="is-sub">${escapeHtml(t("changed {n}", { n: counts.sub }))}</span><span>${escapeHtml(t("of {n}", { n: counts.ref_tokens }))}</span>`;
      const loop = track.looped ? `<p class="transcribe-warning">${escapeHtml(t("The recogniser looped on this track — its transcript is over 1.5× the reference — so this number says more about the recogniser than about the audio."))}</p>` : "";
      return `<div class="transcribe-card" data-card="${escapeHtml(track.id)}"><div class="transcribe-card-head"><i style="background:${escapeHtml(color)}"></i><strong>${escapeHtml(t(track.label))}</strong><div class="transcribe-stats">${stats}</div></div>${loop}<p class="transcribe-text">${this.joinTokens(items, track.id) || `<em>${escapeHtml(t("(nothing recognised)"))}</em>`}</p></div>`;
    }

    /* One display item per alignment operation.  A lost word has no time in
     * this track: it takes the reference's, or sits between its neighbours. */
    alignedItems(track, referenceTokens, referenceTimes) {
      const items = track.alignment.map((op) => {
        if (op.op === "del") return { token: referenceTokens[op.ref], className: "is-del", title: "missing here", time: referenceTimes ? referenceTimes[op.ref] : null };
        const time = track.times[op.hyp];
        if (op.op === "ins") return { token: track.tokens[op.hyp], className: "is-ins", title: "not in the reference", time };
        if (op.op === "sub") return { token: track.tokens[op.hyp], className: "is-sub", title: t("reference: {word}", { word: referenceTokens[op.ref] }), time };
        return { token: track.tokens[op.hyp], className: "", time };
      });
      items.forEach((item, index) => {
        if (item.time && item.time[0] != null) return;
        const before = items.slice(0, index).reverse().find((other) => other.time && other.time[1] != null);
        const after = items.slice(index + 1).find((other) => other.time && other.time[0] != null);
        if (before || after) item.time = [before ? before.time[1] : after.time[0] - 0.3, after ? after.time[0] : before.time[1] + 0.3];
      });
      return items;
    }

    joinTokens(items, trackId = "") {
      return items.map((item, index) => {
        const previous = items[index - 1];
        const gap = index > 0 && !(CJK.test(item.token) && CJK.test(previous.token)) ? " " : "";
        const time = item.time && item.time[0] != null ? ` data-t0="${item.time[0]}" data-t1="${item.time[1] ?? item.time[0]}" data-track="${escapeHtml(trackId)}"` : "";
        const tag = item.className === "is-del" ? "del" : item.className === "is-ins" ? "ins" : item.className === "is-sub" ? "mark" : "span";
        return `${gap}<${tag} class="transcribe-token ${item.className}"${item.title ? ` title="${escapeHtml(item.title)}"` : ""}${time}>${escapeHtml(item.token)}</${tag}>`;
      }).join("");
    }

    seekToken(event) {
      const token = event.target.closest(".transcribe-token[data-t0]");
      if (!token) return;
      const start = Number(token.dataset.t0);
      const end = Number(token.dataset.t1);
      if (!Number.isFinite(start)) return;
      if (token.dataset.track && this.deck.track(token.dataset.track)?.buffer) this.deck.select(token.dataset.track);
      this.deck.setRegion({ start: Math.max(0, start - 0.3), end: Math.min(this.deck.duration, Math.max(end, start + 0.1) + 0.3) });
      this.deck.play();
    }
  }

  window.PureSoundTranscribe = TranscribePanel;
})();
