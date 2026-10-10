/* Comparison deck: one transport over several time-aligned renderings of the
 * same clip (input, outputs, what was removed).  Every track plays at once
 * behind its own gain and only the selected one is audible, so switching is
 * sample-aligned and gapless -- the A/B a listener needs to hear a difference
 * rather than remember one.
 *
 * Viewing: time and frequency both zoom (buttons, Ctrl/⌘ + wheel for time,
 * Alt + wheel for frequency) and scroll (a time bar under the ruler, a
 * frequency bar beside the lanes, Shift + wheel).  A region is dragged out,
 * its edges dragged to adjust, Shift + click extends it; it loops, zooms and
 * exports.  Spectrogram and waveform settings persist per browser and apply
 * to every deck on the page.  Playback only; nothing here touches exports. */
(() => {
  "use strict";

  const A = window.PureSoundAudio;
  const DRAG_PIXELS = 4;
  const EDGE_PIXELS = 6;
  const LEVEL_MATCH_LIMIT_DB = 12;
  const MIN_REGION_SECONDS = 0.02;
  const MIN_VIEW_SECONDS = 0.005;
  const MIN_FREQ_AXIS = 0.02;
  const LOG_MIN_HZ = 30;
  // Level-change lane: frames of 20 ms, drawn over +-24 dB; frames where the
  // reference itself is below -60 dBFS are silence and are greyed out.
  const LEVEL_FRAME_SECONDS = 0.02;
  const LEVEL_RANGE_DB = 24;
  const LEVEL_SILENCE_DBFS = -60;
  const SETTINGS_KEY = "puresound.deck-view";
  const DEFAULT_VIEW = Object.freeze({ ...A.DEFAULT_SPECTROGRAM, waveScale: "linear", waveGain: 1 });

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  }

  function loadSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
      return { ...DEFAULT_VIEW, ...saved };
    } catch {
      return { ...DEFAULT_VIEW };
    }
  }

  const t = (key, vars) => (window.PureSoundI18n ? window.PureSoundI18n.t(key, vars) : key);

  function saveSettings(settings) {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* storage may be disabled */ }
    window.dispatchEvent(new CustomEvent("puresound:view-settings", { detail: settings }));
  }

  class CompareDeck {
    /* tracks: [{ id, label, color, hint?, levelReference?, diagnostic? }] */
    constructor(root, { emptyText = "No audio loaded.", analysis = true, compact = false, listening = false } = {}) {
      this.root = root;
      this.emptyText = emptyText;
      this.analysis = analysis;
      this.compact = compact;
      this.listeningEnabled = listening;
      this.listening = listening;
      this.viewBeforeListening = "both";
      this.visibleTracks = new Set();
      this.visibilityKey = `puresound.deck-tracks.${root.id || "default"}`;
      try {
        const saved = JSON.parse(localStorage.getItem(this.visibilityKey) || "null");
        this.savedVisibility = Array.isArray(saved) ? saved.filter((id) => typeof id === "string") : null;
      } catch { this.savedVisibility = null; }
      this.settings = loadSettings();
      this.curves = [];
      this.levelTraces = new Map();
      this.analysisOn = false;
      this.tracks = [];
      this.selectedId = null;
      this.playing = false;
      this.offset = 0;
      this.startedAt = 0;
      this.playOffset = 0;
      this.playLoop = null;
      this.region = null;
      this.loop = false;
      this.view = null;
      this.freqView = null;
      this.boostDb = 0;
      this.levelMatch = false;
      this.chain = null;
      this.animation = null;
      this.viewMode = listening ? "wave" : "both";
      this.renderShell();
      this.bindEvents();
      this.applySettingsToControls();
      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(this.area);
      window.addEventListener("puresound:view-settings", (event) => {
        this.settings = { ...DEFAULT_VIEW, ...event.detail };
        this.applySettingsToControls();
        this.clampFreqView();
        this.draw();
      });
      // Marked text follows a language switch by itself; these are composed.
      window.addEventListener("puresound:lang", () => {
        this.renderTrackButtons();
        this.tracks.forEach((track) => { const name = track.lane?.querySelector("[data-lane-name]"); if (name) name.textContent = t(track.label); });
        this.curvesPanel.querySelectorAll("[data-curve]").forEach((lane) => {
          const curve = this.curves[Number(lane.dataset.curve)];
          const [name, hint] = lane.querySelectorAll(".deck-lane-label span, .deck-lane-label small");
          if (curve && name) name.textContent = t(curve.label);
          if (curve && hint) hint.textContent = t(curve.hint || "");
        });
        this.updateLaneMeta();
        this.updateRegionLabel();
        if (this.tracks.length) this.setAnalysis(this.analysisOn);
        this.playButton?.setAttribute("aria-label", t(this.playing ? "Pause" : "Play"));
      });
    }

    renderShell() {
      this.root.classList.add("compare-deck");
      this.root.classList.toggle("is-compact", this.compact);
      this.root.classList.toggle("has-listening", this.listeningEnabled);
      this.root.classList.toggle("is-listening", this.listening);
      const option = (value, label, mark = true) => `<option value="${value}"${mark ? " data-i18n" : ""}>${label}</option>`;
      this.root.innerHTML = `
        ${this.listeningEnabled ? `<div class="deck-listening-head">
          <div><strong data-i18n>Listen and compare</strong><p data-i18n>Switch tracks without losing your place.</p></div>
          <div class="segmented" role="group" aria-label="Listening controls" data-i18n-attr="aria-label">
            <button type="button" data-listening="true" aria-pressed="true" data-i18n>Listen</button>
            <button type="button" data-listening="false" aria-pressed="false" data-i18n>Inspect audio</button>
          </div></div>` : ""}
        <div class="audio-tools deck-tools">
          <div class="deck-row">
            <div class="audio-transport">
              <button class="audio-tool-button audio-play" type="button" aria-label="Play" title="Play / pause (Space)" disabled data-i18n-attr="title,aria-label">▶</button>
              <button class="audio-tool-button audio-stop" type="button" aria-label="Stop" title="Stop" disabled data-i18n-attr="title,aria-label">■</button>
              <span class="audio-clock"><span data-clock-current>0:00.000</span><span> / </span><span data-clock-duration>0:00.000</span></span>
            </div>
            <div class="deck-tracks" role="radiogroup" aria-label="Audible track" data-i18n-attr="aria-label"></div>
            <details class="deck-track-picker"><summary><span data-i18n>Visible tracks</span> <span data-visible-count></span></summary>
              <fieldset><legend data-i18n>Choose tracks to compare</legend><div data-track-choices></div><small data-i18n>Checked tracks stay visible in both views. Click a track card to listen.</small></fieldset>
            </details>
            <div class="audio-boost" title="Playback-only gain with a safety limiter; inference and exports are unchanged" data-i18n-attr="title">
              <button class="audio-tool-button" type="button" data-level-match aria-pressed="false" title="Play every track at the input's loudness, so the louder one does not simply sound better" data-i18n-attr="title"><span data-i18n>Match level</span></button>
              <label><span data-i18n>Boost</span> <input type="range" min="0" max="48" step="0.5" value="0" data-boost /></label>
              <output data-boost-value>0.0 dB</output>
              <button class="audio-auto" type="button" disabled data-i18n>Auto</button>
              <span class="audio-limiter" data-limiter>LIM</span>
            </div>
          </div>
          <div class="deck-row">
            <div class="deck-group deck-loop" role="group" aria-label="Selection" data-i18n-attr="aria-label">
              <button class="audio-tool-button" type="button" data-loop aria-pressed="false" title="Loop the selection (L)" disabled data-i18n data-i18n-attr="title">Loop</button>
              <span class="deck-region-label" data-region-label data-i18n>Drag to select · Shift + click extends</span>
              <button class="audio-tool-button" type="button" data-zoom title="Zoom to the selection (Z)" disabled data-i18n data-i18n-attr="title">Zoom</button>
              <button class="audio-tool-button" type="button" data-export-region title="Download the selection of the audible track as WAV" disabled data-i18n-attr="title"><span data-i18n>Export</span> ↓</button>
              <button class="audio-tool-button" type="button" data-annotate title="Open the audible track in Annotate, to mark spans on it" disabled hidden data-i18n-attr="title"><span data-i18n>Annotate</span> ↗</button>
              <button class="audio-tool-button" type="button" data-clear-region aria-label="Clear selection" title="Clear selection and zoom (Esc)" disabled data-i18n-attr="title,aria-label">×</button>
            </div>
            <div class="deck-group" role="group" aria-label="Time zoom" data-i18n-attr="aria-label">
              <span class="deck-group-label" data-i18n>Time</span>
              <button class="audio-tool-button" type="button" data-time-zoom="out" aria-label="Zoom out in time" title="Zoom out (−, or Ctrl/⌘ + wheel)" disabled data-i18n-attr="title,aria-label">−</button>
              <button class="audio-tool-button" type="button" data-time-zoom="in" aria-label="Zoom in in time" title="Zoom in (+, or Ctrl/⌘ + wheel)" disabled data-i18n-attr="title,aria-label">+</button>
              <button class="audio-tool-button" type="button" data-time-zoom="fit" title="Show the whole clip and frequency range (0)" disabled data-i18n data-i18n-attr="title">Fit</button>
            </div>
            <div class="audio-view-switch" role="group" aria-label="Audio view" data-i18n-attr="aria-label">
              <button type="button" data-view="wave" data-i18n>Wave</button><button class="is-active" type="button" data-view="both" data-i18n>Both</button><button type="button" data-view="spec" data-i18n>Spec</button>
            </div>
            <button class="audio-tool-button" type="button" data-analysis aria-pressed="false" title="Show how the audible track differs from the reference track, bin by bin and in level" data-i18n-attr="title">Δ vs input</button>
            <button class="audio-tool-button" type="button" data-settings-toggle aria-expanded="false" title="Spectrogram and waveform settings" data-i18n-attr="title"><span data-i18n>View</span> ⚙</button>
          </div>
          <div class="deck-settings" hidden>
            <div class="deck-settings-group">
              <p data-i18n>Spectrogram</p>
              <label><span data-i18n>FFT size</span><select data-setting="fftSize">${[256, 512, 1024, 2048, 4096, 8192].map((size) => option(size, `${size}`, false)).join("")}</select><small data-fft-note></small></label>
              <label><span data-i18n>Frequency range</span><select data-setting="maxHz">${[[4000, "0–4 kHz"], [8000, "0–8 kHz"], [11025, "0–11 kHz"], [16000, "0–16 kHz"], [24000, "0–24 kHz (up to Nyquist)"]].map(([value, label]) => option(value, label)).join("")}</select></label>
              <label><span data-i18n>Frequency axis</span><select data-setting="scale">${option("linear", "Linear")}${option("log", "Logarithmic")}</select></label>
              <label><span data-i18n>Floor</span><input type="range" min="-140" max="-40" step="2" data-setting="floorDb"><output data-setting-output="floorDb"></output></label>
              <label><span data-i18n>Dynamic range</span><input type="range" min="30" max="140" step="2" data-setting="rangeDb"><output data-setting-output="rangeDb"></output></label>
              <label><span data-i18n>Colours</span><select data-setting="colormap">${option("puresound", "PureSound", false)}${option("magma", "Magma", false)}${option("viridis", "Viridis", false)}${option("gray", "Grey")}</select></label>
            </div>
            <div class="deck-settings-group">
              <p data-i18n>Waveform</p>
              <label><span data-i18n>Amplitude</span><select data-setting="waveScale">${option("linear", "Linear")}${option("db", "dB (0 to −60 dBFS)")}</select></label>
              <label><span data-i18n>Gain</span><select data-setting="waveGain">${[1, 2, 4, 8, 16, 32].map((gain) => option(gain, `×${gain}`, false)).join("")}</select></label>
              <button class="audio-tool-button" type="button" data-settings-reset data-i18n>Reset to defaults</button>
            </div>
          </div>
        </div>
        <div class="deck-stage">
          <div class="deck-area">
            <div class="audio-ruler deck-ruler"><canvas aria-hidden="true"></canvas></div>
            <div class="deck-tscroll" title="Drag to scroll in time" data-i18n-attr="title"><span class="deck-thumb"></span></div>
            <div class="deck-lanes"></div>
            <div class="deck-analysis" hidden>
              <div class="deck-lane deck-analysis-lane" data-analysis-level>
                <span class="deck-lane-label is-static"><span data-i18n>Level change</span><small data-analysis-level-label>selected − reference · 20 ms frames · speech, noise and reverb together</small></span>
                <div class="audio-pane"><canvas aria-label="Level change over time" data-i18n-attr="aria-label"></canvas></div>
              </div>
              <div class="deck-lane deck-analysis-lane" data-analysis-spec>
                <span class="deck-lane-label is-static"><span data-i18n>Δ spectrum</span><small data-analysis-spec-label>cool = removed · warm = added · ±30 dB</small></span>
                <div class="audio-pane audio-diff-pane"><canvas aria-label="Spectral difference" data-i18n-attr="aria-label"></canvas><div class="audio-progress"></div></div>
              </div>
            </div>
            <div class="deck-curves" hidden></div>
            <div class="deck-hover" hidden></div>
            <div class="deck-region" hidden></div>
            <div class="audio-playhead" aria-hidden="true"></div>
            <div class="deck-empty" data-i18n>${escapeHtml(this.emptyText)}</div>
          </div>
          <div class="deck-fscroll" aria-label="Frequency zoom" data-i18n-attr="aria-label">
            <button type="button" data-freq-zoom="in" aria-label="Zoom in in frequency" title="Zoom in in frequency (Alt + wheel)" data-i18n-attr="title,aria-label">+</button>
            <button type="button" data-freq-zoom="out" aria-label="Zoom out in frequency" title="Zoom out in frequency (Alt + wheel)" data-i18n-attr="title,aria-label">−</button>
            <div class="deck-fscroll-track" title="Drag to scroll in frequency" data-i18n-attr="title"><span class="deck-thumb"></span></div>
            <span class="deck-fscroll-label" data-freq-label>kHz</span>
          </div>
        </div>
        <p class="deck-listening-note" data-i18n>Space to play · click a card to switch · drag a region to loop. Level matching affects listening only.</p>
        <p class="audio-audition-note" data-i18n>Space play · 1–9 switch track · drag to select, drag its edges to adjust, Shift + click extends · L loop · Z zoom to selection · + / − / 0 zoom · Ctrl/⌘ + wheel zooms time, Shift + wheel scrolls it, Alt + wheel zooms frequency · Esc clear. Boost and level match affect browser audition only — model input and exported WAV stay unchanged.</p>`;
      window.PureSoundI18n?.apply(this.root);
      const $ = (selector) => this.root.querySelector(selector);
      this.stage = $(".deck-stage");
      this.area = $(".deck-area");
      this.ruler = $(".deck-ruler canvas");
      this.timeScroll = $(".deck-tscroll");
      this.timeThumb = $(".deck-tscroll .deck-thumb");
      this.freqScroll = $(".deck-fscroll");
      this.freqTrack = $(".deck-fscroll-track");
      this.freqThumb = $(".deck-fscroll-track .deck-thumb");
      this.freqLabel = $("[data-freq-label]");
      this.lanes = $(".deck-lanes");
      this.regionOverlay = $(".deck-region");
      this.playhead = $(".audio-playhead");
      this.empty = $(".deck-empty");
      this.trackButtons = $(".deck-tracks");
      this.playButton = $(".audio-play");
      this.stopButton = $(".audio-stop");
      this.currentClock = $("[data-clock-current]");
      this.durationClock = $("[data-clock-duration]");
      this.loopButton = $("[data-loop]");
      this.zoomButton = $("[data-zoom]");
      this.exportButton = $("[data-export-region]");
      this.annotateButton = $("[data-annotate]");
      this.clearButton = $("[data-clear-region]");
      this.regionLabel = $("[data-region-label]");
      this.levelButton = $("[data-level-match]");
      this.analysisButton = $("[data-analysis]");
      this.settingsButton = $("[data-settings-toggle]");
      this.settingsPanel = $(".deck-settings");
      this.analysisPanel = $(".deck-analysis");
      this.levelCanvas = $("[data-analysis-level] canvas");
      this.levelLabel = $("[data-analysis-level-label]");
      this.diffView = new A.SpectrogramView($("[data-analysis-spec] canvas"), $("[data-analysis-spec] .audio-progress"));
      this.diffLabel = $("[data-analysis-spec-label]");
      this.curvesPanel = $(".deck-curves");
      this.hover = $(".deck-hover");
      this.boost = $("[data-boost]");
      this.boostValue = $("[data-boost-value]");
      this.autoButton = $(".audio-auto");
      this.limiterBadge = $("[data-limiter]");
    }

    bindEvents() {
      this.root.querySelectorAll("[data-listening]").forEach((button) => button.addEventListener("click", () => this.setListening(button.dataset.listening === "true")));
      this.root.querySelector("[data-track-choices]").addEventListener("change", (event) => {
        const input = event.target.closest("[data-visible-track]");
        if (input) this.setTrackVisible(input.dataset.visibleTrack, input.checked);
      });
      this.playButton.addEventListener("click", () => this.toggle());
      this.stopButton.addEventListener("click", () => this.stop());
      this.loopButton.addEventListener("click", () => this.toggleLoop());
      this.zoomButton.addEventListener("click", () => this.toggleZoom());
      this.exportButton.addEventListener("click", () => this.exportRegion());
      this.annotateButton.addEventListener("click", () => this.sendToAnnotate());
      this.clearButton.addEventListener("click", () => this.clearRegion());
      this.levelButton.addEventListener("click", () => this.setLevelMatch(!this.levelMatch));
      this.analysisButton.addEventListener("click", () => this.setAnalysis(!this.analysisOn));
      this.settingsButton.addEventListener("click", () => this.toggleSettings());
      this.root.querySelectorAll("[data-time-zoom]").forEach((button) => button.addEventListener("click", () => {
        const kind = button.dataset.timeZoom;
        if (kind === "fit") this.fitView();
        else this.zoomTime(kind === "in" ? 0.5 : 2, this.zoomAnchor());
      }));
      this.root.querySelectorAll("[data-freq-zoom]").forEach((button) => button.addEventListener("click", () => this.zoomFreq(button.dataset.freqZoom === "in" ? 0.5 : 2)));
      this.root.querySelectorAll("[data-setting]").forEach((control) => control.addEventListener(control.type === "range" ? "input" : "change", () => this.changeSetting(control.dataset.setting, control.value)));
      this.root.querySelector("[data-settings-reset]").addEventListener("click", () => { this.settings = { ...DEFAULT_VIEW }; saveSettings(this.settings); });
      this.area.addEventListener("pointermove", (event) => this.pointerHover(event));
      this.area.addEventListener("pointerleave", () => { this.hover.hidden = true; this.area.style.cursor = ""; });
      this.area.addEventListener("wheel", (event) => this.wheel(event), { passive: false });
      this.boost.addEventListener("input", () => this.setBoost(Number(this.boost.value)));
      this.autoButton.addEventListener("click", () => {
        const track = this.selectedTrack();
        if (track?.peak) this.setBoost(A.clamp(A.SAFE_PEAK_DBFS - A.levelDb(track.peak) - this.trackGainDb(track), 0, A.MAX_BOOST_DB));
      });
      this.root.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => this.setView(button.dataset.view)));
      this.trackButtons.addEventListener("click", (event) => {
        const button = event.target.closest("[data-track]");
        if (button) this.select(button.dataset.track);
      });
      this.lanes.addEventListener("click", (event) => {
        const label = event.target.closest(".deck-lane-label");
        if (label) this.select(label.dataset.track);
      });
      this.area.addEventListener("pointerdown", (event) => this.pointerDown(event));
      this.timeScroll.addEventListener("pointerdown", (event) => this.scrollbarDown(event, "time"));
      this.freqTrack.addEventListener("pointerdown", (event) => this.scrollbarDown(event, "freq"));
      this.root.addEventListener("pointerdown", () => { CompareDeck.focused = this; });
      document.addEventListener("pointerdown", (event) => {
        if (!this.settingsPanel.hidden && !this.settingsPanel.contains(event.target) && event.target !== this.settingsButton) this.toggleSettings(false);
      });
    }

    /* Settings --------------------------------------------------------------- */
    toggleSettings(open = this.settingsPanel.hidden) {
      this.settingsPanel.hidden = !open;
      this.settingsButton.setAttribute("aria-expanded", open ? "true" : "false");
      this.settingsButton.classList.toggle("is-active", open);
    }

    changeSetting(name, raw) {
      const numeric = ["fftSize", "maxHz", "floorDb", "rangeDb", "waveGain"].includes(name);
      this.settings = { ...this.settings, [name]: numeric ? Number(raw) : raw };
      if (name === "maxHz" || name === "scale") this.freqView = null;
      saveSettings(this.settings);
    }

    applySettingsToControls() {
      this.root.querySelectorAll("[data-setting]").forEach((control) => { control.value = String(this.settings[control.dataset.setting]); });
      this.root.querySelector('[data-setting-output="floorDb"]').textContent = `${this.settings.floorDb} dB`;
      this.root.querySelector('[data-setting-output="rangeDb"]').textContent = `${this.settings.rangeDb} dB`;
      const rate = this.sampleRate;
      this.root.querySelector("[data-fft-note]").textContent = `${(1000 * this.settings.fftSize / rate).toFixed(1)} ms window · ${(rate / this.settings.fftSize).toFixed(1)} Hz bins`;
    }

    /* Tracks ---------------------------------------------------------------- */
    setTracks(specs) {
      this.stop();
      this.tracks = specs.map((spec) => ({ ...spec, buffer: null, samples: null, bytes: null, peak: 0, rms: 0, error: null }));
      const defaults = this.listeningEnabled ? this.tracks.filter((track) => track.levelReference || track.defaultVisible || track.id === "aligned") : this.tracks;
      this.visibleTracks = new Set((this.savedVisibility || defaults.map((track) => track.id)).filter((id) => this.track(id)));
      if (!this.visibleTracks.size && this.tracks.length) this.visibleTracks.add(this.tracks[0].id);
      this.selectedId = [...this.visibleTracks][0] || null;
      this.region = null;
      this.loop = false;
      this.view = null;
      this.freqView = null;
      this.lanes.innerHTML = this.tracks.map((track) => `
        <div class="deck-lane${track.levelReference ? " is-reference" : ""}" data-lane="${escapeHtml(track.id)}">
          <button class="deck-lane-label" type="button" data-track="${escapeHtml(track.id)}"><i style="background:${escapeHtml(track.color)}"></i><span data-lane-name>${escapeHtml(t(track.label))}</span><small data-lane-meta></small></button>
          <div class="audio-pane audio-wave-pane"><canvas aria-label="${escapeHtml(t("{track} waveform", { track: t(track.label) }))}"></canvas></div>
          <div class="audio-pane audio-spec-pane"><canvas aria-label="${escapeHtml(t("{track} spectrogram", { track: t(track.label) }))}"></canvas><div class="audio-progress"></div></div>
        </div>`).join("");
      this.tracks.forEach((track) => {
        const lane = this.lanes.querySelector(`[data-lane="${CSS.escape(track.id)}"]`);
        track.lane = lane;
        track.waveCanvas = lane.querySelector(".audio-wave-pane canvas");
        track.spectrogram = new A.SpectrogramView(lane.querySelector(".audio-spec-pane canvas"), lane.querySelector(".audio-progress"));
        track.meta = lane.querySelector("[data-lane-meta]");
      });
      this.levelTraces = new Map();
      this.applyTrackVisibility();
      this.root.classList.toggle("is-single", this.tracks.length < 2);
      this.renderTrackButtons();
      this.setAnalysis(this.analysis && this.tracks.some((track) => track.levelReference));
      this.setCurves([]);
      this.setView(this.viewMode);
      this.updateControls();
      this.root.dispatchEvent(new CustomEvent("deck:tracks"));
    }

    referenceTrack() {
      return this.tracks.find((track) => track.levelReference && track.buffer) || null;
    }

    setAnalysis(on) {
      const available = this.analysis && this.tracks.length > 1 && this.tracks.some((track) => track.levelReference);
      this.analysisOn = Boolean(on) && available;
      this.analysisButton.hidden = !available;
      this.analysisButton.setAttribute("aria-pressed", this.analysisOn ? "true" : "false");
      this.analysisButton.classList.toggle("is-active", this.analysisOn);
      this.analysisPanel.hidden = !this.analysisOn;
      const reference = this.tracks.find((track) => track.levelReference);
      this.analysisButton.textContent = t("Δ vs {track}", { track: reference ? t(reference.label) : t("input") });
      requestAnimationFrame(() => this.draw());
    }

    /* Per-frame side information (e.g. VAD heads): [{ label, values, hopSeconds,
     * offsetSeconds, color, range: [low, high] }], drawn as line lanes. */
    setCurves(curves) {
      this.curves = (curves || []).filter((curve) => curve?.values?.length);
      this.curvesPanel.hidden = !this.curves.length;
      this.curvesPanel.innerHTML = this.curves.map((curve, index) => `
        <div class="deck-lane deck-curve-lane" data-curve="${index}">
          <span class="deck-lane-label is-static"><i style="background:${escapeHtml(curve.color || "#7ee0a1")}"></i><span>${escapeHtml(t(curve.label))}</span><small>${escapeHtml(t(curve.hint || ""))}</small></span>
          <div class="audio-pane"><canvas aria-label="${escapeHtml(t(curve.label))}"></canvas></div>
        </div>`).join("");
      requestAnimationFrame(() => this.draw());
    }

    renderTrackButtons() {
      const focused = this.trackButtons.contains(document.activeElement) ? document.activeElement.dataset.track : null;
      this.trackButtons.innerHTML = this.tracks.map((track, index) => `<button type="button" role="radio" data-track="${escapeHtml(track.id)}"${this.visibleTracks.has(track.id) ? "" : " hidden"} aria-checked="${track.id === this.selectedId}" title="${escapeHtml(t(track.hint || track.label))} (${index + 1})"${track.buffer ? "" : " disabled"}><kbd>${index + 1}</kbd><i style="background:${escapeHtml(track.color)}"></i>${this.listeningEnabled ? `<span class="deck-track-copy"><strong>${escapeHtml(t(track.label))}</strong><small>${escapeHtml(t(track.hint || track.label))}</small></span>` : escapeHtml(t(track.label))}</button>`).join("");
      this.renderTrackChoices();
      if (focused) this.trackButtons.querySelector(`[data-track="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
    }

    renderTrackChoices() {
      const host = this.root.querySelector("[data-track-choices]");
      const focused = host.contains(document.activeElement) ? document.activeElement.dataset.visibleTrack : null;
      host.innerHTML = this.tracks.map((track) => `<label><input type="checkbox" data-visible-track="${escapeHtml(track.id)}"${this.visibleTracks.has(track.id) ? " checked" : ""}${this.visibleTracks.size === 1 && this.visibleTracks.has(track.id) ? " disabled" : ""} /><i style="background:${escapeHtml(track.color)}"></i><span>${escapeHtml(t(track.label))}</span></label>`).join("");
      this.root.querySelector("[data-visible-count]").textContent = `${this.visibleTracks.size} / ${this.tracks.length}`;
      if (focused) host.querySelector(`[data-visible-track="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
    }

    applyTrackVisibility() {
      this.tracks.forEach((track) => { if (track.lane) track.lane.hidden = !this.visibleTracks.has(track.id); });
    }

    saveTrackVisibility() {
      this.savedVisibility = [...this.visibleTracks];
      try { localStorage.setItem(this.visibilityKey, JSON.stringify(this.savedVisibility)); } catch { /* optional preference */ }
    }

    setTrackVisible(id, visible) {
      if (!this.track(id) || (!visible && this.visibleTracks.size === 1 && this.visibleTracks.has(id))) return;
      if (visible) this.visibleTracks.add(id); else this.visibleTracks.delete(id);
      if (!visible && this.selectedId === id) {
        const next = this.tracks.find((track) => this.visibleTracks.has(track.id) && track.buffer);
        if (next) this.select(next.id);
        else { this.pause(); this.selectedId = [...this.visibleTracks][0]; }
      }
      this.saveTrackVisibility();
      this.applyTrackVisibility();
      this.renderTrackButtons();
      this.updateControls();
      requestAnimationFrame(() => this.draw());
    }

    setListening(on) {
      if (!this.listeningEnabled || on === this.listening) return;
      if (on) this.viewBeforeListening = this.viewMode;
      this.listening = on;
      this.root.classList.toggle("is-listening", on);
      this.root.querySelectorAll("[data-listening]").forEach((button) => button.setAttribute("aria-pressed", String((button.dataset.listening === "true") === on)));
      this.setView(on ? "wave" : this.viewBeforeListening);
      requestAnimationFrame(() => this.draw());
    }

    track(id) {
      return this.tracks.find((item) => item.id === id) || null;
    }

    selectedTrack() {
      return this.track(this.selectedId);
    }

    get duration() {
      return this.tracks.reduce((longest, track) => Math.max(longest, track.buffer?.duration || 0), 0);
    }

    get sampleRate() {
      // No AudioContext is created just to answer this: before any audio is
      // decoded the browser's usual 48 kHz is close enough for a label.
      return this.tracks.find((track) => track.buffer)?.buffer.sampleRate || 48000;
    }

    async loadTrack(id, source) {
      const track = this.track(id);
      if (!track) return null;
      track.error = null;
      track.lane.classList.add("is-loading");
      try {
        let bytes = source;
        if (typeof source === "string") {
          const response = await fetch(source);
          if (!response.ok) throw new Error(t("Could not load audio ({status})", { status: response.status }));
          bytes = await response.arrayBuffer();
        } else if (source instanceof Blob) {
          bytes = await source.arrayBuffer();
        }
        const buffer = await A.audioContext().decodeAudioData(bytes.slice(0));
        if (this.track(id) !== track) return null; // replaced while decoding
        track.bytes = bytes;
        track.buffer = buffer;
        track.samples = A.mixToMono(buffer);
        track.peak = A.peakOf(track.samples);
        track.rms = A.rmsOf(track.samples);
        track.spectrogram.invalidate();
        this.levelTraces?.delete(track.id);
      } catch (error) {
        track.error = error.message;
        throw error;
      } finally {
        track.lane.classList.remove("is-loading");
        this.renderTrackButtons();
        this.applySettingsToControls();
        this.updateControls();
        this.draw();
      }
      return track.buffer;
    }

    select(id) {
      const track = this.track(id);
      if (!track?.buffer) return;
      this.selectedId = id;
      if (!this.visibleTracks.has(id)) {
        this.visibleTracks.add(id);
        this.saveTrackVisibility();
        this.applyTrackVisibility();
      }
      if (this.playing) this.applyGains(0.004);
      this.renderTrackButtons();
      this.tracks.forEach((item) => item.lane.classList.toggle("is-selected", item.id === id));
      this.updateLimiter();
      if (this.analysisOn) this.drawAnalysis();
      if (this.listening) requestAnimationFrame(() => this.draw());
    }

    selectIndex(index) {
      const track = this.tracks[index];
      if (track) this.select(track.id);
    }

    /* Loudness a track is played at relative to its file: 0 unless level
     * matching is on, then the gain that brings it to the reference RMS. */
    trackGainDb(track) {
      if (!this.levelMatch || track.diagnostic) return 0;
      const reference = this.tracks.find((item) => item.levelReference && item.buffer) || this.tracks.find((item) => item.buffer);
      if (!reference || reference === track || !(track.rms > 0) || !(reference.rms > 0)) return 0;
      return A.clamp(A.levelDb(reference.rms) - A.levelDb(track.rms), -LEVEL_MATCH_LIMIT_DB, LEVEL_MATCH_LIMIT_DB);
    }

    setLevelMatch(on) {
      this.levelMatch = Boolean(on);
      this.levelButton.setAttribute("aria-pressed", this.levelMatch ? "true" : "false");
      this.levelButton.classList.toggle("is-active", this.levelMatch);
      if (this.playing) this.applyGains(0.02);
      this.updateLaneMeta();
    }

    applyGains(timeConstant) {
      const context = A.audioContext();
      this.tracks.forEach((track) => {
        if (!track.gainNode) return;
        const target = track.id === this.selectedId ? A.linearGain(this.trackGainDb(track)) : 0;
        track.gainNode.gain.setTargetAtTime(target, context.currentTime, timeConstant);
      });
    }

    updateLaneMeta() {
      this.tracks.forEach((track) => {
        if (!track.meta) return;
        if (track.error) { track.meta.textContent = track.error; return; }
        if (!track.buffer) { track.meta.textContent = t("loading…"); return; }
        const gain = this.trackGainDb(track);
        const parts = [`RMS ${A.levelDb(track.rms).toFixed(1)} dBFS`];
        if (Math.abs(gain) >= 0.05) parts.push(t("{gain} dB matched", { gain: `${gain > 0 ? "+" : ""}${gain.toFixed(1)}` }));
        if (track.hint) parts.push(t(track.hint));
        track.meta.textContent = parts.join(" · ");
      });
    }

    /* Transport ------------------------------------------------------------- */
    async play() {
      const duration = this.duration;
      if (!duration) return;
      const context = A.audioContext();
      if (!(await A.resumeAudio())) return;
      A.claimPlayback(this);
      this.stopSources();
      const loop = this.loop && this.region ? { ...this.region } : null;
      let offset = this.offset;
      if (loop && (offset < loop.start || offset >= loop.end - 0.005)) offset = loop.start;
      if (!loop && offset >= duration - 0.005) offset = 0;
      this.chain = A.createOutputChain(context, this.boostDb);
      const when = context.currentTime + 0.02;
      this.tracks.forEach((track) => {
        if (!track.buffer || offset >= track.buffer.duration) return;
        const source = context.createBufferSource();
        source.buffer = track.buffer;
        if (loop) {
          source.loop = true;
          source.loopStart = loop.start;
          source.loopEnd = Math.min(loop.end, track.buffer.duration);
        }
        const gain = context.createGain();
        gain.gain.value = track.id === this.selectedId ? A.linearGain(this.trackGainDb(track)) : 0;
        source.connect(gain).connect(this.chain.input);
        source.start(when, offset);
        track.source = source;
        track.gainNode = gain;
      });
      this.startedAt = when;
      this.playOffset = offset;
      this.playLoop = loop;
      this.playing = true;
      this.playButton.textContent = "❚❚";
      this.playButton.setAttribute("aria-label", t("Pause"));
      this.tick();
    }

    pause() {
      if (!this.playing) return;
      this.offset = this.currentTime();
      this.finishPlayback();
    }

    toggle() {
      if (this.playing) this.pause();
      else this.play();
    }

    stop() {
      this.offset = this.loop && this.region ? this.region.start : 0;
      this.finishPlayback();
      this.updatePosition();
    }

    finishPlayback() {
      this.playing = false;
      A.releasePlayback(this);
      this.stopSources();
      cancelAnimationFrame(this.animation);
      this.playButton.textContent = "▶";
      this.playButton.setAttribute("aria-label", t("Play"));
      this.updatePosition();
      this.updateLimiter();
    }

    stopSources() {
      this.tracks.forEach((track) => {
        if (track.source) {
          try { track.source.stop(); } catch { /* already stopped */ }
          track.source.disconnect();
          track.source = null;
        }
        if (track.gainNode) {
          track.gainNode.disconnect();
          track.gainNode = null;
        }
      });
      if (this.chain) {
        this.chain.disconnect();
        this.chain = null;
      }
    }

    currentTime() {
      if (!this.playing) return this.offset;
      let time = this.playOffset + Math.max(0, A.audioContext().currentTime - this.startedAt);
      const loop = this.playLoop;
      if (loop && time >= loop.end) time = loop.start + ((time - loop.start) % (loop.end - loop.start));
      return A.clamp(time, 0, this.duration);
    }

    tick() {
      if (!this.playing) return;
      if (!this.playLoop && this.currentTime() >= this.duration - 0.002) {
        this.offset = this.duration;
        this.finishPlayback();
        return;
      }
      this.updatePosition();
      this.updateLimiter();
      this.followPlayhead();
      this.animation = requestAnimationFrame(() => this.tick());
    }

    /* While zoomed, keep the playhead on screen by paging the view. */
    followPlayhead() {
      if (!this.view) return;
      const current = this.currentTime();
      const { start, end } = this.viewWindow();
      if (current > end || current < start) {
        const span = end - start;
        this.setTimeView(current - span * 0.1, current + span * 0.9);
      }
    }

    seekTo(seconds) {
      if (!this.duration) return;
      const resume = this.playing;
      this.offset = A.clamp(seconds, 0, this.duration);
      if (this.loop && this.region && (this.offset < this.region.start || this.offset > this.region.end)) this.loop = false;
      this.finishPlayback();
      this.updateControls();
      if (resume) this.play();
    }

    nudge(seconds) {
      this.seekTo(this.currentTime() + seconds);
    }

    setBoost(value) {
      this.boostDb = A.clamp(value, 0, A.MAX_BOOST_DB);
      this.boost.value = this.boostDb;
      this.boostValue.textContent = `${this.boostDb.toFixed(1)} dB`;
      if (this.chain) this.chain.setBoost(this.boostDb);
      this.updateLimiter();
    }

    updateLimiter() {
      const reduction = this.chain && this.playing ? Math.max(0, -this.chain.limiter.reduction) : 0;
      const track = this.selectedTrack();
      const willLimit = Boolean(track?.buffer) && A.levelDb(track.peak) + this.trackGainDb(track) + this.boostDb > -1;
      this.limiterBadge.classList.toggle("is-active", reduction > 0.4);
      this.limiterBadge.classList.toggle("is-warning", reduction <= 0.4 && willLimit);
      this.limiterBadge.textContent = reduction > 0.4 ? `LIM ${reduction.toFixed(1)}` : "LIM";
    }

    /* Time view ------------------------------------------------------------- */
    viewWindow() {
      const duration = this.duration;
      if (this.view && this.view.end > this.view.start) return { start: this.view.start, end: Math.min(this.view.end, duration) };
      return { start: 0, end: duration };
    }

    setTimeView(start, end) {
      const duration = this.duration;
      if (!duration) return;
      const span = A.clamp(end - start, Math.min(MIN_VIEW_SECONDS, duration), duration);
      if (span >= duration - 1e-9) {
        this.view = null;
      } else {
        const first = A.clamp(start, 0, duration - span);
        this.view = { start: first, end: first + span };
      }
      this.updateControls();
      this.draw();
    }

    /* Keep `anchor` (seconds) under the same pixel while the span scales. */
    zoomTime(factor, anchor) {
      const { start, end } = this.viewWindow();
      const span = end - start;
      const fraction = span > 0 ? A.clamp((anchor - start) / span, 0, 1) : 0.5;
      const next = span * factor;
      this.setTimeView(anchor - fraction * next, anchor - fraction * next + next);
    }

    panTime(seconds) {
      const { start, end } = this.viewWindow();
      this.setTimeView(start + seconds, end + seconds);
    }

    zoomAnchor() {
      const current = this.currentTime();
      const { start, end } = this.viewWindow();
      if (current >= start && current <= end && (this.playing || current > 0)) return current;
      if (this.region) return (this.region.start + this.region.end) / 2;
      return (start + end) / 2;
    }

    fitView() {
      this.view = null;
      this.freqView = null;
      this.updateControls();
      this.draw();
    }

    timeAt(clientX) {
      const bounds = this.area.getBoundingClientRect();
      const { start, end } = this.viewWindow();
      return start + A.clamp((clientX - bounds.left) / Math.max(1, bounds.width), 0, 1) * (end - start);
    }

    /* Frequency view --------------------------------------------------------- */
    maxHz() {
      return Math.min(this.settings.maxHz, this.sampleRate / 2);
    }

    /* Axis units in [0, 1] over [0, maxHz]: linear in Hz, or in log Hz. */
    toAxis(hz) {
      const max = this.maxHz();
      if (this.settings.scale !== "log") return hz / max;
      return (Math.log(Math.max(hz, LOG_MIN_HZ)) - Math.log(LOG_MIN_HZ)) / (Math.log(max) - Math.log(LOG_MIN_HZ));
    }

    fromAxis(unit) {
      const max = this.maxHz();
      if (this.settings.scale !== "log") return unit * max;
      return Math.exp(Math.log(LOG_MIN_HZ) + unit * (Math.log(max) - Math.log(LOG_MIN_HZ)));
    }

    freqWindow() {
      const low = this.freqView ? this.freqView.low : 0;
      const high = this.freqView ? this.freqView.high : 1;
      return { low, high, lowHz: low <= 0 ? 0 : this.fromAxis(low), highHz: this.fromAxis(high) };
    }

    setFreqView(low, high) {
      const span = A.clamp(high - low, MIN_FREQ_AXIS, 1);
      if (span >= 1 - 1e-9) this.freqView = null;
      else {
        const first = A.clamp(low, 0, 1 - span);
        this.freqView = { low: first, high: first + span };
      }
      this.updateControls();
      this.draw();
    }

    clampFreqView() {
      if (this.freqView) this.setFreqView(this.freqView.low, this.freqView.high);
    }

    zoomFreq(factor, anchor = null) {
      const { low, high } = this.freqWindow();
      const center = anchor ?? (low + high) / 2;
      const fraction = high > low ? A.clamp((center - low) / (high - low), 0, 1) : 0.5;
      const next = (high - low) * factor;
      this.setFreqView(center - fraction * next, center - fraction * next + next);
    }

    /* Axis position under the pointer, if it is over a spectrogram. */
    freqAxisAt(event) {
      const pane = event.target.closest?.(".audio-spec-pane, .audio-diff-pane");
      if (!pane) return null;
      const bounds = pane.getBoundingClientRect();
      const { low, high } = this.freqWindow();
      return low + (1 - A.clamp((event.clientY - bounds.top) / bounds.height, 0, 1)) * (high - low);
    }

    /* Pointer --------------------------------------------------------------- */
    wheel(event) {
      if (!this.duration) return;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        this.zoomTime(event.deltaY > 0 ? 1.25 : 0.8, this.timeAt(event.clientX));
      } else if (event.altKey) {
        event.preventDefault();
        this.zoomFreq(event.deltaY > 0 ? 1.25 : 0.8, this.freqAxisAt(event));
      } else if (this.view && (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY))) {
        event.preventDefault();
        const delta = event.shiftKey ? event.deltaY || event.deltaX : event.deltaX;
        const { start, end } = this.viewWindow();
        this.panTime(delta / Math.max(1, this.area.clientWidth) * (end - start));
      }
    }

    regionEdgeAt(clientX) {
      if (!this.region) return null;
      const bounds = this.area.getBoundingClientRect();
      const { start, end } = this.viewWindow();
      const x = (time) => bounds.left + (time - start) / (end - start) * bounds.width;
      if (Math.abs(clientX - x(this.region.start)) <= EDGE_PIXELS) return "start";
      if (Math.abs(clientX - x(this.region.end)) <= EDGE_PIXELS) return "end";
      return null;
    }

    pointerDown(event) {
      if (!this.duration || event.button !== 0 || event.target.closest(".deck-lane-label, .deck-tscroll")) return;
      CompareDeck.focused = this;
      const originX = event.clientX;
      const clicked = this.timeAt(originX);
      if (event.shiftKey) {
        // Extend: from the selection's far edge, or from the playhead.
        const anchor = this.region
          ? (Math.abs(clicked - this.region.start) > Math.abs(clicked - this.region.end) ? this.region.start : this.region.end)
          : this.currentTime();
        if (Math.abs(clicked - anchor) >= MIN_REGION_SECONDS) this.setRegion({ start: Math.min(anchor, clicked), end: Math.max(anchor, clicked) });
        return;
      }
      const edge = this.regionEdgeAt(originX);
      const fixed = edge === "start" ? this.region.end : edge === "end" ? this.region.start : clicked;
      let dragging = Boolean(edge);
      const move = (moveEvent) => {
        if (!dragging && Math.abs(moveEvent.clientX - originX) < DRAG_PIXELS) return;
        dragging = true;
        const time = this.timeAt(moveEvent.clientX);
        this.region = { start: Math.min(fixed, time), end: Math.max(fixed, time) };
        this.drawRegion();
        this.updateRegionLabel();
      };
      const release = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("pointercancel", cancel);
      };
      const settle = () => {
        if (this.region.end - this.region.start < MIN_REGION_SECONDS) {
          this.clearRegion({ keepView: true });
          return;
        }
        this.setRegion(this.region);
      };
      const up = (upEvent) => {
        release();
        if (!dragging) {
          this.seekTo(this.timeAt(upEvent.clientX));
          return;
        }
        settle();
      };
      // The browser took the gesture over (a touch that scrolls the page): no click happened.
      const cancel = () => {
        release();
        if (dragging) settle();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      window.addEventListener("pointercancel", cancel);
    }

    /* Drag a scrollbar thumb, or click its track to jump there. */
    scrollbarDown(event, axis) {
      if (!this.duration || event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      const track = axis === "time" ? this.timeScroll : this.freqTrack;
      const bounds = track.getBoundingClientRect();
      const along = (pointer) => (axis === "time" ? (pointer.clientX - bounds.left) / bounds.width : 1 - (pointer.clientY - bounds.top) / bounds.height);
      const current = () => {
        if (axis === "time") {
          const { start, end } = this.viewWindow();
          return { low: start / this.duration, high: end / this.duration };
        }
        return this.freqWindow();
      };
      const apply = (low, high) => {
        if (axis === "time") this.setTimeView(low * this.duration, high * this.duration);
        else this.setFreqView(low, high);
      };
      const { low, high } = current();
      const span = high - low;
      const grab = along(event);
      const onThumb = grab >= low && grab <= high;
      const offset = onThumb ? grab - low : span / 2;
      if (!onThumb) apply(grab - offset, grab - offset + span);
      const move = (moveEvent) => {
        const position = along(moveEvent) - offset;
        apply(position, position + span);
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("pointercancel", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      window.addEventListener("pointercancel", up);
    }

    /* Region, loop and zoom ------------------------------------------------- */
    setRegion(region) {
      const duration = this.duration;
      this.region = { start: A.clamp(region.start, 0, duration), end: A.clamp(region.end, 0, duration) };
      const resume = this.playing;
      this.loop = true;
      this.offset = this.region.start;
      this.finishPlayback();
      this.updateControls();
      this.draw();
      if (resume) this.play();
    }

    clearRegion({ keepView = false } = {}) {
      const resume = this.playing;
      const position = this.currentTime();
      this.region = null;
      this.loop = false;
      if (!keepView) {
        this.view = null;
        this.freqView = null;
      }
      this.offset = position;
      this.finishPlayback();
      this.updateControls();
      this.draw();
      if (resume) this.play();
    }

    toggleLoop() {
      if (!this.region) return;
      const resume = this.playing;
      const position = this.currentTime();
      this.loop = !this.loop;
      this.offset = position;
      this.finishPlayback();
      this.updateControls();
      if (resume) this.play();
    }

    /* Zoom to the selection; again, back to the whole clip. */
    toggleZoom() {
      if (!this.region) {
        if (this.view) this.setTimeView(0, this.duration);
        return;
      }
      const pad = (this.region.end - this.region.start) * 0.08;
      const start = Math.max(0, this.region.start - pad);
      const end = Math.min(this.duration, this.region.end + pad);
      const { start: viewStart, end: viewEnd } = this.viewWindow();
      if (this.view && Math.abs(viewStart - start) < 1e-6 && Math.abs(viewEnd - end) < 1e-6) this.setTimeView(0, this.duration);
      else this.setTimeView(start, end);
    }

    /* The audible track's selection as WAV: cut from the file itself when it
     * is 16-bit PCM (so a 16 kHz output stays 16 kHz), else from the decoded
     * audio at the browser's rate. */
    exportRegion() {
      const track = this.selectedTrack();
      if (!track?.buffer || !this.region) return;
      const { start, end } = this.region;
      const sliced = A.sliceWav(track.bytes, start, end);
      let bytes;
      if (sliced) bytes = sliced.bytes;
      else {
        const rate = track.buffer.sampleRate;
        bytes = A.encodeWav(track.samples.subarray(Math.floor(start * rate), Math.ceil(end * rate)), rate);
      }
      const name = `${String(track.label).replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "") || "track"}_${start.toFixed(2)}-${end.toFixed(2)}s.wav`;
      const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    updateRegionLabel() {
      this.regionLabel.textContent = this.region
        ? `${A.formatTime(this.region.start)} – ${A.formatTime(this.region.end)} · ${(this.region.end - this.region.start).toFixed(3)} s`
        : t("Drag to select · Shift + click extends");
    }

    /* The audible track, whole, as a new file on the Annotate screen. */
    async sendToAnnotate() {
      const track = this.selectedTrack();
      const annotate = window.PureSoundAnnotate;
      if (!track?.buffer || !annotate) return;
      const bytes = track.bytes || A.encodeWav(track.samples, track.buffer.sampleRate);
      // The name is what windows.json keys by, so it must not follow the page's
      // language or carry the audio extension twice.
      const label = String(track.label || "").replace(/\.(wav|wave|flac|mp3|ogg|oga|opus|m4a|mp4|aac|aif|aiff|aifc|caf|webm)$/i, "");
      const stem = label.replace(/[^\p{L}\p{N}_.-]+/gu, "_").replace(/^_+|_+$/g, "") || track.id;
      this.stop();
      await annotate.addAudio({ name: `${stem}.wav`, bytes: new Blob([bytes], { type: "audio/wav" }) });
      window.PureSoundShell?.show("annotate");
    }

    updateControls() {
      const ready = this.duration > 0;
      this.playButton.disabled = !ready;
      this.stopButton.disabled = !ready;
      this.autoButton.disabled = !ready;
      this.empty.hidden = ready || this.tracks.some((track) => track.lane?.classList.contains("is-loading"));
      this.durationClock.textContent = A.formatTime(this.duration);
      this.loopButton.disabled = !this.region;
      this.exportButton.disabled = !this.region;
      this.annotateButton.hidden = !window.PureSoundAnnotate;
      this.annotateButton.disabled = !this.selectedTrack()?.buffer;
      this.clearButton.disabled = !this.region && !this.view && !this.freqView;
      this.zoomButton.disabled = !this.region && !this.view;
      this.root.querySelectorAll("[data-time-zoom]").forEach((button) => { button.disabled = !ready; });
      this.loopButton.setAttribute("aria-pressed", this.loop ? "true" : "false");
      this.loopButton.classList.toggle("is-active", this.loop);
      this.zoomButton.classList.toggle("is-active", Boolean(this.view));
      this.levelButton.hidden = this.tracks.length < 2;
      this.updateRegionLabel();
      this.tracks.forEach((track) => track.lane?.classList.toggle("is-selected", track.id === this.selectedId));
      this.updateLaneMeta();
      this.updateScrollbars();
      this.updatePosition();
    }

    updateScrollbars() {
      const duration = this.duration;
      const { start, end } = this.viewWindow();
      const zoomedTime = Boolean(this.view) && duration > 0;
      this.timeScroll.classList.toggle("is-zoomed", zoomedTime);
      this.timeThumb.style.left = `${duration ? start / duration * 100 : 0}%`;
      this.timeThumb.style.width = `${duration ? (end - start) / duration * 100 : 100}%`;
      const { low, high, lowHz, highHz } = this.freqWindow();
      this.freqScroll.classList.toggle("is-zoomed", Boolean(this.freqView));
      this.freqThumb.style.bottom = `${low * 100}%`;
      this.freqThumb.style.height = `${(high - low) * 100}%`;
      this.freqLabel.textContent = `${A.formatHz(lowHz)}–${A.formatHz(highHz)}`;
      this.freqLabel.title = `${Math.round(lowHz)}–${Math.round(highHz)} Hz shown`;
    }

    updatePosition() {
      const current = this.currentTime();
      const { start, end } = this.viewWindow();
      this.currentClock.textContent = A.formatTime(current);
      const fraction = end > start ? (current - start) / (end - start) : 0;
      this.playhead.hidden = fraction < 0 || fraction > 1;
      this.playhead.style.left = `${A.clamp(fraction, 0, 1) * 100}%`;
    }

    /* Drawing --------------------------------------------------------------- */
    setView(mode) {
      this.viewMode = mode;
      this.stage.classList.remove("is-wave", "is-both", "is-spec");
      this.stage.classList.add(`is-${mode}`);
      this.root.querySelectorAll("[data-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.view === mode));
      requestAnimationFrame(() => this.draw());
    }

    drawRegion() {
      const { start, end } = this.viewWindow();
      if (!this.region || !(end > start)) {
        this.regionOverlay.hidden = true;
      } else {
        const left = A.clamp((this.region.start - start) / (end - start), 0, 1);
        const right = A.clamp((this.region.end - start) / (end - start), 0, 1);
        this.regionOverlay.hidden = right <= left;
        this.regionOverlay.style.left = `${left * 100}%`;
        this.regionOverlay.style.width = `${(right - left) * 100}%`;
      }
      A.drawRuler(this.ruler, start, end, { region: this.region });
    }

    spectrogramOptions(extra = {}) {
      const { lowHz, highHz } = this.freqWindow();
      return { settings: this.settings, minHz: lowHz, maxHz: highHz, ...extra };
    }

    draw() {
      const { start, end } = this.viewWindow();
      this.drawRegion();
      this.tracks.forEach((track) => {
        if (!track.waveCanvas || track.lane.hidden) return;
        const rate = track.buffer?.sampleRate || this.sampleRate;
        const startSample = Math.floor(start * rate);
        const endSample = Math.ceil(end * rate);
        A.drawWave(track.waveCanvas, track.samples, track.color, { startSample, endSample, scale: this.settings.waveScale, gain: this.settings.waveGain });
        if (this.viewMode !== "wave") track.spectrogram.render(track.samples, rate, start, end, this.spectrogramOptions());
      });
      if (this.analysisOn) this.drawAnalysis();
      this.drawCurves();
      this.updateScrollbars();
      this.updatePosition();
    }

    levelTrace(track) {
      if (!track?.samples) return null;
      if (!this.levelTraces.has(track.id)) this.levelTraces.set(track.id, A.frameLevels(track.samples, track.buffer.sampleRate, LEVEL_FRAME_SECONDS));
      return this.levelTraces.get(track.id);
    }

    /* The selected track against the reference: level per frame and the
     * spectral difference.  Selecting the reference itself shows why empty. */
    drawAnalysis() {
      const reference = this.referenceTrack();
      const target = this.selectedTrack();
      const { start, end } = this.viewWindow();
      const { context, width, height, ratio } = A.fitCanvas(this.levelCanvas);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#010120";
      context.fillRect(0, 0, width, height);
      const same = !reference || !target?.buffer || target === reference;
      this.levelLabel.textContent = same
        ? t("select a track other than {track} to compare", { track: reference ? t(reference.label) : t("the reference") })
        : `${t(target.label)} − ${t(reference.label)} · ${t("20 ms frames · speech, noise and reverb together")}`;
      this.diffLabel.textContent = same
        ? t("cool = removed · warm = added · ±30 dB")
        : `${t(target.label)} vs ${t(reference.label)} · ${t("cool = removed · warm = added · ±30 dB")}`;
      const zero = height / 2;
      context.strokeStyle = "rgba(255,255,255,.18)";
      context.beginPath(); context.moveTo(0, zero); context.lineTo(width, zero); context.stroke();
      context.fillStyle = "rgba(255,255,255,.4)";
      context.font = `${8 * ratio}px ui-monospace, monospace`;
      context.textBaseline = "top";
      context.fillText(`+${LEVEL_RANGE_DB}`, width - 26 * ratio, 3 * ratio);
      context.textBaseline = "bottom";
      context.fillText(`−${LEVEL_RANGE_DB}`, width - 26 * ratio, height - 2 * ratio);
      if (same || !(end > start)) {
        this.diffView.render(null, 1, 0, 0);
        return;
      }
      const base = this.levelTrace(reference);
      const other = this.levelTrace(target);
      const frames = Math.min(base.levels.length, other.levels.length);
      const firstFrame = Math.max(0, Math.floor(start / base.frameSeconds));
      const lastFrame = Math.min(frames, Math.ceil(end / base.frameSeconds));
      const xOf = (frame) => ((frame + 0.5) * base.frameSeconds - start) / (end - start) * width;
      for (let frame = firstFrame; frame < lastFrame; frame += 1) {
        const x = xOf(frame);
        const barWidth = Math.max(1, base.frameSeconds / (end - start) * width);
        if (base.levels[frame] < LEVEL_SILENCE_DBFS) {
          context.fillStyle = "rgba(255,255,255,.05)";
          context.fillRect(x - barWidth / 2, 0, barWidth, height);
          continue;
        }
        const change = A.clamp(other.levels[frame] - base.levels[frame], -LEVEL_RANGE_DB, LEVEL_RANGE_DB);
        const y = zero - change / LEVEL_RANGE_DB * (height / 2);
        context.fillStyle = change < 0 ? "rgba(80,200,255,.75)" : "rgba(255,120,40,.8)";
        context.fillRect(x - barWidth / 2, Math.min(y, zero), barWidth, Math.max(1, Math.abs(y - zero)));
      }
      this.diffView.render(target.samples, target.buffer.sampleRate, start, end, this.spectrogramOptions({ reference: reference.samples, referenceKey: `${reference.id}>${target.id}` }));
    }

    drawCurves() {
      if (!this.curves.length) return;
      const { start, end } = this.viewWindow();
      this.curves.forEach((curve, index) => {
        const canvas = this.curvesPanel.querySelector(`[data-curve="${index}"] canvas`);
        if (!canvas) return;
        const { context, width, height } = A.fitCanvas(canvas);
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#010120";
        context.fillRect(0, 0, width, height);
        const [low, high] = curve.range || [0, 1];
        const hop = curve.hopSeconds || 0.01;
        const offset = curve.offsetSeconds || 0;
        context.strokeStyle = "rgba(255,255,255,.12)";
        context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
        context.strokeStyle = curve.color || "#7ee0a1";
        context.lineWidth = 1.5;
        context.beginPath();
        let started = false;
        curve.values.forEach((value, frame) => {
          const time = offset + frame * hop;
          if (time < start - hop || time > end + hop) return;
          if (!Number.isFinite(value)) { started = false; return; } // a gap, not a jump to the axis
          const x = (time - start) / (end - start) * width;
          const y = (1 - A.clamp((value - low) / (high - low || 1), 0, 1)) * (height - 4) + 2;
          if (started) context.lineTo(x, y);
          else { context.moveTo(x, y); started = true; }
        });
        context.stroke();
        context.lineWidth = 1;
      });
    }

    /* A readout under the pointer: time always; frequency and level over a
     * spectrogram, level over a waveform, change over the analysis lanes.
     * Near a selection edge the cursor offers to resize it. */
    pointerHover(event) {
      if (!this.duration) return;
      this.area.style.cursor = this.regionEdgeAt(event.clientX) ? "ew-resize" : "";
      const pane = event.target.closest?.(".audio-pane");
      const lane = pane?.closest(".deck-lane");
      if (!pane || !lane) { this.hover.hidden = true; return; }
      const bounds = pane.getBoundingClientRect();
      const xFraction = A.clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
      const yFraction = A.clamp((event.clientY - bounds.top) / bounds.height, 0, 1);
      const { start, end } = this.viewWindow();
      const seconds = start + xFraction * (end - start);
      const parts = [A.formatTime(seconds)];
      const track = this.track(lane.dataset.lane);
      if (lane.hasAttribute("data-analysis-spec")) {
        const value = this.diffView.valueAt(xFraction, yFraction);
        if (value) parts.push(`${Math.round(value.hz)} Hz`, `${value.db > 0 ? "+" : ""}${value.db.toFixed(1)} dB`);
      } else if (lane.hasAttribute("data-analysis-level")) {
        const reference = this.referenceTrack();
        const target = this.selectedTrack();
        if (reference && target && target !== reference) {
          const base = this.levelTrace(reference);
          const other = this.levelTrace(target);
          const frame = Math.floor(seconds / base.frameSeconds);
          if (frame < base.levels.length && frame < other.levels.length) {
            parts.push(`${reference.label} ${base.levels[frame].toFixed(1)} dBFS`, `${target.label} ${other.levels[frame].toFixed(1)} dBFS`, `Δ ${(other.levels[frame] - base.levels[frame]).toFixed(1)} dB`);
          }
        }
      } else if (track && pane.classList.contains("audio-spec-pane")) {
        const value = track.spectrogram.valueAt(xFraction, yFraction);
        if (value) parts.push(`${Math.round(value.hz)} Hz`, `${value.db.toFixed(1)} dB`);
      } else if (track?.samples) {
        const trace = this.levelTrace(track);
        const frame = Math.floor(seconds / trace.frameSeconds);
        if (frame < trace.levels.length) parts.push(`${trace.levels[frame].toFixed(1)} dBFS`);
      }
      const areaBounds = this.area.getBoundingClientRect();
      this.hover.textContent = parts.join(" · ");
      this.hover.hidden = false;
      const left = event.clientX - areaBounds.left + 12;
      this.hover.style.left = `${Math.max(0, Math.min(left, areaBounds.width - this.hover.offsetWidth - 4))}px`;
      this.hover.style.top = `${event.clientY - areaBounds.top + 12}px`;
    }

    destroy() {
      this.stop();
      this.resizeObserver.disconnect();
    }
  }

  CompareDeck.focused = null;
  window.PureSoundCompareDeck = CompareDeck;
})();
