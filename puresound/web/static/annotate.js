/* Annotate: mark spans on a recording (keep, suppress, double-talk, ...) and
 * export them for evaluation.  Everything happens in the browser: audio is
 * decoded here and never uploaded.  A folder opened read-write (Chromium)
 * can take <name>.spans.json files written next to its audio.
 *
 * The engine was the standalone tools/audio_annotator.html; its parsing and
 * export live in annotate-model.js (Node-tested byte for byte against it).
 * This file builds the screen's markup, draws the stage and handles input,
 * scoped to #screen-annotate. */
(() => {
  "use strict";

  const M = window.PureSoundAnnotateModel;
  const I = window.PureSoundI18n;
  const t = (key, vars) => (I ? I.t(key, vars) : key);
  const shell = () => window.PureSoundShell;
  const screen = document.getElementById("screen-annotate");
  if (!screen || !M) return;
  const mainHost = document.getElementById("annotate-main");
  const inspectorHost = document.getElementById("annotate-inspector");
  const helpHost = document.getElementById("help-annotate-body");

  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const db = (v) => 20 * Math.log10(Math.max(1e-7, v));
  const lin = (d) => Math.pow(10, d / 20);
  const fmt = (time) => {
    const total = isFinite(time) ? Math.round(Math.max(0, time) * 1000) : 0;
    const m = Math.floor(total / 60000), s = Math.floor(total / 1000) % 60, ms = total % 1000;
    return `${m}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
  };
  const toast = (message, isError = false) => shell()?.toast(message, isError);

  // ------------------------------------------------------------------ markup

  mainHost.innerHTML = `
    <div class="panel annotate-player">
      <div class="annotate-transport">
        <div class="annotate-group">
          <button class="button button-icon" type="button" data-an="play" disabled aria-label="Play" data-i18n-attr="aria-label">▶</button>
          <button class="button button-icon" type="button" data-an="stop" disabled aria-label="Stop" title="Stop" data-i18n-attr="aria-label,title">■</button>
          <button class="button button-icon" type="button" data-an="loop" disabled aria-pressed="false" aria-label="Loop the selection (L)" title="Loop the selection (L)" data-i18n-attr="aria-label,title">↻</button>
          <button class="button button-small" type="button" data-an="playSel" disabled title="Play the selection (⇧Space)" data-i18n-attr="title"><span data-i18n>Play selection</span></button>
          <span class="annotate-clock"><span data-an="cur">0:00.000</span> / <span data-an="dur">0:00.000</span></span>
        </div>
        <div class="annotate-group">
          <span class="annotate-label" data-i18n>Zoom</span>
          <input class="range" type="range" data-an="zoom" min="0" max="1000" step="1" value="0" disabled aria-label="Time zoom" title="Magnify along time — keys + and −, or the wheel over the waveform" data-i18n-attr="aria-label,title" />
          <output data-an="zoomV">1.0×</output>
          <button class="button button-small" type="button" data-an="zfit" disabled title="Fit the whole file (0)" data-i18n-attr="title"><span data-i18n>Fit</span></button>
          <button class="button button-small" type="button" data-an="zsel" disabled title="Zoom to the selection" data-i18n-attr="title"><span data-i18n>To selection</span></button>
        </div>
        <div class="annotate-group">
          <span class="annotate-label" title="Waveform display gain — drawing only, never the audio or the export" data-i18n-attr="title" data-i18n>Amp</span>
          <input class="range" type="range" data-an="amp" min="0" max="1000" step="1" value="0" aria-label="Waveform display gain" data-i18n-attr="aria-label" />
          <output data-an="ampV">1.0×</output>
        </div>
        <div class="annotate-group">
          <span class="annotate-label" title="Playback gain, with a limiter so it cannot clip" data-i18n-attr="title" data-i18n>Boost</span>
          <input class="range" type="range" data-an="boost" min="0" max="48" step="0.5" value="0" aria-label="Playback gain boost" data-i18n-attr="aria-label" />
          <output data-an="boostV">0.0 dB</output>
          <button class="button button-small" type="button" data-an="norm" disabled title="The largest boost this file takes without clipping (G)" data-i18n-attr="title"><span data-i18n>Auto</span></button>
          <span class="audio-limiter annotate-lim" data-an="lim" title="Limiter" data-i18n-attr="title">LIM</span>
        </div>
        <div class="segmented annotate-mode" role="group" aria-label="Panes" data-i18n-attr="aria-label">
          <button type="button" data-mode="wave" aria-pressed="true" data-i18n>Wave</button><button type="button" data-mode="both" aria-pressed="false" data-i18n>Both</button><button type="button" data-mode="spec" aria-pressed="false" data-i18n>Spec</button>
        </div>
      </div>
      <div class="annotate-meta"><span data-an="mFile"></span><span data-an="mFmt"></span><span data-an="mLevel"></span><span data-an="mSel" class="is-selection"></span><span data-an="mView"></span></div>
      <div class="annotate-stage" data-an="stage">
        <div class="annotate-pane is-ruler" data-an="rulerPane"><canvas data-an="ruler"></canvas></div>
        <div class="annotate-pane is-wave" data-an="wavePane"><span class="annotate-pane-tag" data-an="waveTag"></span><canvas data-an="wave"></canvas></div>
        <div class="annotate-gutter" data-an="gSpec" hidden title="Drag to resize · double-click to reset" data-i18n-attr="title"></div>
        <div class="annotate-pane is-spec" data-an="specPane" hidden><span class="annotate-pane-tag" data-i18n>Spectrogram · 0–8 kHz</span><canvas data-an="spec"></canvas></div>
        <div class="annotate-gutter" data-an="gLane" title="Drag to resize · double-click to reset" data-i18n-attr="title"></div>
        <div class="annotate-pane is-lane" data-an="lanePane"><span class="annotate-pane-tag" data-i18n>Spans · drag the edges to adjust, the body to move</span><canvas data-an="lane"></canvas></div>
        <div class="annotate-hscroll is-off" data-an="hscroll" title="Drag to scroll the timeline" data-i18n-attr="title"><div class="annotate-hthumb" data-an="hthumb"></div></div>
        <div class="annotate-drop" data-an="drop">
          <div>
            <strong data-i18n>Drop audio, a folder, or a span file</strong>
            <p data-i18n>Drag a range on the waveform, tag it, and export every file's spans as JSON, CSV, or an evaluation windows.json.</p>
            <div class="annotate-drop-actions"><button class="button button-secondary" type="button" data-act="openFiles"><span data-i18n>Open audio</span></button><button class="button button-secondary" type="button" data-act="openFolder"><span data-i18n>Open folder</span></button></div>
            <small data-i18n>wav · flac · mp3 · ogg · m4a — decoded in this browser, nothing is uploaded. Span files next to the audio (.spans.json, csv, Audacity labels) are picked up.</small>
          </div>
        </div>
        <div class="annotate-busy" data-an="busy" hidden></div>
      </div>
    </div>
    <div class="panel panel-pad annotate-spans">
      <div class="panel-title">
        <div class="annotate-spans-title"><h2><span data-i18n>Spans</span> <span class="annotate-count" data-an="count">0</span></h2><span class="quiet" data-an="totalCount"></span></div>
        <div class="annotate-spans-actions"><span class="annotate-newtag"><span data-i18n>New spans</span> <span class="chip annotate-tag-chip" data-an="curTag"></span></span><button class="button button-ghost button-small" type="button" data-an="clearAll"><span data-i18n>Clear all</span></button></div>
      </div>
      <div class="annotate-table-wrap" data-an="tblwrap">
        <div class="empty-state" data-an="empty"><strong data-i18n>No spans yet</strong><span data-i18n>Drag a range across the waveform, then press Enter.</span></div>
        <table class="table annotate-table" data-an="tbl" hidden>
          <thead><tr><th class="num">#</th><th data-i18n>Tag</th><th class="num" data-i18n>Start</th><th class="num" data-i18n>End</th><th class="num" data-i18n>Length</th><th data-i18n>Label</th><th></th></tr></thead>
          <tbody data-an="tbody"></tbody>
        </table>
      </div>
    </div>
    <input type="file" data-an="file" accept="audio/*,.wav,.flac,.mp3,.ogg,.m4a" multiple hidden />
    <input type="file" data-an="folder" webkitdirectory directory multiple hidden />
    <input type="file" data-an="spanfile" accept=".json,.csv,.txt,.tsv" multiple hidden />
    <dialog class="dialog" data-an="ask" aria-labelledby="annotate-ask-title"><div class="dialog-head"><h2 data-an="askTitle" id="annotate-ask-title"></h2></div><div class="dialog-body"><p data-an="askMsg"></p><div class="annotate-ask-actions" data-an="askBtns"></div></div></dialog>`;

  inspectorHost.innerHTML = `
    <section class="inspector-section">
      <div class="annotate-section-head"><p class="eyebrow" data-i18n>Files</p><span class="quiet" data-an="folderName"></span><span class="annotate-count" data-an="fcount">0</span></div>
      <div class="annotate-files" data-an="filelist"></div>
      <div class="annotate-file-actions">
        <button class="button button-secondary button-small" type="button" data-act="openFiles" data-inspector-focus><span data-i18n>Open audio</span></button>
        <button class="button button-secondary button-small" type="button" data-act="openFolder" title="Read-write where the browser allows it, so spans can be saved next to the audio" data-i18n-attr="title"><span data-i18n>Open folder</span></button>
        <button class="button button-secondary button-small" type="button" data-act="importSpans"><span data-i18n>Import spans</span></button>
        <button class="button button-secondary button-small" type="button" data-act="saveSidecars" title="Write <name>.spans.json next to each audio file" data-i18n-attr="title"><span data-i18n>Save into folder</span></button>
      </div>
      <div class="annotate-file-actions is-quiet">
        <button class="button button-ghost button-small" type="button" data-act="openFolderInput"><span data-i18n>Open folder read-only</span></button>
        <button class="button button-ghost button-small" type="button" data-act="closeFile"><span data-i18n>Remove this file</span></button>
        <button class="button button-ghost button-small" type="button" data-act="closeAll"><span data-i18n>Close all</span></button>
      </div>
    </section>
    <section class="inspector-section">
      <p class="eyebrow" data-i18n>Tag for new spans</p>
      <div class="annotate-tags" data-an="tagrow"></div>
      <form class="annotate-newtag-form" data-an="newtagForm"><input class="control" data-an="newtag" maxlength="24" placeholder="Add a tag…" aria-label="New tag name" data-i18n-attr="placeholder,aria-label" /><button class="button button-secondary" type="submit"><span data-i18n>Add</span></button></form>
    </section>
    <section class="inspector-section">
      <p class="eyebrow" data-i18n>Export</p>
      <div class="segmented annotate-formats" data-an="fmt" role="group" aria-label="Export format" data-i18n-attr="aria-label"><button type="button" data-f="json" aria-pressed="true">JSON</button><button type="button" data-f="csv" aria-pressed="false">CSV</button><button type="button" data-f="windows" aria-pressed="false">windows.json</button><button type="button" data-f="audacity" aria-pressed="false">Audacity</button></div>
      <p class="toggle-hint" data-an="scope"></p>
      <textarea class="control annotate-out" data-an="out" readonly spellcheck="false" aria-label="Exported spans" data-i18n-attr="aria-label"></textarea>
      <div class="annotate-file-actions"><button class="button button-secondary button-small" type="button" data-act="copy"><span data-i18n>Copy</span></button><button class="button button-secondary button-small" type="button" data-act="preview"><span data-i18n>Open in a new tab</span></button></div>
    </section>
    <section class="inspector-section">
      <p class="eyebrow" data-i18n>Playback</p>
      <div class="field-label range-label"><label for="annotate-rate" data-i18n>Rate</label><output data-an="rateV">1.00×</output></div>
      <input class="range" id="annotate-rate" type="range" data-an="rate" min="0.25" max="2" step="0.05" value="1" />
      <div class="field-label range-label"><label for="annotate-vol" data-i18n>Volume</label><output data-an="volV">100%</output></div>
      <input class="range" id="annotate-vol" type="range" data-an="vol" min="0" max="1" step="0.01" value="1" />
    </section>`;

  const KEYS = [
    ["Transport", [["Space", "play / pause"], ["⇧ Space", "play the selection"], ["L", "loop the selection"], ["← →", "nudge 1 s (⇧ 0.1 s)"]]],
    ["Spans", [["Enter", "add a span from the selection"], ["1…9", "pick a tag"], ["S", "start at the playhead"], ["E", "end at the playhead"], ["⌫", "delete the selected span"], ["Esc", "clear the selection"]]],
    ["View", [["+ − 0", "zoom in / out / fit"], ["wheel", "zoom at the pointer"], ["⇧ wheel", "scroll the timeline"]]],
    ["Files", [["[ ]", "previous / next file"], ["⌘/Ctrl O", "open audio"], ["⇧ ⌘/Ctrl O", "open a folder"], ["⌘/Ctrl I", "import spans"], ["⌘/Ctrl S", "save spans into the folder"], ["⌘/Ctrl E", "download the export"]]],
    ["Level", [["G", "auto boost (clip-free)"]]],
  ];
  helpHost.innerHTML = `
    <p data-i18n>Open a recording, drag a range on the waveform or spectrogram, and press Enter to turn it into a span with the current tag. Spans can be moved and resized in the lane under the waveform, retagged and labelled in the table.</p>
    <p data-i18n>windows.json is what evaluation reads: tags containing keep, near or double become keep windows (speech that must survive); tags containing sup or far become suppress windows. Other tags are left out of it.</p>
    <p data-i18n>Nothing is uploaded. A folder opened read-write (Chromium browsers) can take the spans back as &lt;name&gt;.spans.json next to each file; anywhere else, download the export.</p>
    <h3 data-i18n>Keyboard</h3>
    <p class="quiet" data-i18n>Keys work while Annotate is on screen and you are not typing in a field.</p>
    <div class="annotate-keys">${KEYS.map(([group, rows]) => `<p class="eyebrow" data-i18n>${group}</p>${rows.map(([key, text]) => `<kbd>${esc(key)}</kbd><span data-i18n>${esc(text)}</span>`).join("")}`).join("")}</div>`;
  I?.apply(mainHost);
  I?.apply(inspectorHost);
  I?.apply(helpHost);

  const q = (name) => screen.querySelector(`[data-an="${name}"]`);
  const qa = (selector) => [...screen.querySelectorAll(selector)];
  const AUDIO_BUTTONS = ["play", "stop", "loop", "playSel", "zfit", "zsel", "norm", "zoom"];

  // ------------------------------------------------------------------- state

  const S = {
    files: [], cur: -1, folderName: "", dirHandle: null,
    buf: null, name: "", mono: null, sr: 0, mip: null, peak: 0,
    view: { a: 0, b: 1 }, sel: null, spans: [], selId: null, nextId: 1,
    tag: "near / keep",
    tags: [
      { name: "near / keep", color: "#5ec58a" }, { name: "far / suppress", color: "#ef2cc1" },
      { name: "double-talk", color: "#bdbbff" }, { name: "exclude", color: "#8b96a5" },
    ],
    playing: false, t: 0, loop: false, rate: 1, vol: 1, boostDb: 0, amp: 1, mode: "wave", specCache: null,
    resizing: false, format: "json",
    layout: { spec: 160, lane: 64 },
  };
  const DEFAULT_LAYOUT = { spec: 160, lane: 64 };
  let ctx = null, srcNode = null, boostNode = null, limNode = null, shaper = null, gainNode = null;
  let startedAt = 0, startedFrom = 0, raf = 0;

  const dur = () => (S.buf ? S.buf.duration : 0);
  const vA = () => S.view.a * dur(), vB = () => S.view.b * dur();
  const vSpan = () => Math.max(1e-6, vB() - vA());
  const tagColor = (name) => (S.tags.find((tag) => tag.name === name) || { color: "#8b96a5" }).color;
  const F = () => (S.cur >= 0 ? S.files[S.cur] : null);
  const stageStyle = () => getComputedStyle(q("stage"));
  const css = (name) => stageStyle().getPropertyValue(name).trim();

  // ----------------------------------------------------------------- dialogs

  function ask(title, message, buttons) {
    return new Promise((resolve) => {
      const dialog = q("ask");
      q("askTitle").textContent = title;
      q("askMsg").textContent = message;
      const row = q("askBtns");
      row.innerHTML = "";
      buttons.forEach((button) => {
        const element = document.createElement("button");
        element.type = "button";
        element.className = `button ${button.primary ? "button-primary" : "button-secondary"}`;
        element.textContent = button.label;
        element.onclick = () => { dialog.close(); resolve(button.value); };
        row.appendChild(element);
      });
      dialog.oncancel = (event) => { event.preventDefault(); dialog.close(); resolve(buttons[0].value); };
      dialog.showModal();
      (row.querySelector(".button-primary") || row.firstChild)?.focus();
    });
  }

  // ------------------------------------------------------------------ layout

  const LAYOUT_KEY = "puresound.annotate.layout";
  function loadLayout() {
    try { const value = JSON.parse(localStorage.getItem(LAYOUT_KEY)); if (value && typeof value === "object") Object.assign(S.layout, value); } catch { /* storage may be disabled */ }
  }
  function saveLayout() { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(S.layout)); } catch { /* storage may be disabled */ } }
  function applyLayout() {
    q("specPane").style.height = `${S.layout.spec}px`;
    q("lanePane").style.height = `${S.layout.lane}px`;
  }
  function makeGutter(element, { key, min, max }) {
    let start = 0, base0 = 0, active = false;
    element.addEventListener("pointerdown", (event) => {
      element.setPointerCapture(event.pointerId); element.classList.add("is-dragging"); active = true; S.resizing = true;
      start = event.clientY; base0 = S.layout[key]; event.preventDefault();
    });
    element.addEventListener("pointermove", (event) => {
      if (!active || !q("stage").clientHeight) return;
      S.layout[key] = clamp(Math.round(base0 - (event.clientY - start)), min(), max());
      applyLayout(); resize(); renderStage();
    });
    const end = () => { if (!active) return; active = false; element.classList.remove("is-dragging"); S.resizing = false; S.specCache = null; saveLayout(); resize(); render(); };
    element.addEventListener("pointerup", end);
    element.addEventListener("pointercancel", end);
    element.addEventListener("dblclick", () => { S.layout[key] = DEFAULT_LAYOUT[key]; applyLayout(); S.specCache = null; saveLayout(); resize(); render(); });
  }
  makeGutter(q("gSpec"), { key: "spec", min: () => 56, max: () => Math.max(60, q("stage").clientHeight - 190) });
  makeGutter(q("gLane"), { key: "lane", min: () => 40, max: () => Math.max(44, q("stage").clientHeight - 190) });

  // ----------------------------------------------------------- file registry

  let fileSeq = 1;
  const CACHE_MAX = 2;
  const decoded = new Map(); // id -> {buf, mono, mip, peak, sr}

  function addEntries(list) {
    let added = 0;
    const current = F();
    for (const item of list) {
      if (S.files.some((file) => file.path === item.path)) continue;
      S.files.push({ id: fileSeq++, name: item.name, path: item.path, file: item.file || null, handle: item.handle || null, dir: item.dir || null, spans: [], nextId: 1, sr: 0, duration: 0, peak: 0, probed: false, dirty: false });
      added++;
    }
    S.files.sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true, sensitivity: "base" }));
    // The sort moves entries; S.cur keeps naming the file on screen.
    if (current) S.cur = S.files.indexOf(current);
    return added;
  }
  const fileOf = async (entry) => entry.file || (entry.handle ? await entry.handle.getFile() : null);
  function syncSpans() { const file = F(); if (file) { file.spans = S.spans; file.nextId = S.nextId; } }

  async function selectFile(index, { keepView = false } = {}) {
    if (index < 0 || index >= S.files.length) return;
    syncSpans(); stopPlayback();
    S.cur = index;
    const entry = S.files[index];
    S.spans = entry.spans; S.nextId = entry.nextId; S.selId = null; S.sel = null; S.t = 0;
    if (!keepView) S.view = { a: 0, b: 1 };
    await ensureDecoded(entry);
    renderFileList(); render(); updateExport();
  }

  async function ensureDecoded(entry) {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    let data = decoded.get(entry.id);
    if (!data) {
      const busy = q("busy");
      busy.hidden = false;
      busy.textContent = t("Decoding {name}…", { name: entry.name });
      try {
        const file = await fileOf(entry);
        if (!file) throw new Error("no file");
        const bytes = await file.arrayBuffer();
        const srFile = headerRate(bytes); // before decodeAudioData detaches the buffer
        const buffer = await ctx.decodeAudioData(bytes);
        const n = buffer.length, channels = buffer.numberOfChannels, mono = new Float32Array(n);
        for (let c = 0; c < channels; c++) { const channel = buffer.getChannelData(c); for (let k = 0; k < n; k++) mono[k] += channel[k]; }
        if (channels > 1) for (let k = 0; k < n; k++) mono[k] /= channels;
        let peak = 0;
        for (let k = 0; k < n; k++) { const v = mono[k] < 0 ? -mono[k] : mono[k]; if (v > peak) peak = v; }
        data = { buf: buffer, mono, mip: buildMip(mono), peak, sr: buffer.sampleRate, srFile: srFile || buffer.sampleRate, ch: channels };
        decoded.set(entry.id, data);
        entry.sr = data.srFile; entry.duration = buffer.duration; entry.peak = peak; entry.ch = channels; entry.probed = true;
        // Keep only the most recent few: an AudioBuffer is 4 bytes per sample per channel.
        while (decoded.size > CACHE_MAX) { const key = decoded.keys().next().value; if (key === entry.id) break; decoded.delete(key); }
      } catch (error) {
        console.error(error);
        toast(t("Could not decode {name}", { name: entry.name }), true);
        busy.hidden = true;
        entry.broken = true;
        if (F() !== entry) return false; // another file was selected while this one decoded
        S.buf = null; S.mono = null; S.mip = null; S.peak = 0; S.sr = 0; S.name = entry.name; S.specCache = null;
        AUDIO_BUTTONS.forEach((id) => { q(id).disabled = true; });
        q("mFile").innerHTML = `<b>${esc(entry.name)}</b>`;
        q("mFmt").textContent = t("could not decode");
        q("mLevel").textContent = "";
        q("dur").textContent = fmt(0);
        resize();
        return false;
      }
      busy.hidden = true;
    }
    if (F() !== entry) return false; // another file was selected while this one decoded
    S.buf = data.buf; S.mono = data.mono; S.mip = data.mip; S.peak = data.peak; S.sr = data.sr; S.name = entry.name;
    S.specCache = null;
    q("drop").classList.add("is-gone");
    AUDIO_BUTTONS.forEach((id) => { q(id).disabled = false; });
    q("dur").textContent = fmt(data.buf.duration);
    q("cur").textContent = fmt(0);
    q("mFile").innerHTML = `<b>${esc(entry.path)}</b>`;
    renderFormat(data);
    resize(); updateLevelMeta(); syncZoomUI(); syncDownload();
    return true;
  }

  // Web Audio resamples to the context rate, so say which number is which.
  function renderFormat(data = S.cur >= 0 ? decoded.get(F()?.id) : null) {
    if (!data) { q("mFmt").textContent = ""; return; }
    const channels = data.ch === 1 ? t("mono") : data.ch === 2 ? t("stereo") : t("{n} ch", { n: data.ch });
    q("mFmt").textContent = `${data.srFile.toLocaleString()} Hz${data.srFile !== data.sr ? ` (${t("decoded {rate} kHz", { rate: Math.round(data.sr / 1000) })})` : ""} · ${channels} · ${fmt(data.buf.duration)}`;
  }

  /* The true rate off the WAV or FLAC header: decodeAudioData reports the
   * context rate, which would put a wrong sample_rate into every export. */
  function headerRate(bytes) {
    try {
      const view = new DataView(bytes);
      // FLAC: the 20-bit rate in STREAMINFO, right after the "fLaC" marker and its block header.
      if (view.byteLength >= 21 && view.getUint32(0, false) === 0x664c6143) return (view.getUint8(18) << 12) | (view.getUint8(19) << 4) | (view.getUint8(20) >> 4);
      if (view.byteLength < 44 || view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) return 0;
      let position = 12;
      while (position + 8 <= view.byteLength) {
        const id = view.getUint32(position, false), size = view.getUint32(position + 4, true);
        if (id === 0x666d7420) return view.getUint32(position + 12, true);
        position += 8 + size + (size & 1);
      }
    } catch { /* neither */ }
    return 0;
  }

  /* A min/max pyramid, so redrawing the waveform does not depend on the
   * file's length. */
  function buildMip(samples) {
    const B0 = 256, FACTOR = 4, levels = [];
    let n = Math.ceil(samples.length / B0);
    const mn = new Float32Array(n), mx = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let lo = 1, hi = -1;
      const s = i * B0, e = Math.min(samples.length, s + B0);
      for (let j = s; j < e; j++) { const v = samples[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
      if (lo > hi) { lo = hi = 0; }
      mn[i] = lo; mx[i] = hi;
    }
    levels.push({ bucket: B0, mn, mx });
    while (levels[levels.length - 1].mn.length > 4) {
      const previous = levels[levels.length - 1], pn = previous.mn.length, nn = Math.ceil(pn / FACTOR);
      const a = new Float32Array(nn), b = new Float32Array(nn);
      for (let i = 0; i < nn; i++) {
        let lo = 1, hi = -1;
        const s = i * FACTOR, e = Math.min(pn, s + FACTOR);
        for (let j = s; j < e; j++) { if (previous.mn[j] < lo) lo = previous.mn[j]; if (previous.mx[j] > hi) hi = previous.mx[j]; }
        if (lo > hi) { lo = hi = 0; }
        a[i] = lo; b[i] = hi;
      }
      levels.push({ bucket: previous.bucket * FACTOR, mn: a, mx: b });
    }
    return levels;
  }

  function renderFileList() {
    const list = q("filelist");
    q("fcount").textContent = S.files.length;
    q("folderName").textContent = S.folderName || "";
    if (!S.files.length) { list.innerHTML = `<p class="quiet annotate-files-empty">${esc(t("No files yet. Open audio or a folder, or drop them on the stage."))}</p>`; return; }
    list.innerHTML = S.files.map((file, index) => `<button type="button" class="annotate-file${index === S.cur ? " is-current" : ""}${file.dirty ? " is-dirty" : ""}" data-file="${index}" title="${esc(file.path)}"><span class="annotate-file-dot" aria-hidden="true"></span><span class="annotate-file-name">${esc(file.name)}</span><span class="annotate-file-count${file.spans.length ? " has-spans" : ""}">${file.spans.length || "·"}</span></button>`).join("");
    list.querySelector(".is-current")?.scrollIntoView({ block: "nearest" });
  }
  q("filelist").addEventListener("click", (event) => { const item = event.target.closest("[data-file]"); if (item) selectFile(Number(item.dataset.file)); });

  // ----------------------------------------------------------------- opening

  q("file").onchange = (event) => { intake([...event.target.files].map((file) => ({ name: file.name, path: file.name, file }))); event.target.value = ""; };
  q("folder").onchange = (event) => {
    const list = [...event.target.files].map((file) => ({ name: file.name, path: file.webkitRelativePath || file.name, file }));
    S.folderName = (list[0]?.path.split("/")[0]) || t("folder"); S.dirHandle = null;
    intake(list); event.target.value = "";
  };
  q("spanfile").onchange = (event) => { intakeSpanFiles([...event.target.files]); event.target.value = ""; };

  async function openFolder() {
    if (window.showDirectoryPicker) {
      try {
        const dir = await window.showDirectoryPicker({ mode: "readwrite" });
        const out = [];
        await scanDir(dir, "", out, { n: 0 });
        S.dirHandle = dir; S.folderName = dir.name;
        await intake(out);
      } catch (error) { if (error && error.name !== "AbortError") { console.error(error); toast(t("Could not open that folder"), true); } }
    } else q("folder").click();
  }
  async function scanDir(dir, prefix, out, guard) {
    for await (const [name, handle] of dir.entries()) {
      if (guard.n++ > 4000) return;
      if (handle.kind === "file") out.push({ name, path: prefix + name, handle, dir });
      else if (handle.kind === "directory") await scanDir(handle, `${prefix}${name}/`, out, guard);
    }
  }

  /* One way in for every source: file picker, folder, drag and drop, and
   * another screen's "Annotate". */
  async function intake(items, { select = null } = {}) {
    const audio = items.filter((item) => M.AUDIO_RE.test(item.name));
    // Only files that look like span files: a folder of unrelated .txt is not read.
    const names = new Set([...audio, ...S.files].map((file) => M.base(file.name).toLowerCase()));
    const spanFiles = items.filter((item) => !M.AUDIO_RE.test(item.name) && M.SPAN_FILE_RE.test(item.name)
      && (/\.spans\.(json|csv)$/i.test(item.name) || /windows\.json$/i.test(item.name) || names.has(M.base(item.name).toLowerCase())));
    if (!audio.length && !spanFiles.length) { toast(t("Nothing to open in there"), true); return; }
    const added = addEntries(audio);
    let auto = 0;
    for (const spanFile of spanFiles) {
      const text = await readText(spanFile);
      if (text == null) continue;
      auto += applyParsed(M.parseSpans(text, spanFile.name, S.tag), "merge", true);
    }
    renderFileList();
    const wanted = select ? S.files.findIndex((file) => file.path === select) : -1;
    if (wanted >= 0) await selectFile(wanted);
    else if (S.cur < 0 && S.files.length) await selectFile(0);
    else { renderFileList(); render(); updateExport(); }
    if (auto) toast(t(auto === 1 ? "{n} span read from the span files" : "{n} spans read from the span files", { n: auto }));
  }
  const readText = async (item) => { try { const file = await fileOf(item); return file ? await file.text() : null; } catch { return null; } };

  // Drag and drop, whole folders included.
  const drop = q("drop");
  const player = mainHost.querySelector(".annotate-player");
  ["dragenter", "dragover"].forEach((name) => player.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("is-gone"); drop.classList.add("is-over"); }));
  player.addEventListener("dragleave", (event) => { event.preventDefault(); drop.classList.remove("is-over"); if (S.buf) drop.classList.add("is-gone"); });
  player.addEventListener("drop", async (event) => {
    event.preventDefault(); drop.classList.remove("is-over"); if (S.buf) drop.classList.add("is-gone");
    const transfer = event.dataTransfer;
    const roots = [...transfer.items].filter((item) => item.kind === "file").map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry()).filter(Boolean); // before any await
    let items = [];
    if (roots.length) {
      for (const entry of roots) { if (entry.isDirectory) { S.folderName = entry.name; S.dirHandle = null; } await walkEntry(entry, "", items); }
    } else items = [...transfer.files].map((file) => ({ name: file.name, path: file.name, file }));
    if (items.length) intake(items);
  });
  function walkEntry(entry, prefix, out) {
    return new Promise((resolve) => {
      if (entry.isFile) { entry.file((file) => { out.push({ name: entry.name, path: prefix + entry.name, file }); resolve(); }, () => resolve()); return; }
      if (!entry.isDirectory) { resolve(); return; }
      const reader = entry.createReader(), all = [];
      const step = () => reader.readEntries(async (entries) => {
        if (!entries.length) { for (const child of all) await walkEntry(child, `${prefix}${entry.name}/`, out); resolve(); return; }
        all.push(...entries); step();
      }, () => resolve());
      step();
    });
  }

  // ------------------------------------------------------------ span import

  /* mode "merge" | "replace"; quiet only matches files already open. */
  function applyParsed(parsed, mode, quiet) {
    if (!parsed) return 0;
    let n = 0;
    for (const key in parsed) {
      const rows = parsed[key];
      if (!rows || !rows.length) continue;
      let entry = S.files.find((file) => M.base(file.name).toLowerCase() === key.toLowerCase()) || S.files.find((file) => M.base(file.path).toLowerCase() === key.toLowerCase());
      if (!entry) {
        if (quiet) continue;
        if (S.files.length === 1) entry = S.files[0];
        else if (F()) entry = F();
        else continue;
      }
      if (mode === "replace") { entry.spans = []; entry.nextId = 1; }
      for (const row of rows) {
        if (!(row.b > row.a)) continue;
        ensureTag(row.tag);
        entry.spans.push({ id: entry.nextId++, a: row.a, b: row.b, tag: row.tag, label: row.label || "" });
        n++;
      }
      entry.spans.sort((x, y) => x.a - y.a);
      entry.dirty = true;
      if (entry === F()) { S.spans = entry.spans; S.nextId = entry.nextId; }
    }
    if (n) { renderTags(); renderFileList(); render(); updateExport(); if (!quiet) toast(t(n === 1 ? "Imported {n} span" : "Imported {n} spans", { n })); }
    return n;
  }
  function ensureTag(name) {
    if (!name || S.tags.some((tag) => tag.name === name)) return;
    const palette = ["#c77dff", "#e2686a", "#d9a441", "#4fb4d8", "#b0879c", "#8fbf6a", "#e0836b", "#6fd3db"];
    S.tags.push({ name, color: palette[S.tags.length % palette.length] });
  }
  async function intakeSpanFiles(files) {
    if (!S.files.length) { toast(t("Open the audio first, then import its spans"), true); return; }
    let mode = "merge";
    if (S.files.some((file) => file.spans.length)) {
      mode = await ask(t("Import spans"), t("Some files already have spans. Replace them, or add the imported ones on top?"),
        [{ label: t("Cancel"), value: null }, { label: t("Add"), value: "merge" }, { label: t("Replace"), value: "replace", primary: true }]);
      if (!mode) return;
    }
    let n = 0;
    for (const file of files) { try { n += applyParsed(M.parseSpans(await file.text(), file.name, S.tag), mode); } catch (error) { console.error(error); } }
    if (!n) toast(t("No spans found in that file"), true);
  }

  // ------------------------------------------------ writing spans to the folder

  async function saveSidecars() {
    syncSpans();
    const annotated = S.files.filter((file) => file.spans.length);
    if (!annotated.length) { toast(t("No spans to save"), true); return; }
    if (!S.dirHandle) {
      await ask(t("The folder is read-only"), t("This folder was opened read-only, so spans cannot be written next to the audio. Open it again with Open folder in a Chromium browser, or download the export instead."), [{ label: t("OK"), value: 1, primary: true }]);
      return;
    }
    const writable = annotated.filter((file) => file.dir);
    if (!writable.length) { toast(t("Those files did not come from the chosen folder"), true); return; }
    const ok = await ask(t("Save spans into the folder"), t("Write {n} files named <name>.spans.json next to the audio in “{folder}”. Existing files with those names are overwritten.", { n: writable.length, folder: S.folderName }),
      [{ label: t("Cancel"), value: false }, { label: t("Write {n} files", { n: writable.length }), value: true, primary: true }]);
    if (!ok) return;
    try {
      if (S.dirHandle.queryPermission) {
        let permission = await S.dirHandle.queryPermission({ mode: "readwrite" });
        if (permission !== "granted") permission = await S.dirHandle.requestPermission({ mode: "readwrite" });
        if (permission !== "granted") { toast(t("Write permission was denied"), true); return; }
      }
      let n = 0;
      for (const file of writable) {
        const handle = await file.dir.getFileHandle(`${M.base(file.name)}.spans.json`, { create: true });
        const writer = await handle.createWritable();
        await writer.write(M.singleFileJson(file));
        await writer.close();
        file.dirty = false; n++;
      }
      renderFileList();
      toast(t(n === 1 ? "Wrote {n} span file" : "Wrote {n} span files", { n }));
    } catch (error) { console.error(error); toast(t("Could not write: {reason}", { reason: error.message || error.name }), true); }
  }

  // ---------------------------------------------------------------- playback

  // The last safety stage: unity below 0.7, a soft knee above, never a hard clip.
  const SOFT = (() => {
    const n = 2048, curve = new Float32Array(n), knee = 0.7;
    for (let i = 0; i < n; i++) { const x = i / (n - 1) * 2 - 1, ax = Math.abs(x); curve[i] = Math.sign(x) * (ax <= knee ? ax : knee + (1 - knee) * Math.tanh((ax - knee) / (1 - knee))); }
    return curve;
  })();
  function stopPlayback() {
    if (srcNode) { try { srcNode.onended = null; srcNode.stop(); } catch { /* already stopped */ } srcNode = null; }
    S.playing = false; setPlayLabel(); cancelAnimationFrame(raf); setLim(0);
  }
  function setPlayLabel() {
    q("play").textContent = S.playing ? "❚❚" : "▶";
    q("play").setAttribute("aria-label", t(S.playing ? "Pause" : "Play"));
  }
  function play(from, until) {
    if (!S.buf) return;
    if (ctx.state === "suspended") ctx.resume();
    stopPlayback();
    srcNode = ctx.createBufferSource(); srcNode.buffer = S.buf; srcNode.playbackRate.value = S.rate;
    boostNode = ctx.createGain(); boostNode.gain.value = lin(S.boostDb);
    limNode = ctx.createDynamicsCompressor();
    limNode.threshold.value = -1; limNode.knee.value = 0; limNode.ratio.value = 20; limNode.attack.value = 0.003; limNode.release.value = 0.1;
    shaper = ctx.createWaveShaper(); shaper.curve = SOFT; shaper.oversample = "4x";
    gainNode = ctx.createGain(); gainNode.gain.value = S.vol;
    srcNode.connect(boostNode).connect(limNode).connect(shaper).connect(gainNode).connect(ctx.destination);
    const looping = S.loop && S.sel;
    if (looping) { srcNode.loop = true; srcNode.loopStart = S.sel.a; srcNode.loopEnd = S.sel.b; }
    startedFrom = clamp(from, 0, dur()); startedAt = ctx.currentTime;
    if (until != null && !looping) srcNode.start(0, startedFrom, Math.max(0.01, until - startedFrom));
    else srcNode.start(0, startedFrom);
    srcNode.onended = () => { if (S.playing) { stopPlayback(); S.t = until != null ? until : dur(); render(); } };
    S.playing = true; setPlayLabel(); tick();
  }
  function tick() {
    if (!S.playing) return;
    let time = startedFrom + (ctx.currentTime - startedAt) * S.rate;
    if (S.loop && S.sel && time > S.sel.b) { const span = Math.max(1e-3, S.sel.b - S.sel.a); time = S.sel.a + ((time - S.sel.a) % span); }
    S.t = clamp(time, 0, dur()); q("cur").textContent = fmt(S.t);
    if (limNode && typeof limNode.reduction === "number") setLim(-limNode.reduction);
    drawOverlays(); raf = requestAnimationFrame(tick);
  }
  function setLim(reduction) {
    const element = q("lim");
    element.classList.toggle("is-active", reduction > 0.4);
    element.classList.toggle("is-warning", reduction <= 0.4 && willLimit());
    element.textContent = reduction > 0.4 ? `LIM ${reduction.toFixed(1)}` : "LIM";
  }
  // The limiter sits at −1 dBFS, so a clip-free target lands below it.
  const BOOST_MAX = 48, SAFE_DBFS = -1.5;
  const willLimit = () => !!S.buf && db(S.peak) + S.boostDb > -1;
  const safeBoost = () => SAFE_DBFS - db(S.peak);
  function updateLevelMeta() {
    if (!S.buf) { q("mLevel").textContent = ""; return; }
    const headroom = safeBoost();
    q("mLevel").textContent = `${t("peak {level} dBFS", { level: db(S.peak).toFixed(1) })} · ${headroom > 0.05 ? t("{db} dB free", { db: `+${headroom.toFixed(1)}` }) : t("no headroom")}`;
    setLim(0);
  }
  const togglePlay = () => (S.playing ? (stopPlayback(), render()) : play(S.t >= dur() - 0.01 ? 0 : S.t));
  q("play").onclick = togglePlay;
  q("stop").onclick = () => { stopPlayback(); S.t = 0; q("cur").textContent = fmt(0); render(); };
  q("playSel").onclick = () => { if (S.sel) play(S.sel.a, S.sel.b); };
  q("loop").onclick = () => { S.loop = !S.loop; q("loop").setAttribute("aria-pressed", S.loop ? "true" : "false"); q("loop").classList.toggle("is-on", S.loop); };
  q("rate").oninput = (event) => { S.rate = +event.target.value; q("rateV").textContent = `${S.rate.toFixed(2)}×`; if (srcNode) srcNode.playbackRate.value = S.rate; };
  q("vol").oninput = (event) => { S.vol = +event.target.value; q("volV").textContent = `${Math.round(S.vol * 100)}%`; if (gainNode) gainNode.gain.value = S.vol; };
  q("boost").oninput = (event) => setBoost(+event.target.value);
  function setBoost(value) {
    S.boostDb = clamp(value, 0, BOOST_MAX);
    q("boost").value = S.boostDb; q("boostV").textContent = `${S.boostDb.toFixed(1)} dB`;
    if (boostNode && ctx) boostNode.gain.setTargetAtTime(lin(S.boostDb), ctx.currentTime, 0.02);
    setLim(0);
  }
  q("norm").onclick = () => autoBoost();
  function autoBoost() {
    if (!S.buf) return;
    const want = safeBoost(), head = clamp(want, 0, BOOST_MAX);
    setBoost(head);
    if (head < 0.05) toast(t("Already at full scale: no clip-free boost is available"));
    else if (want > BOOST_MAX + 0.05) toast(t("Boost {db} dB (the slider's maximum): the peak lands at {peak} dBFS, still no limiting", { db: head.toFixed(1), peak: (db(S.peak) + head).toFixed(1) }));
    else toast(t("Boost {db} dB: the peak lands at {peak} dBFS, no limiting", { db: head.toFixed(1), peak: SAFE_DBFS }));
  }

  // ---------------------------------------------------------- canvas plumbing

  const cv = { ruler: q("ruler"), wave: q("wave"), spec: q("spec"), lane: q("lane") };
  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const key in cv) {
      const canvas = cv[key], rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr)); canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas._dpr = dpr; canvas._w = rect.width; canvas._h = rect.height;
    }
    if (!S.resizing) S.specCache = null;
  }
  new ResizeObserver(() => { if (!S.resizing) fitPanes(); resize(); renderStage(); }).observe(q("stage"));
  const xOf = (time) => (time - vA()) / vSpan();
  const tOf = (x) => vA() + x * vSpan();

  function drawRuler() {
    const c = cv.ruler, g = c.getContext("2d"), w = c._w, h = c._h;
    g.setTransform(c._dpr, 0, 0, c._dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.fillStyle = css("--an-bg2"); g.fillRect(0, 0, w, h);
    g.strokeStyle = css("--an-line"); g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
    if (!S.buf) return;
    const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((s) => vSpan() / s < 12) || 900;
    g.font = `10px ${css("--font-mono")}`; g.textBaseline = "middle";
    for (let time = Math.ceil(vA() / step) * step; time <= vB(); time += step) {
      const x = xOf(time) * w;
      g.strokeStyle = css("--an-line"); g.beginPath(); g.moveTo(x + 0.5, h - 7); g.lineTo(x + 0.5, h); g.stroke();
      g.fillStyle = css("--an-dim"); g.fillText(step < 1 ? time.toFixed(3) : fmt(time).replace(/\.000$/, ""), x + 4, h / 2 - 1);
    }
  }
  function drawWave() {
    const c = cv.wave, g = c.getContext("2d"), w = c._w, h = c._h;
    g.setTransform(c._dpr, 0, 0, c._dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.fillStyle = css("--an-bg"); g.fillRect(0, 0, w, h);
    if (!S.buf) return;
    const mid = h / 2, amp = S.amp;
    g.strokeStyle = css("--an-line-soft"); g.beginPath();
    [0.25, 0.5, 0.75].forEach((f) => { g.moveTo(0, h * f + 0.5); g.lineTo(w, h * f + 0.5); }); g.stroke();
    const i0 = vA() * S.sr, i1 = vB() * S.sr, per = (i1 - i0) / w, m = S.mono;
    let level = null;
    if (S.mip) for (const L of S.mip) { if (L.bucket <= per) level = L; else break; }
    g.fillStyle = css("--an-wave");
    for (let px = 0; px < w; px++) {
      let lo = 1, hi = -1;
      if (level) {
        const s = Math.max(0, Math.floor((i0 + px * per) / level.bucket));
        const e = Math.min(level.mn.length, Math.max(s + 1, Math.ceil((i0 + (px + 1) * per) / level.bucket)));
        for (let i = s; i < e; i++) { if (level.mn[i] < lo) lo = level.mn[i]; if (level.mx[i] > hi) hi = level.mx[i]; }
      } else if (per < 1) {
        const v = m[clamp(Math.round(i0 + px * per), 0, m.length - 1)] || 0; lo = Math.min(0, v); hi = Math.max(0, v);
      } else {
        const s = Math.max(0, Math.floor(i0 + px * per)), e = Math.min(m.length, Math.ceil(i0 + (px + 1) * per));
        for (let i = s; i < e; i++) { const v = m[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      }
      if (lo > hi) { lo = hi = 0; }
      const y1 = mid - clamp(hi * amp, -1, 1) * mid * 0.94, y2 = mid - clamp(lo * amp, -1, 1) * mid * 0.94;
      g.fillRect(px, y1, 1, Math.max(1, y2 - y1));
    }
    g.strokeStyle = css("--an-line"); g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(w, mid + 0.5); g.stroke();
    // An amplified view says what full height now means, so nobody misreads the level.
    if (amp > 1.01) {
      g.strokeStyle = css("--an-guide"); g.setLineDash([3, 4]); g.globalAlpha = 0.55;
      g.beginPath(); g.moveTo(0, mid - mid * 0.94 + 0.5); g.lineTo(w, mid - mid * 0.94 + 0.5); g.moveTo(0, mid + mid * 0.94 - 0.5); g.lineTo(w, mid + mid * 0.94 - 0.5); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
    }
    q("waveTag").textContent = amp > 1.01 ? t("Waveform · ×{amp} — full height = {level} dBFS", { amp: amp.toFixed(1), level: db(1 / amp).toFixed(1) }) : t("Waveform");
  }
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const angle = -2 * Math.PI / len, wr = Math.cos(angle), wi = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci, vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi; re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }
  const N = 1024;
  const HANN = (() => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)); return a; })();
  function ramp(v) {
    const stops = [[1, 1, 32], [40, 52, 110], [38, 120, 124], [214, 150, 60], [248, 244, 235]];
    const p = clamp(v, 0, 1) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i, a = stops[i], b = stops[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function drawSpec() {
    const c = cv.spec, g = c.getContext("2d"), w = c._w, h = c._h;
    g.setTransform(c._dpr, 0, 0, c._dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.fillStyle = css("--an-bg"); g.fillRect(0, 0, w, h);
    if (!S.buf) return;
    // While a gutter is dragged, stretch the last image instead of recomputing.
    if (S.resizing && S.specCache) { g.drawImage(S.specCache.off, 0, 0, w, h); return; }
    const key = `${S.view.a.toFixed(6)}|${S.view.b.toFixed(6)}|${Math.round(w)}|${Math.round(h)}|${S.amp.toFixed(2)}|${S.name}`;
    if (!S.specCache || S.specCache.key !== key) {
      const cols = Math.min(1400, Math.max(180, Math.floor(w))), rows = Math.min(360, Math.max(120, Math.floor(h)));
      const image = new ImageData(cols, rows);
      const i0 = vA() * S.sr, step = (vB() - vA()) * S.sr / cols;
      const re = new Float32Array(N), im = new Float32Array(N), m = S.mono;
      const maxBin = Math.max(8, Math.min(N / 2, Math.floor(8000 / (S.sr / N))));
      const lift = db(S.amp); // the Amp control lifts the spectrogram floor too
      for (let x = 0; x < cols; x++) {
        const s = Math.floor(i0 + x * step - N / 2);
        for (let i = 0; i < N; i++) { const k = s + i; re[i] = (k >= 0 && k < m.length ? m[k] : 0) * HANN[i]; im[i] = 0; }
        fft(re, im);
        for (let y = 0; y < rows; y++) {
          const bin = Math.floor((1 - y / rows) * maxBin);
          const mag = Math.hypot(re[bin], im[bin]) / (N / 4);
          const [r, gg, b] = ramp((20 * Math.log10(mag + 1e-9) + 92 + lift) / 82);
          const o = (y * cols + x) * 4;
          image.data[o] = r; image.data[o + 1] = gg; image.data[o + 2] = b; image.data[o + 3] = 255;
        }
      }
      const off = document.createElement("canvas"); off.width = cols; off.height = rows;
      off.getContext("2d").putImageData(image, 0, 0);
      S.specCache = { key, off };
    }
    g.drawImage(S.specCache.off, 0, 0, w, h);
  }
  /* The span lane: the marked regions on the same time axis. */
  function drawLane() {
    const c = cv.lane, g = c.getContext("2d"), w = c._w, h = c._h;
    g.setTransform(c._dpr, 0, 0, c._dpr, 0, 0); g.clearRect(0, 0, w, h);
    g.fillStyle = css("--an-bg2"); g.fillRect(0, 0, w, h);
    if (!S.buf) return;
    const top = Math.min(20, h * 0.28), bh = Math.max(8, h - top - 9);
    g.strokeStyle = css("--an-line-soft"); g.beginPath(); g.moveTo(0, top - 0.5); g.lineTo(w, top - 0.5); g.stroke();
    for (const span of S.spans) {
      if (span.b < vA() || span.a > vB()) continue;
      const x1 = xOf(span.a) * w, x2 = xOf(span.b) * w, bw = Math.max(2, x2 - x1), color = tagColor(span.tag), on = span.id === S.selId;
      g.fillStyle = color; g.globalAlpha = on ? 0.42 : 0.24; g.fillRect(x1, top, bw, bh); g.globalAlpha = 1;
      g.strokeStyle = color; g.lineWidth = on ? 2 : 1; g.strokeRect(x1 + 0.5, top + 0.5, bw - 1, bh - 1);
      g.fillStyle = color; g.fillRect(x1, top, Math.min(4, bw), bh); g.fillRect(x2 - Math.min(4, bw), top, Math.min(4, bw), bh);
      if (bw > 46 && bh > 14) {
        g.font = `600 11px ${css("--font-sans")}`; g.fillStyle = css("--an-text");
        g.save(); g.beginPath(); g.rect(x1 + 7, top, bw - 14, bh); g.clip();
        g.fillText(span.label || span.tag, x1 + 8, top + bh / 2 + 4); g.restore();
      }
    }
    if (!S.spans.length) {
      g.fillStyle = css("--an-faint"); g.font = `11.5px ${css("--font-sans")}`;
      g.fillText(t("Spans you add appear here on the same time axis."), 10, Math.min(h - 6, h / 2 + 8));
    }
  }
  function drawOverlays() {
    [["wave", q("wavePane")], ["spec", q("specPane")], ["lane", q("lanePane")]].forEach(([key, pane]) => {
      if (pane.hidden || !pane.offsetHeight) return;
      const c = cv[key];
      let overlay = pane.querySelector("canvas.annotate-overlay");
      if (!overlay) { overlay = document.createElement("canvas"); overlay.className = "annotate-overlay"; pane.appendChild(overlay); }
      if (overlay.width !== c.width || overlay.height !== c.height) { overlay.width = c.width; overlay.height = c.height; }
      const g = overlay.getContext("2d"), w = c._w, h = c._h;
      g.setTransform(c._dpr, 0, 0, c._dpr, 0, 0); g.clearRect(0, 0, w, h);
      if (key !== "lane") {
        for (const span of S.spans) {
          if (span.b < vA() || span.a > vB()) continue;
          const x1 = xOf(span.a) * w, x2 = xOf(span.b) * w, color = tagColor(span.tag);
          g.fillStyle = color; g.globalAlpha = span.id === S.selId ? 0.2 : 0.11; g.fillRect(x1, 0, Math.max(1, x2 - x1), h); g.globalAlpha = 1;
          g.strokeStyle = color; g.lineWidth = 1;
          g.beginPath(); g.moveTo(x1 + 0.5, 0); g.lineTo(x1 + 0.5, h); g.moveTo(x2 - 0.5, 0); g.lineTo(x2 - 0.5, h); g.stroke();
        }
      }
      if (S.sel) {
        const x1 = xOf(S.sel.a) * w, x2 = xOf(S.sel.b) * w;
        g.fillStyle = css("--an-accent"); g.globalAlpha = 0.14; g.fillRect(x1, 0, x2 - x1, h); g.globalAlpha = 1;
        g.strokeStyle = css("--an-accent"); g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(x1 + 0.5, 0); g.lineTo(x1 + 0.5, h); g.moveTo(x2 - 0.5, 0); g.lineTo(x2 - 0.5, h); g.stroke();
        g.fillStyle = css("--an-accent"); [x1, x2].forEach((x) => { g.fillRect(x - 3, 0, 6, 7); g.fillRect(x - 3, h - 7, 6, 7); });
      }
      const px = xOf(S.t) * w;
      if (px >= -2 && px <= w + 2) {
        g.strokeStyle = css("--an-text"); g.lineWidth = 1;
        g.beginPath(); g.moveTo(px + 0.5, 0); g.lineTo(px + 0.5, h); g.stroke();
        g.fillStyle = css("--an-text"); g.beginPath(); g.moveTo(px - 4, 0); g.lineTo(px + 4, 0); g.lineTo(px, 6); g.closePath(); g.fill();
      }
    });
    drawRuler();
    const r = cv.ruler, g = r.getContext("2d"), px = xOf(S.t) * r._w;
    g.strokeStyle = css("--an-text"); g.beginPath(); g.moveTo(px + 0.5, 0); g.lineTo(px + 0.5, r._h); g.stroke();
  }
  /* The time axis only: panning must not rebuild the table, or its scroll
   * position and a label being typed are lost. */
  function renderStage() {
    if (!q("stage").offsetParent) return; // not on screen: drawn when it is
    drawRuler(); drawWave();
    if (!q("specPane").hidden) drawSpec();
    drawLane(); drawOverlays(); updateMeta(); drawHScroll(); syncZoomUI();
  }
  function render() { syncSpans(); renderStage(); renderTable(); }
  function updateMeta() {
    q("mSel").textContent = S.sel ? t("selection {from} → {to} ({seconds} s)", { from: fmt(S.sel.a), to: fmt(S.sel.b), seconds: (S.sel.b - S.sel.a).toFixed(3) }) : "";
    q("mView").textContent = S.buf ? t("view {from} → {to} · {zoom}× zoom", { from: fmt(vA()), to: fmt(vB()), zoom: (dur() / vSpan()).toFixed(1) }) : "";
    if (!S.buf && !S.files.length) q("mFile").textContent = t("No file open");
  }

  // -------------------------------------------------------------------- zoom

  const zMax = () => Math.max(2, dur() / 0.02); // down to a 20 ms window
  const zoomNow = () => clamp(1 / Math.max(1e-9, S.view.b - S.view.a), 1, zMax());
  const sliderToZoom = (value) => Math.exp(value / 1000 * Math.log(zMax()));
  const zoomToSlider = (zoom) => clamp(Math.log(Math.max(1, zoom)) / Math.log(zMax()) * 1000, 0, 1000);
  function syncZoomUI() {
    if (!S.buf) { q("zoomV").textContent = "1.0×"; return; }
    const zoom = zoomNow();
    if (document.activeElement !== q("zoom")) q("zoom").value = Math.round(zoomToSlider(zoom));
    q("zoomV").textContent = `${zoom < 10 ? zoom.toFixed(1) : Math.round(zoom)}×`;
  }
  function setZoom(zoom, anchor) {
    if (!S.buf) return;
    const span = clamp(1 / clamp(zoom, 1, zMax()), 1 / zMax(), 1);
    const f = anchor != null ? anchor : xOf(S.t) >= 0 && xOf(S.t) <= 1 ? xOf(S.t) : 0.5;
    const at = S.view.a + (S.view.b - S.view.a) * f;
    const a = clamp(at - span * f, 0, 1 - span);
    S.view = { a, b: a + span }; S.specCache = null; renderStage();
  }
  q("zoom").oninput = (event) => setZoom(sliderToZoom(+event.target.value));
  function zoomAt(fraction, k) {
    const a = S.view.a, b = S.view.b, span = b - a, at = a + span * fraction;
    const next = clamp(span * k, 1 / zMax(), 1), start = clamp(at - (at - a) * (next / span), 0, 1 - next);
    S.view = { a: start, b: start + next }; S.specCache = null; renderStage();
  }
  function pan(fraction) {
    const span = S.view.b - S.view.a, a = clamp(S.view.a + fraction * span, 0, 1 - span);
    if (Math.abs(a - S.view.a) < 1e-9) return;
    S.view = { a, b: a + span }; S.specCache = null; renderStage();
  }
  const zoomFit = () => { S.view = { a: 0, b: 1 }; S.specCache = null; render(); };
  const zoomSel = () => {
    if (!S.sel) return;
    const pad = (S.sel.b - S.sel.a) * 0.15;
    S.view = { a: clamp((S.sel.a - pad) / dur(), 0, 1), b: clamp((S.sel.b + pad) / dur(), 0, 1) }; S.specCache = null; render();
  };
  q("zfit").onclick = zoomFit;
  q("zsel").onclick = zoomSel;
  // Display gain for the waveform (and the spectrogram floor). Never the audio.
  q("amp").oninput = (event) => { S.amp = Math.exp(+event.target.value / 1000 * Math.log(32)); q("ampV").textContent = `${S.amp.toFixed(1)}×`; S.specCache = null; renderStage(); };

  // Timeline scrollbar.
  const hscroll = q("hscroll"), hthumb = q("hthumb");
  const THUMB_MIN = 26;
  function hGeometry() {
    const W = Math.max(1, hscroll.clientWidth), span = clamp(S.view.b - S.view.a, 1e-6, 1);
    const width = Math.min(W, Math.max(THUMB_MIN, span * W));
    return { W, span, width, track: Math.max(1e-6, W - width) };
  }
  function drawHScroll() {
    const { W, span, width, track } = hGeometry(), full = !S.buf || span >= 0.9999;
    hscroll.classList.toggle("is-off", full);
    hthumb.style.left = `${full ? 0 : clamp(S.view.a / (1 - span), 0, 1) * track}px`;
    hthumb.style.width = `${full ? W : width}px`;
  }
  function viewFromThumb(left) {
    const { span, track } = hGeometry();
    const a = clamp(clamp(left, 0, track) / track * (1 - span), 0, 1 - span);
    if (Math.abs(a - S.view.a) < 1e-9) return;
    S.view = { a, b: a + span }; S.specCache = null; renderStage();
  }
  (() => {
    let grab = null;
    hscroll.addEventListener("pointerdown", (event) => {
      const { span, width, track } = hGeometry();
      if (!S.buf || span >= 0.9999) return;
      const x = event.clientX - hscroll.getBoundingClientRect().left, current = clamp(S.view.a / (1 - span), 0, 1) * track;
      hscroll.setPointerCapture(event.pointerId); hthumb.classList.add("is-dragging");
      if (x < current || x > current + width) { grab = width / 2; viewFromThumb(x - width / 2); } else grab = x - current;
    });
    hscroll.addEventListener("pointermove", (event) => { if (grab != null) viewFromThumb(event.clientX - hscroll.getBoundingClientRect().left - grab); });
    const end = () => { if (grab != null) { grab = null; hthumb.classList.remove("is-dragging"); renderTable(); } };
    hscroll.addEventListener("pointerup", end);
    hscroll.addEventListener("pointercancel", end);
  })();

  // Wheel: sideways pans, vertical zooms at the pointer, Ctrl/⌘ + wheel zooms (pinch).
  function bindWheel(element, canZoom) {
    element.addEventListener("wheel", (event) => {
      if (!S.buf) return;
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      if (canZoom && (event.ctrlKey || event.metaKey)) { zoomAt(clamp((event.clientX - rect.left) / rect.width, 0, 1), event.deltaY > 0 ? 1.18 : 0.85); return; }
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) { pan(event.deltaX / Math.max(1, rect.width)); return; }
      if (!canZoom || event.shiftKey) { pan(event.deltaY / Math.max(1, rect.width)); return; }
      zoomAt(clamp((event.clientX - rect.left) / rect.width, 0, 1), event.deltaY > 0 ? 1.25 : 0.8);
    }, { passive: false });
  }
  [cv.wave, cv.spec, cv.lane].forEach((canvas) => bindWheel(canvas, true));
  [cv.ruler, hscroll].forEach((element) => bindWheel(element, false));

  // Selecting on the waveform or spectrogram.
  const paneTime = (event, canvas) => { const rect = canvas.getBoundingClientRect(); return tOf(clamp((event.clientX - rect.left) / rect.width, 0, 1)); };
  [cv.wave, cv.spec].forEach((canvas) => {
    let drag = null;
    canvas.addEventListener("pointerdown", (event) => {
      if (!S.buf) return;
      canvas.setPointerCapture(event.pointerId);
      const time = paneTime(event, canvas);
      if (S.sel) {
        const w = canvas._w, x = xOf(time) * w;
        if (Math.abs(xOf(S.sel.a) * w - x) < 6) { drag = { fix: S.sel.b, mode: "edge" }; return; }
        if (Math.abs(xOf(S.sel.b) * w - x) < 6) { drag = { fix: S.sel.a, mode: "edge" }; return; }
      }
      drag = { mode: "new", anchor: time, moved: false };
      S.sel = { a: time, b: time };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!S.buf) return;
      if (!drag) {
        if (S.sel) { const w = canvas._w, x = xOf(paneTime(event, canvas)) * w; canvas.style.cursor = Math.abs(xOf(S.sel.a) * w - x) < 6 || Math.abs(xOf(S.sel.b) * w - x) < 6 ? "ew-resize" : "text"; } else canvas.style.cursor = "text";
        return;
      }
      const time = paneTime(event, canvas), fixed = drag.mode === "edge" ? drag.fix : drag.anchor;
      S.sel = { a: Math.min(fixed, time), b: Math.max(fixed, time) };
      if (drag.mode === "new" && Math.abs(time - drag.anchor) > vSpan() * 0.002) drag.moved = true;
      drawOverlays(); updateMeta();
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!drag) return;
      const click = drag.mode === "new" && !drag.moved;
      drag = null;
      if (click) { const time = paneTime(event, canvas); S.sel = null; S.t = time; q("cur").textContent = fmt(time); if (S.playing) play(time); S.selId = null; render(); return; }
      drawOverlays(); updateMeta();
    });
  });

  // Editing in the lane: drag an edge to resize, the body to move.
  (() => {
    const canvas = cv.lane;
    let drag = null;
    const hit = (x, w) => {
      for (let i = S.spans.length - 1; i >= 0; i--) {
        const span = S.spans[i], x1 = xOf(span.a) * w, x2 = xOf(span.b) * w;
        if (x >= x1 - 4 && x <= x2 + 4) {
          if (Math.abs(x - x1) <= 5) return { span, edge: "a" };
          if (Math.abs(x - x2) <= 5) return { span, edge: "b" };
          return { span, edge: null };
        }
      }
      return null;
    };
    canvas.addEventListener("pointerdown", (event) => {
      if (!S.buf) return;
      const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, time = tOf(clamp(x / rect.width, 0, 1));
      const h = hit(x, canvas._w);
      if (!h) { S.t = time; q("cur").textContent = fmt(time); if (S.playing) play(time); S.selId = null; render(); return; }
      canvas.setPointerCapture(event.pointerId);
      S.selId = h.span.id;
      drag = { span: h.span, edge: h.edge, grab: time, a0: h.span.a, b0: h.span.b };
      render();
    });
    canvas.addEventListener("pointermove", (event) => {
      const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, time = tOf(clamp(x / rect.width, 0, 1));
      if (!drag) { const h = S.buf ? hit(x, canvas._w) : null; canvas.style.cursor = !h ? "crosshair" : h.edge ? "ew-resize" : "grab"; return; }
      if (drag.edge === "a") drag.span.a = clamp(Math.min(time, drag.span.b - 0.01), 0, dur());
      else if (drag.edge === "b") drag.span.b = clamp(Math.max(time, drag.span.a + 0.01), 0, dur());
      else { const length = drag.b0 - drag.a0; drag.span.a = clamp(drag.a0 + time - drag.grab, 0, dur() - length); drag.span.b = drag.span.a + length; }
      canvas.style.cursor = drag.edge ? "ew-resize" : "grabbing";
      markDirty(); drawLane(); drawOverlays(); renderTable();
    });
    canvas.addEventListener("pointerup", () => { if (drag) { S.spans.sort((x, y) => x.a - y.a); drag = null; render(); updateExport(); } });
    canvas.addEventListener("dblclick", (event) => {
      const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left;
      const h = S.buf ? hit(x, canvas._w) : null;
      if (h) { S.sel = { a: h.span.a, b: h.span.b }; S.t = h.span.a; play(h.span.a, h.span.b); render(); }
    });
  })();
  cv.ruler.addEventListener("pointerdown", (event) => {
    if (!S.buf) return;
    const rect = cv.ruler.getBoundingClientRect(), time = tOf(clamp((event.clientX - rect.left) / rect.width, 0, 1));
    S.t = time; q("cur").textContent = fmt(time); if (S.playing) play(time); drawOverlays();
  });

  // The fixed-height panes must not starve the waveform when a pane is switched on.
  const WAVE_MIN = 110;
  function fitPanes() {
    const stageHeight = q("stage").clientHeight;
    if (!stageHeight || S.mode === "spec") return;
    const chrome = 26 + 12 + 5 + (S.mode === "both" ? 5 : 0);
    const spec = S.mode === "both" ? S.layout.spec : 0;
    let need = WAVE_MIN - (stageHeight - chrome - spec - S.layout.lane);
    if (need <= 0) return;
    if (S.mode === "both") { const cut = Math.min(need, S.layout.spec - 56); S.layout.spec -= cut; need -= cut; }
    if (need > 0) S.layout.lane -= Math.min(need, S.layout.lane - 40);
    applyLayout();
  }
  function setMode(mode) {
    S.mode = mode;
    q("wavePane").hidden = mode === "spec";
    q("specPane").hidden = mode === "wave";
    q("gSpec").hidden = mode !== "both";
    q("stage").classList.toggle("is-spec-main", mode === "spec");
    qa(".annotate-mode [data-mode]").forEach((button) => button.setAttribute("aria-pressed", button.dataset.mode === mode ? "true" : "false"));
    fitPanes(); resize(); render();
  }
  qa(".annotate-mode [data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));

  // -------------------------------------------------------------------- tags

  function renderTags() {
    q("tagrow").innerHTML = S.tags.map((tag, index) => `<button type="button" class="chip annotate-tag${tag.name === S.tag ? " is-active" : ""}" style="--tag:${esc(tag.color)}" aria-pressed="${tag.name === S.tag}" data-tag="${esc(tag.name)}"><span class="annotate-tag-dot" aria-hidden="true"></span><span>${esc(tag.name)}</span>${index < 9 ? `<kbd>${index + 1}</kbd>` : ""}</button>`).join("");
    const chip = q("curTag");
    chip.style.setProperty("--tag", tagColor(S.tag));
    chip.innerHTML = `<span class="annotate-tag-dot" aria-hidden="true"></span>${esc(S.tag)}`;
  }
  q("tagrow").addEventListener("click", (event) => { const button = event.target.closest("[data-tag]"); if (button) { S.tag = button.dataset.tag; renderTags(); } });
  q("newtagForm").onsubmit = (event) => {
    event.preventDefault();
    const value = q("newtag").value.trim();
    if (!value) return;
    ensureTag(value); S.tag = value; q("newtag").value = ""; renderTags(); renderTable();
  };

  // ---------------------------------------------------------- spans and table

  function markDirty() { const file = F(); if (file) file.dirty = true; }
  function addSpan() {
    if (!S.buf) { toast(t("Open a file first"), true); return; }
    if (!S.sel || S.sel.b - S.sel.a < 0.001) { toast(t("Select a range on the waveform first"), true); return; }
    const span = { id: S.nextId++, a: S.sel.a, b: S.sel.b, tag: S.tag, label: "" };
    S.spans.push(span); S.spans.sort((x, y) => x.a - y.a);
    S.selId = span.id; S.sel = null; markDirty(); render(); renderFileList(); updateExport();
  }
  function deleteSpan(id) {
    S.spans = S.spans.filter((span) => span.id !== id);
    if (S.selId === id) S.selId = null;
    markDirty(); render(); renderFileList(); updateExport();
  }
  function selectSpan(id) {
    const span = S.spans.find((item) => item.id === id);
    if (!span) return;
    S.selId = id; S.sel = { a: span.a, b: span.b }; S.t = span.a; q("cur").textContent = fmt(span.a); render();
  }
  let shownSelId = null;
  function renderTable() {
    const body = q("tbody"), wrap = q("tblwrap");
    // The table is rebuilt, so remember where the user was first.
    const scroll = wrap.scrollTop, active = document.activeElement;
    const row = active && active.closest ? active.closest('[data-an="tbody"] tr') : null;
    const focus = row ? { id: row.dataset.id, field: active.dataset.field, start: active.selectionStart, end: active.selectionEnd } : null;
    q("count").textContent = S.spans.length;
    const total = S.files.reduce((n, file) => n + file.spans.length, 0);
    q("totalCount").textContent = S.files.length > 1 ? t("· {total} across {files} files", { total, files: S.files.length }) : "";
    q("tbl").hidden = !S.spans.length;
    q("empty").hidden = Boolean(S.spans.length);
    body.innerHTML = S.spans.map((span, index) => {
      const options = S.tags.map((tag) => `<option value="${esc(tag.name)}"${tag.name === span.tag ? " selected" : ""}>${esc(tag.name)}</option>`).join("");
      return `<tr data-id="${span.id}" class="${span.id === S.selId ? "is-selected" : ""}">
        <td class="num">${index + 1}</td>
        <td><span class="annotate-row-tag" style="--tag:${esc(tagColor(span.tag))}"><span class="annotate-tag-dot" aria-hidden="true"></span><select class="control" data-field="tag" aria-label="${esc(t("Tag"))}">${options}</select></span></td>
        <td class="num">${fmt(span.a)}</td><td class="num">${fmt(span.b)}</td><td class="num">${(span.b - span.a).toFixed(3)} s</td>
        <td><input class="control" data-field="label" value="${esc(span.label || "")}" placeholder="—" aria-label="${esc(t("Label"))}" /></td>
        <td><button class="button button-ghost button-icon button-small" type="button" data-delete title="${esc(t("Delete"))}" aria-label="${esc(t("Delete"))}">×</button></td>
      </tr>`;
    }).join("");
    wrap.scrollTop = scroll;
    if (focus) {
      const element = body.querySelector(`tr[data-id="${focus.id}"] [data-field="${focus.field}"]`);
      if (element) {
        element.focus({ preventScroll: true });
        if (focus.start != null && element.setSelectionRange) { try { element.setSelectionRange(focus.start, focus.end); } catch { /* not a text field */ } }
      }
    }
    // Follow the selection only when it moved, so scrolling by hand stays put.
    if (S.selId !== shownSelId) {
      shownSelId = S.selId;
      const selected = S.selId != null && body.querySelector(`tr[data-id="${S.selId}"]`);
      if (selected) selected.scrollIntoView({ block: "nearest" });
    }
  }
  q("tbody").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-id]");
    if (!row || event.target.closest("select, input")) return;
    const id = Number(row.dataset.id);
    if (event.target.closest("[data-delete]")) deleteSpan(id); else selectSpan(id);
  });
  q("tbody").addEventListener("change", (event) => {
    if (event.target.dataset.field !== "tag") return;
    const span = S.spans.find((item) => item.id === Number(event.target.closest("tr").dataset.id));
    if (span) { span.tag = event.target.value; markDirty(); render(); updateExport(); }
  });
  q("tbody").addEventListener("input", (event) => {
    if (event.target.dataset.field !== "label") return;
    const span = S.spans.find((item) => item.id === Number(event.target.closest("tr").dataset.id));
    if (span) { span.label = event.target.value; markDirty(); drawLane(); updateExport(); }
  });
  q("clearAll").onclick = async () => {
    if (!S.spans.length) return;
    const ok = await ask(t("Clear spans"), t("Delete all {n} spans on {name}? Other files keep theirs.", { n: S.spans.length, name: S.name }),
      [{ label: t("Cancel"), value: false }, { label: t("Delete"), value: true, primary: true }]);
    if (ok) { S.spans = []; S.selId = null; markDirty(); render(); renderFileList(); updateExport(); }
  };

  // ------------------------------------------------------------------ export

  function withSpans() { syncSpans(); return S.files.filter((file) => file.spans.length); }
  const exportText = () => M.buildExport(S.format, S.files, { current: F(), folderName: S.folderName });
  const exportFileName = () => M.exportName(S.format, { files: S.files, current: F(), folderName: S.folderName });
  const canExport = () => (S.format === "audacity" ? Boolean(F()?.spans.length) : withSpans().length > 0);
  function updateExport() {
    const ready = canExport();
    q("out").value = ready ? exportText() : "";
    q("out").placeholder = S.format === "audacity" && withSpans().length ? t("The current file has no spans; Audacity labels are per file.") : t("No spans yet.");
    const n = withSpans().length;
    q("scope").textContent = S.format === "audacity" ? t("The current file only.") : S.files.length > 1 ? t(n === 1 ? "All {n} annotated file." : "All {n} annotated files.", { n }) : "";
    qa('[data-an="fmt"] [data-f]').forEach((button) => button.setAttribute("aria-pressed", button.dataset.f === S.format ? "true" : "false"));
    syncDownload();
    if (previewWindow && !previewWindow.closed) writePreview();
  }
  function syncDownload() {
    const button = document.getElementById("annotate-download");
    if (button) button.disabled = !canExport();
    const hasSpans = withSpans().length > 0;
    qa('[data-act="copy"], [data-act="preview"]').forEach((item) => { item.disabled = !canExport(); });
    qa('[data-act="saveSidecars"]').forEach((item) => { item.disabled = !hasSpans; item.title = S.dirHandle ? "" : t("Open a folder read-write first (Chromium browsers)"); });
    qa('[data-act="closeFile"], [data-act="closeAll"]').forEach((item) => { item.disabled = !S.files.length; });
    q("clearAll").disabled = !S.spans.length;
  }
  q("fmt").addEventListener("click", (event) => { const button = event.target.closest("[data-f]"); if (button) { S.format = button.dataset.f; updateExport(); } });

  function download() {
    if (!canExport()) { toast(t("Nothing to export yet"), true); return; }
    const name = exportFileName();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([exportText()], { type: "text/plain" }));
    link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
    toast(t("Downloaded {name}", { name }));
  }
  async function copy() {
    if (!canExport()) return;
    try { await navigator.clipboard.writeText(exportText()); } catch { q("out").select(); document.execCommand("copy"); }
    toast(t("Copied to the clipboard"));
  }
  // The preview opens in its own tab: a folder's JSON is too long for the Inspector.
  let previewWindow = null;
  function writePreview() {
    const w = previewWindow;
    if (!w || w.closed) return;
    const name = exportFileName();
    w.document.open();
    w.document.write(`<!doctype html><meta charset="utf-8"><title>${esc(name)}</title><style>:root{color-scheme:light dark}body{margin:0;font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}header{position:sticky;top:0;padding:10px 16px;background:Canvas;border-bottom:1px solid GrayText;font-weight:600}pre{margin:0;padding:16px;white-space:pre}</style><header>${esc(name)}</header><pre>${esc(exportText())}</pre>`);
    w.document.close();
  }
  function preview() {
    if (!canExport()) return;
    if (previewWindow && !previewWindow.closed) { writePreview(); previewWindow.focus(); return; }
    let w = null;
    try { w = window.open("", "puresoundSpanPreview", "width=920,height=780"); } catch { /* blocked */ }
    if (!w) { toast(t("The browser blocked the new tab; the export is in the Inspector"), true); return; }
    previewWindow = w; writePreview(); w.focus();
  }
  document.getElementById("annotate-download")?.addEventListener("click", download);

  // ----------------------------------------------------------------- actions

  function resetToEmpty() {
    stopPlayback();
    S.files = []; S.cur = -1; S.buf = null; S.mono = null; S.mip = null; S.spans = []; S.selId = null; S.sel = null;
    S.folderName = ""; S.dirHandle = null; S.name = ""; S.t = 0; S.view = { a: 0, b: 1 }; S.specCache = null;
    decoded.clear();
    AUDIO_BUTTONS.forEach((id) => { q(id).disabled = true; });
    q("drop").classList.remove("is-gone");
    q("mFile").textContent = t("No file open"); q("mFmt").textContent = ""; q("mLevel").textContent = "";
    q("dur").textContent = fmt(0); q("cur").textContent = fmt(0);
    renderFileList(); render(); updateExport();
  }
  const ACT = {
    openFiles: () => q("file").click(),
    openFolder,
    openFolderInput: () => q("folder").click(),
    importSpans: () => q("spanfile").click(),
    saveSidecars,
    copy,
    preview,
    closeFile: async () => {
      const file = F();
      if (!file) return;
      if (file.spans.length) {
        const ok = await ask(t("Remove the file"), t("{name} has {n} spans. Remove it from the list? The file on disk is untouched.", { name: file.name, n: file.spans.length }),
          [{ label: t("Cancel"), value: false }, { label: t("Remove"), value: true, primary: true }]);
        if (!ok) return;
      }
      decoded.delete(file.id);
      const at = S.cur;
      S.files.splice(at, 1);
      // The index now names another entry: drop the working spans before
      // selectFile's syncSpans() writes them onto the neighbour.
      S.cur = -1; S.spans = []; S.nextId = 1;
      if (!S.files.length) resetToEmpty(); else selectFile(clamp(at, 0, S.files.length - 1));
    },
    closeAll: async () => {
      if (!S.files.length) return;
      const list = withSpans(), n = list.reduce((total, file) => total + file.spans.length, 0);
      if (n) {
        const ok = await ask(t("Close all"), t("{n} spans across {files} files will be discarded. Export or save them first if you need them.", { n, files: list.length }),
          [{ label: t("Cancel"), value: false }, { label: t("Discard"), value: true, primary: true }]);
        if (!ok) return;
      }
      resetToEmpty();
    },
  };
  screen.addEventListener("click", (event) => {
    const button = event.target.closest("[data-act]");
    if (!button || button.disabled) return;
    const action = ACT[button.dataset.act];
    if (action) action();
  });

  // ---------------------------------------------------------------- keyboard

  const onScreen = () => shell()?.current() === "annotate";
  document.addEventListener("keydown", (event) => {
    if (!onScreen() || document.querySelector("dialog[open]") || shell()?.overlayOpen?.()) return;
    const mod = event.metaKey || event.ctrlKey;
    if (M.shortcutTarget(event, "annotate") === null) return;
    const key = event.key;
    if (mod) {
      const lower = key.toLowerCase();
      if (lower === "o") { event.preventDefault(); if (event.shiftKey) openFolder(); else q("file").click(); return; }
      if (lower === "i") { event.preventDefault(); q("spanfile").click(); return; }
      if (lower === "s") { event.preventDefault(); saveSidecars(); return; }
      if (lower === "e") { event.preventDefault(); download(); return; }
      return;
    }
    if (event.altKey) return;
    const control = event.target.closest?.("button, a");
    if (key === "?") { event.preventDefault(); shell()?.openHelp("annotate"); return; }
    if (key === "[" || key === "]") { if (S.files.length < 2) return; event.preventDefault(); selectFile(clamp(S.cur + (key === "[" ? -1 : 1), 0, S.files.length - 1)); return; }
    if (key === " ") {
      if (control && !mainHost.contains(control)) return; // Space clicks a focused button elsewhere
      event.preventDefault();
      if (event.shiftKey && S.sel) play(S.sel.a, S.sel.b); else togglePlay();
      return;
    }
    if (!S.buf) return;
    if (key === "Enter") { if (control) return; event.preventDefault(); addSpan(); return; }
    if (key === "Backspace" || key === "Delete") { if (S.selId) { event.preventDefault(); deleteSpan(S.selId); } return; }
    if (key >= "1" && key <= "9") { const tag = S.tags[+key - 1]; if (tag) { S.tag = tag.name; renderTags(); } return; }
    if (key === "s" || key === "S") { S.sel = { a: S.t, b: Math.max(S.t, S.sel ? S.sel.b : S.t) }; drawOverlays(); updateMeta(); return; }
    if (key === "e" || key === "E") { S.sel = { a: Math.min(S.t, S.sel ? S.sel.a : S.t), b: S.t }; drawOverlays(); updateMeta(); return; }
    if (key === "l" || key === "L") { q("loop").click(); return; }
    if (key === "g" || key === "G") { autoBoost(); return; }
    if (key === "+" || key === "=") { zoomAt(0.5, 0.6); return; }
    if (key === "-" || key === "_") { zoomAt(0.5, 1.7); return; }
    if (key === "0") { zoomFit(); return; }
    if (key === "ArrowLeft" || key === "ArrowRight") {
      event.preventDefault();
      S.t = clamp(S.t + (key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 0.1 : 1), 0, dur());
      q("cur").textContent = fmt(S.t);
      if (S.playing) play(S.t); else drawOverlays();
      return;
    }
    if (key === "Escape") { S.sel = null; S.selId = null; render(); }
  });
  window.addEventListener("beforeunload", (event) => {
    if (S.files.some((file) => file.spans.length && file.dirty)) { event.preventDefault(); event.returnValue = ""; }
  });

  // --------------------------------------------------------------- language

  window.addEventListener("puresound:lang", () => {
    setPlayLabel();
    renderFormat();
    updateLevelMeta();
    renderFileList();
    render();
    updateExport();
  });

  // -------------------------------------------------------------------- boot

  loadLayout(); applyLayout(); setMode("wave");
  renderTags(); renderFileList(); resize(); render(); updateExport(); syncZoomUI(); setPlayLabel();
  q("mFile").textContent = t("No file open");

  window.PureSoundAnnotate = {
    /* Called when the screen comes on: canvases get their size back. */
    show() { resize(); fitPanes(); renderStage(); },
    /* Another screen's audio, as a new file here (a Blob, a File, or bytes). */
    async addAudio({ name, bytes }) {
      const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: "audio/wav" });
      let path = name;
      for (let n = 2; S.files.some((file) => file.path === path); n++) path = name.replace(/(\.[^.]+)?$/, ` (${n})$1`);
      const file = new File([blob], path, { type: blob.type || "audio/wav" });
      await intake([{ name: path, path, file }], { select: path });
    },
  };
  shell()?.onShow("annotate", () => window.PureSoundAnnotate.show());
})();
