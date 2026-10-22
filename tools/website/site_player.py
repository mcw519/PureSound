"""The site's audio player: a recording and what a model returned, compared on
one time axis, in the playground's dark player style.

The waveform is the timeline: click or drag it to seek, hover it to read the
time. The two tracks play from one clock through Web Audio, so switching
between them is seamless and keeps the place. The track being heard is drawn in
its colour (mint in, periwinkle out); the other stays faintly behind it. While
a model runs, a scan line shows how far it has got.
"""

from __future__ import annotations

from site_chrome import t

ZH_TW = {
    "Play": "播放", "Pause": "暫停", "Playback position": "播放位置", "Listen to": "聆聽",
}

PLAY_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true" class="pp-icon-play"><path d="M4.5 2.6v10.8L13.3 8z"/></svg>'
PAUSE_ICON = ('<svg viewBox="0 0 16 16" aria-hidden="true" class="pp-icon-pause">'
              '<path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>')


def markup(a_label: str, b_label: str, actions: str = "") -> str:
    """A player with tracks named ``a_label`` (the recording) and ``b_label``
    (the model's output), and ``actions`` at the end of its transport."""
    return f"""<div class="pp" data-player>
<div class="pp-stage">
<div class="pp-ruler" aria-hidden="true"></div>
<div class="pp-wave" role="slider" tabindex="0" aria-label="Playback position" data-i18n-attr="aria-label"
 aria-valuemin="0" aria-valuemax="0" aria-valuenow="0">
<canvas class="pp-dim"></canvas><div class="pp-played"><canvas class="pp-bright"></canvas></div>
<div class="pp-scan" hidden></div><div class="pp-hover" hidden><span></span></div><div class="pp-head"></div>
</div>
</div>
<div class="pp-transport">
<button class="pp-play" type="button" aria-label="Play" data-i18n-attr="aria-label" disabled>{PLAY_ICON}{PAUSE_ICON}</button>
<span class="pp-clock"><span class="pp-now">0:00.0</span><span class="pp-total">/ 0:00.0</span></span>
<div class="pp-ab" role="group" aria-label="Listen to" data-i18n-attr="aria-label">
<button type="button" data-ab="a" aria-pressed="true"><i class="in"></i>{t(a_label)}</button>
<button type="button" data-ab="b" aria-pressed="false" disabled><i class="out"></i>{t(b_label)}</button>
</div>
<div class="pp-actions">{actions}</div>
</div>
</div>"""


CSS = """
/* The player: dark in both themes, as the playground's audio players are. */
.pp { min-width: 0; color: var(--on-nav); background: var(--nav); border: 1px solid var(--nav-line);
  border-radius: var(--radius-panel); overflow: hidden; }
.pp-stage { padding: 8px 16px 14px; }
.pp-ruler { position: relative; height: 20px; color: var(--on-nav-3); font-family: var(--font-mono); font-size: 10px; }
.pp-ruler span { position: absolute; top: 2px; bottom: 0; padding-left: 5px; border-left: 1px solid var(--nav-line);
  line-height: 12px; font-variant-numeric: tabular-nums; }
.pp-wave { position: relative; height: 132px; cursor: crosshair; touch-action: none; user-select: none; }
.pp-wave:focus-visible { outline: 2px solid var(--focus); outline-offset: 4px; border-radius: 2px; }
.pp-wave canvas { position: absolute; top: 0; left: 0; display: block; height: 100%; }
.pp-dim { width: 100%; }
.pp-played { position: absolute; top: 0; bottom: 0; left: 0; width: 0; overflow: hidden; }
.pp-head { position: absolute; top: -6px; bottom: -6px; left: 0; width: 2px; margin-left: -1px; background: var(--accent);
  border-radius: 1px; pointer-events: none; }
.pp-head::before { content: ""; position: absolute; top: -3px; left: -3px; width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); }
.pp-hover { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(255, 255, 255, 0.4); pointer-events: none; }
.pp-hover span { position: absolute; bottom: 4px; left: 6px; padding: 2px 5px; color: var(--on-nav); background: var(--nav-3);
  border-radius: 3px; font-family: var(--font-mono); font-size: 10px; white-space: nowrap; }
.pp-hover.is-flipped span { right: 6px; left: auto; }
.pp-scan { position: absolute; top: 0; bottom: 0; left: 0; width: 0; pointer-events: none;
  background: linear-gradient(90deg, transparent 60%, rgba(189, 187, 255, 0.14));
  border-right: 2px solid var(--periwinkle); transition: width 0.12s linear; }
.pp-transport { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 10px 16px;
  background: var(--nav-2); border-top: 1px solid var(--nav-line); }
.pp-play { width: 42px; height: 42px; flex: none; display: grid; place-items: center; padding: 0; color: var(--on-mint);
  background: var(--mint); border: 0; border-radius: 50%; }
.pp-play:hover:not(:disabled) { background: color-mix(in srgb, var(--mint) 70%, #ffffff); }
.pp-play:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.pp-play:disabled { opacity: 0.35; }
.pp-play svg { width: 16px; height: 16px; fill: currentColor; }
.pp-play .pp-icon-pause, .pp.is-playing .pp-play .pp-icon-play { display: none; }
.pp.is-playing .pp-play .pp-icon-pause { display: block; }
.pp-clock { display: inline-flex; gap: 6px; font-family: var(--font-mono); font-size: 13px; font-variant-numeric: tabular-nums; }
.pp-total { color: var(--on-nav-3); }
.pp-ab { display: flex; gap: 2px; padding: 2px; background: rgba(255, 255, 255, 0.06); border-radius: var(--radius-panel); }
.pp-ab button { display: inline-flex; align-items: center; gap: 7px; height: 30px; padding: 0 12px; color: var(--on-nav-2);
  background: transparent; border: 0; border-radius: var(--radius-control); font-size: var(--text-small); font-weight: 500; }
.pp-ab button:hover:not(:disabled) { color: var(--on-nav); }
.pp-ab button:disabled { opacity: 0.35; }
.pp-ab button[aria-pressed="true"] { color: var(--on-nav); background: var(--nav-3); box-shadow: inset 0 0 0 1px var(--nav-line); }
.pp-ab i { width: 8px; height: 8px; border-radius: 50%; }
.pp-ab i.in { background: var(--mint); } .pp-ab i.out { background: var(--periwinkle); }
.pp-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-left: auto; }
.pp-actions .button-primary { height: 36px; padding: 0 16px; color: var(--nav); background: var(--on-nav); border-color: var(--on-nav); }
.pp-actions .button-primary:hover:not(:disabled) { color: var(--nav); background: var(--periwinkle); border-color: var(--periwinkle); }
.pp-actions .button-secondary { color: var(--on-nav); background: transparent; border-color: var(--nav-line); }
.pp-actions .button-secondary:hover:not(:disabled) { background: var(--nav-3); border-color: var(--on-nav-3); }
@media (max-width: 560px) { .pp-actions { width: 100%; margin-left: 0; } .pp-actions .button { flex: 1; } }
"""

SCRIPT = """
window.PureSoundPlayer = class {
  constructor(root, tr = (text) => text) {
    this.root = root;
    this.tr = tr;
    const $ = (selector) => root.querySelector(selector);
    this.wave = $(".pp-wave"); this.dim = $(".pp-dim"); this.bright = $(".pp-bright"); this.played = $(".pp-played");
    this.head = $(".pp-head"); this.hover = $(".pp-hover"); this.scan = $(".pp-scan"); this.ruler = $(".pp-ruler");
    this.now = $(".pp-now"); this.total = $(".pp-total"); this.playButton = $(".pp-play");
    this.ab = [...root.querySelectorAll("[data-ab]")];
    this.input = null; this.output = null; this.rate = 16000; this.active = "a";
    this.offset = 0; this.playing = false; this.reveal = 1; this.sources = []; this.gains = {}; this.buffers = {};

    this.playButton.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    for (const button of this.ab) button.addEventListener("click", () => this.listen(button.dataset.ab));
    const at = (event) => {
      const box = this.wave.getBoundingClientRect();
      return Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * this.duration;
    };
    this.wave.addEventListener("pointerdown", (event) => {
      if (!this.input) return;
      this.wave.setPointerCapture(event.pointerId);
      this.dragging = true;
      this.seek(at(event));
    });
    this.wave.addEventListener("pointermove", (event) => {
      if (!this.input) return;
      if (this.dragging) this.seek(at(event));
      const box = this.wave.getBoundingClientRect();
      this.hover.hidden = false;
      this.hover.style.left = (event.clientX - box.left) + "px";
      this.hover.classList.toggle("is-flipped", event.clientX - box.left > box.width - 80);
      this.hover.firstElementChild.textContent = this.format(at(event));
    });
    this.wave.addEventListener("pointerup", () => { this.dragging = false; });
    this.wave.addEventListener("pointerleave", () => { this.hover.hidden = true; });
    this.wave.addEventListener("keydown", (event) => {
      if (!this.input) return;
      const jump = event.shiftKey ? 5 : 1;
      const moves = { ArrowRight: this.position() + jump, ArrowLeft: this.position() - jump, Home: 0, End: this.duration };
      if (event.key in moves) this.seek(moves[event.key]);
      else if (event.key === " " || event.key === "Enter") this.playing ? this.pause() : this.play();
      else return;
      event.preventDefault();
    });
    new ResizeObserver(() => this.draw()).observe(this.wave);
  }

  get duration() { return this.input ? this.input.length / this.rate : 0; }

  format(seconds) {
    const t = Math.max(0, seconds);
    return Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0");
  }

  /* A new recording: the model's output, if any, is dropped. */
  load(input, rate) {
    this.stopSources();
    this.playing = false;
    Object.assign(this, { input, rate, output: null, offset: 0, peaks: null, buffers: {}, reveal: 1 });
    this.ab[1].disabled = true;
    this.playButton.disabled = false;
    this.listen("a");
    this.renderRuler();
    this.update();
  }

  /* What the model returned, on the recording's time axis; it is heard next. */
  setOutput(output) {
    Object.assign(this, { output, peaks: null });
    delete this.buffers.b;
    if (this.playing) this.restart();
    this.ab[1].disabled = false;
    this.listen("b");
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { this.reveal = 1; this.draw(); return; }
    const started = performance.now();
    const step = (now) => {
      const x = Math.min(1, (now - started) / 700);
      this.reveal = 1 - (1 - x) ** 3;
      this.draw();
      if (x < 1) requestAnimationFrame(step);
    };
    this.reveal = 0;
    requestAnimationFrame(step);
  }

  /* How far a model run has got, 0 to 1; null when none is running. */
  progress(value) {
    this.scan.hidden = value == null;
    if (value != null) this.scan.style.width = Math.round(value * 1000) / 10 + "%";
  }

  listen(which) {
    this.active = which === "b" && this.output ? "b" : "a";
    for (const button of this.ab) button.setAttribute("aria-pressed", String(button.dataset.ab === this.active));
    const ctx = this.ctx;
    for (const [key, gain] of Object.entries(this.gains)) {
      gain.gain.setTargetAtTime(key === this.active ? 1 : 0, ctx.currentTime, 0.006);
    }
    this.draw();
  }

  // ------------------------------------------------------------ playback

  buffer(samples) {
    const buffer = this.ctx.createBuffer(1, samples.length, this.rate);
    buffer.copyToChannel(samples, 0);
    return buffer;
  }

  position() {
    if (!this.playing) return this.offset;
    return Math.min(this.duration, this.ctx.currentTime - this.startedAt);
  }

  async play() {
    if (!this.input) return;
    this.ctx = this.ctx || new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();
    if (this.offset >= this.duration - 0.02) this.offset = 0;
    this.playing = true;
    this.restart();
    const tick = () => {
      if (!this.playing) return;
      if (this.position() >= this.duration) { this.pause(); this.offset = this.duration; this.update(); return; }
      this.update();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* Both tracks from `offset`, one of them silent. */
  restart() {
    const offset = this.playing && this.sources.length ? this.position() : this.offset;
    this.stopSources();
    this.buffers.a = this.buffers.a || this.buffer(this.input);
    if (this.output) this.buffers.b = this.buffers.b || this.buffer(this.output);
    for (const key of ["a", "b"]) {
      if (!this.buffers[key]) continue;
      const source = this.ctx.createBufferSource();
      source.buffer = this.buffers[key];
      const gain = this.ctx.createGain();
      gain.gain.value = key === this.active ? 1 : 0;
      source.connect(gain).connect(this.ctx.destination);
      source.start(0, offset);
      this.sources.push(source);
      this.gains[key] = gain;
    }
    this.startedAt = this.ctx.currentTime - offset;
    this.offset = offset;
    this.update();
  }

  pause() {
    this.offset = this.position();
    this.playing = false;
    this.stopSources();
    this.update();
  }

  seek(seconds) {
    this.offset = Math.min(this.duration, Math.max(0, seconds));
    if (this.playing) { this.stopSources(); this.restart(); }
    this.update();
  }

  stopSources() {
    for (const source of this.sources) { try { source.stop(); } catch { /* not started */ } source.disconnect(); }
    this.sources = [];
    this.gains = {};
  }

  // ------------------------------------------------------------ drawing

  update() {
    const fraction = this.duration ? this.position() / this.duration : 0;
    this.played.style.width = fraction * 100 + "%";
    this.head.style.left = fraction * 100 + "%";
    this.now.textContent = this.format(this.position());
    this.total.textContent = "/ " + this.format(this.duration);
    this.root.classList.toggle("is-playing", this.playing);
    this.playButton.setAttribute("aria-label", this.tr(this.playing ? "Pause" : "Play"));
    this.wave.setAttribute("aria-valuemax", this.duration.toFixed(1));
    this.wave.setAttribute("aria-valuenow", this.position().toFixed(1));
    this.wave.setAttribute("aria-valuetext", this.now.textContent);
  }

  renderRuler() {
    const d = this.duration;
    const step = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => d / s <= 8) || 600;
    const label = (s) => (d < 60 ? (Math.round(s * 10) / 10) + "s" : this.format(s).replace(/\\.\\d$/, ""));
    let html = "";
    for (let s = 0; s <= d * 0.94; s += step) html += `<span style="left:${(s / d) * 100}%">${label(s)}</span>`;
    this.ruler.innerHTML = d ? html : "";
  }

  /* Per bar, the loudest sample of each track, scaled by the recording's loudest. */
  peaksFor(count) {
    if (this.peaks && this.peaks.count === count) return this.peaks;
    const reduce = (samples) => {
      if (!samples) return null;
      const out = new Float32Array(count);
      for (let bar = 0; bar < count; bar += 1) {
        let peak = 0;
        for (let i = Math.floor((bar * samples.length) / count), end = Math.floor(((bar + 1) * samples.length) / count); i < end; i += 1) {
          const value = Math.abs(samples[i]);
          if (value > peak) peak = value;
        }
        out[bar] = peak;
      }
      return out;
    };
    const a = reduce(this.input), b = reduce(this.output);
    const top = Math.max(1e-4, ...a);
    // A gentle curve, so quiet passages still show beside loud ones.
    for (const values of [a, b]) if (values) for (let i = 0; i < count; i += 1) values[i] = Math.min(1, values[i] / top) ** 0.75;
    this.peaks = { count, a, b };
    return this.peaks;
  }

  draw() {
    const ratio = devicePixelRatio || 1;
    const cssWidth = this.wave.clientWidth;
    const width = Math.round(cssWidth * ratio), height = Math.round(this.wave.clientHeight * ratio);
    if (!width || !height) return;
    this.bright.style.width = cssWidth + "px";
    const style = getComputedStyle(document.documentElement);
    const colour = { a: style.getPropertyValue("--mint").trim(), b: style.getPropertyValue("--periwinkle").trim() };
    const bar = Math.max(1, Math.round(2 * ratio)), step = bar + Math.max(1, Math.round(ratio));
    const count = Math.floor(width / step);
    for (const [canvas, dimmed] of [[this.dim, true], [this.bright, false]]) {
      canvas.width = width;
      canvas.height = height;
      const g = canvas.getContext("2d");
      g.clearRect(0, 0, width, height);
      g.globalAlpha = 0.12;
      g.fillStyle = "#ffffff";
      g.fillRect(0, Math.floor(height / 2), width, Math.max(1, Math.round(ratio)));
      if (!this.input) continue;
      const peaks = this.peaksFor(count);
      const order = !this.output ? [["a", 1]] : this.active === "b" ? [["a", 0.3], ["b", 1]] : [["b", 0.3], ["a", 1]];
      for (const [key, alpha] of order) {
        const values = peaks[key];
        g.globalAlpha = alpha * (dimmed ? 0.36 : 1);
        g.fillStyle = colour[key];
        const limit = key === "b" ? Math.floor(count * this.reveal) : count;
        for (let i = 0; i < limit; i += 1) {
          const half = Math.max(ratio, values[i] * (height / 2 - 2 * ratio));
          g.fillRect(i * step, height / 2 - half, bar, half * 2);
        }
      }
    }
  }
};
"""
