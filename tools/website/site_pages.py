"""The site's pages other than Benchmarks, each made from the repository.

Home is the README: its introduction, then every section after the released
models, which the task tiles and the Models page stand for. Models is the
catalog (``model_zoo/catalog.yaml``). Each model card a catalog entry names is a
page of its own, in both languages. Try it runs the released streaming models in
the reader's browser with the playground's on-device payload.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import yaml

import site_player
from site_chrome import FILTER_SCRIPT, REPO_ROOT, TASKS, bilingual, esc, page, segmented, t
from site_markdown import Links, read, render, split_sections, translation

CATALOG = "model_zoo/catalog.yaml"
SYMBOLS = {"speaker_embedding": "◌"}

ZH_TW = {
    "Try it in your browser": "在瀏覽器試用", "Streaming": "串流",
    "Model zoo": "模型庫",
    "Sample rate": "取樣率", "Look-ahead": "前瞻延遲", "Recommended": "建議設定",
    "Download": "下載", "Run it": "執行方式", "Model cards": "模型卡",
    "float weights": "浮點權重", "Models in this card": "此模型卡涵蓋的模型",
    "yes": "是", "no": "否",
    # Try it
    "Try a model in your browser": "在瀏覽器試用模型",
    "The released streaming models run on this device in ONNX Runtime Web (WebAssembly). "
    "Audio never leaves the browser.":
        "已發布的串流模型在這台裝置上以 ONNX Runtime Web（WebAssembly）執行，音訊不會離開瀏覽器。",
    "Model": "模型", "Audio": "音訊", "Samples": "範例",
    "Drop audio here": "將音檔拖放到這裡",
    "or click to browse · any rate, prepared to mono 16 kHz": "或點選瀏覽 · 任何取樣率，會轉成單聲道 16 kHz",
    "Replace audio": "替換音檔",
    "{name} · drop or click to choose another file": "{name} · 拖放或點選以選擇其他檔案",
    "Model output share": "模型輸出比例",
    "dry_blend: the share of the model output mixed with the untouched input.":
        "dry_blend：模型輸出與原始輸入混合時，模型輸出所佔的比例。",
    "input": "原始輸入", "model": "模型輸出",
    "Run": "執行", "Running": "執行中", "Ready": "就緒", "Unavailable": "無法使用",
    "Downloading the model": "下載模型中", "Processing {percent}%": "處理中 {percent}%",
    "Done in {seconds} s": "{seconds} 秒完成",
    "Input": "輸入", "Output": "輸出", "Download output": "下載輸出",
    "Real-time factor": "即時率", "Threads": "執行緒", "Processing time": "處理時間",
    "Choose a sample or a file first.": "請先選一段範例或一個檔案。",
    "Two talkers": "兩位說話者", "Near talker, far talker": "近處與遠處的說話者", "Speech in noise": "噪音中的語音",
    "This page is not cross-origin isolated, so models run on one thread here.":
        "這個頁面沒有跨來源隔離（cross-origin isolation），模型只能以單執行緒執行。",
    "Lower is faster; below 1 keeps up with real time.": "數字越小越快；低於 1 即可跟上即時。",
    "This build of the site has no on-device payload.": "這個版本的網站沒有附上裝置端推論檔案。",
}


@dataclass
class Site:
    """What every page is made from."""

    catalog: list[dict]
    links: Links
    #: Model card README path -> page, for the catalog's cards.
    cards: dict[str, str]
    #: Models the on-device payload carries.
    device_models: list[str]


def load_catalog() -> list[dict]:
    return yaml.safe_load(read(CATALOG))["models"]


def card_of(model: dict) -> str | None:
    """The first README among a model's references is its card."""
    return next((ref for ref in model.get("benchmark_references") or [] if ref.endswith("README.md")), None)


def card_pages(catalog: list[dict]) -> dict[str, str]:
    pages = {}
    for model in catalog:
        card = card_of(model)
        if card and card not in pages:
            pages[card] = f"cards/{model['task'].replace('_', '-')}.html"
    return pages


def page_links(cards: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Both editions of each card lead to its page; the Chinese one's anchors
    carry the Chinese rendering's prefix."""
    pages = {}
    for card, target in cards.items():
        pages[card] = (target, "")
        if zh := translation(card):
            pages[zh] = (target, "zh-")
    return pages


def file_size(path: str) -> int | None:
    """A repository file's size; a Git LFS pointer gives its object's."""
    file = REPO_ROOT / path
    if not file.is_file():
        return None
    with file.open("rb") as handle:
        head = handle.read(512)
    if head.startswith(b"version https://git-lfs"):
        match = re.search(rb"^size (\d+)", head, re.M)
        return int(match.group(1)) if match else None
    return file.stat().st_size


def lookahead_ms(model: dict) -> float | None:
    for artifact in model.get("artifacts") or []:
        manifest = artifact.get("manifest")
        if manifest and (REPO_ROOT / manifest).is_file():
            latency = json.loads(read(manifest)).get("algorithmic_latency") or {}
            if latency.get("lookahead_ms") is not None:
                return float(latency["lookahead_ms"])
    return None


def command(model: dict) -> str:
    inputs = " ".join(f"--input {name}={name}.wav" for name in model.get("inputs") or [])
    outputs = " --output audio=output.wav" if model.get("outputs") == ["audio"] else ""
    return f"uv run puresound infer {model['id']} \\\n  {inputs.replace('audio=audio.wav', 'audio=input.wav')}{outputs}"


def _order(model: dict) -> tuple:
    return (list(TASKS).index(model["task"]) if model["task"] in TASKS else len(TASKS),
            "default" not in (model.get("roles") or []), model["id"])


def _doc(toc_en: list, toc_zh: list, body: str) -> str:
    """Prose beside a table of its contents."""

    def contents(toc):
        return "".join(f'<li class="toc-{level}"><a href="#{esc(anchor)}">{esc(text)}</a></li>'
                       for level, text, anchor in toc)

    nav = bilingual(f"<ul>{contents(toc_en)}</ul>", f"<ul>{contents(toc_zh)}</ul>" if toc_zh else None)
    return (f'<div class="doc"><nav class="toc" aria-label="Contents"><span class="caps">{t("Contents")}</span>'
            f'{nav}</nav><article class="prose">{body}</article></div>')


DOC_CSS = """
.doc { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 32px; align-items: start; }
.toc { position: sticky; top: 16px; max-height: calc(100dvh - 32px); overflow: auto; font-size: var(--text-small); }
.toc ul { margin: 8px 0 0; padding: 0; list-style: none; }
.toc li { margin: 0; }
.toc a { display: block; padding: 4px 8px; color: var(--ink-2); border-left: 2px solid var(--line); text-decoration: none; line-height: 1.35; }
.toc a:hover { color: var(--ink); border-left-color: var(--ink-3); }
.toc .toc-3 a { padding-left: 20px; font-size: 11.5px; }
@media (max-width: 900px) { .doc { grid-template-columns: minmax(0, 1fr); } .toc { position: static; max-height: none; } }
"""


# ---------------------------------------------------------------- shared audio

#: Samples as a 16-bit WAV file, for downloads.
WAV_SCRIPT = """
window.PureSoundWav = (samples, rate) => {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((value, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, value)) * 0x7fff, true));
  return new Blob([view], { type: "audio/wav" });
};
"""


# ---------------------------------------------------------------- home

#: The clips the home page's player offers, and the task each one shows off.
DEMO_CLIPS = {"noisy-speech": "noise_suppression", "near-and-far": "voice_isolation"}

HOME_ZH = {
    "Clean up speech in real time, on an ordinary CPU": "在一般 CPU 上即時清理語音",
    "Remove background noise and distant talkers from a recording or a live stream. The released models "
    "run in this browser, from the command line, or inside your own app.":
        "把錄音或即時串流裡的背景噪音與遠處的說話聲去掉。已發布的模型可以直接在瀏覽器、指令列，或你自己的應用程式裡執行。",
    "Get started": "開始使用",
    "Clean it up": "開始清理", "Original": "原始", "Cleaned": "清理後",
    "Runs in this browser. Nothing is uploaded.": "在這個瀏覽器裡執行，不會上傳任何資料。",
    "Downloading the model": "下載模型中", "Cleaning {percent}%": "清理中 {percent}%",
    "Cleaned in {seconds} s on this device": "在這台裝置上花了 {seconds} 秒清理完成",
    "Pick a clip, then press Clean it up.": "選一段音訊，再按「開始清理」。",
    "This browser can't run the models here. Try the command line instead.": "這個瀏覽器無法在這裡執行模型，請改用指令列。",
    "What do you need?": "你需要什麼？",
    "How to run it": "執行方式",
    "In your browser": "在瀏覽器裡",
    "Nothing to install: run the released models on your own audio.": "不用安裝，直接用已發布的模型處理你的音訊。",
    "From the command line": "從指令列",
    "Process files with one command after installing with uv or pip.": "用 uv 或 pip 安裝後，一行指令就能處理檔案。",
    "In a local web app": "在本機 Web 介面",
    "Compare models, record from a microphone and inspect the results on your own machine.":
        "在自己的電腦上比較模型、用麥克風錄音並檢視結果。",
    "Inside your own app": "放進你的應用程式",
    "A standalone runtime that needs only NumPy, ONNX Runtime and the model file.":
        "獨立的 runtime，只需要 NumPy、ONNX Runtime 與模型檔。",
    "Install guide": "安裝說明", "Python SDK": "Python SDK", "Web SDK": "Web SDK",
    "Speed on one CPU thread": "單一 CPU 執行緒的速度",
    "Real-time factor of each default model in the latest CI run: 0.25 means a minute of audio takes "
    "15 seconds. Lower is faster; below 1 keeps up with a live stream.":
        "各預設模型在最近一次 CI 量測的即時率：0.25 代表一分鐘的音訊要 15 秒處理。數字越小越快；低於 1 就跟得上即時串流。",
    "See all benchmarks": "查看所有量測",
    "Learn more": "延伸閱讀", "Documentation": "文件", "Train your own model": "訓練自己的模型",
    "License": "授權",
    "Voice isolation model card": "人聲分離模型卡", "Noise suppression model card": "雜訊抑制模型卡",
    "Speaker verification model card": "語者驗證模型卡",
}

HOME_CSS = """
.hero { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 32px 40px; align-items: center;
  margin: 0 calc(-1 * max(24px, env(safe-area-inset-left, 0px))) 40px;
  padding: 48px max(24px, env(safe-area-inset-left, 0px)) 52px; color: var(--on-nav); background: var(--nav);
  border-bottom: 1px solid var(--nav-line); }
.hero h1 { margin: 0; max-width: 16ch; font-size: clamp(30px, 3.6vw, 44px); font-weight: 600; line-height: 1.1;
  letter-spacing: -0.025em; text-wrap: balance; }
.hero-text { max-width: 52ch; margin: 16px 0 0; color: var(--on-nav-2); font-size: var(--text-lead); line-height: 1.6; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 24px; }
.hero .button { height: 36px; padding: 0 14px; }
.hero .button-primary { color: var(--nav); background: var(--on-nav); border-color: var(--on-nav); }
.hero .button-primary:hover:not(:disabled) { color: var(--nav); background: var(--mint); border-color: var(--mint); }
.hero .button-secondary { color: var(--on-nav); background: transparent; border-color: var(--nav-line); }
.hero .button-secondary:hover:not(:disabled) { background: var(--nav-3); border-color: var(--on-nav-3); }

/* The player: the one thing on this page that does something. */
.player { min-width: 0; background: var(--nav-2); border: 1px solid var(--nav-line); border-radius: var(--radius-panel);
  overflow: hidden; box-shadow: 0 24px 64px rgba(0, 0, 0, 0.35); }
.player-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 14px 16px 6px; }
.player-clips { display: flex; flex-wrap: wrap; gap: 6px; }
.player-clips button { display: grid; gap: 1px; padding: 7px 12px 8px; color: var(--on-nav-2); background: transparent;
  border: 1px solid var(--nav-line); border-radius: var(--radius-panel); text-align: left; }
.player-clips button span { font-size: var(--text-ui); font-weight: 500; }
.player-clips button small { color: var(--on-nav-3); font-size: 11.5px; }
.player-clips button:hover { color: var(--on-nav); border-color: var(--on-nav-3); }
.player-clips button[aria-pressed="true"] { color: var(--on-nav); background: var(--nav-3); border-color: var(--on-nav-3); }
.player-head .private { margin-left: auto; color: var(--on-nav-3); font-size: 11.5px; }
.player .pp { background: transparent; border: 0; border-radius: 0; }
.player-status { min-height: 18px; margin: 0; padding: 0 16px 12px; color: var(--on-nav-2); background: var(--nav-2);
  font-size: var(--text-small); }
.player-status.is-run { color: var(--on-nav); }
.player-status.is-error { color: var(--nav-bad); }

.home-section { margin: 0 0 44px; }
.home-section > h2 { margin: 0 0 6px; font-size: var(--text-title); font-weight: 600; letter-spacing: -0.01em; }
.home-section > .lead { max-width: 72ch; margin: 0 0 16px; color: var(--ink-2); font-size: var(--text-body); line-height: 1.55; }
.home-section > .lead p { margin: 0; }
.tasks { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 12px; }
.task-tile { display: flex; flex-direction: column; gap: 8px; }
.task-tile h3 { margin: 4px 0 0; font-size: var(--text-lead); font-weight: 600; }
.task-tile .use { margin: 0; color: var(--ink); font-size: var(--text-body); line-height: 1.5; }
.task-tile .use p { margin: 0; }
.task-tile .model-card-subtitle { margin: 0; }
.task-tile .card-links { margin-top: auto; padding-top: 8px; }

/* Ways to run it: one row, divided by rules rather than boxed. */
.ways { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid var(--line); }
.way { display: flex; flex-direction: column; gap: 8px; min-width: 0; padding: 16px 20px 4px 0; }
.way + .way { padding-left: 20px; border-left: 1px solid var(--line); }
.way h3 { margin: 0; font-size: var(--text-lead); font-weight: 600; }
.way p { margin: 0; color: var(--ink-2); font-size: var(--text-ui); line-height: 1.5; }
.way code.cmd { display: block; padding: 8px 10px; overflow-wrap: anywhere; color: var(--on-nav); background: var(--nav);
  border-radius: var(--radius-control); font-size: 11.5px; line-height: 1.5; white-space: pre-wrap; }
.way .links { display: flex; flex-wrap: wrap; gap: 12px; margin-top: auto; padding-top: 4px; font-size: var(--text-ui); }
.way .links a, .more a { color: var(--ink); text-decoration-color: var(--line-strong); text-underline-offset: 3px; }
.way .links a:hover, .more a:hover { text-decoration-color: currentColor; }

.speed { max-width: 760px; overflow-x: auto; background: var(--canvas); border: 1px solid var(--line); border-radius: var(--radius-panel); }
.speed th.platform { text-transform: none; letter-spacing: 0; }
.speed td.model strong { display: block; font-weight: 500; }
.speed td.model span { color: var(--ink-3); font-family: var(--font-mono); font-size: 11px; }
.speed-link { margin: 10px 0 0; font-size: var(--text-ui); }
.more { display: flex; flex-wrap: wrap; gap: 8px 24px; font-size: var(--text-body); }

@media (max-width: 980px) {
  .hero { grid-template-columns: minmax(0, 1fr); padding-top: 36px; }
  .ways { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .way:nth-child(3) { padding-left: 0; border-left: 0; }
  .way:nth-child(n + 3) { border-top: 1px solid var(--line); }
}
@media (max-width: 560px) {
  .ways { grid-template-columns: minmax(0, 1fr); }
  .way + .way { padding-left: 0; border-left: 0; border-top: 1px solid var(--line); }
}
"""

HOME_SCRIPT = """
(() => {
  const site = window.PureSoundSite, device = window.PureSoundDevice;
  const frame = document.getElementById("player");
  if (!frame) return;
  const CLIPS = JSON.parse(document.getElementById("demo-clips").textContent);
  const player = new window.PureSoundPlayer(frame.querySelector("[data-player]"), site.tr);
  const run = document.getElementById("clean"), status = document.getElementById("player-status");
  let clip = null, busy = false, token = 0;

  const fmt = (text, vars = {}) => site.tr(text).replace(/\\{(\\w+)\\}/g, (m, k) => (vars[k] ?? m));
  function say(text, vars = {}, kind = "") {
    status.dataset.text = text; status.dataset.vars = JSON.stringify(vars);
    status.textContent = text ? fmt(text, vars) : "";
    status.className = "player-status" + (kind ? " is-" + kind : "");
  }
  async function choose(id) {
    clip = CLIPS.find((c) => c.id === id);
    const mine = ++token;
    for (const button of frame.querySelectorAll("[data-clip]")) button.setAttribute("aria-pressed", String(button.dataset.clip === id));
    player.progress(null);
    say("Pick a clip, then press Clean it up.");
    try {
      const decoded = await device.decode(await (await fetch(clip.file)).blob(), { sampleRate: 16000 });
      if (mine === token) player.load(decoded.samples, decoded.sampleRate);
    } catch (error) {
      if (mine === token) say(error.message || String(error), {}, "error");
    }
  }
  async function clean() {
    if (busy || !clip || !player.input) return;
    busy = true; run.disabled = true;
    const mine = token;
    try {
      const result = await device.process(player.input, {
        model: clip.model, sampleRate: player.rate,
        onProgress: (value, phase) => {
          if (mine !== token) return;
          if (phase === "load") { say("Downloading the model", {}, "run"); player.progress(0); }
          else { player.progress(value); say("Cleaning {percent}%", { percent: Math.round(value * 100) }, "run"); }
        },
      });
      if (mine !== token) return;
      player.progress(null);
      player.setOutput(result.output);
      say("Cleaned in {seconds} s on this device", { seconds: result.seconds.toFixed(1) });
    } catch (error) {
      player.progress(null);
      say(error.message || String(error), {}, "error");
    } finally {
      busy = false; run.disabled = false;
    }
  }
  for (const button of frame.querySelectorAll("[data-clip]")) button.addEventListener("click", () => choose(button.dataset.clip));
  run.addEventListener("click", clean);
  site.onLanguage.push(() => {
    if (status.dataset.text) status.textContent = fmt(status.dataset.text, JSON.parse(status.dataset.vars || "{}"));
    player.update();
  });
  choose(CLIPS[0].id);
  device.status().then((state) => {
    if (state.ready && CLIPS.every((c) => state.models.includes(c.model))) return;
    run.disabled = true;
    say("This browser can't run the models here. Try the command line instead.", {}, "error");
  });
})();
"""


def readme_uses(path: str) -> dict[str, str]:
    """Each model's "use it for" cell in a README's released-models table."""
    uses = {}
    for line in read(path).splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and (match := re.fullmatch(r"`([^`]+)`", cells[1])):
            uses[match.group(1)] = cells[-1]
    return uses


def _inline(text: str, source: str, links: Links) -> str:
    return render(text, source=source, links=links).html


def _speed(site: Site, history: dict) -> str:
    from site_benchmarks import LABELS, _latest_item

    runs = history.get("runs") or []
    if not runs:
        return ""
    latest = runs[-1]
    platforms = [p for p in LABELS if p in latest["platforms"]]
    rows = []
    for task in TASKS:
        models = sorted((m for m in site.catalog if m["task"] == task
                         and (m.get("capabilities") or {}).get("realtime_streaming")), key=_order)
        if not models:
            continue
        model = models[0]
        cells = []
        for platform in platforms:
            item = _latest_item(latest, model["id"], "default", platform)
            cells.append(f'<td class="num">{float(item["rtf"]):.2f}</td>' if item else '<td class="num quiet">—</td>')
        latency = lookahead_ms(model)
        rows.append(f'<tr><td class="model"><strong>{t(TASKS[task])}</strong><span>{esc(model["id"])}</span></td>'
                    f'{"".join(cells)}<td class="num">{f"{latency:.0f} ms" if latency is not None else "—"}</td></tr>')
    if not rows:
        return ""
    head = "".join(f'<th class="num platform">{LABELS[p]}</th>' for p in platforms)
    return f"""<section class="home-section">
<h2>{t("Speed on one CPU thread")}</h2>
<p class="lead">{t("Real-time factor of each default model in the latest CI run: 0.25 means a minute of audio takes 15 seconds. Lower is faster; below 1 keeps up with a live stream.")}</p>
<div class="speed"><table class="table"><thead><tr><th>{t("Model")}</th>{head}<th class="num">{t("Look-ahead")}</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="speed-link"><a href="benchmarks.html">{t("See all benchmarks")}</a></p>
</section>"""


def home(site: Site, history: dict, samples: list[dict]) -> str:
    links = site.links
    catalog = site.catalog
    zh_readme = translation("README.md")
    uses_en = readme_uses("README.md")
    uses_zh = readme_uses(zh_readme) if zh_readme else {}
    defaults = {task: sorted((m for m in catalog if m["task"] == task), key=_order)[0]
                for task in TASKS if any(m["task"] == task for m in catalog)}

    # The player: clips whose task's default model this browser build carries.
    clips = []
    for clip in samples:
        task = DEMO_CLIPS.get(clip["id"])
        if task in defaults and defaults[task]["id"] in site.device_models:
            clips.append({"id": clip["id"], "file": f"samples/{clip['file']}", "model": defaults[task]["id"],
                          "title": clip.get("title") or clip["id"], "task": task})
    clips.sort(key=lambda clip: list(DEMO_CLIPS).index(clip["id"]))
    player = ""
    if clips:
        buttons = "".join(f'<button type="button" aria-pressed="false" data-clip="{esc(c["id"])}">'
                          f'<span>{t(c["title"])}</span><small>{t(TASKS[c["task"]])}</small></button>' for c in clips)
        clean = f'<button class="button button-primary" id="clean" type="button">{t("Clean it up")}</button>'
        player = f"""<div class="player" id="player">
<div class="player-head"><div class="player-clips" role="group" aria-label="Samples">{buttons}</div>
<span class="private">{t("Runs in this browser. Nothing is uploaded.")}</span></div>
{site_player.markup("Original", "Cleaned", clean)}
<p class="player-status" id="player-status" role="status"></p>
</div>
<script type="application/json" id="demo-clips">{json.dumps(clips).replace("</", "<\\/")}</script>"""

    tiles = []
    for task, default in defaults.items():
        card = card_of(default)
        use = bilingual(_inline(uses_en[default["id"]][:1].upper() + uses_en[default["id"]][1:], "README.md", links),
                        _inline(uses_zh[default["id"]], zh_readme, links) if default["id"] in uses_zh else None) \
            if default["id"] in uses_en else esc(default.get("description") or "")
        actions = []
        if default["id"] in site.device_models:
            actions.append(f'<a class="card-link" href="try.html?model={esc(default["id"])}">{t("Try it")}</a>')
        if card in site.cards:
            actions.append(f'<a class="card-link" href="{esc(site.cards[card])}">{t("Model card")}</a>')
        actions.append(f'<a class="card-link" href="models.html?task={esc(task)}">{t("Models")}</a>')
        speaker = " sv" if task == "speaker_embedding" else ""
        tiles.append(f"""<article class="model-card task-tile">
<div class="model-card-top"><div class="model-symbol{speaker}" aria-hidden="true">{SYMBOLS.get(task, "∿")}</div>
<span class="badge badge-default">{t("default")}</span></div>
<h3>{t(TASKS[task])}</h3>
<div class="use">{use}</div>
<p class="model-card-subtitle">{esc(default["id"])}</p>
<div class="card-links">{"".join(actions)}</div>
</article>""")

    vi = defaults.get("voice_isolation") or next(iter(defaults.values()))
    sdk_python = links.to("sdk/python/README.md", "README.md")
    sdk_web = links.to("sdk/web/README.md", "README.md")
    ways = [
        ("In your browser", "Nothing to install: run the released models on your own audio.", None,
         [("try.html", "Try it")] if site.device_models else []),
        ("From the command line", "Process files with one command after installing with uv or pip.",
         f"uv run puresound infer {vi['id']} --input audio=in.wav --output audio=out.wav",
         [("start.html#quick-start", "Install guide")]),
        ("In a local web app", "Compare models, record from a microphone and inspect the results on your own machine.",
         "uv run puresound web", [("start.html#quick-start", "Get started")]),
        ("Inside your own app", "A standalone runtime that needs only NumPy, ONNX Runtime and the model file.", None,
         [(sdk_python, "Python SDK"), (sdk_web, "Web SDK")]),
    ]
    way_html = "".join(
        f'<div class="way"><h3>{t(title)}</h3><p>{t(text)}</p>'
        + (f'<code class="cmd">{esc(cmd)}</code>' if cmd else "")
        + '<div class="links">' + "".join(f'<a href="{esc(href)}">{t(label)}</a>' for href, label in targets)
        + "</div></div>"
        for title, text, cmd, targets in ways
    )
    more = [(links.to("docs/index.md", "README.md"), t("Documentation")),
            ("start.html#train-a-model", t("Train your own model"))]
    for card_path, target in site.cards.items():
        task = next(m["task"] for m in catalog if card_of(m) == card_path)
        more.append((target, t(f"{TASKS.get(task, task)} model card")))
    more.append((links.to("LICENSE", "README.md"), t("License")))

    github = f'<a class="button button-secondary" href="{esc(links.repository)}">GitHub</a>' if links.repository else ""
    primary = (f'<a class="button button-primary" href="try.html">{t("Try it in your browser")}</a>'
               if site.device_models else "")
    body = f"""<section class="hero">
<div><h1>{t("Clean up speech in real time, on an ordinary CPU")}</h1>
<p class="hero-text">{t("Remove background noise and distant talkers from a recording or a live stream. The released models run in this browser, from the command line, or inside your own app.")}</p>
<div class="hero-actions">{primary}<a class="button button-secondary" href="start.html">{t("Get started")}</a>{github}</div></div>
{player}
</section>
<section class="home-section"><h2>{t("What do you need?")}</h2><div class="tasks">{"".join(tiles)}</div></section>
<section class="home-section"><h2>{t("How to run it")}</h2><div class="ways">{way_html}</div></section>
{_speed(site, history)}
<section class="home-section"><h2>{t("Learn more")}</h2>
<nav class="more">{"".join(f'<a href="{esc(href)}">{label}</a>' for href, label in more)}</nav></section>"""
    head = '<script src="device/device.js"></script>' if player else ""
    return page(title="Home", active="index.html", body=body, repository=links.repository,
                extra_css=site_player.CSS + HOME_CSS, extra_js=site_player.SCRIPT + HOME_SCRIPT,
                zh={**ZH_TW, **HOME_ZH, **site_player.ZH_TW}, head=head,
                footer=_footer(site))


# ---------------------------------------------------------------- get started

def start(site: Site) -> str:
    """The README in full, after its released-models table."""
    links = site.links
    project_url = f"{links.repository}.git" if links.repository else "<project-url>"

    def parts(path, prefix):
        preamble, sections = split_sections(read(path).replace("<project-url>", project_url))
        # The README's own licence badge; the License section says it in full.
        intro = render(re.sub(r"^\[[^\]\n]*\]\(LICENSE\)\s*$", "", preamble, flags=re.M), source=path, links=links)
        rest = "".join(f"## {heading}\n{body}" for heading, body in sections[1:])
        return intro, render(rest, source=path, links=links, anchor_prefix=prefix)

    intro_en, body_en = parts("README.md", "")
    zh_path = translation("README.md")
    intro_zh, body_zh = parts(zh_path, "zh-") if zh_path else (None, None)
    body = f"""<header class="page-header is-static">
<div class="page-heading"><h1>{t("Get started")}</h1>
<div class="page-desc">{bilingual(intro_en.html, intro_zh.html if intro_zh else None)}</div></div>
<div class="page-actions"><a class="button button-secondary button-small" href="{esc(links.to("README.md", "README.md"))}">{t("Source on GitHub")}</a></div>
</header>
{_doc(body_en.toc, body_zh.toc if body_zh else [], bilingual(body_en.html, body_zh.html if body_zh else None))}"""
    return page(title="Get started", active="start.html", body=body, repository=links.repository,
                extra_css=DOC_CSS + ".page-desc p { margin: 0; }", zh={**ZH_TW, **HOME_ZH}, footer=_footer(site))


def _footer(site: Site) -> str:
    links = site.links
    if not links.repository:
        return ""
    return (f'{t("Built from")} <a href="{esc(links.repository)}/tree/{esc(links.revision)}">'
            f'<code>{esc(links.revision[:12])}</code></a> · Apache-2.0')


# ---------------------------------------------------------------- models

MODELS_CSS = """
.model-card:target { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.downloads { display: grid; gap: 6px; margin: 12px 0 0; padding: 10px 0 0; list-style: none; border-top: 1px solid var(--line);
  font-size: var(--text-small); }
.downloads li { display: grid; grid-template-columns: 4.5em minmax(0, 1fr) auto; gap: 4px 8px; align-items: baseline; }
.downloads .variant-name { color: var(--ink); font-weight: 500; }
.downloads .note { grid-column: 2 / -1; color: var(--ink-2); line-height: 1.4; }
.downloads a { color: var(--ink); text-underline-offset: 3px; }
.command { margin: 12px 0 0; padding: 9px 11px; overflow-x: auto; color: var(--on-nav); background: var(--nav);
  border-radius: var(--radius-panel); font-family: var(--font-mono); font-size: 11.5px; line-height: 1.5;
  white-space: pre-wrap; overflow-wrap: anywhere; }
.model-card .model-footer { margin-top: auto; }
.model-card .command + .model-footer, .model-card .downloads + .model-footer { margin-top: 12px; }
"""


def _model_card(model: dict, site: Site) -> str:
    links = site.links
    task = model["task"]
    roles = model.get("roles") or []
    speaker = " sv" if task == "speaker_embedding" else ""
    badges = [f'<span class="badge badge-task{speaker}">{t(TASKS.get(task, task))}</span>']
    badges += [f'<span class="badge badge-{esc(role)}">{t(role)}</span>' for role in roles]
    facts = [(t("Sample rate"), f'{(model.get("audio") or {}).get("sample_rate", 16000) / 1000:g} kHz'),
             (t("Streaming"), t("yes" if (model.get("capabilities") or {}).get("realtime_streaming") else "no"))]
    if (latency := lookahead_ms(model)) is not None:
        facts.append((t("Look-ahead"), f"{latency:.0f} ms"))
    for key, value in (model.get("recommended_inference") or {}).items():
        facts.append((esc(key), esc(value)))
    downloads = []
    for artifact in model.get("artifacts") or []:
        size = file_size(artifact["path"])
        href = links.to(artifact["path"], "README.md", raw=True)
        note = artifact.get("description")
        note_html = esc(note) if note else (t("float weights") if artifact.get("variant") == "default" else "")
        downloads.append(
            f'<li><span class="variant-name">{esc(artifact.get("variant"))}</span>'
            f'<span class="mono quiet">{f"{size / 1e6:.1f} MB · " if size else ""}{esc(artifact.get("format"))}</span>'
            f'<a href="{esc(href)}" title="SHA-256 {esc(artifact.get("sha256", ""))}" download>{t("Download")}</a>'
            + (f'<span class="note">{note_html}</span>' if note_html else "") + "</li>"
        )
    actions = []
    card = card_of(model)
    if card in site.cards:
        actions.append(f'<a class="card-link" href="{esc(site.cards[card])}">{t("Model card")}</a>')
    if model["id"] in site.device_models:
        actions.append(f'<a class="card-link" href="try.html?model={esc(model["id"])}">{t("Try it")}</a>')
    if (model.get("capabilities") or {}).get("realtime_streaming"):
        actions.append(f'<a class="card-link" href="benchmarks.html#{esc(model["id"])}">{t("Benchmarks")}</a>')
    text = " ".join([model["id"], model.get("display_name") or "", task, " ".join(roles),
                     model.get("description") or ""]).lower()
    return f"""<article class="model-card{" is-default" if "default" in roles else ""}" id="{esc(model["id"])}"
 data-model="{esc(model["id"])}" data-task="{esc(task)}" data-roles="{esc(" ".join(roles))}" data-text="{esc(text)}">
<div class="model-card-top"><div class="model-symbol{speaker}" aria-hidden="true">{SYMBOLS.get(task, "∿")}</div>
<span class="badge badge-quiet">{t(model.get("lifecycle") or "released")}</span></div>
<div class="card-heading"><h3>{esc(model.get("display_name") or model["id"])}</h3>
<p class="model-card-subtitle">{esc(model["id"])}</p></div>
<div class="model-badges">{"".join(badges)}</div>
<p class="model-description" data-clamp>{esc(model.get("description") or "")}</p>
<button class="card-more" type="button" data-more hidden>{t("More")}</button>
<dl class="facts">{"".join(f"<div><dt>{label}</dt><dd class='mono'>{value}</dd></div>" for label, value in facts)}</dl>
<ul class="downloads">{"".join(downloads)}</ul>
<pre class="command"><code>{esc(command(model))}</code></pre>
<div class="model-footer"><div class="card-links">{"".join(actions)}</div></div>
</article>"""


def models(site: Site) -> str:
    catalog = sorted(site.catalog, key=_order)
    tasks = [task for task in TASKS if any(model["task"] == task for model in catalog)]
    roles = sorted({role for model in catalog for role in model.get("roles") or []},
                   key=lambda role: (role != "default", role))

    def lead(path):
        # The README's account of the catalog: its released-models section up to the table.
        _, sections = split_sections(read(path))
        text = sections[0][1].split("\n|", 1)[0] if sections else ""
        return render(text, source=path, links=site.links).html

    zh = translation("README.md")
    body = f"""<header class="page-header">
<div class="page-heading"><h1>{t("Model zoo")}</h1><div class="page-desc">{bilingual(lead("README.md"), lead(zh) if zh else None)}</div></div>
<div class="page-actions"><a class="button button-secondary button-small" href="{esc(site.links.to(CATALOG, "README.md"))}">catalog.yaml</a></div>
</header>
<div class="toolbar" id="filters">
<label class="search-field"><span aria-hidden="true">⌕</span><input id="q" type="search" autocomplete="off"
 placeholder="Search models" aria-label="Search models" data-i18n-attr="placeholder,aria-label"></label>
<div class="filter-group"><span class="caps">{t("Task")}</span><div class="segmented" role="group" aria-label="Task">{segmented("task", tasks, TASKS)}</div></div>
<div class="filter-group"><span class="caps">{t("Role")}</span><div class="segmented" role="group" aria-label="Role">{segmented("role", roles, {})}</div></div>
<span class="toolbar-count" id="count">{len(catalog)} of {len(catalog)} models</span>
</div>
<div class="model-grid" id="grid">{"".join(_model_card(model, site) for model in catalog)}
<div class="empty-state" id="no-match" hidden>{t("No models match these filters.")}</div></div>"""
    return page(title="Models", active="models.html", body=body, repository=site.links.repository,
                extra_css=MODELS_CSS + ".page-desc p { margin: 0; }", extra_js=FILTER_SCRIPT, zh=ZH_TW,
                footer=_footer(site))


# ---------------------------------------------------------------- model cards

def card(path: str, target: str, site: Site) -> str:
    links = Links(site.links.repository, site.links.revision, site.links.pages, prefix="../")
    english = render(read(path), source=path, links=links)
    zh_path = translation(path)
    chinese = render(read(zh_path), source=zh_path, links=links, anchor_prefix="zh-") if zh_path else None
    covered = [model for model in sorted(site.catalog, key=_order) if card_of(model) == path]
    chips = "".join(f'<a class="chip" href="../models.html#{esc(m["id"])}">{esc(m["id"])}'
                    + (f' <small>{esc(" ".join(m.get("roles") or []))}</small>' if m.get("roles") else "") + "</a>"
                    for m in covered)
    title_en = english.title or path
    source = links.to(path, "README.md")
    body = f"""<header class="page-header is-static">
<div class="page-heading"><span class="caps">{t("Model card")}</span>
<h1>{bilingual(esc(title_en), esc(chinese.title) if chinese and chinese.title else None, "span")}</h1>
<div class="card-models" aria-label="Models in this card">{chips}</div></div>
<div class="page-actions"><a class="button button-secondary button-small" href="{esc(source)}">{t("Source on GitHub")}</a></div>
</header>
{_doc(english.toc, chinese.toc if chinese else [], bilingual(english.html, chinese.html if chinese else None))}"""
    css = DOC_CSS + """
.card-models { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.card-models .chip { text-decoration: none; }
"""
    return page(title=title_en, active="models.html", body=body, prefix="../", repository=links.repository,
                extra_css=css, zh=ZH_TW, footer=_footer(site))


# ---------------------------------------------------------------- try it

TRY_CSS = """
.try { display: grid; grid-template-columns: minmax(280px, 380px) minmax(0, 1fr); gap: 12px; align-items: start; }
.try > .panel { min-width: 0; }
.try .field { grid-template-columns: minmax(0, 1fr); }
.sample-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.sample-row .chip { max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.dropzone { min-width: 0; }
.dropzone .upload-icon { flex: none; }
.drop-text { display: grid; gap: 3px; min-width: 0; }
.drop-text > span { color: var(--ink-3); font-size: var(--text-small); line-height: 1.4; }
.dropzone.has-file { align-items: center; padding: 10px 12px; }
.dropzone.has-file .drop-text > * { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dropzone.has-file .drop-text > span { color: var(--ink-2); }
.run-row .button { min-width: 96px; }
.model-note { margin: 6px 0 0; color: var(--ink-2); font-size: var(--text-small); line-height: 1.45; }
.run-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 18px; }
.results { display: grid; gap: 12px; min-width: 0; }
.results .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 140px), 1fr)); gap: 8px; }
@media (max-width: 900px) { .try { grid-template-columns: minmax(0, 1fr); } }
"""

TRY_SCRIPT = """
(() => {
  const site = window.PureSoundSite;
  const device = window.PureSoundDevice;
  const MODELS = JSON.parse(document.getElementById("try-models").textContent);
  const $ = (id) => document.getElementById(id);
  const select = $("model"), blend = $("blend"), run = $("run"), state = $("state"), note = $("note");
  const file = $("file"), download = $("download");
  const player = new window.PureSoundPlayer(document.querySelector("[data-player]"), site.tr);
  let sourceName = "", busy = false, token = 0, outputUrl = null;

  const fmt = (text, vars) => site.tr(text).replace(/\\{(\\w+)\\}/g, (m, k) => (vars[k] ?? m));
  function setState(kind, text, vars = {}) {
    state.className = "state" + (kind ? " is-" + kind : "");
    state.dataset.text = text;
    state.dataset.vars = JSON.stringify(vars);
    state.textContent = fmt(text, vars);
  }
  function showNote(text, error = false) {
    note.textContent = text ? site.tr(text) : "";
    note.classList.toggle("is-error", error);
  }

  function describe() {
    const model = MODELS.find((m) => m.id === select.value);
    if (!model) return;
    $("model-note").textContent = model.description || "";
    $("model-card").hidden = !model.card;
    if (model.card) $("model-card").href = model.card;
    blend.value = model.dry_blend;
    $("blend-value").value = Number(blend.value).toFixed(2);
  }

  /* The drop target's words: an invitation, or once a file is chosen, a replace bar. */
  function dropText() {
    const chosen = $("drop").classList.contains("has-file");
    $("drop-title").textContent = site.tr(chosen ? "Replace audio" : "Drop audio here");
    $("drop-hint").textContent = chosen ? fmt("{name} · drop or click to choose another file", { name: sourceName })
      : site.tr("or click to browse · any rate, prepared to mono 16 kHz");
  }

  /* A recording to run: decoded at the model rate and shown in the player. */
  async function pick(blob, name, chip = null) {
    const mine = ++token;
    sourceName = name;
    for (const other of document.querySelectorAll("[data-sample]")) {
      other.classList.toggle("is-active", other === chip);
      other.setAttribute("aria-pressed", String(other === chip));
    }
    $("drop").classList.toggle("has-file", !chip);
    dropText();
    showNote("");
    download.hidden = true;
    player.progress(null);
    try {
      const { samples, sampleRate } = await device.decode(blob, { sampleRate: 16000 });
      if (mine === token) player.load(samples, sampleRate);
    } catch (error) {
      if (mine === token) showNote(error.message || String(error), true);
    }
  }

  async function start() {
    if (busy) return;
    if (!player.input) { showNote("Choose a sample or a file first.", true); return; }
    busy = true;
    run.disabled = true;
    const mine = token;
    showNote("");
    setState("run", "Running");
    try {
      const result = await device.process(player.input, {
        model: select.value, sampleRate: player.rate, dryBlend: Number(blend.value),
        onProgress: (value, phase) => {
          if (mine !== token) return;
          if (phase === "load") { setState("run", "Downloading the model"); player.progress(0); }
          else {
            player.progress(value);
            setState("run", "Processing {percent}%", { percent: Math.round(value * 100) });
          }
        },
      });
      if (mine !== token) return;
      player.progress(null);
      player.setOutput(result.output);
      if (outputUrl) URL.revokeObjectURL(outputUrl);
      outputUrl = URL.createObjectURL(window.PureSoundWav(result.output, result.sampleRate));
      download.href = outputUrl;
      download.download = sourceName.replace(/\\.[^.]+$/, "") + "." + result.model + ".wav";
      download.hidden = false;
      $("rtf").textContent = result.rtf.toFixed(3);
      $("threads").textContent = result.threads;
      $("latency").textContent = Math.round(result.latencyMs) + " ms";
      $("seconds").textContent = result.seconds.toFixed(1) + " s";
      $("figures").hidden = false;
      setState("ok", "Done in {seconds} s", { seconds: result.seconds.toFixed(1) });
    } catch (error) {
      player.progress(null);
      setState("err", "Unavailable");
      showNote(error.message || String(error), true);
    } finally {
      busy = false;
      run.disabled = false;
    }
  }

  for (const chip of document.querySelectorAll("[data-sample]")) {
    chip.addEventListener("click", async () => {
      const response = await fetch(chip.dataset.sample);
      pick(await response.blob(), chip.dataset.sample.split("/").pop(), chip);
    });
  }
  file.addEventListener("change", () => { if (file.files[0]) pick(file.files[0], file.files[0].name); });
  const drop = $("drop");
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("is-dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("is-dragging"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("is-dragging");
    const dropped = event.dataTransfer.files[0];
    if (dropped) pick(dropped, dropped.name);
  });
  select.addEventListener("change", describe);
  blend.addEventListener("input", () => { $("blend-value").value = Number(blend.value).toFixed(2); });
  run.addEventListener("click", start);
  site.onLanguage.push(() => {
    if (state.dataset.text) state.textContent = fmt(state.dataset.text, JSON.parse(state.dataset.vars || "{}"));
    player.update();
    dropText();
  });

  const params = new URLSearchParams(location.search);
  if (MODELS.some((m) => m.id === params.get("model"))) select.value = params.get("model");
  describe();
  const samples = [...document.querySelectorAll("[data-sample]")];
  (samples.find((chip) => chip.dataset.sampleId === params.get("sample")) || samples[0])?.click();
  device.status().then((status) => {
    if (!status.ready) {
      setState("err", "Unavailable");
      showNote(status.reason, true);
      run.disabled = true;
      return;
    }
    for (const option of select.options) option.disabled = !status.models.includes(option.value);
    $("isolation").hidden = status.isolated;
    setState("ok", "Ready");
  });
})();
"""


def try_it(site: Site, samples: list[dict]) -> str:
    models = [model for model in sorted(site.catalog, key=_order) if model["id"] in site.device_models]
    data = [{"id": m["id"], "task": m["task"], "description": m.get("description") or "",
             "card": site.cards.get(card_of(m)), "dry_blend": (m.get("recommended_inference") or {}).get("dry_blend", 1.0)}
            for m in models]
    groups = []
    for task in TASKS:
        options = "".join(f'<option value="{esc(m["id"])}">{esc(m.get("display_name") or m["id"])}'
                          f'{" · " + esc(", ".join(m.get("roles") or [])) if m.get("roles") else ""}</option>'
                          for m in models if m["task"] == task)
        if options:
            groups.append(f'<optgroup label="{esc(TASKS[task])}" data-i18n-attr="label">{options}</optgroup>')
    chips = "".join(f'<button type="button" class="chip" aria-pressed="false" data-sample="samples/{esc(clip["file"])}"'
                    f' data-sample-id="{esc(clip["id"])}"'
                    f' title="{esc(clip.get("description") or "")}">{t(clip.get("title") or clip["id"])}'
                    f' <small>{float(clip.get("seconds") or 0):.1f} s</small></button>' for clip in samples)
    if not models:
        body = f"""<header class="page-header"><div class="page-heading"><h1>{t("Try a model in your browser")}</h1></div></header>
<div class="empty-state">{t("This build of the site has no on-device payload.")}</div>"""
        return page(title="Try it", active="try.html", body=body, repository=site.links.repository, zh=ZH_TW)
    download = f'<a class="button button-secondary" id="download" href="#" hidden>{t("Download output")}</a>'
    body = f"""<header class="page-header">
<div class="page-heading"><h1>{t("Try a model in your browser")}</h1>
<p class="page-desc">{t("The released streaming models run on this device in ONNX Runtime Web (WebAssembly). Audio never leaves the browser.")}</p></div>
<div class="page-actions"><span class="state" id="state"></span></div>
</header>
<div class="try">
<section class="panel panel-pad">
<div class="field"><label for="model">{t("Model")}</label><select class="control" id="model">{"".join(groups)}</select>
<p class="model-note"><span id="model-note"></span> <a id="model-card" href="#" hidden>{t("Model card")}</a></p></div>
<div class="field"><span class="field-name">{t("Audio")}</span>
<div class="sample-row" role="group" aria-label="Samples">{chips}</div>
<label class="dropzone" id="drop"><input id="file" type="file" accept="audio/*,.wav,.flac,.mp3,.m4a"><span class="upload-icon" aria-hidden="true">↑</span>
<span class="drop-text"><strong id="drop-title">{t("Drop audio here")}</strong>
<span id="drop-hint">{t("or click to browse · any rate, prepared to mono 16 kHz")}</span></span></label></div>
<div class="field-label range-label"><label for="blend">{t("Model output share")}</label><output id="blend-value" class="mono">1.00</output></div>
<input class="range" id="blend" type="range" min="0.1" max="1" step="0.05" value="1">
<div class="range-hint"><span>{t("input")}</span><span>{t("model")}</span></div>
<p class="toggle-hint">{t("dry_blend: the share of the model output mixed with the untouched input.")}</p>
<div class="run-row"><button class="button button-primary" id="run" type="button">{t("Run")}</button></div>
<p class="form-note" id="note" role="status"></p>
<p class="panel-help" id="isolation" hidden>{t("This page is not cross-origin isolated, so models run on one thread here.")}</p>
</section>
<section class="results">
{site_player.markup("Input", "Output", download)}
<div id="figures" hidden>
<div class="stats">
<div class="stat"><span>{t("Real-time factor")}</span><strong id="rtf">—</strong></div>
<div class="stat"><span>{t("Threads")}</span><strong id="threads">—</strong></div>
<div class="stat"><span>{t("Look-ahead")}</span><strong id="latency">—</strong></div>
<div class="stat"><span>{t("Processing time")}</span><strong id="seconds">—</strong></div>
</div>
<p class="panel-help">{t("Lower is faster; below 1 keeps up with real time.")}</p>
</div>
</section>
</div>
<script type="application/json" id="try-models">{json.dumps(data).replace("</", "<\\/")}</script>"""
    return page(title="Try it", active="try.html", body=body, repository=site.links.repository,
                extra_css=site_player.CSS + TRY_CSS, extra_js=WAV_SCRIPT + site_player.SCRIPT + TRY_SCRIPT,
                zh={**ZH_TW, **site_player.ZH_TW}, head='<script src="device/device.js"></script>',
                footer=_footer(site))
