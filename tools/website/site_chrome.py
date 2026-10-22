"""What every page of the project site shares: the frame, navigation, theme and
language switches, and the playground's design system copied beside them.

The site is drawn with the web playground's own stylesheets (tokens, base,
components) and fonts, read from ``puresound/web/static``, so a change to the
playground's look reaches the site on its next build.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "puresound/web/static"
#: The playground stylesheets every screen shares; the site uses them unchanged.
SHARED_CSS = ("tokens.css", "base.css", "components.css")

TASKS = {
    "voice_isolation": "Voice isolation",
    "noise_suppression": "Noise suppression",
    "speaker_embedding": "Speaker verification",
}

NAV = (
    ("index.html", "Home"),
    ("start.html", "Get started"),
    ("models.html", "Models"),
    ("try.html", "Try it"),
    ("benchmarks.html", "Benchmarks"),
)

#: The site's own words in Traditional Chinese, keyed by their English. Text
#: rendered from the repository arrives in both languages already.
ZH_TW = {
    "Home": "首頁", "Get started": "開始使用", "Models": "模型庫", "Try it": "線上試用", "Benchmarks": "效能量測",
    "Theme": "主題", "Auto": "自動", "Light": "淺色", "Dark": "深色", "Language": "語言",
    "Voice isolation": "人聲分離", "Noise suppression": "雜訊抑制", "Speaker verification": "語者驗證",
    "default": "預設", "candidate": "候選", "alternative": "替代", "fallback": "備援",
    "released": "已發布", "reference": "參考", "experimental": "實驗",
    "All": "全部", "More": "展開", "Less": "收合",
    "Search models": "搜尋模型", "Task": "任務", "Role": "角色",
    "{shown} of {total} models": "{shown} / {total} 個模型",
    "No models match these filters.": "沒有符合篩選條件的模型。",
    "Model card": "模型卡", "Contents": "目錄",
    "Source on GitHub": "GitHub 原始檔",
    "Built from": "產生自",
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def t(text: str) -> str:
    """English page text the reader can switch to Traditional Chinese."""
    return f'<span data-i18n="{esc(text)}">{esc(text)}</span>'


def bilingual(english: str, chinese: str | None, tag: str = "div") -> str:
    """Content written in both languages; the language switch shows one."""
    if not chinese:
        return english
    return f'<{tag} class="lang-en" lang="en">{english}</{tag}><{tag} class="lang-zh" lang="zh-TW">{chinese}</{tag}>'


def segmented(name: str, values: list[str], labels: dict[str, str]) -> str:
    """One filter's choices, "All" first and pressed."""
    buttons = [f'<button type="button" aria-pressed="true" data-filter="{name}" data-value="all">{t("All")}</button>']
    buttons += [f'<button type="button" aria-pressed="false" data-filter="{name}" data-value="{esc(v)}">'
                f'{t(labels.get(v, v))}</button>' for v in values]
    return "".join(buttons)


def copy_assets(out: Path) -> None:
    """The playground's stylesheets, fonts and mark, beside the pages."""
    css = out / "css"
    css.mkdir(parents=True, exist_ok=True)
    for name in SHARED_CSS:
        # The playground serves its fonts from the site root; a project site
        # lives under a sub-path, so they are found beside the stylesheet.
        text = (STATIC / "css" / name).read_text(encoding="utf-8")
        (css / name).write_text(text.replace('url("/fonts/', 'url("../fonts/'), encoding="utf-8")
    shutil.copytree(STATIC / "fonts", out / "fonts", dirs_exist_ok=True)
    shutil.copyfile(STATIC / "favicon.svg", out / "favicon.svg")


CSS = """
/* Platform series: three hues outside the playground's signal colours. */
:root { --linux: #4b4fd1; --macos: #0f7b84; --windows: #b04a92; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --linux: #a9abff; --macos: #6fd3db; --windows: #f19bd6; } }
:root[data-theme="dark"] { --linux: #a9abff; --macos: #6fd3db; --windows: #f19bd6; }

/* Content written in both languages shows the reader's one. */
:root:not([lang="zh-TW"]) .lang-zh, :root[lang="zh-TW"] .lang-en { display: none; }

/* Top bar: the playground's always-dark navigation, laid flat. */
.topbar { position: relative; z-index: 30; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; min-height: 52px;
  padding: 8px max(24px, env(safe-area-inset-right, 0px)) 8px max(24px, env(safe-area-inset-left, 0px));
  color: var(--on-nav); background: var(--nav); border-bottom: 1px solid var(--nav-line); }
.topbar :focus-visible { outline-color: var(--mint); }
.brand { display: flex; align-items: center; gap: 9px; color: var(--on-nav); font-size: 16px; font-weight: 600;
  letter-spacing: -0.01em; text-decoration: none; }
.brand img { width: 24px; height: 24px; }
.site-nav { display: flex; flex-wrap: wrap; gap: 2px; }
.site-nav a { position: relative; display: inline-flex; align-items: center; height: 32px; padding: 0 10px; color: var(--on-nav-2);
  border-radius: var(--radius-control); font-size: var(--text-body); text-decoration: none; }
.site-nav a:hover { color: var(--on-nav); background: rgba(255, 255, 255, 0.06); }
.site-nav a[aria-current="page"] { color: var(--on-nav); background: var(--nav-3); }
.site-nav a[aria-current="page"]::after { content: ""; position: absolute; left: 10px; right: 10px; bottom: 3px; height: 2px;
  border-radius: 2px; background: var(--accent); }
.topbar-end { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-left: auto; }
.topbar-end > a { color: var(--on-nav-2); font-size: var(--text-small); text-decoration: none; }
.topbar-end > a:hover { color: var(--on-nav); }
.nav-switch { display: flex; gap: 2px; padding: 2px; background: rgba(255, 255, 255, 0.06); border-radius: var(--radius-panel); }
.nav-switch button { min-width: 0; height: 24px; padding: 0 8px; color: var(--on-nav-2); background: transparent; border: 0;
  border-radius: var(--radius-control); font-size: 11.5px; font-weight: 500; }
.nav-switch button:hover { color: var(--on-nav); }
.nav-switch button[aria-pressed="true"] { color: var(--on-nav); background: var(--nav-3); }

/* Page frame and header, as the playground's screens. */
.page { max-width: 1680px; margin: 0 auto; padding: 0 max(24px, env(safe-area-inset-right, 0px)) 48px max(24px, env(safe-area-inset-left, 0px)); }
.page-header { position: sticky; z-index: 25; top: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto;
  align-items: center; gap: 6px 16px; margin: 0 calc(-1 * max(24px, env(safe-area-inset-left, 0px))) 16px;
  padding: 14px max(24px, env(safe-area-inset-left, 0px));
  background: color-mix(in srgb, var(--page) 94%, transparent); border-bottom: 1px solid var(--line);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px); }
.page-desc { max-width: 96ch; margin: 3px 0 0; color: var(--ink-2); font-size: var(--text-ui); line-height: 1.45; }
.page-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 8px; }
a.button { text-decoration: none; }

/* Toolbar and model cards: the playground's Models screen. */
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-bottom: 14px; }
.search-field { position: relative; min-width: 220px; flex: 1; display: flex; align-items: center; }
.search-field > span { position: absolute; left: 10px; color: var(--ink-3); font-size: 16px; line-height: 1; pointer-events: none; }
.search-field input { width: 100%; height: var(--control-h); padding: 0 10px 0 30px; color: var(--ink); background: var(--canvas);
  border: 1px solid var(--line-strong); border-radius: var(--radius-control); outline: 0; font-size: var(--text-ui); }
.search-field input:focus { border-color: var(--ink); }
.search-field input:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
.search-field input::placeholder { color: var(--ink-3); }
.filter-group { display: flex; align-items: center; gap: 6px; }
.filter-group > .caps { margin-right: 2px; }
.toolbar-count { margin-left: auto; color: var(--ink-3); font-family: var(--font-mono); font-size: 11px; white-space: nowrap; }
.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap: 12px; }
.model-card { position: relative; display: flex; flex-direction: column; padding: 16px; overflow: hidden;
  background: var(--canvas); border: 1px solid var(--line); border-radius: var(--radius-panel); transition: border-color 0.15s ease; }
.model-card:hover { border-color: var(--line-strong); }
.model-card.is-default { border-color: color-mix(in srgb, var(--good) 45%, var(--line)); box-shadow: inset 0 2px 0 var(--good); }
.model-card-top { display: flex; justify-content: space-between; align-items: start; gap: 12px; }
.model-symbol { width: 36px; height: 36px; display: grid; place-items: center; color: var(--periwinkle); background: var(--nav);
  border-radius: var(--radius-panel); font-family: var(--font-mono); font-size: 18px; line-height: 1; }
.model-symbol.sv { color: var(--speaker-ink); background: var(--speaker); }
.card-heading { margin-top: 12px; }
.model-card h3 { margin: 0; font-size: var(--text-lead); font-weight: 600; }
.model-card-subtitle { margin: 2px 0 0; color: var(--ink-3); font-family: var(--font-mono); font-size: 11px; overflow-wrap: anywhere; }
.model-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 10px; }
.badge { display: inline-flex; align-items: center; height: 22px; padding: 0 8px; color: var(--ink-2); background: var(--sunken);
  border-radius: var(--radius-pill); font-size: 11.5px; font-weight: 500; line-height: 1; white-space: nowrap; }
.badge-task.sv { color: var(--speaker-ink); background: var(--speaker); }
.badge-default { color: var(--good); background: var(--good-soft); }
.badge-candidate, .badge-alternative { color: var(--ink); background: transparent; box-shadow: inset 0 0 0 1px var(--line-strong); }
.badge-quiet { color: var(--ink-3); background: transparent; box-shadow: inset 0 0 0 1px var(--line); }
.model-description { display: -webkit-box; margin: 0; overflow: hidden; color: var(--ink-2); font-size: var(--text-ui); line-height: 1.5;
  -webkit-box-orient: vertical; -webkit-line-clamp: 4; line-clamp: 4; }
.model-card.is-expanded .model-description { display: block; -webkit-line-clamp: unset; line-clamp: none; }
.card-more { align-self: flex-start; padding: 4px 0 0; color: var(--ink-2); background: transparent; border: 0; font-size: var(--text-small);
  text-decoration: underline; text-underline-offset: 3px; }
.card-more:hover { color: var(--ink); }
.model-footer { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; padding-top: 12px;
  margin-top: auto; border-top: 1px solid var(--line); }
.model-meta { display: grid; gap: 2px; min-width: 0; color: var(--ink-3); font-size: 11.5px; }
.model-meta strong { overflow: hidden; color: var(--ink); font-family: var(--font-mono); font-size: 11.5px; font-weight: 500;
  text-overflow: ellipsis; white-space: nowrap; }
.card-links { display: flex; flex-wrap: wrap; gap: 6px; }
.card-link { flex: none; display: inline-flex; align-items: center; height: var(--control-h-small); padding: 0 9px; color: var(--ink);
  background: var(--canvas); border: 1px solid var(--line-strong); border-radius: var(--radius-control); font-size: var(--text-small);
  font-weight: 500; text-decoration: none; }
.card-link:hover { background: var(--sunken); }
.facts { display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 12px 0 0; font-size: var(--text-small); }
.facts div { display: flex; gap: 6px; align-items: baseline; }
.facts dt { color: var(--ink-3); } .facts dd { margin: 0; color: var(--ink); }

/* Text rendered from the repository's Markdown. */
.prose { min-width: 0; max-width: 92ch; color: var(--ink); font-size: var(--text-body); line-height: 1.6; }
.prose h1 { margin: 0 0 12px; font-size: 22px; }
.prose h2 { margin: 28px 0 10px; padding-top: 4px; font-size: 17px; }
.prose h3 { margin: 20px 0 8px; font-size: 15px; }
.prose h2, .prose h3, .prose h4 { scroll-margin-top: 96px; }
.prose p, .prose ul, .prose ol { margin: 0 0 12px; color: var(--ink-2); }
.prose li + li { margin-top: 4px; }
.prose a { color: var(--ink); text-decoration-color: var(--line-strong); text-underline-offset: 3px; }
.prose a:hover { text-decoration-color: currentColor; }
.prose strong { color: var(--ink); }
.prose code { padding: 1px 4px; background: var(--sunken); border-radius: 3px; }
.prose pre { margin: 0 0 14px; padding: 12px 14px; overflow-x: auto; color: var(--on-nav); background: var(--nav);
  border-radius: var(--radius-panel); font-size: 12.5px; line-height: 1.5; }
.prose pre code { padding: 0; color: inherit; background: none; }
.prose blockquote { margin: 0 0 12px; padding: 2px 14px; border-left: 3px solid var(--line-strong); }
.prose .table-wrap { margin: 0 0 16px; overflow-x: auto; background: var(--canvas); border: 1px solid var(--line);
  border-radius: var(--radius-panel); }
.prose .table th { white-space: normal; }
.prose .table td { color: var(--ink); }
.prose hr { margin: 24px 0; border: 0; border-top: 1px solid var(--line); }

.site-footer { margin-top: 32px; padding-top: 16px; color: var(--ink-3); border-top: 1px solid var(--line); font-size: var(--text-small); }
.site-footer a { color: var(--ink-2); }

@media (max-width: 760px) {
  .topbar-end > a { display: none; }
  .page-header { grid-template-columns: minmax(0, 1fr); }
  .page-actions { justify-content: flex-start; }
  .toolbar-count { margin-left: 0; }
}
"""

SCRIPT = """
(() => {
  const ZH = __ZH__;
  const root = document.documentElement;
  const tr = (text) => (root.lang === "zh-TW" && ZH[text]) || text;
  const site = { tr, onLanguage: [] };
  window.PureSoundSite = site;

  function translate() {
    for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = tr(el.dataset.i18n);
    for (const el of document.querySelectorAll("[data-i18n-attr]")) {
      for (const attr of el.dataset.i18nAttr.split(",")) {
        // A dataset key may not hold "-" before a lowercase letter.
        const key = "src_" + attr.replace(/[^a-z]/gi, "");
        el.dataset[key] ??= el.getAttribute(attr) || "";
        el.setAttribute(attr, tr(el.dataset[key]));
      }
    }
    for (const button of document.querySelectorAll("[data-lang]")) button.setAttribute("aria-pressed", String(button.dataset.lang === root.lang));
    for (const listener of site.onLanguage) listener();
  }
  site.translate = translate;

  const themeButtons = document.querySelectorAll("[data-theme-choice]");
  function setTheme(choice) {
    try { localStorage.setItem("puresound.theme", choice); } catch {}
    const dark = choice === "dark" || (choice === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "dark" : "light";
    for (const button of themeButtons) button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice));
  }
  let theme = "auto";
  try { theme = localStorage.getItem("puresound.theme") || "auto"; } catch {}
  for (const button of themeButtons) button.addEventListener("click", () => setTheme(button.dataset.themeChoice));
  setTheme(theme);
  for (const button of document.querySelectorAll("[data-lang]")) {
    button.addEventListener("click", () => {
      root.lang = button.dataset.lang;
      try { localStorage.setItem("puresound.lang", root.lang); } catch {}
      translate();
    });
  }
  document.addEventListener("DOMContentLoaded", translate);
})();
"""

#: Search, task and role filters over a grid of model cards, as the playground's
#: Models screen; ``?task=`` picks the task on arrival. Each card's description
#: shows four lines, with "More" where that hides text.
FILTER_SCRIPT = """
(() => {
  const site = window.PureSoundSite;
  const count = document.getElementById("count");
  const cards = [...document.querySelectorAll(".model-card[data-model]")];
  const state = { task: "all", role: "all", query: "" };

  function apply() {
    let shown = 0;
    for (const card of cards) {
      const visible = (state.task === "all" || card.dataset.task === state.task)
        && (state.role === "all" || card.dataset.roles.split(" ").includes(state.role))
        && (!state.query || card.dataset.text.includes(state.query));
      card.hidden = !visible;
      shown += visible ? 1 : 0;
      for (const row of document.querySelectorAll(`tr[data-model="${CSS.escape(card.dataset.model)}"]`)) row.hidden = !visible;
    }
    document.getElementById("no-match").hidden = shown > 0 || cards.length === 0;
    count.textContent = site.tr("{shown} of {total} models").replace("{shown}", shown).replace("{total}", cards.length);
    clamp();
  }
  // The full height is measured with the clamp lifted: under the standard
  // line-clamp, scrollHeight reports the clamped box, not the text. Fonts
  // arriving or a resize can change it.
  function clamp() {
    requestAnimationFrame(() => {
      for (const card of cards) {
        if (card.hidden || card.classList.contains("is-expanded")) continue;
        const text = card.querySelector("[data-clamp]");
        const clamped = text.getBoundingClientRect().height;
        text.style.display = "block";
        text.style.webkitLineClamp = "unset";
        const full = text.getBoundingClientRect().height;
        text.style.display = "";
        text.style.webkitLineClamp = "";
        card.querySelector("[data-more]").hidden = full <= clamped + 2;
      }
    });
  }
  function choose(name, value) {
    state[name] = value;
    for (const button of document.querySelectorAll(`[data-filter="${name}"]`)) {
      button.setAttribute("aria-pressed", String(button.dataset.value === value));
    }
  }
  document.fonts?.ready.then(clamp);
  addEventListener("resize", clamp);
  for (const button of document.querySelectorAll("[data-filter]")) {
    button.addEventListener("click", () => { choose(button.dataset.filter, button.dataset.value); apply(); });
  }
  const task = new URLSearchParams(location.search).get("task");
  if (task && document.querySelector(`[data-filter="task"][data-value="${CSS.escape(task)}"]`)) choose("task", task);
  document.getElementById("q").addEventListener("input", (event) => {
    state.query = event.target.value.trim().toLowerCase();
    apply();
  });
  for (const card of cards) {
    const more = card.querySelector("[data-more]");
    more.addEventListener("click", () => {
      const open = card.classList.toggle("is-expanded");
      more.firstElementChild.dataset.i18n = open ? "Less" : "More";
      more.firstElementChild.textContent = site.tr(more.firstElementChild.dataset.i18n);
    });
  }
  site.onLanguage.push(apply);
})();
"""

#: Theme and language before first paint, as the playground does.
PRE_PAINT = """
(() => {
  let theme = "auto", lang = null;
  try { theme = localStorage.getItem("puresound.theme") || "auto"; lang = localStorage.getItem("puresound.lang"); } catch {}
  const dark = theme === "dark" || (theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const zh = lang ? lang === "zh-TW" : String(navigator.language || "").toLowerCase().startsWith("zh");
  document.documentElement.lang = zh ? "zh-TW" : "en";
})();
"""


def page(*, title: str, active: str, body: str, prefix: str = "", repository: str | None = None,
         extra_css: str = "", extra_js: str = "", zh: dict[str, str] | None = None,
         head: str = "", footer: str = "") -> str:
    """A complete page: the frame around ``body``.

    ``prefix`` leads from the page to the site root ("" at the root, "../" one
    folder down); ``active`` is the file of the navigation entry to mark.
    """
    words = {**ZH_TW, **(zh or {})}
    nav = "".join(
        f'<a href="{prefix}{href}"{" aria-current=\"page\"" if href == active else ""}>{t(label)}</a>'
        for href, label in NAV
    )
    github = f'<a href="{esc(repository)}">GitHub</a>' if repository else ""
    script = SCRIPT.replace("__ZH__", json.dumps(words, ensure_ascii=False).replace("</", "<\\/"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#010120">
<link rel="icon" type="image/svg+xml" href="{prefix}favicon.svg">
<title>{esc(title)} · PureSound</title>
<link rel="stylesheet" href="{prefix}css/tokens.css">
<link rel="stylesheet" href="{prefix}css/base.css">
<link rel="stylesheet" href="{prefix}css/components.css">
<script>{PRE_PAINT}</script>
<style>{CSS}{extra_css}</style>
{head}
</head>
<body>
<header class="topbar">
<a class="brand" href="{prefix}index.html"><img src="{prefix}favicon.svg" alt="">PureSound</a>
<nav class="site-nav" aria-label="Site">{nav}</nav>
<div class="topbar-end">{github}
<div class="nav-switch" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">{t("Auto")}</button><button type="button" data-theme-choice="light">{t("Light")}</button><button type="button" data-theme-choice="dark">{t("Dark")}</button></div>
<div class="nav-switch" role="group" aria-label="Language"><button type="button" data-lang="en">EN</button><button type="button" data-lang="zh-TW">中文</button></div>
</div>
</header>
<main class="page">
{body}
{f'<footer class="site-footer">{footer}</footer>' if footer else ''}
</main>
<script>{script}</script>
{f'<script>{extra_js}</script>' if extra_js else ''}
</body>
</html>
"""
