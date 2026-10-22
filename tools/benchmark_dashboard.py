"""Fold `puresound models benchmark` records into a history and render it.

    python tools/benchmark_dashboard.py --history site/history.json \\
        --add bench/*.json --out site

Records with one revision -- one CI run on several runners -- become one entry
in the history; entries stay in the order they ran, the newest last, and the
oldest drop off past ``--keep``. The history also keeps the newest records'
model cards: the catalog's account of each model, carried in every record.

``--out`` receives the history and the page: one card per model, laid out like
the web playground's Models screen, with its catalog description, artifacts, a
chart of its single-thread RTF on each platform across runs and its latest
numbers, then a table of the latest run. Filters narrow it by task, role,
platform, variant and text; without JavaScript everything is shown. The page
uses the playground's own design tokens, components and fonts, copied from
``puresound/web/static`` beside it, and loads nothing from elsewhere. Standard
library only, so it runs without the project installed.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

SCHEMA_VERSION = 1
#: The single-thread RTF the project holds its default artifacts under.
RTF_BUDGET = 0.5
STATIC = Path(__file__).resolve().parents[1] / "puresound/web/static"
#: The playground stylesheets every screen shares; the page uses them unchanged.
SHARED_CSS = ("tokens.css", "base.css", "components.css")
PLATFORMS = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
LABELS = {"linux": "Linux", "macos": "macOS", "windows": "Windows"}
TASKS = {"voice_isolation": "Voice isolation", "noise_suppression": "Noise suppression"}
RESULT_KEYS = ("model_id", "variant", "rtf", "cold_start_seconds", "native_ssm", "intra_op_threads",
               "latency_ms")
MACHINE_KEYS = ("system", "machine", "cpu", "logical_cpus", "onnxruntime", "ci_image")

#: The page's own words in Traditional Chinese, keyed by their English. Catalog
#: text (names, descriptions) stays as the catalog writes it.
ZH_TW = {
    "Benchmarks": "效能量測",
    "Inference benchmarks": "推論效能量測",
    "Single-thread real-time factor of every streaming model in the zoo, measured on GitHub-hosted "
    "runners after each merge to main and once a week. Lower is faster; 0.5 is the budget for the "
    "default artifacts. Shared runners move these numbers by tens of percent, so read the trend, not "
    "a single point.":
        "Model zoo 每個串流模型的單執行緒即時率（RTF），在每次合併進 main 後與每週一次於 GitHub "
        "代管機器上量測。數字越小越快；預設版本的預算是 0.5。共用機器的數字會浮動數十個百分點，"
        "請看趨勢，不要只看單一點。",
    "Search models": "搜尋模型",
    "Task": "任務", "Role": "角色", "Platforms": "平台", "Variants": "版本",
    "All": "全部", "Voice isolation": "人聲分離", "Noise suppression": "雜訊抑制",
    "default": "預設", "candidate": "候選", "alternative": "替代", "released": "已發布",
    "More": "展開", "Less": "收合",
    "Latest RTF, one thread": "最新 RTF（單執行緒）",
    "Model card": "模型卡",
    "Look-ahead": "前瞻延遲", "Default artifact": "預設模型檔", "float weights": "浮點權重",
    "Latest run": "最新一次量測", "Machines": "機器", "logical CPUs": "個邏輯 CPU",
    "budget": "預算", "other variants dashed": "其他版本為虛線",
    "No models match these filters.": "沒有符合篩選條件的模型。",
    "No runs yet.": "還沒有量測紀錄。",
    "Theme": "主題", "Auto": "自動", "Light": "淺色", "Dark": "深色", "Language": "語言",
    "model": "模型", "variant": "版本",
    "{shown} of {total} models": "{shown} / {total} 個模型",
    "Measured by": "量測工具",
    "Values above the budget are red.": "超過預算的數值以紅色標示。",
}


def _summary(record: dict) -> dict:
    """What a run keeps of one record: its numbers and its machine."""
    return {
        "platform": PLATFORMS.get(record["machine"]["system"], record["machine"]["system"].lower()),
        "machine": {key: record["machine"].get(key) for key in MACHINE_KEYS},
        "results": [{key: item.get(key) for key in RESULT_KEYS} for item in record["results"]],
    }


def merge(history: dict, records: list[dict], keep: int) -> dict:
    runs = list(history.get("runs", []))
    by_revision: dict[str, dict] = {}
    for record in records:
        revision = record.get("revision") or "unknown"
        run = by_revision.setdefault(revision, {
            "revision": revision, "created_at": record["created_at"], "platforms": {},
        })
        run["created_at"] = min(run["created_at"], record["created_at"])
        summary = _summary(record)
        run["platforms"][summary.pop("platform")] = summary
    for revision, run in by_revision.items():
        # A rerun of one revision replaces its earlier entry instead of adding one.
        runs = [existing for existing in runs if existing["revision"] != revision]
        runs.append(run)
    runs.sort(key=lambda run: run["created_at"])
    merged = {
        "schema_version": SCHEMA_VERSION,
        "models": history.get("models", {}),
        "repository": history.get("repository"),
        "runs": runs[-keep:],
    }
    # The cards describe the catalog as of the newest record that carries them.
    for record in sorted(records, key=lambda record: record["created_at"]):
        if record.get("models"):
            merged["models"] = {**merged["models"], **record["models"]}
        if record.get("repository"):
            merged["repository"] = record["repository"]
    return merged


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _t(text: str) -> str:
    """English text the page translates on the reader's request."""
    return f'<span data-i18n="{_esc(text)}">{_esc(text)}</span>'


def _variant_order(variants) -> list[str]:
    return sorted(variants, key=lambda variant: (variant != "default", variant))


def _series(runs: list[dict], model: str, variant: str, platform: str) -> list[tuple[int, float]]:
    points = []
    for index, run in enumerate(runs):
        for item in run["platforms"].get(platform, {}).get("results", []):
            if item["model_id"] == model and item["variant"] == variant:
                points.append((index, float(item["rtf"])))
    return points


def _latest_item(run: dict | None, model: str, variant: str, platform: str) -> dict | None:
    if run is None:
        return None
    for item in run["platforms"].get(platform, {}).get("results", []):
        if item["model_id"] == model and item["variant"] == variant:
            return item
    return None


def _chart(runs: list[dict], model: str, variants: list[str], platforms: list[str]) -> str:
    width, height, left, right, top, bottom = 520, 188, 40, 8, 12, 26
    values = [rtf for p in platforms for v in variants for _, rtf in _series(runs, model, v, p)]
    ceiling = max([RTF_BUDGET, *values]) * 1.12
    span = max(1, len(runs) - 1)

    def x(index: int) -> float:
        return left + (width - left - right) * (index / span if len(runs) > 1 else 0.5)

    def y(value: float) -> float:
        return top + (height - top - bottom) * (1 - value / ceiling)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(model)}: '
             f'single-thread RTF per run on each platform">']
    for tick in [ceiling * i / 4 for i in range(5)]:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 6}" y="{y(tick) + 3.5:.1f}" text-anchor="end">{tick:.2f}</text>')
    parts.append(f'<line class="budget" x1="{left}" x2="{width - right}" y1="{y(RTF_BUDGET):.1f}" '
                 f'y2="{y(RTF_BUDGET):.1f}"/>')
    parts.append(f'<text class="budget-label" x="{width - right}" y="{y(RTF_BUDGET) - 5:.1f}" '
                 f'text-anchor="end"><tspan data-i18n="budget">budget</tspan> {RTF_BUDGET}</text>')
    if runs:
        parts.append(f'<text class="axis" x="{x(0):.1f}" y="{height - 7}" text-anchor="start">'
                     f'{_esc(runs[0]["revision"][:7])}</text>')
        if len(runs) > 1:
            parts.append(f'<text class="axis" x="{x(len(runs) - 1):.1f}" y="{height - 7}" text-anchor="end">'
                         f'{_esc(runs[-1]["revision"][:7])}</text>')
    for platform in platforms:
        for variant in variants:
            points = _series(runs, model, variant, platform)
            if not points:
                continue
            tags = f"p-{platform} v-{_esc(variant)}"
            dash = " alt" if variant != "default" else ""
            path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in points)
            parts.append(f'<polyline class="line c-{platform}{dash} {tags}" points="{path}"/>')
            for i, v in points:
                label = f"{LABELS.get(platform, platform)} {variant} @ {runs[i]['revision'][:7]}: RTF {v:.3f}"
                parts.append(f'<circle class="dot c-{platform} {tags}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.75">'
                             f'<title>{_esc(label)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def _rtf_cell(item: dict | None, variant: str, platform: str) -> str:
    if item is None:
        return f'<td class="num p-{platform} quiet">—</td>'
    rtf = float(item["rtf"])
    over = " is-over" if variant == "default" and rtf >= RTF_BUDGET else ""
    return f'<td class="num p-{platform}{over}">{rtf:.3f}</td>'


def _card(model: str, info: dict, variants: list[str], runs: list[dict], platforms: list[str],
          repository: str | None) -> str:
    latest = runs[-1] if runs else None
    task = info.get("task", "")
    roles = info.get("roles") or []
    variants = _variant_order(variants)
    badges = [f'<span class="badge badge-task">{_t(TASKS.get(task, task))}</span>']
    badges += [f'<span class="badge badge-{_esc(role)}">{_t(role)}</span>' for role in roles]
    lifecycle = info.get("lifecycle") or "released"

    artifacts = info.get("artifacts") or {}
    artifact_rows = []
    for variant in _variant_order(set(artifacts) | set(variants)):
        artifact = artifacts.get(variant, {})
        size = artifact.get("size_bytes")
        note = artifact.get("description")
        note_html = _esc(note) if note else (_t("float weights") if variant == "default" else "")
        artifact_rows.append(
            f'<li class="v-{_esc(variant)}"><span class="variant-name">{_esc(variant)}</span>'
            f'<span class="mono quiet">{f"{size / 1e6:.1f} MB" if size else ""}</span>'
            f'<span class="artifact-note">{note_html}</span></li>'
        )

    facts = []
    latency = next((item.get("latency_ms") for p in platforms
                    if (item := _latest_item(latest, model, "default", p)) and item.get("latency_ms") is not None),
                   None)
    if latency is not None:
        facts.append(f'<div><dt>{_t("Look-ahead")}</dt><dd class="mono">{latency:.0f} ms</dd></div>')
    for key, value in (info.get("recommended_inference") or {}).items():
        facts.append(f'<div><dt>{_esc(key)}</dt><dd class="mono">{_esc(value)}</dd></div>')

    head = "".join(f'<th class="num p-{p}">{LABELS[p]}</th>' for p in platforms)
    rows = "".join(
        f'<tr class="v-{_esc(v)}"><td>{_esc(v)}</td>'
        + "".join(_rtf_cell(_latest_item(latest, model, v, p), v, p) for p in platforms) + "</tr>"
        for v in variants
    )
    default_file = (artifacts.get("default") or {}).get("file") or "—"
    default_size = (artifacts.get("default") or {}).get("size_bytes")
    card = info.get("card")
    if card and repository and latest:
        link = (f'<a class="card-link" href="{_esc(repository)}/blob/{_esc(latest["revision"])}/{_esc(card)}">'
                f'{_t("Model card")} →</a>')
    else:
        link = ""
    text = " ".join([model, info.get("display_name") or "", task, " ".join(roles),
                     info.get("description") or ""]).lower()
    default_class = " is-default" if "default" in roles else ""

    return f"""<article class="model-card bench-card{default_class}" data-model="{_esc(model)}"
 data-task="{_esc(task)}" data-roles="{_esc(' '.join(roles))}" data-text="{_esc(text)}">
<div class="model-card-top"><div class="model-symbol" aria-hidden="true">∿</div>
<span class="badge badge-quiet">{_t(lifecycle)}</span></div>
<div class="card-heading"><h3>{_esc(info.get("display_name") or model)}</h3>
<p class="model-card-subtitle">{_esc(model)}</p></div>
<div class="model-badges">{''.join(badges)}</div>
<p class="model-description" data-clamp>{_esc(info.get("description") or "")}</p>
<button class="card-more" type="button" data-more hidden>{_t("More")}</button>
{f'<dl class="facts">{"".join(facts)}</dl>' if facts else ''}
<ul class="artifacts">{''.join(artifact_rows)}</ul>
<figure class="chart">{_chart(runs, model, variants, platforms)}</figure>
<table class="table mini"><caption class="caps">{_t("Latest RTF, one thread")}</caption>
<thead><tr><th>{_t("variant")}</th>{head}</tr></thead><tbody>{rows}</tbody></table>
<div class="model-footer"><div class="model-meta"><span>{_t("Default artifact")}</span>
<strong>{_esc(default_file)}{f" · {default_size / 1e6:.1f} MB" if default_size else ""}</strong></div>{link}</div>
</article>"""


def render(history: dict) -> str:
    runs = history["runs"]
    cards_info = history.get("models", {})
    repository = history.get("repository")
    platforms = [p for p in LABELS if any(p in run["platforms"] for run in runs)]
    models: dict[str, list[str]] = {}
    for run in runs:
        for summary in run["platforms"].values():
            for item in summary["results"]:
                variants = models.setdefault(item["model_id"], [])
                if item["variant"] not in variants:
                    variants.append(item["variant"])
    latest = runs[-1] if runs else None
    tasks = sorted({cards_info.get(m, {}).get("task", "") for m in models} - {""})
    roles = sorted({r for m in models for r in cards_info.get(m, {}).get("roles", [])},
                   key=lambda role: (role != "default", role))
    variants_all = _variant_order({v for vs in models.values() for v in vs})

    def order(model: str) -> tuple:
        info = cards_info.get(model, {})
        return (info.get("task", ""), "default" not in info.get("roles", []), model)

    cards = "".join(_card(m, cards_info.get(m, {}), models[m], runs, platforms, repository)
                    for m in sorted(models, key=order))

    def segmented(name: str, values: list[str], labels: dict[str, str]) -> str:
        buttons = [f'<button type="button" aria-pressed="true" data-filter="{name}" data-value="all">'
                   f'{_t("All")}</button>']
        buttons += [f'<button type="button" aria-pressed="false" data-filter="{name}" data-value="{_esc(v)}">'
                    f'{_t(labels.get(v, v))}</button>' for v in values]
        return "".join(buttons)

    platform_chips = "".join(
        f'<button type="button" class="chip is-active" aria-pressed="true" data-toggle="p-{p}">'
        f'<i class="swatch c-{p}" aria-hidden="true"></i>{LABELS[p]}</button>' for p in platforms
    )
    variant_chips = "".join(
        f'<button type="button" class="chip is-active" aria-pressed="true" data-toggle="v-{_esc(v)}">'
        f'<i class="swatch{" dash" if v != "default" else " solid"}" aria-hidden="true"></i>{_esc(v)}</button>'
        for v in variants_all
    )

    summary = ""
    if latest:
        head = "".join(f'<th class="num p-{p}">{LABELS[p]}</th>' for p in platforms)
        rows = []
        for model in sorted(models, key=order):
            for variant in _variant_order(models[model]):
                cells = "".join(_rtf_cell(_latest_item(latest, model, variant, p), variant, p) for p in platforms)
                rows.append(f'<tr data-model="{_esc(model)}" class="v-{_esc(variant)}">'
                            f'<td class="mono">{_esc(model)}</td><td>{_esc(variant)}</td>{cells}</tr>')
        machines = "".join(
            f'<div class="stat p-{p}"><span>{LABELS.get(p, p)}</span><strong>{_esc(s["machine"].get("cpu"))}</strong>'
            f'<small>{_esc(s["machine"].get("logical_cpus"))} {_t("logical CPUs")} · onnxruntime '
            f'{_esc(s["machine"].get("onnxruntime"))}'
            + (f' · {_esc(s["machine"].get("ci_image"))}' if s["machine"].get("ci_image") else "")
            + "</small></div>"
            for p, s in sorted(latest["platforms"].items(),
                               key=lambda kv: list(LABELS).index(kv[0]) if kv[0] in LABELS else len(LABELS))
        )
        summary = f"""<section class="panel panel-pad summary">
<div class="panel-title"><h2>{_t("Latest run")}</h2><code class="quiet">{_esc(latest["revision"][:12])} · {_esc(latest["created_at"][:10])}</code></div>
<div class="machines">{machines}</div>
<div class="scroll"><table class="table"><thead><tr><th>{_t("model")}</th><th>{_t("variant")}</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="panel-help">{_t("Values above the budget are red.")}</p>
</section>"""

    latest_state = (f'<span class="state is-ok"><span class="mono">{_esc(latest["revision"][:7])}</span>'
                    f' · {_esc(latest["created_at"][:10])}</span>') if latest else ""
    repo_links = (f'<a href="{_esc(repository)}">GitHub</a>'
                  f'<a href="{_esc(repository)}/tree/main/model_zoo">Model zoo</a>') if repository else ""
    logo = (STATIC / "favicon.svg").is_file()
    variant_rules = "".join(f"main.hide-v-{_esc(v)} .v-{_esc(v)} {{ display: none; }}" for v in variants_all)
    intro = ("Single-thread real-time factor of every streaming model in the zoo, measured on GitHub-hosted "
             "runners after each merge to main and once a week. Lower is faster; 0.5 is the budget for the "
             "default artifacts. Shared runners move these numbers by tens of percent, so read the trend, not "
             "a single point.")
    zh = json.dumps(ZH_TW, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#010120">
{'<link rel="icon" type="image/svg+xml" href="favicon.svg">' if logo else ''}
<title>PureSound · Inference benchmarks</title>
<link rel="stylesheet" href="css/tokens.css">
<link rel="stylesheet" href="css/base.css">
<link rel="stylesheet" href="css/components.css">
<script>
  // Theme and language before first paint, as the playground does.
  (() => {{
    let theme = "auto", lang = null;
    try {{ theme = localStorage.getItem("puresound.theme") || "auto"; lang = localStorage.getItem("puresound.lang"); }} catch {{}}
    const dark = theme === "dark" || (theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const zh = lang ? lang === "zh-TW" : String(navigator.language || "").toLowerCase().startsWith("zh");
    document.documentElement.lang = zh ? "zh-TW" : "en";
  }})();
</script>
<style>
/* Platform series: three hues outside the playground's signal colours, so a
   platform never reads as input, output or interferer. */
:root {{ --linux: #4b4fd1; --macos: #0f7b84; --windows: #b04a92; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --linux: #a9abff; --macos: #6fd3db; --windows: #f19bd6; }} }}
:root[data-theme="dark"] {{ --linux: #a9abff; --macos: #6fd3db; --windows: #f19bd6; }}

/* Top bar: the playground's always-dark navigation, laid flat. */
.topbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; min-height: 52px;
  padding: 8px max(24px, env(safe-area-inset-right, 0px)) 8px max(24px, env(safe-area-inset-left, 0px));
  color: var(--on-nav); background: var(--nav); border-bottom: 1px solid var(--nav-line); }}
.topbar :focus-visible {{ outline-color: var(--mint); }}
.brand {{ display: flex; align-items: center; gap: 9px; color: var(--on-nav); font-size: 16px; font-weight: 600;
  letter-spacing: -0.01em; text-decoration: none; }}
.brand img {{ width: 24px; height: 24px; }}
.topbar-section {{ color: var(--on-nav-2); font-size: var(--text-body); }}
.topbar-section::before {{ content: "/"; margin-right: 12px; color: var(--on-nav-3); }}
.topbar-links {{ display: flex; gap: 14px; margin-left: auto; }}
.topbar-links a {{ color: var(--on-nav-2); font-size: var(--text-small); text-decoration: none; }}
.topbar-links a:hover {{ color: var(--on-nav); }}
.nav-switch {{ display: flex; gap: 2px; padding: 2px; background: rgba(255, 255, 255, 0.06); border-radius: var(--radius-panel); }}
.nav-switch button {{ min-width: 0; height: 24px; padding: 0 8px; color: var(--on-nav-2); background: transparent; border: 0;
  border-radius: var(--radius-control); font-size: 11.5px; font-weight: 500; }}
.nav-switch button:hover {{ color: var(--on-nav); }}
.nav-switch button[aria-pressed="true"] {{ color: var(--on-nav); background: var(--nav-3); }}

/* Page frame and header, as the playground's screens. */
.page {{ max-width: 1680px; margin: 0 auto; padding: 0 max(24px, env(safe-area-inset-right, 0px)) 48px max(24px, env(safe-area-inset-left, 0px)); }}
.page-header {{ position: sticky; z-index: 25; top: 0; display: grid; grid-template-columns: minmax(0, 1fr) auto;
  align-items: center; gap: 6px 16px; margin: 0 calc(-1 * max(24px, env(safe-area-inset-left, 0px))) 16px;
  padding: 14px max(24px, env(safe-area-inset-left, 0px));
  background: color-mix(in srgb, var(--page) 94%, transparent); border-bottom: 1px solid var(--line);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px); }}
.page-desc {{ max-width: 96ch; margin: 3px 0 0; color: var(--ink-2); font-size: var(--text-ui); line-height: 1.45; }}

/* Toolbar and model cards: the playground's Models screen. */
.toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin-bottom: 14px; }}
.search-field {{ position: relative; min-width: 220px; flex: 1; display: flex; align-items: center; }}
.search-field > span {{ position: absolute; left: 10px; color: var(--ink-3); font-size: 16px; line-height: 1; pointer-events: none; }}
.search-field input {{ width: 100%; height: var(--control-h); padding: 0 10px 0 30px; color: var(--ink); background: var(--canvas);
  border: 1px solid var(--line-strong); border-radius: var(--radius-control); outline: 0; font-size: var(--text-ui); }}
.search-field input:focus {{ border-color: var(--ink); }}
.search-field input:focus-visible {{ outline: 2px solid var(--focus); outline-offset: 1px; }}
.search-field input::placeholder {{ color: var(--ink-3); }}
.filter-group {{ display: flex; align-items: center; gap: 6px; }}
.filter-group > .caps {{ margin-right: 2px; }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
/* Show/hide toggles, all on by default: on is a plain chip, off is dashed and
   faint. The accent stays for focus, as everywhere in the playground. */
.chip-row .chip {{ cursor: pointer; }}
.chip-row .chip.is-active {{ color: var(--ink); border-color: var(--line-strong); box-shadow: none; }}
.chip-row .chip:not(.is-active) {{ color: var(--ink-3); background: transparent; border-style: dashed; }}
.chip-row .chip:not(.is-active) .swatch {{ opacity: 0.3; }}
.toolbar-count {{ margin-left: auto; color: var(--ink-3); font-family: var(--font-mono); font-size: 11px; white-space: nowrap; }}
.swatch {{ display: inline-block; width: 14px; height: 3px; border-radius: 2px; }}
.swatch.c-linux {{ background: var(--linux); }} .swatch.c-macos {{ background: var(--macos); }}
.swatch.c-windows {{ background: var(--windows); }} .swatch.solid {{ background: var(--ink-2); }}
.swatch.dash {{ background: repeating-linear-gradient(90deg, var(--ink-2) 0 4px, transparent 4px 7px); }}

.model-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 400px), 1fr)); gap: 12px; }}
.model-card {{ position: relative; display: flex; flex-direction: column; gap: 0; padding: 16px; overflow: hidden;
  background: var(--canvas); border: 1px solid var(--line); border-radius: var(--radius-panel); transition: border-color 0.15s ease; }}
.model-card:hover {{ border-color: var(--line-strong); }}
.model-card.is-default {{ border-color: color-mix(in srgb, var(--good) 45%, var(--line)); box-shadow: inset 0 2px 0 var(--good); }}
.model-card-top {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
.model-symbol {{ width: 36px; height: 36px; display: grid; place-items: center; color: var(--periwinkle); background: var(--nav);
  border-radius: var(--radius-panel); font-family: var(--font-mono); font-size: 18px; line-height: 1; }}
.card-heading {{ margin-top: 12px; }}
.model-card h3 {{ margin: 0; font-size: var(--text-lead); font-weight: 600; }}
.model-card-subtitle {{ margin: 2px 0 0; color: var(--ink-3); font-family: var(--font-mono); font-size: 11px; overflow-wrap: anywhere; }}
.model-badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 10px; }}
.badge {{ display: inline-flex; align-items: center; height: 22px; padding: 0 8px; color: var(--ink-2); background: var(--sunken);
  border-radius: var(--radius-pill); font-size: 11.5px; font-weight: 500; line-height: 1; white-space: nowrap; }}
.badge-default {{ color: var(--good); background: var(--good-soft); }}
.badge-candidate {{ color: var(--ink); background: transparent; box-shadow: inset 0 0 0 1px var(--line-strong); }}
.badge-quiet {{ color: var(--ink-3); background: transparent; box-shadow: inset 0 0 0 1px var(--line); }}
.model-description {{ display: -webkit-box; margin: 0; overflow: hidden; color: var(--ink-2); font-size: var(--text-ui); line-height: 1.5;
  -webkit-box-orient: vertical; -webkit-line-clamp: 4; line-clamp: 4; }}
.model-card.is-expanded .model-description {{ display: block; -webkit-line-clamp: unset; line-clamp: none; }}
.card-more {{ align-self: flex-start; padding: 4px 0 0; color: var(--ink-2); background: transparent; border: 0; font-size: var(--text-small);
  text-decoration: underline; text-underline-offset: 3px; }}
.card-more:hover {{ color: var(--ink); }}
.facts {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 12px 0 0; font-size: var(--text-small); }}
.facts div {{ display: flex; gap: 6px; align-items: baseline; }}
.facts dt {{ color: var(--ink-3); }} .facts dd {{ margin: 0; color: var(--ink); }}
.artifacts {{ display: grid; gap: 4px; margin: 10px 0 0; padding: 10px 0 0; list-style: none; border-top: 1px solid var(--line);
  font-size: var(--text-small); }}
.artifacts li {{ display: grid; grid-template-columns: 4.5em 4.8em minmax(0, 1fr); gap: 8px; align-items: baseline; }}
.variant-name {{ color: var(--ink); font-weight: 500; }}
.artifact-note {{ color: var(--ink-2); line-height: 1.4; }}
.chart {{ margin: 12px 0 0; }}
.chart svg {{ width: 100%; height: auto; }}
.chart .grid {{ stroke: var(--line); stroke-width: 1; }}
.chart .axis {{ fill: var(--ink-3); font-family: var(--font-mono); font-size: 10.5px; }}
.chart .budget {{ stroke: var(--bad); stroke-width: 1.25; stroke-dasharray: 2 4; }}
.chart .budget-label {{ fill: var(--bad); font-family: var(--font-mono); font-size: 10.5px; }}
.chart .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.chart .line.alt {{ stroke-dasharray: 6 4; }}
.chart .line.c-linux {{ stroke: var(--linux); }} .chart .dot.c-linux {{ fill: var(--linux); }}
.chart .line.c-macos {{ stroke: var(--macos); }} .chart .dot.c-macos {{ fill: var(--macos); }}
.chart .line.c-windows {{ stroke: var(--windows); }} .chart .dot.c-windows {{ fill: var(--windows); }}
.table.mini {{ margin-top: 8px; }}
.table caption {{ padding-bottom: 4px; text-align: left; }}
.table td.is-over {{ color: var(--bad); font-weight: 600; }}
/* Platform names keep their own case under the caps table header. */
.table th.p-linux, .table th.p-macos, .table th.p-windows {{ text-transform: none; letter-spacing: 0; }}
.model-footer {{ display: flex; align-items: end; justify-content: space-between; gap: 12px; padding-top: 12px; margin-top: auto;
  border-top: 1px solid var(--line); }}
.model-card > .table + .model-footer {{ margin-top: 12px; }}
.model-meta {{ display: grid; gap: 2px; min-width: 0; color: var(--ink-3); font-size: 11.5px; }}
.model-meta strong {{ overflow: hidden; color: var(--ink); font-family: var(--font-mono); font-size: 11.5px; font-weight: 500;
  text-overflow: ellipsis; white-space: nowrap; }}
.card-link {{ flex: none; display: inline-flex; align-items: center; height: var(--control-h-small); padding: 0 9px; color: var(--ink);
  background: var(--canvas); border: 1px solid var(--line-strong); border-radius: var(--radius-control); font-size: var(--text-small);
  font-weight: 500; text-decoration: none; }}
.card-link:hover {{ background: var(--sunken); }}

.summary {{ display: grid; gap: 14px; margin-top: 16px; }}
.summary .panel-title {{ margin-bottom: 0; }}
.machines {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 8px; }}
.machines .stat strong {{ font-size: 13px; white-space: normal; }}
.scroll {{ overflow-x: auto; }}
.footnote {{ margin: 16px 0 0; color: var(--ink-3); font-size: var(--text-small); }}

main.hide-p-linux .p-linux, main.hide-p-macos .p-macos, main.hide-p-windows .p-windows {{ display: none; }}
{variant_rules}
@media (max-width: 760px) {{
  .topbar-links {{ display: none; }}
  .topbar .nav-switch:first-of-type {{ margin-left: auto; }}
  .page-header {{ grid-template-columns: minmax(0, 1fr); }}
  .toolbar-count {{ margin-left: 0; }}
  .artifacts li {{ grid-template-columns: 4.5em minmax(0, 1fr); }}
  .artifacts .artifact-note {{ grid-column: 1 / -1; }}
}}
</style>
</head>
<body>
<header class="topbar">
<a class="brand" href="{_esc(repository or '#')}">{'<img src="favicon.svg" alt="">' if logo else ''}PureSound</a>
<span class="topbar-section">{_t("Benchmarks")}</span>
<nav class="topbar-links" aria-label="Project">{repo_links}</nav>
<div class="nav-switch" role="group" aria-label="Theme">
<button type="button" data-theme-choice="auto">{_t("Auto")}</button><button type="button" data-theme-choice="light">{_t("Light")}</button><button type="button" data-theme-choice="dark">{_t("Dark")}</button></div>
<div class="nav-switch" role="group" aria-label="Language">
<button type="button" data-lang="en">EN</button><button type="button" data-lang="zh-TW">中文</button></div>
</header>
<main class="page">
<header class="page-header">
<div class="page-heading"><h1>{_t("Inference benchmarks")}</h1><p class="page-desc">{_t(intro)}</p></div>
<div class="page-actions">{latest_state}</div>
</header>
<div class="toolbar" id="filters">
<label class="search-field"><span aria-hidden="true">⌕</span><input id="q" type="search" autocomplete="off"
 placeholder="Search models" aria-label="Search models" data-i18n-attr="placeholder,aria-label"></label>
<div class="filter-group"><span class="caps">{_t("Task")}</span><div class="segmented" role="group" aria-label="Task">{segmented("task", tasks, TASKS)}</div></div>
<div class="filter-group"><span class="caps">{_t("Role")}</span><div class="segmented" role="group" aria-label="Role">{segmented("role", roles, {})}</div></div>
<div class="filter-group"><span class="caps">{_t("Platforms")}</span><div class="chip-row" role="group" aria-label="Platforms">{platform_chips}</div></div>
<div class="filter-group"><span class="caps">{_t("Variants")}</span><div class="chip-row" role="group" aria-label="Variants">{variant_chips}</div></div>
<span class="toolbar-count" id="count" data-total="{len(models)}">{len(models)} of {len(models)} models</span>
</div>
<div class="model-grid" id="grid">{cards or f'<div class="empty-state">{_t("No runs yet.")}</div>'}
<div class="empty-state" id="no-match" hidden>{_t("No models match these filters.")}</div></div>
{summary}
<p class="footnote">{_t("Measured by")} <code>puresound models benchmark</code> · <span class="swatch dash" aria-hidden="true"></span> {_t("other variants dashed")}</p>
</main>
<script>
(() => {{
  const ZH = {zh};
  const root = document.documentElement;
  const main = document.querySelector("main");
  const count = document.getElementById("count");
  const cards = [...document.querySelectorAll(".model-card[data-model]")];
  const state = {{ task: "all", role: "all", query: "" }};
  const zh = () => root.lang === "zh-TW";
  const tr = (text) => (zh() && ZH[text]) || text;

  function translate() {{
    for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = tr(el.dataset.i18n);
    for (const el of document.querySelectorAll("[data-i18n-attr]")) {{
      for (const attr of el.dataset.i18nAttr.split(",")) {{
        // A dataset key may not hold "-" before a lowercase letter.
        const key = "src_" + attr.replace(/[^a-z]/gi, "");
        el.dataset[key] ??= el.getAttribute(attr) || "";
        el.setAttribute(attr, tr(el.dataset[key]));
      }}
    }}
    for (const button of document.querySelectorAll("[data-lang]")) button.setAttribute("aria-pressed", String(button.dataset.lang === root.lang));
    apply();
  }}
  function apply() {{
    let shown = 0;
    for (const card of cards) {{
      const visible = (state.task === "all" || card.dataset.task === state.task)
        && (state.role === "all" || card.dataset.roles.split(" ").includes(state.role))
        && (!state.query || card.dataset.text.includes(state.query));
      card.hidden = !visible;
      shown += visible ? 1 : 0;
      for (const row of document.querySelectorAll(`tr[data-model="${{CSS.escape(card.dataset.model)}}"]`)) row.hidden = !visible;
    }}
    document.getElementById("no-match").hidden = shown > 0 || cards.length === 0;
    count.textContent = tr("{{shown}} of {{total}} models").replace("{{shown}}", shown).replace("{{total}}", cards.length);
    clamp();
  }}
  // "More" shows only where four lines hide text. The full height is measured
  // with the clamp lifted: under the standard line-clamp, scrollHeight reports
  // the clamped box, not the text. Fonts arriving or a resize can change it.
  function clamp() {{
    requestAnimationFrame(() => {{
      for (const card of cards) {{
        if (card.hidden || card.classList.contains("is-expanded")) continue;
        const text = card.querySelector("[data-clamp]");
        const clamped = text.getBoundingClientRect().height;
        text.style.display = "block";
        text.style.webkitLineClamp = "unset";
        const full = text.getBoundingClientRect().height;
        text.style.display = "";
        text.style.webkitLineClamp = "";
        card.querySelector("[data-more]").hidden = full <= clamped + 2;
      }}
    }});
  }}
  document.fonts?.ready.then(clamp);
  addEventListener("resize", clamp);
  for (const button of document.querySelectorAll("[data-filter]")) {{
    button.addEventListener("click", () => {{
      state[button.dataset.filter] = button.dataset.value;
      for (const other of document.querySelectorAll(`[data-filter="${{button.dataset.filter}}"]`)) {{
        other.setAttribute("aria-pressed", String(other === button));
      }}
      apply();
    }});
  }}
  for (const chip of document.querySelectorAll("[data-toggle]")) {{
    chip.addEventListener("click", () => {{
      const on = chip.getAttribute("aria-pressed") !== "true";
      chip.setAttribute("aria-pressed", String(on));
      chip.classList.toggle("is-active", on);
      main.classList.toggle("hide-" + chip.dataset.toggle, !on);
    }});
  }}
  document.getElementById("q").addEventListener("input", (event) => {{
    state.query = event.target.value.trim().toLowerCase();
    apply();
  }});
  for (const card of cards) {{
    const more = card.querySelector("[data-more]");
    more.addEventListener("click", () => {{
      const open = card.classList.toggle("is-expanded");
      more.firstElementChild.dataset.i18n = open ? "Less" : "More";
      more.firstElementChild.textContent = tr(more.firstElementChild.dataset.i18n);
    }});
  }}
  const themeButtons = document.querySelectorAll("[data-theme-choice]");
  function setTheme(choice) {{
    try {{ localStorage.setItem("puresound.theme", choice); }} catch {{}}
    const dark = choice === "dark" || (choice === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "dark" : "light";
    for (const button of themeButtons) button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice));
  }}
  let theme = "auto";
  try {{ theme = localStorage.getItem("puresound.theme") || "auto"; }} catch {{}}
  for (const button of themeButtons) button.addEventListener("click", () => setTheme(button.dataset.themeChoice));
  setTheme(theme);
  for (const button of document.querySelectorAll("[data-lang]")) {{
    button.addEventListener("click", () => {{
      root.lang = button.dataset.lang;
      try {{ localStorage.setItem("puresound.lang", root.lang); }} catch {{}}
      translate();
    }});
  }}
  translate();
}})();
</script>
</body>
</html>
"""


def write_site(out: Path, history: dict) -> None:
    """The page, its history, and the playground assets it is drawn with."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render(history), encoding="utf-8")
    css = out / "css"
    css.mkdir(exist_ok=True)
    for name in SHARED_CSS:
        # The playground serves its fonts from the site root; this page may live
        # under a project path, so they are found beside the stylesheet instead.
        text = (STATIC / "css" / name).read_text(encoding="utf-8")
        (css / name).write_text(text.replace('url("/fonts/', 'url("../fonts/'), encoding="utf-8")
    shutil.copytree(STATIC / "fonts", out / "fonts", dirs_exist_ok=True)
    if (STATIC / "favicon.svg").is_file():
        shutil.copyfile(STATIC / "favicon.svg", out / "favicon.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", type=Path, required=True, help="history JSON to extend (may not exist yet)")
    parser.add_argument("--add", type=Path, nargs="*", default=[], help="benchmark records to fold in")
    parser.add_argument("--out", type=Path, required=True, help="directory for the page, its assets and history.json")
    parser.add_argument("--keep", type=int, default=200, help="runs to keep (default: 200)")
    args = parser.parse_args()
    history = (json.loads(args.history.read_text(encoding="utf-8")) if args.history.is_file()
               else {"schema_version": SCHEMA_VERSION, "runs": []})
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.add]
    history = merge(history, records, args.keep)
    write_site(args.out, history)
    print(f"{len(history['runs'])} runs -> {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
