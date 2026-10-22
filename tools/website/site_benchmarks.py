"""The Benchmarks page: `puresound models benchmark` records folded into a
history and drawn.

Records with one revision -- one CI run on several runners -- become one entry
in the history; entries stay in the order they ran, the newest last, and the
oldest drop off past ``keep``. The history also keeps the newest records' model
cards: the catalog's account of each model, carried in every record.

The page has one card per model with its catalog description, artifacts, a
chart of its single-thread RTF on each platform across runs and its latest
numbers, then a table of the latest run. Filters narrow it by task, role,
platform, variant and text; without JavaScript everything is shown.
"""

from __future__ import annotations

from site_chrome import FILTER_SCRIPT, TASKS, esc, page, segmented, t
from site_markdown import Links

SCHEMA_VERSION = 1
#: The single-thread RTF the project holds its default artifacts under.
RTF_BUDGET = 0.5
#: An int8 graph this far from its float graph is broken, not merely coarser:
#: puresound.streaming.quantization.INT8_MAX_RELATIVE_ERROR.
INT8_MAX_RELATIVE_ERROR = 0.1
PLATFORMS = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
LABELS = {"linux": "Linux", "macos": "macOS", "windows": "Windows"}
RESULT_KEYS = ("model_id", "variant", "rtf", "cold_start_seconds", "native_ssm", "intra_op_threads",
               "latency_ms", "int8_relative_error")
MACHINE_KEYS = ("system", "machine", "cpu", "logical_cpus", "onnxruntime", "ci_image")

INTRO = ("Single-thread real-time factor of every streaming model in the zoo, measured on GitHub-hosted "
         "runners after each merge to main and once a week. Lower is faster; 0.5 is the budget for the "
         "default artifacts. Shared runners move these numbers by tens of percent, so read the trend, not "
         "a single point.")

ZH_TW = {
    "Inference benchmarks": "推論效能量測",
    INTRO: "Model zoo 每個串流模型的單執行緒即時率（RTF），在每次合併進 main 後與每週一次於 GitHub "
           "代管機器上量測。數字越小越快；預設版本的預算是 0.5。共用機器的數字會浮動數十個百分點，"
           "請看趨勢，不要只看單一點。",
    "Platforms": "平台", "Variants": "版本",
    "Latest RTF, one thread": "最新 RTF（單執行緒）",
    "Look-ahead": "前瞻延遲", "Default artifact": "預設模型檔", "float weights": "浮點權重",
    "Latest run": "最新一次量測", "logical CPUs": "個邏輯 CPU",
    "budget": "預算", "other variants dashed": "其他版本為虛線",
    "No runs yet.": "還沒有量測紀錄。",
    "model": "模型", "variant": "版本",
    "Measured by": "量測工具",
    "Values above the budget are red.": "超過預算的數值以紅色標示。",
    "int8 vs float": "int8 與浮點差距",
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

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(model)}: '
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
                     f'{esc(runs[0]["revision"][:7])}</text>')
        if len(runs) > 1:
            parts.append(f'<text class="axis" x="{x(len(runs) - 1):.1f}" y="{height - 7}" text-anchor="end">'
                         f'{esc(runs[-1]["revision"][:7])}</text>')
    for platform in platforms:
        for variant in variants:
            points = _series(runs, model, variant, platform)
            if not points:
                continue
            tags = f"p-{platform} v-{esc(variant)}"
            dash = " alt" if variant != "default" else ""
            path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in points)
            parts.append(f'<polyline class="line c-{platform}{dash} {tags}" points="{path}"/>')
            for i, v in points:
                label = f"{LABELS.get(platform, platform)} {variant} @ {runs[i]['revision'][:7]}: RTF {v:.3f}"
                parts.append(f'<circle class="dot c-{platform} {tags}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.75">'
                             f'<title>{esc(label)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def _rtf_cell(item: dict | None, variant: str, platform: str) -> str:
    if item is None:
        return f'<td class="num p-{platform} quiet">—</td>'
    rtf = float(item["rtf"])
    over = " is-over" if variant == "default" and rtf >= RTF_BUDGET else ""
    return f'<td class="num p-{platform}{over}">{rtf:.3f}</td>'


def _card(model: str, info: dict, variants: list[str], runs: list[dict], platforms: list[str],
          links: Links) -> str:
    latest = runs[-1] if runs else None
    task = info.get("task", "")
    roles = info.get("roles") or []
    variants = _variant_order(variants)
    badges = [f'<span class="badge badge-task">{t(TASKS.get(task, task))}</span>']
    badges += [f'<span class="badge badge-{esc(role)}">{t(role)}</span>' for role in roles]
    lifecycle = info.get("lifecycle") or "released"

    artifacts = info.get("artifacts") or {}
    artifact_rows = []
    for variant in _variant_order(set(artifacts) | set(variants)):
        artifact = artifacts.get(variant, {})
        size = artifact.get("size_bytes")
        note = artifact.get("description")
        note_html = esc(note) if note else (t("float weights") if variant == "default" else "")
        artifact_rows.append(
            f'<li class="v-{esc(variant)}"><span class="variant-name">{esc(variant)}</span>'
            f'<span class="mono quiet">{f"{size / 1e6:.1f} MB" if size else ""}</span>'
            f'<span class="artifact-note">{note_html}</span></li>'
        )

    facts = []
    latency = next((item.get("latency_ms") for p in platforms
                    if (item := _latest_item(latest, model, "default", p)) and item.get("latency_ms") is not None),
                   None)
    if latency is not None:
        facts.append(f'<div><dt>{t("Look-ahead")}</dt><dd class="mono">{latency:.0f} ms</dd></div>')
    for key, value in (info.get("recommended_inference") or {}).items():
        facts.append(f'<div><dt>{esc(key)}</dt><dd class="mono">{esc(value)}</dd></div>')

    head = "".join(f'<th class="num p-{p}">{LABELS[p]}</th>' for p in platforms)
    rows = "".join(
        f'<tr class="v-{esc(v)}"><td>{esc(v)}</td>'
        + "".join(_rtf_cell(_latest_item(latest, model, v, p), v, p) for p in platforms) + "</tr>"
        for v in variants
    )
    # How far each int8 graph landed from the float one on each runner's CPU.
    for v in variants:
        items = {p: _latest_item(latest, model, v, p) for p in platforms}
        if not any(item and item.get("int8_relative_error") is not None for item in items.values()):
            continue
        cells = []
        for p, item in items.items():
            error = item.get("int8_relative_error") if item else None
            if error is None:
                cells.append(f'<td class="num p-{p} quiet">—</td>')
            else:
                over = " is-over" if error >= INT8_MAX_RELATIVE_ERROR else ""
                cells.append(f'<td class="num p-{p} quiet{over}">{float(error):.3f}</td>')
        rows += (f'<tr class="v-{esc(v)} int8-row"><td class="quiet">{esc(v)} · {t("int8 vs float")}</td>'
                 + "".join(cells) + "</tr>")
    default_file = (artifacts.get("default") or {}).get("file") or "—"
    default_size = (artifacts.get("default") or {}).get("size_bytes")
    card = info.get("card")
    href = links.to(card, "README.md") if card else None
    link = f'<a class="card-link" href="{esc(href)}">{t("Model card")} →</a>' if href and href != card else ""
    text = " ".join([model, info.get("display_name") or "", task, " ".join(roles),
                     info.get("description") or ""]).lower()
    default_class = " is-default" if "default" in roles else ""

    return f"""<article class="model-card bench-card{default_class}" id="{esc(model)}" data-model="{esc(model)}"
 data-task="{esc(task)}" data-roles="{esc(' '.join(roles))}" data-text="{esc(text)}">
<div class="model-card-top"><div class="model-symbol" aria-hidden="true">∿</div>
<span class="badge badge-quiet">{t(lifecycle)}</span></div>
<div class="card-heading"><h3>{esc(info.get("display_name") or model)}</h3>
<p class="model-card-subtitle">{esc(model)}</p></div>
<div class="model-badges">{''.join(badges)}</div>
<p class="model-description" data-clamp>{esc(info.get("description") or "")}</p>
<button class="card-more" type="button" data-more hidden>{t("More")}</button>
{f'<dl class="facts">{"".join(facts)}</dl>' if facts else ''}
<ul class="artifacts">{''.join(artifact_rows)}</ul>
<figure class="chart">{_chart(runs, model, variants, platforms)}</figure>
<table class="table mini"><caption class="caps">{t("Latest RTF, one thread")}</caption>
<thead><tr><th>{t("variant")}</th>{head}</tr></thead><tbody>{rows}</tbody></table>
<div class="model-footer"><div class="model-meta"><span>{t("Default artifact")}</span>
<strong>{esc(default_file)}{f" · {default_size / 1e6:.1f} MB" if default_size else ""}</strong></div>{link}</div>
</article>"""


CSS = """
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
/* Show/hide toggles, all on by default: on is a plain chip, off is dashed and
   faint. The accent stays for focus, as everywhere in the playground. */
.chip-row .chip { cursor: pointer; }
.chip-row .chip.is-active { color: var(--ink); border-color: var(--line-strong); box-shadow: none; }
.chip-row .chip:not(.is-active) { color: var(--ink-3); background: transparent; border-style: dashed; }
.chip-row .chip:not(.is-active) .swatch { opacity: 0.3; }
.swatch { display: inline-block; width: 14px; height: 3px; border-radius: 2px; }
.swatch.c-linux { background: var(--linux); } .swatch.c-macos { background: var(--macos); }
.swatch.c-windows { background: var(--windows); } .swatch.solid { background: var(--ink-2); }
.swatch.dash { background: repeating-linear-gradient(90deg, var(--ink-2) 0 4px, transparent 4px 7px); }
.model-grid.bench { grid-template-columns: repeat(auto-fill, minmax(min(100%, 400px), 1fr)); }
.artifacts { display: grid; gap: 4px; margin: 10px 0 0; padding: 10px 0 0; list-style: none; border-top: 1px solid var(--line);
  font-size: var(--text-small); }
.artifacts li { display: grid; grid-template-columns: 4.5em 4.8em minmax(0, 1fr); gap: 8px; align-items: baseline; }
.variant-name { color: var(--ink); font-weight: 500; }
.artifact-note { color: var(--ink-2); line-height: 1.4; }
.chart { margin: 12px 0 0; }
.chart svg { width: 100%; height: auto; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .axis { fill: var(--ink-3); font-family: var(--font-mono); font-size: 10.5px; }
.chart .budget { stroke: var(--bad); stroke-width: 1.25; stroke-dasharray: 2 4; }
.chart .budget-label { fill: var(--bad); font-family: var(--font-mono); font-size: 10.5px; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.chart .line.alt { stroke-dasharray: 6 4; }
.chart .line.c-linux { stroke: var(--linux); } .chart .dot.c-linux { fill: var(--linux); }
.chart .line.c-macos { stroke: var(--macos); } .chart .dot.c-macos { fill: var(--macos); }
.chart .line.c-windows { stroke: var(--windows); } .chart .dot.c-windows { fill: var(--windows); }
.table.mini { margin-top: 8px; }
.table caption { padding-bottom: 4px; text-align: left; }
.table td.is-over { color: var(--bad); font-weight: 600; }
/* Platform names keep their own case under the caps table header. */
.table th.p-linux, .table th.p-macos, .table th.p-windows { text-transform: none; letter-spacing: 0; }
.model-card > .table + .model-footer { margin-top: 12px; }
.summary { display: grid; gap: 14px; margin-top: 16px; }
.summary .panel-title { margin-bottom: 0; }
.machines { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 8px; }
.machines .stat strong { font-size: 13px; white-space: normal; }
.scroll { overflow-x: auto; }
.footnote { margin: 16px 0 0; color: var(--ink-3); font-size: var(--text-small); }
main.hide-p-linux .p-linux, main.hide-p-macos .p-macos, main.hide-p-windows .p-windows { display: none; }
@media (max-width: 760px) {
  .artifacts li { grid-template-columns: 4.5em minmax(0, 1fr); }
  .artifacts .artifact-note { grid-column: 1 / -1; }
}
"""

TOGGLES = """
for (const chip of document.querySelectorAll("[data-toggle]")) {
  chip.addEventListener("click", () => {
    const on = chip.getAttribute("aria-pressed") !== "true";
    chip.setAttribute("aria-pressed", String(on));
    chip.classList.toggle("is-active", on);
    document.querySelector("main").classList.toggle("hide-" + chip.dataset.toggle, !on);
  });
}
"""


def render(history: dict, links: Links) -> str:
    runs = history["runs"]
    cards_info = history.get("models", {})
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

    cards = "".join(_card(m, cards_info.get(m, {}), models[m], runs, platforms, links)
                    for m in sorted(models, key=order))

    platform_chips = "".join(
        f'<button type="button" class="chip is-active" aria-pressed="true" data-toggle="p-{p}">'
        f'<i class="swatch c-{p}" aria-hidden="true"></i>{LABELS[p]}</button>' for p in platforms
    )
    variant_chips = "".join(
        f'<button type="button" class="chip is-active" aria-pressed="true" data-toggle="v-{esc(v)}">'
        f'<i class="swatch{" dash" if v != "default" else " solid"}" aria-hidden="true"></i>{esc(v)}</button>'
        for v in variants_all
    )

    summary = ""
    if latest:
        head = "".join(f'<th class="num p-{p}">{LABELS[p]}</th>' for p in platforms)
        rows = []
        for model in sorted(models, key=order):
            for variant in _variant_order(models[model]):
                cells = "".join(_rtf_cell(_latest_item(latest, model, variant, p), variant, p) for p in platforms)
                rows.append(f'<tr data-model="{esc(model)}" class="v-{esc(variant)}">'
                            f'<td class="mono">{esc(model)}</td><td>{esc(variant)}</td>{cells}</tr>')
        machines = "".join(
            f'<div class="stat p-{p}"><span>{LABELS.get(p, p)}</span><strong>{esc(s["machine"].get("cpu"))}</strong>'
            f'<small>{esc(s["machine"].get("logical_cpus"))} {t("logical CPUs")} · onnxruntime '
            f'{esc(s["machine"].get("onnxruntime"))}'
            + (f' · {esc(s["machine"].get("ci_image"))}' if s["machine"].get("ci_image") else "")
            + "</small></div>"
            for p, s in sorted(latest["platforms"].items(),
                               key=lambda kv: list(LABELS).index(kv[0]) if kv[0] in LABELS else len(LABELS))
        )
        summary = f"""<section class="panel panel-pad summary">
<div class="panel-title"><h2>{t("Latest run")}</h2><code class="quiet">{esc(latest["revision"][:12])} · {esc(latest["created_at"][:10])}</code></div>
<div class="machines">{machines}</div>
<div class="scroll"><table class="table"><thead><tr><th>{t("model")}</th><th>{t("variant")}</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="panel-help">{t("Values above the budget are red.")}</p>
</section>"""

    latest_state = (f'<span class="state is-ok"><span class="mono">{esc(latest["revision"][:7])}</span>'
                    f' · {esc(latest["created_at"][:10])}</span>') if latest else ""
    variant_rules = "".join(f"main.hide-v-{esc(v)} .v-{esc(v)} {{ display: none; }}" for v in variants_all)

    body = f"""<header class="page-header">
<div class="page-heading"><h1>{t("Inference benchmarks")}</h1><p class="page-desc">{t(INTRO)}</p></div>
<div class="page-actions">{latest_state}</div>
</header>
<div class="toolbar" id="filters">
<label class="search-field"><span aria-hidden="true">⌕</span><input id="q" type="search" autocomplete="off"
 placeholder="Search models" aria-label="Search models" data-i18n-attr="placeholder,aria-label"></label>
<div class="filter-group"><span class="caps">{t("Task")}</span><div class="segmented" role="group" aria-label="Task">{segmented("task", tasks, TASKS)}</div></div>
<div class="filter-group"><span class="caps">{t("Role")}</span><div class="segmented" role="group" aria-label="Role">{segmented("role", roles, {})}</div></div>
<div class="filter-group"><span class="caps">{t("Platforms")}</span><div class="chip-row" role="group" aria-label="Platforms">{platform_chips}</div></div>
<div class="filter-group"><span class="caps">{t("Variants")}</span><div class="chip-row" role="group" aria-label="Variants">{variant_chips}</div></div>
<span class="toolbar-count" id="count">{len(models)} of {len(models)} models</span>
</div>
<div class="model-grid bench" id="grid">{cards or f'<div class="empty-state">{t("No runs yet.")}</div>'}
<div class="empty-state" id="no-match" hidden>{t("No models match these filters.")}</div></div>
{summary}
<p class="footnote">{t("Measured by")} <code>puresound models benchmark</code> · <span class="swatch dash" aria-hidden="true"></span> {t("other variants dashed")}</p>"""
    return page(title="Benchmarks", active="benchmarks.html", body=body, repository=links.repository,
                extra_css=CSS + variant_rules, extra_js=FILTER_SCRIPT + TOGGLES, zh=ZH_TW)
