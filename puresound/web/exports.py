"""Take a finished run out of the web workspace: its audio as a ZIP, or the
whole comparison as one self-contained HTML page (tables and players, audio
embedded) that can be sent to someone who has no PureSound server."""

from __future__ import annotations

import base64
import html
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_RUN_URL = re.compile(r"^/api/runs/([0-9a-f]{32})/([^/]+)$")
#: Above this much audio a report keeps its tables and drops the players, so
#: a large comparison still makes a page a mail client will carry.
MAX_EMBEDDED_AUDIO_BYTES = 40 * 1024 * 1024

Fetch = Callable[[str, str], Any]  # (token, output name) -> StoredOutput | None


def _slug(text: Any, default: str = "file") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("._")
    return slug[:80] or default


def _stem(name: Any) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", str(name or ""))


def _label(model: Mapping[str, Any]) -> str:
    label = model.get("label")
    return str(label if label and label != model.get("model_id") else model.get("display_name") or model.get("model_id") or "model")


def job_files(job: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(archive path, run URL)`` for every audio output a finished job holds."""

    result = job.get("result") or {}
    files: list[tuple[str, str]] = []
    if job.get("kind") in {"world_render", "world_sweep"}:
        cells = result.get("cells") or [{"index": 0, "result": result}]
        for cell in cells:
            folder = (
                f"cell_{cell['index']:02d}/" if job.get("kind") == "world_sweep" else ""
            )
            for key, url in (cell.get("result") or {}).get("output_urls", {}).items():
                files.append(
                    (
                        folder
                        + _slug(key)
                        + (".zip" if key == "scene-package" else ".wav"),
                        url,
                    )
                )
        return files
    if job.get("kind") == "measurement":
        clips = result.get("clips") or [result]
        for index, clip in enumerate(clips):
            folder = f"{index + 1:02d}_{_slug(_stem((clip.get('input_files') or {}).get('audio')), 'clip')}/" if len(clips) > 1 else ""
            if (clip.get("baseline") or {}).get("output_url"):
                files.append((f"{folder}00_unprocessed.wav", clip["baseline"]["output_url"]))
            if clip.get("reference_url"):
                files.append((f"{folder}reference.wav", clip["reference_url"]))
            for model in clip.get("models") or []:
                if model.get("output_url") and not model.get("error"):
                    files.append((f"{folder}{int(model.get('candidate_id', 0)) + 1:02d}_{_slug(_label(model))}.wav", model["output_url"]))
        return files
    urls = result.get("output_urls") or {}
    names = {"input": "input.wav", "aligned": "output.wav", "audio": "output_stream.wav", "removed": "removed.wav"}
    stages = (result.get("pipeline") or {}).get("stages") or []
    for key, url in urls.items():
        if key in names:
            files.append((names[key], url))
        elif key.startswith("stage-"):
            position = int(key.split("-", 1)[1]) if key.split("-", 1)[1].isdigit() else 0
            stage = stages[position - 1] if 0 < position <= len(stages) else {}
            files.append((f"stage_{position}_{_slug(stage.get('display_name') or stage.get('model_id'), 'model')}.wav", url))
    return files


def _load(fetch: Fetch, url: str) -> bytes | None:
    match = _RUN_URL.match(str(url or ""))
    if not match:
        return None
    stored = fetch(match.group(1), match.group(2))
    return stored.data if stored is not None else None


def build_zip(job: Mapping[str, Any], fetch: Fetch) -> tuple[bytes, list[str]]:
    """The job's audio plus ``report.json``; returns the bytes and what is missing."""

    missing = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, url in job_files(job):
            data = _load(fetch, url)
            if data is None:
                missing.append(name)
                continue
            archive.writestr(name, data)
        archive.writestr("report.json", json.dumps(job, indent=2, ensure_ascii=False, default=str))
        if missing:
            archive.writestr("MISSING.txt", "These outputs had already expired from the run store:\n" + "\n".join(missing) + "\n")
    return buffer.getvalue(), missing


# Report ------------------------------------------------------------------------

def _zone(name: str | None):
    try:
        return ZoneInfo(name) if name else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _when(epoch: Any, zone) -> str:
    try:
        return datetime.fromtimestamp(float(epoch), zone).strftime("%Y-%m-%d %H:%M:%S %Z")
    except (TypeError, ValueError, OSError):
        return "—"


def _number(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _delta(value: Any, base: Any, digits: int) -> str:
    try:
        change = float(value) - float(base)
    except (TypeError, ValueError):
        return ""
    tone = "up" if change >= 0 else "down"
    return f'<small class="{tone}">{change:+.{digits}f}</small>'


_SCORES = (
    ("DNSMOS OVR", lambda row: ((row.get("reference_free") or {}).get("dnsmos") or {}).get("dnsmos_ovr"), 2, ""),
    ("SI-SDR", lambda row: (row.get("quality") or {}).get("si_sdr_db"), 2, " dB"),
    ("STOI", lambda row: (row.get("quality") or {}).get("stoi"), 3, ""),
    ("PESQ", lambda row: (row.get("quality") or {}).get("pesq_wb"), 2, ""),
)


def _audio_tag(fetch: Fetch, url: str | None, label: str, budget: list[int]) -> str:
    data = _load(fetch, url) if url else None
    if data is None:
        return f'<div class="track"><span>{html.escape(label)}</span><em>audio no longer stored</em></div>'
    if budget[0] + len(data) > MAX_EMBEDDED_AUDIO_BYTES:
        return f'<div class="track"><span>{html.escape(label)}</span><em>left out: the report is at its size limit</em></div>'
    budget[0] += len(data)
    encoded = base64.b64encode(data).decode("ascii")
    return f'<div class="track"><span>{html.escape(label)}</span><audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio></div>'


def _clip_section(clip: Mapping[str, Any], fetch: Fetch, budget: list[int], title: str) -> str:
    baseline = clip.get("baseline") or {}
    rows = []
    header = "".join(f"<th>{name}</th>" for name, *_ in _SCORES)
    rows.append(
        f'<tr class="baseline"><td><b>Unprocessed input</b></td><td>—</td><td>{_number((baseline.get("output") or {}).get("rms_dbfs"), 1, " dB")}</td>'
        + "".join(f"<td>{_number(read(baseline), digits, suffix)}</td>" for _, read, digits, suffix in _SCORES)
        + "</tr>"
    )
    for model in clip.get("models") or []:
        if model.get("error"):
            rows.append(f'<tr><td><b>{html.escape(_label(model))}</b><small class="down">{html.escape(str(model["error"]))}</small></td><td colspan="{2 + len(_SCORES)}">—</td></tr>')
            continue
        settings = " · ".join(filter(None, [
            ("after " + " → ".join(stage.get("display_name") or stage.get("model_id", "") for stage in model["stages"])) if model.get("stages") else "",
            f"blend {_number(model.get('dry_blend'), 2)}",
            f"guard {'on' if model.get('onset_guard') else 'off'}",
        ]))
        rows.append(
            f'<tr><td><b>{html.escape(_label(model))}</b><small>{html.escape(settings)}</small></td><td>{_number(model.get("rtf"), 3)}</td><td>{_number((model.get("output") or {}).get("rms_dbfs"), 1, " dB")}</td>'
            + "".join(f"<td>{_number(read(model), digits, suffix)}{_delta(read(model), read(baseline), digits)}</td>" for _, read, digits, suffix in _SCORES)
            + "</tr>"
        )
    tracks = [_audio_tag(fetch, baseline.get("output_url"), "Unprocessed input", budget)]
    tracks += [_audio_tag(fetch, model.get("output_url"), _label(model), budget) for model in clip.get("models") or [] if model.get("output_url") and not model.get("error")]
    if clip.get("reference_url"):
        tracks.append(_audio_tag(fetch, clip["reference_url"], "Clean reference", budget))
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<table><thead><tr><th>Row</th><th>RTF</th><th>Level</th>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<div class="tracks">{"".join(tracks)}</div></section>'
    )


def _aggregate_section(result: Mapping[str, Any]) -> str:
    rows = []
    for row in result.get("aggregate") or []:
        cells = []
        for key, digits in (("dnsmos_ovr", 2), ("si_sdr_db", 2), ("stoi", 3), ("pesq_wb", 2)):
            summary = (row.get("metrics") or {}).get(key) or {}
            if summary.get("mean") is None:
                cells.append("<td>—</td>")
                continue
            interval = "one clip" if summary.get("ci_low") is None else f"[{summary['ci_low']:+.{digits}f}, {summary['ci_high']:+.{digits}f}]"
            verdict = "" if summary.get("resolved") else (" · too few clips" if summary.get("n", 0) < 5 else " · not resolved")
            tone = "up" if summary.get("resolved") and summary["mean"] > 0 else "down" if summary.get("resolved") else "flat"
            cells.append(f'<td class="{tone}"><b>{summary["mean"]:+.{digits}f}</b><small>{interval} · {summary.get("wins")}/{summary.get("n")} up{verdict}</small></td>')
        rows.append(f'<tr><td><b>{html.escape(str(row.get("label") or row.get("model_id")))}</b></td><td>{row.get("clips_ok")}</td><td>{_number(row.get("rtf_median"), 3)}</td>{"".join(cells)}</tr>')
    return (
        "<section><h2>Across clips</h2><p>Change from the unprocessed input of the same clip, averaged, with a 95% bootstrap interval. "
        "“Not resolved”: the interval spans zero. Fewer than five clips never resolve.</p>"
        "<table><thead><tr><th>Candidate</th><th>Clips</th><th>RTF median</th><th>Δ DNSMOS OVR</th><th>Δ SI-SDR</th><th>Δ STOI</th><th>Δ PESQ</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _run_section(job: Mapping[str, Any], fetch: Fetch, budget: list[int]) -> str:
    result = job.get("result") or {}
    metadata = result.get("metadata") or {}
    urls = result.get("output_urls") or {}
    dnsmos = (((result.get("measurements") or {}).get("reference_free") or {}).get("dnsmos") or {}).get("dnsmos_ovr")
    facts = [
        ("Input", (result.get("input_files") or {}).get("audio") or "—"),
        ("Duration", _number(metadata.get("duration_seconds"), 2, " s")),
        ("RTF", _number((result.get("pipeline") or {}).get("rtf", result.get("rtf")), 3)),
        ("Look-ahead", _number(metadata.get("latency_ms"), 0, " ms")),
        ("Dry blend", _number(metadata.get("dry_blend"), 2)),
        ("Onset guard", "on" if metadata.get("onset_guard") else "off"),
        ("DNSMOS OVR", _number(dnsmos, 2)),
    ]
    if result.get("pipeline"):
        facts.insert(1, ("Chain", " → ".join(stage.get("display_name") or stage.get("model_id", "") for stage in result["pipeline"].get("stages", [])) + " → " + str(job.get("model_id"))))
    tracks = [_audio_tag(fetch, urls[key], label, budget) for key, label in (("input", "Input"), ("aligned", "Output"), ("removed", "Removed (input − output)")) if urls.get(key)]
    return (
        "<section><h2>Run</h2><dl>" + "".join(f"<dt>{html.escape(name)}</dt><dd>{html.escape(str(value))}</dd>" for name, value in facts) + "</dl>"
        f'<div class="tracks">{"".join(tracks)}</div></section>'
    )

def _world_section(
    result: Mapping[str, Any], fetch: Fetch, budget: list[int], title: str
) -> str:
    metadata = result.get("metadata") or {}
    scene = result.get("scene") or {}
    rows = []
    for policy, scores in (result.get("metrics") or {}).items():
        metric = scores.get("output") or {}
        rows.append(
            f"<tr><td>{html.escape(policy)}</td><td>{html.escape(scores.get('purpose', ''))}</td><td>{_number(metric.get('si_sdr_db'), 2, ' dB')}</td><td>{_number(metric.get('stoi'), 3)}</td><td>{_number(metric.get('pesq_wb'), 2)}</td></tr>"
        )
    comparison_notes = []
    for provider, comparison in (result.get("comparisons") or {}).items():
        label = f"{provider} · {comparison.get('display_name') or comparison.get('model_id') or ''}".rstrip(" ·")
        if comparison.get("status") != "succeeded":
            comparison_notes.append(f"{label}: {comparison.get('error') or 'comparison unavailable'}")
        else:
            comparison_notes.append(
                f"{label} · model {comparison.get('resolved_model_id', '—')} · SDK {comparison.get('sdk_version', '—')} · "
                f"strength {_number(comparison.get('enhancement_level'), 2)} · compensated delay {_number(comparison.get('audio_delay_ms'), 1, ' ms')} · "
                f"processing RTF {_number(comparison.get('rtf'), 3)} (excludes setup/download)"
            )
        for policy, scores in comparison.get("metrics", {}).items():
            metric = scores.get("output") or {}
            rows.append(
                f"<tr><td>{html.escape(label)} · {html.escape(policy)}</td><td>{html.escape(scores.get('purpose', ''))}</td><td>{_number(metric.get('si_sdr_db'), 2, ' dB')}</td><td>{_number(metric.get('stoi'), 3)}</td><td>{_number(metric.get('pesq_wb'), 2)}</td></tr>"
            )
    tracks = [
        _audio_tag(fetch, url, key, budget)
        for key, url in (result.get("output_urls") or {}).items()
        if key in {"input", "aligned", "removed", "reference-target", "reference-speech", "reference-near", "aicoustics", "aicoustics-removed"}
        or key.startswith("source-")
    ]
    facts = f"Seed {scene.get('seed', '—')} · renderer {metadata.get('renderer', '—')} · source {_number(metadata.get('source_duration_s'), 2, ' s')} · tail {_number(metadata.get('tail_s'), 2, ' s')}"
    notes = "".join(f"<p>{html.escape(note)}</p>" for note in comparison_notes)
    return f'<section><h2>{html.escape(title)}</h2><p>{html.escape(facts)}</p><p>{html.escape(metadata.get("late_field", ""))}</p>{notes}<table><thead><tr><th>Policy</th><th>Purpose</th><th>SI-SDR</th><th>STOI</th><th>PESQ</th></tr></thead><tbody>{"".join(rows)}</tbody></table><div class="tracks">{"".join(tracks)}</div></section>'


_STYLE = """
body{margin:0;padding:32px 20px 60px;font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:#111;background:#fff}
main{max-width:1100px;margin:0 auto}h1{margin:0 0 4px;font-size:30px;font-weight:600}h2{margin:32px 0 10px;font-size:19px}
.meta{color:#666;font-size:13px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #e5e5e5;text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}th{color:#666;font-size:11px;font-weight:500;text-transform:uppercase}td small{display:block;color:#666;font-size:11px}
.up{color:#1e8d5b}.down{color:#b42318}.flat{color:#666}tr.baseline td{background:#f6fbfb}
.tracks{display:grid;gap:8px;margin-top:14px}.track{display:grid;grid-template-columns:minmax(140px,260px) 1fr;align-items:center;gap:12px}
.track span{font-size:13px}.track em{color:#999;font-size:12px}audio{width:100%}dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;margin:0}
dt{color:#666}dd{margin:0}.note{margin-top:32px;color:#666;font-size:12px}
@media (prefers-color-scheme: dark){body{color:#eee;background:#111}th,td{border-color:#333}tr.baseline td{background:#1b2424}.meta,th,td small,dt,.note{color:#aaa}}
"""


def build_report(job: Mapping[str, Any], fetch: Fetch, *, tz: str | None = None) -> str:
    """One HTML page for a finished run, audio embedded as data URIs."""

    zone = _zone(tz)
    result = job.get("result") or {}
    budget = [0]
    host = result.get("host") or {}
    load = host.get("load_average_1m")
    kind = job.get("kind")
    if kind in {"world_render", "world_sweep"}:
        title = (
            "PureSound acoustic world"
            if kind == "world_render"
            else "PureSound model limit map"
        )
        subtitle = str(job.get("status") or "")
        cells = result.get("cells") or [
            {"index": 0, "status": job.get("status"), "result": result}
        ]
        body = ""
        for cell in cells:
            label = f"Cell {cell['index']} · {cell.get('x', '')} × {cell.get('y', '')} · {cell.get('status', '')}"
            body += (
                _world_section(cell["result"], fetch, budget, label)
                if cell.get("result")
                else f"<section><h2>{html.escape(label)}</h2><p>{html.escape(cell.get('error', ''))}</p></section>"
            )
    elif kind == "measurement":
        title = "PureSound comparison"
        clips = result.get("clips") or [result]
        body = _aggregate_section(result) if result.get("clips") else ""
        body += "".join(
            _clip_section(clip, fetch, budget, (clip.get("input_files") or {}).get("audio") or f"Clip {index + 1}")
            for index, clip in enumerate(clips)
        )
        request = result.get("request") or {}
        subtitle = f"{len(clips)} clip{'s' if len(clips) != 1 else ''} · provider {request.get('provider', 'auto')} · RTF {'median of ' + str(request.get('measurements', {}).get('timing_repeats')) + ' runs' if (request.get('measurements') or {}).get('timing_repeats', 1) > 1 else 'one run'}"
    else:
        title = f"PureSound {'live session' if kind == 'live' else 'run'} · {job.get('model_id')}"
        body = _run_section(job, fetch, budget)
        subtitle = str(result.get("provider") or "")
    meta = " · ".join(filter(None, [
        _when(job.get("finished_at") or job.get("created_at"), zone),
        subtitle,
        f"host load {load:.1f} / {host.get('cpu_count')} cores" if isinstance(load, (int, float)) else "",
    ]))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p class=\"meta\">{html.escape(meta)}</p>{body}"
        "<p class=\"note\">Scores are from one run of each model on each clip; a difference of a few hundredths on one clip is not a ranking. "
        f"Made by the PureSound web workspace on {html.escape(_when(time.time(), zone))}.</p>"
        "</main></body></html>"
    )
