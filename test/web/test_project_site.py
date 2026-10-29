"""The project site (`tools/website/build_site.py`): what each page is made
from, where its links lead, and how benchmark runs fold into the history."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

WEBSITE = Path(__file__).resolve().parents[2] / "tools/website"
REPOSITORY = "https://github.com/example/PureSound"


@pytest.fixture(scope="module")
def site():
    sys.path.insert(0, str(WEBSITE))
    try:
        yield importlib.import_module("build_site")
    finally:
        sys.path.remove(str(WEBSITE))


@pytest.fixture(scope="module")
def bench(site):
    return sys.modules["site_benchmarks"]


@pytest.fixture(scope="module")
def markdown(site):
    return sys.modules["site_markdown"]


def _record(revision, system, created_at, rtf, *, description="A model."):
    return {
        "schema_version": 1,
        "created_at": created_at,
        "revision": revision,
        "repository": REPOSITORY,
        "machine": {"system": system, "machine": "x86_64", "cpu": f"{system} cpu", "logical_cpus": 4,
                    "onnxruntime": "1.26.0", "ci_image": None},
        "models": {"ns-model": {
            "display_name": "NS model", "task": "noise_suppression", "lifecycle": "released",
            "roles": ["default"], "description": description, "recommended_inference": {"dry_blend": 1.0},
            "card": "egs/noise_suppression/pretrained_ckpt/README.md",
            "artifacts": {"default": {"file": "ns.onnx", "description": None, "size_bytes": 2_000_000},
                          "flash": {"file": "ns_flash.onnx", "description": "int8", "size_bytes": 1_000_000}},
        }},
        "results": [
            {"model_id": "ns-model", "variant": "default", "rtf": rtf, "cold_start_seconds": 0.02,
             "native_ssm": False, "intra_op_threads": 1, "latency_ms": 30.0},
            {"model_id": "ns-model", "variant": "flash", "rtf": rtf * 0.9, "cold_start_seconds": 0.02,
             "native_ssm": False, "intra_op_threads": 1, "latency_ms": 30.0,
             "int8_relative_error": 0.0123 if system == "Linux" else 0.31},
        ],
    }


@pytest.fixture(scope="module")
def built(site, bench, tmp_path_factory):
    """The site as CI builds it, with a device payload of one model."""
    out = tmp_path_factory.mktemp("site")
    payload = tmp_path_factory.mktemp("device")
    (payload / "catalog.json").write_text(json.dumps({"version": "x", "files": {}, "models": [
        {"id": "noise-suppression-dpcrn-mamba-v2", "manifest": "assets/ns.json", "model": "assets/ns.onnx"}]}))
    history = bench.merge({}, [_record("abc1234def", "Linux", "2026-10-01T00:00:00+00:00", 0.62),
                               _record("abc1234def", "Windows", "2026-10-01T00:00:00+00:00", 0.31)], keep=10)
    site.build(out, history, repository=REPOSITORY, revision="abc1234def", device=payload)
    return out


def _page(built, name):
    return (built / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------ history

def test_one_revision_is_one_run_and_a_rerun_replaces_it(bench):
    history = bench.merge({}, [_record("aaa", "Linux", "2026-10-01T00:00:00+00:00", 0.3),
                               _record("aaa", "Darwin", "2026-10-01T00:01:00+00:00", 0.2)], keep=10)
    assert [run["revision"] for run in history["runs"]] == ["aaa"]
    assert sorted(history["runs"][0]["platforms"]) == ["linux", "macos"]

    history = bench.merge(history, [_record("aaa", "Linux", "2026-10-02T00:00:00+00:00", 0.4)], keep=10)
    (run,) = history["runs"]
    assert list(run["platforms"]) == ["linux"]
    assert run["platforms"]["linux"]["results"][0]["rtf"] == 0.4


def test_the_oldest_runs_drop_off_and_the_newest_cards_win(bench):
    history = {}
    for day in range(1, 5):
        record = _record(f"rev{day}", "Linux", f"2026-10-0{day}T00:00:00+00:00", 0.3,
                         description=f"as of day {day}")
        history = bench.merge(history, [record], keep=3)
    assert [run["revision"] for run in history["runs"]] == ["rev2", "rev3", "rev4"]
    assert history["models"]["ns-model"]["description"] == "as of day 4"
    assert history["repository"] == REPOSITORY


def test_the_int8_bound_is_the_exports(bench):
    from puresound.streaming.quantization import INT8_MAX_RELATIVE_ERROR

    assert bench.INT8_MAX_RELATIVE_ERROR == INT8_MAX_RELATIVE_ERROR


# ------------------------------------------------------------------ markdown

def test_markdown_links_lead_to_site_pages_or_the_revision_on_github(markdown):
    links = markdown.Links(REPOSITORY, "rev", {"egs/vi/pretrained_ckpt/README.md": ("cards/vi.html", "")},
                           prefix="../")
    text = ("# Card\n\nTraditional Chinese: [README.zh-TW.md](README.zh-TW.md)\n\n"
            "## Flash (int8) variants\n\n[cfg](../config/a.yaml) [vi](../../vi/pretrained_ckpt/README.md#v3) "
            "[up](#flash-int8-variants) [web](https://example.com)\n")
    page = markdown.render(text, source="egs/ns/pretrained_ckpt/README.md", links=links, anchor_prefix="zh-")

    assert page.title == "Card"
    assert "Traditional Chinese" not in page.html
    assert page.toc == [(2, "Flash (int8) variants", "zh-flash-int8-variants")]
    assert 'id="zh-flash-int8-variants"' in page.html
    assert f'href="{REPOSITORY}/blob/rev/egs/ns/config/a.yaml"' in page.html
    assert 'href="../cards/vi.html#v3"' in page.html
    assert 'href="#zh-flash-int8-variants"' in page.html and 'href="https://example.com"' in page.html


def test_sections_split_on_second_level_headings_outside_code(markdown):
    preamble, sections = markdown.split_sections("intro\n## A\n```text\n## not a heading\n```\n## B\nb\n")
    assert preamble == "intro\n"
    assert [heading for heading, _ in sections] == ["A", "B"]
    assert "## not a heading" in sections[0][1]


def test_an_lfs_pointer_gives_its_objects_size(site, tmp_path, monkeypatch):
    pages = sys.modules["site_pages"]
    (tmp_path / "model.onnx").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 21500000\n")
    monkeypatch.setattr(pages, "REPO_ROOT", tmp_path)
    assert pages.file_size("model.onnx") == 21_500_000


# ------------------------------------------------------------------ pages

def test_home_plays_what_the_payload_can_clean_and_routes_by_task(built):
    page = _page(built, "index.html")
    # The player offers only clips whose task's default model the payload carries.
    clips = json.loads(page.split('id="demo-clips">', 1)[1].split("</script>", 1)[0])
    assert [(clip["id"], clip["model"]) for clip in clips] == [("noisy-speech", "noise-suppression-dpcrn-mamba-v2")]
    assert '<script src="device/device.js"></script>' in page
    # One tile per task, worded by the README's released-models table in both languages.
    assert page.count('class="model-card task-tile"') == 3
    assert "Keeps the talker near the microphone" in page and "保留靠近麥克風的說話者" in page
    assert 'href="try.html?model=noise-suppression-dpcrn-mamba-v2"' in page
    # Speed from the latest run, for the default streaming models only.
    assert "noise-suppression-dpcrn-mamba-v2" in page.split("Speed on one CPU thread", 1)[1]
    assert 'href="start.html#quick-start"' in page and 'href="start.html#train-a-model"' in page


def test_home_and_try_it_share_the_designed_player(built):
    for name in ("index.html", "try.html"):
        page = _page(built, name)
        assert page.count('<div class="pp" data-player>') == 1 and "window.PureSoundPlayer = class" in page
        # The waveform is the timeline; there is no native audio bar.
        assert 'class="pp-wave" role="slider"' in page and "<audio" not in page


def test_get_started_is_the_readme_in_both_languages(built):
    page = _page(built, "start.html")
    assert "PyTorch toolkit for single-channel speech enhancement" in page
    assert 'id="quick-start"' in page and 'id="train-a-model"' in page and 'id="zh-' in page
    # The released-models table is the Models page's to show.
    assert 'id="released-models"' not in page
    assert "Traditional Chinese:" not in page
    assert "<project-url>" not in page and f"git clone {REPOSITORY}.git" in page


def test_models_lists_the_catalog_with_downloads_at_the_revision(built):
    page = _page(built, "models.html")
    catalog = sys.modules["site_pages"].load_catalog()
    assert page.count('<article class="model-card') == len(catalog)
    assert f'href="{REPOSITORY}/raw/abc1234def/egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8_flash.onnx"' in page
    assert 'href="cards/voice-isolation.html"' in page
    assert "uv run puresound infer speaker-verification-ps-spk-v1-1" in page
    # Only what the device payload carries can be tried.
    assert page.count('href="try.html?model=') == 1


def test_every_model_card_is_a_page_in_both_languages(built):
    pages = sys.modules["site_pages"]
    cards = pages.card_pages(pages.load_catalog())
    assert set(cards.values()) == {"cards/noise-suppression.html", "cards/voice-isolation.html",
                                   "cards/speaker-embedding.html"}
    page = _page(built, "cards/voice-isolation.html")
    assert 'href="../css/tokens.css"' in page
    assert 'class="lang-zh"' in page and 'id="zh-' in page
    assert f'href="{REPOSITORY}/blob/abc1234def/model_zoo/catalog.yaml"' in page
    assert 'href="../models.html#voice-isolate-dpcrn-curriculum-v1"' in page


def test_try_it_runs_the_device_payload_on_the_sample_clips(built):
    page = _page(built, "try.html")
    assert '<script src="device/device.js"></script>' in page
    assert '"id": "noise-suppression-dpcrn-mamba-v2"' in page and "voice-isolate-dpcrn-v8" not in page
    assert (built / "device/device.js").is_file() and (built / "device/worker.js").is_file()
    assert (built / "device/catalog.json").is_file()
    clips = [clip for clip in json.loads((WEBSITE.parents[1] / "puresound/web/static/samples/index.json")
                                          .read_text())["clips"] if clip["task"] == "enhancement"]
    for clip in clips:
        assert (built / "samples" / clip["file"]).is_file() and f'data-sample="samples/{clip["file"]}"' in page


def test_without_a_payload_try_it_says_so(site, bench, tmp_path):
    site.build(tmp_path, bench.merge({}, [], keep=10), repository=None, revision="main",
               device=tmp_path / "no-payload")
    assert "no on-device payload" in (tmp_path / "try.html").read_text(encoding="utf-8")
    assert "No runs yet." in (tmp_path / "benchmarks.html").read_text(encoding="utf-8")


def test_benchmarks_carry_cards_filters_and_flags(built):
    page = _page(built, "benchmarks.html")
    assert page.count('class="model-card bench-card is-default"') == 1
    assert 'data-filter="task"' in page and 'data-toggle="p-linux"' in page and 'data-toggle="v-flash"' in page
    # A default artifact over the budget is flagged; the flash one is not held to it.
    assert '<td class="num p-linux is-over">0.620</td>' in page
    assert '<td class="num p-linux">0.558</td>' in page
    # The flash graph's distance from its float graph, per platform; a broken one is flagged.
    assert '<td class="num p-linux quiet">0.012</td>' in page
    assert '<td class="num p-windows quiet is-over">0.310</td>' in page
    assert 'href="cards/noise-suppression.html"' in page
    assert json.loads((built / "history.json").read_text())["runs"][0]["revision"] == "abc1234def"


def test_pages_use_the_playground_assets_under_any_base_path(site, built):
    for name in sys.modules["site_chrome"].SHARED_CSS:
        assert (built / "css" / name).is_file()
    base = (built / "css/base.css").read_text(encoding="utf-8")
    assert 'url("/fonts/' not in base and 'url("../fonts/IBMPlexSans-400.woff2")' in base
    assert (built / "fonts/IBMPlexSans-400.woff2").is_file()
    for page in [*built.glob("*.html"), *built.glob("cards/*.html")]:
        assert 'href="/' not in page.read_text(encoding="utf-8"), page.name
