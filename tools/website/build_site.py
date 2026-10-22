"""Build the project site: home, models, model cards, an in-browser try-out and
the benchmark dashboard, all made from this repository.

    python tools/website/build_site.py --out site \\
        --history site/history.json --add bench/*.json --device build/device

Every page is drawn with the web playground's own stylesheets and fonts and
loads nothing from elsewhere. Text comes from the READMEs (English and
Traditional Chinese, switched on the page), the model catalog and the model
cards; links into the repository lead to GitHub at the revision built from.

``--history`` is the benchmark history to extend with the records in ``--add``
(see site_benchmarks.py); it is written to ``--out`` as history.json.
``--device`` is a folder holding the on-device payload, as
``sdk/web/tools/build_assets.py --dest`` writes it; without one, the payload
beside the playground's device.js is used if it has been built. Needs PyYAML
and markdown-it-py, not the project.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from site_benchmarks import SCHEMA_VERSION, merge  # noqa: E402
from site_benchmarks import render as benchmarks  # noqa: E402
from site_chrome import REPO_ROOT, STATIC, copy_assets  # noqa: E402
from site_markdown import Links  # noqa: E402
from site_pages import Site, card, card_pages, home, load_catalog, models, page_links, start, try_it  # noqa: E402

#: The device payload's own files, kept in the repository beside what is built.
DEVICE_SOURCES = ("device.js", "worker.js")
#: The sample clips a speech-enhancement model can be tried on.
SAMPLE_TASK = "enhancement"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def repository() -> str | None:
    """The repository's web address: GitHub Actions says it; otherwise origin."""
    server, name = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
    if server and name:
        return f"{server}/{name}"
    url = _git("config", "--get", "remote.origin.url")
    if not url:
        return None
    if url.startswith("git@"):
        host, _, path = url[4:].partition(":")
        url = f"https://{host}/{path}"
    return url.removesuffix(".git")


def revision() -> str:
    return os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "main"


def copy_device(payload: Path | None, out: Path) -> list[str]:
    """The on-device payload beside the pages; the models it carries."""
    payload = payload or STATIC / "device"
    if not (payload / "catalog.json").is_file():
        return []
    dest = out / "device"
    shutil.copytree(payload, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*"))
    for name in DEVICE_SOURCES:
        shutil.copyfile(STATIC / "device" / name, dest / name)
    catalog = json.loads((payload / "catalog.json").read_text(encoding="utf-8"))
    return [entry["id"] for entry in catalog.get("models", [])]


def copy_samples(out: Path) -> list[dict]:
    index = json.loads((STATIC / "samples/index.json").read_text(encoding="utf-8"))
    clips = [clip for clip in index.get("clips", []) if clip.get("task") == SAMPLE_TASK]
    (out / "samples").mkdir(parents=True, exist_ok=True)
    for clip in clips:
        shutil.copyfile(STATIC / "samples" / clip["file"], out / "samples" / clip["file"])
    return clips


def build(out: Path, history: dict, *, repository: str | None, revision: str, device: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    copy_assets(out)
    catalog = load_catalog()
    cards = card_pages(catalog)
    links = Links(repository, revision, page_links(cards))
    site = Site(catalog=catalog, links=links, cards=cards, device_models=copy_device(device, out))
    samples = copy_samples(out)
    pages = {
        "index.html": home(site, history, samples),
        "start.html": start(site),
        "models.html": models(site),
        "try.html": try_it(site, samples),
        "benchmarks.html": benchmarks(history, links),
    }
    for path, target in cards.items():
        pages[target] = card(path, target, site)
    for target, html in pages.items():
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        (out / target).write_text(html, encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="directory for the site")
    parser.add_argument("--history", type=Path, help="benchmark history to extend (default: OUT/history.json)")
    parser.add_argument("--add", type=Path, nargs="*", default=[], help="benchmark records to fold in")
    parser.add_argument("--keep", type=int, default=200, help="benchmark runs to keep (default: 200)")
    parser.add_argument("--device", type=Path, help="on-device payload folder (from build_assets.py --dest)")
    parser.add_argument("--repository", help="repository web address (default: from GitHub Actions or origin)")
    parser.add_argument("--revision", help="revision the site is built from (default: GITHUB_SHA or HEAD)")
    args = parser.parse_args()
    history_path = args.history or args.out / "history.json"
    history = (json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file()
               else {"schema_version": SCHEMA_VERSION, "runs": []})
    history = merge(history, [json.loads(path.read_text(encoding="utf-8")) for path in args.add], args.keep)
    build(args.out, history, repository=args.repository or history.get("repository") or repository(),
          revision=args.revision or revision(), device=args.device)
    print(f"{len(history['runs'])} benchmark runs -> {args.out}")


if __name__ == "__main__":
    main()
