"""The repository's Markdown as site HTML.

A link into the repository leads to the site's own page for that file where
there is one (the model cards) and to the file on GitHub, at the revision the
site was built from, otherwise. Headings get the anchors GitHub gives them, so
links written for GitHub keep working.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt

from site_chrome import REPO_ROOT

#: "Traditional Chinese: [README.zh-TW.md](README.zh-TW.md)" and its kin: a
#: README's pointer to its other language, which the site's switch replaces.
LANGUAGE_LINE = re.compile(r"^[^\[\n]{1,40}[:：]\s*\[[^\]\n]*\]\((?:\./)?README(?:\.zh-TW)?\.md\)\s*$", re.M)
SCHEME = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)


def slug(text: str) -> str:
    """The anchor GitHub gives a heading."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


@dataclass
class Links:
    """Where the repository's paths lead from one page."""

    repository: str | None
    revision: str
    #: Repository paths with a page of their own: path -> (page, anchor prefix).
    pages: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: From the page being rendered to the site root.
    prefix: str = ""

    def to(self, href: str, source: str, *, raw: bool = False) -> str:
        path, hashed, fragment = href.partition("#")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        if target.startswith("../") or target == "..":
            return href
        if target in self.pages:
            page, prefix = self.pages[target]
            return f"{self.prefix}{page}" + (f"#{prefix}{fragment}" if hashed else "")
        if not self.repository:
            return href
        kind = "raw" if raw else "tree" if (REPO_ROOT / target).is_dir() else "blob"
        return f"{self.repository}/{kind}/{self.revision}/{target}" + (f"#{fragment}" if hashed else "")


@dataclass
class Rendered:
    title: str | None
    html: str
    #: (level, text, anchor) of each second- and third-level heading.
    toc: list[tuple[int, str, str]]


def _parser() -> MarkdownIt:
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    md.add_render_rule("table_open", lambda *args: '<div class="table-wrap"><table class="table">\n')
    md.add_render_rule("table_close", lambda *args: "</table></div>\n")
    return md


MD = _parser()


def _text(inline) -> str:
    return "".join(child.content for child in inline.children or [] if child.type in ("text", "code_inline"))


def render(text: str, *, source: str, links: Links, anchor_prefix: str = "") -> Rendered:
    """``text`` written at ``source`` (a repository path), as HTML.

    A leading first-level heading is the title, returned apart from the body.
    ``anchor_prefix`` keeps two renderings on one page (the two languages) from
    sharing heading anchors; links within the text follow it.
    """
    tokens = MD.parse(LANGUAGE_LINE.sub("", text))
    title = None
    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
        title = _text(tokens[1])
        tokens = tokens[3:]
    seen: dict[str, int] = {}
    toc = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            heading = _text(tokens[index + 1])
            anchor = slug(heading)
            if anchor in seen:
                seen[anchor] += 1
                anchor = f"{anchor}-{seen[anchor]}"
            else:
                seen[anchor] = 0
            token.attrSet("id", anchor_prefix + anchor)
            if token.tag in ("h2", "h3"):
                toc.append((int(token.tag[1]), heading, anchor_prefix + anchor))
        for child in token.children or []:
            if child.type == "link_open":
                href = str(child.attrGet("href") or "")
                if href.startswith("#"):
                    child.attrSet("href", f"#{anchor_prefix}{href[1:]}")
                elif href and not SCHEME.match(href):
                    child.attrSet("href", links.to(href, source))
            elif child.type == "image":
                src = str(child.attrGet("src") or "")
                if src and not SCHEME.match(src):
                    child.attrSet("src", links.to(src, source, raw=True))
    return Rendered(title, MD.renderer.render(tokens, MD.options, {}), toc)


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """A document's text before its first second-level heading, and each
    second-level section as (heading, body). Code fences are not split."""
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0] * len(marker.group(1))
            elif marker.group(1).startswith(fence):
                fence = None
        if fence is None and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return "".join(preamble), [(heading, "".join(body)) for heading, body in sections]


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def translation(path: str) -> str | None:
    """The Traditional Chinese edition of a repository Markdown file, if any."""
    candidate = Path(path).with_suffix(".zh-TW.md").as_posix()
    return candidate if (REPO_ROOT / candidate).is_file() else None
