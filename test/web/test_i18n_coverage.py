"""Every English string the interface shows has a Traditional Chinese entry.

Keys come from three places: text and attributes marked in ``index.html``
(``data-i18n``, ``data-i18n-attr``), the same marks inside markup the scripts
generate, and literal ``t("…")`` calls.  A missing entry would show English in
a Chinese page; an empty one would blank the string, so both fail here."""

from __future__ import annotations

import html as html_module
import json
import re
from html.parser import HTMLParser
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "puresound" / "web" / "static"
SCRIPTS = ("world.js", "app.js", "shell.js", "pipeline.js", "compare-deck.js", "transcribe.js", "capture.js", "audio-view.js", "annotate.js", "device/device.js")
ENTRY = re.compile(r'^\s*("(?:[^"\\]|\\.)*"):\s*("(?:[^"\\]|\\.)*"),?\s*$')


def _dictionary() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (STATIC / "i18n-zh.js").read_text().splitlines():
        match = ENTRY.match(line)
        if match:
            key, value = json.loads(match.group(1)), json.loads(match.group(2))
            assert key not in entries, f"duplicate entry: {key}"
            entries[key] = value
    return entries


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _Marks(HTMLParser):
    """data-i18n keys and data-i18n-attr values, and marked elements that
    contain other elements (their text would be replaced wholesale)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.keys: set[str] = set()
        self.nested: list[str] = []
        self.open: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for mark in self.open:
            mark["children"] = True
        for name in (attrs.get("data-i18n-attr") or "").split(","):
            name = name.strip()
            if name and attrs.get(name):
                self.keys.add(_clean(attrs[name]))
        if "data-i18n" in attrs and tag not in {"input", "img", "br"}:
            self.open.append({"tag": tag, "key": attrs["data-i18n"] or "", "text": "", "children": False})

    def handle_data(self, data):
        for mark in self.open:
            mark["text"] += data

    def handle_endtag(self, tag):
        if self.open and self.open[-1]["tag"] == tag:
            mark = self.open.pop()
            key = _clean(mark["key"] or mark["text"])
            if key:
                self.keys.add(key)
            if mark["children"]:
                self.nested.append(key)


def _markup_keys(html: str) -> tuple[set[str], list[str]]:
    parser = _Marks()
    parser.feed(html)
    return parser.keys, parser.nested


STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|`((?:[^`\\]|\\.)*)`')
# Calls whose string arguments are keys: t("…"), a ternary inside t(…), setKey(el, "…").
CALL = re.compile(r"\b(?:t|setKey)\(")


def _argument(code: str, start: int) -> str:
    """The first argument of the call whose "(" is at ``start - 1``."""
    depth, index = 0, start
    while index < len(code):
        char = code[index]
        if char in "\"'`":
            match = STRING.match(code, index)
            if match:
                index = match.end()
                continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif char == "," and depth == 0:
            break
        index += 1
    return code[start:index]


def _script_keys(code: str) -> set[str]:
    keys: set[str] = set()
    for call in CALL.finditer(code):
        argument = _argument(code, call.end())
        if call.group(0).startswith("setKey"):  # setKey(element, key): the key is the second argument
            argument = _argument(code[call.end() + len(argument) + 1:], 0)
        # Keys are the argument's own string literals (a ternary's branches),
        # not those inside a nested call or index, nor those compared against.
        depth = 0
        for match in STRING.finditer(argument):
            between = argument[:match.start()]
            depth = sum(between.count(char) for char in "([{") - sum(between.count(char) for char in ")]}")
            text = next(group for group in match.groups() if group is not None)
            tail = argument[match.end():].lstrip()
            head = between.rstrip()
            if depth or tail.startswith(("===", "!==", "==", "!=")) or head.endswith(("===", "!==", "==", "!=")):
                continue
            if "${" in text or not re.search(r"[A-Za-z]", text):
                continue
            keys.add(json.loads(f'"{text}"') if match.group(1) is not None else text)
    # Marked text inside generated markup: data-i18n>English</
    for match in re.finditer(r'data-i18n(?:="")?>([^<>${}]+)</', code):
        keys.add(_clean(html_module.unescape(match.group(1))))
    for match in re.finditer(r'emptyText: "([^"]+)"', code):
        keys.add(match.group(1))
    return keys


# Keys looked up by value (t(model.lifecycle), t(job.status), ...): every value
# the server or the page can produce.
DYNAMIC = {
    # lifecycle, roles, job states, gate verdicts
    "released", "candidate", "reference", "experimental", "diagnostic", "default", "deprecated",
    "succeeded", "failed", "running", "queued", "cancelled", "pass", "fail", "unresolved",
    # tasks, input modes, monitor modes
    "Voice isolation", "Noise suppression", "Speaker verification",
    "Upload a noisy recording", "Record with your microphone", "Live through the model",
    "Nothing plays. The model still processes everything; after you stop, listen to the input and the output side by side below.",
    "You hear the model's output as you speak, a little over a tenth of a second late. Use headphones — through speakers the output feeds back into the microphone.",
    "You hear your microphone straight through, unprocessed. Switch between this and Model output while talking to hear what the model takes out: a live A/B.",
    # pipeline stage states and source roles
    "applied on this row", "not in this recipe", "probability 0 for these rows", "off in the inspector",
    "not needed on this row", "did not fire on this row", "did not fire on this row · p = {p}",
    "foreground", "interferer", "media source", "residual echo", "noise, through the room", "other",
    # the word check's language menu
    "Auto-detect", "English (US)", "English (UK)",
    # track and curve names the player translates when it draws them
    "Input", "Output", "Removed", "Previous", "Unprocessed", "Clean reference", "Before", "After", "Mixture", "Target",
    "Interferers", "Model output", "Stage change", "Effective SNR", "as the model hears it", "input − output",
    "aligned, before the next model", "the input, at the model rate", "what the output should approach",
    "the microphone, as streamed", "what the loss compares the output with", "the other talkers' bus at this point",
    "after − before: what this stage added or removed", "per 20 ms: target against everything else, −30 to 60 dB",
    "Microphone", "This device", "Target talkers", "All speech", "Near region", "what the microphone picked up",
    "microphone − output", "reference: every target talker", "reference: every talker", "reference: talkers inside the near radius",
    "this source alone, at the microphone", "the same model in this browser",
    # Annotate's keyboard list (annotate.js KEYS)
    "Transport", "Spans", "View", "Files", "Level", "play / pause", "play the selection", "loop the selection",
    "nudge 1 s (⇧ 0.1 s)", "add a span from the selection", "pick a tag", "start at the playhead", "end at the playhead",
    "delete the selected span", "clear the selection", "zoom in / out / fit", "zoom at the pointer", "scroll the timeline",
    "previous / next file", "open audio", "open a folder", "import spans", "save spans into the folder",
    "download the export", "auto boost (clip-free)",
}


def test_every_marked_or_translated_string_has_a_traditional_chinese_entry():
    zh = _dictionary()
    html = (STATIC / "index.html").read_text()
    keys, nested = _markup_keys(html)
    for name in SCRIPTS:
        path = STATIC / name
        if path.exists():
            keys |= _script_keys(path.read_text())
    keys |= DYNAMIC
    missing = sorted(key for key in keys if key not in zh)
    assert not missing, f"{len(missing)} strings have no zh-TW entry:\n" + "\n".join(missing)
    assert not nested, f"data-i18n on elements with children (their markup would be lost): {nested}"


def test_the_dictionary_has_no_empty_or_untranslated_entries():
    zh = _dictionary()
    assert len(zh) > 100
    assert not [key for key, value in zh.items() if not value.strip()]
    # Brand, model and unit strings stay English on purpose; everything else
    # should read as Chinese.
    same = [key for key, value in zh.items() if key == value and re.search(r"[a-z]{3}", key) and not re.search(r"[A-Z]{2}|\d", key)]
    assert not same, f"entries identical to their English: {same}"
