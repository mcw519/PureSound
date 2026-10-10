"""Tokens and word alignment for reading transcripts against each other.

``tools/wer.py`` scores corpora of English read speech.  A listening check in
the web workspace needs two more things: the *alignment itself* (which words
were dropped, added or changed, and where they sit in time), and scripts
without spaces.  Text in Chinese, Japanese or Korean is scored per character
-- a character error rate -- since a recogniser's word segmentation of those
languages is its own guess and not something to count errors against.

The edit costs and tie-breaking match ``tools.wer.edit_counts``, so a count
read here and one read there agree on the same pair of strings.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

# Ideographs, kana and hangul: scored one character per token.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
#: A hypothesis this many times longer than its reference is a recogniser loop
#: (see ``tools.wer.LOOP_RATIO``), not a transcription.
LOOP_RATIO = 1.5


def is_cjk(token: str) -> bool:
    return bool(_CJK.fullmatch(token))


def normalise(text: str) -> str:
    """NFKC, lower case, punctuation and symbols to spaces, whitespace collapsed.

    Apostrophes inside words stay ("don't"), hyphens split ("well-known"), so
    English reads as ``tools.wer.normalise`` reads it.
    """

    text = unicodedata.normalize("NFKC", str(text or "")).lower().replace("-", " ")
    kept = "".join(char if char == "'" or unicodedata.category(char)[0] in {"L", "N", "M"} else " " for char in text)
    # An apostrophe is kept only between two letters or digits.
    kept = re.sub(r"(?<![^\W_])'|'(?![^\W_])", " ", kept)
    return " ".join(kept.split())


def tokens(text: str) -> list[str]:
    """Scoring tokens: words, except CJK characters, which stand alone."""

    out: list[str] = []
    for chunk in normalise(text).split():
        word = ""
        for char in chunk:
            if is_cjk(char):
                if word:
                    out.append(word)
                    word = ""
                out.append(char)
            else:
                word += char
        if word:
            out.append(word)
    return out


def unit(token_list: Sequence[str]) -> str:
    """``character`` when most tokens are CJK characters, else ``word``."""

    if not token_list:
        return "word"
    cjk = sum(1 for token in token_list if is_cjk(token))
    return "character" if cjk * 2 >= len(token_list) else "word"


def timed_tokens(words: Iterable[Any]) -> tuple[list[str], list[tuple[float | None, float | None]]]:
    """Tokens of recognised words, each with the time span of its word.

    A word that splits into several tokens (CJK characters) shares its span
    out evenly, so every token can still be found on the timeline.
    """

    out: list[str] = []
    times: list[tuple[float | None, float | None]] = []
    for word in words:
        text = getattr(word, "text", None) if not isinstance(word, dict) else word.get("text")
        start = getattr(word, "start", None) if not isinstance(word, dict) else word.get("start")
        end = getattr(word, "end", None) if not isinstance(word, dict) else word.get("end")
        pieces = tokens(text or "")
        for index, piece in enumerate(pieces):
            if start is not None and end is not None and len(pieces) > 1:
                step = (float(end) - float(start)) / len(pieces)
                times.append((float(start) + index * step, float(start) + (index + 1) * step))
            else:
                times.append((None if start is None else float(start), None if end is None else float(end)))
            out.append(piece)
    return out, times


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> list[dict[str, Any]]:
    """Levenshtein alignment as operations, in order.

    Each is ``{"op": "hit" | "sub" | "del" | "ins", "ref": i | None,
    "hyp": j | None}`` with indices into the two token lists.  A deletion is a
    reference token the hypothesis lacks -- the failure an enhancement model
    must not cause.
    """

    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[0] * cols for _ in range(rows)]  # 0 = match/sub, 1 = deletion, 2 = insertion
    for i in range(1, rows):
        cost[i][0] = i
        back[i][0] = 1
    for j in range(1, cols):
        cost[0][j] = j
        back[0][j] = 2
    for i in range(1, rows):
        ref_token = reference[i - 1]
        previous, current = cost[i - 1], cost[i]
        for j in range(1, cols):
            sub = previous[j - 1] + (ref_token != hypothesis[j - 1])
            dele = previous[j] + 1
            ins = current[j - 1] + 1
            best = min(sub, dele, ins)
            current[j] = best
            back[i][j] = 0 if best == sub else (1 if best == dele else 2)
    ops: list[dict[str, Any]] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 0:
            ops.append({"op": "hit" if reference[i - 1] == hypothesis[j - 1] else "sub", "ref": i - 1, "hyp": j - 1})
            i, j = i - 1, j - 1
        elif move == 1:
            ops.append({"op": "del", "ref": i - 1, "hyp": None})
            i -= 1
        else:
            ops.append({"op": "ins", "ref": None, "hyp": j - 1})
            j -= 1
    ops.reverse()
    return ops


def counts(ops: Sequence[dict[str, Any]], reference_length: int) -> dict[str, Any]:
    tally = {"hit": 0, "sub": 0, "del": 0, "ins": 0}
    for op in ops:
        tally[op["op"]] += 1
    words = max(int(reference_length), 1)
    return {
        **tally,
        "ref_tokens": int(reference_length),
        "error_rate": (tally["sub"] + tally["del"] + tally["ins"]) / words,
        "del_rate": tally["del"] / words,
        "ins_rate": tally["ins"] / words,
        "sub_rate": tally["sub"] / words,
    }


def looped(reference_length: int, hypothesis_length: int) -> bool:
    return reference_length > 0 and hypothesis_length > LOOP_RATIO * reference_length


__all__ = ["LOOP_RATIO", "align", "counts", "is_cjk", "looped", "normalise", "timed_tokens", "tokens", "unit"]
