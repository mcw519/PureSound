"""Word-error accounting: each error type counted as itself."""

import pytest

from puresound.evaluation.tools.wer import (
    LOOP_RATIO,
    edit_counts,
    hypothesis_row,
    loop_count,
    normalise,
    rates,
)


@pytest.mark.parametrize(
    "text, normalised",
    [("Please call Stella.", "please call stella"), ("please  call stella", "please call stella"),
     ("well-known", "well known"), ("He said, 'hello there.'", "he said hello there"),
     ("rock 'n' roll", "rock n roll"), ("Don't stop", "don't stop")],
    ids=["punctuation-and-case", "whitespace", "hyphen-splits", "quotation-marks",
         "quoted-letter", "contraction-kept"],
)
def test_normalisation_removes_what_is_not_an_error(text, normalised):
    assert normalise(text) == normalised


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        ("please call stella", "please call stella", {"wer": 0.0, "del": 0.0, "ins": 0.0, "sub": 0.0}),
        ("please call stella", "please stella", {"wer": 1 / 3, "del": 1 / 3, "ins": 0.0, "sub": 0.0}),
        ("please call stella", "please do call stella", {"wer": 1 / 3, "del": 0.0, "ins": 1 / 3, "sub": 0.0}),
        ("please call stella", "please call stellar", {"wer": 1 / 3, "del": 0.0, "ins": 0.0, "sub": 1 / 3}),
        ("please call stella", "", {"wer": 1.0, "del": 1.0, "ins": 0.0, "sub": 0.0}),
        ("", "hello", {"ins": 1.0}),
        ("a b c d e", "a x c e f", None),
    ],
    ids=["exact", "deletion", "insertion", "substitution", "dropped-sentence",
         "empty-reference", "mixed"],
)
def test_each_error_type_is_counted_as_itself(reference, hypothesis, expected):
    """Deletions are the one with a direction, so they must not be reported as
    substitutions -- a model removing speech has to be distinguishable from a
    recogniser mishearing it. WER is always the sum of the three rates, and an
    empty reference does not divide by zero."""
    scored = rates(edit_counts(reference, hypothesis))
    for key, value in (expected or {}).items():
        assert scored[key] == pytest.approx(value), key
    if reference:
        assert scored["wer"] == pytest.approx(scored["del"] + scored["ins"] + scored["sub"])


def test_a_hypothesis_row_carries_the_words_and_their_counts():
    """The stage reports rates; the row is what lets a rate be traced back to
    what the recogniser actually wrote."""
    row = hypothesis_row("u1", "model", "the cat sat", "the cat sat down")
    assert row["id"] == "u1" and row["system"] == "model"
    assert row["hypothesis"] == "the cat sat down" and row["ins"] == pytest.approx(1 / 3)
    assert row["ref_words"] == 3 and row["hit"] == 3


def test_a_looped_transcription_is_counted_apart_from_the_rates():
    """Whisper answers a segment it cannot parse by repeating a phrase. One such
    hypothesis has an insertion rate far above the corpus mean, so a handful of
    them decide the corpus number -- they have to be countable before a delta is
    read as evidence about the model."""
    reference = " ".join(["word"] * 10)
    normal = hypothesis_row("a", "model", reference, " ".join(["word"] * 11))
    looped = hypothesis_row("b", "model", reference, " ".join(["holy spirit"] * 30))
    assert loop_count([normal, looped]) == 1
    assert loop_count([normal, looped], ratio=100.0) == 0
    assert LOOP_RATIO > 1.0
