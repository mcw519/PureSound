"""The screen's verdict: words a recogniser is sure of are speech; babble is noise."""
from types import SimpleNamespace

import pytest

from puresound.dataset.corpus.speech_screen import judge


def _seg(text, logprob=-0.1, no_speech=0.05):
    return SimpleNamespace(text=text, avg_logprob=logprob, no_speech_prob=no_speech)


@pytest.mark.parametrize(
    "segments, intelligible, details",
    [
        ([_seg(" please call stella and ask her")], True, {}),
        ([_seg(" the and so i", logprob=-1.4), _seg(" yeah", no_speech=0.8)], False, {"words": 0}),
        ([_seg(" yes")], False, {}),
        ([_seg(" 지하철이 곧 도착합니다")], True, {}),  # unspaced scripts count by character
        ([], False, {"min_no_speech": 1.0}),
    ],
    ids=["confident-sentence", "unsure-babble", "single-word", "unspaced-script", "no-segments"],
)
def test_only_confident_multi_word_speech_is_intelligible(segments, intelligible, details):
    verdict = judge(segments)
    assert verdict["intelligible"] == intelligible
    for key, value in details.items():
        assert verdict[key] == value
