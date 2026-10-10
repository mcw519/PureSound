"""LibriLight selection and segmentation: cleanest first, held-out readers never."""

import json

import numpy as np
import pytest
import soundfile as sf

from puresound.dataset.corpus.librilight import main, plan_segments, select_chapters
from puresound.dataset.corpus.records import read_metafile


def _chapter(speaker, snr, seconds=3600, name=None):
    return {"flac": name or f"{speaker}_{snr}.flac", "speaker": speaker, "snr": snr,
            "voice_activity": [[0.0, seconds]], "subset": "small"}


@pytest.mark.parametrize(
    "chapters, hours, cap, excluded, expected",
    [
        ([_chapter("a", 5.0), _chapter("b", 20.0), _chapter("c", 12.0)], 2, None, set(), ["b", "c"]),
        # A held-out reader is refused however clean.
        ([_chapter("test", 40.0), _chapter("train", 10.0)], 10, None, {"test"}, ["train"]),
        # The per-reader cap spreads the hours.
        ([_chapter("a", 30.0 - i, name=f"a{i}.flac") for i in range(5)] + [_chapter("b", 1.0)],
         3, 2, set(), ["a", "a", "b"]),
        # A NaN SNR does not scramble the ranking.
        ([_chapter("a", float("nan")), _chapter("b", -20.0), _chapter("c", 30.0), _chapter("d", 25.0)],
         2, None, set(), ["c", "d"]),
    ],
    ids=["cleanest-first", "held-out-reader", "per-reader-cap", "nan-snr"],
)
def test_selection_is_cleanest_first_within_the_budget(chapters, hours, cap, excluded, expected):
    chosen = select_chapters(chapters, hours=hours, max_hours_per_speaker=cap, excluded=excluded)
    assert [c["speaker"] for c in chosen] == expected


@pytest.mark.parametrize(
    "activity, context, duration, expected",
    [
        # 0-9 merged across a 0.3 s pause; 12-15.5 merged; the 40 s region split in two.
        ([[0.0, 4.0], [4.3, 9.0], [12.0, 15.0], [15.2, 15.5], [20.0, 60.0]], 0.0, None,
         [(0.0, 9.0), (12.0, 15.5), (20.0, 40.0), (40.0, 60.0)]),
        # Short pieces are dropped and context is clipped to the file.
        ([[0.1, 1.0], [5.0, 9.9]], 0.25, 10.0, [(4.75, 10.0)]),
    ],
    ids=["pauses-and-limits", "short-and-context"],
)
def test_segments_break_at_long_pauses_and_respect_the_length_limits(activity, context, duration, expected):
    kwargs = {"duration": duration} if duration is not None else {}
    pieces = plan_segments(activity, min_seconds=3.0, max_seconds=20.0, max_gap=1.0,
                           context=context, **kwargs)
    assert pieces == expected


def test_segment_writes_pieces_and_a_metafile(tmp_path):
    root = tmp_path / "LibriLight"
    chapter = root / "small" / "1234" / "book" / "ch1.flac"
    chapter.parent.mkdir(parents=True)
    sf.write(str(chapter), (0.1 * np.random.default_rng(0).standard_normal(16000 * 30)).astype("float32"), 16000)
    meta = {"speaker": "1234", "snr": 15.0, "voice_activity": [[1.0, 8.0], [12.0, 20.0]]}
    chapter.with_suffix(".json").write_text(json.dumps(meta))
    held_out = tmp_path / "test-clean" / "999"
    held_out.mkdir(parents=True)

    selection = tmp_path / "sel.jsonl"
    assert main(["select", str(root), "--hours", "1", "--exclude-speakers-in",
                 str(tmp_path / "test-clean"), "--out", str(selection)]) == 0
    out = tmp_path / "ll.csv"
    assert main(["segment", str(selection), "--source-root", str(root),
                 "--dest-root", str(tmp_path / "seg"), "--out", str(out), "--jobs", "1"]) == 0

    records = read_metafile(out)
    assert len(records) == 2 and {r.spkid for r in records} == {"ll_1234"}
    assert all(r.path.is_file() and r.sample_rate == 16000 for r in records)


def test_a_rerun_with_other_knobs_describes_the_pieces_that_are_on_disk(tmp_path):
    root = tmp_path / "LibriLight"
    chapter = root / "small" / "1234" / "book" / "ch1.flac"
    chapter.parent.mkdir(parents=True)
    sf.write(str(chapter), (0.1 * np.random.default_rng(0).standard_normal(16000 * 30)).astype("float32"), 16000)
    meta = {"speaker": "1234", "snr": 15.0, "voice_activity": [[1.0, 8.0], [12.0, 20.0]]}
    chapter.with_suffix(".json").write_text(json.dumps(meta))
    selection = tmp_path / "sel.jsonl"
    assert main(["select", str(root), "--hours", "1", "--out", str(selection)]) == 0

    def segment(context):
        out = tmp_path / f"ll{context}.csv"
        assert main(["segment", str(selection), "--source-root", str(root),
                     "--dest-root", str(tmp_path / "seg"), "--out", str(out),
                     "--context", str(context), "--jobs", "1"]) == 0
        return read_metafile(out)

    first, second = segment(0.25), segment(1.0)

    assert [r.path for r in first] == [r.path for r in second]
    assert [r.length for r in second] == [r.length for r in first] == [
        sf.info(str(r.path)).frames for r in second
    ]
