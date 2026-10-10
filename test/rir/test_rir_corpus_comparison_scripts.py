import json
import sys

import numpy as np
import pytest
import soundfile as sf

from egs.rir_generation.phases.m6_bank.scripts import (
    compare_measured_tilt_by_corpus,
    compare_noise_floor_by_corpus,
)

SAMPLE_RATE = 16000


def _write_flat_bank(root, rooms_by_corpus):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    for corpus, room_count in rooms_by_corpus.items():
        for room in range(room_count):
            stem = f"{corpus}_room{room}_0000"
            decay = np.exp(-6.9 * time / (0.3 + 0.1 * room))
            rir = (rng.standard_normal((SAMPLE_RATE, 2)) * decay[:, None]).astype(np.float32)
            rir[50] = 3.0
            sf.write(root / f"{stem}.wav", rir, SAMPLE_RATE, subtype="FLOAT")
            channel_map = [
                {"channel": 0, "label": "near_0", "distance_m": 0.5},
                {"channel": 1, "label": "far_0", "distance_m": 2.5},
            ]
            (root / f"{stem}.json").write_text(
                json.dumps({"scene": {"channel_map": channel_map}}), encoding="utf-8"
            )


@pytest.mark.parametrize(
    "rooms_by_corpus",
    [{"solo": 4}, {"a": 2, "b": 2}, {"a": 4, "b": 1}],
    ids=["one-corpus", "too-few-rooms", "one-eligible-corpus"],
)
def test_tilt_comparison_is_inconclusive_without_two_comparable_corpora(
    rooms_by_corpus, tmp_path, monkeypatch, capsys
):
    _write_flat_bank(tmp_path / "bank", rooms_by_corpus)
    monkeypatch.setattr(
        sys, "argv", ["compare_measured_tilt_by_corpus.py", "--root", str(tmp_path / "bank")]
    )

    assert compare_measured_tilt_by_corpus.main() == 1
    assert "INCONCLUSIVE" in capsys.readouterr().err


def test_tilt_comparison_reaches_a_verdict_with_two_comparable_corpora(
    tmp_path, monkeypatch, capsys
):
    _write_flat_bank(tmp_path / "bank", {"a": 3, "b": 3})
    monkeypatch.setattr(
        sys, "argv", ["compare_measured_tilt_by_corpus.py", "--root", str(tmp_path / "bank")]
    )

    assert compare_measured_tilt_by_corpus.main() in (0, 1)
    output = capsys.readouterr().out
    assert "POOLABLE" in output or "NOT POOLABLE" in output


def test_noise_floor_comparison_skips_the_level_test_with_one_corpus(
    tmp_path, monkeypatch, capsys
):
    _write_flat_bank(tmp_path / "bank", {"solo": 4})
    monkeypatch.setattr(
        sys, "argv", ["compare_noise_floor_by_corpus.py", "--measured", str(tmp_path / "bank")]
    )

    compare_noise_floor_by_corpus.main()

    output = capsys.readouterr().out
    assert "not comparable" in output
    assert "poolable as a target   None" in output
