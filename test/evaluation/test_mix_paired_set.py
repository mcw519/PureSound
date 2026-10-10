"""A WER set mixed from a transcript-bearing corpus and a noise pool."""

import json
import random

import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.evaluation.tools.mix_paired_set import crop_noise, main, mix_set

SR = 16000


@pytest.fixture
def corpus(tmp_path, write_tone_wav):
    speech = tmp_path / "speech"
    for index in range(6):
        # Nested like LibriTTS (<speaker>/<chapter>/<utt>), long and short utterances.
        folder = speech / f"{100 + index % 2}" / "1"
        duration = 1.0 if index < 4 else 0.3
        write_tone_wav(folder / f"u{index}.wav", sample_rate=24000, duration=duration, freq=300 + index)
        (folder / f"u{index}.normalized.txt").write_text(f"sentence {index}\n", encoding="utf-8")
    write_tone_wav(speech / "100" / "1" / "orphan.wav", sample_rate=24000, duration=1.0)

    noise = tmp_path / "noise"
    noise.mkdir()
    generator = torch.Generator().manual_seed(0)
    AudioIO.save(0.05 * torch.randn(1, 2 * SR, generator=generator), str(noise / "bus.wav"), SR)
    AudioIO.save(0.05 * torch.randn(1, SR // 4, generator=generator), str(noise / "short.wav"), SR)
    return speech, noise, tmp_path / "out"


def _rows(out):
    return [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_item_carries_its_transcript_and_lands_in_the_snr_window(corpus):
    speech, noise, out = corpus
    written = mix_set(
        speech, noise, out, n=10, snr_range=(-10.0, 0.0), min_duration=0.5,
        transcript_suffix=".normalized.txt", sample_rate=SR,
    )
    rows = _rows(out)
    # Four utterances are long enough; the orphan has no transcript.
    assert written == len(rows) == 4
    for row in rows:
        assert row["transcript"].startswith("sentence ")
        assert row["sample_rate"] == SR and row["noise"] in {"bus", "short"}
        # No reverb, so the measured SNR is the target -- that is the point of the set.
        assert row["snr_db"] == pytest.approx(row["snr_target_db"], abs=0.05)
        assert -10.0 <= row["snr_target_db"] <= 0.0
        assert (out / row["mix"]).is_file() and (out / row["clean"]).is_file()
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["kind"] == "mixed" and provenance["skipped_without_transcript"] == 1
    assert provenance["skipped_outside_duration"] == 2 and provenance["noise_tracks"] == ["bus", "short"]


def test_the_same_seed_writes_the_same_set(corpus, tmp_path):
    speech, noise, _ = corpus
    outs = [tmp_path / "a", tmp_path / "b"]
    for out in outs:
        mix_set(speech, noise, out, n=4, snr_range=(0.0, 5.0), min_duration=0.5,
                transcript_suffix=".normalized.txt", sample_rate=SR, seed=7)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "source"} for r in rows]
    assert strip(_rows(outs[0])) == strip(_rows(outs[1]))


def test_a_track_shorter_than_the_utterance_is_tiled_not_refused():
    track = torch.arange(5, dtype=torch.float32).view(1, -1)
    cropped = crop_noise(track, 12, random.Random(0))
    assert cropped.shape == (1, 12) and cropped[0, 5] == 0 and cropped[0, 11] == 1


def test_a_window_nothing_falls_in_is_refused_with_the_counts(corpus):
    speech, noise, out = corpus
    with pytest.raises(ValueError, match="6 outside the window"):
        mix_set(speech, noise, out, n=4, snr_range=(0.0, 5.0), min_duration=30.0,
                transcript_suffix=".normalized.txt", sample_rate=SR)


def test_the_cli_refuses_an_inverted_window_and_reports_a_shortfall(corpus, capsys):
    speech, noise, out = corpus
    common = ["--speech-dir", str(speech), "--noise-dir", str(noise), "--out-dir", str(out),
              "--transcript-suffix", ".normalized.txt"]
    assert main([*common, "--snr", "5", "0"]) == 1
    assert main([*common, "--n", "50", "--min-duration", "0.5"]) == 0
    assert "only 4 utterances met" in capsys.readouterr().out
