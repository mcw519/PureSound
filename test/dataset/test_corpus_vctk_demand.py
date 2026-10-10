"""VCTK-DEMAND's layout, as knowledge the repo keeps rather than retypes."""

import json
import re

import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.dataset.corpus.vctk_demand import (
    DEFAULT_CLEAN_TEST,
    DEFAULT_CLEAN_TRAIN,
    NOISE_SPLITS,
    SPEAKER_PATTERN,
    extract_noise,
    main,
    read_noise_log,
    resolve_subdir,
)


@pytest.fixture
def corpus_root(tmp_path, write_tone_wav):
    (tmp_path / "testset_txt").mkdir()
    for index in range(4):
        speaker = f"p{232 + index % 2}"
        name = f"{speaker}_{index:03d}.wav"
        write_tone_wav(tmp_path / "noisy_testset_wav" / name, sample_rate=48000, duration=0.4)
        write_tone_wav(tmp_path / "clean_testset_wav" / name, sample_rate=48000, duration=0.4)
        (tmp_path / "testset_txt" / f"{speaker}_{index:03d}.txt").write_text(
            "Please call Stella.", encoding="utf-8"
        )
    return tmp_path


def _ensure(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def noise_root(tmp_path):
    """Two noise types across the two splits; noisy = clean + a known noise."""
    generator = torch.Generator().manual_seed(0)
    for split, (noisy_name, clean_name, log_name) in NOISE_SPLITS.items():
        lines = []
        for index in range(3):
            stem = f"p{240 + index}_{index:03d}"
            noise_type = "bus" if split == "testset" else ("car" if index else "bus")
            clean = 0.1 * torch.sin(torch.arange(4800) / 10.0).view(1, -1)
            noise = 0.02 * torch.randn(1, 4800, generator=generator)
            AudioIO.save(clean, str(_ensure(tmp_path / clean_name) / f"{stem}.wav"), 48000)
            AudioIO.save(clean + noise, str(_ensure(tmp_path / noisy_name) / f"{stem}.wav"), 48000)
            lines.append(f"{stem} {noise_type} 5")
        (tmp_path / log_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_the_layout_names_the_speaker_and_prefers_the_published_training_split():
    """The files sit in one flat directory, so without the pattern every utterance
    would look like its own speaker and the split would be disjoint in name only.
    The 28-speaker split is the one the published baselines train on."""
    match = re.search(SPEAKER_PATTERN, "p232_001.wav")
    assert match is not None and match.group(1) == "p232"
    assert DEFAULT_CLEAN_TRAIN[0] == "clean_trainset_28spk_wav"


def test_known_folders_are_found_without_being_named_and_an_override_wins(corpus_root, tmp_path):
    assert resolve_subdir(corpus_root, None, DEFAULT_CLEAN_TEST) == corpus_root / "clean_testset_wav"
    (corpus_root / "elsewhere").mkdir()
    assert resolve_subdir(corpus_root, "elsewhere", DEFAULT_CLEAN_TEST) == corpus_root / "elsewhere"
    assert resolve_subdir(corpus_root / "testset_txt", None, DEFAULT_CLEAN_TEST) is None


@pytest.mark.parametrize("with_transcripts", [True, False])
def test_testset_writes_a_scored_set_with_optional_transcripts(corpus_root, tmp_path, with_transcripts):
    out = tmp_path / "set"
    extra = [] if with_transcripts else ["--no-transcripts"]
    assert main(["testset", str(corpus_root), "--out-dir", str(out), *extra]) == 0

    rows = [line for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 4
    assert ('"transcript": "Please call Stella."' in rows[0]) == with_transcripts
    assert ("transcript" in "".join(rows)) == with_transcripts


@pytest.mark.parametrize("subcommand", ["testset", "noise"])
def test_a_root_without_the_expected_folders_is_refused(tmp_path, subcommand):
    assert main([subcommand, str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 1


def test_the_noise_log_maps_utterance_to_type_and_drops_the_snr_column(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("p257_001 bus 1.750000e+01\np257_002 cafe 1.250000e+01\n", encoding="utf-8")
    assert read_noise_log(log) == {"p257_001": "bus", "p257_002": "cafe"}


def test_noise_is_the_difference_grouped_by_type_across_splits(noise_root, tmp_path):
    out = tmp_path / "noise"
    report = extract_noise(noise_root, out, sample_rate=16000)
    # 3 bus (test) + 1 bus (train) + 2 car (train) -> two tracks, six files.
    assert set(report.tracks) == {"bus", "car"} and report.files == 6
    bus, sr = AudioIO.open(str(out / "bus.wav"))
    assert sr == 16000 and bus.shape[-1] == 4 * 1600
    car, _ = AudioIO.open(str(out / "car.wav"))
    assert car.shape[-1] == 2 * 1600
    # It is noise, not speech: no tone survives the subtraction.
    assert float(bus.abs().mean()) < 0.02
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["tracks"]["bus"]["files"] == 4 and index["splits"] == ["testset", "trainset_28spk"]


def test_an_unlogged_file_is_named_not_guessed_and_an_unknown_split_is_refused(noise_root, tmp_path):
    (noise_root / "log_testset.txt").write_text("p240_000 bus 5\n", encoding="utf-8")
    report = extract_noise(noise_root, tmp_path / "noise", splits=("testset",))
    assert report.skipped_unlogged == ["p241_001", "p242_002"]
    assert "not in the log" in report.summary()

    with pytest.raises(ValueError, match="unknown split"):
        extract_noise(noise_root, tmp_path / "out", splits=("dev",))
