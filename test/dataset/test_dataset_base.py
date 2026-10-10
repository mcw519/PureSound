"""The dataset layer under the task datasets: the metafile parser, the dynamic
base dataset, and the frozen Kaldi-format reader."""

import random

import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.audio.vad import frame_count
from puresound.dataset.dynamic_base import DynamicBaseDataset
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from puresound.dataset.parser import MetafileParser

HEADER = "uttid, spkid, gender, path, length, sample rate, channels"
ROW0 = "utt0, corpus_spk0, m, /tmp/utt0.wav, 16000, 16000, 1"
ROW1 = "utt1, corpus_spk0, m, /tmp/utt1.wav, 32000, 16000, 1"


@pytest.mark.parametrize(
    "rows",
    [[ROW0, ROW1], [HEADER, ROW0, HEADER, ROW1]],
    ids=["no-header", "embedded-headers"],
)
def test_metafile_parser_keeps_every_data_row_and_no_header_row(tmp_path, rows):
    metafile = tmp_path / "meta.csv"
    metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")

    meta = MetafileParser.read_from_metafile(str(metafile), use_speaker_as_key=True)

    assert sorted(meta["corpus_spk0"]["utts"]) == ["utt0", "utt1"]


def _dataset_args(metafile_path):
    return {
        "metafile_path": str(metafile_path),
        "min_utt_length_in_seconds": 0.05,
        "min_utts_in_each_speaker": 2,
        "target_sr": 16000,
        "training_sample_length_in_seconds": 0.1,
        "audio_gain_normalized_to": None,
    }


def test_dynamic_dataset_filters_metadata_and_creates_vad_targets(
    tmp_path, write_puresound_metafile
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv")
    dataset = DynamicBaseDataset(
        **_dataset_args(metafile),
        vad_label_args={
            "used": True,
            "backend": "energy",
            "args": {"frame_length": 80, "hop_length": 40},
        },
    )
    wav = torch.ones(1, 1600)

    assert sorted(dataset.total_spks) == ["corpus_spk0", "corpus_spk1"]
    assert dataset.spk2idx["corpus_spk0"] == 0
    assert dataset.sr_meta[16000]
    assert dataset.create_vad_target(wav, 16000).shape[0] == frame_count(1600, 80, 40)
    assert dataset.create_empty_vad_target(wav).sum() == 0


def test_sr_keyed_utterance_pool_keeps_metafile_order(
    tmp_path, write_puresound_metafile, monkeypatch
):
    """The sr-keyed branch must hand `random.sample` an ordered list.

    Built from a set, the index `random.sample` draws lands in a str-hash-ordered
    sequence, so the same per-item seed picks a different utterance in every
    process -- silently defeating the seeded sampler. The failure only shows
    across PYTHONHASHSEED, so pin the observable invariant instead: the pool is
    exactly `sr_meta[sr][spk]`, in metafile order, before and after
    `ignoring_utt_list` filtering.
    """
    metafile = write_puresound_metafile(tmp_path / "meta.csv", utterances_per_speaker=8)
    dataset = DynamicBaseDataset(**_dataset_args(metafile))
    expected = list(dataset.sr_meta[16000]["corpus_spk0"])
    assert len(expected) == 8, "need a pool long enough that set order cannot coincide"

    pools = []

    def _spy(population, k):
        pools.append(list(population))
        return list(population)[:k]

    monkeypatch.setattr(random, "sample", _spy)

    dataset.choose_an_utterance_by_speaker_name(
        target_speaker_name="corpus_spk0", select_with_sr_as_key=16000
    )
    assert pools == [expected]

    pools.clear()
    ignored = [expected[1], expected[5]]
    dataset.choose_an_utterance_by_speaker_name(
        target_speaker_name="corpus_spk0",
        ignoring_utt_list=ignored,
        select_with_sr_as_key=16000,
    )
    assert pools == [[key for key in expected if key not in set(ignored)]]


def _write_speaker_metafile(tmp_path, utterances):
    """One speaker ``spk``; ``utterances`` is ``[(sample_rate, silent)]``."""
    rows = [HEADER]
    for index, (sample_rate, silent) in enumerate(utterances):
        path = tmp_path / f"utt{index}.wav"
        wav = torch.zeros(1, sample_rate) if silent else 0.1 * torch.randn(1, sample_rate)
        AudioIO.save(wav, str(path), sample_rate)
        rows.append(f"c_spk_{index}, c_spk, m, {path}, {sample_rate}, {sample_rate}, 1")
    metafile = tmp_path / "meta.csv"
    metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metafile


@pytest.mark.parametrize("target_sr, key_rate", [(None, 8000), (16000, 8000)])
def test_row_length_comes_from_the_target_rate_or_else_the_keys_own(
    tmp_path, write_puresound_metafile, target_sr, key_rate
):
    """With no target rate the utterance is not open yet when the key is parsed,
    so the key's sample rate is the only rate there is to convert seconds with."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv")
    dataset = DynamicBaseDataset(**{**_dataset_args(metafile), "target_sr": target_sr})

    dataset.parse_item_key(("corpus_spk0", key_rate, None, 3.0))

    assert dataset.sample_length == 3 * (target_sr or key_rate)


def test_a_redraw_after_a_silent_file_keeps_the_keyed_sample_rate(tmp_path):
    metafile = _write_speaker_metafile(
        tmp_path, [(16000, True)] + [(16000, False)] * 3 + [(8000, False)] * 4
    )
    dataset = DynamicBaseDataset(
        metafile_path=str(metafile), min_utt_length_in_seconds=0.1, min_utts_in_each_speaker=2
    )
    state = random.getstate()
    try:
        for seed in range(40):
            random.seed(seed)
            _, sample_rate, _ = dataset.choose_an_utterance_by_speaker_name(
                "c_spk", select_with_sr_as_key=16000
            )
            assert sample_rate == 16000
    finally:
        random.setstate(state)


def test_a_speaker_of_silent_files_fails_with_the_timeout_not_a_recursion_error(tmp_path):
    metafile = _write_speaker_metafile(tmp_path, [(16000, True)] * 4)
    dataset = DynamicBaseDataset(
        metafile_path=str(metafile),
        min_utt_length_in_seconds=0.1,
        min_utts_in_each_speaker=2,
        target_sr=16000,
    )

    with pytest.raises(RuntimeError, match="Timeout"):
        dataset.choose_an_utterance_by_speaker_name("c_spk")


def _write_mapping(path, mapping):
    path.write_text(
        "\n".join(f"{key} {value}" for key, value in mapping.items()) + "\n",
        encoding="utf-8",
    )


def test_kaldi_dataset_reads_dev_triples_and_chunks_eval_audio(tmp_path, write_tone_wav):
    noisy = tmp_path / "noisy.wav"
    clean = tmp_path / "clean.wav"
    enroll = tmp_path / "enroll.wav"
    write_tone_wav(noisy, duration=0.3)
    write_tone_wav(clean, duration=0.3, freq=330.0)
    write_tone_wav(enroll, duration=0.3, freq=440.0)
    _write_mapping(tmp_path / "wav2scp.txt", {"utt": noisy})
    _write_mapping(tmp_path / "wav2ref.txt", {"utt": clean})
    _write_mapping(tmp_path / "wav2enroll.txt", {"utt": enroll})

    dev = KaldiFormBaseDataset(str(tmp_path), mode="dev")
    dev.folder_content = {"wav2enroll": "wav2enroll.txt"}
    sample = dev[0]
    assert len(dev) == 1
    assert sample["name"] == "utt"
    assert sample["sr"] == 16000
    assert sample["noisy_speech"].shape == sample["clean_speech"].shape
    assert sample["conditional_speech"].numel() == sample["noisy_speech"].numel()

    # eval mode needs no reference list
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_mapping(eval_dir / "wav2scp.txt", {"utt": noisy})
    evaluation = KaldiFormBaseDataset(
        str(eval_dir), mode="eval", split_to_chunks_with_size=0.1
    )
    sample = evaluation[0]
    assert sample["clean_speech"].numel() == 0
    assert sample["noisy_speech"].dim() == 2
    assert sample["noisy_speech"].shape[-1] == 1600
    assert torch.isfinite(sample["noisy_speech"]).all()
