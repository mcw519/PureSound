import random

import torch

from puresound.audio.vad import frame_count
from puresound.dataset.dynamic_base import DynamicBaseDataset
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset


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

    It once built that pool with `list(set(...))`, so the index `random.sample`
    draws landed in a str-hash-ordered sequence and the same per-item seed
    picked a different utterance in every process -- silently defeating the
    reproducibility the seeded sampler exists to provide. The failure is
    invisible in-process (it only shows up across PYTHONHASHSEED), so pin the
    observable invariant instead: the pool is exactly `sr_meta[sr][spk]`, in
    metafile order, before and after `ignoring_utt_list` filtering.
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


def test_noise_suppression_dataset_returns_training_contract(
    tmp_path, write_puresound_metafile
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv")
    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        vad_label_args={
            "used": True,
            "backend": "energy",
            "args": {"frame_length": 80, "hop_length": 40},
        },
    )

    sample = dataset[("corpus_spk0", 16000)]

    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)
    assert sample["consistency_noise"].shape == (1, 1600)
    assert sample["audio_sr"] == 16000
    assert sample["audio_length"] == 1600
    assert sample["vad_target"].shape[0] == frame_count(1600, 80, 40)

    batch = NoiseSuppressionCollateFunc()([sample, sample])

    assert batch["noisy_speech"].shape == (2, 1600)
    assert batch["clean_speech"].shape == (2, 1600)
    assert batch["vad_target"].shape[0] == 2


def test_noise_suppression_dataset_samples_speech_interference_from_sequence(
    tmp_path, write_puresound_metafile
):
    metafile = write_puresound_metafile(
        tmp_path / "meta.csv", speakers=3, utterances_per_speaker=2
    )
    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        augmentation_speech_args={
            "used": True,
            "is_target": False,
            "prob": 1.0,
            "add_n_cases": 1,
            "snr_range": [-5, 5],
        },
        vad_label_args={"used": False},
    )

    sample = dataset[("corpus_spk0", 16000)]

    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)


def test_noise_suppression_dataset_resamples_interference_from_mixed_sr_pool(
    tmp_path, write_tone_wav
):
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
    for spk_idx, sample_rate in enumerate([22050, 24000, 48000]):
        spkid = f"corpus_spk{spk_idx}"
        for utt_idx in range(2):
            uttid = f"{spkid}_utt{utt_idx}"
            wav_path = tmp_path / "wav" / spkid / f"{uttid}.wav"
            write_tone_wav(
                wav_path,
                sample_rate=sample_rate,
                duration=0.25,
                freq=180.0 + spk_idx * 80.0 + utt_idx * 10.0,
            )
            rows.append(
                f"{uttid}, {spkid}, m, {wav_path}, "
                f"{int(sample_rate * 0.25)}, {sample_rate}, 1"
            )
    metafile = tmp_path / "meta.csv"
    metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        augmentation_speech_args={
            "used": True,
            "is_target": False,
            "prob": 1.0,
            "add_n_cases": 1,
            "snr_range": [-5, 5],
        },
        vad_label_args={"used": False},
    )

    sample = dataset[("corpus_spk0", None)]

    assert sample["audio_sr"] == 16000
    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)
