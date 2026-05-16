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
        "audio_gain_nomalized_to": None,
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
