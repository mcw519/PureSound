import torch

from puresound.dataset.kaldi_base import KaldiFormBaseDataset


def _write_mapping(path, mapping):
    path.write_text(
        "\n".join(f"{key} {value}" for key, value in mapping.items()) + "\n",
        encoding="utf-8",
    )


def test_kaldi_dataset_reads_noisy_clean_and_enrollment(tmp_path, write_tone_wav):
    noisy = tmp_path / "noisy.wav"
    clean = tmp_path / "clean.wav"
    enroll = tmp_path / "enroll.wav"
    write_tone_wav(noisy)
    write_tone_wav(clean, freq=330.0)
    write_tone_wav(enroll, freq=440.0)
    _write_mapping(tmp_path / "wav2scp.txt", {"utt": noisy})
    _write_mapping(tmp_path / "wav2ref.txt", {"utt": clean})
    _write_mapping(tmp_path / "wav2enroll.txt", {"utt": enroll})

    dataset = KaldiFormBaseDataset(str(tmp_path), mode="dev")
    dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["name"] == "utt"
    assert sample["sr"] == 16000
    assert sample["noisy_speech"].shape == sample["clean_speech"].shape
    assert sample["conditional_speech"].numel() == sample["noisy_speech"].numel()


def test_kaldi_eval_dataset_splits_long_audio_to_chunks(tmp_path, write_tone_wav):
    noisy = tmp_path / "noisy.wav"
    write_tone_wav(noisy, duration=0.3)
    _write_mapping(tmp_path / "wav2scp.txt", {"utt": noisy})

    dataset = KaldiFormBaseDataset(
        str(tmp_path),
        mode="eval",
        split_to_chunks_with_size=0.1,
    )
    sample = dataset[0]

    assert sample["clean_speech"].numel() == 0
    assert sample["noisy_speech"].dim() == 2
    assert sample["noisy_speech"].shape[-1] == 1600
    assert torch.isfinite(sample["noisy_speech"]).all()
