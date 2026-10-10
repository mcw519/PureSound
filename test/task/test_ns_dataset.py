"""`NoiseSuppressionDataset`: the training-row contract and the dataset-level
behaviour of its augmentation blocks."""

import math

import pytest
import torch

from puresound.audio.vad import frame_count
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset

SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 1.0,
    "add_n_cases": 1,
    "snr_range": [-5, 5],
}


def _dataset_args(metafile_path, **overrides):
    args = {
        "metafile_path": str(metafile_path),
        "min_utt_length_in_seconds": 0.05,
        "min_utts_in_each_speaker": 2,
        "target_sr": 16000,
        "training_sample_length_in_seconds": 0.1,
        "audio_gain_normalized_to": None,
    }
    args.update(overrides)
    return args


def test_a_row_and_a_batch_have_the_training_shapes(tmp_path, write_puresound_metafile):
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


def _mixed_rate_metafile(tmp_path, write_tone_wav):
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
                f"{uttid}, {spkid}, m, {wav_path}, {int(sample_rate * 0.25)}, {sample_rate}, 1"
            )
    metafile = tmp_path / "meta.csv"
    metafile.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metafile


@pytest.mark.parametrize("pool", ["one-rate", "mixed-rates"])
def test_speech_interference_is_drawn_and_resampled_to_the_target_rate(
    tmp_path, write_puresound_metafile, write_tone_wav, pool
):
    if pool == "one-rate":
        metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
        key = ("corpus_spk0", 16000)
    else:
        metafile = _mixed_rate_metafile(tmp_path, write_tone_wav)
        key = ("corpus_spk0", None)
    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=SPEECH_ARGS,
        vad_label_args={"used": False},
    )

    sample = dataset[key]

    assert sample["audio_sr"] == 16000
    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)


def test_the_capture_floor_lands_in_the_mixture_and_never_in_the_reference(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    write_tone_wav(tmp_path / "noise" / "n0.wav", duration=0.5, freq=500.0)

    def _sample(floor_cfg):
        noise_args = {
            "used": False,  # SNR-relative noise off
            "prob": 0.0,
            "noise_folder": str(tmp_path / "noise"),
        }
        if floor_cfg is not None:
            noise_args["absolute_floor"] = floor_cfg
        ds = NoiseSuppressionDataset(
            **_dataset_args(
                metafile,
                min_utts_in_each_speaker=1,
                training_sample_length_in_seconds=0.2,
                audio_gain_normalized_to=-28,
            ),
            augmentation_speech_args=None,
            augmentation_noise_args=noise_args,
            vad_label_args={"used": False},
        )
        return ds[("corpus_spk0", 16000, 7)]

    with_floor = _sample({"used": True, "prob": 1.0, "level_dbfs_range": [-45.0, -45.0]})
    without = _sample(None)

    floor = with_floor["noisy_speech"] - without["noisy_speech"]
    rms_dbfs = 20.0 * math.log10(float(floor.square().mean().sqrt()))
    assert rms_dbfs == pytest.approx(-45.0, abs=1.5)
    assert torch.equal(with_floor["clean_speech"], without["clean_speech"])


def test_voice_isolation_blocks_are_rejected_rather_than_ignored(
    tmp_path, write_puresound_metafile
):
    """mix_mode belongs to `VoiceIsolationDataset`; the generic dataset fails
    fast instead of silently dropping it."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    speech_args = dict(SPEECH_ARGS)
    speech_args["mix_mode"] = {
        "used": True,
        "modes": [{"name": "physical", "prob": 1.0, "physical": True}],
    }
    with pytest.raises(ValueError, match="voice_isolation"):
        NoiseSuppressionDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=speech_args,
            vad_label_args={"used": False},
        )


def test_a_key_that_names_an_utterance_loads_that_utterance(tmp_path, write_puresound_metafile):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", utterances_per_speaker=4)
    dataset = NoiseSuppressionDataset(**_dataset_args(metafile))
    for _ in range(5):
        _, _, picked = dataset.choose_an_utterance_by_speaker_name(
            "corpus_spk1", select_channel=0, utterance="corpus_spk1_utt2"
        )
        assert picked == ("corpus_spk1", "corpus_spk1_utt2")
    with pytest.raises(KeyError):
        dataset.choose_an_utterance_by_speaker_name("corpus_spk1", utterance="corpus_spk0_utt0")

    asked = []
    choose = dataset.choose_an_utterance_by_speaker_name
    dataset.choose_an_utterance_by_speaker_name = lambda **kw: asked.append(kw) or choose(**kw)
    row = dataset[("corpus_spk1", None, 3, None, None, "corpus_spk1_utt2")]
    assert asked[0]["utterance"] == "corpus_spk1_utt2"
    assert row["clean_speech"].shape == (1, 1600)


def test_named_empty_audio_is_not_silently_replaced_by_another_utterance(
    tmp_path, write_puresound_metafile,
):
    import numpy as np
    import soundfile as sf
    metafile = write_puresound_metafile(tmp_path / "meta.csv", utterances_per_speaker=4)
    dataset = NoiseSuppressionDataset(**_dataset_args(metafile))
    utterance = "corpus_spk1_utt2"
    path = dataset.meta["corpus_spk1"]["utts"][utterance]["path"]
    sf.write(path, np.zeros(1600), 16000)
    with pytest.raises(ValueError, match="contains no usable audio"):
        dataset.choose_an_utterance_by_speaker_name("corpus_spk1", select_channel=0, utterance=utterance)
