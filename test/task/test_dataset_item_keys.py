"""Sampler-selected utterances and row lengths across the training datasets."""

import pytest

from puresound.task.ns import NoiseSuppressionDataset
from puresound.task.sv import SpeakerEmbeddingDataset
from puresound.task.tse import TargetSpeakerExtractDataset


@pytest.mark.parametrize("dataset_cls", [
    NoiseSuppressionDataset, SpeakerEmbeddingDataset, TargetSpeakerExtractDataset,
])
@pytest.mark.parametrize("interferers", [False, True])
def test_selected_utterances_keep_the_scheduled_length_and_clear_it_on_next_item(
    tmp_path, write_puresound_metafile, monkeypatch, dataset_cls, interferers,
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", utterances_per_speaker=4)
    kwargs = {}
    if interferers:
        kwargs["augmentation_speech_args"] = {
            "used": True, "is_target": False, "prob": 1.0,
            "add_n_cases": 1, "snr_range": [-5, 5],
        }
    if dataset_cls is TargetSpeakerExtractDataset:
        kwargs["enroll_speech_args"] = {
            "enroll_length_seconds": 0.1, "gain_normalized_to": None,
            "add_inactive_target": {"used": False}, "add_noise": {"used": False},
            "add_reverb": {"used": False}, "add_volume": {"used": False},
        }
    dataset = dataset_cls(
        metafile_path=str(metafile), min_utt_length_in_seconds=0.05,
        min_utts_in_each_speaker=2, target_sr=16000,
        training_sample_length_in_seconds=0.1, audio_gain_normalized_to=None, **kwargs,
    )
    selected = []
    choose = dataset.choose_an_utterance_by_speaker_name

    def load(*args, **kwargs):
        result = choose(*args, **kwargs)
        if kwargs.get("utterance") is not None:
            selected.append(result[2][1])
        return result

    monkeypatch.setattr(dataset, "choose_an_utterance_by_speaker_name", load)
    for seconds in (0.05, 0.2, None):
        row = dataset[("corpus_spk1", None, 3, seconds, None, "corpus_spk1_utt2")]
        assert row["noisy_speech"].shape == (1, int((seconds or 0.1) * 16000))
        if "clean_speech" in row:
            assert row["clean_speech"].shape == row["noisy_speech"].shape
    assert selected == ["corpus_spk1_utt2"] * 3
