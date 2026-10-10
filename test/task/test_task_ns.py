"""``augmentation_row_initial_ambient`` on the noise-suppression dataset.

The two contracts: (1) inside the lead there is NO speech in noisy or clean --
only what the noise stage adds -- and the lead edge is a ramp, not a click;
(2) with the block absent or disabled, nothing changes, RNG stream included.
"""
import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.config.augmentation import RowInitialAmbientAugmentation
from puresound.task.ns import NoiseSuppressionDataset

SR = 16000


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("ns_corpus")
    rows = ["uttid, spkid, gender, path, length, sample rate, channels"]
    time = torch.arange(SR * 6) / SR
    for speaker in range(2):
        for utterance in range(2):
            path = root / f"s{speaker}_{utterance}.wav"
            wav = 0.1 * torch.sin(2 * torch.pi * (200 + 50 * speaker) * time)
            AudioIO.save(wav.view(1, -1), str(path), SR)
            rows.append(f"s{speaker}_{utterance}, spk{speaker}, m, {path}, {SR * 6}, {SR}, 1")
    meta = root / "meta.csv"
    meta.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return meta


def _row(corpus, ambient, seed=3):
    dataset = NoiseSuppressionDataset(
        metafile_path=str(corpus), min_utt_length_in_seconds=1.0,
        min_utts_in_each_speaker=1, target_sr=SR, training_sample_length_in_seconds=4.0,
        audio_gain_normalized_to=-28, augmentation_row_initial_ambient_args=ambient,
    )
    return dataset[("spk0", SR, seed)]


def test_the_schema_rejects_a_lead_it_cannot_draw():
    with pytest.raises(Exception):
        RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[0.0, 4.0])
    with pytest.raises(Exception):
        RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[4.0, 1.0])
    RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[1.0, 4.0])


def test_the_lead_is_speech_free_and_its_edge_is_a_ramp(corpus):
    """Noise is off, so the lead is digital silence; past the lead the row is the
    one the recipe without the block produces, faded in over ``fade_ms``."""
    plain = _row(corpus, None)
    led = _row(corpus, {"used": True, "prob": 1.0, "lead_seconds_range": [2.0, 2.0]})
    lead, fade = 2 * SR, int(0.05 * SR)
    ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, 1.0, fade) * torch.pi)
    for key in ("noisy_speech", "clean_speech"):
        assert float(led[key][..., :lead].abs().max()) == 0.0
        assert float(plain[key][..., :lead].abs().max()) > 0.0
        assert torch.allclose(led[key][..., lead : lead + fade],
                              plain[key][..., lead : lead + fade] * ramp, atol=1e-6)
        assert torch.equal(led[key][..., lead + fade :], plain[key][..., lead + fade :])


def test_a_disabled_block_is_bit_identical(corpus):
    plain = _row(corpus, None)
    disabled = _row(corpus, {"used": False, "prob": 1.0, "lead_seconds_range": [2.0, 2.0]})
    for key in ("noisy_speech", "clean_speech", "far_target"):
        assert torch.equal(plain[key], disabled[key]), key
