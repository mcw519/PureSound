"""The three synthesis-realism knobs from the domain-gap measurements:
room-colored noise, absolute capture floor, distance-level SIR.

Each must (a) do what it claims when enabled and (b) leave the RNG stream
untouched when absent/disabled, so existing recipes regenerate bit-identically.
"""
import math

import pytest
import torch

from puresound.task.ns import NoiseSuppressionDataset
from puresound.config.augmentation import MixModeEntry
from puresound.task.voice_isolation import VoiceIsolationDataset


_SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 1.0,
    "add_n_cases": 1,
    "snr_range": [-5, 5],
}


def _dataset_args(metafile):
    return {
        "metafile_path": str(metafile),
        "min_utt_length_in_seconds": 0.05,
        "min_utts_in_each_speaker": 1,
        "target_sr": 16000,
        "training_sample_length_in_seconds": 0.2,
        "audio_gain_normalized_to": -28,
    }


# --------------------------------------------------------------------------- #
# absolute capture floor
# --------------------------------------------------------------------------- #
def test_absolute_floor_level_lands_in_range(tmp_path, write_puresound_metafile, write_tone_wav):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    write_tone_wav(tmp_path / "noise" / "n0.wav", duration=0.5, freq=500.0)

    def _sample(floor_cfg):
        noise_args = {
            "used": False,                       # SNR-relative noise off
            "prob": 0.0,
            "noise_folder": str(tmp_path / "noise"),
        }
        if floor_cfg is not None:
            noise_args["absolute_floor"] = floor_cfg
        ds = NoiseSuppressionDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=None,       # no interferers
            augmentation_noise_args=noise_args,
            vad_label_args={"used": False},
        )
        return ds[("corpus_spk0", 16000, 7)]

    with_floor = _sample({"used": True, "prob": 1.0, "level_dbfs_range": [-45.0, -45.0]})
    without = _sample(None)

    floor = with_floor["noisy_speech"] - without["noisy_speech"]
    rms_dbfs = 20.0 * math.log10(float(floor.square().mean().sqrt()))
    assert rms_dbfs == pytest.approx(-45.0, abs=1.5)
    # capture noise never enters the clean reference
    assert torch.equal(with_floor["clean_speech"], without["clean_speech"])


def test_noise_knobs_disabled_are_bit_identical(tmp_path, write_puresound_metafile, write_tone_wav):
    """Absent vs {'used': False} vs missing keys must consume identical RNG."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    write_tone_wav(tmp_path / "noise" / "n0.wav", duration=0.5, freq=500.0)

    def _sample(noise_args):
        ds = NoiseSuppressionDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=_SPEECH_ARGS,
            augmentation_noise_args=noise_args,
            vad_label_args={"used": False},
        )
        return ds[("corpus_spk0", 16000, 123)]

    baseline = _sample(None)
    disabled = _sample({
        "used": False, "prob": 0.9, "snr_range": [0, 20], "prob_white_noise": 0.1,
        "white_noise_snr_range": [10, 30],
        "noise_folder": str(tmp_path / "noise"),
        "room_coloring": {"used": False, "prob": 1.0},
        "absolute_floor": {"used": False, "prob": 1.0, "level_dbfs_range": [-45, -35]},
    })
    assert torch.allclose(baseline["noisy_speech"], disabled["noisy_speech"])
    assert torch.allclose(baseline["clean_speech"], disabled["clean_speech"])


# --------------------------------------------------------------------------- #
# room-colored noise (augmentor-level: the transform reaches the mixing)
# --------------------------------------------------------------------------- #
def test_noise_transform_is_applied_before_mixing(tmp_path, write_tone_wav):
    from puresound.audio.augmentation import AudioEffectAugmentor

    noise_dir = tmp_path / "noise"
    write_tone_wav(noise_dir / "n0.wav", duration=0.5, freq=500.0)

    aug = AudioEffectAugmentor()
    aug.load_bg_noise_from_folder(str(noise_dir))

    torch.manual_seed(0)
    wav = 0.05 * torch.randn(1, 8000)
    seen = {}

    def transform(n):
        seen["called"] = True
        return torch.zeros_like(n)     # silence the noise entirely

    _, (added_dry, _, _) = aug.add_bg_noise(wav=wav.clone(), snr_list=[0.0], sr=16000)
    _, (added_silenced, _, _) = aug.add_bg_noise(
        wav=wav.clone(), snr_list=[0.0], sr=16000, noise_transform=transform
    )
    assert seen.get("called")
    assert float(added_dry[0].abs().sum()) > 0
    assert float(added_silenced[0].abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# distance-level SIR
# --------------------------------------------------------------------------- #
def _bare_vi() -> VoiceIsolationDataset:
    return object.__new__(VoiceIsolationDataset)


def test_distance_level_sir_follows_inverse_distance_law():
    vi = _bare_vi()
    mode = MixModeEntry(name="distance_level", distance_level=True, jitter_db=[0.0, 0.0])
    sir = vi._distance_level_sir(
        mode,
        fg_metadata={"source_receiver_distance": 0.5},
        interferer_metadata=[
            {"source_receiver_distance": 3.0},
            {"source_receiver_distance": 2.0},   # nearest -> the loudest
        ],
    )
    assert sir == pytest.approx(20.0 * math.log10(2.0 / 0.5))  # +12.04 dB


def test_distance_level_sir_falls_back_when_geometry_is_unknown():
    vi = _bare_vi()
    mode = MixModeEntry(name="distance_level", distance_level=True)
    assert vi._distance_level_sir(mode, None, [{"source_receiver_distance": 2.0}]) is None
    assert vi._distance_level_sir(mode, {"source_receiver_distance": 0.5}, []) is None
    assert (
        vi._distance_level_sir(
            mode,
            {"source_receiver_distance": None},
            [{"source_receiver_distance": 2.0}],
        )
        is None
    )
