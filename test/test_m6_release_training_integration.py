import torch
import pytest

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_variant_release
from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.rir.bank.loader import PreGeneratedReleaseBank
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset


def test_augmentor_consumes_m6_release_recipe_without_split_leakage(tmp_path):
    report = validate_m6_variant_release.build_report(tmp_path / "fixture")
    release_root = report["release"]["root"]
    augmentor = AudioEffectAugmentor()
    augmentor.init_room_bank(
        {
            "used": True,
            "bank_type": "release",
            "folder": release_root,
            "recipe_id": "synthetic_calibrated",
            "split": "train",
            "usage_role": "train",
        }
    )

    assert augmentor.room_bank_kind == "release"
    assert isinstance(augmentor.room_bank, PreGeneratedReleaseBank)
    scene = augmentor.sample_room_scene()
    assert scene["split"] == "train"
    assert scene["release_recipe_id"] == "synthetic_calibrated"

    reverberant, (_rir_id, info) = augmentor.apply_rir(
        wav=torch.randn(1, 16000),
        rir_mode="full",
        sr=16000,
        room_scene=scene,
        source_role="foreground",
    )
    metadata = info["metadata"]
    assert reverberant.ndim == 2
    assert metadata["split"] == "train"
    assert metadata["release_recipe_id"] == "synthetic_calibrated"
    assert metadata["release_variant_id"] == "synthetic_calibrated"
    assert metadata["release_sha256"] == report["release"]["release_sha256"]


def test_release_training_config_requires_recipe_and_split(tmp_path):
    augmentor = AudioEffectAugmentor()
    with pytest.raises(ValueError, match="requires recipe_id"):
        augmentor.init_room_bank(
            {
                "bank_type": "release",
                "folder": str(tmp_path),
                "split": "train",
                "usage_role": "train",
            }
        )
    with pytest.raises(ValueError, match="explicit split"):
        augmentor.init_room_bank(
            {
                "bank_type": "release",
                "folder": str(tmp_path),
                "recipe_id": "synthetic_calibrated",
                "usage_role": "train",
            }
        )

    with pytest.raises(ValueError, match="usage_role"):
        augmentor.init_room_bank(
            {
                "bank_type": "release",
                "folder": str(tmp_path),
                "recipe_id": "synthetic_calibrated",
                "split": "test",
            }
        )
    with pytest.raises(ValueError, match="must match"):
        augmentor.init_room_bank(
            {
                "bank_type": "release",
                "folder": str(tmp_path),
                "recipe_id": "synthetic_calibrated",
                "split": "test",
                "usage_role": "train",
            }
        )


def test_room_bank_rejects_release_only_options(tmp_path):
    augmentor = AudioEffectAugmentor()
    with pytest.raises(ValueError, match="release-only options"):
        augmentor.init_room_bank(
            {
                "bank_type": "room",
                "folder": str(tmp_path),
                "recipe_id": "synthetic_calibrated",
            }
        )


def test_release_provenance_survives_dataset_and_collate():
    dataset = object.__new__(NoiseSuppressionDataset)
    sample = {}
    foreground = {
        "release_id": "release-a",
        "release_sha256": "1" * 64,
        "release_recipe_id": "synthetic_calibrated",
        "release_variant_id": "synthetic_calibrated",
        "split": "train",
        "origin": "synthetic",
        "renderer_profile_id": "profile-a",
    }
    dataset._emit_task_metadata(
        sample,
        foreground_metadata=foreground,
        interferer_metadata=[foreground],
        target_absent=False,
        background_speech_reference=None,
    )
    assert sample["rir_release_id"] == "release-a"
    assert sample["rir_recipe_id"] == "synthetic_calibrated"
    assert sample["rir_variant_id"] == "synthetic_calibrated"
    assert sample["rir_split"] == "train"

    collate_item = {
        "noisy_speech": torch.zeros(1, 8),
        "clean_speech": torch.zeros(1, 8),
        "consistency_noise": torch.zeros(1, 8),
        "speaker_id": 0,
        "audio_sr": 16000,
        "audio_length": 8,
        **sample,
    }
    batch = NoiseSuppressionCollateFunc()([collate_item])
    assert batch["rir_release_id"] == ["release-a"]
    assert batch["rir_split"] == ["train"]
