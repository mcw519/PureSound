from pathlib import Path
import pytest
import torch

from egs.rir_generation.phases.m6_bank.scripts import validate_m6_variant_release
from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.rir.bank.loader import PreGeneratedReleaseBank
from puresound.task.ns import NoiseSuppressionCollateFunc, NoiseSuppressionDataset

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow


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


def test_union_can_add_a_release_recipe_beside_a_folder_bank(tmp_path):
    """Widening a folder-bank recipe with an M6 release keeps both provenances.

    This is the bank-expansion shape: the existing pool stays exactly what it
    was and the release rides alongside it at a fixed weight, with usage_role
    injected once at the top level the way the dataset does it.
    """
    report = validate_m6_variant_release.build_report(tmp_path / "fixture")
    release_root = Path(report["release"]["root"])
    folder_member = release_root / "variants" / "calibrated"

    augmentor = AudioEffectAugmentor()
    augmentor.init_room_bank(
        {
            "used": True,
            "usage_role": "train",
            "banks": [
                {
                    "name": "folder",
                    "weight": 0.6,
                    "bank_type": "room",
                    "folder": str(folder_member),
                    "split": "train",
                },
                {
                    "name": "m6",
                    "weight": 0.4,
                    "bank_type": "release",
                    "folder": str(release_root),
                    "recipe_id": "synthetic_calibrated",
                    "split": "train",
                },
            ],
        }
    )

    assert augmentor.room_bank_kind == "union"
    assert augmentor.room_bank.weights == pytest.approx([0.6, 0.4])
    assert "folder w=0.60" in augmentor.room_bank.describe()

    by_member = {}
    for _ in range(120):
        scene = augmentor.sample_room_scene()
        _reverberant, (_rir_id, info) = augmentor.apply_rir(
            wav=torch.randn(1, 16000),
            rir_mode="full",
            sr=16000,
            room_scene=scene,
            source_role="foreground",
        )
        metadata = info["metadata"]
        assert metadata["split"] == "train"
        by_member.setdefault(metadata["union_member_name"], []).append(metadata)
    assert set(by_member) == {"folder", "m6"}
    # Release identity travels only with the release member's draws.
    assert all("release_sha256" not in m for m in by_member["folder"])
    assert all(
        m["release_sha256"] == report["release"]["release_sha256"]
        for m in by_member["m6"]
    )


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


def test_release_audit_is_memoized_and_skippable(tmp_path):
    """A full audit reads the whole release; a training start must not repeat it."""
    from puresound.audio.rir.bank.release import (
        AUDIT_CACHE_NAME,
        audit_m6_variant_release,
    )

    report = validate_m6_variant_release.build_report(tmp_path / "fixture")
    release_root = Path(report["release"]["root"])
    # Building the fixture already audits it once, which now memoizes the verdict.
    cache_path = release_root / AUDIT_CACHE_NAME
    assert cache_path.is_file()
    cache_path.unlink()

    first = audit_m6_variant_release(release_root)
    assert first["valid"]
    assert cache_path.is_file()                      # verdict memoized beside the release

    # A hit skips re-reading the bank's audio: hide one RIR, which only the cached
    # verdict can survive. This is exactly the scope the cache trusts.
    rir = next(release_root.glob("variants/*/**/*.wav"))
    hidden = rir.with_suffix(".hidden")
    rir.rename(hidden)
    try:
        assert audit_m6_variant_release(release_root) == first
        assert not audit_m6_variant_release(release_root, use_cache=False)["valid"]
    finally:
        hidden.rename(rir)

    # Anything the release declares by digest is inside the key, so tampering with
    # a recipe index misses the cache and the audit still catches it -- the
    # negative control the M6.4 validator relies on.
    index_path = next(release_root.glob("recipes/**/*.jsonl"))
    with index_path.open("ab") as handle:
        handle.write(b"cache-key-tamper")
    assert not audit_m6_variant_release(release_root)["valid"]


def test_release_bank_can_skip_the_audit(tmp_path):
    """`audit: False` loads an already-validated release without re-reading it."""
    report = validate_m6_variant_release.build_report(tmp_path / "fixture")
    release_root = Path(report["release"]["root"])

    bank = PreGeneratedReleaseBank(
        release_root,
        recipe_id="synthetic_calibrated",
        split="train",
        audit=False,
    )
    assert len(bank) > 0

    augmentor = AudioEffectAugmentor()
    with pytest.raises(ValueError, match="release-only options"):
        augmentor.init_room_bank(
            {"used": True, "bank_type": "room", "folder": str(release_root), "audit": False}
        )
