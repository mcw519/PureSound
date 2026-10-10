import shutil
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


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A validated variant release, built once and treated as read-only."""
    report = validate_m6_variant_release.build_report(
        tmp_path_factory.mktemp("release") / "fixture"
    )
    return {
        "root": Path(report["release"]["root"]),
        "sha256": report["release"]["release_sha256"],
    }


def test_augmentor_consumes_a_release_recipe_without_split_leakage(release):
    augmentor = AudioEffectAugmentor()
    augmentor.init_room_bank(
        {
            "used": True,
            "bank_type": "release",
            "folder": str(release["root"]),
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
    assert metadata["release_sha256"] == release["sha256"]

    # `audit: False` loads an already-validated release without re-reading it.
    unaudited = PreGeneratedReleaseBank(
        release["root"],
        recipe_id="synthetic_calibrated",
        split="train",
        audit=False,
    )
    assert len(unaudited) > 0


def test_union_can_add_a_release_recipe_beside_a_folder_bank(release):
    """Widening a folder-bank recipe with a release keeps both provenances.

    This is the bank-expansion shape: the existing pool stays exactly what it
    was and the release rides alongside it at a fixed weight, with usage_role
    injected once at the top level the way the dataset does it.
    """
    folder_member = release["root"] / "variants" / "calibrated"

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
                    "name": "release",
                    "weight": 0.4,
                    "bank_type": "release",
                    "folder": str(release["root"]),
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
    assert set(by_member) == {"folder", "release"}
    # Release identity travels only with the release member's draws.
    assert all("release_sha256" not in m for m in by_member["folder"])
    assert all(m["release_sha256"] == release["sha256"] for m in by_member["release"])


@pytest.mark.parametrize(
    "config,match",
    [
        (
            {"bank_type": "release", "split": "train", "usage_role": "train"},
            "requires recipe_id",
        ),
        (
            {"bank_type": "release", "recipe_id": "synthetic_calibrated", "usage_role": "train"},
            "explicit split",
        ),
        (
            {"bank_type": "release", "recipe_id": "synthetic_calibrated", "split": "test"},
            "usage_role",
        ),
        (
            {
                "bank_type": "release",
                "recipe_id": "synthetic_calibrated",
                "split": "test",
                "usage_role": "train",
            },
            "must match",
        ),
        ({"bank_type": "room", "recipe_id": "synthetic_calibrated"}, "release-only options"),
        ({"used": True, "bank_type": "room", "audit": False}, "release-only options"),
    ],
    ids=[
        "no_recipe",
        "no_split",
        "no_usage_role",
        "split_role_mismatch",
        "room_with_recipe",
        "room_with_audit",
    ],
)
def test_bank_config_requires_recipe_split_and_role_and_keeps_options_in_their_bank_type(
    tmp_path, config, match
):
    with pytest.raises(ValueError, match=match):
        AudioEffectAugmentor().init_room_bank({"folder": str(tmp_path), **config})


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


def test_release_audit_is_memoized_and_skippable(release, tmp_path):
    """A full audit reads the whole release; a training start must not repeat it."""
    from puresound.audio.rir.bank.release import (
        AUDIT_CACHE_NAME,
        audit_m6_variant_release,
    )

    # Building the fixture already audited the release, which memoizes the verdict.
    assert (release["root"] / AUDIT_CACHE_NAME).is_file()
    # The rest mutates the release, so it works on a private copy.
    release_root = tmp_path / "release"
    shutil.copytree(release["root"], release_root)
    cache_path = release_root / AUDIT_CACHE_NAME
    cache_path.unlink()

    first = audit_m6_variant_release(release_root)
    assert first["valid"]
    assert cache_path.is_file()

    # A hit skips re-reading the bank's audio: hide one RIR, which only the cached
    # verdict can survive. This is exactly the scope the cache trusts.
    rir = next(release_root.glob("variants/*/**/*.wav"))
    rir.rename(rir.with_suffix(".hidden"))
    assert audit_m6_variant_release(release_root) == first
    assert not audit_m6_variant_release(release_root, use_cache=False)["valid"]
    rir.with_suffix(".hidden").rename(rir)

    # Anything the release declares by digest is inside the key, so tampering with
    # a recipe index misses the cache and the audit still catches it.
    index_path = next(release_root.glob("recipes/**/*.jsonl"))
    with index_path.open("ab") as handle:
        handle.write(b"cache-key-tamper")
    assert not audit_m6_variant_release(release_root)["valid"]
