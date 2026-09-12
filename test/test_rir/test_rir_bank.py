import json

import numpy as np
import pytest
import torch
import torchaudio

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_bank_contract import (
    build_fixture,
)
from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.rir.bank.loader import PreGeneratedRoomBank


def _write_room(root, room_id, distances, sr=16000, length=2400):
    """Write one room item: a 5-channel RIR WAV + metadata.json.

    Channel i gets a unit impulse at the sample matching ``distances[i]`` so the
    direct path (and therefore the recovered distance/DRR ordering) is exact.
    """
    sample_dir = root / room_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels = ["near_0", "near_1", "far_0", "far_1", "far_2"]
    rir = torch.zeros(5, length)
    for ch, dist in enumerate(distances):
        peak = int(round(dist / 343.0 * sr))
        rir[ch, peak] = 1.0
        # Tail placed beyond the 2.5 ms direct window so DRR is finite.
        rir[ch, peak + 100] = 0.3
    torchaudio.save(str(sample_dir / "rir_5ch.wav"), rir, sr, encoding="PCM_F")
    channel_map = [
        {"channel": ch, "label": labels[ch], "distance_m": float(dist)}
        for ch, dist in enumerate(distances)
    ]
    metadata = {"scene": {"rt60": 0.45, "channel_map": channel_map}}
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_named_rir(root, room_id, rir_id, distances, sr=16000, length=2400):
    sample_dir = root / room_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels = ["near_0", "near_1", "far_0", "far_1", "far_2"]
    rir = torch.zeros(5, length)
    for ch, dist in enumerate(distances):
        peak = int(round(dist / 343.0 * sr))
        rir[ch, peak] = 1.0
    torchaudio.save(str(sample_dir / f"{rir_id}.wav"), rir, sr, encoding="PCM_F")
    channel_map = [
        {"channel": ch, "label": labels[ch], "distance_m": float(dist)}
        for ch, dist in enumerate(distances)
    ]
    metadata = {"scene": {"rt60": 0.45, "channel_map": channel_map}}
    (sample_dir / f"{rir_id}.json").write_text(json.dumps(metadata), encoding="utf-8")


def _make_bank_folder(tmp_path):
    root = tmp_path / "hybrid_rir_db"
    _write_room(root, "room_000000", [0.5, 0.8, 2.5, 3.5, 4.5])
    _write_room(root, "room_000001", [0.4, 0.9, 2.2, 4.0, 5.2])
    return root


def test_bank_indexes_rooms_and_splits_near_far(tmp_path):
    root = _make_bank_folder(tmp_path)
    bank = PreGeneratedRoomBank(str(root), wav_name="rir_5ch.wav")

    assert len(bank) == 2
    scene = bank.sample_scene()
    assert {c["label"] for c in scene["near"]} == {"near_0", "near_1"}
    assert {c["label"] for c in scene["far"]} == {"far_0", "far_1", "far_2"}


def test_bank_indexes_multiple_named_rirs_per_room_folder(tmp_path):
    root = tmp_path / "hybrid_rir_db"
    _write_named_rir(
        root, "room_000000", "room_000000_000000", [0.5, 0.8, 2.5, 3.5, 4.5]
    )
    _write_named_rir(
        root, "room_000000", "room_000000_000001", [0.6, 0.9, 2.6, 3.6, 4.6]
    )

    bank = PreGeneratedRoomBank(str(root))

    assert len(bank) == 2
    assert {room["id"] for room in bank._rooms} == {
        "room_000000_000000",
        "room_000000_000001",
    }


def test_manifestless_m6_bank_refuses_legacy_split_mixing(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "rir_bank_manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest is missing"):
        PreGeneratedRoomBank(str(tmp_path))


def test_foreground_uses_near_interferer_uses_far(tmp_path):
    root = _make_bank_folder(tmp_path)
    bank = PreGeneratedRoomBank(str(root))
    scene = bank.sample_scene()

    fg_imp, fg_meta, sr = bank.select_channel(scene, source_role="foreground")
    itf_imp, itf_meta, _ = bank.select_channel(scene, source_role="interferer")

    assert fg_meta["source_receiver_distance"] < 1.0
    assert itf_meta["source_receiver_distance"] > 2.0
    assert fg_meta["rt60"] == 0.45
    # DRR must be a finite number derived from the channel, not a placeholder.
    assert np.isfinite(fg_meta["drr_db"])
    assert fg_imp.shape[0] == 1 and itf_imp.shape[0] == 1


def test_distance_range_override_filters_channels(tmp_path):
    root = _make_bank_folder(tmp_path)
    bank = PreGeneratedRoomBank(str(root))
    scene = bank.sample_scene()
    # Restrict interferers to the far end; the nearest far channel (~2.2-2.5)
    # must be excluded.
    _, meta, _ = bank.select_channel(
        scene, source_role="interferer", distance_range_override=[4.0, 6.0]
    )
    assert meta["source_receiver_distance"] >= 4.0


def test_augmentor_bank_path_reuses_channel_for_target_rir_type(tmp_path):
    root = _make_bank_folder(tmp_path)
    aug = AudioEffectAugmentor()
    aug.init_room_bank({"used": True, "folder": str(root)})

    scene = aug.sample_room_scene()
    assert scene.get("_bank") is True

    wav = torch.randn(1, 16000)
    reverb, (rir_id, info) = aug.apply_rir(
        wav=wav, rir_mode="full", sr=16000, room_scene=scene, source_role="foreground"
    )
    assert rir_id.startswith("bank-")
    assert info["metadata"]["source_receiver_distance"] < 1.0

    # The clean-target second call reuses the cached channel via rir_id.
    clean, (rir_id2, _) = aug.apply_rir(
        wav=wav, rir_id=rir_id, rir_mode="direct", sr=16000
    )
    assert rir_id2 == rir_id
    assert reverb.shape[-1] >= wav.shape[-1]
    assert clean.shape[-1] >= 1


def test_union_bank_serves_both_members_and_routes_scenes_home(tmp_path):
    """A union widens the pool without letting one member answer for another."""
    hybrid = tmp_path / "hybrid"
    _write_room(hybrid, "room_000000", [0.5, 0.8, 2.5, 3.5, 4.5])
    added = tmp_path / "added"
    _write_room(added, "room_000100", [0.4, 0.9, 2.2, 4.0, 5.2])

    aug = AudioEffectAugmentor()
    aug.init_room_bank(
        {
            "used": True,
            "banks": [
                {"name": "hybrid", "weight": 0.7, "folder": str(hybrid)},
                {"name": "added", "weight": 0.3, "folder": str(added)},
            ],
        }
    )
    assert aug.room_bank_kind == "union"
    assert len(aug.room_bank) == 2
    assert aug.room_bank.weights == pytest.approx([0.7, 0.3])

    seen = set()
    for _ in range(200):
        scene = aug.room_bank.sample_scene()
        seen.add(scene["union_member_name"])
        # Every draw must resolve against the bank that produced it.
        _impulse, metadata, _sr = aug.room_bank.select_channel(
            scene, source_role="foreground"
        )
        assert metadata["union_member_name"] == scene["union_member_name"]
        assert metadata["source_receiver_distance"] < 1.0
    assert seen == {"hybrid", "added"}

    # A scene the union did not hand out cannot be redeemed against it.
    stray = PreGeneratedRoomBank(str(hybrid)).sample_scene()
    with pytest.raises(ValueError, match="does not belong"):
        aug.room_bank.select_channel(stray)


def test_union_bank_rejects_bad_weights_and_stray_top_level_options(tmp_path):
    hybrid = tmp_path / "hybrid"
    _write_room(hybrid, "room_000000", [0.5, 0.8, 2.5, 3.5, 4.5])
    aug = AudioEffectAugmentor()

    with pytest.raises(ValueError, match="weights must be positive"):
        aug.init_room_bank(
            {"used": True, "banks": [{"folder": str(hybrid), "weight": 0}]}
        )
    # A folder left at the top level would silently apply to no member.
    with pytest.raises(ValueError, match="per-bank options at the top level"):
        aug.init_room_bank(
            {"used": True, "folder": str(hybrid), "banks": [{"folder": str(hybrid)}]}
        )
    with pytest.raises(ValueError, match="non-empty list"):
        aug.init_room_bank({"used": True, "banks": []})


def test_simulated_rir_cache_is_bounded(tmp_path):
    root = _make_bank_folder(tmp_path)
    aug = AudioEffectAugmentor()
    aug.simulated_rir_cache_size = 2
    aug.init_room_bank({"used": True, "folder": str(root)})
    wav = torch.randn(1, 800)

    ids = []
    for _ in range(4):
        _reverb, (rir_id, _info) = aug.apply_rir(
            wav=wav,
            rir_mode="full",
            sr=16000,
        )
        ids.append(rir_id)

    assert len(aug.simulated_rir) == 2
    assert ids[0] not in aug.simulated_rir
    assert ids[-1] in aug.simulated_rir
