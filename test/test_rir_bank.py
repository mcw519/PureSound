import json

import numpy as np
import torch
import torchaudio

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.rir_bank import PreGeneratedRoomBank


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
