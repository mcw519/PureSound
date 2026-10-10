from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_bank_contract import build_fixture
from puresound.audio.rir.bank.loader import PreGeneratedRoomBank
from puresound.audio.rir.bank.schema import (
    BANK_SPLITS,
    BankSplitPolicy,
    RIRBankManifest,
    audit_rir_bank_manifest,
    canonicalize_float_wav_header,
    sha256_file,
)


def test_split_policy_and_manifest_digest_are_deterministic(tmp_path):
    policy = BankSplitPolicy(
        seed=37,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    identities = [f"room-{index}" for index in range(100)]

    assert [policy.assign(value) for value in identities] == [
        policy.assign(value) for value in identities
    ]
    assert set(policy.assign(value) for value in identities) == set(BANK_SPLITS)

    manifest = build_fixture(tmp_path)
    round_trip = RIRBankManifest.from_json(manifest.to_json())
    assert round_trip.content_sha256() == manifest.manifest_sha256
    assert round_trip.to_dict() == manifest.to_dict()


def test_bank_audit_fails_closed_on_tamper_and_unbacked_claims(tmp_path):
    manifest = build_fixture(tmp_path)
    valid = audit_rir_bank_manifest(manifest, tmp_path)
    assert valid["ready_for_m6_bank_generation"] is True

    rir_path = tmp_path / manifest.items[0].rir_path
    with rir_path.open("ab") as handle:
        handle.write(b"tamper")
    tampered = audit_rir_bank_manifest(manifest, tmp_path, inspect_audio=False)

    assert tampered["checks"]["all_asset_hashes_match"] is False
    assert tampered["ready_for_m6_bank_generation"] is False

    false_production = replace(
        manifest,
        release_status="production",
        manifest_sha256=None,
    ).with_content_sha256()
    audit = audit_rir_bank_manifest(false_production, tmp_path)

    assert audit["checks"]["production_claim_is_evidence_backed"] is False
    assert audit["ready_for_production"] is False
    with pytest.raises(ValueError, match="safe bank-relative path"):
        replace(manifest.items[0], metadata_path="../outside.json")


def test_bank_reader_requires_a_split_and_propagates_provenance(tmp_path):
    manifest = build_fixture(tmp_path)

    with pytest.raises(ValueError, match="explicit split"):
        PreGeneratedRoomBank(str(tmp_path))
    train = PreGeneratedRoomBank(str(tmp_path), split="train")
    scene = train.sample_scene()
    _rir, metadata, _sample_rate = train.select_channel(scene)

    assert len(train) == sum(item.split == "train" for item in manifest.items)
    assert metadata["split"] == "train"
    assert metadata["physical_room_id"] == "m6_train_room"
    assert metadata["acoustic_space_id"].startswith("m6-fixture-acoustic-space")
    assert metadata["renderer_profile_id"] == "m6-fixture-path-events-m4"


def _set_peak_timestamp(path, timestamp):
    payload = bytearray(path.read_bytes())
    offset = payload.index(b"PEAK")
    payload[offset + 12 : offset + 16] = int(timestamp).to_bytes(4, "little")
    path.write_bytes(payload)


def test_float_wav_peak_timestamp_is_canonicalized(tmp_path):
    samples = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, samples, 16000, subtype="FLOAT")
    sf.write(second, samples, 16000, subtype="FLOAT")
    _set_peak_timestamp(first, 1)
    _set_peak_timestamp(second, 4_000_000_000)

    assert sha256_file(first) != sha256_file(second)
    canonicalize_float_wav_header(first)
    canonicalize_float_wav_header(second)

    assert sha256_file(first) == sha256_file(second)
    first_audio, first_sr = sf.read(first, dtype="float32")
    second_audio, second_sr = sf.read(second, dtype="float32")
    assert first_sr == second_sr == 16000
    np.testing.assert_array_equal(first_audio, second_audio)
