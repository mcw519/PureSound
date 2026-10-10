"""Serialization contracts for the core RIR types.

``RoomSceneV2``, ``PathEventSet`` and the bank manifest all travel as JSON, and
the bank resume path hashes what it reads back. These tests hold the three
properties that path depends on: a fixed seed samples the same scene twice, a
round trip through JSON preserves the content hash, and the one known
asymmetry (integer coordinates widening to floats) is pinned rather than
discovered later.
"""

from __future__ import annotations

import json

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene
from puresound.audio.rir.bank.schema import (
    BankGeneratorProvenance,
    BankRendererProfile,
    BankSplitIndex,
    BankSplitPolicy,
    RIRBankItem,
    RIRBankManifest,
    canonical_json_bytes,
    canonical_json_sha256,
    task_plan_rows,
)
from puresound.audio.rir.path_events import PathEventSet, generate_shoebox_path_events
from puresound.audio.rir.scene.schema import RoomSceneV2, SCENE_SCHEMA_VERSION


SCENE_SEED = 7


def build_scene() -> RoomSceneV2:
    """A fixed reference scene."""

    config = HybridRIRConfig(sample_rate=16000, duration=0.2)
    return sample_material_first_rir_scene(
        config,
        seed=SCENE_SEED,
        room_type="office",
        scene_id="reference-scene",
    )


def build_path_event_set() -> PathEventSet:
    """A fixed order-2 shoebox path-event set."""

    return generate_shoebox_path_events(
        dimensions_m=(4.2, 5.1, 2.7),
        source_position_m=(1.1, 1.7, 1.4),
        receiver_position_m=(3.0, 3.6, 1.2),
        sound_speed_m_s=343.0,
        scene_id="reference-room",
        max_order=2,
    )


def build_manifest() -> RIRBankManifest:
    """A self-contained three-split manifest with hand-fixed identities.

    Every hash in this fixture is a literal, so the digest depends only on the
    schema and serialization code, not on any file on disk.
    """

    policy = BankSplitPolicy(
        seed=11,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    profile = BankRendererProfile(
        profile_id="reference-profile",
        renderer_id="puresound.hybrid_rir",
        renderer_version="reference",
        low_backend="analytic",
        high_backend="path-events-m4",
        scene_schema_version=SCENE_SCHEMA_VERSION,
        renderer_config_sha256="11" * 32,
        evidence_tier="development",
    )
    items = tuple(
        RIRBankItem(
            item_id=f"reference_{index:03d}",
            room_id=f"room_{index:03d}",
            acoustic_space_id=f"synthetic:reference-{split}",
            scene_id=f"scene_{index:03d}",
            split=split,
            generation_seed=1000 + index,
            origin="synthetic",
            renderer_profile_id=profile.profile_id,
            signal_variant="physical",
            level_policy="calibrated",
            rir_path=f"room_{index:03d}/room_{index:03d}.wav",
            metadata_path=f"room_{index:03d}/room_{index:03d}.json",
            rir_sha256=f"{index:02d}" * 32,
            metadata_sha256=f"{index + 10:02d}" * 32,
            scene_sha256=f"{index + 20:02d}" * 32,
            sample_rate=16000,
            channel_count=5,
            frame_count=3200,
        )
        for index, split in enumerate(("train", "validation", "test"))
    )
    generation_config = {
        "fixture": "reference",
        "sample_rate": 16000,
        "frame_count": 3200,
        "split_policy": policy.to_dict(),
    }
    generator = BankGeneratorProvenance(
        generator_id="test.test_rir_reference_fixtures",
        generator_version="1",
        code_revision="reference-fixture",
        config_sha256=canonical_json_sha256(generation_config),
        task_plan_sha256=canonical_json_sha256(task_plan_rows(items)),
        seed=11,
    )
    split_indexes = {
        split: BankSplitIndex(
            path=f"indexes/{split}.jsonl",
            sha256=f"{index + 30:02d}" * 32,
            item_count=1,
        )
        for index, split in enumerate(("train", "validation", "test"))
    }
    return RIRBankManifest(
        bank_id="puresound-reference",
        release_status="draft",
        split_policy=policy,
        generator=generator,
        renderer_profiles=(profile,),
        items=items,
        split_indexes=split_indexes,
    ).with_content_sha256()


def test_scene_sampling_is_deterministic_and_its_metadata_hash_survives_json():
    """The form the bank resume path hashes must be stable through a file.

    ``RoomSceneV2.from_dict(...).to_dict()`` widens integer-valued coordinates
    to floats, so ``[0, 0, 0]`` becomes ``[0.0, 0.0, 0.0]``: numerically equal,
    not byte-stable. The resume path is unaffected because it hashes the dict
    loaded straight from JSON, but code that rebuilds a scene object and
    re-hashes it gets a different ``scene_sha256``. The assertion pins that
    asymmetry so a change to it is deliberate; if the round trip becomes
    byte-stable, update it and check nothing depended on the old digest.
    """

    scene = build_scene()
    assert canonical_json_sha256(scene.to_dict()) == canonical_json_sha256(
        build_scene().to_dict()
    )
    metadata = scene.to_metadata()
    assert canonical_json_sha256(json.loads(json.dumps(metadata))) == (
        canonical_json_sha256(metadata)
    )

    original = scene.to_dict()
    rebuilt = RoomSceneV2.from_dict(original).to_dict()
    assert rebuilt == original
    assert canonical_json_bytes(rebuilt) != canonical_json_bytes(original)


def test_path_event_generation_is_deterministic_and_round_trips_byte_stable():
    events = build_path_event_set()
    assert canonical_json_sha256(events.to_dict()) == canonical_json_sha256(
        build_path_event_set().to_dict()
    )
    rebuilt = PathEventSet.from_dict(events.to_dict())
    assert canonical_json_bytes(rebuilt.to_dict()) == canonical_json_bytes(
        events.to_dict()
    )


def test_manifest_hash_is_deterministic_self_consistent_and_survives_json():
    manifest = build_manifest()
    assert manifest.manifest_sha256 == build_manifest().manifest_sha256
    assert manifest.manifest_sha256 == manifest.content_sha256()
    rebuilt = RIRBankManifest.from_json(manifest.to_json())
    assert rebuilt.manifest_sha256 == manifest.manifest_sha256
    assert rebuilt.content_sha256() == manifest.content_sha256()

    spaces = [f"synthetic:space-{index:04d}" for index in range(64)]
    assert [BankSplitPolicy(seed=1).assign(space) for space in spaces] != [
        BankSplitPolicy(seed=2).assign(space) for space in spaces
    ]
