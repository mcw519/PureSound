"""Serialization contracts for the core RIR types.

``RoomSceneV2``, ``PathEventSet`` and the M6 bank manifest all travel as JSON,
and the M6 resume path hashes what it reads back. These tests hold the three
properties that path depends on: a fixed seed samples the same scene twice, a
round trip through JSON preserves the content hash, and the one known
asymmetry (integer coordinates widening to floats) is documented rather than
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


SCENE_SEED = 20260802


def build_scene() -> RoomSceneV2:
    """The frozen R0 reference scene."""

    config = HybridRIRConfig(sample_rate=16000, duration=0.2)
    return sample_material_first_rir_scene(
        config,
        seed=SCENE_SEED,
        room_type="office",
        scene_id="r0-golden-scene",
    )


def build_path_event_set() -> PathEventSet:
    """The frozen R0 reference order-2 shoebox path-event set."""

    return generate_shoebox_path_events(
        dimensions_m=(4.2, 5.1, 2.7),
        source_position_m=(1.1, 1.7, 1.4),
        receiver_position_m=(3.0, 3.6, 1.2),
        sound_speed_m_s=343.0,
        scene_id="r0-golden",
        max_order=2,
    )


def build_manifest() -> RIRBankManifest:
    """A self-contained three-split manifest with hand-fixed identities.

    Every hash in this fixture is a literal, so the digest depends only on the
    schema and serialization code — not on any file on disk.  That is what R6
    must preserve when the bank package is split apart.
    """

    policy = BankSplitPolicy(
        seed=20260803,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    profile = BankRendererProfile(
        profile_id="r0-golden-profile",
        renderer_id="puresound.hybrid_rir",
        renderer_version="R0-golden",
        low_backend="analytic",
        high_backend="path-events-m4",
        scene_schema_version=SCENE_SCHEMA_VERSION,
        renderer_config_sha256="11" * 32,
        evidence_tier="development",
    )
    items = tuple(
        RIRBankItem(
            item_id=f"r0_golden_{index:03d}",
            room_id=f"room_{index:03d}",
            acoustic_space_id=f"synthetic:r0-golden-{split}",
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
        "fixture": "R0-golden",
        "sample_rate": 16000,
        "frame_count": 3200,
        "split_policy": policy.to_dict(),
    }
    generator = BankGeneratorProvenance(
        generator_id="test.test_rir_r0_golden_fixtures",
        generator_version="R0",
        code_revision="r0-golden-fixture",
        config_sha256=canonical_json_sha256(generation_config),
        task_plan_sha256=canonical_json_sha256(task_plan_rows(items)),
        seed=20260803,
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
        bank_id="puresound-r0-golden",
        release_status="draft",
        split_policy=policy,
        generator=generator,
        renderer_profiles=(profile,),
        items=items,
        split_indexes=split_indexes,
    ).with_content_sha256()


class TestSceneFixture:
    def test_sampling_is_deterministic(self):
        assert canonical_json_sha256(build_scene().to_dict()) == canonical_json_sha256(
            build_scene().to_dict()
        )

    def test_metadata_survives_json_text_round_trip(self):
        """The form M6 actually hashes must be stable through a file."""

        metadata = build_scene().to_metadata()
        reloaded = json.loads(json.dumps(metadata))
        assert canonical_json_sha256(reloaded) == canonical_json_sha256(metadata)

    def test_object_round_trip_is_numerically_equal_but_not_byte_stable(self):
        """Pin a known asymmetry so a migration cannot change it silently.

        ``RoomSceneV2.from_dict(...).to_dict()`` widens integer-valued
        coordinates to floats, so ``[0, 0, 0]`` becomes ``[0.0, 0.0, 0.0]``.
        The values are equal and the M6 resume path is unaffected — it hashes
        the dict loaded straight from JSON, and JSON text preserves the
        distinction (see the test above).  But any future code that rebuilds a
        scene object and re-hashes it will get a different ``scene_sha256``.

        This test documents the current behaviour rather than endorsing it.
        If a later stage makes the round-trip byte-stable, that is an
        improvement — update this test deliberately.
        """

        original = build_scene().to_dict()
        rebuilt = RoomSceneV2.from_dict(original).to_dict()
        assert rebuilt == original, "round trip must stay numerically equal"
        assert canonical_json_bytes(rebuilt) != canonical_json_bytes(original), (
            "the int->float widening documented here appears to be fixed; "
            "update this test and check nothing depended on the old digest"
        )


class TestPathEventFixture:
    def test_generation_is_deterministic(self):
        assert canonical_json_sha256(
            build_path_event_set().to_dict()
        ) == canonical_json_sha256(build_path_event_set().to_dict())

    def test_object_round_trip_is_byte_stable(self):
        events = build_path_event_set()
        rebuilt = PathEventSet.from_dict(events.to_dict())
        assert canonical_json_bytes(rebuilt.to_dict()) == canonical_json_bytes(
            events.to_dict()
        )

class TestManifestFixture:
    def test_construction_is_deterministic(self):
        assert build_manifest().manifest_sha256 == build_manifest().manifest_sha256

    def test_content_hash_is_self_consistent(self):
        manifest = build_manifest()
        assert manifest.manifest_sha256 == manifest.content_sha256()

    def test_json_round_trip_preserves_the_content_hash(self):
        manifest = build_manifest()
        rebuilt = RIRBankManifest.from_json(manifest.to_json())
        assert rebuilt.manifest_sha256 == manifest.manifest_sha256
        assert rebuilt.content_sha256() == manifest.content_sha256()

    def test_split_assignment_depends_on_the_seed(self):
        spaces = [f"synthetic:r0-{index:04d}" for index in range(64)]
        a = [BankSplitPolicy(seed=1).assign(space) for space in spaces]
        b = [BankSplitPolicy(seed=2).assign(space) for space in spaces]
        assert a != b

