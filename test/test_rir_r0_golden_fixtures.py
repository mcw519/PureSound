"""R0 golden fixtures: pin the serialized form of the core RIR contracts.

``RIR_EXP_LOG.md`` §8 item 2 requires contract fixtures for
``RoomSceneV2``, ``PathEventSet`` and the M6 manifest before any code moves.
§5 then requires that after each migration stage the same seed still produces
the same scene, path events and metadata.

Each fixture is stored as canonical JSON under ``test/fixtures/rir_r0/`` *and*
pinned by a SHA-256 constant in this file.  The stored file makes a change
reviewable as a diff; the constant makes it impossible to regenerate the file
and quietly move the goalposts in the same commit without the change showing
up in the source too.

Regenerate deliberately with::

    PYTHONPATH=. python test/test_rir_r0_golden_fixtures.py --regenerate

and expect to justify every hash that moves.
"""

from __future__ import annotations

import json
import pathlib

import pytest

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


FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "rir_r0"

SCENE_SEED = 20260802
SCENE_FIXTURE = FIXTURE_DIR / "room_scene_v2.json"
PATH_EVENT_FIXTURE = FIXTURE_DIR / "path_event_set.json"
MANIFEST_FIXTURE = FIXTURE_DIR / "bank_manifest.json"

#: Frozen canonical-JSON digests.  A migration stage that changes one of these
#: has changed observable behaviour, not just file layout.
SCENE_SHA256 = "d9512794380eec0a86b72cfcb284c2348bbd659927c53e334902f5c539ab2be6"
PATH_EVENT_SHA256 = (
    "772a16da13cf2c16538c48516122917b3ac908b72c3f55b960e28091c83e722c"
)


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


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestSceneFixture:
    def test_sampling_is_deterministic(self):
        assert canonical_json_sha256(build_scene().to_dict()) == canonical_json_sha256(
            build_scene().to_dict()
        )

    def test_matches_frozen_digest(self):
        assert canonical_json_sha256(build_scene().to_dict()) == SCENE_SHA256

    def test_matches_stored_fixture(self):
        assert _load(SCENE_FIXTURE) == build_scene().to_dict()

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

    def test_matches_frozen_digest(self):
        assert canonical_json_sha256(build_path_event_set().to_dict()) == (
            PATH_EVENT_SHA256
        )

    def test_matches_stored_fixture(self):
        assert _load(PATH_EVENT_FIXTURE) == build_path_event_set().to_dict()

    def test_object_round_trip_is_byte_stable(self):
        events = build_path_event_set()
        rebuilt = PathEventSet.from_dict(events.to_dict())
        assert canonical_json_bytes(rebuilt.to_dict()) == canonical_json_bytes(
            events.to_dict()
        )

    def test_order_two_shoebox_has_the_expected_path_count(self):
        assert len(build_path_event_set().events) == 25


class TestManifestFixture:
    def test_construction_is_deterministic(self):
        assert build_manifest().manifest_sha256 == build_manifest().manifest_sha256

    def test_matches_stored_fixture(self):
        assert _load(MANIFEST_FIXTURE) == build_manifest().to_dict()

    def test_content_hash_is_self_consistent(self):
        manifest = build_manifest()
        assert manifest.manifest_sha256 == manifest.content_sha256()

    def test_json_round_trip_preserves_the_content_hash(self):
        manifest = build_manifest()
        rebuilt = RIRBankManifest.from_json(manifest.to_json())
        assert rebuilt.manifest_sha256 == manifest.manifest_sha256
        assert rebuilt.content_sha256() == manifest.content_sha256()

    def test_split_assignment_is_deterministic_for_a_fixed_policy(self):
        policy = BankSplitPolicy(seed=20260803)
        spaces = [f"synthetic:r0-{index:04d}" for index in range(64)]
        first = [policy.assign(space) for space in spaces]
        second = [BankSplitPolicy(seed=20260803).assign(space) for space in spaces]
        assert first == second
        assert set(first) <= {"train", "validation", "test"}

    def test_split_assignment_depends_on_the_seed(self):
        spaces = [f"synthetic:r0-{index:04d}" for index in range(64)]
        a = [BankSplitPolicy(seed=1).assign(space) for space in spaces]
        b = [BankSplitPolicy(seed=2).assign(space) for space in spaces]
        assert a != b


@pytest.mark.parametrize(
    "path", [SCENE_FIXTURE, PATH_EVENT_FIXTURE, MANIFEST_FIXTURE]
)
def test_fixture_files_are_canonical_json(path):
    """Stored fixtures must already be in canonical form, byte for byte."""

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw[:-1] == canonical_json_bytes(json.loads(raw))


def _regenerate() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (SCENE_FIXTURE, build_scene().to_dict()),
        (PATH_EVENT_FIXTURE, build_path_event_set().to_dict()),
        (MANIFEST_FIXTURE, build_manifest().to_dict()),
    ):
        path.write_bytes(canonical_json_bytes(payload) + b"\n")
        print(f"wrote {path.name}  sha256={canonical_json_sha256(payload)}")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
