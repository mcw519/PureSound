import types
from collections import Counter

import pytest

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_bank_contract import (
    build_fixture,
)
from puresound.audio.rir.bank import listening
from puresound.audio.rir.bank.evidence import (
    RENDERER_APPROVAL_KIND,
    RendererApproval,
    approve_bank_renderer_profiles,
    build_evidence_bundle,
)
from puresound.audio.rir.bank.listening import (
    TRIAL_ROLES,
    ListeningProtocol,
    build_listening_assignment,
)
from puresound.audio.rir.bank.production import REQUIRED_EVIDENCE_KINDS
from puresound.audio.rir.bank.release import build_m6_variant_release


@pytest.fixture
def release_root(tmp_path, monkeypatch):
    """A release stand-in: the assignment only reads recipe rows and the release hash."""
    rows = [
        {"release_item_id": f"variant:{index}", "acoustic_space_id": f"space{index % 7}"}
        for index in range(40)
    ]
    monkeypatch.setattr(listening, "_recipe_rows", lambda root, recipe_id, split: rows)
    monkeypatch.setattr(
        listening.RIRBankReleaseManifest,
        "from_json",
        classmethod(lambda cls, text: types.SimpleNamespace(release_sha256="a" * 64)),
    )
    (tmp_path / "rir_bank_release.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("trials", [3, 5, 9, 10, 25])
def test_every_session_hides_a_reference_and_an_anchor_at_unpredictable_positions(
    release_root, trials
):
    """The protocol claims randomisation and both validity trials in every session."""
    protocol = ListeningProtocol(
        participant_count=20, trials_per_participant=trials, noninferiority_margin=0.5
    )
    records = build_listening_assignment(release_root, protocol)["assignment_records"]

    positions: dict[str, set[int]] = {role: set() for role in TRIAL_ROLES}
    for participant in {row["participant_id"] for row in records}:
        session = [row for row in records if row["participant_id"] == participant]
        assert len(session) == trials
        counts = Counter(row["role"] for row in session)
        assert all(counts[role] >= 1 for role in TRIAL_ROLES), counts
        for row in session:
            positions[row["role"]].add(row["presentation_index"])

    # A role pinned to one presentation index across participants gives it away.
    assert len(positions["hidden_reference"]) > 1
    assert len(positions["degraded_anchor"]) > 1


@pytest.mark.parametrize("trials", [1, 2])
def test_protocol_needs_room_for_a_comparison_a_reference_and_an_anchor(trials):
    with pytest.raises(ValueError, match="trials_per_participant"):
        ListeningProtocol(
            participant_count=20,
            trials_per_participant=trials,
            noninferiority_margin=0.5,
        )


def test_a_rejecting_record_stops_the_approval_before_anything_is_written(tmp_path):
    bank = tmp_path / "bank"
    manifest = build_fixture(bank)
    manifest_path = bank / "rir_bank_manifest.json"
    before = manifest_path.read_bytes()
    approval = RendererApproval(
        profile_id=manifest.renderer_profiles[0].profile_id,
        approver_id="someone",
        reviewed_scope="scope",
        evidence_sha256={"evidence": "c" * 64},
        decision="reject",
    )

    with pytest.raises(ValueError, match="missing or rejected"):
        approve_bank_renderer_profiles(
            bank, [approval], approval_dir=tmp_path / "approvals"
        )

    assert manifest_path.read_bytes() == before
    assert not (tmp_path / "approvals").exists()


def test_evidence_bundle_needs_a_file_for_every_required_kind(tmp_path):
    kinds = (*REQUIRED_EVIDENCE_KINDS, RENDERER_APPROVAL_KIND)
    artifacts = {}
    for kind in kinds:
        (tmp_path / f"{kind}.json").write_text("{}", encoding="utf-8")
        artifacts[kind] = [f"{kind}.json"]
    bundle = build_evidence_bundle(
        tmp_path, artifacts, release_sha256="a" * 64, evaluation_sha256="b" * 64
    )
    assert {row["kind"] for row in bundle["artifacts"]} == set(kinds)

    for kind in kinds:
        with pytest.raises(ValueError, match="evidence bundle"):
            build_evidence_bundle(
                tmp_path,
                {**artifacts, kind: []},
                release_sha256="a" * 64,
                evaluation_sha256="b" * 64,
            )


@pytest.mark.parametrize(
    "weights",
    [
        {"synthetic": 2.0, "real": 1.0},
        {"synthetic": 0.4, "real": 0.4},
        {"synthetic": float("nan"), "real": 1.0},
    ],
    ids=["above_one", "below_one", "not_finite"],
)
def test_release_refuses_unusable_mixed_weights_before_building_anything(
    tmp_path, weights
):
    """The recipe constructor rejects these, but only after the banks are copied."""
    output = tmp_path / "release"

    with pytest.raises(ValueError, match="mixed_origin_weights"):
        build_m6_variant_release(
            tmp_path / "source",
            output,
            measured_bank_root=tmp_path / "measured",
            mixed_origin_weights=weights,
        )

    assert not output.exists()
