import json
from pathlib import Path

import numpy as np
import pytest

from puresound.audio.rir.bank.evaluation import (
    EVIDENCE_TIERS,
    evaluate_m6_release,
    validate_listening_report,
    validate_throughput_report,
)
from puresound.audio.rir.bank.evidence import (
    RendererApproval,
    approve_bank_renderer_profiles,
    build_evidence_bundle,
    build_throughput_report,
    write_production_signoff,
    write_renderer_approval_record,
)
from puresound.audio.rir.bank.listening import (
    DRY_RUN_EVIDENCE_TIER,
    EMPIRICAL_EVIDENCE_TIER,
    MINIMUM_EMPIRICAL_PARTICIPANTS,
    ListeningProtocol,
    build_dry_run_report,
    build_listening_assignment,
    ingest_listening_responses,
)
from puresound.audio.rir.bank.production import (
    REQUIRED_EVIDENCE_KINDS,
    audit_m6_production_evidence,
)
from puresound.audio.rir.bank.schema import RIRBankManifest, sha256_file

AUDIT = {
    "bank_id": "test-bank",
    "root": "/somewhere",
    "generation_run": {
        "task_count": 10,
        "items_generated": 7,
        "items_skipped_as_complete": 3,
        "items_failed": 0,
        "num_workers": 4,
        "elapsed_seconds": 14.0,
        "resume_requested": True,
        "items_failed_policy": "any_item_failure_aborts_the_run_before_manifest_emit",
    },
}
RELEASE_SHA = "a" * 64
EVALUATION_SHA = "b" * 64


def test_throughput_report_satisfies_its_own_validator():
    report = build_throughput_report(AUDIT, release_sha256=RELEASE_SHA)
    validated = validate_throughput_report(report, release_sha256=RELEASE_SHA)

    assert validated["valid"] is True
    assert report["items_per_second"] == pytest.approx(7 / 14.0)


def test_throughput_report_refuses_an_untimed_or_unfailed_run():
    """A missing measurement must stay missing rather than become a default.

    The contract requires a positive duration and an explicit failure count; filling
    either in here would convert "nobody measured this" into a passing check.
    """
    untimed = {"generation_run": {**AUDIT["generation_run"], "elapsed_seconds": None}}
    unfailed = {"generation_run": {**AUDIT["generation_run"], "items_failed": None}}

    with pytest.raises(ValueError, match="elapsed_seconds"):
        build_throughput_report(untimed, release_sha256=RELEASE_SHA)
    with pytest.raises(ValueError, match="items_failed"):
        build_throughput_report(unfailed, release_sha256=RELEASE_SHA)
    with pytest.raises(ValueError, match="generation_run"):
        build_throughput_report({}, release_sha256=RELEASE_SHA)


def test_throughput_report_rejects_inconsistent_counts():
    broken = {"generation_run": {**AUDIT["generation_run"], "task_count": 99}}

    with pytest.raises(ValueError, match="inconsistent"):
        build_throughput_report(broken, release_sha256=RELEASE_SHA)


def test_renderer_approval_needs_a_person_and_stated_evidence():
    with pytest.raises(ValueError, match="approver_id"):
        RendererApproval(
            profile_id="p", approver_id="  ", reviewed_scope="scope",
            evidence_sha256={"e": "c" * 64},
        )
    with pytest.raises(ValueError, match="reviewed_scope"):
        RendererApproval(
            profile_id="p", approver_id="someone", reviewed_scope="",
            evidence_sha256={"e": "c" * 64},
        )
    with pytest.raises(ValueError, match="at least one evidence hash"):
        RendererApproval(
            profile_id="p", approver_id="someone", reviewed_scope="scope",
            evidence_sha256={},
        )
    with pytest.raises(ValueError, match="decision must be"):
        RendererApproval(
            profile_id="p", approver_id="someone", reviewed_scope="scope",
            evidence_sha256={"e": "c" * 64}, decision="looks-fine",
        )


def test_approval_record_hash_is_the_file_hash_the_bundle_will_check(tmp_path):
    """The profile must carry the file's hash, not the payload's content hash.

    The bundle audit verifies sha256_file(path) and then requires that value to be
    the profile's approval_report_sha256. A canonical-JSON content hash would differ
    once the file is indented and newline terminated, and the mismatch surfaces only
    at the final decision.
    """
    approval = RendererApproval(
        profile_id="prof", approver_id="someone", reviewed_scope="scope",
        evidence_sha256={"m6_5_evaluation": EVALUATION_SHA},
    )
    path, returned = write_renderer_approval_record(
        tmp_path / "approval.json", approval, profile={"renderer_id": "r"}
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert returned == sha256_file(path)
    assert returned != record["approval_sha256"]


def test_signoff_binds_to_one_release_and_one_evaluation(tmp_path):
    path, _sha = write_production_signoff(
        tmp_path / "acoustics.json",
        role="acoustics", reviewer_id="reviewer", reviewed_scope="scope",
        release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["release_sha256"] == RELEASE_SHA
    assert record["evaluation_sha256"] == EVALUATION_SHA
    assert record["decision"] == "approve"
    with pytest.raises(ValueError, match="reviewer_id"):
        write_production_signoff(
            tmp_path / "bad.json", role="ml", reviewer_id="",
            reviewed_scope="scope", release_sha256=RELEASE_SHA,
            evaluation_sha256=EVALUATION_SHA,
        )
    with pytest.raises(ValueError, match="role must be"):
        write_production_signoff(
            tmp_path / "bad.json", role="whoever", reviewer_id="reviewer",
            reviewed_scope="scope", release_sha256=RELEASE_SHA,
            evaluation_sha256=EVALUATION_SHA,
        )


def _stub_evidence(root: Path, kinds=REQUIRED_EVIDENCE_KINDS):
    artifacts: dict[str, list[str]] = {}
    for kind in kinds:
        name = f"{kind}.json"
        (root / name).write_text(json.dumps({"kind": kind}) + "\n", encoding="utf-8")
        artifacts[kind] = [name]
    (root / "renderer_approval.json").write_text("{}\n", encoding="utf-8")
    artifacts["renderer_approval"] = ["renderer_approval.json"]
    return artifacts


def test_bundle_refuses_to_describe_a_file_that_is_not_there(tmp_path):
    artifacts = _stub_evidence(tmp_path)
    artifacts["listening_analysis"] = ["listening/absent.json"]

    with pytest.raises(FileNotFoundError, match="listening_analysis"):
        build_evidence_bundle(
            tmp_path, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )


def test_bundle_names_the_required_kinds_it_is_missing(tmp_path):
    partial = [kind for kind in REQUIRED_EVIDENCE_KINDS if "downstream" not in kind]
    artifacts = _stub_evidence(tmp_path, partial)

    with pytest.raises(ValueError, match="downstream_training_recipe"):
        build_evidence_bundle(
            tmp_path, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )


def test_bundle_requires_a_renderer_approval_and_unique_paths(tmp_path):
    artifacts = _stub_evidence(tmp_path)
    del artifacts["renderer_approval"]
    with pytest.raises(ValueError, match="renderer_approval"):
        build_evidence_bundle(
            tmp_path, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )

    duplicated = _stub_evidence(tmp_path)
    duplicated["ml_signoff"] = ["acoustics_signoff.json"]
    with pytest.raises(ValueError, match="declared twice"):
        build_evidence_bundle(
            tmp_path, duplicated,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )


def test_bundle_escapes_are_rejected(tmp_path):
    artifacts = _stub_evidence(tmp_path)
    artifacts["release_signoff"] = ["../outside.json"]

    with pytest.raises(ValueError, match="relative path"):
        build_evidence_bundle(
            tmp_path, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )


def test_bundle_audit_accepts_a_bundle_this_module_produced(tmp_path):
    """Producer and validator have to agree, and only an audit can show it."""
    artifacts = _stub_evidence(tmp_path)
    approval_path, approval_sha = write_renderer_approval_record(
        tmp_path / "renderer_approval.json",
        RendererApproval(
            profile_id="prof", approver_id="someone", reviewed_scope="scope",
            evidence_sha256={"m6_5_evaluation": EVALUATION_SHA},
        ),
        profile={"renderer_id": "r"},
    )
    artifacts["renderer_approval"] = [
        str(approval_path.relative_to(tmp_path))
    ]
    for role, kind in (
        ("acoustics", "acoustics_signoff"),
        ("ml", "ml_signoff"),
        ("release", "release_signoff"),
    ):
        path, _sha = write_production_signoff(
            tmp_path / f"{kind}.json", role=role, reviewer_id="reviewer",
            reviewed_scope="scope", release_sha256=RELEASE_SHA,
            evaluation_sha256=EVALUATION_SHA,
        )
        artifacts[kind] = [str(path.relative_to(tmp_path))]
    listening_hashes = {}
    for kind, name in (
        ("listening_assignment", "assignment_sha256"),
        ("listening_responses", "responses_sha256"),
        ("listening_analysis", "analysis_sha256"),
    ):
        listening_hashes[name] = sha256_file(tmp_path / f"{kind}.json")
    downstream_hashes = {}
    for kind, name in (
        ("downstream_training_recipe", "training_recipe_sha256"),
        ("downstream_checkpoints", "checkpoints_sha256"),
        ("downstream_analysis", "analysis_sha256"),
    ):
        downstream_hashes[name] = sha256_file(tmp_path / f"{kind}.json")

    bundle = build_evidence_bundle(
        tmp_path, artifacts,
        release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
    )
    audit = audit_m6_production_evidence(
        bundle,
        evidence_root=tmp_path,
        release_sha256=RELEASE_SHA,
        evaluation={
            "evaluation_sha256": EVALUATION_SHA,
            "listening_evaluation": {"artifact_sha256": listening_hashes},
            "downstream_evaluation": {"artifact_sha256": downstream_hashes},
        },
        renderer_approval_sha256=[approval_sha],
    )

    assert audit["valid"] is True, [
        name for name, ok in audit["checks"].items() if not ok
    ]


def test_bundle_audit_rejects_a_signoff_for_a_different_evaluation(tmp_path):
    artifacts = _stub_evidence(tmp_path)
    for role, kind in (
        ("acoustics", "acoustics_signoff"),
        ("ml", "ml_signoff"),
        ("release", "release_signoff"),
    ):
        path, _sha = write_production_signoff(
            tmp_path / f"{kind}.json", role=role, reviewer_id="reviewer",
            reviewed_scope="scope", release_sha256=RELEASE_SHA,
            evaluation_sha256="d" * 64,
        )
        artifacts[kind] = [str(path.relative_to(tmp_path))]
    bundle = build_evidence_bundle(
        tmp_path, artifacts,
        release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
    )

    audit = audit_m6_production_evidence(
        bundle, evidence_root=tmp_path, release_sha256=RELEASE_SHA,
        evaluation={"evaluation_sha256": EVALUATION_SHA},
        renderer_approval_sha256=["e" * 64],
    )

    assert audit["checks"]["all_three_role_signoffs_approve"] is False


# --- listening -------------------------------------------------------------


def test_protocol_cannot_opt_out_of_the_contract():
    with pytest.raises(ValueError, match="all required"):
        ListeningProtocol(
            participant_count=24, trials_per_participant=10,
            noninferiority_margin=0.5, double_blind=False,
        )
    with pytest.raises(ValueError, match="noninferiority_margin"):
        ListeningProtocol(
            participant_count=24, trials_per_participant=10,
            noninferiority_margin=-1.0,
        )


def test_dry_run_tier_is_the_listening_vocabulary_not_the_profile_one():
    """Two different tuples are both named EVIDENCE_TIERS; using the wrong one fails.

    The renderer profile tiers include "development" and the listening contract's do
    not, and the resulting failure surfaces as evidence_tier_is_declared with no
    hint about which vocabulary was meant.
    """
    assert DRY_RUN_EVIDENCE_TIER in EVIDENCE_TIERS
    assert EMPIRICAL_EVIDENCE_TIER in EVIDENCE_TIERS
    assert "development" not in EVIDENCE_TIERS


def _responses(assignment, *, seed=3, offset=-0.05, scale=0.3, subset=None):
    """Stand-in responses for exercising the ingest. Never shipped as evidence."""
    rng = np.random.default_rng(seed)
    rows = assignment["assignment_records"]
    if subset is not None:
        rows = rows[:subset]
    return [
        {
            "participant_id": row["participant_id"],
            "stimulus_label": row["stimulus_label"],
            "primary_endpoint_difference": float(rng.normal(offset, scale)),
        }
        for row in rows
    ]


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A release with both a synthetic and a measured variant, built once."""
    from egs.rir_generation.phases.m6_bank.scripts.validate_m6_reproducible_generation import (
        _run_generator,
    )
    from puresound.audio.rir.bank.measured_ingest import (
        build_measured_m6_bank,
        prune_bank_to_qc_passed,
    )
    from puresound.audio.rir.bank.qc import run_rir_bank_qc
    from puresound.audio.rir.bank.release import build_m6_variant_release

    from test_rir_measured_ingest import _write_corpus_view

    root = tmp_path_factory.mktemp("evidence-release")
    # Two corpora, so the measured bank carries more than one renderer profile:
    # each measurement chain gets its own, and approving them is per profile.
    _write_corpus_view(root / "view", corpus="probe", rooms=20, per_room=2)
    _write_corpus_view(root / "view", corpus="other", rooms=20, per_room=2, seed=7000)
    build_measured_m6_bank(root / "view", root / "measured", code_revision="test-rev")
    prune_bank_to_qc_passed(root / "measured", root / "measured_pruned")
    generated = _run_generator(root / "synth", workers=1, code_revision="test-rev")
    run_rir_bank_qc(root / "synth")
    manifest = build_m6_variant_release(
        root / "synth", root / "release", release_id="test-release",
        measured_bank_root=root / "measured_pruned",
    )
    return {
        "root": root,
        "release_root": root / "release",
        "release_sha256": str(manifest.release_sha256),
        "audit": generated["audit"],
    }


def test_assignment_pairs_only_across_acoustic_spaces(release):
    """Room-disjoint means the two arms of a trial never share an acoustic space.

    Otherwise a judgement can ride on room familiarity instead of on the renderer.
    """
    protocol = ListeningProtocol(
        participant_count=24, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)

    assert len(assignment["assignment_records"]) == 24 * 10
    assert assignment["meets_empirical_participant_minimum"] is True
    assert not [
        row
        for row in assignment["assignment_records"]
        if row["system_acoustic_space_id"] == row["reference_acoustic_space_id"]
    ]
    labels = {row["stimulus_label"] for row in assignment["assignment_records"]}
    assert len(labels) == 24 * 10
    roles = {row["role"] for row in assignment["assignment_records"]}
    assert {"comparison", "hidden_reference", "degraded_anchor"} == roles


def test_assignment_is_reproducible_from_its_seed(release):
    protocol = ListeningProtocol(
        participant_count=21, trials_per_participant=10, noninferiority_margin=0.5
    )
    first = build_listening_assignment(
        release["release_root"], protocol, assignment_seed=99
    )
    again = build_listening_assignment(
        release["release_root"], protocol, assignment_seed=99
    )
    other = build_listening_assignment(
        release["release_root"], protocol, assignment_seed=100
    )

    assert first["assignment_sha256"] == again["assignment_sha256"]
    assert first["assignment_sha256"] != other["assignment_sha256"]


def test_assignment_flags_a_participant_count_too_small_to_be_empirical(release):
    protocol = ListeningProtocol(
        participant_count=MINIMUM_EMPIRICAL_PARTICIPANTS - 1,
        trials_per_participant=10, noninferiority_margin=0.5,
    )
    assignment = build_listening_assignment(release["release_root"], protocol)

    assert assignment["meets_empirical_participant_minimum"] is False


def test_ingest_refuses_a_response_to_a_stimulus_nobody_was_given(release):
    protocol = ListeningProtocol(
        participant_count=21, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)
    rows = _responses(assignment)
    rows[0]["stimulus_label"] = "stim_deadbeefcafe"

    with pytest.raises(ValueError, match="never issued"):
        ingest_listening_responses(assignment, rows)


def test_ingest_refuses_a_partial_run_at_empirical_tier(release):
    """Scoring what came back would silently change who the estimate represents."""
    protocol = ListeningProtocol(
        participant_count=21, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)

    with pytest.raises(ValueError, match="every issued trial"):
        ingest_listening_responses(
            assignment, _responses(assignment, subset=50),
            evidence_tier=EMPIRICAL_EVIDENCE_TIER,
        )


def test_ingest_rejects_duplicates_and_unusable_values(release):
    protocol = ListeningProtocol(
        participant_count=21, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)
    rows = _responses(assignment)
    with pytest.raises(ValueError, match="duplicate"):
        ingest_listening_responses(assignment, rows + [rows[0]])

    missing = _responses(assignment)
    missing[3]["primary_endpoint_difference"] = None
    with pytest.raises(ValueError, match="finite"):
        ingest_listening_responses(assignment, missing)


def test_dry_run_report_validates_but_is_not_empirical_evidence(release):
    """The whole point: the pipeline can be exercised without claiming human data."""
    protocol = ListeningProtocol(
        participant_count=24, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)
    report = build_dry_run_report(assignment, _responses(assignment))

    validated = validate_listening_report(
        report, release_sha256=release["release_sha256"]
    )

    assert validated["valid"] is True, [
        name for name, ok in validated["checks"].items() if not ok
    ]
    assert validated["empirical"] is False
    assert report["evidence_tier"] == DRY_RUN_EVIDENCE_TIER
    assert report["protocol"]["explicitly_not_human_responses"] is True


def test_empirical_report_passes_the_recomputation_it_will_be_audited_by(release):
    """The validator re-derives the estimate and CI; the ingest must already agree.

    Both sides call the same helper, so agreement is structural rather than a pair
    of hand-written numbers that happen to match.
    """
    protocol = ListeningProtocol(
        participant_count=24, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)
    report = ingest_listening_responses(
        assignment, _responses(assignment), evidence_tier=EMPIRICAL_EVIDENCE_TIER
    )

    validated = validate_listening_report(
        report, release_sha256=release["release_sha256"]
    )

    assert validated["valid"] is True, [
        name for name, ok in validated["checks"].items() if not ok
    ]
    assert validated["empirical"] is True
    assert validated["checks"]["empirical_participant_statistics_are_recomputed"]
    assert validated["checks"]["empirical_artifacts_are_parsed_and_content_addressed"]


def test_tampering_with_a_response_record_breaks_its_content_hash(release):
    protocol = ListeningProtocol(
        participant_count=24, trials_per_participant=10, noninferiority_margin=0.5
    )
    assignment = build_listening_assignment(release["release_root"], protocol)
    report = ingest_listening_responses(
        assignment, _responses(assignment), evidence_tier=EMPIRICAL_EVIDENCE_TIER
    )
    report["artifacts"]["response_records"][0]["primary_endpoint_difference"] = 9.0

    validated = validate_listening_report(
        report, release_sha256=release["release_sha256"]
    )

    assert validated["valid"] is False
    assert not validated["checks"][
        "empirical_artifacts_are_parsed_and_content_addressed"
    ]


# --- the chain -------------------------------------------------------------


def test_renderer_approval_closes_its_production_check(release):
    """Approval must land before QC, or it invalidates the release it blesses.

    A bank's QC summary is bound to its manifest hash, so stamping a profile after
    QC breaks manifest_hash_matches_summary. Re-running QC is the cost of putting
    the approval inside the hash chain.
    """
    from puresound.audio.rir.bank.production import _production_decision_components
    from puresound.audio.rir.bank.qc import audit_rir_bank_qc_release
    from puresound.audio.rir.bank.release import build_m6_variant_release

    root = release["root"]
    evaluation = evaluate_m6_release(
        release["release_root"],
        throughput_report=build_throughput_report(
            release["audit"], release_sha256=release["release_sha256"]
        ),
    )
    approvals_dir = root / "approvals"
    for bank in (root / "synth", root / "measured_pruned"):
        manifest = RIRBankManifest.from_json(
            (bank / "rir_bank_manifest.json").read_text(encoding="utf-8")
        )
        approve_bank_renderer_profiles(
            bank,
            [
                RendererApproval(
                    profile_id=profile.profile_id,
                    approver_id="test-approver",
                    reviewed_scope="scope",
                    evidence_sha256={
                        "m6_5_evaluation": evaluation["evaluation_sha256"]
                    },
                )
                for profile in manifest.renderer_profiles
            ],
            approval_dir=approvals_dir,
        )
        assert audit_rir_bank_qc_release(bank)["valid"] is True

    rebuilt = build_m6_variant_release(
        root / "synth", root / "release_approved", release_id="test-release",
        measured_bank_root=root / "measured_pruned",
    )
    checks, _evidence, _release_audit = _production_decision_components(
        root / "release_approved",
        rebuilt,
        evaluate_m6_release(root / "release_approved"),
        evidence_bundle=None,
        evidence_root=root / "approvals",
    )

    assert checks["all_renderer_profiles_are_production_approved"] is True
    assert checks["all_variant_items_are_qc_passed"] is True
    assert checks["all_required_recipes_are_ready"] is True


def test_approval_requires_every_profile_in_the_bank(release):
    """A partial pass only moves the failure somewhere less obvious."""
    manifest = RIRBankManifest.from_json(
        (release["root"] / "measured_pruned" / "rir_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = list(manifest.renderer_profiles)
    assert len(profiles) > 1

    with pytest.raises(ValueError, match="every renderer profile"):
        approve_bank_renderer_profiles(
            release["root"] / "measured_pruned",
            [
                RendererApproval(
                    profile_id=profiles[0].profile_id,
                    approver_id="test-approver", reviewed_scope="scope",
                    evidence_sha256={"e": EVALUATION_SHA},
                )
            ],
            approval_dir=release["root"] / "partial_approvals",
        )


def test_approval_rejects_an_unknown_profile(release):
    with pytest.raises(ValueError, match="unknown renderer profiles"):
        approve_bank_renderer_profiles(
            release["root"] / "synth",
            [
                RendererApproval(
                    profile_id="not-a-profile", approver_id="test-approver",
                    reviewed_scope="scope", evidence_sha256={"e": EVALUATION_SHA},
                )
            ],
            approval_dir=release["root"] / "unknown_approvals",
        )
