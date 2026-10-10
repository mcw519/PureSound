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

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow

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
SIGNOFF_ROLES = (
    ("acoustics", "acoustics_signoff"),
    ("ml", "ml_signoff"),
    ("release", "release_signoff"),
)


def _generation_run(**overrides):
    return {"generation_run": {**AUDIT["generation_run"], **overrides}}


def test_throughput_report_satisfies_its_own_validator():
    report = build_throughput_report(AUDIT, release_sha256=RELEASE_SHA)
    validated = validate_throughput_report(report, release_sha256=RELEASE_SHA)

    assert validated["valid"] is True
    assert report["items_per_second"] == pytest.approx(7 / 14.0)


@pytest.mark.parametrize(
    "audit,match",
    [
        (_generation_run(elapsed_seconds=None), "elapsed_seconds"),
        (_generation_run(items_failed=None), "items_failed"),
        ({}, "generation_run"),
        (_generation_run(task_count=99), "inconsistent"),
    ],
    ids=["untimed", "unfailed", "no_run", "inconsistent_counts"],
)
def test_throughput_report_refuses_a_run_it_cannot_vouch_for(audit, match):
    """A missing measurement must stay missing rather than become a default.

    The contract requires a positive duration, an explicit failure count and
    counts that add up; filling any of them in here would convert "nobody
    measured this" into a passing check.
    """
    with pytest.raises(ValueError, match=match):
        build_throughput_report(audit, release_sha256=RELEASE_SHA)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"approver_id": "  "}, "approver_id"),
        ({"reviewed_scope": ""}, "reviewed_scope"),
        ({"evidence_sha256": {}}, "at least one evidence hash"),
        ({"decision": "looks-fine"}, "decision must be"),
    ],
)
def test_renderer_approval_needs_a_person_and_stated_evidence(overrides, match):
    kwargs = {
        "profile_id": "p",
        "approver_id": "someone",
        "reviewed_scope": "scope",
        "evidence_sha256": {"e": "c" * 64},
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        RendererApproval(**kwargs)


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


def _point_at_absent_file(artifacts):
    artifacts["listening_analysis"] = ["listening/absent.json"]


def _drop_downstream_recipe(artifacts):
    del artifacts["downstream_training_recipe"]


def _drop_renderer_approval(artifacts):
    del artifacts["renderer_approval"]


def _declare_a_path_twice(artifacts):
    artifacts["ml_signoff"] = ["acoustics_signoff.json"]


def _escape_the_root(artifacts):
    artifacts["release_signoff"] = ["../outside.json"]


@pytest.mark.parametrize(
    "corrupt,error,match",
    [
        (_point_at_absent_file, FileNotFoundError, "listening_analysis"),
        (_drop_downstream_recipe, ValueError, "downstream_training_recipe"),
        (_drop_renderer_approval, ValueError, "renderer_approval"),
        (_declare_a_path_twice, ValueError, "declared twice"),
        (_escape_the_root, ValueError, "relative path"),
    ],
)
def test_bundle_refuses_evidence_it_cannot_describe(tmp_path, corrupt, error, match):
    artifacts = _stub_evidence(tmp_path)
    corrupt(artifacts)

    with pytest.raises(error, match=match):
        build_evidence_bundle(
            tmp_path, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )


def test_bundle_audit_accepts_its_own_bundle_and_rejects_foreign_signoffs(tmp_path):
    """Producer and validator have to agree, and only an audit can show it."""

    def signed_bundle(root, signed_evaluation_sha):
        root.mkdir()
        artifacts = _stub_evidence(root)
        approval_path, approval_sha = write_renderer_approval_record(
            root / "renderer_approval.json",
            RendererApproval(
                profile_id="prof", approver_id="someone", reviewed_scope="scope",
                evidence_sha256={"m6_5_evaluation": EVALUATION_SHA},
            ),
            profile={"renderer_id": "r"},
        )
        artifacts["renderer_approval"] = [str(approval_path.relative_to(root))]
        for role, kind in SIGNOFF_ROLES:
            path, _sha = write_production_signoff(
                root / f"{kind}.json", role=role, reviewer_id="reviewer",
                reviewed_scope="scope", release_sha256=RELEASE_SHA,
                evaluation_sha256=signed_evaluation_sha,
            )
            artifacts[kind] = [str(path.relative_to(root))]
        bundle = build_evidence_bundle(
            root, artifacts,
            release_sha256=RELEASE_SHA, evaluation_sha256=EVALUATION_SHA,
        )
        return bundle, approval_sha

    root = tmp_path / "own"
    bundle, approval_sha = signed_bundle(root, EVALUATION_SHA)
    evaluation = {
        "evaluation_sha256": EVALUATION_SHA,
        "listening_evaluation": {
            "artifact_sha256": {
                name: sha256_file(root / f"{kind}.json")
                for kind, name in (
                    ("listening_assignment", "assignment_sha256"),
                    ("listening_responses", "responses_sha256"),
                    ("listening_analysis", "analysis_sha256"),
                )
            }
        },
        "downstream_evaluation": {
            "artifact_sha256": {
                name: sha256_file(root / f"{kind}.json")
                for kind, name in (
                    ("downstream_training_recipe", "training_recipe_sha256"),
                    ("downstream_checkpoints", "checkpoints_sha256"),
                    ("downstream_analysis", "analysis_sha256"),
                )
            }
        },
    }
    audit = audit_m6_production_evidence(
        bundle,
        evidence_root=root,
        release_sha256=RELEASE_SHA,
        evaluation=evaluation,
        renderer_approval_sha256=[approval_sha],
    )
    assert audit["valid"] is True, [
        name for name, ok in audit["checks"].items() if not ok
    ]

    foreign_root = tmp_path / "foreign"
    foreign, _sha = signed_bundle(foreign_root, "d" * 64)
    rejected = audit_m6_production_evidence(
        foreign, evidence_root=foreign_root, release_sha256=RELEASE_SHA,
        evaluation={"evaluation_sha256": EVALUATION_SHA},
        renderer_approval_sha256=["e" * 64],
    )
    assert rejected["checks"]["all_three_role_signoffs_approve"] is False


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


def _protocol(participant_count=24):
    return ListeningProtocol(
        participant_count=participant_count,
        trials_per_participant=10,
        noninferiority_margin=0.5,
    )


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """A release with both a synthetic and a measured variant, built once."""
    from egs.rir_generation.phases.m6_bank.scripts.validate_m6_reproducible_generation import (
        _run_generator,
    )
    from puresound.audio.rir.bank.measured_ingest import build_measured_m6_bank
    from puresound.audio.rir.bank.qc import run_rir_bank_qc
    from puresound.audio.rir.bank.release import (
        build_m6_variant_release,
        prune_bank_to_qc_passed,
    )

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


def test_assignment_is_room_disjoint_seeded_and_flags_small_panels(release):
    """Room-disjoint means the two arms of a trial never share an acoustic space.

    Otherwise a judgement can ride on room familiarity instead of on the renderer.
    """
    assignment = build_listening_assignment(release["release_root"], _protocol())

    records = assignment["assignment_records"]
    assert len(records) == 24 * 10
    assert assignment["meets_empirical_participant_minimum"] is True
    assert not [
        row
        for row in records
        if row["system_acoustic_space_id"] == row["reference_acoustic_space_id"]
    ]
    assert len({row["stimulus_label"] for row in records}) == 24 * 10
    assert {row["role"] for row in records} == {
        "comparison", "hidden_reference", "degraded_anchor"
    }

    first, again, other = (
        build_listening_assignment(
            release["release_root"], _protocol(21), assignment_seed=seed
        )
        for seed in (99, 99, 100)
    )
    assert first["assignment_sha256"] == again["assignment_sha256"]
    assert first["assignment_sha256"] != other["assignment_sha256"]

    small = build_listening_assignment(
        release["release_root"], _protocol(MINIMUM_EMPIRICAL_PARTICIPANTS - 1)
    )
    assert small["meets_empirical_participant_minimum"] is False


def _unknown_stimulus(assignment):
    rows = _responses(assignment)
    rows[0]["stimulus_label"] = "stim_deadbeefcafe"
    return rows, {}


def _partial_empirical_run(assignment):
    # Scoring what came back would silently change who the estimate represents.
    return _responses(assignment, subset=50), {"evidence_tier": EMPIRICAL_EVIDENCE_TIER}


def _duplicate_response(assignment):
    rows = _responses(assignment)
    return rows + [rows[0]], {}


def _missing_value(assignment):
    rows = _responses(assignment)
    rows[3]["primary_endpoint_difference"] = None
    return rows, {}


@pytest.mark.parametrize(
    "make_rows,match",
    [
        (_unknown_stimulus, "never issued"),
        (_partial_empirical_run, "every issued trial"),
        (_duplicate_response, "duplicate"),
        (_missing_value, "finite"),
    ],
)
def test_ingest_refuses_responses_it_cannot_attribute(release, make_rows, match):
    assignment = build_listening_assignment(release["release_root"], _protocol(21))
    rows, kwargs = make_rows(assignment)

    with pytest.raises(ValueError, match=match):
        ingest_listening_responses(assignment, rows, **kwargs)


def test_dry_run_report_validates_but_is_not_empirical_evidence(release):
    """The whole point: the pipeline can be exercised without claiming human data.

    The listening tiers are their own vocabulary: the renderer-profile tiers
    include "development" and the listening contract's do not.
    """
    assert DRY_RUN_EVIDENCE_TIER in EVIDENCE_TIERS
    assert EMPIRICAL_EVIDENCE_TIER in EVIDENCE_TIERS
    assert "development" not in EVIDENCE_TIERS

    assignment = build_listening_assignment(release["release_root"], _protocol())
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


def test_empirical_report_passes_its_recomputation_and_detects_tampering(release):
    """The validator re-derives the estimate and CI; the ingest must already agree.

    Both sides call the same helper, so agreement is structural rather than a pair
    of hand-written numbers that happen to match.
    """
    assignment = build_listening_assignment(release["release_root"], _protocol())
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

    report["artifacts"]["response_records"][0]["primary_endpoint_difference"] = 9.0
    tampered = validate_listening_report(
        report, release_sha256=release["release_sha256"]
    )
    assert tampered["valid"] is False
    assert not tampered["checks"][
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
    from puresound.audio.rir.bank.release import (
        build_m6_variant_release,
    )

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


@pytest.mark.parametrize(
    "bank,which,match",
    [
        # A partial pass only moves the failure somewhere less obvious.
        ("measured_pruned", "first_profile", "every renderer profile"),
        ("synth", "unknown_profile", "unknown renderer profiles"),
    ],
)
def test_approval_must_cover_exactly_the_banks_profiles(release, bank, which, match):
    bank_root = release["root"] / bank
    manifest = RIRBankManifest.from_json(
        (bank_root / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    profiles = list(manifest.renderer_profiles)
    if which == "first_profile":
        assert len(profiles) > 1
        profile_id = profiles[0].profile_id
    else:
        profile_id = "not-a-profile"

    with pytest.raises(ValueError, match=match):
        approve_bank_renderer_profiles(
            bank_root,
            [
                RendererApproval(
                    profile_id=profile_id,
                    approver_id="test-approver", reviewed_scope="scope",
                    evidence_sha256={"e": EVALUATION_SHA},
                )
            ],
            approval_dir=release["root"] / f"{which}_approvals",
        )
