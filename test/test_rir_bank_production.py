import json

from puresound.audio.rir.bank.schema import canonical_json_sha256, sha256_file
from puresound.audio.rir.bank.production import (
    M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
    M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION,
    audit_m6_production_evidence,
)


def test_missing_m6_6_evidence_bundle_fails_closed():
    audit = audit_m6_production_evidence(
        None,
        evidence_root=".",
        release_sha256="a" * 64,
        evaluation={"evaluation_sha256": "b" * 64},
        renderer_approval_sha256=(),
    )

    assert audit["valid"] is False
    assert audit["checks"]["bundle_present"] is False
    assert not any(audit["checks"].values())


def test_complete_m6_6_evidence_bundle_links_actual_files(tmp_path):
    release_sha256 = "a" * 64
    evaluation_sha256 = "b" * 64
    evaluation = {
        "evaluation_sha256": evaluation_sha256,
        "listening_evaluation": {"artifact_sha256": {}},
        "downstream_evaluation": {"artifact_sha256": {}},
    }
    artifact_specs = (
        ("listening_assignment", "listening_evaluation", "assignment_sha256"),
        ("listening_responses", "listening_evaluation", "responses_sha256"),
        ("listening_analysis", "listening_evaluation", "analysis_sha256"),
        (
            "downstream_training_recipe",
            "downstream_evaluation",
            "training_recipe_sha256",
        ),
        (
            "downstream_checkpoints",
            "downstream_evaluation",
            "checkpoints_sha256",
        ),
        ("downstream_analysis", "downstream_evaluation", "analysis_sha256"),
    )
    artifacts = []
    for index, (kind, section, field) in enumerate(artifact_specs):
        path = tmp_path / f"artifact_{index}.bin"
        path.write_bytes(f"m6.6-{kind}".encode())
        digest = sha256_file(path)
        evaluation[section]["artifact_sha256"][field] = digest
        artifacts.append({"kind": kind, "path": path.name, "sha256": digest})

    renderer = tmp_path / "renderer_approval.json"
    renderer.write_text('{"approved":true}\n', encoding="utf-8")
    renderer_sha256 = sha256_file(renderer)
    artifacts.append(
        {
            "kind": "renderer_approval",
            "path": renderer.name,
            "sha256": renderer_sha256,
        }
    )
    for role, kind in (
        ("acoustics", "acoustics_signoff"),
        ("ml", "ml_signoff"),
        ("release", "release_signoff"),
    ):
        path = tmp_path / f"{kind}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": M6_PRODUCTION_SIGNOFF_SCHEMA_VERSION,
                    "role": role,
                    "reviewer_id": f"reviewer-{role}",
                    "decision": "approve",
                    "release_sha256": release_sha256,
                    "evaluation_sha256": evaluation_sha256,
                    "reviewed_scope": "M6 production evidence bundle",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {"kind": kind, "path": path.name, "sha256": sha256_file(path)}
        )
    payload = {
        "schema_version": M6_PRODUCTION_EVIDENCE_SCHEMA_VERSION,
        "release_sha256": release_sha256,
        "evaluation_sha256": evaluation_sha256,
        "artifacts": artifacts,
    }
    bundle = {**payload, "bundle_sha256": canonical_json_sha256(payload)}

    audit = audit_m6_production_evidence(
        bundle,
        evidence_root=tmp_path,
        release_sha256=release_sha256,
        evaluation=evaluation,
        renderer_approval_sha256=(renderer_sha256,),
    )
    assert audit["valid"] is True
    assert all(audit["checks"].values())

    renderer.write_text('{"approved":false}\n', encoding="utf-8")
    tampered = audit_m6_production_evidence(
        bundle,
        evidence_root=tmp_path,
        release_sha256=release_sha256,
        evaluation=evaluation,
        renderer_approval_sha256=(renderer_sha256,),
    )
    assert tampered["valid"] is False
    assert tampered["checks"]["artifact_paths_and_hashes_match"] is False
