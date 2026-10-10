import pytest

from puresound.audio.rir.bank.evaluation import (
    M6_LISTENING_SCHEMA_VERSION,
    M6_THROUGHPUT_SCHEMA_VERSION,
    validate_listening_report,
    validate_throughput_report,
)

RELEASE_SHA = "a" * 64


def test_missing_or_malformed_external_evidence_is_invalid_instead_of_raising():
    assert validate_throughput_report(None, release_sha256=RELEASE_SHA)["valid"] is False
    assert validate_listening_report(None, release_sha256=RELEASE_SHA)["empirical"] is False

    throughput = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": RELEASE_SHA,
        "task_count": None,
        "items_generated": "not-an-integer",
        "items_skipped": [],
        "items_failed": {},
        "elapsed_seconds": "not-a-number",
        "items_per_second": None,
        "num_workers": False,
    }
    listening = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": RELEASE_SHA,
        "evidence_tier": "empirical",
        "protocol": [],
        "results": "invalid",
        "artifacts": None,
    }

    assert validate_throughput_report(throughput, release_sha256=RELEASE_SHA)[
        "valid"
    ] is False
    assert validate_listening_report(listening, release_sha256=RELEASE_SHA)[
        "valid"
    ] is False


def test_throughput_contract_recomputes_rate_and_counts():
    report = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": RELEASE_SHA,
        "task_count": 6,
        "items_generated": 5,
        "items_skipped": 1,
        "items_failed": 0,
        "elapsed_seconds": 2.0,
        "items_per_second": 2.5,
        "num_workers": 2,
    }
    assert validate_throughput_report(report, release_sha256=RELEASE_SHA)["valid"]
    report["items_per_second"] = 3.0
    assert not validate_throughput_report(report, release_sha256=RELEASE_SHA)["valid"]


@pytest.mark.parametrize(
    "protocol_overrides,failed_check",
    [
        ({"double_blind": False}, "randomized_and_double_blind"),
        (
            {"explicitly_not_human_responses": True},
            "empirical_claim_is_not_explicitly_nonhuman",
        ),
    ],
    ids=["unblinded", "explicitly_nonhuman"],
)
def test_listening_contract_rejects_an_empirical_claim_it_cannot_back(
    protocol_overrides, failed_check
):
    report = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": RELEASE_SHA,
        "evidence_tier": "empirical",
        "protocol": {
            "randomized": True,
            "double_blind": True,
            "common_loudness_master_gain": True,
            "hidden_reference": True,
            "degraded_anchor": True,
            "room_disjoint_stimuli": True,
            "participant_count": 24,
            **protocol_overrides,
        },
        "results": {
            "completed": True,
            "noninferiority_margin": 0.1,
            "confidence_interval_low": 0.0,
        },
        "artifacts": {
            "assignment_sha256": "1" * 64,
            "responses_sha256": "2" * 64,
            "analysis_sha256": "3" * 64,
        },
    }

    result = validate_listening_report(report, release_sha256=RELEASE_SHA)

    assert result["valid"] is False
    assert result["empirical"] is False
    assert result["checks"][failed_check] is False
