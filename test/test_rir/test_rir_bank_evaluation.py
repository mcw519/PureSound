from puresound.audio.rir.bank.evaluation import (
    M6_LISTENING_SCHEMA_VERSION,
    M6_THROUGHPUT_SCHEMA_VERSION,
    validate_listening_report,
    validate_throughput_report,
)


def test_missing_m6_5_external_evidence_fails_closed():
    assert validate_throughput_report(None, release_sha256="a" * 64)["valid"] is False
    assert validate_listening_report(None, release_sha256="a" * 64)["empirical"] is False


def test_throughput_contract_recomputes_rate_and_counts():
    report = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": "a" * 64,
        "task_count": 6,
        "items_generated": 5,
        "items_skipped": 1,
        "items_failed": 0,
        "elapsed_seconds": 2.0,
        "items_per_second": 2.5,
        "num_workers": 2,
    }
    assert validate_throughput_report(report, release_sha256="a" * 64)["valid"]
    report["items_per_second"] = 3.0
    assert not validate_throughput_report(report, release_sha256="a" * 64)["valid"]


def test_malformed_external_evidence_returns_invalid_instead_of_raising():
    throughput = {
        "schema_version": M6_THROUGHPUT_SCHEMA_VERSION,
        "release_sha256": "a" * 64,
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
        "release_sha256": "a" * 64,
        "evidence_tier": "empirical",
        "protocol": [],
        "results": "invalid",
        "artifacts": None,
    }

    assert validate_throughput_report(throughput, release_sha256="a" * 64)[
        "valid"
    ] is False
    assert validate_listening_report(listening, release_sha256="a" * 64)[
        "valid"
    ] is False


def test_listening_contract_rejects_unblinded_empirical_claim():
    report = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": "a" * 64,
        "evidence_tier": "empirical",
        "protocol": {
            "randomized": True,
            "double_blind": False,
            "common_loudness_master_gain": True,
            "hidden_reference": True,
            "degraded_anchor": True,
            "room_disjoint_stimuli": True,
            "participant_count": 24,
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
    result = validate_listening_report(report, release_sha256="a" * 64)
    assert result["valid"] is False
    assert result["empirical"] is False
    assert result["checks"]["randomized_and_double_blind"] is False


def test_listening_contract_rejects_explicit_nonhuman_empirical_claim():
    report = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": "a" * 64,
        "evidence_tier": "empirical",
        "protocol": {
            "randomized": True,
            "double_blind": True,
            "common_loudness_master_gain": True,
            "hidden_reference": True,
            "degraded_anchor": True,
            "room_disjoint_stimuli": True,
            "participant_count": 24,
            "explicitly_not_human_responses": True,
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

    result = validate_listening_report(report, release_sha256="a" * 64)
    assert result["valid"] is False
    assert result["empirical"] is False
    assert result["checks"][
        "empirical_claim_is_not_explicitly_nonhuman"
    ] is False
