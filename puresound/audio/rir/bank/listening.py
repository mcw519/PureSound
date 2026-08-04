"""Design a controlled listening test on an M6 release, and ingest its responses.

``validate_listening_report`` is strict in the right way: at
``evidence_tier: empirical`` it re-derives the estimate and the paired-t
confidence interval from raw per-participant records and requires the declared
numbers to match, and it will not call anything empirical below twenty
participants. What it has never had is a producer. Nothing built the stimulus
assignment a test would run, and nothing turned collected responses into a report,
so the only way to satisfy it was to hand-write the numbers — which is to say, to
make them up.

This module closes that gap from both ends and deliberately keeps them apart.

``build_listening_assignment`` designs the experiment: it draws room-disjoint
stimulus pairs from the release, adds the hidden reference and degraded anchor the
protocol requires, and produces a randomized per-participant presentation order
under labels that carry no hint of condition. The mapping from label back to
condition lives in the assignment, which is what makes the test double-blind and
also what makes it scorable later.

``ingest_listening_responses`` reads what the participants actually said, checks it
against the assignment, and computes the statistics. It cannot invent a response:
every record must name a participant and a stimulus label that the assignment
issued, and a run missing responses is reported as incomplete rather than scored
on what arrived.

There is no function here that produces responses. A dry run — verifying the
pipeline end to end without human data — goes through ``build_dry_run_report``,
which emits ``evidence_tier: contract_fixture`` with
``explicitly_not_human_responses: true``. The validator accepts that as a
well-formed contract and correctly refuses to count it as empirical evidence. That
tier is the honest way to exercise this machinery, and the reason it exists.

Beware that the listening contract's tiers are ``("contract_fixture",
"empirical")``, which is a *different* vocabulary from the renderer profile tiers
in ``bank.schema`` — two module-level names spelled ``EVIDENCE_TIERS`` hold
different tuples. Declaring a listening report ``development`` fails
``evidence_tier_is_declared``, and the check name does not hint at why, so both
tiers are named as constants here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import (
    EVIDENCE_TIERS,
    M6_LISTENING_SCHEMA_VERSION,
    _paired_t_confidence_interval,
)
from .release import RIRBankReleaseManifest
from .schema import canonical_json_sha256

M6_LISTENING_ASSIGNMENT_SCHEMA_VERSION = "puresound.m6_listening_assignment.v1"

#: The tier a run without human listeners must declare. Note that the listening
#: contract's tiers are ``("contract_fixture", "empirical")`` — a different
#: vocabulary from the renderer profile tiers in ``bank.schema``, which use
#: ``development``. Reaching for the wrong one produces a report that fails
#: ``evidence_tier_is_declared`` for a reason the message does not explain.
DRY_RUN_EVIDENCE_TIER = "contract_fixture"
EMPIRICAL_EVIDENCE_TIER = "empirical"

#: ``validate_listening_report`` refuses to call a run empirical below this many
#: participants, so an assignment that plans fewer cannot produce usable evidence
#: and says so at design time rather than after the sessions are run.
MINIMUM_EMPIRICAL_PARTICIPANTS = 20

#: The roles a trial can carry. ``hidden_reference`` and ``degraded_anchor`` are
#: required by the protocol checks: the first catches a participant who cannot hear
#: the difference at all, the second catches one who marks everything the same.
TRIAL_ROLES = ("comparison", "hidden_reference", "degraded_anchor")


def _blind_label(assignment_seed: int, participant: str, index: int) -> str:
    """A stimulus label that leaks nothing about the condition it stands for.

    Derived from the seed so the whole assignment is reproducible, and hashed so
    that sorting or eyeballing the labels tells a participant — or an analyst —
    nothing about which arm a trial belongs to.
    """
    digest = hashlib.sha256(
        f"{assignment_seed}\0{participant}\0{index}".encode()
    ).hexdigest()
    return f"stim_{digest[:12]}"


@dataclass(frozen=True)
class ListeningProtocol:
    """The protocol claims ``validate_listening_report`` checks.

    Each field maps to a check rather than to prose, and they are all required to
    be true, so an assignment cannot be designed that quietly omits blinding or
    loudness matching and then fails validation after the sessions are run.
    """

    participant_count: int
    trials_per_participant: int
    noninferiority_margin: float
    primary_endpoint: str = "degradation_mean_opinion_score_difference"
    common_loudness_master_gain: float = 1.0
    randomized: bool = True
    double_blind: bool = True
    hidden_reference: bool = True
    degraded_anchor: bool = True
    room_disjoint_stimuli: bool = True

    def __post_init__(self) -> None:
        if self.participant_count < 1:
            raise ValueError("participant_count must be positive")
        if self.trials_per_participant < 1:
            raise ValueError("trials_per_participant must be positive")
        if not math.isfinite(self.noninferiority_margin) or (
            self.noninferiority_margin < 0.0
        ):
            raise ValueError("noninferiority_margin must be finite and non-negative")
        if self.common_loudness_master_gain <= 0.0:
            raise ValueError("common_loudness_master_gain must be positive")
        if not all(
            (
                self.randomized,
                self.double_blind,
                self.hidden_reference,
                self.degraded_anchor,
                self.room_disjoint_stimuli,
            )
        ):
            raise ValueError(
                "randomization, double blinding, a hidden reference, a degraded "
                "anchor and room-disjoint stimuli are all required by the M6.5 "
                "listening contract; an assignment cannot opt out of them"
            )
        if not str(self.primary_endpoint).strip():
            raise ValueError("primary_endpoint must be named")

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_count": int(self.participant_count),
            "trials_per_participant": int(self.trials_per_participant),
            "noninferiority_margin": float(self.noninferiority_margin),
            "primary_endpoint": str(self.primary_endpoint),
            "common_loudness_master_gain": float(self.common_loudness_master_gain),
            "randomized": True,
            "double_blind": True,
            "hidden_reference": True,
            "degraded_anchor": True,
            "room_disjoint_stimuli": True,
        }


def _recipe_rows(release_root: Path, recipe_id: str, split: str) -> list[dict[str, Any]]:
    release = RIRBankReleaseManifest.from_json(
        (release_root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    recipe = next(
        (item for item in release.recipes if item.recipe_id == recipe_id), None
    )
    if recipe is None or recipe.status != "ready" or not recipe.split_indexes:
        raise ValueError(f"recipe {recipe_id} is not ready in this release")
    index = recipe.split_indexes[split]
    return [
        json.loads(line)
        for line in (release_root / index.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_listening_assignment(
    release_root: str | Path,
    protocol: ListeningProtocol,
    *,
    reference_recipe_id: str = "real_native",
    system_recipe_id: str = "synthetic_calibrated",
    split: str = "test",
    assignment_seed: int = 20260804,
) -> dict[str, Any]:
    """Design the stimulus assignment for a controlled listening test.

    Pairs each synthetic item against a measured one from a *different* acoustic
    space, which is what ``room_disjoint_stimuli`` means here: no participant hears
    the same room in both arms of a comparison, so a judgement cannot be carried by
    room familiarity. The pairing is drawn from the release's own recipe indexes,
    so the stimuli are exactly the items the release ships.

    Uses the ``test`` split by default. Designing a listening test on ``train``
    would measure the renderer on rooms the downstream model also trains on, and
    the two pieces of empirical evidence would stop being independent.
    """

    root = Path(release_root)
    reference = _recipe_rows(root, reference_recipe_id, split)
    system = _recipe_rows(root, system_recipe_id, split)
    if not reference or not system:
        raise ValueError(
            f"both {system_recipe_id} and {reference_recipe_id} need items in the "
            f"{split} split to build a comparison"
        )

    rng = np.random.default_rng(assignment_seed)
    trials: list[dict[str, Any]] = []
    participants: list[dict[str, Any]] = []
    for participant_index in range(protocol.participant_count):
        participant_id = f"P{participant_index + 1:03d}"
        order: list[dict[str, Any]] = []
        for trial_index in range(protocol.trials_per_participant):
            system_row = system[int(rng.integers(len(system)))]
            # Draw the reference from a different acoustic space than the system
            # item, retrying rather than accepting a same-room pair.
            candidates = [
                row
                for row in reference
                if row["acoustic_space_id"] != system_row["acoustic_space_id"]
            ]
            if not candidates:
                raise ValueError(
                    "no measured item sits in a different acoustic space from "
                    f"{system_row['acoustic_space_id']}; the release cannot support "
                    "a room-disjoint comparison"
                )
            reference_row = candidates[int(rng.integers(len(candidates)))]
            role = TRIAL_ROLES[0]
            if trial_index % 10 == 4:
                role = "hidden_reference"
            elif trial_index % 10 == 9:
                role = "degraded_anchor"
            label = _blind_label(assignment_seed, participant_id, trial_index)
            trials.append(
                {
                    "participant_id": participant_id,
                    "presentation_index": trial_index,
                    "stimulus_label": label,
                    "role": role,
                    "system_release_item_id": system_row["release_item_id"],
                    "system_acoustic_space_id": system_row["acoustic_space_id"],
                    "reference_release_item_id": reference_row["release_item_id"],
                    "reference_acoustic_space_id": (
                        reference_row["acoustic_space_id"]
                    ),
                }
            )
            order.append({"presentation_index": trial_index, "stimulus_label": label})
        participants.append({"participant_id": participant_id, "order": order})

    release = RIRBankReleaseManifest.from_json(
        (root / "rir_bank_release.json").read_text(encoding="utf-8")
    )
    assignment: dict[str, Any] = {
        "schema_version": M6_LISTENING_ASSIGNMENT_SCHEMA_VERSION,
        "release_sha256": release.release_sha256,
        "assignment_seed": int(assignment_seed),
        "split": split,
        "system_recipe_id": system_recipe_id,
        "reference_recipe_id": reference_recipe_id,
        "protocol": protocol.to_dict(),
        "assignment_records": trials,
        "participants": participants,
        "meets_empirical_participant_minimum": bool(
            protocol.participant_count >= MINIMUM_EMPIRICAL_PARTICIPANTS
        ),
        "minimum_empirical_participants": MINIMUM_EMPIRICAL_PARTICIPANTS,
    }
    assignment["assignment_sha256"] = canonical_json_sha256(trials)
    return assignment


def _load_records(value: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        parsed = json.loads(text)
        if isinstance(parsed, Mapping):
            parsed = parsed.get("response_records", parsed.get("responses"))
        if not isinstance(parsed, list):
            raise ValueError(f"{path} does not contain a list of response records")
        return [dict(row) for row in parsed]
    return [dict(row) for row in value]


def ingest_listening_responses(
    assignment: str | Path | Mapping[str, Any],
    responses: str | Path | Sequence[Mapping[str, Any]],
    *,
    evidence_tier: str = EMPIRICAL_EVIDENCE_TIER,
    explicitly_not_human_responses: bool = False,
    analysis_notes: str = "",
) -> dict[str, Any]:
    """Score collected responses against their assignment into an M6.5 report.

    Every response must name a participant and a stimulus label the assignment
    issued; anything else is a response to a stimulus nobody was given. Coverage is
    reported and, at ``empirical`` tier, required to be complete — scoring a
    partially returned run would silently change which participants the estimate
    represents.

    The estimate is the mean of per-participant means and the interval is a paired-t
    on those means, computed with the same helper the validator uses to re-derive
    them, so a report produced here agrees with its own audit by construction
    rather than by a matching pair of hand-written numbers.
    """

    if isinstance(assignment, (str, Path)):
        plan = json.loads(Path(assignment).read_text(encoding="utf-8"))
    else:
        plan = dict(assignment)
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(
            f"evidence_tier must be one of {EVIDENCE_TIERS}; the listening contract "
            "uses a different vocabulary from the renderer profile tiers"
        )
    records = _load_records(responses)
    protocol = dict(plan["protocol"])
    issued = {
        (row["participant_id"], row["stimulus_label"]): row
        for row in plan["assignment_records"]
    }

    scored: list[dict[str, Any]] = []
    for row in records:
        participant_id = str(row.get("participant_id", "")).strip()
        label = str(row.get("stimulus_label", "")).strip()
        key = (participant_id, label)
        if key not in issued:
            raise ValueError(
                f"response cites a stimulus the assignment never issued: {key}"
            )
        value = row.get("primary_endpoint_difference")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"response for {key} has no finite primary_endpoint_difference"
            )
        trial = issued[key]
        scored.append(
            {
                "participant_id": participant_id,
                "stimulus_label": label,
                "role": trial["role"],
                "primary_endpoint_difference": float(value),
            }
        )
    if len({(row["participant_id"], row["stimulus_label"]) for row in scored}) != len(
        scored
    ):
        raise ValueError("responses contain duplicate participant/stimulus pairs")

    # The hidden reference and the degraded anchor screen participants; they are not
    # part of the endpoint. They are therefore kept out of ``response_records`` and
    # summarized in the analysis instead — not merely skipped in the average here.
    # ``validate_listening_report`` re-derives the estimate from *every* record it is
    # handed, so leaving anchors in that list would make the audit disagree with the
    # analysis, and the only ways to reconcile that are to fold anchors into the
    # endpoint or to weaken the audit.
    comparison = [row for row in scored if row["role"] == "comparison"]
    validity = [row for row in scored if row["role"] != "comparison"]
    per_participant: dict[str, list[float]] = {}
    for row in comparison:
        per_participant.setdefault(row["participant_id"], []).append(
            row["primary_endpoint_difference"]
        )
    # Participants are the unit of analysis: averaging trials first stops a
    # participant who completed more trials from carrying more weight.
    means = [float(np.mean(values)) for _pid, values in sorted(per_participant.items())]
    interval = _paired_t_confidence_interval(means)
    if interval is None:
        raise ValueError(
            "a confidence interval needs at least two participants with comparison "
            "trials"
        )
    estimate = float(np.mean(means))
    margin = float(protocol["noninferiority_margin"])

    expected_participants = int(protocol["participant_count"])
    coverage = {
        "issued_trials": len(issued),
        "returned_trials": len(scored),
        "endpoint_trials": len(comparison),
        "validity_trials": len(validity),
        "participants_expected": expected_participants,
        "participants_returned": len(per_participant),
        "complete": bool(
            len(scored) == len(issued) and len(per_participant) == expected_participants
        ),
    }
    if evidence_tier == EMPIRICAL_EVIDENCE_TIER and not coverage["complete"]:
        raise ValueError(
            "an empirical claim needs every issued trial returned by every "
            f"participant; got {coverage['returned_trials']}/{coverage['issued_trials']} "
            f"trials from {coverage['participants_returned']}/"
            f"{expected_participants} participants"
        )

    validity_by_role: dict[str, list[float]] = {}
    for row in validity:
        validity_by_role.setdefault(row["role"], []).append(
            row["primary_endpoint_difference"]
        )
    analysis_record = {
        "estimate": estimate,
        "confidence_interval_low": interval[0],
        "confidence_interval_high": interval[1],
        "participant_count": len(per_participant),
        "comparison_trial_count": len(comparison),
        "primary_endpoint": protocol["primary_endpoint"],
        "noninferiority_margin": margin,
        "analysis_policy": (
            "mean_of_per_participant_means_with_paired_t_interval"
            "_over_comparison_trials_only"
        ),
        "validity_screening": {
            role: {
                "trial_count": len(values),
                "mean": float(np.mean(values)),
                "policy": (
                    "reported for participant screening; excluded from the endpoint"
                ),
            }
            for role, values in sorted(validity_by_role.items())
        },
        "notes": analysis_notes,
    }
    report = {
        "schema_version": M6_LISTENING_SCHEMA_VERSION,
        "release_sha256": plan["release_sha256"],
        "evidence_tier": evidence_tier,
        "protocol": {
            **protocol,
            "explicitly_not_human_responses": bool(explicitly_not_human_responses),
        },
        "results": {
            "completed": True,
            "estimate": estimate,
            "confidence_interval_low": interval[0],
            "confidence_interval_high": interval[1],
            "noninferiority_margin": margin,
            "noninferior": bool(interval[0] >= -margin),
        },
        "coverage": coverage,
        "artifacts": {
            "assignment_records": plan["assignment_records"],
            # Endpoint trials only, so the validator's recomputation over this list
            # reproduces the estimate above rather than a different quantity.
            "response_records": comparison,
            "validity_records": validity,
            "analysis_record": analysis_record,
            "assignment_sha256": canonical_json_sha256(plan["assignment_records"]),
            "responses_sha256": canonical_json_sha256(comparison),
            "analysis_sha256": canonical_json_sha256(analysis_record),
        },
    }
    return report


def build_dry_run_report(
    assignment: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    *,
    analysis_notes: str = "pipeline dry run; responses are not from human listeners",
) -> dict[str, Any]:
    """Score non-human responses into a report that is honest about being one.

    This is how the listening pipeline gets exercised without human data. The report
    declares ``evidence_tier: contract_fixture`` and sets
    ``explicitly_not_human_responses``, which ``validate_listening_report`` accepts
    as a well-formed contract while refusing to treat as empirical evidence. Any
    other way of producing a report without listeners would either fail validation
    or, worse, pass it while claiming something untrue.

    Coverage is not required to be complete here, because a dry run's purpose is to
    check the wiring rather than to represent a full session.
    """

    return ingest_listening_responses(
        assignment,
        responses,
        evidence_tier=DRY_RUN_EVIDENCE_TIER,
        explicitly_not_human_responses=True,
        analysis_notes=analysis_notes,
    )


__all__ = [
    "DRY_RUN_EVIDENCE_TIER",
    "EMPIRICAL_EVIDENCE_TIER",
    "M6_LISTENING_ASSIGNMENT_SCHEMA_VERSION",
    "MINIMUM_EMPIRICAL_PARTICIPANTS",
    "TRIAL_ROLES",
    "ListeningProtocol",
    "build_dry_run_report",
    "build_listening_assignment",
    "ingest_listening_responses",
]
