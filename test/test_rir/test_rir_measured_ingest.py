import json
import math
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from egs.rir_generation.phases.m6_bank.scripts.validate_m6_reproducible_generation import (
    _run_generator,
)
from puresound.audio.rir.bank.measured_ingest import (
    ASSUMED_MEASURED_ENVIRONMENT,
    MEASURED_TIME_ORIGIN_POLICY,
    MeasuredAlignmentPolicy,
    align_measured_channel,
    build_measured_m6_bank,
    scan_measured_corpus_view,
)
from puresound.audio.rir.bank.production import (
    REQUIRED_RECIPE_IDS,
    _recipe_semantics_valid,
)
from puresound.audio.rir.bank.qc import run_rir_bank_qc
from puresound.audio.rir.bank.release import (
    audit_m6_variant_release,
    build_m6_variant_release,
    prune_bank_to_qc_passed,
)
from puresound.audio.rir.bank.schema import RIRBankManifest

# End-to-end evidence-chain validator: builds, QCs and releases a real bank, so it
# runs for tens of seconds. Excluded by `run_repo_checks.py --suite standard`.
pytestmark = pytest.mark.slow

SAMPLE_RATE = 16000
SPEED = float(ASSUMED_MEASURED_ENVIRONMENT.sound_speed_m_s)


def _decaying_rir(
    direct_sample: int,
    *,
    length: int = 8000,
    rt60_s: float = 0.4,
    seed: int = 0,
    noise_relative: float = 1e-4,
) -> np.ndarray:
    """A causal RIR: an impulsive direct path, then exponentially decaying noise."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(length)
    time = np.arange(length) / SAMPLE_RATE
    envelope = np.zeros(length)
    tail = np.arange(direct_sample, length)
    envelope[tail] = 10.0 ** (
        -3.0 * (time[tail] - time[direct_sample]) / rt60_s
    )
    signal = signal * envelope * 0.05
    signal[:direct_sample] = 0.0
    # Below unity so the sum with noise still satisfies the native_measured peak
    # gate, as every published corpus in the reference does.
    signal[direct_sample] = 0.5
    return signal + rng.standard_normal(length) * noise_relative


def _geometric_sample(distance_m: float) -> int:
    return int(math.floor(distance_m / SPEED * SAMPLE_RATE))


def test_alignment_policy_is_versioned_content_addressed_and_json_safe():
    policy = MeasuredAlignmentPolicy()

    assert policy.policy_id == MEASURED_TIME_ORIGIN_POLICY
    assert len(policy.policy_sha256) == 64
    json.dumps(policy.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="below the channel peak"):
        replace(policy, onset_threshold_db_below_peak=0.0)
    with pytest.raises(ValueError, match="removed energy"):
        replace(policy, maximum_removed_energy_fraction=1.0)
    with pytest.raises(ValueError, match="shift bounds"):
        replace(policy, maximum_delay_ms=0.0)
    with pytest.raises(ValueError, match="unsupported measured time-origin"):
        replace(policy, policy_id="something-else")


def test_alignment_puts_the_response_start_on_the_geometric_arrival():
    policy = MeasuredAlignmentPolicy()
    distance = 3.0
    target = _geometric_sample(distance)
    # A corpus that referenced its RIR to the direct arrival: no leading delay.
    published = _decaying_rir(direct_sample=0, seed=1)

    aligned, record = align_measured_channel(
        published,
        SAMPLE_RATE,
        distance_m=distance,
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert record.status == "aligned"
    assert record.geometric_arrival_sample == target
    # Silence before the arrival is the whole point, and it must be exact rather
    # than merely small: QC's gate sits at 1e-7 of the channel peak.
    assert not np.any(aligned[:target])
    assert record.prearrival_relative_db_after is None
    assert np.argmax(np.abs(aligned)) == target + policy.fade_in_samples


def test_alignment_leaves_a_corpus_that_already_has_the_delay_alone():
    """The BRUDEX invariant: a shared emission origin needs no correction.

    BRUDEX publishes RIRs whose onset tracks source distance with slope 1.00, so
    the propagation delay is already in the data. An estimator that wants to move
    such a channel is not finding the direct path, and nothing it does to the
    other corpora can be trusted. Only the deliberate fade offset may remain.
    """
    policy = MeasuredAlignmentPolicy()
    for distance in (0.5, 1.7, 4.2):
        target = _geometric_sample(distance)
        published = _decaying_rir(direct_sample=target, seed=2)

        _aligned, record = align_measured_channel(
            published,
            SAMPLE_RATE,
            distance_m=distance,
            sound_speed_m_s=SPEED,
            policy=policy,
        )

        assert record.status == "aligned"
        assert record.shift_samples == policy.fade_in_samples, distance


def test_alignment_preserves_the_direct_path_and_the_response_after_it():
    """The mute may only ever consume pre-onset signal.

    An earlier version ramped in *at* the arrival, which attenuated the direct
    peak itself and silently discarded about half of a near-field channel's
    energy while every gate still reported success.
    """
    policy = MeasuredAlignmentPolicy()
    distance = 2.0
    published = _decaying_rir(direct_sample=0, seed=3)

    aligned, record = align_measured_channel(
        published,
        SAMPLE_RATE,
        distance_m=distance,
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert record.removed_energy_fraction < 1e-3
    onset = record.geometric_arrival_sample + policy.fade_in_samples
    np.testing.assert_allclose(
        aligned[onset : onset + 4000], published[:4000], atol=0.0, rtol=0.0
    )


def test_forward_onset_scan_finds_a_direct_path_weaker_than_a_later_reflection():
    """Regression guard: the onset scan must run forward, not back from the peak.

    Searching backward from the peak returns the last quiet sample before it, so a
    response whose loudest arrival is a reflection gets its start placed after the
    direct path, and every earlier sample is then treated as pre-arrival energy to
    be muted.
    """
    policy = MeasuredAlignmentPolicy()
    distance = 1.0
    direct = _decaying_rir(direct_sample=0, length=8000, seed=4) * 0.2
    reflection = np.zeros(8000)
    reflection[300:] = _decaying_rir(direct_sample=0, length=7700, seed=5)
    published = direct + reflection

    assert np.argmax(np.abs(published)) == 300

    aligned, record = align_measured_channel(
        published,
        SAMPLE_RATE,
        distance_m=distance,
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert record.detected_onset_sample < 300
    assert record.removed_energy_fraction < 1e-2
    # The reflection keeps its 300-sample spacing from the direct path.
    target = record.geometric_arrival_sample + policy.fade_in_samples
    assert np.argmax(np.abs(aligned)) == target + 300


def test_alignment_refuses_a_channel_whose_direct_path_is_not_the_first_arrival():
    """A separated earlier arrival means the anchor is a reflection: reject it.

    A gap is what distinguishes this from the direct path's own rising edge; level
    cannot, because the pre-onset region legitimately holds that edge 40 dB or
    more above the noise floor.
    """
    policy = MeasuredAlignmentPolicy()
    published = _decaying_rir(direct_sample=1200, seed=6, noise_relative=1e-6)
    # An isolated arrival well before the response, separated by quiet.
    published[200] += 0.02

    _aligned, record = align_measured_channel(
        published,
        SAMPLE_RATE,
        distance_m=1.0,
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert record.status == "rejected"
    assert "earlier_arrival" in record.rejection_reasons


def test_alignment_refuses_a_shift_the_record_cannot_justify():
    policy = MeasuredAlignmentPolicy(maximum_delay_ms=1.0, maximum_advance_ms=1.0)
    published = _decaying_rir(direct_sample=0, seed=7)

    _aligned, record = align_measured_channel(
        published,
        SAMPLE_RATE,
        distance_m=8.0,
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert record.status == "rejected"
    assert "implausible_shift" in record.rejection_reasons


def test_alignment_refuses_a_silent_or_undistanced_channel():
    policy = MeasuredAlignmentPolicy()

    _silent, silent = align_measured_channel(
        np.zeros(4000),
        SAMPLE_RATE,
        distance_m=1.0,
        sound_speed_m_s=SPEED,
        policy=policy,
    )
    _bad, bad = align_measured_channel(
        _decaying_rir(direct_sample=0, seed=8),
        SAMPLE_RATE,
        distance_m=float("nan"),
        sound_speed_m_s=SPEED,
        policy=policy,
    )

    assert "silent_channel" in silent.rejection_reasons
    assert "invalid_distance" in bad.rejection_reasons


def _write_view_item(root, stem, channels, distances):
    """One published-corpus record: audio plus the distance metadata beside it."""
    root.mkdir(parents=True, exist_ok=True)
    sf.write(
        root / f"{stem}.wav", np.stack(channels, axis=1), SAMPLE_RATE, subtype="FLOAT"
    )
    (root / f"{stem}.json").write_text(
        json.dumps(
            {
                "scene": {
                    "origin": "real",
                    "rt60": 0.4,
                    "channel_map": [
                        {
                            "channel": channel,
                            "label": "near_0" if channel == 0 else f"far_{channel}",
                            "distance_m": distance,
                        }
                        for channel, distance in enumerate(distances)
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_corpus_view(root, *, corpus, rooms, per_room, seed=0):
    """A published-corpus view: distance metadata, no propagation delay."""
    for room_index in range(rooms):
        for item_index in range(per_room):
            distances = [0.8 + 0.4 * room_index, 2.0 + 0.5 * item_index, 3.4]
            channels = [
                _decaying_rir(
                    direct_sample=0,
                    seed=seed + 100 * room_index + 10 * item_index + channel,
                )
                for channel in range(len(distances))
            ]
            _write_view_item(
                root, f"{corpus}_room{room_index}_{item_index:04d}", channels, distances
            )


def _tailless_channel(seed):
    """A direct path with no reverberant tail behind it.

    This is the shape DIFFRIR's faded records take, and the reason a measured bank
    always carries some quarantine: QC rejects it on tail energy and on C50/C80,
    which is the right answer, so the release has to take a pruned copy rather
    than have the gate loosened.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(4000) * 1e-7
    signal[0] = 0.5
    return signal


def test_scan_limit_samples_across_rooms_rather_than_truncating_one(tmp_path):
    """A prefix of a sorted corpus is a prefix of one room, which collapses splits."""
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=4, per_room=5)

    limited = scan_measured_corpus_view(view, limit_per_corpus=4)

    assert len(limited) == 4
    assert len({item.room for item in limited}) == 4


def test_scan_reads_distances_and_filters_by_corpus(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="alpha", rooms=2, per_room=2)
    _write_corpus_view(view, corpus="beta", rooms=2, per_room=2, seed=500)

    every = scan_measured_corpus_view(view)
    only_alpha = scan_measured_corpus_view(view, corpora=["alpha"])

    assert {item.corpus for item in every} == {"alpha", "beta"}
    assert {item.corpus for item in only_alpha} == {"alpha"}
    assert all(len(item.channel_map) == 3 for item in every)


def test_ingest_makes_every_channel_causal_and_publishes_a_qc_release(tmp_path):
    """The measured corpora fail item QC only on the time origin; fix it and they pass."""
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)

    report = build_measured_m6_bank(
        view,
        tmp_path / "bank",
        code_revision="test-revision",
        bank_id="test-measured",
    )

    assert report.counts["aligned_items"] == report.counts["source_items"]
    assert report.qc_audit_valid is True
    assert report.qc_summary["counts"]["passed"] == report.counts["aligned_items"]

    manifest = RIRBankManifest.from_json(
        (tmp_path / "bank" / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.items
    for item in manifest.items:
        assert item.origin == "real"
        assert item.signal_variant == "measured"
        assert item.level_policy == "native_measured"
        report_payload = json.loads(
            (tmp_path / "bank" / item.qc_report_path).read_text(encoding="utf-8")
        )
        for channel in report_payload["channels"]:
            geometry = channel["geometry"]
            assert geometry["prearrival_relative_peak"] == 0.0
            assert geometry["arrival_error_ms"] <= 1.0


def test_ingest_records_the_assumed_sound_speed_qc_recomputes_arrivals_from(tmp_path):
    """QC must key on the same sound speed the alignment used, or nothing lines up.

    No reference corpus publishes air temperature or humidity, so this number is an
    assumption; the point of writing it into every scene is that the assumption
    travels with the data instead of living in the ingest script.
    """
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)

    build_measured_m6_bank(view, tmp_path / "bank", code_revision="test-revision")

    manifest = RIRBankManifest.from_json(
        (tmp_path / "bank" / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (tmp_path / "bank" / manifest.items[0].metadata_path).read_text(encoding="utf-8")
    )
    environment = metadata["scene"]["environment"]

    assert environment["sound_speed_m_s"] == pytest.approx(SPEED)
    assert "assumed" in environment["provenance"]
    assert metadata["measured_alignment"]["policy_id"] == MEASURED_TIME_ORIGIN_POLICY
    assert metadata["measured_source"]["audio_sha256"]


def test_ingest_reports_rejections_instead_of_forcing_them_through(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)
    # One record carrying an isolated arrival a full 1000 samples ahead of its
    # response: the anchor cannot be the first thing that arrived.
    spurious = _decaying_rir(direct_sample=1200, seed=900, noise_relative=1e-6)
    spurious[200] += 0.02
    _write_view_item(
        view,
        "probe_room0_0009",
        [spurious, _decaying_rir(direct_sample=1200, seed=901), _decaying_rir(0, seed=902)],
        [0.8, 2.0, 3.4],
    )

    report = build_measured_m6_bank(
        view, tmp_path / "bank", code_revision="test-revision"
    )

    assert report.counts["rejected_items"] == 1
    assert report.rejections[0]["item_id"] == "probe_room0_0009"
    assert report.rejections[0]["reasons"] == ["earlier_arrival"]
    manifest = RIRBankManifest.from_json(
        (tmp_path / "bank" / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )
    assert "probe_room0_0009" not in {item.item_id for item in manifest.items}


def test_ingest_refuses_a_corpus_too_small_to_fill_every_split(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=1, per_room=2)

    with pytest.raises(ValueError, match="split left"):
        build_measured_m6_bank(view, tmp_path / "bank", code_revision="test-revision")


def test_pruning_drops_quarantined_items_and_leaves_none_behind(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)
    # A record that aligns cleanly but has no decay for QC to fit, which is how a
    # genuinely unusable published room reaches quarantine rather than rejection.
    _write_view_item(
        view,
        "probe_room0_0009",
        [_tailless_channel(910 + channel) for channel in range(3)],
        [0.8, 2.0, 3.4],
    )
    report = build_measured_m6_bank(
        view, tmp_path / "bank", code_revision="test-revision"
    )

    assert report.counts["rejected_items"] == 0
    assert report.qc_summary["counts"]["quarantined"] >= 1

    pruned = prune_bank_to_qc_passed(tmp_path / "bank", tmp_path / "pruned")

    assert pruned["dropped_item_count"] >= 1
    assert "probe_room0_0009" in pruned["dropped_item_ids"]
    assert pruned["qc_counts"]["quarantined"] == 0
    assert pruned["qc_counts"]["passed"] == pruned["kept_item_count"]
    assert pruned["qc_audit_valid"] is True


def test_release_stays_blocked_without_a_measured_variant(tmp_path):
    _run_generator(tmp_path / "synth", workers=1)
    run_rir_bank_qc(tmp_path / "synth")

    release = build_m6_variant_release(
        tmp_path / "synth", tmp_path / "release", release_id="test-release"
    )
    recipes = {recipe.recipe_id: recipe for recipe in release.recipes}

    assert audit_m6_variant_release(tmp_path / "release")["valid"] is True
    assert recipes["real_native"].status == "blocked"
    assert recipes["mixed_calibrated_real"].status == "blocked"
    assert _recipe_semantics_valid(release) is False


def test_measured_variant_makes_the_real_and_mixed_recipes_ready(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)
    build_measured_m6_bank(view, tmp_path / "measured", code_revision="test-revision")
    prune_bank_to_qc_passed(tmp_path / "measured", tmp_path / "measured_pruned")
    _run_generator(tmp_path / "synth", workers=1)
    run_rir_bank_qc(tmp_path / "synth")

    release = build_m6_variant_release(
        tmp_path / "synth",
        tmp_path / "release",
        release_id="test-release",
        measured_bank_root=tmp_path / "measured_pruned",
        mixed_origin_weights={"synthetic": 0.25, "real": 0.75},
    )
    recipes = {recipe.recipe_id: recipe for recipe in release.recipes}

    assert audit_m6_variant_release(tmp_path / "release")["valid"] is True
    assert all(recipes[name].status == "ready" for name in REQUIRED_RECIPE_IDS)
    assert _recipe_semantics_valid(release) is True
    assert recipes["real_native"].origin_weights == {"real": 1.0}
    assert recipes["mixed_calibrated_real"].origin_weights == {
        "synthetic": 0.25,
        "real": 0.75,
    }
    assert set(recipes["mixed_calibrated_real"].variant_ids) == {
        "synthetic_calibrated",
        "measured_native",
    }
    # The measured items reach the recipe index rather than only the manifest.
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "release" / recipes["real_native"].split_indexes["train"].path
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows and all(row["origin"] == "real" for row in rows)


def test_release_rejects_a_measured_bank_that_is_not_a_qc_release(tmp_path):
    _run_generator(tmp_path / "synth", workers=1)
    run_rir_bank_qc(tmp_path / "synth")
    (tmp_path / "not_a_bank").mkdir()

    with pytest.raises((ValueError, OSError)):
        build_m6_variant_release(
            tmp_path / "synth",
            tmp_path / "release",
            measured_bank_root=tmp_path / "not_a_bank",
        )


def test_release_rejects_mixed_weights_that_are_not_a_two_origin_mixture(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)
    build_measured_m6_bank(view, tmp_path / "measured", code_revision="test-revision")
    prune_bank_to_qc_passed(tmp_path / "measured", tmp_path / "measured_pruned")
    _run_generator(tmp_path / "synth", workers=1)
    run_rir_bank_qc(tmp_path / "synth")

    with pytest.raises(ValueError, match="positive synthetic and real weight"):
        build_m6_variant_release(
            tmp_path / "synth",
            tmp_path / "release",
            measured_bank_root=tmp_path / "measured_pruned",
            mixed_origin_weights={"synthetic": 1.0},
        )
