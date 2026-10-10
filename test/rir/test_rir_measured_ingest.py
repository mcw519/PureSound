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
    align_measured_item,
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


def _align(signal, distance_m, policy=None):
    return align_measured_channel(
        signal,
        SAMPLE_RATE,
        distance_m=distance_m,
        sound_speed_m_s=SPEED,
        policy=policy or MeasuredAlignmentPolicy(),
    )


def _spurious_early_arrival(seed):
    # An isolated arrival well before the response, separated by quiet: the
    # anchor cannot be the first thing that arrived.
    signal = _decaying_rir(direct_sample=1200, seed=seed, noise_relative=1e-6)
    signal[200] += 0.02
    return signal


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


def test_alignment_moves_the_response_start_to_the_geometric_arrival_intact():
    """The mute may only ever consume pre-onset signal.

    Ramping in *at* the arrival would attenuate the direct peak itself and
    silently discard about half of a near-field channel's energy while every
    gate still reported success.
    """
    policy = MeasuredAlignmentPolicy()
    distance = 3.0
    target = _geometric_sample(distance)
    # A corpus that referenced its RIR to the direct arrival: no leading delay.
    published = _decaying_rir(direct_sample=0, seed=1)

    aligned, record = _align(published, distance, policy)

    assert record.status == "aligned"
    assert record.geometric_arrival_sample == target
    # Silence before the arrival is the whole point, and it must be exact rather
    # than merely small: QC's gate sits at 1e-7 of the channel peak.
    assert not np.any(aligned[:target])
    assert record.prearrival_relative_db_after is None
    onset = target + policy.fade_in_samples
    assert np.argmax(np.abs(aligned)) == onset
    assert record.removed_energy_fraction < 1e-3
    np.testing.assert_allclose(
        aligned[onset : onset + 4000], published[:4000], atol=0.0, rtol=0.0
    )


def test_alignment_leaves_a_corpus_that_already_has_the_delay_alone():
    """A shared emission origin needs no correction.

    A corpus whose onsets already track source distance carries the propagation
    delay in the data. An estimator that wants to move such a channel is not
    finding the direct path, and nothing it does to the other corpora can be
    trusted. Only the deliberate fade offset may remain.
    """
    policy = MeasuredAlignmentPolicy()
    for distance in (0.5, 1.7, 4.2):
        published = _decaying_rir(direct_sample=_geometric_sample(distance), seed=2)

        _aligned, record = _align(published, distance, policy)

        assert record.status == "aligned"
        assert record.shift_samples == policy.fade_in_samples, distance


def test_forward_onset_scan_finds_a_direct_path_weaker_than_a_later_reflection():
    """Regression guard: the onset scan must run forward, not back from the peak.

    Searching backward from the peak returns the last quiet sample before it, so a
    response whose loudest arrival is a reflection gets its start placed after the
    direct path, and every earlier sample is then treated as pre-arrival energy to
    be muted.
    """
    policy = MeasuredAlignmentPolicy()
    direct = _decaying_rir(direct_sample=0, length=8000, seed=4) * 0.2
    reflection = np.zeros(8000)
    reflection[300:] = _decaying_rir(direct_sample=0, length=7700, seed=5)
    published = direct + reflection

    assert np.argmax(np.abs(published)) == 300

    aligned, record = _align(published, 1.0, policy)

    assert record.detected_onset_sample < 300
    assert record.removed_energy_fraction < 1e-2
    # The reflection keeps its 300-sample spacing from the direct path.
    target = record.geometric_arrival_sample + policy.fade_in_samples
    assert np.argmax(np.abs(aligned)) == target + 300


@pytest.mark.parametrize(
    "signal,distance_m,policy,reason",
    [
        # A separated earlier arrival means the anchor is a reflection. A gap is
        # what distinguishes this from the direct path's own rising edge; level
        # cannot, because the pre-onset region legitimately holds that edge 40 dB
        # or more above the noise floor.
        (_spurious_early_arrival(6), 1.0, None, "earlier_arrival"),
        (
            _decaying_rir(direct_sample=0, seed=7),
            8.0,
            MeasuredAlignmentPolicy(maximum_delay_ms=1.0, maximum_advance_ms=1.0),
            "implausible_shift",
        ),
        (np.zeros(4000), 1.0, None, "silent_channel"),
        (_decaying_rir(direct_sample=0, seed=8), float("nan"), None, "invalid_distance"),
    ],
    ids=["earlier_arrival", "implausible_shift", "silent", "undistanced"],
)
def test_alignment_rejects_a_channel_it_cannot_anchor(signal, distance_m, policy, reason):
    _aligned, record = _align(signal, distance_m, policy)

    assert record.status == "rejected"
    assert reason in record.rejection_reasons


def test_a_channel_without_a_distance_is_rejected_and_reported_as_strict_json(tmp_path):
    """A missing or null distance is a rejection, and the ingest report must hold it.

    The report is written with ``allow_nan=False`` after the whole bank is built,
    so a NaN distance in a rejection record would lose the report at the very end.
    """
    channels = [_decaying_rir(direct_sample=0, seed=seed) for seed in (31, 32)]
    _write_view_item(tmp_path, "probe_room0_0000", channels, [1.0, 2.0])
    metadata_path = tmp_path / "probe_room0_0000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scene"]["channel_map"][0]["distance_m"] = None
    del metadata["scene"]["channel_map"][1]["distance_m"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (item,) = scan_measured_corpus_view(tmp_path)

    alignment = align_measured_item(item)

    assert alignment.status == "rejected"
    assert alignment.rejection_reasons == ("invalid_distance",)
    records = [channel.to_dict() for channel in alignment.channels]
    json.dumps(records, allow_nan=False)
    assert [record["distance_m"] for record in records] == [None, None]


@pytest.mark.parametrize("rt60", [float("nan"), float("inf")])
def test_a_non_finite_published_rt60_is_treated_as_absent(tmp_path, rt60):
    """Strict-JSON metadata cannot hold it, and the item is otherwise usable."""
    channels = [_decaying_rir(direct_sample=0, seed=seed) for seed in (41, 42)]
    _write_view_item(tmp_path, "probe_room0_0000", channels, [1.0, 2.0])
    metadata_path = tmp_path / "probe_room0_0000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scene"]["rt60"] = rt60
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    (item,) = scan_measured_corpus_view(tmp_path)

    assert item.rt60_s is None


def test_alignment_treats_digital_silence_after_the_direct_path_as_prominent():
    """A direct path with nothing after it has no reverberation to be buried in.

    The prominence reading is undefined there rather than weak; the channel must
    be judged on the other gates, not crash the ingest.
    """
    gated = np.zeros(8000)
    gated[100] = 0.5

    _aligned, record = _align(gated, 1.0)

    assert record.status == "aligned"
    assert record.direct_prominence_db is None
    assert "weak_direct_path" not in record.rejection_reasons


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

    This is the shape faded published records take, and the reason a measured
    bank always carries some quarantine: QC rejects it on tail energy and on
    C50/C80, which is the right answer, so the release has to take a pruned copy
    rather than have the gate loosened.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(4000) * 1e-7
    signal[0] = 0.5
    return signal


def _manifest(bank):
    return RIRBankManifest.from_json(
        (bank / "rir_bank_manifest.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def measured_bank(tmp_path_factory):
    """A clean measured bank and its QC-passed copy, built once."""
    root = tmp_path_factory.mktemp("measured")
    _write_corpus_view(root / "view", corpus="probe", rooms=20, per_room=2)
    report = build_measured_m6_bank(
        root / "view",
        root / "bank",
        code_revision="test-revision",
        bank_id="test-measured",
    )
    prune_bank_to_qc_passed(root / "bank", root / "pruned")
    return {"report": report, "bank": root / "bank", "pruned": root / "pruned"}


@pytest.fixture(scope="module")
def synthetic_bank(tmp_path_factory):
    """A QC'd synthetic bank from the reproducible generator, built once."""
    root = tmp_path_factory.mktemp("synthetic") / "synth"
    _run_generator(root, workers=1)
    run_rir_bank_qc(root)
    return root


def test_scan_reads_distances_filters_by_corpus_and_samples_across_rooms(tmp_path):
    """A prefix of a sorted corpus is a prefix of one room, which collapses splits."""
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="alpha", rooms=4, per_room=5)
    _write_corpus_view(view, corpus="beta", rooms=2, per_room=2, seed=500)

    every = scan_measured_corpus_view(view)
    only_alpha = scan_measured_corpus_view(view, corpora=["alpha"])
    limited = scan_measured_corpus_view(view, corpora=["alpha"], limit_per_corpus=4)

    assert {item.corpus for item in every} == {"alpha", "beta"}
    assert {item.corpus for item in only_alpha} == {"alpha"}
    assert all(len(item.channel_map) == 3 for item in every)
    assert len(limited) == 4
    assert len({item.room for item in limited}) == 4


def test_ingest_makes_every_channel_causal_and_publishes_a_qc_release(measured_bank):
    """The measured corpora fail item QC only on the time origin; fix it and they pass.

    QC must also key on the same sound speed the alignment used, or nothing lines
    up. No reference corpus publishes air temperature or humidity, so that number
    is an assumption; writing it into every scene makes the assumption travel with
    the data instead of living in the ingest script.
    """
    report = measured_bank["report"]
    bank = measured_bank["bank"]

    assert report.counts["aligned_items"] == report.counts["source_items"]
    assert report.qc_audit_valid is True
    assert report.qc_summary["counts"]["passed"] == report.counts["aligned_items"]

    manifest = _manifest(bank)
    assert manifest.items
    for item in manifest.items:
        assert item.origin == "real"
        assert item.signal_variant == "measured"
        assert item.level_policy == "native_measured"
        report_payload = json.loads(
            (bank / item.qc_report_path).read_text(encoding="utf-8")
        )
        for channel in report_payload["channels"]:
            geometry = channel["geometry"]
            assert geometry["prearrival_relative_peak"] == 0.0
            assert geometry["arrival_error_ms"] <= 1.0

    metadata = json.loads(
        (bank / manifest.items[0].metadata_path).read_text(encoding="utf-8")
    )
    environment = metadata["scene"]["environment"]
    assert environment["sound_speed_m_s"] == pytest.approx(SPEED)
    assert "assumed" in environment["provenance"]
    assert metadata["measured_alignment"]["policy_id"] == MEASURED_TIME_ORIGIN_POLICY
    assert metadata["measured_source"]["audio_sha256"]


def test_ingest_reports_rejections_instead_of_forcing_them_through(tmp_path):
    view = tmp_path / "view"
    _write_corpus_view(view, corpus="probe", rooms=20, per_room=2)
    _write_view_item(
        view,
        "probe_room0_0009",
        [
            _spurious_early_arrival(900),
            _decaying_rir(direct_sample=1200, seed=901),
            _decaying_rir(0, seed=902),
        ],
        [0.8, 2.0, 3.4],
    )

    report = build_measured_m6_bank(
        view, tmp_path / "bank", code_revision="test-revision"
    )

    assert report.counts["rejected_items"] == 1
    assert report.rejections[0]["item_id"] == "probe_room0_0009"
    assert report.rejections[0]["reasons"] == ["earlier_arrival"]
    assert "probe_room0_0009" not in {
        item.item_id for item in _manifest(tmp_path / "bank").items
    }


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


def test_release_without_a_qc_measured_variant_stays_blocked(synthetic_bank, tmp_path):
    release = build_m6_variant_release(
        synthetic_bank, tmp_path / "release", release_id="test-release"
    )
    recipes = {recipe.recipe_id: recipe for recipe in release.recipes}

    assert audit_m6_variant_release(tmp_path / "release")["valid"] is True
    assert recipes["real_native"].status == "blocked"
    assert recipes["mixed_calibrated_real"].status == "blocked"
    assert _recipe_semantics_valid(release) is False

    (tmp_path / "not_a_bank").mkdir()
    with pytest.raises((ValueError, OSError)):
        build_m6_variant_release(
            synthetic_bank,
            tmp_path / "release_bad",
            measured_bank_root=tmp_path / "not_a_bank",
        )


def test_measured_variant_makes_the_real_and_mixed_recipes_ready(
    synthetic_bank, measured_bank, tmp_path
):
    release = build_m6_variant_release(
        synthetic_bank,
        tmp_path / "release",
        release_id="test-release",
        measured_bank_root=measured_bank["pruned"],
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

    with pytest.raises(ValueError, match="positive synthetic and real weight"):
        build_m6_variant_release(
            synthetic_bank,
            tmp_path / "release_single_origin",
            measured_bank_root=measured_bank["pruned"],
            mixed_origin_weights={"synthetic": 1.0},
        )
