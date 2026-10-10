"""The statistics exist to stop a claim bigger than the measurement supporting it."""

import numpy as np
import pytest

from puresound.evaluation.statistics import (
    Interval,
    block_summary,
    bootstrap_ci,
    paired_bootstrap_ci,
    verdict,
    wilcoxon_p,
)


@pytest.fixture
def paired_samples():
    rng = np.random.default_rng(0)
    baseline = rng.normal(3.0, 0.5, 300)
    return baseline, rng


@pytest.mark.parametrize(
    "shift, spread, higher, lower",
    [
        (0.12, 0.05, "pass", "fail"),
        (-0.12, 0.05, "fail", "pass"),
        (0.0, 0.5, "no-resolution", "no-resolution"),
    ],
    ids=["improvement", "regression", "noise-of-the-same-size"],
)
def test_a_real_effect_resolves_in_its_direction_and_noise_does_not(
    paired_samples, shift, spread, higher, lower
):
    baseline, rng = paired_samples
    treatment = baseline + shift + rng.normal(0, spread, baseline.size)

    interval = paired_bootstrap_ci(treatment, baseline, aggregate=np.mean)
    assert interval.resolves == (higher != "no-resolution")
    if shift:
        assert abs(interval.point - shift) < 0.02
    assert verdict(interval, direction="higher_is_better") == higher
    assert verdict(interval, direction="lower_is_better") == lower


def test_pairing_is_what_makes_the_effect_visible(paired_samples):
    """Resampling the two sides independently reports the metric's spread, not the
    difference's -- which is how a real effect gets called noise."""
    baseline, rng = paired_samples
    treatment = baseline + 0.12 + rng.normal(0, 0.05, baseline.size)

    paired = paired_bootstrap_ci(treatment, baseline, aggregate=np.mean)
    unpaired = bootstrap_ci(treatment, aggregate=np.mean)
    assert (paired.high - paired.low) < (unpaired.high - unpaired.low) / 3


def test_tolerance_absorbs_a_small_resolved_regression_but_never_an_unresolved_one(paired_samples):
    baseline, rng = paired_samples
    worse = baseline - 0.02 + rng.normal(0, 0.01, baseline.size)
    interval = paired_bootstrap_ci(worse, baseline, aggregate=np.mean)

    assert verdict(interval, direction="higher_is_better") == "fail"
    assert verdict(interval, direction="higher_is_better", tolerance=0.05) == "pass"
    unresolved = Interval(point=0.0, low=-0.01, high=0.01, n=50)
    assert verdict(unresolved, direction="higher_is_better", tolerance=1.0) == "no-resolution"


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: paired_bootstrap_ci([1.0, 2.0], [1.0]), "equal lengths"),
        (lambda: bootstrap_ci([]), "empty sample"),
        (lambda: verdict(Interval(1.0, 0.5, 1.5, 10), direction="bigger"), "direction"),
    ],
    ids=["mismatched-lengths", "empty-sample", "unknown-direction"],
)
def test_malformed_input_is_an_error_not_a_number(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_bootstrap_is_deterministic_for_a_seed_and_one_item_has_no_width(paired_samples):
    baseline, _ = paired_samples
    first = bootstrap_ci(baseline, seed=7)
    second = bootstrap_ci(baseline, seed=7)
    assert (first.low, first.high) == (second.low, second.high)

    single = bootstrap_ci([2.5])
    assert (single.point, single.low, single.high, single.n) == (2.5, 2.5, 2.5, 1)


@pytest.mark.parametrize(
    "treatment, baseline",
    [([0.5], [0.0]), ([0.5, float("nan"), 0.7], [0.0, 0.0, 0.0])],
    ids=["one-item", "non-finite-score"],
)
def test_an_interval_without_a_spread_to_read_never_resolves(treatment, baseline):
    """One item gives a zero-width interval and a NaN gives none at all; reading
    either as an interval that excludes zero would promote a version on nothing."""
    interval = paired_bootstrap_ci(treatment, baseline, aggregate=np.mean)
    assert not interval.resolves
    for direction in ("higher_is_better", "lower_is_better"):
        assert verdict(interval, direction=direction) == "no-resolution"


def test_block_reports_the_range_not_a_standard_deviation():
    summary = block_summary([-0.1, -0.3, -0.2, -0.5, -0.25])
    assert summary.point == pytest.approx(-0.25)
    assert summary.spread == pytest.approx(0.4)
    assert summary.n_checkpoints == 5


def test_wilcoxon_detects_a_consistent_shift_and_not_an_identical_sequence(paired_samples):
    baseline, rng = paired_samples
    treatment = baseline + 0.12 + rng.normal(0, 0.05, baseline.size)
    assert wilcoxon_p(treatment, baseline) < 1e-6
    assert wilcoxon_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
