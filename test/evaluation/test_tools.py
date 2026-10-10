"""The pure parts of the stage tools: banding, measured SNR, NaN handling."""

import math
from pathlib import Path

import pytest
import torch

from puresound.dataset.corpus.records import AudioRecord
from puresound.evaluation.tools.build_eval_set import (
    band_of,
    measured_snr_db,
    recipe_sha256,
    rms_dbfs,
)
from puresound.evaluation.tools.noreference import by_tag
from puresound.evaluation.tools.reference import METRICS, _finite_pairs, by_band, score_pair


def test_measured_snr_reads_what_ended_up_in_the_mixture_and_degenerate_levels_stay_finite():
    clean = torch.ones(1, 1000)
    noise = torch.full((1, 1000), 0.1)
    # signal power 1.0, noise power 0.01 -> 20 dB, whatever the config asked for.
    assert measured_snr_db(clean, clean + noise) == pytest.approx(20.0, abs=1e-4)
    assert math.isfinite(measured_snr_db(clean, clean))  # no noise at all
    assert math.isfinite(rms_dbfs(torch.zeros(1, 100)))  # silence is a floor, not -inf


@pytest.mark.parametrize(
    "snr, band", [(-3.0, "-inf..0"), (0.0, "0..5"), (12.0, "10..15"), (99.0, "15..inf")]
)
def test_bands_cover_the_whole_line_including_both_tails(snr, band):
    assert band_of(snr) == band


def test_recipe_identity_hashes_the_file_contents(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("task: noise_suppression\n", encoding="utf-8")
    first = recipe_sha256(recipe)
    recipe.write_text("task: voice_isolation\n", encoding="utf-8")
    assert recipe_sha256(recipe) != first


def test_nan_pairs_are_dropped_together_not_one_side_at_a_time():
    left, right = _finite_pairs([1.0, float("nan"), 3.0], [0.5, 0.4, float("nan")])
    assert left == [1.0] and right == [0.5]


def test_band_breakdown_groups_by_the_manifest_key():
    items = [{"snr_band": "0..5"}, {"snr_band": "0..5"}, {"snr_band": "10..15"}]
    treatment = [{"pesq_wb": 2.0}, {"pesq_wb": 2.2}, {"pesq_wb": 3.0}]
    baseline = [{"pesq_wb": 1.5}, {"pesq_wb": 1.5}, {"pesq_wb": 2.9}]

    table = by_band(items, treatment, baseline, key="snr_band", metric="pesq_wb")
    assert table["0..5"][0] == 2
    assert table["0..5"][1] == pytest.approx(0.6)
    assert table["10..15"][0] == 1


def test_tag_breakdown_counts_untagged_clips_rather_than_dropping_them():
    records = [
        AudioRecord("a", "s", "None", Path("/a.wav"), 1, 1, 1, tags={"category": "fan"}),
        AudioRecord("b", "s", "None", Path("/b.wav"), 1, 1, 1),
    ]
    treatment = [{"dnsmos_ovr": 3.0}, {"dnsmos_ovr": 2.0}]
    baseline = [{"dnsmos_ovr": 2.0}, {"dnsmos_ovr": 2.0}]

    table = by_tag(records, treatment, baseline, tag="category", score="dnsmos_ovr")
    assert table["fan"] == (1, pytest.approx(1.0))
    assert table["unknown"] == (1, 0.0)


def test_scoring_trims_a_shorter_output_and_reports_nan_for_a_degenerate_pair():
    """A model's output is not its input's length -- the STFT round trip drops a
    partial frame -- so the scorer trims rather than raising on the mismatch a
    real checkpoint always produces. An all-zero pair is scored, not raised on."""
    clean = torch.randn(1, 16000) * 0.1
    scores = score_pair(clean[..., :-128].clone(), clean, 16000)
    assert set(scores) == set(METRICS)
    assert scores["sisdr"] > 20  # a trimmed copy of the target is still the target

    degenerate = score_pair(torch.zeros(1, 16000), torch.zeros(1, 16000), 16000)
    assert set(degenerate) == set(METRICS)
