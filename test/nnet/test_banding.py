"""Perceptual banding: what it must allocate, and what it must not change."""

import pytest
import torch

from puresound.nnet.lobe.banding import (
    BandBottleneck,
    band_edges_hz,
    triangular_band_matrix,
)


@pytest.mark.parametrize("scale, min_ratio", [("erb", 10), ("mel", 5)])
def test_bands_span_the_range_narrow_low_and_wide_high(scale, min_ratio):
    """The whole point: spend units where the harmonics are. A uniform split
    has a width ratio of exactly 1 and its midpoint at 4 kHz."""
    edges = band_edges_hz(32, 50.0, 8000.0, scale)
    assert len(edges) == 33
    assert float(edges[0]) == pytest.approx(50.0, abs=1e-3)
    assert float(edges[-1]) == pytest.approx(8000.0, abs=1e-2)

    widths = edges.diff()
    assert torch.all(widths > 0)
    assert float(widths[-1] / widths[0]) > min_ratio
    assert float(edges[16]) < 8000.0 / 4


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: band_edges_hz(32, 50.0, 8000.0, "bark-ish"), "scale"),
        (lambda: band_edges_hz(0, 50.0, 8000.0), None),
        (lambda: band_edges_hz(32, 8000.0, 50.0), None),
        (lambda: band_edges_hz(32, -1.0, 8000.0), None),
        (lambda: BandBottleneck(32, 64), "not reduce"),
    ],
)
def test_a_degenerate_band_request_is_refused(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize("n_bands", [32, 48])
def test_pooling_is_an_average_and_leaves_no_band_empty(n_bands):
    """Rows sum to one, so wide and narrow bands share a scale -- including a
    band narrower than one unit, which must still claim its nearest unit."""
    matrix = triangular_band_matrix(n_bands, 64, f_min=50.0, f_max=8000.0)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(n_bands), atol=1e-5)


def test_the_bottleneck_round_trip_keeps_shape_level_and_frame_independence():
    band = BandBottleneck(64, 32)
    # Every unit gets the same total weight back; otherwise a unit at a band
    # edge returns quieter than one at a centre -- a comb nobody asked for.
    assert torch.allclose(band.expand.sum(dim=1), torch.ones(64), atol=1e-5)

    x = torch.randn(2, 8, 64, 50)
    assert band.to_bands(x).shape == (2, 8, 32, 50)
    assert band.to_units(band.to_bands(x)).shape == x.shape

    flat = torch.ones(1, 4, 64, 10) * 3.0
    assert torch.allclose(band.to_units(band.to_bands(flat)), flat, atol=1e-4)

    # No time state, so the streaming graph can band one frame at a time.
    frames = x[:1, :4, :, :7]
    per_frame = torch.cat([band.to_bands(frames[..., i : i + 1]) for i in range(7)], dim=-1)
    assert torch.allclose(band.to_bands(frames), per_frame, atol=1e-6)

    # Fixed by default, trainable only on request.
    assert not isinstance(band.pool, torch.nn.Parameter)
    assert isinstance(BandBottleneck(64, 32, learnable=True).pool, torch.nn.Parameter)
