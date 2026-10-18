"""`puresound.audio.pcs`: the band table reproduces the authors' bin assignment,
and the stretch is an identity at gamma 1 and a contrast increase otherwise."""

import numpy as np
import torch

from puresound.audio.pcs import pcs_gamma, stretch_magnitude, stretch_magnitude_np


def _authors_512_table():
    """RoyChao19477/PCS `PCS.py`, verbatim bin ranges for a 512-point STFT at 16 kHz."""
    g = np.ones(257)
    g[3:6] = 1.070175439
    g[6:9] = 1.182456140
    g[9:12] = 1.287719298
    g[12:138] = 1.4
    g[138:166] = 1.322807018
    g[166:200] = 1.238596491
    g[200:241] = 1.161403509
    g[241:256] = 1.077192982
    return g


def test_the_band_table_reproduces_the_published_512_point_assignment():
    assert np.array_equal(pcs_gamma(512, 16000), _authors_512_table())


def test_other_stft_sizes_follow_the_same_frequencies():
    g = pcs_gamma(1024, 16000)
    assert g.shape == (513,)
    assert g[64] == 1.4 and g[6] == 1.070175439 and g[512] == 1.0   # 1000 Hz, 93.75 Hz, 8 kHz
    assert pcs_gamma(8, 16000).tolist() == [1.0, 1.4, 1.4, 1.238596491, 1.0]


def test_the_stretch_is_an_identity_at_gamma_one_and_sharpens_peaks_otherwise():
    magnitude = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
    assert torch.allclose(stretch_magnitude(magnitude, torch.ones(5)), magnitude)
    stretched = stretch_magnitude(magnitude, torch.full((5,), 1.4))
    # monotone, and the loud bins gain more than the quiet ones: contrast goes up
    gains = stretched / magnitude
    assert torch.all(stretched[1:] > stretched[:-1])
    assert torch.all(gains[1:] > gains[:-1])
    assert np.allclose(stretch_magnitude_np(magnitude.numpy(), np.full(5, 1.4)), stretched.numpy())
