"""The target-absent SDR path. Its default objective is absolute residual energy
on a SUM, so its eps floor and its value move with row length and level;
``mode="mean"`` removes the length dependence and ``mode="relative"`` scores
the reduction against the noisy input instead of the level."""

import math

import pytest
import torch

from puresound.nnet.loss.sdr import SDRLoss, inactive_sdr_loss

SR = 16000


def _residual(seconds, generator):
    return torch.randn(1, seconds * SR, generator=generator) * 1e-3


@pytest.mark.parametrize("mode, expected_shift_db", [("absolute", 10.0), ("mean", 0.0)])
def test_the_inactive_score_of_a_residual_across_row_lengths(mode, expected_shift_db):
    """A 10x longer row of the same residual: +10 dB on the SUM, 0 on the mean."""
    kwargs = {} if mode == "absolute" else {"mode": mode}
    g = torch.Generator().manual_seed(1)
    r3 = inactive_sdr_loss(_residual(3, g), torch.zeros(1, 3 * SR), reduction=False, **kwargs)
    r30 = inactive_sdr_loss(_residual(30, g), torch.zeros(1, 30 * SR), reduction=False, **kwargs)
    assert abs((r30 - r3).item() - expected_shift_db) < 0.1

    if mode == "absolute":  # an all-zero residual sits on the 1e-8 floor at any length
        for n in (3 * SR, 30 * SR):
            v = inactive_sdr_loss(torch.zeros(1, n), torch.zeros(1, n), reduction=False).item()
            assert math.isclose(v, -80.0, abs_tol=1e-3)


def test_the_relative_inactive_score_is_the_reduction_not_the_level():
    torch.manual_seed(0)
    noisy = torch.randn(1, SR)
    enh = noisy * 10 ** (-20 / 20)                               # 20 dB of suppression
    for gain in (1.0, 10 ** (-30 / 20)):                         # loud and quiet rows
        v = inactive_sdr_loss(enh * gain, torch.zeros_like(enh), reduction=False,
                              mode="relative", noisy=noisy * gain).item()
        assert abs(v + 20.0) < 0.2


def test_sdrloss_relative_mode_declares_and_uses_the_batch():
    loss = SDRLoss(scaled=False, scale_dependent=True, zero_mean=True, inactive_mode="relative")
    assert "batch" in loss.required_inputs
    torch.manual_seed(0)
    noisy = torch.randn(2, SR)
    enh = torch.stack([noisy[0] * 0.1, noisy[1]])                 # row0 suppressed 20 dB, row1 untouched
    v = loss(enh, torch.zeros_like(enh), inactive_labels=torch.tensor([True, True]),
             batch={"noisy_speech": noisy})
    assert -12 < v.item() < -8                                     # mean of (-20, 0)
    assert "batch" not in SDRLoss(scaled=False, scale_dependent=True, zero_mean=True).required_inputs
    with pytest.raises(ValueError):
        SDRLoss(inactive_mode="bogus")
