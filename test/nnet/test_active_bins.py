"""What `ActiveBinLogMagLoss` must do: count log-magnitude error only where the
clean target has speech, so the large errors in silence and noise-only bins
cannot outvote the speech interior.
"""

import pytest
import torch

from puresound.nnet.loss import ActiveBinLogMagLoss

SR = 16000


def _voiced(seed: int, n: int = SR, level: float = 0.3) -> torch.Tensor:
    """Harmonic tone burst: silence for the first third, then a 5-harmonic comb."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    tone = sum(
        torch.sin(2 * torch.pi * f * t + float(torch.rand(1, generator=generator)) * 6.28)
        for f in (150.0, 300.0, 450.0, 600.0, 750.0)
    )
    tone[: n // 3] = 0.0
    return (level * tone / tone.abs().max()).unsqueeze(0)


def test_only_error_inside_speech_is_counted():
    loss = ActiveBinLogMagLoss()
    target = _voiced(1)
    assert float(loss(target, target)) == pytest.approx(0.0, abs=1e-6)

    # Noise only where the target is silent is outside the active set. Stay
    # clear of the onset: a frame straddling it holds target energy and is
    # legitimately active.
    polluted = target.clone()
    span = SR // 3 - 1024
    polluted[..., :span] += 0.05 * torch.randn(1, span)
    assert float(loss(polluted, target)) == pytest.approx(0.0, abs=1e-4)

    shaved = target.clone()
    shaved[..., SR // 3 :] *= 0.5  # -6 dB on every active bin
    assert 5.0 < float(loss(shaved, target)) < 7.0  # in dB units

    enh = (torch.randn(2, SR) * 0.1).requires_grad_(True)
    loss(enh, _voiced(6).repeat(2, 1)).backward()
    assert torch.isfinite(enh.grad).all()


def test_the_same_shave_costs_the_same_at_any_level():
    """The property energy-weighted losses lack: a quiet utterance's error is
    worth as much as a loud one's."""
    loss = ActiveBinLogMagLoss()
    loud_t = _voiced(3, level=0.5)
    quiet_t = _voiced(3, level=0.005)
    assert float(loss(quiet_t * 0.7, quiet_t)) == pytest.approx(
        float(loss(loud_t * 0.7, loud_t)), rel=0.02
    )


def test_under_weight_penalises_undershoot_more_than_overshoot():
    target = _voiced(4)
    down, up = target.clone(), target.clone()
    down[..., SR // 3 :] *= 0.5
    up[..., SR // 3 :] *= 2.0
    symmetric = ActiveBinLogMagLoss(under_weight=1.0)
    leaning = ActiveBinLogMagLoss(under_weight=2.0)
    assert float(symmetric(down, target)) == pytest.approx(float(symmetric(up, target)), rel=0.05)
    assert float(leaning(down, target)) == pytest.approx(2.0 * float(leaning(up, target)), rel=0.05)


@pytest.mark.parametrize(
    "enh, target",
    [
        # A relative threshold alone would make every bin of a silent row active
        # (peak == floor), costing ~100 dB per bin.
        (torch.randn(2, SR) * 0.1, torch.zeros(2, SR)),
        (torch.randn(1, 200), torch.randn(1, 200)),  # shorter than one frame
    ],
    ids=["silent-target", "shorter-than-a-frame"],
)
def test_a_row_with_nothing_to_score_contributes_zero_with_a_finite_gradient(enh, target):
    enh = enh.clone().requires_grad_(True)
    value = ActiveBinLogMagLoss()(enh, target)
    value.backward()
    assert float(value.detach()) == 0.0 and torch.isfinite(enh.grad).all()


def test_a_silent_row_is_dropped_not_averaged_in():
    voiced = _voiced(5)[0]
    target = torch.stack([torch.zeros(SR), voiced])
    enh = torch.stack([torch.randn(SR) * 0.1, voiced])
    assert float(ActiveBinLogMagLoss()(enh, target)) == pytest.approx(0.0, abs=1e-5)


def test_the_call_contract():
    """Default module inputs, a channel axis accepted, unusable settings refused."""
    assert ActiveBinLogMagLoss.required_inputs == ("enhanced", "target")
    loss = ActiveBinLogMagLoss()
    a, b = _voiced(7), _voiced(8)
    assert float(loss(a.unsqueeze(1), b.unsqueeze(1))) == pytest.approx(float(loss(a, b)), abs=1e-6)
    for kwargs in ({"dynamic_range_db": 0.0}, {"under_weight": 0.0}):
        with pytest.raises(ValueError):
            ActiveBinLogMagLoss(**kwargs)
