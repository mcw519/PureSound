"""Activity/VAD losses and the energy labeler that feeds them targets."""

import pytest
import torch

from puresound.audio.vad import EnergyVADLabeler
from puresound.nnet.loss import (
    BackgroundVADHeadBCELoss,
    VADActivityLoss,
    VADHeadBCELoss,
)


def _reference():
    ref = torch.zeros(1, 1600)
    ref[:, 800:] = 0.2 * torch.sin(torch.linspace(0, 40, 800))
    return ref


@pytest.mark.parametrize("external_target", [False, True], ids=["own-labels", "vad_target"])
def test_vad_activity_loss_penalizes_false_activity(external_target):
    loss_func = VADActivityLoss(frame_length=160, hop_length=80, activity_threshold_db=-35,
                                false_positive_weight=2.0)
    ref = _reference()
    kwargs = {}
    if external_target:
        labeler = EnergyVADLabeler(frame_length=160, hop_length=80, activity_threshold_db=-35)
        kwargs["vad_target"] = labeler(ref, sample_rate=16000).unsqueeze(0)
    enh_bad = ref.clone()
    enh_bad[:, :800] = 0.2

    good, bad = loss_func(ref.clone(), ref, **kwargs), loss_func(enh_bad, ref, **kwargs)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert good < bad


def test_vad_activity_loss_can_require_an_external_target():
    loss_func = VADActivityLoss(frame_length=160, hop_length=80, require_vad_target=True)
    ref = torch.zeros(1, 1600)
    with pytest.raises(ValueError, match="requires `vad_target`"):
        loss_func(ref, ref)
    assert torch.isfinite(loss_func(ref, ref, vad_target=torch.zeros(1, 19)))


def test_vad_head_bce_can_balance_imbalanced_frames():
    # At a constant zero logit both classes have BCE=log(2), so balancing
    # preserves that value while changing the gradient contribution by class.
    logits = torch.zeros(1, 10)
    target = torch.tensor([[1.0, 1.0] + [0.0] * 8])
    balanced = VADHeadBCELoss(balance_per_batch=True)
    unbalanced = VADHeadBCELoss(balance_per_batch=False)

    assert torch.isclose(balanced(logits, target), torch.tensor(0.6931472), atol=1e-5)
    assert torch.isclose(unbalanced(logits, target), torch.tensor(0.6931472), atol=1e-5)

    all_positive = torch.full_like(logits, 4.0)
    all_negative = torch.full_like(logits, -4.0)
    assert torch.isclose(
        balanced(all_positive, target), balanced(all_negative, target), atol=1e-5
    )
    assert unbalanced(all_positive, target) > unbalanced(all_negative, target)


def test_missing_head_errors_name_the_head_the_loss_actually_wants():
    # No shipped backbone populates `last_background_vad_logits`, so the
    # background loss reaches its None-logits guard whenever it is configured.
    # It must not send the reader off to enable the FOREGROUND head, which they
    # may already have on.
    target = torch.zeros(1, 10)

    with pytest.raises(ValueError, match=r"`last_vad_logits`.*vad_head"):
        VADHeadBCELoss()(None, target)

    with pytest.raises(
        ValueError, match=r"`last_background_vad_logits`.*background_vad_head"
    ):
        BackgroundVADHeadBCELoss()(None, target)

    with pytest.raises(ValueError, match=r"requires `background_vad_target`"):
        BackgroundVADHeadBCELoss()(torch.zeros(1, 10), None)


def _burst():
    torch.manual_seed(0)
    x = torch.zeros(1, 16000 * 2)
    x[:, 4000:12000] = torch.randn(8000)                          # 0.5 s burst, rest silence
    return x


@pytest.mark.parametrize("eps_mode", ["absolute", "relative"])
def test_the_energy_labeler_on_a_quiet_row(eps_mode):
    """The default fixed clamp marks every frame active once the row's peak is
    below about -40 dBFS; ``eps_mode="relative"`` holds the label at any level."""
    labeler = EnergyVADLabeler() if eps_mode == "absolute" else EnergyVADLabeler(eps_mode=eps_mode)
    loud = labeler(_burst()).mean().item()
    quiet = labeler(_burst() * 10 ** (-60 / 20)).mean().item()
    assert loud < 0.6
    if eps_mode == "absolute":
        assert quiet > 0.95
    else:
        assert abs(loud - quiet) < 0.02
        with pytest.raises(ValueError):
            EnergyVADLabeler(eps_mode="bogus")
