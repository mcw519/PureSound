import torch
import pytest

from puresound.nnet.loss import (
    BackgroundVADHeadBCELoss,
    VADActivityLoss,
    VADHeadBCELoss,
)
from puresound.recipes import init_loss_func
from puresound.audio.vad import EnergyVADLabeler


def test_vad_activity_loss_penalizes_false_activity():
    loss_func = VADActivityLoss(
        frame_length=160,
        hop_length=80,
        activity_threshold_db=-35,
        false_positive_weight=2.0,
    )
    ref = torch.zeros(1, 1600)
    ref[:, 800:] = 0.2 * torch.sin(torch.linspace(0, 40, 800))
    enh_good = ref.clone()
    enh_bad = ref.clone()
    enh_bad[:, :800] = 0.2

    good_loss = loss_func(enh_good, ref)
    bad_loss = loss_func(enh_bad, ref)

    assert good_loss < bad_loss


def test_vad_activity_loss_can_be_loaded_from_recipe():
    loss_list, loss_weights = init_loss_func(
        [
            {
                "type": "VADActivityLoss",
                "weighted": 0.1,
                "args": {
                    "frame_length": 160,
                    "hop_length": 80,
                    "activity_threshold_db": -35,
                },
            }
        ]
    )

    assert isinstance(loss_list[0], VADActivityLoss)
    assert loss_weights == [0.1]


def test_vad_activity_loss_uses_external_vad_target():
    labeler = EnergyVADLabeler(
        frame_length=160,
        hop_length=80,
        activity_threshold_db=-35,
    )
    loss_func = VADActivityLoss(
        frame_length=160,
        hop_length=80,
        activity_threshold_db=-35,
        false_positive_weight=2.0,
    )
    ref = torch.zeros(1, 1600)
    ref[:, 800:] = 0.2 * torch.sin(torch.linspace(0, 40, 800))
    vad_target = labeler(ref, sample_rate=16000).unsqueeze(0)
    enh_bad = ref.clone()
    enh_bad[:, :800] = 0.2
    enh_good = ref.clone()

    external_bad_loss = loss_func(enh_bad, ref, vad_target=vad_target)
    external_good_loss = loss_func(enh_good, ref, vad_target=vad_target)

    assert torch.isfinite(external_bad_loss)
    assert torch.isfinite(external_good_loss)
    assert external_good_loss < external_bad_loss


def test_vad_activity_loss_can_require_external_vad_target():
    loss_func = VADActivityLoss(
        frame_length=160,
        hop_length=80,
        require_vad_target=True,
    )
    ref = torch.zeros(1, 1600)

    with pytest.raises(ValueError, match="requires `vad_target`"):
        loss_func(ref, ref)

    vad_target = torch.zeros(1, 19)
    loss = loss_func(ref, ref, vad_target=vad_target)

    assert torch.isfinite(loss)


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
