import torch
import pytest

from puresound.nnet.loss import VADActivityLoss
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

    fallback_loss = loss_func(enh_bad, ref)
    external_label_loss = loss_func(enh_bad, ref, vad_target=vad_target)

    assert torch.isfinite(external_label_loss)
    assert torch.allclose(fallback_loss, external_label_loss)


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
