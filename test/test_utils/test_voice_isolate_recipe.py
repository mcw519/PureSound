import torch

from puresound.nnet.loss import VADActivityLoss
from puresound.recipes import init_loss_func, init_siso_model, load_siso_recipe_config


def test_voice_isolate_skim_recipe_initializes_model_and_losses():
    config_path = "egs/voice_isolate/config/skim.yaml"
    (
        _corpus_dict,
        _trainer_dict,
        _optim_dict,
        _scheduler_dict,
        loss_dict,
        model_dict,
        _aug_speech_dict,
        _aug_noise_dict,
        aug_reverb_dict,
        _aug_speed_dict,
        _aug_ir_dict,
        _aug_src_dict,
        _aug_hpf_dict,
        _aug_volume_dict,
        vad_label_dict,
    ) = load_siso_recipe_config(config_path)

    model = init_siso_model(model_dict).eval()
    losses, loss_weights = init_loss_func(loss_dict)

    assert aug_reverb_dict["simulator"]["source_level"] is True
    assert vad_label_dict["backend"] == "silero"
    assert any(isinstance(loss, VADActivityLoss) for loss in losses)
    assert loss_weights == [1.0, 0.5, 0.1]

    with torch.no_grad():
        enhanced = model(torch.zeros(2, 4096))

    assert enhanced.shape == (2, 4096)
