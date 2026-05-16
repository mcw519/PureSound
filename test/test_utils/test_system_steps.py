from pathlib import Path

import torch

from puresound.system.siso import EncDecMaskBase


class EchoSystem(EncDecMaskBase):
    def forward(self, noisy):
        return noisy


def _system():
    system = EchoSystem(
        encoder=torch.nn.Identity(),
        feats=torch.nn.Identity(),
        backbone=torch.nn.Identity(),
    )
    system.log = lambda *args, **kwargs: None
    system.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])
    return system


def test_siso_training_and_validation_steps_update_losses():
    system = _system()
    batch = {
        "noisy_speech": torch.ones(2, 8),
        "clean_speech": torch.zeros(2, 8),
    }

    train_out = system.training_step(batch, 0)
    valid_out = system.validation_step(batch, 0)

    assert torch.isclose(train_out["loss"], torch.tensor(1.0))
    assert torch.isclose(valid_out["loss"], torch.tensor(1.0))
    assert system.puresound_logging.average("epoch_train_loss") == 1.0


def test_siso_predict_step_saves_enhanced_audio(tmp_path):
    system = _system()
    system.register_proc_output_folder(str(tmp_path))
    batch = {
        "noisy_speech": torch.zeros(1, 160),
        "sr": torch.tensor(16000),
        "name": ["utt"],
    }

    system.predict_step(batch, 0)

    assert Path(tmp_path / "utt.wav").is_file()
