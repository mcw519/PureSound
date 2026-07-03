from pathlib import Path

import torch

from puresound.system.siso import EncDecMaskBase


class EchoSystem(EncDecMaskBase):
    def forward(self, noisy, query_distance=None):
        return noisy


def _system(loss_funcs=None):
    system = EchoSystem(
        encoder=torch.nn.Identity(),
        feats=torch.nn.Identity(),
        backbone=torch.nn.Identity(),
    )
    system.logged = []
    system.log = lambda *args, **kwargs: system.logged.append((args, kwargs))
    if loss_funcs is None:
        loss_funcs = [torch.nn.L1Loss()]
    system.register_loss_func(torch.nn.ModuleList(loss_funcs), [1.0] * len(loss_funcs))
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


def test_siso_step_logs_do_not_sync_aux_losses():
    # Train step losses back the progress bar and must stay per-rank
    # (sync_dist=False): an all-reduce there deadlocks DDP because the bar's
    # refresh isn't in lockstep across ranks. Validation aux losses are
    # epoch-aggregated (on_step=False, on_epoch=True) and must sync
    # (sync_dist=True) so the logged value averages across all ranks.
    system = _system([torch.nn.L1Loss(), torch.nn.MSELoss()])
    system.verbose = True
    batch = {
        "noisy_speech": torch.ones(2, 8),
        "clean_speech": torch.zeros(2, 8),
    }

    system.training_step(batch, 0)
    system.validation_step(batch, 0)

    train_aux_logs = [
        (args, kwargs) for args, kwargs in system.logged if args[0].startswith("train_step_loss_")
    ]
    valid_aux_logs = [
        (args, kwargs) for args, kwargs in system.logged if args[0].startswith("valid_step_loss_")
    ]
    assert train_aux_logs
    assert valid_aux_logs
    assert all(kwargs["sync_dist"] is False for _, kwargs in train_aux_logs)
    assert all(kwargs["sync_dist"] is True for _, kwargs in valid_aux_logs)


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
