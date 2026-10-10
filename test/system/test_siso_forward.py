import math
from pathlib import Path

import numpy as np
import torch

from puresound.system.onset_guard import OnsetGuard
from puresound.system.presence_gate import PresenceGate
from puresound.system.siso import EncDecMaskBase

REPO_ROOT = Path(__file__).resolve().parents[2]


class EchoSystem(EncDecMaskBase):
    def forward(self, noisy):
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


def test_training_and_validation_steps_compute_and_log_losses():
    """Train step losses back the progress bar and must stay per-rank
    (sync_dist=False): an all-reduce there deadlocks DDP because the bar's
    refresh is not in lockstep across ranks. Validation aux losses are
    epoch-aggregated and must sync (sync_dist=True) so the logged value averages
    across all ranks."""
    system = _system([torch.nn.L1Loss(), torch.nn.MSELoss()])
    system.verbose = True
    batch = {
        "noisy_speech": torch.ones(2, 8),
        "clean_speech": torch.zeros(2, 8),
    }

    train_out = system.training_step(batch, 0)
    valid_out = system.validation_step(batch, 0)

    # L1 and MSE of ones against zeros are both 1.0, weighted 1.0 each
    assert torch.isclose(train_out["loss"], torch.tensor(2.0))
    assert torch.isclose(valid_out["loss"], torch.tensor(2.0))
    assert system.puresound_logging.average("epoch_train_loss") == 2.0

    def logged(prefix):
        return [kwargs for args, kwargs in system.logged if args[0].startswith(prefix)]

    assert logged("train_step_loss_") and logged("valid_step_loss_")
    assert all(kwargs["sync_dist"] is False for kwargs in logged("train_step_loss_"))
    assert all(kwargs["sync_dist"] is True for kwargs in logged("valid_step_loss_"))


def test_predict_step_saves_enhanced_audio(tmp_path):
    system = _system()
    system.register_proc_output_folder(str(tmp_path))
    batch = {
        "noisy_speech": torch.zeros(1, 160),
        "sr": torch.tensor(16000),
        "name": ["utt"],
    }

    system.predict_step(batch, 0)

    assert Path(tmp_path / "utt.wav").is_file()


def test_post_model_stages_run_blend_then_presence_gate_then_onset_guard():
    """The order IS the mechanism.

    `dry_blend` 0.9 keeps 10% of the input, an arithmetic floor at -20 dB, and
    the presence gate is a multiplicative gain, so it has to come after:

        gate, then blend:  0.9 * (g * enh) + 0.1 * mix  ->  |out| >= 0.1 * |mix|
        blend, then gate:  g * (0.9 * enh + 0.1 * mix)  ->  |out| -> 0 as g -> 0

    The onset guard restores the dry input, so it has to come after the gate:

        guard, then gate:  p * (g * mix + (1 - g) * enh)  ->  |out| = p * |mix| at g = 1
        gate, then guard:  g * mix + (1 - g) * (p * enh)  ->  |out| = |mix| at g = 1
    """
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    recipe = load_recipe(
        REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn.yaml",
        expected_task="voice_isolation",
    )
    model = init_siso_model(recipe.model).eval()  # random weights are fine
    rng = np.random.default_rng(0)
    # stationary floor: nothing for the guard to confirm, so it must stay dry
    wav = torch.from_numpy((rng.standard_normal((1, 32000)) * 0.05).astype("float32"))

    # bias -30 => sigmoid ~ 0, so presence falls and the gain reaches its floor
    channels = recipe.model["backbone"]["backbone_args"]["channels"][-1]
    gate = PresenceGate(weight=torch.zeros(channels), bias=-30.0, b_hi=0.5, b_lo=0.1,
                        gain_floor_db=-40.0, tau_up_s=0.05, tau_dn_s=0.05)
    with torch.no_grad():
        gated = model(wav.clone(), dry_blend=0.9, presence_gate=gate)
        guarded = model(wav.clone(), dry_blend=0.9, presence_gate=gate,
                        onset_guard=OnsetGuard())

    # The stash flag the gate needs is a loan, not a setting: left on, every
    # later forward -- a training step included -- keeps a tensor nothing reads.
    assert model.backbone.stash_bottleneck is False

    n = min(gated.shape[-1], guarded.shape[-1], wav.shape[-1])
    tail = slice(n // 2, n)  # past the integrator settling

    def level(x):
        return 10 * math.log10(float(x[..., tail].square().mean())
                               / float(wav[..., tail].square().mean()))

    assert level(gated) < -21.0, (
        f"output is {level(gated):.1f} dB below input; gate-before-blend floors it at -20"
    )
    assert torch.equal(guarded[..., :n], wav[..., :n]), "the guard must undo the gate"
