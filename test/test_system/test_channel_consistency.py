"""Tests for the channel-perturbation mask-consistency regularizer."""

import torch

from puresound.nnet import DPCRN, FeatureEncoder
from puresound.nnet.lobe.encoder import ConvEncDec
from puresound.system.siso import EncDecMaskBase


class MaskEchoSystem(EncDecMaskBase):
    """Forward stub whose 'mask' is input-dependent -> nonzero channel sensitivity."""

    def __init__(self, **kw):
        super().__init__(
            encoder=torch.nn.Identity(),
            feats=torch.nn.Identity(),
            backbone=torch.nn.Identity(),
            **kw,
        )
        self.p = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, noisy):
        self.last_mask = noisy * self.p
        return noisy


def _wire(system):
    system.logged = []
    system.log = lambda *args, **kwargs: system.logged.append((args, kwargs))
    system.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])
    return system


def _batch():
    torch.manual_seed(0)
    return {
        "noisy_speech": torch.randn(2, 256).clamp(-1, 1),
        "clean_speech": torch.zeros(2, 256),
    }


def test_perturb_changes_signal_but_not_shape():
    system = _wire(MaskEchoSystem(
        channel_consistency={"enabled": True, "prob": 1.0}))
    wav = torch.randn(3, 512).clamp(-1, 1)
    out = system._random_channel_perturb(wav)
    assert out.shape == wav.shape
    assert not torch.allclose(out, wav)
    assert out.abs().max() <= 1.0
    assert not out.requires_grad


def test_training_step_adds_consistency_loss_and_grad():
    system = _wire(MaskEchoSystem(
        channel_consistency={"enabled": True, "prob": 1.0, "weight": 1.0}))
    out = system.training_step(_batch(), 0)
    cons_logs = [a for a, k in system.logged if a[0] == "train_step_cons_loss"]
    assert cons_logs, "consistency loss was not applied/logged at prob=1"
    # the consistency term flows gradient into the mask path parameter
    out["loss"].backward()
    assert system.p.grad is not None and system.p.grad.abs() > 0


def test_consistency_schedule_is_deterministic_rank_safe():
    """prob 0.3 / period 10 fires exactly on batch_idx % 10 in {0,1,2}: the
    schedule must be a pure function of batch_idx (rank-synchronized), never a
    per-rank random draw -- that desyncs SyncBN collectives and deadlocks DDP."""
    system = _wire(MaskEchoSystem(
        channel_consistency={"enabled": True, "prob": 0.3, "weight": 1.0}))
    fired = []
    for idx in range(10):
        system.logged = []
        system.training_step(_batch(), idx)
        fired.append(bool([a for a, k in system.logged
                           if a[0] == "train_step_cons_loss"]))
    assert fired == [True, True, True] + [False] * 7


def test_consistency_max_rows_subbatch():
    """max_rows caps the second forward to a leading sub-batch (memory guard)."""
    system = _wire(MaskEchoSystem(
        channel_consistency={"enabled": True, "prob": 1.0, "weight": 1.0,
                             "max_rows": 1}))
    seen = []
    orig_forward = system.forward

    def spy(noisy):
        seen.append(tuple(noisy.shape))
        return orig_forward(noisy)

    system.forward = spy
    system.training_step(_batch(), 0)
    # first call = main forward on the full batch, second = perturbed sub-batch
    assert seen[0][0] == 2 and seen[1][0] == 1


def test_disabled_consistency_is_noop():
    system = _wire(MaskEchoSystem(channel_consistency=None))
    baseline = _wire(MaskEchoSystem(
        channel_consistency={"enabled": False, "prob": 1.0}))
    out_a = system.training_step(_batch(), 0)
    out_b = baseline.training_step(_batch(), 0)
    assert torch.isclose(out_a["loss"], out_b["loss"])
    assert not [a for a, k in system.logged if a[0] == "train_step_cons_loss"]
    assert not [a for a, k in baseline.logged if a[0] == "train_step_cons_loss"]


def test_end_to_end_tiny_dpcrn_training_step_runs():
    encoder = ConvEncDec(fft_length=64, win_length=64, hop_length=32,
                         fmin=0, fmax=8000, sr=16000, trainable=False)
    features = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True,
                              trainable=False)
    backbone = DPCRN(
        input_dim=32, channels=(2, 4, 8), kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=4,
    )
    system = EncDecMaskBase(
        encoder, features, backbone,
        channel_consistency={"enabled": True, "prob": 1.0, "weight": 0.5},
    )
    system.logged = []
    system.log = lambda *args, **kwargs: system.logged.append((args, kwargs))
    system.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])

    torch.manual_seed(1)
    batch = {"noisy_speech": torch.randn(2, 1024).clamp(-1, 1),
             "clean_speech": torch.randn(2, 1024).clamp(-1, 1)}
    out = system.training_step(batch, 0)
    assert torch.isfinite(out["loss"])
    assert [a for a, k in system.logged if a[0] == "train_step_cons_loss"]
