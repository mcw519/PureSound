"""Param-group / lr-factor plumbing: the path every training run configures its
optimizer through (get_total_param_groups -> create_optimizer_and_scheduler)."""
import torch.nn as nn

from puresound.nnet import ConvEncDec, FeatureEncoder
from puresound.nnet.lobe.heads import VADHead
from puresound.config.recipe import OptimizerConfig, SchedulerConfig
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.system.siso import EncDecMaskBase


def _tiny_system(**module_args):
    encoder = ConvEncDec(fft_length=64, win_type="hann", win_length=64,
                         hop_length=32, fmin=0, fmax=4000, sr=8000, trainable=False)
    feats = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True,
                           trainable=False, include_specaug=False)

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(2, 2, 1)
            self.vad_head = VADHead(enc_channels=2, hidden=4, kernel_t=3)

        def forward(self, x):
            return self.proj(x)

    return EncDecMaskBase(encoder=encoder, feats=feats, backbone=TinyBackbone(),
                          mask_type="complex", **module_args)


def test_param_groups_carry_lr_factors_into_optimizer():
    model = _tiny_system(encoder_lr_factor=0.1, feats_lr_factor=0.5, backbone_lr_factor=1.0)
    groups = model.get_total_param_groups()
    assert set(groups) == {"encoder", "feats", "backbone"}

    optimizer, scheduler = create_optimizer_and_scheduler(
        groups,
        OptimizerConfig(type="AdamW", learning_rate=1e-3, args={"weight_decay": 0.0}),
        SchedulerConfig(
            type="CosineAnnealingWarmRestarts", warmup_step=0, args={"T_0": 20}
        ),
    )
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert lrs == [1e-4, 5e-4, 1e-3]          # learning_rate * lr_factor, group order
    assert scheduler.optimizer is optimizer


