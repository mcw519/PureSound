"""MetricGAN: the learned PESQ critic and the two terms it adds to training.

The property that matters most is separation: the generator term must move
only the enhancer, and the critic's term only the critic. Both live in one loss
stepped by one optimizer, so a leak in either direction is silent -- the
critic would be trained to call enhanced audio perfect, or the enhancer would
be trained on stale replayed audio.
"""

import time

import numpy as np
import pytest
import torch

from puresound.nnet.lobe.metric_discriminator import MetricDiscriminator
from puresound.system.metric_gan import (
    MetricGanConfig,
    PesqReplay,
    discriminator_loss,
    generator_loss,
    normalised_pesq,
)
from puresound.system.siso import EncDecMaskBase

SR = 16000


def _speechlike(seconds=3.0, seed=0):
    """Harmonic source with a syllable-rate envelope: PESQ finds utterances in it."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 0.5
    return (0.2 * x * env + 0.001 * rng.standard_normal(t.size)).astype(np.float32)


def _pair(noise=0.05, seed=0):
    clean = torch.from_numpy(_speechlike(seed=seed))
    noisy = clean + noise * torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    return clean, noisy


def test_the_critic_scores_each_row_in_range():
    torch.manual_seed(0)
    disc = MetricDiscriminator()
    clean, noisy = _pair()
    scores = disc(torch.stack([clean, clean]), torch.stack([clean, noisy]))
    assert scores.shape == (2,)
    assert torch.all(scores >= 0) and torch.all(scores <= 1.2)


def test_replay_labels_rows_with_unit_pesq_and_drops_a_silent_reference():
    assert np.allclose(normalised_pesq([1.0, 4.5, 2.75, 0.5, 5.0]), [0.0, 1.0, 0.5, 0.0, 1.0])

    cfg = MetricGanConfig(enabled=True, pesq_workers=0, rows_per_step=3)
    replay = PesqReplay(cfg)
    clean, noisy = _pair()
    ref = torch.stack([clean, clean, torch.zeros_like(clean)])
    est = torch.stack([clean, noisy, noisy])
    replay.submit(ref, est)
    assert replay.scored == 2 and replay.rejected == 1
    labels = sorted(y for _, _, y in replay.buffer)
    assert labels[1] > 0.9, "clean vs clean should score near the top"
    assert labels[0] < labels[1], "noisy should score below clean"
    ref_s, est_s, target = replay.sample(4, "cpu")
    assert ref_s.shape == est_s.shape == (4, clean.numel()) and target.shape == (4,)


@pytest.mark.slow  # starts a PESQ worker pool
def test_the_pool_returns_the_same_labels_as_inline():
    clean, noisy = _pair()
    ref, est = torch.stack([clean, clean]), torch.stack([clean, noisy])
    inline = PesqReplay(MetricGanConfig(enabled=True, pesq_workers=0))
    inline.submit(ref, est)
    pooled = PesqReplay(MetricGanConfig(enabled=True, pesq_workers=2))
    try:
        pooled.submit(ref, est)
        deadline = time.time() + 120
        while pooled.scored < 2 and time.time() < deadline:
            pooled.harvest(); time.sleep(0.2)
    finally:
        pooled.close()
    assert [y for *_, y in pooled.buffer] == pytest.approx([y for *_, y in inline.buffer])


def test_each_term_moves_only_its_own_side():
    torch.manual_seed(1)
    clean, noisy = _pair()

    disc = MetricDiscriminator()
    estimate = noisy.clone().unsqueeze(0).requires_grad_(True)
    loss, _ = generator_loss(disc, clean.unsqueeze(0), estimate)
    loss.backward()
    assert estimate.grad is not None and estimate.grad.abs().sum() > 0
    assert all(p.grad is None for p in disc.parameters()), "generator term moved the critic"
    assert all(p.requires_grad for p in disc.parameters()), "flags were not restored"

    disc = MetricDiscriminator()
    replay = PesqReplay(MetricGanConfig(enabled=True, pesq_workers=0))
    gain = torch.nn.Parameter(torch.tensor(1.0))
    replay.submit(clean.unsqueeze(0), (noisy * gain).unsqueeze(0))   # stored detached
    loss, fake = discriminator_loss(disc, replay, clean.unsqueeze(0), rows=2)
    assert fake is not None
    loss.backward()
    assert gain.grad is None, "critic term moved the enhancer"
    assert all(p.grad is not None for p in disc.parameters() if p.requires_grad)


class _EchoSystem(EncDecMaskBase):
    """Enhancer stub: output = input * gain, so the generator term has a target."""

    def __init__(self, **kw):
        super().__init__(encoder=torch.nn.Identity(), feats=torch.nn.Identity(),
                         backbone=torch.nn.Identity(), **kw)
        self.gain = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, noisy):
        return noisy * self.gain


def _wired(**metric_gan):
    system = _EchoSystem(metric_gan={"enabled": True, "pesq_workers": 0, **metric_gan})
    system.logged = {}
    system.log = lambda name, value, **kw: system.logged.__setitem__(name, value)
    system.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])
    return system


def _batch():
    clean, noisy = _pair()
    return {"noisy_speech": noisy.unsqueeze(0).repeat(2, 1), "clean_speech": clean.unsqueeze(0).repeat(2, 1)}


def test_a_training_step_trains_both_sides():
    system = _wired(weight=0.05, warmup_steps=0)
    batch = _batch()
    system.training_step(batch, 0)                      # first step fills the replay
    out = system.training_step(batch, 1)
    out["loss"].backward()
    assert "train_metric_gan_g_score" in system.logged
    assert "train_metric_gan_fake_mse" in system.logged
    assert system.gain.grad is not None
    assert all(p.grad is not None for p in system.metric_disc.parameters() if p.requires_grad)

    # The warm-up keeps the generator term off while the critic still trains.
    system = _wired(weight=0.05, warmup_steps=10_000)
    system.training_step(batch, 0)
    assert "train_metric_gan_g_score" not in system.logged
    assert "train_metric_gan_d_loss" in system.logged


def test_an_all_absent_batch_still_gives_every_critic_parameter_a_gradient():
    """DDP needs every parameter used on every rank, every step."""
    system = _wired(weight=0.05, warmup_steps=0)
    clean, noisy = _pair()
    batch = {"noisy_speech": noisy.unsqueeze(0), "clean_speech": torch.zeros(1, clean.numel())}
    out = system.training_step(batch, 0)
    out["loss"].backward()
    assert all(p.grad is not None for p in system.metric_disc.parameters() if p.requires_grad)


def test_the_critic_gets_its_own_parameter_group():
    groups = _wired(lr_factor=0.5).get_total_param_groups()
    assert groups["metric_disc"]["lr_factor"] == 0.5
    assert "metric_disc" not in _EchoSystem().get_total_param_groups()
