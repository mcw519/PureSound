"""The three contracts `DeviceChain` claims, each pinned separately.

The end-to-end fingerprint proves the chain as a whole did not change. These
pin *why* it is allowed to change: which signals a stage may touch, and what a
disabled stage costs.
"""

import random

import pytest
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.config.augmentation import (
    CodecAugmentation,
    HighPassAugmentation,
    PacketLossAugmentation,
    SimpleProbAugmentation,
    SourceRateAugmentation,
    VolumeAugmentation,
)
from puresound.task.device_chain import DeviceChain

SR = 16000


def _pair():
    time = torch.arange(SR, dtype=torch.float32) / SR
    noisy = (0.3 * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)
    target = (0.2 * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)
    return noisy, target


ALWAYS = {"used": True, "prob": 1.0}
NEVER = {"used": False}

ENABLED = dict(
    src=SourceRateAugmentation(**ALWAYS, src_range=[8000], prob_each=[1.0]),
    ir_response=SimpleProbAugmentation(**ALWAYS),
    hpf=HighPassAugmentation(**ALWAYS, cutoff=[120.0], prob_each=[1.0]),
    volume=VolumeAugmentation(
        **ALWAYS,
        perturbed_range=[0.5, 1.5],
        clipping_prob=0.0,
        clipping_range={"min": [0.01, 0.05], "max": [0.95, 0.99]},
    ),
    codec=CodecAugmentation(**ALWAYS, codecs=["libopus"]),
    packet_loss=PacketLossAugmentation(
        **ALWAYS, packet_ms_choices=[20], loss_rate_range=[0.05, 0.05]
    ),
)


def _rng_draws(build_chain) -> int:
    """How many values the chain pulls off the shared streams.

    Counted rather than compared: the point is that a disabled stage costs
    *nothing*, not merely that it produces the same audio.
    """
    torch.manual_seed(0)
    random.seed(0)
    before_torch = torch.rand(1)  # anchor
    chain = build_chain()
    chain.apply(*_pair(), sample_rate=SR)
    after = torch.rand(1)
    del before_torch
    return float(after)


def test_a_disabled_stage_consumes_no_randomness():
    """The whole "absent/disabled block never touches the RNG stream" discipline.

    Without it, turning a knob off would shift every later stage's draw, and an
    old recipe would stop regenerating what it used to.
    """
    augmentor = AudioEffectAugmentor()
    all_off = _rng_draws(lambda: DeviceChain(augmentor))
    torch.manual_seed(0)
    random.seed(0)
    torch.rand(1)
    nothing_at_all = float(torch.rand(1))
    assert all_off == nothing_at_all

    # And a stage that is present but disabled costs the same as absent.
    disabled = _rng_draws(
        lambda: DeviceChain(
            augmentor,
            hpf=HighPassAugmentation(**NEVER),
            volume=VolumeAugmentation(**NEVER),
            codec=CodecAugmentation(**NEVER),
        )
    )
    assert disabled == all_off


@pytest.mark.parametrize("stage", ["codec", "packet_loss"])
def test_transmission_damage_never_touches_the_target(stage):
    """The target is the undamaged reference the model is scored against."""
    chain = DeviceChain(AudioEffectAugmentor(), **{stage: ENABLED[stage]})
    noisy, target = _pair()
    result = chain.apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert not torch.equal(result.noisy, noisy), f"{stage} did nothing to the mixture"
    assert torch.equal(result.target, target)


@pytest.mark.parametrize("stage", ["src", "ir_response", "hpf", "volume"])
def test_a_linear_channel_stage_moves_both_signals(stage):
    """The target is what the model must recover *through* the channel, so it
    goes through the same filter with the same parameters."""
    chain = DeviceChain(AudioEffectAugmentor(), **{stage: ENABLED[stage]})
    noisy, target = _pair()
    result = chain.apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert not torch.equal(result.noisy, noisy), f"{stage} did nothing to the mixture"
    assert not torch.equal(result.target, target), f"{stage} skipped the target"


def test_the_overload_guard_rescales_the_pair_together():
    """A target above full scale is one the model's clamped output cannot reach;
    dividing both by the same peak keeps the level relationship."""
    chain = DeviceChain(AudioEffectAugmentor())
    noisy = torch.full((1, 16), 4.0)
    target = torch.full((1, 16), 2.0)
    result = chain.apply(noisy, target, sample_rate=SR)
    assert float(result.noisy.max()) == pytest.approx(1.0)
    assert float(result.target.max()) == pytest.approx(0.5)


def test_an_untouched_pair_passes_through_unchanged():
    chain = DeviceChain(AudioEffectAugmentor())
    noisy, target = _pair()
    result = chain.apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert torch.equal(result.noisy, noisy)
    assert torch.equal(result.target, target)
