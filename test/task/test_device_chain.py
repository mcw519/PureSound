"""The contracts `DeviceChain` claims, each pinned separately.

Which signals a stage may touch, what a disabled stage costs, what the chain
records about itself, and -- the one with real physics behind it -- that
everything upstream of the converter is linear.
"""

import math
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
from puresound.task.device_chain import DEVICE_CHAIN_SCALARS, DeviceChain

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


def test_a_chain_with_nothing_enabled_is_free_and_leaves_the_pair_alone():
    """The "absent/disabled block never touches the RNG stream" discipline.

    Without it, turning a knob off would shift every later stage's draw, and an
    old recipe would stop regenerating what it used to. A stage that is present
    but disabled costs the same as an absent one.
    """
    augmentor = AudioEffectAugmentor()
    all_off = _rng_draws(lambda: DeviceChain(augmentor))
    torch.manual_seed(0)
    random.seed(0)
    torch.rand(1)
    nothing_at_all = float(torch.rand(1))
    assert all_off == nothing_at_all

    disabled = _rng_draws(
        lambda: DeviceChain(
            augmentor,
            hpf=HighPassAugmentation(**NEVER),
            volume=VolumeAugmentation(**NEVER),
            codec=CodecAugmentation(**NEVER),
        )
    )
    assert disabled == all_off

    noisy, target = _pair()
    result = DeviceChain(augmentor).apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert torch.equal(result.noisy, noisy)
    assert torch.equal(result.target, target)


@pytest.mark.parametrize(
    "stage,target_moves",
    [
        ("src", True),
        ("ir_response", True),
        ("hpf", True),
        ("volume", True),
        ("codec", False),
        ("packet_loss", False),
    ],
)
def test_channel_stages_move_the_target_and_transmission_damage_does_not(
    stage, target_moves
):
    """The target is what the model must recover *through* the channel, so a
    linear channel stage filters it with the same parameters; transmission
    damage is what the model must undo, so the target stays the undamaged
    reference."""
    chain = DeviceChain(AudioEffectAugmentor(), **{stage: ENABLED[stage]})
    noisy, target = _pair()
    result = chain.apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert not torch.equal(result.noisy, noisy), f"{stage} did nothing to the mixture"
    assert torch.equal(result.target, target) is not target_moves


def test_the_converter_gain_stages_rather_than_clips_and_says_so():
    """Crossing into the digital domain is the engineer setting the preamp, not
    the rails being hit: one scalar on both, so the pair's level relationship
    comes out exactly as it went in.

    Clipping here instead would squash the mixture -- the louder of the two --
    while the target sailed through, and the mixture would stop being the sum of
    its sources at the SIR the recipe asked for.
    """
    chain = DeviceChain(AudioEffectAugmentor())
    assert chain.apply(*_pair(), sample_rate=SR).applied["overload_rescaled"] == 0.0

    result = chain.apply(torch.full((1, 16), 4.0), torch.full((1, 16), 2.0), sample_rate=SR)
    assert float(result.noisy.max()) == pytest.approx(1.0)
    assert float(result.target.max()) == pytest.approx(0.5)
    assert result.applied["overload_rescaled"] == 1.0


@pytest.mark.parametrize("stage", ["src", "ir_response", "hpf", "volume"])
def test_the_analogue_path_is_linear(stage):
    """H(a*x) == a*H(x), for every stage upstream of the converter.

    Not a style preference -- the reason these stages exist is that the target
    is what the model must recover *through* the channel, and that only holds
    while superposition does. Every DSP backend underneath them saturates at
    full scale by default (torchaudio's `lfilter` clamps, sox round-trips
    through fixed point), and upstream of the converter the pair is routinely
    hot. When it fires it hits the mixture, which is the louder signal, and
    leaves the quieter target alone --
    so the pair's level relationship is broken by an operator that is not
    supposed to exist. `puresound.audio.dsp.apply_linear` is what holds this.

    Level is a sound pressure up here, not a sample value, so 3.0 is not an
    error to be clamped away: it is a loud room, and it stays loud until the
    converter says otherwise.
    """
    chain = DeviceChain(
        AudioEffectAugmentor(), overload_guard=False, **{stage: ENABLED[stage]}
    )
    noisy, target = _pair()
    noisy, target = noisy * 10.0, target * 10.0  # peak 3.0: over full scale
    quiet_by = 100.0

    def run(seed, scale):
        torch.manual_seed(seed)
        random.seed(seed)
        return chain.apply(noisy * scale, target * scale, sample_rate=SR)

    # Several seeds, not one: a stage can pick between backends -- SRC tosses a
    # coin between sox and torchaudio -- so a single seed tests whichever
    # branch it happened to land on.
    for seed in range(4):
        loud, quiet = run(seed, 1.0), run(seed, 1.0 / quiet_by)
        for name, hot, cold in (
            ("mixture", loud.noisy, quiet.noisy),
            ("target", loud.target, quiet.target),
        ):
            error = float((hot - cold * quiet_by).abs().amax() / hot.abs().amax())
            assert error < 1e-3, (
                f"{stage} is not linear on the {name} at seed {seed}: {error:.2e}"
            )


def test_the_converter_runs_before_the_codec():
    """A codec is a digital sink; it cannot encode past full scale.

    Ordering, not decoration: with the converter at the *end* of the chain the
    codec is handed an out-of-range signal and clips it internally, which is a
    nonlinearity applied to the mixture alone and recorded nowhere.
    """
    seen = []
    augmentor = AudioEffectAugmentor()
    real_codec = augmentor.apply_codec

    def spy(wav, **kwargs):
        seen.append(float(wav.abs().amax()))
        return real_codec(wav=wav, **kwargs)

    augmentor.apply_codec = spy
    noisy, target = _pair()
    DeviceChain(augmentor, codec=ENABLED["codec"]).apply(
        noisy * 12.0, target * 12.0, sample_rate=SR
    )
    assert seen and seen[0] <= 1.0 + 1e-6, f"codec was handed peak {seen}"


# --------------------------------------------------------------------------- #
# What the chain records about itself
# --------------------------------------------------------------------------- #


def test_every_row_reports_every_key_and_a_quiet_stage_reports_zero_and_nan():
    """Every key, every row: the batch collate is one `torch.cat` per key, so a
    row that omitted a key would silently shorten that tensor and misalign it.
    A flag is 0.0 so it buckets; an absent parameter is NaN so it is skipped
    rather than counted as zero."""
    untouched = DeviceChain(AudioEffectAugmentor()).apply(*_pair(), sample_rate=SR).applied
    everything = DeviceChain(AudioEffectAugmentor(), **ENABLED).apply(
        *_pair(), sample_rate=SR
    ).applied
    assert set(untouched) == set(DEVICE_CHAIN_SCALARS)
    assert set(everything) == set(DEVICE_CHAIN_SCALARS)

    assert untouched["src_applied"] == 0.0
    assert untouched["overload_rescaled"] == 0.0
    assert math.isnan(untouched["src_target_sr"])
    assert math.isnan(untouched["hpf_cutoff"])


def test_the_record_carries_the_value_the_stage_actually_drew():
    """A flag alone cannot answer "worse at which cutoff"; the realized
    parameter is the point. Volume's clipping and gain branches are different
    damage, so they are recorded apart."""
    chain = DeviceChain(
        AudioEffectAugmentor(),
        hpf=HighPassAugmentation(**ALWAYS, cutoff=[137.0], prob_each=[1.0]),
        src=SourceRateAugmentation(**ALWAYS, src_range=[8000], prob_each=[1.0]),
    )
    applied = chain.apply(*_pair(), sample_rate=SR).applied
    assert applied["hpf_applied"] == 1.0
    assert applied["hpf_cutoff"] == pytest.approx(137.0)
    assert applied["src_target_sr"] == pytest.approx(8000.0)

    clipping = VolumeAugmentation(
        **ALWAYS,
        perturbed_range=[0.5, 1.5],
        clipping_prob=1.0,
        clipping_range={"min": [0.01, 0.05], "max": [0.95, 0.99]},
    )
    applied = DeviceChain(AudioEffectAugmentor(), volume=clipping).apply(
        *_pair(), sample_rate=SR
    ).applied
    assert applied["volume_applied"] == 1.0
    assert applied["volume_clipped"] == 1.0
    assert math.isnan(applied["volume_gain"])

    applied = DeviceChain(AudioEffectAugmentor(), volume=ENABLED["volume"]).apply(
        *_pair(), sample_rate=SR
    ).applied
    assert applied["volume_clipped"] == 0.0
    assert not math.isnan(applied["volume_gain"])


@pytest.mark.parametrize("target_clipping", ["mixture_level", "own_quantile"])
def test_overload_clips_the_mixture_the_same_way_and_the_target_as_asked(target_clipping):
    """A quiet talker in loud noise: the mixture's clip levels sit above the
    target's peak. ``mixture_level`` leaves such a target alone, ``own_quantile``
    clips it at its own quantiles; the mixture and the draws are the same."""
    time = torch.arange(SR, dtype=torch.float32) / SR
    target = (0.05 * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)
    noisy = target + 0.5 * torch.randn(1, SR, generator=torch.Generator().manual_seed(0))
    volume = VolumeAugmentation(
        **ALWAYS,
        perturbed_range=[0.5, 1.5],
        clipping_prob=1.0,
        clipping_range={"min": [0.1, 0.1], "max": [0.9, 0.9]},
        target_clipping=target_clipping,
    )
    torch.manual_seed(0)
    random.seed(0)
    result = DeviceChain(AudioEffectAugmentor(), volume=volume, overload_guard=False).apply(
        noisy.clone(), target.clone(), sample_rate=SR
    )
    after = torch.rand(1)

    low, high = torch.quantile(noisy, torch.tensor([0.1, 0.9]), dim=-1)
    torch.testing.assert_close(result.noisy, noisy.clip(low, high))
    if target_clipping == "mixture_level":
        torch.testing.assert_close(result.target, target)
    else:
        own_low, own_high = torch.quantile(target, torch.tensor([0.1, 0.9]), dim=-1)
        torch.testing.assert_close(result.target, target.clip(own_low, own_high))
        assert result.target.abs().max() < target.abs().max() * 0.96
    torch.manual_seed(0)
    random.seed(0)
    other = volume.model_copy(update={"target_clipping": "own_quantile"})
    DeviceChain(AudioEffectAugmentor(), volume=other, overload_guard=False).apply(
        noisy.clone(), target.clone(), sample_rate=SR
    )
    assert torch.rand(1) == after


# "Recording must not itself draw randomness" has no honest test here: any
# check written against this tree runs the recording code on both sides and
# passes whatever it does. It is a before/after property, so it belongs to
# tools/rng_fingerprint.py.
