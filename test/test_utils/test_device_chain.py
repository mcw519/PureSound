"""The contracts `DeviceChain` claims, each pinned separately.

The end-to-end fingerprint proves the chain as a whole did not change. These
pin *why* it is allowed to change: which signals a stage may touch, what a
disabled stage costs, and -- the one with real physics behind it -- that
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


def test_the_converter_gain_stages_rather_than_clips():
    """Crossing into the digital domain is the engineer setting the preamp, not
    the rails being hit: one scalar on both, so the pair's level relationship
    comes out exactly as it went in.

    Clipping here instead would squash the mixture -- the louder of the two --
    while the target sailed through, and the mixture would stop being the sum of
    its sources at the SIR the recipe asked for.
    """
    chain = DeviceChain(AudioEffectAugmentor())
    noisy = torch.full((1, 16), 4.0)
    target = torch.full((1, 16), 2.0)
    result = chain.apply(noisy, target, sample_rate=SR)
    assert float(result.noisy.max()) == pytest.approx(1.0)
    assert float(result.target.max()) == pytest.approx(0.5)
    assert float(result.target.max() / result.noisy.max()) == pytest.approx(0.5)


@pytest.mark.parametrize("stage", ["src", "ir_response", "hpf", "volume"])
def test_the_analogue_path_is_linear(stage):
    """H(a*x) == a*H(x), for every stage upstream of the converter.

    Not a style preference -- the reason these stages exist is that the target
    is what the model must recover *through* the channel, and that only holds
    while superposition does. Every DSP backend underneath them saturates at
    full scale by default (torchaudio's `lfilter` clamps, sox round-trips
    through fixed point), and upstream of the converter the pair is routinely
    hot: on the shipped voice-isolation recipe that default fired on 28% of
    transducer-response calls and 19% of gain calls. When it fires it hits the
    mixture, which is the louder signal, and leaves the quieter target alone --
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
    # coin between sox and torchaudio, and only the sox leg was ever the
    # problem -- so a single seed tests whichever branch it happened to land on.
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


def test_an_untouched_pair_passes_through_unchanged():
    chain = DeviceChain(AudioEffectAugmentor())
    noisy, target = _pair()
    result = chain.apply(noisy.clone(), target.clone(), sample_rate=SR)
    assert torch.equal(result.noisy, noisy)
    assert torch.equal(result.target, target)


# --------------------------------------------------------------------------- #
# What the chain records about itself
# --------------------------------------------------------------------------- #


def test_every_row_reports_every_key():
    """Not "the keys that fired" -- every key, every row.

    The batch collate is one `torch.cat` per key, so a row that omitted a key
    would silently shorten that tensor and misalign it against the others.
    """
    chain = DeviceChain(AudioEffectAugmentor())
    untouched = chain.apply(*_pair(), sample_rate=SR).applied
    everything = DeviceChain(AudioEffectAugmentor(), **ENABLED).apply(
        *_pair(), sample_rate=SR
    ).applied
    assert set(untouched) == set(DEVICE_CHAIN_SCALARS)
    assert set(everything) == set(DEVICE_CHAIN_SCALARS)


def test_a_stage_that_did_not_fire_reports_zero_and_nan():
    """0.0/NaN, matching the task scalars already emitted: a flag is a number so
    it buckets, and an absent parameter is NaN so it is skipped rather than
    counted as zero."""
    applied = DeviceChain(AudioEffectAugmentor()).apply(*_pair(), sample_rate=SR).applied
    assert applied["src_applied"] == 0.0
    assert applied["overload_rescaled"] == 0.0
    assert math.isnan(applied["src_target_sr"])
    assert math.isnan(applied["hpf_cutoff"])


def test_the_record_carries_the_value_the_stage_actually_drew():
    """A flag alone cannot answer "worse at which cutoff"; the realized
    parameter is the point."""
    chain = DeviceChain(
        AudioEffectAugmentor(),
        hpf=HighPassAugmentation(**ALWAYS, cutoff=[137.0], prob_each=[1.0]),
        src=SourceRateAugmentation(**ALWAYS, src_range=[8000], prob_each=[1.0]),
    )
    applied = chain.apply(*_pair(), sample_rate=SR).applied
    assert applied["hpf_applied"] == 1.0
    assert applied["hpf_cutoff"] == pytest.approx(137.0)
    assert applied["src_target_sr"] == pytest.approx(8000.0)


def test_volume_distinguishes_the_clipping_branch_from_the_gain_branch():
    """They are different damage, and grouping them together would hide it."""
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


def test_the_converter_reports_when_it_had_to_rescale():
    chain = DeviceChain(AudioEffectAugmentor())
    quiet = chain.apply(*_pair(), sample_rate=SR).applied
    assert quiet["overload_rescaled"] == 0.0
    loud = chain.apply(
        torch.full((1, 16), 4.0), torch.full((1, 16), 2.0), sample_rate=SR
    ).applied
    assert loud["overload_rescaled"] == 1.0


# "Recording must not itself draw randomness" has no honest test here: any
# check written against this tree runs the recording code on both sides and
# passes whatever it does. It is a before/after property, so it belongs to
# tools/rng_fingerprint.py -- which is how the record was added, with the 840
# pre-existing hashes unchanged and only the new keys appearing.
