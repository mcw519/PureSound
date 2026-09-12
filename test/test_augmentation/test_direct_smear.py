import random

import pytest
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.impulse_response import smear_direct_arrival
from puresound.config.augmentation import DirectSmearConfig


SR = 16000
PEAK = 160


def _impulse(length=4800, peak_at=PEAK, tail_level=0.05):
    """Unit direct peak, structured early samples, flat tail: every region distinct."""
    rir = torch.zeros(1, length)
    rir[0, peak_at] = 1.0
    early = torch.linspace(0.4, 0.1, 40) * torch.tensor([1.0, -1.0]).repeat(20)
    rir[0, peak_at + 1 : peak_at + 1 + 40] = early
    rir[0, peak_at + 200 :] = tail_level
    return rir


# ---------------------------------------------------------------- the function


def test_peak_energy_and_tail_survive_across_seeds():
    """The three preservation contracts hold for ANY draw, not a lucky one.

    Peak index feeds `wav_apply_rir`'s argmax alignment, window energy is what
    keeps this a timing manipulation rather than a level one, and the ordering
    bug this guards against (restoring the peak AFTER renormalising) shows up as
    energy drift -- so the energy check across seeds is the mutation test.
    """
    rir = _impulse()
    n = int(round(5.0 * SR / 1000.0))
    window = slice(PEAK, PEAK + n)
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        out = smear_direct_arrival(rir, SR, smear_ms=5.0, generator=g)
        assert int(out[0].abs().argmax()) == PEAK
        assert float(out[0, window].pow(2).sum()) == pytest.approx(
            float(rir[0, window].pow(2).sum()), rel=1e-4
        )
        assert torch.equal(out[0, :PEAK], rir[0, :PEAK])
        assert torch.equal(out[0, PEAK + n :], rir[0, PEAK + n :])
        # and the window's structure is actually gone
        assert not torch.allclose(out[0, window], rir[0, window], atol=1e-3)


def test_noop_paths_return_the_same_object():
    rir = _impulse()
    assert smear_direct_arrival(rir, SR, smear_ms=0.0) is rir
    assert smear_direct_arrival(rir, SR, smear_ms=-3.0) is rir
    # 0.05 ms at 16 kHz is under 2 samples: nothing to scramble
    assert smear_direct_arrival(rir, SR, smear_ms=0.05) is rir


def test_silent_and_truncated_channels_are_left_alone():
    silent = torch.zeros(1, 1000)
    out = smear_direct_arrival(
        silent, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(out, silent)

    at_the_edge = torch.zeros(1, 500)
    at_the_edge[0, 499] = 1.0
    out = smear_direct_arrival(
        at_the_edge, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(out, at_the_edge)


def test_generator_makes_it_reproducible():
    rir = _impulse()
    a = smear_direct_arrival(rir, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(7))
    b = smear_direct_arrival(rir, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(7))
    c = smear_direct_arrival(rir, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_channels_are_smeared_independently():
    rir = torch.zeros(2, 4800)
    rir[0, 100] = 1.0
    rir[0, 101:130] = 0.3
    rir[1, 300] = 1.0
    rir[1, 301:330] = 0.3
    out = smear_direct_arrival(
        rir, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(0)
    )
    assert int(out[0].abs().argmax()) == 100
    assert int(out[1].abs().argmax()) == 300
    assert not torch.equal(out[0, 100:180], rir[0, 100:180])
    assert not torch.equal(out[1, 300:380], rir[1, 300:380])


# ------------------------------------------------------------------ the config


def test_config_rejects_a_zero_start():
    with pytest.raises(ValueError, match="above 0 ms"):
        DirectSmearConfig(used=True, prob=0.5, smear_ms_range=[0.0, 2.0])


def test_config_enabled_contract():
    with pytest.raises(ValueError):
        DirectSmearConfig(used=True)
    DirectSmearConfig(used=False)  # a dark knob needs no other field
    DirectSmearConfig(used=True, prob=0.25, smear_ms_range=[2.0, 6.0])


# --------------------------------------------------------------- the augmentor


def _armed(prob=1.0, span=(2.0, 2.0)):
    aug = AudioEffectAugmentor()
    aug.init_direct_smear({"used": True, "prob": prob, "smear_ms_range": list(span)})
    return aug


def test_init_rejects_bad_knobs():
    aug = AudioEffectAugmentor()
    with pytest.raises(ValueError, match="unknown"):
        aug.init_direct_smear(
            {"used": True, "prob": 0.5, "smear_ms_range": [1.0, 2.0], "mode": "x"}
        )
    with pytest.raises(ValueError, match="prob"):
        aug.init_direct_smear({"used": True, "prob": 0.0, "smear_ms_range": [1.0, 2.0]})
    with pytest.raises(ValueError, match="smear_ms_range"):
        aug.init_direct_smear({"used": True, "prob": 0.5, "smear_ms_range": [2.0, 1.0]})
    with pytest.raises(ValueError, match="smear_ms_range"):
        aug.init_direct_smear({"used": True, "prob": 0.5, "smear_ms_range": [0.0, 1.0]})


def test_disabled_knob_consumes_no_randomness():
    """A recipe without the knob must regenerate bit-identically: the draw sits
    inside the short circuit, so switching the knob off may not shift the RNG
    stream every other augmentation reads from."""
    rir = _impulse()
    aug = AudioEffectAugmentor()
    random.seed(123)
    baseline = random.random()
    random.seed(123)
    out, meta = aug._apply_direct_smear(rir, SR, {"drr_db": 1.0})
    assert out is rir
    assert meta == {"drr_db": 1.0}
    assert random.random() == baseline


def test_applied_row_records_the_draw_without_mutating_the_input():
    rir = _impulse()
    aug = _armed(prob=1.0, span=(2.0, 4.0))
    original = {"drr_db": 1.0}
    random.seed(0)
    out, meta = aug._apply_direct_smear(rir, SR, original)
    assert out is not rir
    assert int(out[0].abs().argmax()) == PEAK
    assert 2.0 <= meta["direct_smear_ms"] <= 4.0
    assert meta["drr_db"] == 1.0
    assert original == {"drr_db": 1.0}  # caller's dict untouched


def test_skipped_draw_returns_the_input():
    rir = _impulse()
    aug = _armed(prob=1e-9)
    random.seed(0)  # first draw is ~0.844 > prob: skipped
    out, meta = aug._apply_direct_smear(rir, SR, None)
    assert out is rir
    assert meta is None
