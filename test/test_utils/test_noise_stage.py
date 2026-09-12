"""What `NoiseStage` promises, pinned one contract at a time.

The fingerprint proves the extraction from `ns.__getitem__` changed nothing.
These pin why it is allowed to change: what a source that does not fire costs,
which signal noise may touch, and which SNR the row reports -- the last one
being a bug the shape of the old code invited.
"""

import math
import random

import pytest
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.config.augmentation import AbsoluteFloorConfig, NoiseAugmentation
from puresound.task.noise_stage import NoiseStage

SR = 16000


def _mixture(seconds=0.5):
    time = torch.arange(int(SR * seconds), dtype=torch.float32) / SR
    return (0.2 * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)


def _config(noise_dir=None, **overrides):
    base = dict(
        used=True,
        prob=1.0,
        noise_folder=str(noise_dir) if noise_dir else None,
        snr_range=[5.0, 5.0],
        prob_white_noise=0.0,
        white_noise_snr_range=[20.0, 20.0],
    )
    base.update(overrides)
    return NoiseAugmentation(**base)


def _stage(config, noise_dir=None):
    augmentor = AudioEffectAugmentor()
    if noise_dir is not None:
        augmentor.load_bg_noise_from_folder(str(noise_dir))
    return NoiseStage(augmentor, config)


@pytest.fixture
def noise_pool(tmp_path):
    """Several files, not one: the dynamic-noise branch samples more than one."""
    from puresound.audio.io import AudioIO

    folder = tmp_path / "noise"
    folder.mkdir()
    generator = torch.Generator().manual_seed(0)
    for index in range(5):
        AudioIO.save(
            0.05 * torch.randn(1, SR, generator=generator),
            str(folder / f"n{index}.wav"),
            SR,
        )
    return folder


def _next_draw(build, mixture=None):
    """The next value off the shared stream after one `apply`.

    Counted rather than compared: the point is that a source which does not fire
    costs *nothing*, not merely that it produces the same audio.
    """
    torch.manual_seed(0)
    random.seed(0)
    build().apply(mixture if mixture is not None else _mixture(), sample_rate=SR)
    return float(torch.rand(1))


def _nothing_at_all():
    torch.manual_seed(0)
    random.seed(0)
    return float(torch.rand(1))


# --------------------------------------------------------------------------- #
# What a source that does not fire costs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,build",
    [
        ("no block at all", lambda: _stage(None)),
        ("block disabled", lambda: _stage(NoiseAugmentation(used=False))),
    ],
)
def test_a_stage_that_adds_nothing_consumes_no_randomness(name, build):
    """Otherwise turning noise off would shift every later stage's draw and an
    old recipe would stop regenerating what it used to."""
    assert _next_draw(build) == _nothing_at_all(), name


def test_a_disabled_capture_floor_consumes_no_randomness():
    """The floor is the innermost guard, so it is the easiest one to get wrong:
    its draw has to sit inside the short circuit like every other."""
    without = _next_draw(lambda: _stage(NoiseAugmentation(used=False)))
    disabled = _next_draw(
        lambda: _stage(
            NoiseAugmentation(
                used=False, absolute_floor=AbsoluteFloorConfig(used=False)
            )
        )
    )
    assert disabled == without


def test_a_stage_that_adds_nothing_reports_no_snr():
    """NaN, not 0.0: eval skips a missing measurement and averages a zero."""
    result = _stage(None).apply(_mixture(), sample_rate=SR)
    assert math.isnan(result.snr)


def test_a_stage_that_adds_nothing_returns_the_mixture_untouched():
    mixture = _mixture()
    assert torch.equal(_stage(None).apply(mixture.clone(), sample_rate=SR).noisy, mixture)


# --------------------------------------------------------------------------- #
# The capture floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("level_dbfs", [-60.0, -45.0, -30.0])
def test_the_capture_floor_lands_at_the_level_it_drew(level_dbfs):
    """Absolute, not relative to the speech: that is the whole reason it exists
    alongside the SNR-relative source."""
    stage = _stage(
        NoiseAugmentation(
            used=False,
            absolute_floor=AbsoluteFloorConfig(
                used=True, prob=1.0, level_dbfs_range=[level_dbfs, level_dbfs]
            ),
        )
    )
    torch.manual_seed(0)
    silence = torch.zeros(1, SR)
    floor = stage.apply(silence, sample_rate=SR).noisy
    measured = 20 * math.log10(float(floor.pow(2).mean().sqrt()))
    assert measured == pytest.approx(level_dbfs, abs=0.5)


def test_the_capture_floor_does_not_scale_with_the_speech():
    """Two mixtures 20 dB apart must get the same floor, which is what an
    SNR-relative source cannot do."""
    config = NoiseAugmentation(
        used=False,
        absolute_floor=AbsoluteFloorConfig(
            used=True, prob=1.0, level_dbfs_range=[-45.0, -45.0]
        ),
    )
    levels = []
    for gain in (1.0, 0.1):
        torch.manual_seed(0)
        quiet = torch.zeros(1, SR) + 0.0  # silent, so only the floor is present
        levels.append(
            float(_stage(config).apply(quiet * gain, sample_rate=SR).noisy.pow(2).mean())
        )
    assert levels[0] == pytest.approx(levels[1], rel=1e-6)


# --------------------------------------------------------------------------- #
# The number the row reports
# --------------------------------------------------------------------------- #


def test_the_reported_snr_is_the_recorded_noise_one_not_the_white_noise_one(
    noise_pool,
):
    """The code this replaces drew the white-noise SNR into the same local, so
    reading it after that branch would label the row with the wrong number.
    Pinned because the shape of the old code invited exactly that.
    """
    stage = _stage(
        _config(
            noise_pool,
            snr_range=[7.0, 7.0],
            prob_white_noise=1.0,
            white_noise_snr_range=[33.0, 33.0],
        ),
        noise_pool,
    )
    # Several seeds, because the white-noise branch only runs on rows the
    # dynamic-noise draw did not claim -- seed 0 happens to be one of those, so a
    # single seed leaves the mutation this guards against alive.
    for seed in range(6):
        torch.manual_seed(seed)
        random.seed(seed)
        assert stage.apply(_mixture(), sample_rate=SR).snr == pytest.approx(7.0), seed


def test_room_coloring_without_a_room_is_a_no_op_that_draws_nothing(noise_pool):
    """A row with no room has nothing to colour the noise with.

    Checked through a *firing* noise stage, not a disabled one: with the outer
    block off the guard is never reached and the test would be pinning the wrong
    short circuit.
    """
    from puresound.config.augmentation import RoomColoringConfig

    with_block = _config(
        noise_pool, room_coloring=RoomColoringConfig(used=True, prob=1.0)
    )
    without_block = _config(noise_pool)

    stage = _stage(with_block, noise_pool)
    assert stage._room_coloring(SR, None) is None

    def stream_position(config):
        torch.manual_seed(0)
        random.seed(0)
        _stage(config, noise_pool).apply(
            _mixture(), sample_rate=SR, room_scene=None
        )
        return float(torch.rand(1))

    assert stream_position(with_block) == stream_position(without_block)
