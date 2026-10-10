"""What `NoiseStage` promises, pinned one contract at a time.

What a source that does not fire costs, where the capture floor lands, and
which SNR the row reports.
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
    "build",
    [
        lambda: _stage(None),
        lambda: _stage(NoiseAugmentation(used=False)),
        # the floor is the innermost guard, so its draw is the easiest one to
        # leave outside the short circuit
        lambda: _stage(
            NoiseAugmentation(used=False, absolute_floor=AbsoluteFloorConfig(used=False))
        ),
    ],
    ids=["no-block", "block-disabled", "floor-disabled"],
)
def test_a_stage_that_adds_nothing_is_free_and_says_so(build):
    """No randomness drawn -- otherwise turning noise off would shift every later
    stage's draw and an old recipe would stop regenerating what it used to --
    the mixture untouched, and NaN rather than 0.0 for the SNR, because eval
    skips a missing measurement and averages a zero."""
    assert _next_draw(build) == _nothing_at_all()
    mixture = _mixture()
    result = build().apply(mixture.clone(), sample_rate=SR)
    assert torch.equal(result.noisy, mixture)
    assert math.isnan(result.snr)


# --------------------------------------------------------------------------- #
# The capture floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("level_dbfs", [-60.0, -45.0, -30.0])
def test_the_capture_floor_lands_at_the_absolute_level_it_drew(level_dbfs):
    """Absolute, not relative to the speech: two mixtures 20 dB apart get the
    same floor, which an SNR-relative source cannot do."""
    config = NoiseAugmentation(
        used=False,
        absolute_floor=AbsoluteFloorConfig(
            used=True, prob=1.0, level_dbfs_range=[level_dbfs, level_dbfs]
        ),
    )
    floors = []
    for gain in (1.0, 0.1):
        torch.manual_seed(0)
        silence = torch.zeros(1, SR)  # silent, so only the floor is present
        floors.append(_stage(config).apply(silence * gain, sample_rate=SR).noisy)
    measured = 20 * math.log10(float(floors[0].pow(2).mean().sqrt()))
    assert measured == pytest.approx(level_dbfs, abs=0.5)
    assert float(floors[0].pow(2).mean()) == pytest.approx(
        float(floors[1].pow(2).mean()), rel=1e-6
    )


# --------------------------------------------------------------------------- #
# The number the row reports
# --------------------------------------------------------------------------- #


def test_the_reported_snr_is_the_recorded_noise_one_not_the_white_noise_one(
    noise_pool,
):
    """The white-noise branch draws its own SNR; the row must still report the
    recorded-noise one."""
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
    # dynamic-noise draw did not claim, and one seed may not reach it.
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


def test_snr_bands_keep_the_hard_mixtures_while_the_range_reaches_40(noise_pool):
    """A piecewise draw: 80 % inside [-5, 15], 20 % in (15, 40]."""
    config = _config(
        noise_pool,
        snr_range=[-5.0, 40.0],
        snr_bands=[{"low": -5.0, "high": 15.0, "prob": 0.8}, {"low": 15.0, "high": 40.0, "prob": 0.2}],
    )
    stage = _stage(config, noise_pool)
    torch.manual_seed(0)
    random.seed(0)
    snrs = [stage.apply(_mixture(0.1), sample_rate=SR).snr for _ in range(2000)]
    below = sum(s <= 15.0 for s in snrs) / len(snrs)
    assert 0.76 < below < 0.84  # uniform over [-5, 40] would give 0.44
    assert min(snrs) >= -5.0 and max(snrs) <= 40.0


@pytest.mark.parametrize(
    "snr_range,bands,match",
    [
        ([0.0, 40.0], [{"low": 0, "high": 20, "prob": 0.5}], "sum to 1"),
        ([0.0, 20.0], [{"low": 0, "high": 40, "prob": 1.0}], "outside snr_range"),
    ],
)
def test_snr_bands_must_lie_in_the_range_and_sum_to_one(snr_range, bands, match):
    with pytest.raises(ValueError, match=match):
        _config("/noise", snr_range=snr_range, snr_bands=bands)
