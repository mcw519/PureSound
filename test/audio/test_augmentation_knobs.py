"""``AudioEffectAugmentor`` knobs: direct-arrival smear, DRR contrast, noise sources.

Every knob shares one contract besides its own: a recipe that does not turn it on
regenerates bit-identically, so the draw sits inside the short circuit and an off
knob never shifts the RNG stream the other augmentations read from.
"""
import json
import math
import random
from collections import Counter

import pytest
import soundfile as sf
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.impulse_response import compute_drr_db, smear_direct_arrival
from puresound.config.augmentation import DirectSmearConfig, NoiseAugmentation

SR = 16000
PEAK = 160


def _impulse(length=4800, peak_at=PEAK, tail_level=0.05, early=True):
    """Unit direct peak, structured early samples, flat tail: every region distinct.

    Without ``early`` the tail starts well past the 2.5 ms direct window, which is
    the shape the DRR measure reads.
    """
    rir = torch.zeros(1, length)
    rir[0, peak_at] = 1.0
    if early:
        samples = torch.linspace(0.4, 0.1, 40) * torch.tensor([1.0, -1.0]).repeat(20)
        rir[0, peak_at + 1 : peak_at + 1 + 40] = samples
        rir[0, peak_at + 200 :] = tail_level
    else:
        rir[0, peak_at + int(round(2.5e-3 * SR)) + 8 :] = tail_level
    return rir


def _noise_folder(root, name, count, write_tone_wav):
    for i in range(count):
        write_tone_wav(root / name / f"clip_{i}.wav")
    return root / name


def _noise_config(**kw):
    base = dict(used=True, prob=0.9, snr_range=(0, 20), prob_white_noise=0.0,
                white_noise_snr_range=(10, 30))
    return NoiseAugmentation(**{**base, **kw})


# --------------------------------------------------------------------------- #
# Configuration: what each knob refuses, and what a recipe may write
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "configure, match",
    [
        (lambda tmp: DirectSmearConfig(used=True, prob=0.5, smear_ms_range=[0.0, 2.0]), "above 0 ms"),
        (lambda tmp: DirectSmearConfig(used=True), ""),
        (lambda tmp: AudioEffectAugmentor().init_direct_smear(
            {"used": True, "prob": 0.5, "smear_ms_range": [1.0, 2.0], "mode": "x"}), "unknown"),
        (lambda tmp: AudioEffectAugmentor().init_direct_smear(
            {"used": True, "prob": 0.0, "smear_ms_range": [1.0, 2.0]}), "prob"),
        (lambda tmp: AudioEffectAugmentor().init_direct_smear(
            {"used": True, "prob": 0.5, "smear_ms_range": [2.0, 1.0]}), "smear_ms_range"),
        (lambda tmp: AudioEffectAugmentor().init_direct_smear(
            {"used": True, "prob": 0.5, "smear_ms_range": [0.0, 1.0]}), "smear_ms_range"),
        (lambda tmp: AudioEffectAugmentor().init_drr_contrast({"used": True, "prob": 0.0}), "prob"),
        (lambda tmp: AudioEffectAugmentor().init_drr_contrast(
            {"used": True, "near_boost_db": [4.0, 1.0]}), "ranges"),
        (lambda tmp: AudioEffectAugmentor().init_drr_contrast(
            {"used": True, "near_boost": [0, 4]}), "unknown drr_contrast options"),
        (lambda tmp: _noise_config(noise_folder="/x", noise_sources=[{"name": "a", "folder": "/y"}]),
         "not both"),
        (lambda tmp: _noise_config(), "noise_folder or noise_sources"),
        (lambda tmp: ((tmp / "empty").mkdir(), AudioEffectAugmentor().load_bg_noise_sources(
            [("empty", str(tmp / "empty"), 1.0)])), "no .wav files"),
    ],
    ids=["smear-zero-start", "smear-enabled-without-fields", "smear-unknown-option",
         "smear-zero-prob", "smear-reversed-range", "smear-range-from-zero",
         "drr-zero-prob", "drr-reversed-range", "drr-unknown-option",
         "noise-named-twice", "noise-not-named", "noise-empty-source"],
)
def test_a_bad_knob_is_refused_when_it_is_configured(tmp_path, configure, match):
    with pytest.raises(ValueError, match=match):
        configure(tmp_path)


def test_the_configs_accept_what_a_recipe_writes():
    DirectSmearConfig(used=False)  # a dark knob needs no other field
    DirectSmearConfig(used=True, prob=0.25, smear_ms_range=[2.0, 6.0])
    source = _noise_config(noise_sources=[{"name": "a", "folder": "/y", "weight": 2.0}])
    assert source.noise_sources[0].weight == 2.0


# --------------------------------------------------------------------------- #
# Direct-arrival smear
# --------------------------------------------------------------------------- #


def test_the_smear_keeps_peak_window_energy_and_tail_for_any_draw():
    """Peak index feeds `wav_apply_rir`'s argmax alignment, window energy is what
    keeps this a timing manipulation rather than a level one, and restoring the
    peak AFTER renormalising shows up as energy drift -- so the energy check
    across seeds is the mutation test."""
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


def test_nothing_to_smear_returns_the_input():
    rir = _impulse()
    assert smear_direct_arrival(rir, SR, smear_ms=0.0) is rir
    assert smear_direct_arrival(rir, SR, smear_ms=-3.0) is rir
    # 0.05 ms at 16 kHz is under 2 samples: nothing to scramble
    assert smear_direct_arrival(rir, SR, smear_ms=0.05) is rir

    silent = torch.zeros(1, 1000)
    at_the_edge = torch.zeros(1, 500)
    at_the_edge[0, 499] = 1.0
    for channel in (silent, at_the_edge):
        out = smear_direct_arrival(channel, SR, smear_ms=5.0,
                                   generator=torch.Generator().manual_seed(0))
        assert torch.equal(out, channel)


def test_the_smear_is_reproducible_from_its_generator_and_per_channel():
    rir = _impulse()
    a, b, c = (smear_direct_arrival(rir, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(s))
               for s in (7, 7, 8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)

    two = torch.zeros(2, 4800)
    for channel, peak in ((0, 100), (1, 300)):
        two[channel, peak] = 1.0
        two[channel, peak + 1 : peak + 30] = 0.3
    out = smear_direct_arrival(two, SR, smear_ms=5.0, generator=torch.Generator().manual_seed(0))
    for channel, peak in ((0, 100), (1, 300)):
        assert int(out[channel].abs().argmax()) == peak
        assert not torch.equal(out[channel, peak : peak + 80], two[channel, peak : peak + 80])


def test_an_unconfigured_smear_consumes_no_randomness():
    rir = _impulse()
    aug = AudioEffectAugmentor()
    random.seed(123)
    baseline = random.random()
    random.seed(123)
    out, meta = aug._apply_direct_smear(rir, SR, {"drr_db": 1.0})
    assert out is rir
    assert meta == {"drr_db": 1.0}
    assert random.random() == baseline


def test_an_applied_smear_records_its_draw_and_a_skipped_one_returns_the_input():
    rir = _impulse()
    aug = AudioEffectAugmentor()
    aug.init_direct_smear({"used": True, "prob": 1.0, "smear_ms_range": [2.0, 4.0]})
    original = {"drr_db": 1.0}
    random.seed(0)
    out, meta = aug._apply_direct_smear(rir, SR, original)
    assert out is not rir
    assert int(out[0].abs().argmax()) == PEAK
    assert 2.0 <= meta["direct_smear_ms"] <= 4.0
    assert meta["drr_db"] == 1.0
    assert original == {"drr_db": 1.0}  # caller's dict untouched

    rare = AudioEffectAugmentor()
    rare.init_direct_smear({"used": True, "prob": 1e-9, "smear_ms_range": [2.0, 2.0]})
    random.seed(0)  # first draw is ~0.844 > prob: skipped
    out, meta = rare._apply_direct_smear(rir, SR, None)
    assert out is rir
    assert meta is None


# --------------------------------------------------------------------------- #
# DRR contrast
# --------------------------------------------------------------------------- #


def _drr_knob(prob=1.0, near=(3.0, 3.0), far=(3.0, 3.0)):
    aug = AudioEffectAugmentor()
    aug.init_drr_contrast({"used": True, "prob": prob, "near_boost_db": list(near),
                           "far_cut_db": list(far), "direct_window_ms": 2.5})
    return aug


def _write_room(root, room_id, distances, length=4800):
    sample_dir = root / room_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels = ["near_0", "near_1", "far_0", "far_1", "far_2"]
    rir = torch.zeros(5, length)
    for ch, dist in enumerate(distances):
        peak = int(round(dist / 343.0 * SR))
        rir[ch, peak] = 1.0
        rir[ch, peak + 100 :] = 0.05
    sf.write(str(sample_dir / "rir_5ch.wav"), rir.numpy().T, SR, subtype="FLOAT")
    channel_map = [
        {"channel": ch, "label": labels[ch], "distance_m": float(dist)}
        for ch, dist in enumerate(distances)
    ]
    metadata = {"scene": {"rt60": 0.45, "channel_map": channel_map}}
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_the_shift_lands_exactly_on_the_pipeline_drr_measure_for_role_aware_channels():
    rir = _impulse(early=False)
    before = compute_drr_db(rir, SR, 2.5)

    aug = _drr_knob(near=(3.0, 3.0), far=(4.0, 4.0))
    up, meta_up = aug._apply_drr_contrast(rir, SR, "foreground", {"drr_db": before})
    down, meta_down = aug._apply_drr_contrast(rir, SR, "interferer", {"drr_db": before})

    assert compute_drr_db(up, SR, 2.5) - before == pytest.approx(3.0, abs=0.05)
    assert compute_drr_db(down, SR, 2.5) - before == pytest.approx(-4.0, abs=0.05)
    # metadata carries the recomputed truth, not the pre-shift value
    assert meta_up["drr_contrast_shift_db"] == pytest.approx(3.0)
    assert meta_up["drr_db"] == pytest.approx(compute_drr_db(up, SR, 2.5), abs=1e-6)
    assert meta_down["drr_contrast_shift_db"] == pytest.approx(-4.0)
    # the direct block is untouched either way
    assert torch.equal(up[..., :200], rir[..., :200])
    assert torch.equal(down[..., :200], rir[..., :200])

    for role in ("source", "", None):  # only role-aware channels are touched
        out, meta = aug._apply_drr_contrast(rir, SR, role, {"drr_db": 1.0})
        assert out is rir and meta == {"drr_db": 1.0}


def test_an_unconfigured_drr_contrast_consumes_no_randomness():
    rir = _impulse(early=False)
    random.seed(1234)
    state = random.getstate()
    out, meta = AudioEffectAugmentor()._apply_drr_contrast(rir, SR, "foreground", {"drr_db": 1.0})
    assert out is rir and meta == {"drr_db": 1.0}
    assert random.getstate() == state

    # and the configured knob DOES consume the stream (one gate + one uniform)
    _drr_knob()._apply_drr_contrast(rir, SR, "foreground", {"drr_db": 1.0})
    assert random.getstate() != state


def test_clean_target_reuse_sees_the_shifted_impulse(tmp_path):
    """The cache must hold the modified RIR so mixture and target agree."""
    root = tmp_path / "bank"
    _write_room(root, "room_000000", [0.5, 0.8, 2.5, 3.5, 4.5])
    aug = _drr_knob(near=(4.0, 4.0))
    aug.init_room_bank({"used": True, "folder": str(root)})

    scene = aug.sample_room_scene()
    wav = torch.randn(1, SR)
    _mix, (rir_id, info) = aug.apply_rir(
        wav=wav, rir_mode="full", sr=SR, room_scene=scene, source_role="foreground"
    )
    shift = info["metadata"]["drr_contrast_shift_db"]
    assert shift == pytest.approx(4.0)

    _clean, (rir_id2, info2) = aug.apply_rir(wav=wav, rir_id=rir_id, rir_mode="direct", sr=SR)
    assert rir_id2 == rir_id
    # the cached (reused) metadata is the post-shift one
    assert info2["metadata"]["drr_contrast_shift_db"] == pytest.approx(shift)
    assert info2["metadata"]["drr_db"] == pytest.approx(info["metadata"]["drr_db"])


def test_deterministic_mode_is_a_fixed_function_of_distance():
    """Same channel -> same shift, steeper pool gradient, zero RNG use."""
    aug = AudioEffectAugmentor()
    aug.init_drr_contrast(
        {"used": True, "mode": "deterministic", "extra_db_per_decade": -3.2, "pivot_m": 1.0}
    )
    rir = _impulse(early=False)
    before = compute_drr_db(rir, SR, 2.5)

    random.seed(77)
    state = random.getstate()
    near, _ = aug._apply_drr_contrast(rir, SR, "foreground", {"source_receiver_distance": 0.5})
    far, _ = aug._apply_drr_contrast(rir, SR, "interferer", {"source_receiver_distance": 4.0})
    assert random.getstate() == state  # deterministic mode never touches the RNG

    assert compute_drr_db(near, SR, 2.5) - before == pytest.approx(-3.2 * math.log10(0.5), abs=0.05)
    assert compute_drr_db(far, SR, 2.5) - before == pytest.approx(-3.2 * math.log10(4.0), abs=0.05)
    # role does not matter -- only distance does; repeat draws are identical
    far2, _ = aug._apply_drr_contrast(rir, SR, "foreground", {"source_receiver_distance": 4.0})
    assert torch.equal(far2, far)
    # missing/invalid distance -> untouched
    same, _ = aug._apply_drr_contrast(rir, SR, "interferer", {"rt60": 0.4})
    assert same is rir


# --------------------------------------------------------------------------- #
# Noise sources: a corpus's share is chosen, not its file count
# --------------------------------------------------------------------------- #


def test_noise_draws_follow_source_weights_not_file_counts(tmp_path, write_tone_wav):
    big = _noise_folder(tmp_path, "big", 40, write_tone_wav)
    small = _noise_folder(tmp_path / "other", "big", 2, write_tone_wav)  # same basename
    augmentor = AudioEffectAugmentor()
    augmentor.load_bg_noise_sources([("big", str(big), 1.0), ("small", str(small), 1.0)])

    # Sources sharing a folder basename do not overwrite each other.
    assert len(augmentor.bg_noise) == 42
    draws = Counter(augmentor._draw_noise_id().split("/")[0] for _ in range(4000))
    assert 0.45 < draws["small"] / 4000 < 0.55  # by file count it would be ~5%
