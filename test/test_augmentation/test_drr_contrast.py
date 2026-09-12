import json
import random

import pytest
import torch
import torchaudio

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.impulse_response import compute_drr_db


SR = 16000


def _impulse(sr=SR, length=4800, peak_at=160, tail_level=0.05):
    """Unit direct peak plus a flat tail well past the 2.5 ms window."""
    rir = torch.zeros(1, length)
    rir[0, peak_at] = 1.0
    tail_start = peak_at + int(round(2.5e-3 * sr)) + 8
    rir[0, tail_start:] = tail_level
    return rir


def _write_room(root, room_id, distances, sr=SR, length=4800):
    sample_dir = root / room_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels = ["near_0", "near_1", "far_0", "far_1", "far_2"]
    rir = torch.zeros(5, length)
    for ch, dist in enumerate(distances):
        peak = int(round(dist / 343.0 * sr))
        rir[ch, peak] = 1.0
        rir[ch, peak + 100 :] = 0.05
    torchaudio.save(str(sample_dir / "rir_5ch.wav"), rir, sr, encoding="PCM_F")
    channel_map = [
        {"channel": ch, "label": labels[ch], "distance_m": float(dist)}
        for ch, dist in enumerate(distances)
    ]
    metadata = {"scene": {"rt60": 0.45, "channel_map": channel_map}}
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _knobbed(prob=1.0, near=(3.0, 3.0), far=(3.0, 3.0)):
    aug = AudioEffectAugmentor()
    aug.init_drr_contrast(
        {
            "used": True,
            "prob": prob,
            "near_boost_db": list(near),
            "far_cut_db": list(far),
            "direct_window_ms": 2.5,
        }
    )
    return aug


def test_shift_lands_exactly_on_the_pipeline_drr_measure():
    rir = _impulse()
    before = compute_drr_db(rir, SR, 2.5)

    aug = _knobbed(near=(3.0, 3.0), far=(4.0, 4.0))
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


def test_only_role_aware_channels_are_touched():
    rir = _impulse()
    aug = _knobbed()
    for role in ("source", "", None):
        out, meta = aug._apply_drr_contrast(rir, SR, role, {"drr_db": 1.0})
        assert out is rir
        assert meta == {"drr_db": 1.0}


def test_unconfigured_knob_consumes_no_randomness():
    """Old configs must stay bit-identical: no draws unless the knob is on."""
    rir = _impulse()
    aug = AudioEffectAugmentor()  # knob never initialized
    random.seed(1234)
    state = random.getstate()
    out, meta = aug._apply_drr_contrast(rir, SR, "foreground", {"drr_db": 1.0})
    assert out is rir and meta == {"drr_db": 1.0}
    assert random.getstate() == state

    # and the configured knob DOES consume the stream (one gate + one uniform)
    knobbed = _knobbed()
    knobbed._apply_drr_contrast(rir, SR, "foreground", {"drr_db": 1.0})
    assert random.getstate() != state


def test_clean_target_reuse_sees_the_shifted_impulse(tmp_path):
    """The cache must hold the modified RIR so mixture and target agree."""
    root = tmp_path / "bank"
    _write_room(root, "room_000000", [0.5, 0.8, 2.5, 3.5, 4.5])
    aug = _knobbed(near=(4.0, 4.0))
    aug.init_room_bank({"used": True, "folder": str(root)})

    scene = aug.sample_room_scene()
    wav = torch.randn(1, SR)
    _mix, (rir_id, info) = aug.apply_rir(
        wav=wav, rir_mode="full", sr=SR, room_scene=scene, source_role="foreground"
    )
    shift = info["metadata"]["drr_contrast_shift_db"]
    assert shift == pytest.approx(4.0)

    _clean, (rir_id2, info2) = aug.apply_rir(
        wav=wav, rir_id=rir_id, rir_mode="direct", sr=SR
    )
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
    rir = _impulse()
    before = compute_drr_db(rir, SR, 2.5)

    random.seed(77)
    state = random.getstate()
    near, m_near = aug._apply_drr_contrast(rir, SR, "foreground", {"source_receiver_distance": 0.5})
    far, m_far = aug._apply_drr_contrast(rir, SR, "interferer", {"source_receiver_distance": 4.0})
    assert random.getstate() == state  # deterministic mode never touches the RNG

    import math
    assert compute_drr_db(near, SR, 2.5) - before == pytest.approx(-3.2 * math.log10(0.5), abs=0.05)
    assert compute_drr_db(far, SR, 2.5) - before == pytest.approx(-3.2 * math.log10(4.0), abs=0.05)
    # role does not matter -- only distance does; repeat draws are identical
    far2, _ = aug._apply_drr_contrast(rir, SR, "foreground", {"source_receiver_distance": 4.0})
    assert torch.equal(far2, far)
    # missing/invalid distance -> untouched
    same, _ = aug._apply_drr_contrast(rir, SR, "interferer", {"rt60": 0.4})
    assert same is rir


def test_init_rejects_bad_options():
    aug = AudioEffectAugmentor()
    with pytest.raises(ValueError, match="prob"):
        aug.init_drr_contrast({"used": True, "prob": 0.0})
    with pytest.raises(ValueError, match="ranges"):
        aug.init_drr_contrast({"used": True, "near_boost_db": [4.0, 1.0]})
    with pytest.raises(ValueError, match="unknown drr_contrast options"):
        aug.init_drr_contrast({"used": True, "near_boost": [0, 4]})
