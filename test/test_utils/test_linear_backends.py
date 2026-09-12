"""`apply_linear` is the guarantee that a linear stage stays linear.

Every DSP backend this package filters and gains with saturates at digital full
scale, and none of them says so: `torchaudio.functional.lfilter` clamps to
[-1, 1] unless told otherwise, every `biquad` is built on it, and every sox
effect round-trips through a fixed-point sample format. Upstream of the
converter the signal is a sound pressure, routinely above 1.0, and those
ceilings turn a microphone response into a waveshaper.

`test_device_chain.py` pins the consequence -- the analogue path is linear.
This pins the mechanism, including the cases the chain never reaches.
"""

import math
import pytest
import torch
import torchaudio

from puresound.audio.dsp import FULL_SCALE, apply_linear, wav_resampling

SR = 16000


def _hot(peak=4.0):
    time = torch.arange(SR, dtype=torch.float32) / SR
    return (peak * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)


def _saturating(gain=1.0):
    """A backend that behaves the way every real one here does."""

    def backend(wav):
        return (wav * gain).clamp(-FULL_SCALE, FULL_SCALE)

    return backend


def test_it_recovers_the_operator_the_backend_was_hiding():
    hot = _hot()
    assert float(_saturating()(hot).abs().amax()) == pytest.approx(1.0)  # the bug
    assert torch.allclose(apply_linear(_saturating(), hot), hot, atol=1e-6)


def test_it_escalates_when_the_operator_has_more_gain_than_the_headroom():
    """A 40 dB operator blows straight through the first attempt's 6 dB.

    Backing off and retrying is the difference between a number that is right
    and a number that is quietly wrong.
    """
    hot = _hot()
    out = apply_linear(_saturating(gain=100.0), hot)
    assert torch.allclose(out, hot * 100.0, rtol=1e-5)


def test_it_gives_up_loudly_rather_than_return_a_clipped_answer():
    with pytest.raises(RuntimeError, match="saturates"):
        apply_linear(lambda w: torch.full_like(w, FULL_SCALE), _hot())


@pytest.mark.parametrize("wav", [torch.zeros(1, 64), torch.zeros(1, 0)])
def test_silence_has_no_headroom_question_to_answer(wav):
    """Scaling by 1/0 is undefined and there is nothing to protect anyway."""
    assert torch.equal(apply_linear(lambda w: w * 2.0, wav), wav * 2.0)


@pytest.mark.parametrize("backend", ["sox", "torchaudio"])
def test_both_resampler_legs_are_linear(backend):
    """The SRC stage tosses a coin between these two.

    They are supposed to differ in their anti-alias filters -- that is the point
    of offering both. They are not supposed to differ in whether they clip,
    which is what made the coin toss a confound rather than an augmentation.
    """
    hot = _hot()
    # The torchaudio leg draws a fresh anti-alias filter (width / rolloff /
    # window) on every call unless handed one, so two unpinned round trips are
    # two different filters and comparing them measures nothing.
    params = (
        {"lp_width": 16, "rolloff": 0.9, "window": "sinc_interp_hann"}
        if backend == "torchaudio"
        else None
    )

    def round_trip(wav):
        down = wav_resampling(
            wav, SR, 8000, backend=backend, torch_backend_params=params
        )
        up = wav_resampling(
            down[0], 8000, SR, backend=backend, torch_backend_params=params
        )
        return up[0]

    loud = round_trip(hot)
    quiet = round_trip(hot / 100.0) * 100.0
    error = float((loud - quiet).abs().amax() / loud.abs().amax())
    assert error < 1e-3, f"{backend} resampler is not linear: {error:.2e}"


def test_a_biquad_is_a_transfer_function_not_a_limiter():
    hot = _hot()
    naked = torchaudio.functional.highpass_biquad(hot, SR, 100.0, 0.707)
    assert float(naked.abs().amax()) == pytest.approx(1.0)  # the bug
    wrapped = apply_linear(
        lambda w: torchaudio.functional.highpass_biquad(w, SR, 100.0, 0.707), hot
    )
    assert float(wrapped.abs().amax()) > 3.9


# --------------------------------------------------------------------------- #
# compressor_gain -- a time-varying gain, so superposition survives it
# --------------------------------------------------------------------------- #


def test_compressor_gain_preserves_superposition():
    """THE reason this returns a curve instead of applying itself.

    The device chain's linear group puts the same parameters on the mixture and
    the target because that is what keeps the mixture equal to the sum of its
    parts. A gain does; the |x|^p waveshaper in apply_media_coloring does not,
    which is why that one is a source-level effect and this one is not.
    """
    from puresound.audio.dsp import compressor_gain

    torch.manual_seed(0)
    near, far = torch.randn(1, 16000) * 0.2, torch.randn(1, 16000) * 0.1
    gain = compressor_gain(near + far, 16000, threshold_db=-30.0, ratio=4.0)
    assert torch.allclose(gain * (near + far), gain * near + gain * far, atol=1e-6)


def test_compressor_gain_actually_compresses():
    """Loud down, quiet up -- the envelope must flatten, or this augmentation is
    teaching nothing about the cue it exists to attack."""
    from puresound.audio.dsp import compressor_gain

    torch.manual_seed(0)
    sr = 16000
    x = torch.cat([torch.randn(1, sr) * 0.5, torch.randn(1, sr) * 0.02], dim=-1)
    gain = compressor_gain(x, sr, threshold_db=-30.0, ratio=4.0)
    assert float(gain[0, :sr].mean()) < float(gain[0, sr:].mean())
    before = 20 * math.log10(float(x[0, :sr].std() / x[0, sr:].std()))
    after = 20 * math.log10(float((x * gain)[0, :sr].std() / (x * gain)[0, sr:].std()))
    assert after < before - 5.0, f"contrast {before:.1f} -> {after:.1f} dB"


def test_compressor_gain_is_level_neutral():
    """Unit-mean makeup, so compression and a gain change are separable.

    Without it a model could satisfy this augmentation by learning level
    invariance, which the renormalisation probe already showed it has -- and
    which is not the cue in question.
    """
    from puresound.audio.dsp import compressor_gain

    torch.manual_seed(0)
    for scale in (0.02, 0.2, 0.9):
        g = compressor_gain(torch.randn(1, 8000) * scale, 16000,
                            threshold_db=-30.0, ratio=4.0)
        assert abs(float(g.mean()) - 1.0) < 1e-4


def test_compressor_ratio_one_is_a_no_op():
    from puresound.audio.dsp import compressor_gain

    g = compressor_gain(torch.randn(1, 4000) * 0.4, 16000,
                        threshold_db=-40.0, ratio=1.0)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


@pytest.mark.parametrize("kw", [
    dict(ratio=0.5),
    dict(ratio=4.0, attack_ms=0.0),
    dict(ratio=4.0, release_ms=-1.0),
])
def test_compressor_gain_rejects_incoherent_settings(kw):
    from puresound.audio.dsp import compressor_gain

    kw.setdefault("threshold_db", -30.0)
    with pytest.raises(ValueError):
        compressor_gain(torch.randn(1, 1000) * 0.2, 16000, **kw)


def test_compressor_attack_is_faster_than_release():
    """The asymmetry IS the detector: a symmetric follower pumps between
    syllables and stops modelling a compressor."""
    from puresound.audio.dsp import compressor_gain

    sr = 16000
    burst = torch.cat([torch.zeros(1, sr // 2), torch.ones(1, sr // 4) * 0.5,
                       torch.zeros(1, sr)], dim=-1)
    g = compressor_gain(burst, sr, threshold_db=-40.0, ratio=8.0,
                        attack_ms=2.0, release_ms=200.0, makeup=False)
    onset = sr // 2
    offset = onset + sr // 4
    # gain is down within a few ms of the onset ...
    assert float(g[0, onset + int(0.004 * sr)]) < 0.6
    # ... and still down well after the burst ends
    assert float(g[0, offset + int(0.05 * sr)]) < 0.9
