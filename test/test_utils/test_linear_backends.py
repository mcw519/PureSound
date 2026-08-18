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
