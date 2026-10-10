"""`puresound.audio.dsp`: `apply_linear` and `compressor_gain`.

`apply_linear` is the guarantee that a linear stage stays linear. Every DSP
backend this package filters and gains with saturates at digital full scale,
and none of them says so: `torchaudio.functional.lfilter` clamps to [-1, 1]
unless told otherwise, every `biquad` is built on it, and every sox effect
round-trips through a fixed-point sample format. Upstream of the converter the
signal is a sound pressure, routinely above 1.0, and those ceilings turn a
microphone response into a waveshaper. `test_device_chain.py` pins the
consequence -- the analogue path is linear; this pins the mechanism.
"""

import math

import pytest
import torch
import torchaudio

from puresound.audio.dsp import FULL_SCALE, apply_linear, compressor_gain, wav_resampling

SR = 16000


def _hot(peak=4.0):
    time = torch.arange(SR, dtype=torch.float32) / SR
    return (peak * torch.sin(2 * torch.pi * 220.0 * time)).view(1, -1)


def _saturating(gain=1.0):
    """A backend that behaves the way every real one here does."""

    def backend(wav):
        return (wav * gain).clamp(-FULL_SCALE, FULL_SCALE)

    return backend


@pytest.mark.parametrize("gain", [1.0, 100.0], ids=["unity", "more-gain-than-headroom"])
def test_it_recovers_the_operator_the_backend_was_hiding(gain):
    """At 40 dB the operator blows through the first attempt's headroom, so
    apply_linear has to back off and retry rather than return a clipped answer."""
    hot = _hot()
    assert float(_saturating(gain)(hot).abs().amax()) == pytest.approx(1.0)  # the bug
    assert torch.allclose(apply_linear(_saturating(gain), hot), hot * gain, rtol=1e-5, atol=1e-6)


def test_it_gives_up_loudly_rather_than_return_a_clipped_answer():
    with pytest.raises(RuntimeError, match="saturates"):
        apply_linear(lambda w: torch.full_like(w, FULL_SCALE), _hot())


@pytest.mark.parametrize("wav", [torch.zeros(1, 64), torch.zeros(1, 0)])
def test_silence_has_no_headroom_question_to_answer(wav):
    """Scaling by 1/0 is undefined and there is nothing to protect anyway."""
    assert torch.equal(apply_linear(lambda w: w * 2.0, wav), wav * 2.0)


def test_a_biquad_is_a_transfer_function_not_a_limiter():
    hot = _hot()
    naked = torchaudio.functional.highpass_biquad(hot, SR, 100.0, 0.707)
    assert float(naked.abs().amax()) == pytest.approx(1.0)  # the bug
    wrapped = apply_linear(
        lambda w: torchaudio.functional.highpass_biquad(w, SR, 100.0, 0.707), hot
    )
    assert float(wrapped.abs().amax()) > 3.9


@pytest.mark.parametrize("backend", ["sox", "torchaudio"])
def test_both_resampler_legs_are_linear(backend):
    """The SRC stage tosses a coin between these two. They may differ in their
    anti-alias filters -- that is the point of offering both -- but not in
    whether they clip."""
    hot = _hot()
    # The torchaudio leg draws a fresh anti-alias filter on every call unless
    # handed one, and two different filters are not comparable.
    params = (
        {"lp_width": 16, "rolloff": 0.9, "window": "sinc_interp_hann"}
        if backend == "torchaudio"
        else None
    )

    def round_trip(wav):
        down = wav_resampling(wav, SR, 8000, backend=backend, torch_backend_params=params)
        up = wav_resampling(down[0], 8000, SR, backend=backend, torch_backend_params=params)
        return up[0]

    loud = round_trip(hot)
    quiet = round_trip(hot / 100.0) * 100.0
    error = float((loud - quiet).abs().amax() / loud.abs().amax())
    assert error < 1e-3, f"{backend} resampler is not linear: {error:.2e}"


# --------------------------------------------------------------------------- #
# compressor_gain -- a time-varying gain, so superposition survives it
# --------------------------------------------------------------------------- #


def test_compressor_gain_preserves_superposition():
    """The reason this returns a curve instead of applying itself: the device
    chain puts the same gain on the mixture and the target, which keeps the
    mixture equal to the sum of its parts."""
    torch.manual_seed(0)
    near, far = torch.randn(1, 16000) * 0.2, torch.randn(1, 16000) * 0.1
    gain = compressor_gain(near + far, 16000, threshold_db=-30.0, ratio=4.0)
    assert torch.allclose(gain * (near + far), gain * near + gain * far, atol=1e-6)


def test_compressor_gain_flattens_the_envelope_at_unit_mean():
    """Loud down, quiet up; and unit-mean makeup, so compression and a level
    change stay separable. Ratio 1 is the identity."""
    torch.manual_seed(0)
    x = torch.cat([torch.randn(1, SR) * 0.5, torch.randn(1, SR) * 0.02], dim=-1)
    gain = compressor_gain(x, SR, threshold_db=-30.0, ratio=4.0)
    assert float(gain[0, :SR].mean()) < float(gain[0, SR:].mean())
    before = 20 * math.log10(float(x[0, :SR].std() / x[0, SR:].std()))
    after = 20 * math.log10(float((x * gain)[0, :SR].std() / (x * gain)[0, SR:].std()))
    assert after < before - 5.0, f"contrast {before:.1f} -> {after:.1f} dB"

    for scale in (0.02, 0.2, 0.9):
        g = compressor_gain(torch.randn(1, 8000) * scale, SR, threshold_db=-30.0, ratio=4.0)
        assert abs(float(g.mean()) - 1.0) < 1e-4

    g = compressor_gain(torch.randn(1, 4000) * 0.4, SR, threshold_db=-40.0, ratio=1.0)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


@pytest.mark.parametrize("kw", [
    dict(ratio=0.5),
    dict(ratio=4.0, attack_ms=0.0),
    dict(ratio=4.0, release_ms=-1.0),
])
def test_compressor_gain_rejects_incoherent_settings(kw):
    with pytest.raises(ValueError):
        compressor_gain(torch.randn(1, 1000) * 0.2, SR, threshold_db=-30.0, **kw)


def test_compressor_attack_is_faster_than_release():
    """The asymmetry is the detector: a symmetric follower pumps between
    syllables and stops modelling a compressor."""
    burst = torch.cat([torch.zeros(1, SR // 2), torch.ones(1, SR // 4) * 0.5,
                       torch.zeros(1, SR)], dim=-1)
    g = compressor_gain(burst, SR, threshold_db=-40.0, ratio=8.0,
                        attack_ms=2.0, release_ms=200.0, makeup=False)
    onset = SR // 2
    offset = onset + SR // 4
    assert float(g[0, onset + int(0.004 * SR)]) < 0.6   # down within a few ms
    assert float(g[0, offset + int(0.05 * SR)]) < 0.9   # still down after the burst


def test_a_sampleless_file_is_returned_empty_by_the_neutral_resampler(tmp_path):
    """The datasets retry on an empty waveform; a resampler that raises on one
    would take the DataLoader worker down instead."""
    import soundfile as sf

    from puresound.audio.io import AudioIO

    path = tmp_path / "empty.wav"
    sf.write(str(path), torch.zeros(0).numpy(), 8000)

    wav, rate = AudioIO.open(str(path), resample_to=16000)

    assert wav.shape == (1, 0) and rate == 16000
