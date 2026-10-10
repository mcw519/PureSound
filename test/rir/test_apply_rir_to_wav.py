import pytest
import torch
from torchaudio.functional import highpass_biquad, lowpass_biquad

from egs.rir_generation.tools.audition.apply_rir_to_wav import apply_rir_to_wav


@pytest.mark.parametrize(
    "output_layout,expected_shape,expected_first_sample",
    [
        ("rir-channels", (5, 32), torch.arange(1, 6).float()),
        ("mono-sum", (1, 32), torch.tensor([15.0])),
    ],
)
def test_apply_rir_to_wav_lays_out_one_channel_per_rir_or_a_mono_sum(
    output_layout, expected_shape, expected_first_sample
):
    wav = torch.zeros(1, 32)
    wav[0, 0] = 1.0
    rir = torch.zeros(5, 8)
    rir[:, 0] = torch.arange(1, 6).float()

    wet = apply_rir_to_wav(
        wav=wav,
        rir=rir,
        sample_rate=8000,
        length_mode="same",
        output_layout=output_layout,
    )

    assert tuple(wet.shape) == expected_shape
    assert torch.allclose(wet[:, 0], expected_first_sample)


def test_apply_rir_to_wav_can_trim_output_low_band():
    sample_rate = 8000
    t = torch.arange(sample_rate).float() / sample_rate
    wav = (
        torch.sin(2.0 * torch.pi * 200.0 * t)
        + torch.sin(2.0 * torch.pi * 2000.0 * t)
    ).view(1, -1)
    rir = torch.zeros(1, 16)
    rir[0, 0] = 1.0

    wet = apply_rir_to_wav(
        wav=wav,
        rir=rir,
        sample_rate=sample_rate,
        output_low_gain_db=-12.0,
        output_low_cutoff_hz=1000.0,
    )
    low = lowpass_biquad(wet, sample_rate, 1000.0)
    high = highpass_biquad(wet, sample_rate, 1000.0)

    assert torch.sqrt(torch.mean(low**2)) < torch.sqrt(torch.mean(high**2))
