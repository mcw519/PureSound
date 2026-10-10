"""Speech-shaped noise has speech's spectrum, not white noise's."""
import numpy as np

from puresound.dataset.corpus.ssn import envelope, long_term_spectrum, shaped_noise


def test_shaped_noise_follows_the_target_spectrum():
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 4) / 16000
    lowpass_speechlike = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate(range(150, 3000, 150)))
    spectrum = long_term_spectrum([lowpass_speechlike * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))])
    noise = shaped_noise(spectrum, 16000 * 4, rng)
    power = np.abs(np.fft.rfft(noise)) ** 2
    bins = np.fft.rfftfreq(len(noise), 1 / 16000)
    assert power[bins < 3000].sum() > 50 * power[bins > 5000].sum()
    assert abs(np.sqrt(np.mean(noise**2)) - 1.0) < 1e-6


def test_envelope_is_tiled_and_mean_one():
    env = envelope(np.abs(np.random.default_rng(1).standard_normal(8000)), 20000)
    assert env.shape == (20000,) and abs(env.mean() - 1.0) < 0.05
