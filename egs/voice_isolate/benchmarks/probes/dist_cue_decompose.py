"""Separate WHEN from WHAT, from first principles.

A real signal is completely described by its magnitude spectrum (which
frequencies are present) and its phase spectrum (when they happen). So the two
manipulations that isolate them are:

  phase-randomised : magnitude EXACTLY preserved, phase replaced by noise
                     -> "same frequency content, timing destroyed"
  magnitude-flattened : phase EXACTLY preserved, magnitude made flat
                     -> "same timing, frequency content destroyed"

Both preserve total energy (Parseval), so level is not a confound in either.

Unlike the earlier smear, neither forces the peak index: the probe convolves
directly and never reads argmax, so leaving the peak alone keeps the
manipulation pure. That also means these are not directly comparable to the
smear numbers, which did force it.
"""
import torch


def _split(window):
    X = torch.fft.rfft(window.double())
    return X.abs(), X.angle(), X.shape[-1]


def phase_randomise(window, generator=None):
    """Keep |X| exactly, replace its phase. DC and Nyquist stay real, or the
    inverse transform is not a real signal."""
    mag, _, n = _split(window)
    ph = torch.rand(n, generator=generator, dtype=torch.float64) * 2 * torch.pi
    ph[0] = 0.0
    if window.shape[-1] % 2 == 0:
        ph[-1] = 0.0
    Y = mag * torch.exp(1j * ph)
    out = torch.fft.irfft(Y, n=window.shape[-1])
    return out.to(window.dtype)


def magnitude_flatten(window):
    """Keep the phase exactly, make every magnitude equal, same total energy."""
    mag, ph, n = _split(window)
    energy = (mag ** 2).sum()
    flat = torch.full_like(mag, float((energy / n).sqrt()))
    Y = flat * torch.exp(1j * ph)
    out = torch.fft.irfft(Y, n=window.shape[-1])
    # Parseval only holds up to the rfft's half-spectrum weighting; renormalise
    # in the time domain so the window's energy is exactly what it was.
    e_in = window.double().pow(2).sum()
    e_out = out.pow(2).sum().clamp_min(1e-20)
    return (out * (e_in / e_out).sqrt()).to(window.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.zeros(800); w[0] = 1.0; w[40] = 0.5; w[120] = -0.2
    g = torch.Generator().manual_seed(0)
    pr = phase_randomise(w, g)
    mf = magnitude_flatten(w)
    for name, y in (("phase-randomised", pr), ("magnitude-flattened", mf)):
        M0, P0, _ = _split(w); M1, P1, _ = _split(y)
        dmag = float((M0 - M1).abs().max() / M0.abs().max())
        dph = float((torch.remainder(P0 - P1 + torch.pi, 2*torch.pi) - torch.pi).abs().max())
        e = float(y.double().pow(2).sum() / w.double().pow(2).sum())
        print(f"{name:22s} |mag| rel err {dmag:.2e}   max phase err {dph:.2e} rad   energy x{e:.6f}")
