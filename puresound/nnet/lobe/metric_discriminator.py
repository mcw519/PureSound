"""A metric discriminator: a small network that learns to predict PESQ.

MetricGAN's idea, in the form CMGAN and MP-SENet use: a conv net reads the
clean and the enhanced spectrum side by side and regresses the enhanced one's
normalised PESQ, ``(pesq - 1) / 3.5`` in ``[0, 1]``. The enhancer is then trained
to make it say 1. PESQ itself is not differentiable; this is a learned,
differentiable stand-in for it, retrained continuously as the enhancer moves.

It is a training-time module only. Nothing at inference reads it, and the
streaming export never sees it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    """``beta * sigmoid(slope * x)`` with a learnable slope.

    ``beta`` slightly above 1 (MetricGAN+ uses 1.2) lets the output reach the
    clean-vs-clean target of 1.0 without driving the logit to infinity.
    """

    def __init__(self, features: int = 1, beta: float = 1.2):
        super().__init__()
        self.beta = float(beta)
        self.slope = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    """``D(reference, estimate) -> normalised metric(s)`` per row.

    CMGAN's discriminator: four strided 4x4 convs over the pair of power-law
    compressed magnitudes (|X|^0.3, the loudness domain PESQ works in), spectral
    norm and instance norm, max-pooled to a vector, then two linear layers.

    With ``n_outputs > 1`` the last layer predicts several metrics at once --
    one shared trunk, one output and one sigmoid slope per metric -- so a
    critic for PESQ, eSTOI and DNSMOS is one network learning them jointly.

    Args:
        n_fft, hop: STFT of the waveforms it reads (CMGAN uses 400 / 100 at
            16 kHz).
        ndf: width of the first conv; doubles per layer.
        beta: ceiling of the output sigmoid.
        n_outputs: metrics predicted per row.
    """

    def __init__(
        self, n_fft: int = 400, hop: int = 100, ndf: int = 16, beta: float = 1.2, n_outputs: int = 1
    ):
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        sn = nn.utils.spectral_norm

        def block(c_in, c_out):
            return [
                sn(nn.Conv2d(c_in, c_out, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(c_out, affine=True),
                nn.PReLU(c_out),
            ]

        self.layers = nn.Sequential(
            *block(2, ndf), *block(ndf, ndf * 2), *block(ndf * 2, ndf * 4), *block(ndf * 4, ndf * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(ndf * 4),
            sn(nn.Linear(ndf * 4, self.n_outputs)),
            LearnableSigmoid(self.n_outputs, beta=beta),
        )

    def magnitude(self, wav: torch.Tensor) -> torch.Tensor:
        """``[B, T]`` waveform -> ``[B, F, frames]`` |STFT|^0.3, always in fp32.

        STFT under bf16 autocast is not supported and the compression is a
        power of a small number, so this runs with autocast off.
        """
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = torch.stft(
                wav.float(), self.n_fft, self.hop, self.n_fft,
                self.window.to(wav.device), return_complex=True,
            )
            return spec.abs().clamp_min(1e-8).pow(0.3)

    def forward(self, reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """reference, estimate: ``[B, T]`` waveforms -> ``[B]`` scores, or
        ``[B, n_outputs]`` when predicting several metrics."""
        n = min(reference.shape[-1], estimate.shape[-1])
        pair = torch.stack(
            [self.magnitude(reference[..., :n]), self.magnitude(estimate[..., :n])], dim=1
        )
        out = self.layers(pair)
        return out.reshape(-1) if self.n_outputs == 1 else out
