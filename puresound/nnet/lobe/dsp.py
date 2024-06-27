from typing import Dict, Tuple

import torch
import torch.nn as nn

from puresound.audio.dsp import get_biquad_params


class FrequecyEQLayer(nn.Module):
    """Defined trainable Parametric EQ"""

    def __init__(
        self,
        n_fft: int = 512,
        sample_rate: int = 16000,
        eq_band_gain: Tuple[float] = (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),
        eq_band_cutoff: Tuple[float] = (500, 1000, 1500, 2500, 3500, 5500, 6000),
        eq_band_q_factor: Tuple[float] = (
            0.707,
            0.707,
            0.707,
            0.707,
            0.707,
            0.707,
            0.707,
        ),
        low_shelf_gain_dB: float = 0.0,
        low_shelf_cutoff_freq: float = 80,
        low_shelf_q_factor: float = 0.707,
        high_shelf_gain_dB: float = 0.0,
        high_shelf_cutoff_freq: float = 7800,
        high_shelf_q_factor: float = 0.707,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.sr = sample_rate
        self.eq_band_gain = eq_band_gain
        self.eq_band_cutoff = eq_band_cutoff
        self.eq_band_q_factor = eq_band_q_factor
        self.low_shelf_gain_dB = low_shelf_gain_dB
        self.low_shelf_cutoff_freq = low_shelf_cutoff_freq
        self.low_shelf_q_factor = low_shelf_q_factor
        self.high_shelf_gain_dB = high_shelf_gain_dB
        self.high_shelf_cutoff_freq = high_shelf_cutoff_freq
        self.high_shelf_q_factor = high_shelf_q_factor
        self.trainable = trainable

        peq_weights = self.init_eq_weight().float().view(-1, 1)
        if trainable:
            peq_weights = nn.Parameter(peq_weights, requires_grad=True)
            self.register_parameter("peq", peq_weights)
        else:
            self.register_buffer("peq", peq_weights)

    def init_eq_weight(self):
        b_list = []
        a_list = []

        b, a = get_biquad_params(
            gain_dB=self.low_shelf_gain_dB,
            cutoff_freq=self.low_shelf_cutoff_freq,
            q_factor=self.low_shelf_q_factor,
            sample_rate=self.sr,
            filter_type="low_shelf",
        )
        b_list.append(b)
        a_list.append(a)

        for i in range(len(self.eq_band_gain)):
            b, a = get_biquad_params(
                gain_dB=self.eq_band_gain[i],
                cutoff_freq=self.eq_band_cutoff[i],
                q_factor=self.eq_band_q_factor[i],
                sample_rate=self.sr,
                filter_type="peaking",
            )
            b_list.append(b)
            a_list.append(a)

        b, a = get_biquad_params(
            gain_dB=self.high_shelf_gain_dB,
            cutoff_freq=self.high_shelf_cutoff_freq,
            q_factor=self.high_shelf_q_factor,
            sample_rate=self.sr,
            filter_type="high_shelf",
        )
        b_list.append(b)
        a_list.append(a)

        self.n_eq = len(a_list)

        b = torch.stack([torch.from_numpy(x) for x in b_list])
        a = torch.stack([torch.from_numpy(x) for x in a_list])

        B = torch.fft.rfft(b, self.n_fft)
        A = torch.fft.rfft(a, self.n_fft)

        H = B / A
        H = torch.prod(H, dim=0).view(-1)
        H = H.abs()
        return H

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor has shape [..., C, T]
        """
        x = self.peq * x
        return x

    @property
    def get_args(self) -> Dict:
        return {
            "n_fft": self.n_fft,
            "sample_rate": self.sr,
            "eq_band_gain": self.eq_band_gain,
            "eq_band_cutoff": self.eq_band_cutoff,
            "eq_band_q_factor": self.eq_band_q_factor,
            "low_shelf_gain_dB": self.low_shelf_gain_dB,
            "low_shelf_cutoff_freq": self.low_shelf_cutoff_freq,
            "low_shelf_q_factor": self.low_shelf_q_factor,
            "high_shelf_gain_dB": self.high_shelf_gain_dB,
            "high_shelf_cutoff_freq": self.high_shelf_cutoff_freq,
            "high_shelf_q_factor": self.high_shelf_q_factor,
            "trainable": self.trainable,
        }
