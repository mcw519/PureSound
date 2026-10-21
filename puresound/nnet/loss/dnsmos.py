"""DNSMOS P.835 as a differentiable training term.

DNSMOS (Reddy et al., 2022) predicts the ITU-T P.835 listening-test scores --
SIG (speech quality), BAK (background intrusiveness), OVRL -- from the enhanced
signal alone, with a small CNN that reads 9.01 s of 16 kHz audio. It is the
gate's monitor for real recordings, where no clean reference exists. Run through
:class:`~puresound.nnet.lobe.onnx_module.OnnxModule` the same network gives a
gradient, so the enhancer can be trained against the judge it is measured by.

The model is frozen; only the enhancer moves. A no-reference judge can be
gamed, so this term is meant to sit beside reference-based terms and a learned
critic, with a small weight and a short schedule -- it steers an operating
point, it does not define one.

Scores enter the loss through the same polynomial calibration the metric
applies (``_polyfit_val`` in torchmetrics), so a configured floor is in MOS
units. The loss is ``sum_k w_k * (5 - MOS_k) / 4`` over the scores with a
non-zero weight, plus ``floor_weight * relu(sig_floor - SIG)`` when a SIG floor
is set: BAK can then be pushed only as far as SIG stays above the floor.

Rows shorter than 9.01 s are tiled to that length (the network pools over
time with a global max, so repetition changes little); longer rows use the
first 9.01 s.

The judge reads absolute log power (no normalisation inside the graph), and
its scores move with level: on real recordings a 3 dB change shifts SIG by
0.2-0.3 on average and up to 1.3. Left at its natural level the term would
reward the enhancer for simply turning the output down or up. Each row is
therefore brought to ``reference_rms`` by a gradient-free factor before the
judge, which closes that shortcut; the default (-26 dBFS) sits within 2 dB of
where the enhancer's outputs and the gate's clips lie. ``reference_rms=None``
scores rows at their natural level instead, which is what the metric in the
gate sees.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from puresound.nnet.lobe.onnx_module import OnnxModule

#: torchmetrics' default location for the DNSMOS files.
DEFAULT_MODEL = "~/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
INPUT_SAMPLES = 144160  # 9.01 s at 16 kHz
# torchmetrics._polyfit_val (non-personalized): highest power first.
_POLY = {
    "sig": (-0.08397278, 1.22083953, 0.0052439),
    "bak": (-0.13166888, 1.60915514, -0.39604546),
    "ovr": (-0.06766283, 1.11546468, 0.04602535),
}


def _poly(x: torch.Tensor, c) -> torch.Tensor:
    return c[0] * x * x + c[1] * x + c[2]


class DNSMOSLoss(nn.Module):
    """Pull the enhanced signal's predicted P.835 scores up.

    Args:
        model_path: the ``sig_bak_ovr.onnx`` file (torchmetrics downloads it on
            first metric use).
        sig_weight, bak_weight, ovr_weight: weights on ``(5 - MOS)/4`` per score.
        sig_floor: when set, ``floor_weight * relu(sig_floor - SIG)`` is added,
            so pushing BAK cannot pay with SIG below this value (MOS units).
        floor_weight: multiplier on the floor hinge.
        sample_rate: the enhancer's rate; DNSMOS needs 16 kHz, anything else is
            rejected rather than resampled inside the loss.
        max_rows: score at most this many rows per call (a random subset). The
            judge reads 9 s per row, so its activations cost ~0.45 GB a row;
            a subset keeps the term affordable next to the enhancer's own step.
        reference_rms: every row is scaled to this RMS (0.05 is -26 dBFS) before
            the judge, by a gradient-free factor, so the level-sensitive judge
            cannot be pleased by a level change; ``None`` scores rows at their
            natural level. See :meth:`scores`.
        silence_rms: rows quieter than this (RMS, 1e-3 is -60 dBFS) are not scored.
        max_scale: cap on the per-row scale factor, so a nearly suppressed row
            is not amplified without bound to reach the reference level.
    """

    required_inputs = ("enhanced",)

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        sig_weight: float = 0.0,
        bak_weight: float = 1.0,
        ovr_weight: float = 0.0,
        sig_floor: float | None = None,
        floor_weight: float = 1.0,
        sample_rate: int = 16000,
        max_rows: int | None = None,
        reference_rms: float | None = 0.05,
        silence_rms: float = 1e-3,
        max_scale: float = 20.0,
    ):
        super().__init__()
        if int(sample_rate) != 16000:
            raise ValueError("DNSMOSLoss works at 16 kHz only")
        path = Path(os.path.expanduser(model_path))
        if not path.is_file():
            raise FileNotFoundError(f"DNSMOS model not found at {path}; run any DNSMOS metric once to download it")
        self.judge = OnnxModule(path)
        self.judge.eval()
        self.weights = {"sig": float(sig_weight), "bak": float(bak_weight), "ovr": float(ovr_weight)}
        if not any(self.weights.values()) and sig_floor is None:
            raise ValueError("DNSMOSLoss has no active term: set a weight or a SIG floor")
        self.sig_floor = None if sig_floor is None else float(sig_floor)
        self.floor_weight = float(floor_weight)
        self.max_rows = None if max_rows is None else int(max_rows)
        self.reference_rms = None if reference_rms is None else float(reference_rms)
        self.silence_rms = float(silence_rms)
        self.max_scale = float(max_scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.judge.eval()
        return self

    def scores(self, wav: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Calibrated SIG / BAK / OVR per row (``[B]`` each, with gradient) and a
        ``[B]`` mask of the rows that were scored.

        The judge scores absolute level along with everything else, so a row
        is first brought to ``reference_rms`` by a factor that carries no
        gradient: the enhancer is then judged on what it does to the signal,
        never on how loud it leaves it. A silent row (below ``silence_rms``)
        is left out rather than scaled up: inside the graph it would divide 0
        by 0, and that NaN would poison the gradient of every row in the
        batch, masked or not.
        """
        if wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)
        wav = wav.float()
        rms = wav.pow(2).mean(-1).sqrt()
        keep = rms > self.silence_rms
        if not bool(keep.any()):
            zeros = wav.new_zeros(wav.shape[0])
            return {k: zeros for k in ("sig", "bak", "ovr")}, keep
        idx = keep.nonzero(as_tuple=True)[0]
        x = wav[idx]
        if self.reference_rms is not None:
            scale = (self.reference_rms / rms[idx]).clamp(1.0 / self.max_scale, self.max_scale).detach()
            x = x * scale[:, None]
        n = x.shape[-1]
        if n < INPUT_SAMPLES:
            x = x.repeat(1, INPUT_SAMPLES // n + 1)
        x = x[..., :INPUT_SAMPLES]
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = self.judge(x)  # [kept, 3] = sig, bak, ovr before calibration
        full = wav.new_zeros(wav.shape[0], 3)  # an excluded row reads 0 on every score
        full[idx] = torch.stack([_poly(raw[:, i], _POLY[k]) for i, k in enumerate(("sig", "bak", "ovr"))], 1)
        return {k: full[:, i] for i, k in enumerate(("sig", "bak", "ovr"))}, keep

    def forward(self, enh: torch.Tensor) -> torch.Tensor:
        if self.max_rows is not None and enh.shape[0] > self.max_rows:
            idx = torch.randperm(enh.shape[0], device=enh.device)[: self.max_rows]
            enh = enh[idx]
        s, keep = self.scores(enh)
        w_rows = keep.float()
        n = w_rows.sum().clamp_min(1.0)
        loss = enh.sum() * 0.0   # stays attached to the graph when every row is silent
        for k, w in self.weights.items():
            if w:
                loss = loss + w * (((5.0 - s[k]) / 4.0) * w_rows).sum() / n
        if self.sig_floor is not None:
            loss = loss + self.floor_weight * (torch.relu(self.sig_floor - s["sig"]) * w_rows).sum() / n
        return loss
