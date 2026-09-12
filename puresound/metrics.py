import numpy as np
import torch
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pystoi.stoi import stoi

from puresound.nnet.loss.sdr import si_snr


_DNSMOS_METRICS = {}


def _mono_audio_tensor(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.detach().cpu().float()
    while wav.dim() > 1 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    if wav.dim() > 1:
        wav = wav[0]
    return wav.clamp(min=-1.0, max=1.0)


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def check_shape(
        clean: torch.Tensor, enhanced: torch.Tensor, retun_as_tensor: bool = False
    ):
        if clean.shape[0] != 1:
            clean = clean[0, ...]
        if enhanced.shape[0] != 1:
            enhanced = enhanced[0, ...]

        if clean.dim() != 1:
            clean = clean.squeeze()
        if enhanced.dim() != 1:
            enhanced = enhanced.squeeze()

        # align from start
        if clean.shape != enhanced.shape:
            if clean.shape[-1] > enhanced.shape[-1]:
                clean = clean[: enhanced.shape[-1]]
            else:
                enhanced = enhanced[: clean.shape[-1]]

        # convert to numpy
        clean = clean.detach().numpy()
        enhanced = enhanced.detach().numpy()

        # normalize
        clean = clean / abs(clean).max()
        enhanced = enhanced / abs(enhanced).max()

        if retun_as_tensor:
            clean = torch.from_numpy(clean)
            enhanced = torch.from_numpy(enhanced)

        return clean, enhanced

    @staticmethod
    def pesq_wb(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return pesq(16000, clean, enhanced, "wb")

    @staticmethod
    def pesq_nb(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return pesq(8000, clean, enhanced, "nb")

    @staticmethod
    def stoi(clean: np.array, enhanced: np.array, sr: int = 16000):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return stoi(clean, enhanced, sr)

    @staticmethod
    def estoi(clean: np.array, enhanced: np.array, sr: int = 16000):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return stoi(clean, enhanced, sr, extended=True)

    @staticmethod
    def bss_sdr(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return bss_eval_sources(clean, enhanced, False)[0][0]

    @staticmethod
    def sisnr(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return si_snr(
            torch.from_numpy(enhanced).view(1, -1), torch.from_numpy(clean).view(1, -1)
        ).item()

    @staticmethod
    def sisnr_imp(clean: np.array, enhanced: np.array, noisy: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced, retun_as_tensor=True)
        clean, noisy = Metrics.check_shape(
            clean.view(1, -1), noisy, retun_as_tensor=True
        )
        improvement = si_snr(enhanced.reshape(1, -1), clean.reshape(1, -1)).reshape(
            -1
        ) - si_snr(noisy.reshape(1, -1), clean.reshape(1, -1)).reshape(-1)

        return improvement.item()

    @staticmethod
    def dnsmos_p835(
        clean: torch.Tensor,
        enhanced: torch.Tensor,
        sr: int = 16000,
        personalized: bool = False,
    ):
        del clean
        try:
            from torchmetrics.audio.dnsmos import (
                DeepNoiseSuppressionMeanOpinionScore,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DNSMOS requires torchmetrics audio dependencies. "
                "Install with `uv sync`, or install librosa, onnxruntime, and requests."
            ) from exc

        key = (int(sr), bool(personalized))
        if key not in _DNSMOS_METRICS:
            _DNSMOS_METRICS[key] = DeepNoiseSuppressionMeanOpinionScore(
                fs=int(sr),
                personalized=bool(personalized),
                device="cpu",
            )

        wav = _mono_audio_tensor(enhanced)
        score = _DNSMOS_METRICS[key](wav)
        names = ["dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr"]
        return {name: float(value) for name, value in zip(names, score)}

    @staticmethod
    def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true, y_pred = Metrics.check_shape(y_true, y_pred, retun_as_tensor=True)
        tp = torch.sum(torch.logical_and(y_pred, y_true))
        tn = torch.sum(
            torch.logical_and(torch.logical_not(y_pred), torch.logical_not(y_true))
        )
        fp = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_pred))
        fn = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_true))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        f1 = f1.clamp(min=1e-7, max=1 - 1e-7)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    @staticmethod
    def noise_reduction(noisy: torch.Tensor, enhanced: torch.Tensor):
        noisy, enhanced = Metrics.check_shape(noisy, enhanced, retun_as_tensor=True)

        return 10 * torch.log10(
            torch.sum(enhanced ** 2, -1, keepdim=True)
            / torch.sum(noisy ** 2, -1, keepdim=True)
        )
