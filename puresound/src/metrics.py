import numpy as np
import torch
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from puresound.nnet.loss.sdr import si_snr
from pystoi.stoi import stoi


class Metrics():
    def __init__(self) -> None:
        pass

    @staticmethod
    def check_shape(clean: torch.Tensor, enhanced: torch.Tensor, retun_as_tensor: bool = False):
        if clean.shape[0] != 1: clean = clean[0, ...]
        if enhanced.shape[0] != 1: enhanced = enhanced[0, ...]

        if clean.dim() != 1: clean = clean.squeeze()
        if enhanced.dim() != 1: enhanced = enhanced.squeeze()

        # align from start
        if clean.shape != enhanced.shape:
            if clean.shape[-1] > enhanced.shape[-1]:
                clean = clean[:enhanced.shape[-1]]
            else:
                enhanced = enhanced[:clean.shape[-1]]

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
        
        return pesq(16000, clean, enhanced, 'wb')
    
    @staticmethod
    def pesq_nb(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)
        
        return pesq(8000, clean, enhanced, 'nb')
    
    @staticmethod
    def stoi(clean: np.array, enhanced: np.array, sr: int = 16000):
        clean, enhanced = Metrics.check_shape(clean, enhanced)
        
        return stoi(clean, enhanced, sr)
    
    @staticmethod
    def bss_sdr(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)

        return bss_eval_sources(clean, enhanced, False)[0][0]
    
    @staticmethod
    def sisnr(clean: np.array, enhanced: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced)
        
        return si_snr(torch.from_numpy(enhanced).view(1, -1), torch.from_numpy(clean).view(1, -1)).item()

    @staticmethod
    def sisnr_imp(clean: np.array, enhanced: np.array, noisy: np.array):
        clean, enhanced = Metrics.check_shape(clean, enhanced, retun_as_tensor=True)
        clean, noisy = Metrics.check_shape(clean.view(1, -1), noisy, retun_as_tensor=True)
        improvement = si_snr(enhanced.reshape(1, -1), clean.reshape(1, -1)).reshape(-1) - si_snr(noisy.reshape(1, -1), clean.reshape(1, -1)).reshape(-1)
        
        return improvement.item()

    @staticmethod
    def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true, y_pred = Metrics.check_shape(y_true, y_pred, retun_as_tensor=True)
        tp = torch.sum(torch.logical_and(y_pred, y_true))
        tn = torch.sum(torch.logical_and(torch.logical_not(y_pred), torch.logical_not(y_true)))
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
