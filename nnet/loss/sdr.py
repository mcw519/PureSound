from typing import Optional

import torch
import torch.nn as nn


class SDRLoss(nn.Module):
    def __init__(self, scaled: bool = True, scale_dependent: bool = False, zero_mean: bool = True, source_aggregated: bool = False, sdr_max: int = None,
                    eps: float = 1e-8, reduction: bool = True) -> None:
        """
        Signal SDR/SNR loss function and its variations.

        Args:
            scaled: if true, scaling the target signal
            scale_dependent: loss function considered the volumn scale or not
            zero_mean: input normalize the mean to 0
            source_aggregated: source aggregated SDR loss, this mode need different source such as target and interference speech
            sdr_max: if not None, used Soft-maximum threshold, named t-SDR
            eps: small value protect divide zero case
            reduction: is Fasle, return batch result
        """
        super().__init__()
        self.scaled = scaled
        self.scale_dependent = scale_dependent
        self.zero_mean = zero_mean
        self.source_aggregated = source_aggregated
        self.sdr_max = sdr_max
        self.eps = eps
        self.reduction = reduction

    @classmethod
    def init_mode(cls, loss_func: str = 'sisnr', reduction: bool = True) -> None:
        """
        Init loss function module by alias name.\n
        You can implemented different SDR loss here.

        Args:
            loss_func: loss name, include (sisnr, sdsdr, sdr, tsdr, sasdr, sasisnr, satsdr)

        Raises:
            NameError: if loss_func not in (sisnr, sdsdr, sdr, tsdr, sasdr, sasisnr, satsdr)
        """
        loss_func = loss_func.lower()

        if loss_func not in ('sisnr', 'sdsdr', 'sdr', 'tsdr', 'sasdr', 'sasisnr', 'satsdr'):
            raise NameError

        if loss_func == 'sisnr' or loss_func in 'sdsdr' or loss_func == 'sasisdr':
            scaled = True
        else:
            scaled = False
        
        if loss_func == 'sdsdr':
            scale_dependent = True
        else:
            scale_dependent = False
        
        if loss_func == 'sasdr' or loss_func == 'sasisnr' or loss_func == 'satsdr':
            source_aggregated = True
        else:
            source_aggregated = False
        
        if loss_func == 'tsdr' or loss_func == 'satsdr':
            sdr_max = 30
        else:
            sdr_max = None
        
        print(f"init loss function: {loss_func}")
        return cls(scaled=scaled, scale_dependent=scale_dependent, zero_mean=True, source_aggregated=source_aggregated, sdr_max=sdr_max, eps=1e-8, reduction=reduction)
    
    def forward(self, s1: torch.Tensor, s2: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Compute SDR loss.
        
        Args:
            s1: enhanced signal tensor
            s2: reference signal tensor
            threshold: set SDR loss range (minimum)
        
        Returns:
            loss
        """
        self.check_input_shape(s1)
        self.check_input_shape(s2)

        if self.zero_mean:
            s1 = self.apply_zero_mean(s1)
            s2 = self.apply_zero_mean(s2)
        
        s1_s2_norm = self.l2_norm(s1, s2)
        s2_s2_norm = self.l2_norm(s2, s2)

        if self.scaled:
            s_target = s1_s2_norm/(s2_s2_norm+self.eps)*s2
        else:
            s_target = s2

        if not self.scale_dependent:
            e_noise = s1 - s_target
        else:
            e_noise = s1 - s2
        
        target_norm = self.l2_norm(s_target, s_target)
        noise_norm = self.l2_norm(e_noise, e_noise)

        if self.sdr_max is not None:
            tau = 10**(-self.sdr_max/10)
            noise_norm = noise_norm + tau*target_norm

        if not self.source_aggregated:
            snr = 10 * torch.log10((target_norm / (noise_norm+self.eps)) + self.eps)
        else:
            snr = 10 * torch.log10((target_norm.sum(dim=-1)) / (noise_norm.sum(dim=-1)+self.eps) + self.eps)

        snr = -1 * snr

        if threshold is not None:
            # hard threshold
            snr_to_keep = snr[snr > threshold]
            if snr_to_keep.nelement() > 0:
                snr = snr_to_keep.view(-1, 1)

        if self.reduction:
            return torch.mean(snr)
        else:
            return snr

    def l2_norm(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        The equation is || ||^2

        Args:
            s1: tensor with shape [batch, *, length]
            s2: tensor with shape [batch, *, length]
        
        Returns:
            the l2-norm in tenor's last dimension
        """
        norm = torch.sum(s1*s2, -1, keepdim=True)
        
        return norm
    
    def apply_zero_mean(self, s: torch.Tensor) -> torch.Tensor:
        """Zero-mean in last dimension"""
        s_mean = torch.mean(s, dim=-1, keepdim=True)
        s = s - s_mean
        
        return s
    
    def check_input_shape(self, s: torch.Tensor) -> None:
        """Check input tensor shape meets the loss function setting"""
        if self.source_aggregated:
            assert s.dim() == 3, 'source_aggregated need input dimension is 3'
        
        else:
            assert s.dim() == 2, 'need input shape as (batch, length)'


def attenuation_ratio(s1: torch.Tensor, s2: torch.Tensor, mask: torch.Tensor, reduction: bool = True) -> torch.Tensor:
    """
    To judge how well the system can suppress speech where the output sould be silent.

    Args:
        s1: enhanced signal, (N, L)
        s2: noisy (unprocessed) signal, (N, L)
        mask: target speaker mask, (N, L)
        reduction: is Fasle, return batch result
    
    Returns:
        the attenuation ratio present the suppresion level in where we wanted to be close silence.
    """
    batch_size = mask.shape[0]
    score = []
    for i in range(batch_size):
        # compute only non-target speech part
        _r = s1[i][mask[i] == 0].reshape(1, -1) # [1, L]
        _ref = s2[i][mask[i] == 0].reshape(1, -1) # [1, L]
        score.append(10 * torch.log10(l2_norm(_ref, _ref)/l2_norm(_r, _r)))
    
    score = torch.tensor(score)
    if reduction:
        return torch.mean(score, dim=0)
    else:
        return torch.mean(score, keepdim=True)


def l2_norm(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """
    The equation is || ||^2

    Args:
        s1: tensor with shape [batch, *, length]
        s2: tensor with shape [batch, *, length]
    
    Returns:
        the l2-norm in tenor's last dimension
    """
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm


def si_snr(s1: torch.Tensor, s2: torch.Tensor, eps: float = 1e-8, reduction: bool = True) -> torch.Tensor:
    """
    Single source SI-SNR\n
    Si-SNR = 20 * log10(|| alpha * target_s || / || pred_s - alpha * s ||), || || is 2-norm\n
    where alpha = <pred_s, target_s> / <target_s, target_s>,  < > is inner product\n
    
    Args:
        s1: enhance signal, shape as (N, *, L)
        s2: reference signal, shape as  (N, *, L)
        eps: small value protect divide zero case
        reduction: is Fasle, return batch result
    
    Returns:
        sisnr metric, if reduction is False, return batch result
    """
    # zero-mean
    s1_mean = torch.mean(s1, dim=-1, keepdim=True)
    s2_mean = torch.mean(s2, dim=-1, keepdim=True)
    s1 = s1 - s1_mean
    s2 = s2 - s2_mean

    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps) # 20*log10(|| ||) == 20*1/2*log10(|| ||^2)

    if reduction:
        return torch.mean(snr)
    else:
        return snr


def inactive_sdr_loss(s1: torch.Tensor, s2: torch.Tensor, reduction: bool = True) -> torch.Tensor:
    """
    Args:
        s1: enhanced signal, shape is (N, *, L)
        s2: reference signal, shape is (N, *, L)
    """
    # zero-mean
    s1_mean = torch.mean(s1, dim=-1, keepdim=True)
    s2_mean = torch.mean(s2, dim=-1, keepdim=True)
    s1 = s1 - s1_mean
    s2 = s2 - s2_mean

    s1_s1_norm = l2_norm(s1, s1)
    s2_s2_norm = l2_norm(s2, s2)

    if reduction:
        return torch.mean(10 * torch.log10(s1_s1_norm + 0.01*s2_s2_norm + 1e-8))
    else:
        return 10 * torch.log10(s1_s1_norm + 0.01*s2_s2_norm  + 1e-8)
