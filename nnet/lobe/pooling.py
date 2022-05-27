from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L661
def length_to_mask(length: torch.Tensor, max_len: Optional[int] = None, dtype: torch.dtype = None, device: torch.device = None):
    """Creates a binary mask for each sequence.
    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.
    Returns
    -------
    mask : tensor
        The binary mask.
    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class AttentiveStatisticsPooling(nn.Module):
    """
    Calculates mean and std for a batch tensor

    Args:
        channels: equal to 2*output_size
        attention_channels: hidden layer dimension
    """
    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=attention_channels, kernel_size=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels)
            )
        
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, return_weight: bool = False):
        """
        Args:
            x : torch.Tensor of shape [N, C, L].
            return_weight: return the attention weight
        """
        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        L = x.shape[-1]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        attn = x
        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))
        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2)
        if return_weight: return attn
        mean, std = self._compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats
    
    def _compute_statistics(self, x: torch.Tensor, m: torch.Tensor, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt(
            (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps)
        )
        return mean, std
