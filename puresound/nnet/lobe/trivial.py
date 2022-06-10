from types import LambdaType
from typing import Any

import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambda_func: LambdaType):
        super().__init__()
        self.lambd = lambda_func
    
    def forward(self, x: torch.Tensor) -> Any:
        return self.lambd(x)


class Magnitude(nn.Module):
    """
    This is a layer for converting stft-complex form to magnitude form.

    Args:
        if drop_first will remove the first bin value
    """

    def __init__(self, drop_first: bool = True) -> None:
        super().__init__()
        self.drop_first = drop_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor x has shape [N, C, T, 2] or [N, C, T]
        """
        if x.dim() == 4:
            _re = x[..., 0]
            _im = x[..., 1]
        
        elif x.dim() == 3:
            _re, _im = torch.chunk(x, 2, dim=1)
        
        else:
            raise TypeError
        
        if self.drop_first:
            _re = _re[:, 1:, :]
            _im = _im[:, 1:, :]
        
        return torch.sqrt(_re.pow(2) + _im.pow(2) + 1e-8)
