import torch
import torch.nn as nn


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.eps = 1e-8
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + self.eps).sqrt())


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""

    def forward(self, x):
        """Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, *]`
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + self.eps).sqrt())


class InstantLN(_LayerNorm):
    """Instant Layer Normalization
    Applies normalization over frequency and channel."""

    def forward(self, x):
        """
        Args:
            x: Shape [N, Ch, C, T]
        """
        N, CH, C, T = x.shape
        x = x.reshape(N, CH*C, T)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + self.eps).sqrt()).reshape(N, CH, C, T)


# Aliases.
gLN = GlobLN
cLN = ChanLN
iLN = InstantLN
bN1d = nn.BatchNorm1d
bN2d = nn.BatchNorm2d
gGN = lambda x: nn.GroupNorm(1, x, 1e-8)

def get_norm(name: str):
    if name not in ['gLN', 'cLN', 'iLN', 'bN1d', 'gGN', 'bN2d']:
        raise NameError('Could not interpret normalization identifier')

    if isinstance(name, str):
        cls = globals().get(name)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(name))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(name))
