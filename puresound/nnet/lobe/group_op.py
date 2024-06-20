from typing import Optional

import torch
import torch.nn as nn


class TAC(nn.Module):
    """
    Transform-Average-Concateneta.

    Args:
        input_dim: input feature dimension
        hidden_dim: hidden feature dimension

    Reference:
        https://ieeexplore.ieee.org/abstract/document/9414322
        https://github.com/yluo42/GC3/blob/main/utility/basics.py#L28
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.TAC_input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.PReLU())

        self.TAC_mean = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU())

        self.TAC_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim), nn.PReLU()
        )

        # In paper, this normalization layer is non-causal `nn.GroupNorm(1, input_size)`
        self.TAC_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, with shape [N, G, C, T] where G is group
        """
        N, G, C, T = x.shape
        output = x

        # transform
        group_input = output.permute(0, 3, 1, 2).contiguous().view(-1, C)  # [N*T*G, C]
        group_output = self.TAC_input(group_input).view(N, T, G, -1)  # [N, T, G, C]

        # mean pooling
        group_mean = group_output.mean(2).view(N * T, -1)  # [N*T, C]

        # concate
        group_output = group_output.view(N * T, G, -1)  # [N*T, G, C]
        group_mean = (
            self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()
        )  # [N*T, G, C]
        group_output = torch.cat([group_output, group_mean], 2)  # [N*T, G, 2C]
        group_output = self.TAC_output(
            group_output.view(-1, group_output.shape[-1])
        )  # [N*T*G, C]
        group_output = (
            group_output.view(N, T, G, -1).permute(0, 2, 3, 1).contiguous()
        )  # [N, G, C, T]
        group_output = self.TAC_norm(group_output.view(N * G, C, T))
        output = output + group_output.view(x.shape)

        return output


class GroupedGRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        bidirectional: bool = False,
        bias: bool = True,
        droupout: float = 0.0,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = int(self.bidirectional) + 1
        self.groups = groups
        self.layers = nn.ModuleList(
            [
                nn.GRU(
                    self.input_size,
                    self.hidden_size,
                    bias=bias,
                    batch_first=True,
                    droupout=droupout,
                    bidirectional=bidirectional,
                )
                for _ in range(groups)
            ]
        )

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            x   (Tensor): input x of shape is [N, C, T]
            h0  (Tensor): hidden state of shape [G*D, B, H], where G=groups, D=directions, H=hidden_size

        Returns:
            outputs and hidden states
        """
        x = x.permute(0, 2, 1).contiguous()  # [N, T, C]
        batch, _, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(
                self.groups * self.num_directions,
                batch,
                self.hidden_size,
                device=x.device,
            )

        outputs = []
        outstates = []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                x[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)

        outputs = torch.cat(outputs, dim=-1)
        outputs = outputs.permute(0, 2, 1).contiguous()  # [N, C, T]
        h = torch.cat(outstates, dim=0)
        if return_hidden:
            return outputs, h
        else:
            return outputs


class GroupedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bidirectional: bool = False,
        bias: bool = True,
        droupout: float = 0.0,
        shuffle: bool = True,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = int(self.bidirectional) + 1
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle

        self.grus = nn.ModuleList()
        self.grus.append(
            GroupedGRULayer(
                input_size,
                hidden_size,
                groups=groups,
                bias=bias,
                droupout=droupout,
                bidirectional=bidirectional,
            )
        )
        for _ in range(1, num_layers):
            self.grus.append(
                GroupedGRULayer(
                    hidden_size,
                    hidden_size,
                    groups=groups,
                    bias=bias,
                    droupout=droupout,
                    bidirectional=bidirectional,
                )
            )

        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            x   (Tensor): input x of shape is [N, C, T]
            h0  (Tensor): hidden state of shape [L*G*D, B, H], where L=num_layer, G=groups, D=directions, H=hidden_size

        Returns:
            outputs and hidden states
        """
        batch, nfreq, nframes = x.shape

        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * self.groups * self.num_directions,
                batch,
                self.hidden_size,
                device=x.device,
            )

        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            x, s = gru(x, h0[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                x = x.permute(0, 2, 1).contiguous()  # [N, T, C]
                x = (
                    x.view(batch, nframes, -1, self.groups)
                    .transpose(2, 3)
                    .reshape(batch, nframes, -1)
                )
                x = x.permute(0, 2, 1)  # [N, C, T]

        outstates = torch.cat(outstates, dim=0)
        if return_hidden:
            return x, outstates
        else:
            return x


class GroupedLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int = 1,
        shuffle: bool = True,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.groups = groups

        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x   (Tensor): input of shape [N, C, T]
        """
        x = x.permute(0, 2, 1)  # [N, T, C]
        outputs = []
        for i, layer in enumerate(self.layers):
            outputs.append(
                layer(x[..., i * self.input_size : (i + 1) * self.input_size])
            )

        outputs = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = outputs.shape
            outputs = (
                outputs.view(-1, self.hidden_size, self.groups)
                .transpose(-1, -2)
                .reshape(orig_shape)
            )

        outputs = outputs.permute(0, 2, 1).contiguous()  # [N, C, T]
        return outputs


class SqueezedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinear(
                input_size=input_size,
                hidden_size=hidden_size,
                groups=linear_groups,
            ),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinear(hidden_size, output_size, groups=linear_groups),
                nn.ReLU(inplace=True),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            x   (Tensor): input x of shape is [N, C, T]
            h0  (Tensor): hidden state of shape [L, B, H], where L=num_layer, H=hidden_size

        Returns:
            outputs and hidden states
        """
        self.gru.flatten_parameters()
        x = self.linear_in(x)
        x = x.permute(0, 2, 1)  # [N, T, C]
        x, h0 = self.gru(x, h0)
        x = x.permute(0, 2, 1)  # [N, C, T]
        x = self.linear_out(x)
        if return_hidden:
            return x, h0
        else:
            return x
