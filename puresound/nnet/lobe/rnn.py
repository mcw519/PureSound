from typing import Optional, Tuple

import torch
import torch.nn as nn

from .norm import get_norm


class SingleRNN(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        rnn_type = rnn_type.upper()
        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(p=dropout)
        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, C, T]
        
        Returns:
            output tensor has shape [N, C, T]
        """
        output = x.permute(0, 2, 1)  # [N, T, C]
        rnn_output, _ = self.rnn(output)
        rnn_output = self.drop(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        rnn_output = rnn_output.permute(0, 2, 1)  # [N, C, T]
        return rnn_output


class FSMN(nn.Module):
    """
    Feedforward Sequential Memory Networks.

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        project_dim: hidden feature dimension, as same as memory block size
        l_context: past context frames
        r_context: future context frames
    
    References:
        https://github.com/funcwj/aps/blob/c814dc5a8b0bff5efa7e1ecc23c6180e76b8e26c/aps/asr/base/component.py#L310
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        project_dim: int,
        l_context: int,
        r_context: int,
        dilation: int = 1,
        dropout: float = 0.0,
        norm_type: str = "bN1d",
    ):
        super().__init__()
        self.lctx = l_context
        self.rctx = r_context
        k_size = l_context + r_context + 1

        self.in_proj = nn.Conv1d(
            input_dim, project_dim, kernel_size=1, stride=1, bias=False
        )
        self.ctx_conv = nn.Conv1d(
            project_dim,
            project_dim,
            kernel_size=k_size,
            padding=0,
            dilation=dilation,
            groups=project_dim,
            bias=False,
        )
        self.out_proj = nn.Conv1d(project_dim, output_dim, kernel_size=1)

        norm_cls = get_norm(norm_type)
        self.out_norm = nn.Sequential(
            norm_cls(output_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            current input tensor x, [N, C, T]
            previous memory block tensor, [N, P, T] where P is the project_dim

        Returns:
            out -- [N, C, T]
            proj, new memory block for next layer, [N, P, T], where P is the project_dim
        """
        proj = self.in_proj(x)
        proj_pad = torch.nn.functional.pad(
            proj, (self.lctx, self.rctx), "constant", 0.0
        )
        ctx = self.ctx_conv(proj_pad)
        proj = proj + ctx

        if memory is not None:
            proj = proj + memory

        out = self.out_proj(proj)
        out = self.out_norm(out)

        return out, proj


class ConditionFSMN(FSMN):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        project_dim: int,
        embed_dim: int,
        l_context: int,
        r_context: int,
        dilation: int = 1,
        dropout: float = 0,
        norm_type: str = "bN1d",
        use_film: bool = False,
    ):
        super().__init__(
            input_dim,
            output_dim,
            project_dim,
            l_context,
            r_context,
            dilation,
            dropout,
            norm_type,
        )
        self.use_film = use_film

        if not use_film:
            self.embed_proj = nn.Conv1d(
                project_dim + embed_dim,
                project_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            )

        else:
            self.cond_scale = nn.Conv1d(
                embed_dim, project_dim, kernel_size=1, bias=False
            )
            self.cond_bias = nn.Conv1d(
                embed_dim, project_dim, kernel_size=1, bias=False
            )

    def forward(
        self,
        x: torch.Tensor,
        embed: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            current input tensor x, [N, C, T]
            embed: speaker embed, [N, C]
            previous memory block tensor, [N, P, T] where P is the project_dim

        Returns:
            out -- [N, C, T]
            proj, new memory block for next layer, [N, P, T], where P is the project_dim
        """
        proj = self.in_proj(x)
        proj_pad = torch.nn.functional.pad(
            proj, (self.lctx, self.rctx), "constant", 0.0
        )
        ctx = self.ctx_conv(proj_pad)

        if not self.use_film:
            embed = embed.unsqueeze(2)
            embed = embed.repeat(1, 1, x.size(2))
            condi = torch.cat([ctx, embed], dim=1)
            condi = self.embed_proj(condi)
            proj = proj + ctx + condi

        else:
            condi = embed.unsqueeze(-1)
            film_scale = self.cond_scale(condi)
            film_bias = self.cond_bias(condi)
            proj = film_scale * proj + film_bias
            ctx = film_scale * ctx + film_bias
            proj = proj + ctx

        if memory is not None:
            proj = proj + memory

        out = self.out_proj(proj)
        out = self.out_norm(out)

        return out, proj
