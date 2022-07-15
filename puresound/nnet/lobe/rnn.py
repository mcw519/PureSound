from typing import Optional, Tuple

import torch
import torch.nn as nn

from .norm import get_norm


class SingleRNN(nn.Module):
    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, bidirectional: bool = False, dropout: float = 0.):
        super().__init__()
        rnn_type = rnn_type.upper()
        assert rnn_type in ["RNN", "LSTM", "GRU",], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
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
        output = x.permute(0, 2, 1) # [N, T, C]
        rnn_output, _ = self.rnn(output)
        rnn_output = self.drop(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        rnn_output = rnn_output.permute(0, 2, 1) # [N, C, T]
        return rnn_output
