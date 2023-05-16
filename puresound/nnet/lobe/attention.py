import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"
            )

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [N, T, C]
        """
        x = x.permute(1, 0, 2)  # [N, T, C] -> [T, N, C]
        x = x + self.pe[: x.size(0)]

        return self.dropout(x).permute(1, 0, 2)


class MHA(nn.Module):
    """
    Multi-head Attention mechanism.

    Args:
        embed_dim: Total dimension of the model
        heads: embed_dim = head_dim * num_heads
    """

    def __init__(self, embed_dim: int, heads: int = 1):
        super().__init__()

        self.atten = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=0,
            batch_first=True,
            bias=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
        context_range: int = None,
    ):
        """
        Args:
            query: tensor with shape [N, T, C]
            key: tensor with shape [N, T, C]
            value: tensor with shape [N, T, C]
            causal: if causal, attent only past length
            context_range: attent only in context_range length
        
        Returns:
            Tuple of attention outputs and attention weights
        """
        size_q = query.size(1)
        mask = None

        if causal and context_range is None:
            mask = (torch.triu(torch.ones(size_q, size_q)) == 1).transpose(
                0, 1
            )  # [L, S]
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )

        if context_range is not None:
            if causal:
                mask = torch.tril(torch.ones(size_q, size_q), diagonal=-context_range)
                mask = mask + torch.triu(torch.ones(size_q, size_q), diagonal=1)
                mask = (
                    mask.float()
                    .masked_fill(mask == 0, float(0.0))
                    .masked_fill(mask == 1, float("-inf"))
                )

            else:
                mask = torch.tril(
                    torch.ones(size_q, size_q), diagonal=-(context_range - 1)
                )
                mask = mask + mask.transpose(0, 1)
                mask = (
                    mask.float()
                    .masked_fill(mask == 0, float(0.0))
                    .masked_fill(mask == 1, float("-inf"))
                )

        if mask is not None:
            mask = mask.to(query.device)

        return self.atten(query, key, value, attn_mask=mask)


class MhaSelfAttenLayer(nn.Module):
    """
    Implement Transformer encoder block, include self-attention, feed-forward and skip-connect.

    Args:
        feats_dim: Total dimension of the model's input feature
        hidden_dim: feed-forward network's hidden dimension
        nhead: embed_dim = head_dim * nhead
        dropout: dropout probability
        improved: replace linear FF by LSTM. [1]
        bidirectional: used in improved-transformer. [1]
        position_encoding: add postional embedding or not.
    
    References:
        [1] Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation
    """

    def __init__(
        self,
        feats_dim: int,
        hidden_dim: int,
        nhead: int,
        dropout: float = 0.0,
        improved: bool = False,
        bidirectional: bool = False,
        position_encoding: bool = True,
    ):
        super().__init__()
        self.improved = improved
        self.bidirectional = bidirectional
        self.position_encoding = position_encoding

        self.self_atten = MHA(feats_dim, heads=nhead)
        self.self_atten_dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(feats_dim)

        if not improved:
            if self.bidirectional:
                print("Ignored bidirectional option since no LSTM here.")

            if position_encoding:
                self.pos = PositionalEncoding(d_model=feats_dim, dropout=dropout)

            self.feedforward = nn.Sequential(
                nn.Linear(feats_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, feats_dim),
                nn.Dropout(p=dropout),
            )

        else:
            if position_encoding:
                print(
                    "Ignored position_encoding option here replaced by LSTM modeling."
                )

            self.recurrent = nn.LSTM(
                feats_dim, hidden_dim, bidirectional=bidirectional, batch_first=True
            )
            if bidirectional:
                hidden_dim *= 2
            self.feedforward = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, feats_dim),
                nn.Dropout(p=dropout),
            )

        self.norm2 = nn.LayerNorm(feats_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        context_range: Optional[int] = None,
        return_atten_weight: bool = False,
    ):
        """
        Args:
            x: tensor with shape [N, C, T]
            causal: if causal, attent only past length
            context_range: attent only in context_range length
            return_atten_weight: return attention weights
        
        Returns:
            Tuple of attention outputs has shape [N, C, T] and its attention weights
        """
        if self.improved:
            self.recurrent.flatten_parameters()
        x = x.permute(0, 2, 1)  # [N, T, C]
        src = x.clone()

        if self.position_encoding:
            x = self.pos(x)

        # self-attention
        x, w = self.self_atten(x, x, x, causal=causal, context_range=context_range)
        x = self.self_atten_dropout(x)
        x = src + x
        x = self.norm1(x)

        # feed-forward
        src = x.clone()
        if self.improved:
            x, _ = self.recurrent(x)

        x = self.feedforward(x)
        x = src + x
        x = self.norm2(x)

        x = x.permute(0, 2, 1)  # [N, C, T]

        if return_atten_weight:
            return x, w
        else:
            return x
