from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lobe.trivial import FiLM, SplitMerge


class DPRNN(nn.Module):
    """
    Deep dual-path RNN.
    
    Args:
        input_size (int): input feature(channel) dimension
        hidden_size (int): hidden feature(channel) dimension
        output_size (int): output feature(channel) dimension
        n_blocks (int): number of blocks (intra+inter).
        seg_size (int): chunk size
        seg_overlap (bool): if true, chunk stride is half chunk size.
        embed_dim (int): if not zero, concate in right_conv's input.
        embed_norm (bool): applies 2-norm for input embedding.
        causal (bool): padding by causal scenario, others padding to same length between input and output.
        block_with_embed (list): which layer insert embedding.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_blocks: int = 2,
        seg_size: int = 20,
        seg_overlap: bool = False,
        causal: bool = True,
        embed_dim: int = 0,
        embed_norm: bool = False,
        block_with_embed: Optional[List] = None,
        embedding_free_tse: bool = False,
    ):
        super().__init__()

        self.seg_size = seg_size
        self.seg_overlap = seg_overlap

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bi_direct = not causal
        self.n_blocks = n_blocks

        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.block_with_embed = block_with_embed
        self.embedding_free_tse = embedding_free_tse

        self.input_film = nn.ModuleList()

        self.intra_rnn = nn.ModuleList()
        self.intra_proj = nn.ModuleList()
        self.intra_norm = nn.ModuleList()
        self.inter_rnn = nn.ModuleList()
        self.inter_norm = nn.ModuleList()
        self.inter_proj = nn.ModuleList()

        for i in range(n_blocks):
            if embed_dim != 0 and block_with_embed[i]:
                self.intra_rnn.append(
                    nn.LSTM(
                        input_size,
                        hidden_size,
                        num_layers=1,
                        bidirectional=self.bi_direct,
                        batch_first=True,
                    )
                )
                self.input_film.append(FiLM(input_size, embed_dim, input_norm=True))
            else:
                self.intra_rnn.append(
                    nn.LSTM(
                        input_size,
                        hidden_size,
                        num_layers=1,
                        bidirectional=self.bi_direct,
                        batch_first=True,
                    )
                )
                self.input_film.append(None)

            self.intra_proj.append(
                nn.Linear(hidden_size * (int(self.bi_direct) + 1), input_size)
            )
            self.intra_norm.append(nn.LayerNorm(input_size))
            self.inter_rnn.append(
                nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=1,
                    bidirectional=self.bi_direct,
                    batch_first=True,
                )
            )
            self.inter_proj.append(
                nn.Linear(hidden_size * (int(self.bi_direct) + 1), input_size)
            )
            self.inter_norm.append(nn.LayerNorm(input_size))

        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None):
        """
        Args:
            input tensor x has shape [N, C, T]
            Conditional embedding vector has shape [N, C] or [N, C, T]
        
        Returns:
            output tensor shape is [N, C, T]
        """
        if self.embedding_free_tse:
            assert (
                embed.dim() == 3
            ), f"embedding free tse need enrollment waveform as input."
            inter_hidd_init_states = self._get_hidden_states(embed)
        else:
            inter_hidd_init_states = [None for _ in range(self.n_blocks)]

        if self.embed_norm and embed is not None and not self.embedding_free_tse:
            embed = F.normalize(embed, p=2, dim=1)

        N, C, T = x.shape

        if self.seg_overlap:
            x, rest = SplitMerge.split(
                x, seg_size=self.seg_size
            )  # [N, C, T] -> [N, S, K, C]

        else:
            x = x.permute(0, 2, 1)
            N, _, C = x.shape
            rest = self.seg_size - T % self.seg_size
            if rest > 0:
                x = nn.functional.pad(x, (0, 0, 0, rest))

            x = x.reshape(N, -1, self.seg_size, C)

        batch_size, total_seg, seg_size, feat_size = x.shape

        if not self.embedding_free_tse and embed is not None:
            embed = embed.unsqueeze(1)
            embed = embed.repeat(1, total_seg, 1).reshape(batch_size * total_seg, -1)

        output = x
        for i in range(self.n_blocks):
            output = output.reshape(-1, seg_size, feat_size).contiguous()  # [NS, K, C]

            if embed is not None and self.block_with_embed[i]:
                output = self.input_film[i](output.transpose(1, 2), embed)
                output = output.transpose(1, 2)

            intra_input = output
            intra_output, _ = self.intra_rnn[i](intra_input)
            intra_output = self.intra_proj[i](intra_output)
            intra_output = self.intra_norm[i](intra_output)
            output = output + intra_output

            inter_input = output.reshape(batch_size, total_seg, seg_size, feat_size)
            inter_input = inter_input.permute(0, 2, 1, 3)
            inter_input = inter_input.reshape(-1, total_seg, feat_size).contiguous()
            inter_output, _ = self.inter_rnn[i](inter_input, inter_hidd_init_states[i])
            inter_output = self.inter_proj[i](inter_output)
            inter_output = self.inter_norm[i](inter_output)
            output = inter_input + inter_output  # [NK, S, C]

            output = output.reshape(
                batch_size, seg_size, total_seg, feat_size
            ).contiguous()
            output = output.permute(0, 2, 1, 3)

        N, S, K, C = output.shape

        if self.seg_overlap:
            output = output.reshape(N, S, K, C)
            output = SplitMerge.merge(output, rest)
            output = self.output_fc(output)

        else:
            output = output.reshape(N, S * K, C)[:, :T, :]
            output = self.output_fc(output.transpose(1, 2))

        return output

    def _get_hidden_states(self, x: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, C, T]
                    
        Returns:
            output tensor shape is [N, C, T]
        """
        N, C, T = x.shape

        if self.seg_overlap:
            x, rest = SplitMerge.split(
                x, seg_size=self.seg_size
            )  # [N, C, T] -> [N, S, K, C]

        else:
            x = x.permute(0, 2, 1)
            N, _, C = x.shape
            rest = self.seg_size - T % self.seg_size
            if rest > 0:
                x = nn.functional.pad(x, (0, 0, 0, rest))

            x = x.reshape(N, -1, self.seg_size, C)

        batch_size, total_seg, seg_size, feat_size = x.shape

        output = x
        inter_hidd = []
        for i in range(self.n_blocks):
            output = output.reshape(-1, seg_size, feat_size).contiguous()  # [NS, K, C]

            intra_input = output
            intra_output, _ = self.intra_rnn[i](intra_input)
            intra_output = self.intra_proj[i](intra_output)
            intra_output = self.intra_norm[i](intra_output)
            output = output + intra_output

            inter_input = output.reshape(batch_size, total_seg, seg_size, feat_size)
            inter_input = inter_input.permute(0, 2, 1, 3)
            inter_input = inter_input.reshape(-1, total_seg, feat_size).contiguous()
            inter_output, hidd = self.inter_rnn[i](inter_input)
            inter_hidd.append(hidd)
            inter_output = self.inter_proj[i](inter_output)
            inter_output = self.inter_norm[i](inter_output)
            output = inter_input + inter_output  # [NK, S, C]

            output = output.reshape(
                batch_size, seg_size, total_seg, feat_size
            ).contiguous()
            output = output.permute(0, 2, 1, 3)

        return inter_hidd
