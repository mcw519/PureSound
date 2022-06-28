from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .lobe.trivial import FiLM, Gate


class MemLSTM(nn.Module):
    def __init__(self, hidden_size: int, causal: bool = True, dropout: float = 0.):
        super().__init__()
        self.hidden_size = hidden_size
        self.causal = causal
        self.input_size = hidden_size if causal else 2 * hidden_size
        self.bi_direct = not causal
        self.causal = causal

        self.h_net = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=self.bi_direct, batch_first=True)
        self.h_dropout = nn.Dropout(p=dropout)
        self.h_proj = nn.Linear(self.hidden_size * (int(self.bi_direct)+1), self.input_size)
        self.h_norm = nn.LayerNorm(self.input_size)
        self.c_net = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=self.bi_direct, batch_first=True)
        self.c_dropout = nn.Dropout(p=dropout)
        self.c_proj = nn.Linear(self.hidden_size * (int(self.bi_direct)+1), self.input_size)
        self.c_norm = nn.LayerNorm(self.input_size)

    def forward(self,
                h: torch.Tensor,
                c: torch.Tensor,
                h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                c_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_all: bool = False,
                streaming: bool = False):
        """
        Args:
            hidden states h from SegLSTM has shape [N, S, D, C], where N=batch_size, S=total_seg, D=direction_size and C=feats_size
            cell states c from SegLSTM has shape [N, S, D, C], where N=batch_size, S=total_seg, D=direction_size and C=feats_size
            h_states/c_states are the hidden/cell status in LSTM layer, we need this op for streaming inference
            if return_all is true, return all information.
            if streaming is true, let off-line inference result equal to streaming result.
        
        Returns:
            hidden states to next SegLSTM has shape [D, NS, C]
            cell states to next SegLSTM has shape [D, NS, C]
            Mem-LSTM's hidden/cell status, ((hidden status), (cell status))
        """
        self.h_net.flatten_parameters()
        self.c_net.flatten_parameters()
        batch, total_seg, direct, feat_len = h.shape
        h = h.reshape(batch, total_seg, -1) # [N, S, DC]
        c = c.reshape(batch, total_seg, -1) # [N, S, DC]

        if h_states is not None:
            h_, (h_h, h_c) = self.h_net(h, h_states)
        else:
            h_, (h_h, h_c) = self.h_net(h)
        
        h_ = self.h_dropout(h_)
        h_ = self.h_proj(h_.reshape(batch*total_seg, -1)).reshape(batch, total_seg, -1)
        h = h + self.h_norm(h_)
        
        if c_states is not None:
            c_, (c_h, c_c) = self.c_net(c, c_states) # c_h, c_c [D, NS, C]
        else:
            c_, (c_h, c_c) = self.c_net(c)

        c_ = self.c_dropout(c_)
        c_ = self.c_proj(c_.reshape(batch*total_seg, -1)).reshape(batch, total_seg, -1)
        c = c + self.c_norm(c_)

        h = h.reshape(batch*total_seg, direct, feat_len).transpose(1, 0).contiguous() # [D, NS, C]
        c = c.reshape(batch*total_seg, direct, feat_len).transpose(1, 0).contiguous() # [D, NS, C]

        if self.causal and not streaming:
            # In the causal setting, each layer's states for first segment start from zeros
            h_ = torch.zeros_like(h)
            h_[:, 1:, :] = h[:, :-1, :]
            c_ = torch.zeros_like(c)
            c_[:, 1:, :] = c[:, :-1, :]
            h = h_
            c = c_
        
        if return_all:
            return h, c, (h_h, h_c), (c_h, c_c)
        else:
            return h, c

    def streaming_forward(self,
                            h: torch.Tensor,
                            c: torch.Tensor,
                            h_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                            c_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                            return_all: bool = False):
        """Woking as streaming inference."""
        self.h_net.flatten_parameters()
        self.c_net.flatten_parameters()
        batch, total_seg, direct, feat_len = h.shape
        h = h.reshape(batch, total_seg, -1) # [N, S, DC]
        c = c.reshape(batch, total_seg, -1) # [N, S, DC]

        h_out = []
        c_out = []
        for idx in range(h.shape[1]):
            cur_h_inp = h[:, idx, :].view(1, 1, -1)
            cur_c_inp = c[:, idx, :].view(1, 1, -1)
            if h_states is None:
                h_, h_states = self.h_net(cur_h_inp)
            else:
                h_, h_states = self.h_net(cur_h_inp, h_states)

            if c_states is None:
                c_, c_states = self.c_net(cur_c_inp)
            else:
                c_, c_states = self.c_net(cur_c_inp, c_states)
            
            h_ = self.h_proj(h_.reshape(1, -1)).reshape(1, 1, -1)
            out_h = cur_h_inp + self.h_norm(h_)
            h_out.append(out_h)

            c_ = self.c_proj(c_.reshape(1, -1)).reshape(1, 1, -1)
            out_c = cur_c_inp + self.c_norm(c_)
            c_out.append(out_c)
                
        h = torch.cat(h_out, dim=1).reshape(batch*total_seg, direct, feat_len).transpose(1, 0).contiguous()
        c = torch.cat(c_out, dim=1).reshape(batch*total_seg, direct, feat_len).transpose(1, 0).contiguous()

        if return_all:
            return h, c, h_states, c_states
        else:
            return h, c


class SegLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, causal: bool = True, dropout: float = 0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bi_direct = not causal
        self.causal = causal

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=self.bi_direct, batch_first=True)
        self.drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size*(int(self.bi_direct)+1), input_size)
        self.norm = nn.LayerNorm(input_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Args:
            input tensor x has shape [NS, K, C]
            hidden states h from SegLSTM has shape [D, NS, C]
            cell states c from SegLSTM has shape [D, NS, C]
            where N=batch_size, S=total_seg, D=direction_size and C=feats_size
        
        Returns:
            output tensor x has shape [NS, K, C]
            hidden states to next memLSTM has shape [D, NS, C]
            cell states to next memLSTM has shape [D, NS, C]
        """
        self.lstm.flatten_parameters()
        batch, seg_size, feat_len = x.shape

        if h is None:
            d = int(self.bi_direct) + 1
            h = torch.zeros(d, batch, self.hidden_size).to(x.device)
        
        if c is None:
            d = int(self.bi_direct) + 1
            c = torch.zeros(d, batch, self.hidden_size).to(x.device)
        
        x_out, (h, c) = self.lstm(x, (h, c))
        x_out = self.drop(x_out)
        x_out = self.proj(x_out.contiguous().view(-1, x_out.shape[2])).view(batch, seg_size, feat_len)
        x_out = x + self.norm(x_out)

        return x_out, h, c
    
    def streaming_forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        self.lstm.flatten_parameters()
        batch, seg_size, feat_len = x.shape

        if h is None:
            d = int(self.bi_direct) + 1
            h = torch.zeros(d, batch, self.hidden_size).to(x.device)
        
        if c is None:
            d = int(self.bi_direct) + 1
            c = torch.zeros(d, batch, self.hidden_size).to(x.device)
        
        x_out = []
        for idx in range(x.shape[1]):
            x_inp = x[:, idx, :].view(batch, 1, feat_len)
            _out, (h, c) = self.lstm(x_inp, (h, c))
            _out = self.drop(_out)
            _out = self.proj(_out.contiguous().view(-1, _out.shape[2])).view(batch, 1, feat_len)
            _out = x_inp + self.norm(_out)
            x_out.append(_out)
        
        x_out = torch.cat(x_out, dim=1)
        
        return x_out, h, c


class SkiM(nn.Module):
    """
    Skipping memory LSTM.
    
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
    
    References:
        [1]: https://arxiv.org/abs/2201.10800
        [2]: https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/skim.py
    """
    def __init__(self,
                input_size: int,
                hidden_size: int,
                output_size: int,
                n_blocks: int = 2,
                seg_size: int = 20,
                seg_overlap: bool = False,
                causal: bool = True,
                embed_dim: int = 0,
                embed_norm: bool = False,
                embed_fusion: Optional[str] = None,
                block_with_embed: Optional[List] = None,
                dropout: float = 0.,
                ):
        super().__init__()

        self.seg_size = seg_size
        self.seg_overlap = seg_overlap
        self.hidden_size = hidden_size

        self.n_blocks = n_blocks
        self.causal = causal

        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.block_with_embed = block_with_embed

        self.seg_lstm = nn.ModuleList()
        if embed_dim == 0:
            for i in range(n_blocks):
                self.seg_lstm.append(SegLSTM(input_size, hidden_size, causal=causal, dropout=dropout))
        else:
            self.seg_input_fusion = nn.ModuleList()
            for i in range(n_blocks):
                self.seg_lstm.append(SegLSTM(input_size, hidden_size, causal=causal, dropout=dropout))
                if block_with_embed[i]:
                    if embed_fusion.lower() == 'film':
                        self.seg_input_fusion.append(FiLM(input_size, embed_dim, input_norm=True))
                    
                    elif embed_fusion.lower() == 'gate':
                        self.seg_input_fusion.append(Gate(input_size, hidden_size=128, embed_size=embed_dim))

                    else:
                        raise NameError
                
                else:
                    self.seg_input_fusion.append(None)
            
        self.mem_lstm = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.mem_lstm.append(MemLSTM(hidden_size, causal=causal, dropout=dropout))

        self.output_fc = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(input_size, output_size, 1)
        )
    
    def split(self, x: torch.Tensor):
        """
        Args:
            input tensor x has shape [N, C, T]
        """
        seg_stride = self.seg_size // 2

        # padding
        batch, feat_size, seq_len = x.shape # [N, C, T]
        
        rest = self.seg_size - (seg_stride + seq_len % self.seg_size) % self.seg_size
        if rest > 0:
            pad = Variable(torch.zeros(batch, feat_size, rest)).type(x.type())
            x = torch.cat([x, pad], dim=-1)
        
        pad_aux = Variable(torch.zeros(batch, feat_size, seg_stride)).type(x.type())
        x = torch.cat([pad_aux, x, pad_aux], dim=-1)

        # splitting
        batch, feat_size, seq_len = x.shape

        seg_1 = x[:, :, :-seg_stride].contiguous().view(batch, feat_size, -1, self.seg_size)
        seg_2 = x[:, :, seg_stride:].contiguous().view(batch, feat_size, -1, self.seg_size)

        segments = torch.cat([seg_1, seg_2], dim=-1).view(batch, feat_size, -1, self.seg_size) # [N, C, S, K]
        segments = segments.permute(0, 2, 3, 1) # [N, S, K, C]
        
        return segments, rest
    
    def merge(self, x: torch.Tensor, rest: int):
        """
        Args:
            input tensor x has shape [N, S, K, C]
        
        Outputs:
            output tensor has shape [N, C, T]
        """
        batch, total_seg, seg_size, feat_size = x.shape
        seg_stride = seg_size // 2
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, feat_size, -1, seg_size*2)

        x1 = x[:, :, :, :seg_size].contiguous().view(batch, feat_size, -1)[:, :, seg_stride:]
        x2 = x[:, :, :, seg_size:].contiguous().view(batch, feat_size, -1)[:, :, :-seg_stride]

        output = (x1 + x2) / 2
        if rest > 0:
            output = output[..., :-rest]
        
        return output.contiguous()

    def forward(self, x: torch.Tensor, embed: Optional[torch.Tensor] = None):
        """
        Args:
            input tensor shape is [N, C, T]
            Conditional embedding vector has shape [N, C]
        
        Returns:
            output tensor shape is [N, C, T]
        """
        if self.embed_norm and embed is not None:
            embed = F.normalize(embed, p=2, dim=1)

        x = x.permute(0, 2, 1).contiguous()
        N, T, C = x.shape
        
        if self.seg_overlap:
            x, rest = self.split(x.transpose(1, 2))
                    
        else:
            rest = self.seg_size - T % self.seg_size
            if rest > 0:
                x = nn.functional.pad(x, (0, 0, 0, rest))
            
            x = x.view(N, -1, self.seg_size, C)
        
        N, S, K, C = x.shape
        assert K == self.seg_size

        if embed is not None:
            embed = embed.unsqueeze(1)
            embed = embed.repeat(1, S, 1).reshape(N*S, -1)

        output = x.reshape(N*S, K, C).contiguous()
        h, c = None, None

        for i in range(self.n_blocks):
            if embed is not None and self.block_with_embed[i]:
                output = self.seg_input_fusion[i](output.transpose(1, 2), embed)
                output = output.transpose(1, 2)

            output, h, c = self.seg_lstm[i](output, h, c) # x=[NS, K, C], hc=[D, NS, C]
            if i < self.n_blocks - 1:
                h = h.reshape(-1, N, S, self.hidden_size).permute(1, 2, 0, 3) # [D, N, S, C] -> [N, S, D, C]
                c = c.reshape(-1, N, S, self.hidden_size).permute(1, 2, 0, 3) # [D, N, S, C] -> [N, S, D, C]
                h, c = self.mem_lstm[i](h, c) # [D, NS, C]
        
        if self.seg_overlap:
            output = output.reshape(N, S, K, C)
            output = self.merge(output, rest)
            output = self.output_fc(output)
        
        else:
            output = output.reshape(N, S*K, C)[:, :T, :]
            output = self.output_fc(output.transpose(1, 2))
        
        return output
