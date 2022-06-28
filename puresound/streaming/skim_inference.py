from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from puresound.nnet.skim import SkiM


class StreamingSkiM(SkiM):
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
                dropout: float = 0):
        super().__init__(input_size, hidden_size, output_size, n_blocks, seg_size, seg_overlap, causal, embed_dim, embed_norm, embed_fusion, block_with_embed, dropout)
    
    @torch.no_grad()
    def step_chunk(self,
                    x: torch.Tensor,
                    seg_lstm_h_state: Optional[torch.Tensor] = None,
                    mem_lstm_h_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    seg_lstm_c_state: Optional[torch.Tensor] = None,
                    mem_lstm_c_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    embed: Optional[torch.Tensor] = None):
        """
        Args:
            input tensor x has shape [1, K, C], We meet the input x tensor as a series of frames equall to the chunk_size=K.
            seg_lstm_h_state has shape [N-Layer, D, 1, C] or None for first input chunk.
            seg_lstm_c_state has shape [N-Layer, D, 1, C] or None for first input chunk.
            mem_lstm_h_hidden/mem_lstm_c_hidden for mem-LSTM's hidden/cell states continued
            speaker or others embedding tensor has shape [1, embedding_size]
        """
        if self.embed_norm and embed is not None:
            embed = F.normalize(embed, p=2, dim=1)
        
        output = []
        # Initialize each Seg-LSTM layers's hidden/cell state
        # In SkiM, only first layer's initial state is None, others come from past Mem-LSTM's output
        if seg_lstm_h_state is not None and seg_lstm_c_state is not None:
            seg_lstm_h_state = [None] + [seg_lstm_h_state[i] for i in range(self.n_blocks - 1)]
            seg_lstm_c_state = [None] + [seg_lstm_c_state[i] for i in range(self.n_blocks - 1)]
        
        else:
            seg_lstm_h_state = [None] + [None for _ in range(self.n_blocks - 1)]
            seg_lstm_c_state = [None] + [None for _ in range(self.n_blocks - 1)]
        
        # initialize mem-LSTM hidden states
        if mem_lstm_h_hidden is None and mem_lstm_c_hidden is None:
            mem_lstm_h_hidden = [None for _ in range(self.n_blocks - 1)]
            mem_lstm_c_hidden = [None for _ in range(self.n_blocks - 1)]

        # intra-chunk-op
        for f_idx in range(x.shape[1]): # alone chunk_size(K) axis
            cur_inp = x[:, f_idx, :].view(1, 1, -1) # [1, 1, C] -> [1, 1, C] 1 frame input
            for i in range(self.n_blocks):
                if embed is not None and self.block_with_embed[i]:
                    cur_inp = self.seg_input_fusion[i](cur_inp.view(1, -1, 1), embed) # fusion layer need [N, C, T]
                    cur_inp = cur_inp.reshape(1, 1, -1)

                cur_inp, seg_h, seg_c = self.seg_lstm[i](cur_inp, seg_lstm_h_state[i], seg_lstm_c_state[i]) # cur_inp=[1, 1, C], seg_h/seg_c=[D, 1, C]
                # update Seg-LSTM's status
                seg_lstm_h_state[i] = seg_h
                seg_lstm_c_state[i] = seg_c
            
            cur_out = self.output_fc(cur_inp.reshape(1, -1, 1))
            output.append(cur_out) # [1, C, 1]
            
        # inter-segment-op
        for i in range(self.n_blocks-1):
            seg_h = seg_lstm_h_state[i].reshape(-1, 1, 1, self.hidden_size).permute(1, 2, 0, 3)
            seg_c = seg_lstm_c_state[i].reshape(-1, 1, 1, self.hidden_size).permute(1, 2, 0, 3)
            mem_h, mem_c, _hid_hidden, _cell_hidden = self.mem_lstm[i](seg_h, seg_c, h_states=mem_lstm_h_hidden[i], c_states=mem_lstm_c_hidden[i], return_all=True, streaming=True) # [N, S, D, C]
            seg_lstm_h_state[i] = mem_h # [D, 1, C]
            seg_lstm_c_state[i] = mem_c # [D, 1, C]

            # update Mem-LSTM's hidden and cell status
            mem_lstm_h_hidden[i] = _hid_hidden
            mem_lstm_c_hidden[i] = _cell_hidden

        output = torch.cat(output, dim=-1)

        return output, seg_lstm_h_state[:-1], mem_lstm_h_hidden, seg_lstm_c_state[:-1], mem_lstm_c_hidden
