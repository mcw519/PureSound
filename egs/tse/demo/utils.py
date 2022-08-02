from typing import Optional

import torch
import torch.nn as nn
from puresound.nnet.conv_tasnet import TCN
from puresound.nnet.lobe.encoder import FreeEncDec
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling
from puresound.streaming.skim_inference import StreamingSkiM


class DemoSpeakerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True)
        self.speaker_net = nn.ModuleList(
            [TCN(128, 256, 3, dilation=2**i, causal=False, tcn_norm='gLN', dconv_norm='gGN') for i in range(5)] + \
            [AttentiveStatisticsPooling(128, 128), nn.Conv1d(128*2, 192, 1, bias=False)])
    
    def forward(self, enroll: torch.Tensor) -> torch.Tensor:
        dvec = self.encoder(enroll)
        for layer in self.speaker_net:
            dvec = layer(dvec)
        
        return dvec.squeeze()
    
    @torch.no_grad()
    def get_speaker_embedding(self, enroll: torch.Tensor):
        return self.forward(enroll)


class DemoTseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FreeEncDec(win_length=32, hop_length=16, laten_length=128, output_active=True)
        self.masker = StreamingSkiM(input_size=128, hidden_size=256, output_size=128, n_blocks=4, seg_size=150, seg_overlap=False, causal=True,
            embed_dim=192, embed_norm=True, block_with_embed=[1, 1, 1, 1], embed_fusion='FiLM')
        
        self.training = False
        self.queue = None
        self.win_size = 32
        self.hop_size = 16
        self.ola_size = int(self.win_size - self.hop_size)

    def forward(self, noisy: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def streaming_inference(self, chunk: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        if embed.dim() == 1:
            embed = embed.unsqueeze(0)

        if self.queue is None:
            zero_pad = torch.zeros_like(chunk)
            self.queue = torch.cat([zero_pad, chunk], dim=-1)
            return None

        else:
            cur_frame = torch.cat([self.queue[:, self.hop_size:], chunk], dim=-1)
            self.queue = cur_frame

        feats = self.encoder(cur_frame)
        mask = self.masker.step_frame(feats, embed)
        gen_wav = self.encoder.inverse(feats * mask)
        
        return gen_wav
    
    @torch.no_grad()
    def streaming_inference_chunk(self, chunk: torch.Tensor, embed: torch.Tensor, pre_wav: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embed.dim() == 1:
            embed = embed.unsqueeze(0)
        
        total_frames = chunk.shape[-1] // self.hop_size
        for i in range(total_frames):
            s = int(i * self.hop_size)
            e = int(s + self.hop_size)
            cur_frame = chunk[:, s:e].view(1, -1)
            cur_wav = self.streaming_inference(cur_frame, embed)
            if cur_wav is not None:
                pre_wav = overlap_add(pre_wav, cur_wav.squeeze(), self.ola_size)
        
        return pre_wav


def overlap_add(a: torch.Tensor, b: torch.Tensor, overlap_length: int) -> torch.Tensor:
    """a and b are 1D tensor"""
    if a is None:
        return b
    else:
        keep_a, overlap_a = a[:-overlap_length], a[-overlap_length:]
        keep_b, overlap_b = b[overlap_length:], b[:overlap_length]
        return torch.cat([keep_a, (overlap_a+overlap_b)/2, keep_b], dim=-1)
