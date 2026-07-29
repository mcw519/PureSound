from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lobe.rnn import SingleRNN
from .lobe.trivial import FiLM, spectral_compression
from .unet import Unet


class VADHead(nn.Module):
    """Frame-level speech-activity logits from the bottleneck. Causal in time."""

    def __init__(self, enc_channels: int, hidden: int, kernel_t: int):
        super().__init__()
        self.kernel_t = kernel_t
        self.proj = nn.Linear(enc_channels, hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_t, padding=0, groups=1)
        self.act = nn.SiLU()
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> pool over F -> [N, C, T]
        h = x.mean(dim=2)  # [N, C, T]
        h = self.proj(h.transpose(1, 2)).transpose(1, 2)  # [N, hidden, T]
        h = F.pad(h, (self.kernel_t - 1, 0))  # causal
        h = self.act(self.dwconv(h))
        return self.out(h).squeeze(1)  # [N, T]


class DistHead(nn.Module):
    """Utterance-level distance/DRR regression from the bottleneck.

    Auxiliary multi-task pressure that makes the bottleneck encode the physical
    proximity cues (DRR / source distance) rather than the capture-chain
    signature of the training near-field rows. Predicts
    ``[fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]``;
    supervision comes from the dataset's free scalar labels and is NaN-masked
    (see DistHeadRegressionLoss). Training-only: inference never reads it and
    the streaming export is untouched."""

    def __init__(self, enc_channels: int, hidden: int = 128, n_out: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> global pool -> [N, C] -> [N, n_out]
        return self.net(x.mean(dim=(2, 3)))


class DPRNNblock2D(nn.Module):
    """
    DPRNN-2D modul which be parts of DPCRN.

    Args:
        input_size: input 4D-tensor's channel dimension
        hidden_size: RNN's hidden dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        embedding_size: Optional[int] = None,
        fused_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.fused_type = fused_type.lower()

        if embedding_size is not None:
            if self.fused_type == "film":
                self.film = FiLM(
                    feats_size=input_size,
                    embed_size=embedding_size,
                    input_norm=True,
                )
            else:
                raise NameError

        self.intra_rnn = SingleRNN(
            "LSTM", input_size, hidden_size, bidirectional=True, dropout=dropout
        )
        self.intra_norm = nn.LayerNorm(input_size)

        self.inter_rnn = SingleRNN(
            "LSTM", input_size, hidden_size, bidirectional=False, dropout=dropout
        )
        self.inter_norm = nn.LayerNorm(input_size)

    def forward(
        self,
        x: torch.Tensor,
        intra_skip: bool = True,
        inter_skip: bool = True,
        embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input tensor has shape as [N, ch, C, T]
        Inputs:
            x -- [N, ch, C, T]

        Returns:
            output -- [N, ch, C, T]
        """
        self.intra_rnn.rnn.flatten_parameters()
        self.inter_rnn.rnn.flatten_parameters()

        x_intra_skip = x.clone()
        N, CH, C, T = x.shape

        # intra-chunk, time independent and frequency dependent
        x = x.transpose(1, -1).reshape(
            N * T, C, CH
        )  # [N, CH, C, T] -> [N, T, C, CH] -> [N*T, C, CH]
        x = self.intra_rnn(x.permute(0, 2, 1))  # [N*T, C, CH] -> [N*T, CH, C]
        x = x.permute(0, 2, 1)  # [N*T, CH, C] -> [N*T, C, CH]
        x = self.intra_norm(x)
        x = x.reshape(N, T, C, -1)
        x = x.transpose(1, -1)  # [N, CH, C, T]

        if intra_skip:
            x = x_intra_skip + x  # [N, CH, C, T]

        x_inter_skip = x.clone()

        if self.embedding_size is not None and embed is not None:
            if self.fused_type == "film":
                x = x.permute(0, 2, 1, 3)  # [N, C, CH, T]
                x = x.reshape(-1, CH, T)
                embed_inp = embed.unsqueeze(1).repeat(1, C, 1)
                embed_inp = embed_inp.reshape(-1, embed.shape[-1])
                x = self.film(x, embed_inp)
                x = x.reshape(N, C, CH, T)
                x = x.permute(0, 2, 1, 3)
            else:
                raise NotImplementedError

        # inter-chunk, time dependent and frequency independent
        x = x.permute(0, 2, 3, 1).reshape(
            N * C, T, -1
        )  # [N, CH, C, T] -> [N, C, T, CH] -> [N*C, T, CH]
        x = self.inter_rnn(x.permute(0, 2, 1))  # [N*C, T, CH] -> [N*C, CH, T]
        x = x.permute(0, 2, 1)  # [N*C, CH, T] -> [N*C, T, CH]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(N, C, CH, T)
        x = x.permute(0, 2, 1, 3)

        if inter_skip:
            x = x_inter_skip + x

        return x


class DPCRN(Unet):
    def __init__(
        self,
        input_dim: int = 512,
        dvec_dim: Optional[int] = None,
        activation_type: str = "PReLU",
        norm_type: str = "bN2d",
        dropout: float = 0.05,
        channels: Tuple = (1, 32, 32, 32, 64, 128),
        transpose_t_size: int = 2,
        transpose_delay: bool = False,
        skip_conv: bool = False,
        kernel_t: Tuple = (2, 2, 2, 2, 2),
        stride_t: Tuple = (1, 1, 1, 1, 1),
        dilation_t: Tuple = (1, 1, 1, 1, 1),
        kernel_f: Tuple = (5, 3, 3, 3, 3),
        stride_f: Tuple = (2, 2, 1, 1, 1),
        dilation_f: Tuple = (1, 1, 1, 1, 1),
        delay: Tuple = (0, 0, 0, 0, 0),
        rnn_hidden: int = 128,
        spectral_compress: bool = False,
        vad_head: Optional[Dict] = None,
        dist_head: Optional[Dict] = None,
    ):
        super().__init__(
            input_dim,
            activation_type,
            norm_type,
            dropout,
            channels,
            transpose_t_size,
            skip_conv,
            kernel_t,
            stride_t,
            dilation_t,
            kernel_f,
            stride_f,
            dilation_f,
            delay,
        )

        self.transpose_delay = transpose_delay
        self.rnn_hidden = rnn_hidden
        self.spectral_compress = spectral_compress
        self.dvec_dim = dvec_dim
        self.vad_head_args = vad_head

        # DPRNN block
        self.dprnn_block1 = DPRNNblock2D(
            input_size=channels[-1],
            hidden_size=rnn_hidden,
            dropout=dropout,
            embedding_size=dvec_dim,
            fused_type="FiLM",
        )
        self.dprnn_block2 = DPRNNblock2D(
            input_size=channels[-1],
            hidden_size=rnn_hidden,
            dropout=dropout,
            embedding_size=dvec_dim,
            fused_type="FiLM",
        )

        if vad_head is not None and vad_head.get("enabled", False):
            self.vad_head = VADHead(
                enc_channels=channels[-1],
                hidden=vad_head.get("hidden", channels[-1]),
                kernel_t=vad_head.get("kernel_t", 5),
            )
        else:
            self.vad_head = None
        self.last_vad_logits: Optional[torch.Tensor] = None

        if dist_head is not None and dist_head.get("enabled", False):
            self.dist_head = DistHead(
                enc_channels=channels[-1],
                hidden=dist_head.get("hidden", 128),
            )
        else:
            self.dist_head = None
        self.last_dist_preds: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, dvec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, CH, C, T]
            dvec: speaker embedding tensor shape [N, D]

        Returns:
            output tensor has shape [N, CH, C, T]
        """
        if self.spectral_compress:
            x = spectral_compression(x, alpha=0.3, dim=1)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, 1, C, T]

        x = self.input_norm(x)
        if dvec is not None:
            dvec = nn.functional.normalize(dvec, dim=1, p=2)

        skip = [x.clone()]

        # forward CNN-down layers
        for cnn_layer in self.cnn_down:
            x = cnn_layer(x)  # [N, ch, C, T]
            skip.append(x)

        # forward dprnn
        x = self.dprnn_block1(x, embed=dvec)  # [N, ch, C, T]
        x = self.dprnn_block2(x, embed=dvec)  # [N, ch, C, T]

        if self.vad_head is not None:
            self.last_vad_logits = self.vad_head(x)
        else:
            self.last_vad_logits = None

        if self.dist_head is not None:
            self.last_dist_preds = self.dist_head(x)
        else:
            self.last_dist_preds = None

        # forward CNN-up layers
        for i, cnn_layer in enumerate(self.cnn_up):
            if self.skip_conv:
                x += self.skip_cnn[i](skip[-i - 1])
            else:
                x = torch.cat([x, skip[-i - 1]], dim=1)

            x = cnn_layer(x)
            if self.t_kernel != 1:
                if self.transpose_delay:
                    x = x[
                        ..., (self.t_kernel - 1) :
                    ]  # transpose-conv with t-kernel size would increase (t-1) length
                else:
                    x = x[
                        ..., : -(self.t_kernel - 1)
                    ]  # transpose-conv with t-kernel size would increase (t-1) length

        return x
