from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .lobe.norm import LayerNorm2D, cLN, gLN


class ContextFeature(nn.Module):
    """
    Contextual frames information by appending front/back frames.

    Args:
        num_right: right contexts (past), this may create `num_right - 1` frames delay.
        num_left: left contexts (future), this may create `num_left` frames delay and reduce `num_left` frames length.
        equal_length: make input/output tensor has equal length by repeat begin/end frames
    """

    def __init__(
        self, num_right: int, num_left: int, equal_length: bool = True
    ) -> None:
        super().__init__()
        self.right = num_right
        self.left = num_left
        self.equal_length = equal_length

    @staticmethod
    def add_context(
        x: torch.Tensor, num_right: int, num_left: int, equal_length: bool = True
    ) -> torch.Tensor:
        """Unfold tensor with context."""
        context = []
        right_context = []
        left_context = []
        if num_right != 0:
            shift_x = x.clone()
            for _ in range(num_right):
                zeros = (
                    x[..., 0].clone().unsqueeze(-1)
                    if equal_length
                    else torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)
                )
                shift_x = torch.cat([zeros, shift_x], dim=-1)[..., :-1]
                right_context.insert(0, shift_x)

            context.append(torch.cat(right_context, dim=1))

        context.append(x)

        if num_left != 0:
            shift_x = x.clone()
            for _ in range(num_left):
                zeros = (
                    x[..., -1].clone().unsqueeze(-1)
                    if equal_length
                    else torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)
                )
                shift_x = torch.cat([shift_x, zeros], dim=-1)[..., 1:]
                left_context.append(shift_x)

            context.append(torch.cat(left_context, dim=1))

        if equal_length:
            return torch.cat(context, dim=1)

        if num_left != 0:
            return torch.cat(context, dim=1)[..., num_right:-num_left]
        else:
            return torch.cat(context, dim=1)[..., num_right:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape is (N, C, T)
        """
        x_context = self.add_context(x, self.right, self.left, self.equal_length)

        return x_context


class IntraSpectralLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        hidd_size: int,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        intra_in_channels = channels * kernel_size

        self.intra_context = ContextFeature(
            num_right=kernel_size // 2, num_left=kernel_size // 2, equal_length=True
        )
        self.intra_norm = cLN(channel_size=intra_in_channels)
        self.intra_rnn = nn.LSTM(
            intra_in_channels,
            hidd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.intra_linear = nn.ConvTranspose1d(
            hidd_size * 2, channels, kernel_size=kernel_size, stride=1
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        Return:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        """
        self.intra_rnn.flatten_parameters()

        x_backup = x
        batch, chdim, fdim, nframes = x.shape
        # intra-chunk, time independent

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, chdim, fdim)  # [N * T, CH, C]
        x = self.intra_context(x)  # [N * T, CH * X, C]
        x = self.intra_norm(x)  # [N * T, CH * X, C]
        x = x.permute(0, 2, 1)  # [N * T, C, CH * X]

        x, _ = self.intra_rnn(x)
        x = x.transpose(1, 2).contiguous()  # [N * T, 2 * hidd_size, C]
        x = self.intra_linear(x).contiguous()[
            ..., self.kernel_size - 1 :
        ]  # [N * T, 2 * hidd_size, C + (kernel_f - 1)] -> # [N * T, CH, C]
        x = x.reshape(batch, nframes, chdim, fdim)
        x = x.permute(0, 2, 3, 1)  # [N, CH, C, T]
        x = x_backup + x

        return x


class SubbandTemporalLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        hidd_size: int,
        n_delay: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_delay = n_delay

        self.inter_context = ContextFeature(
            num_right=n_delay, num_left=kernel_size - n_delay - 1, equal_length=True
        )

        inter_in_channels = channels * kernel_size
        self.inter_norm = cLN(channel_size=inter_in_channels)

        self.inter_rnn = nn.LSTM(
            inter_in_channels,
            hidd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.inter_linear = nn.ConvTranspose1d(
            hidd_size, channels, kernel_size=kernel_size, stride=1
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        Return:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        """
        self.inter_rnn.flatten_parameters()

        x_backup = x
        batch, chdim, fdim, nframes = x.shape

        # inter-chunk, frequency dependent
        x = x.permute(0, 2, 1, 3)  # [N, C, CH, T]
        x = x.reshape(-1, chdim, nframes)  # [N * C, CH, T]
        x = self.inter_context(x)  # [N * C, CH * X, T]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1)  # [N * C, T, CH * X]

        x, _ = self.inter_rnn(x)
        x = x.transpose(1, 2).contiguous()  # [N * C, CH * X, T]
        x = self.inter_linear(x).contiguous()
        x = x[..., self.n_delay : self.n_delay + nframes]
        x = x.reshape(batch, fdim, chdim, nframes)
        x = x.permute(0, 2, 1, 3)  # [N, CH, C, T]
        x = x_backup + x

        return x


class FullbandSelfAttention(nn.Module):
    def __init__(
        self,
        fdim: int,
        channels: int,
        channels_qk: int,
        n_head: int,
        attent_range: Optional[int] = None,
    ):
        super().__init__()
        assert (
            channels % n_head == 0
        ), f"input channel {channels} can't be devided by {n_head} heads."

        self.n_head = n_head
        self.attent_range = attent_range

        self.atten_q = nn.ModuleList()
        self.atten_k = nn.ModuleList()
        self.atten_v = nn.ModuleList()

        for _ in range(n_head):
            self.atten_q.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels_qk,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.PReLU(),
                    LayerNorm2D(ch=channels_qk, f=fdim),
                )
            )

            self.atten_k.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels_qk,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.PReLU(),
                    LayerNorm2D(ch=channels_qk, f=fdim),
                )
            )

            self.atten_v.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels // n_head,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.PReLU(),
                    LayerNorm2D(ch=channels // n_head, f=fdim),
                )
            )

        self.atten_concat_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels, kernel_size=1, stride=1
            ),
            nn.PReLU(),
            LayerNorm2D(ch=channels, f=fdim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        Return:
            x   (torch.Tensor): input feature with shape [N, CH, C, T]
        """
        all_Q = []
        all_K = []
        all_V = []

        atten_x = x
        for ii in range(self.n_head):
            all_Q.append(self.atten_q[ii](atten_x))  # [N, CH', C, T]
            all_K.append(self.atten_k[ii](atten_x))  # [N, CH', C, T]
            all_V.append(self.atten_v[ii](atten_x))  # [N, CH', C, T]

        Q = torch.stack(all_Q, dim=1)  # [N, n_head, CH', C, T]
        K = torch.stack(all_K, dim=1)  # [N, n_head, CH', C, T]
        V = torch.stack(all_V, dim=1)  # [N, n_head, CH', C, T]

        batch, n_head, chdim, fdim, nframes = Q.shape
        Q = Q.reshape(batch, n_head, -1, nframes)  # [N, n_head, CH' * C, T]
        K = K.reshape(batch, n_head, -1, nframes)  # [N, n_head, CH' * C, T]
        V = V.reshape(batch, n_head, -1, nframes)  # [N, n_head, CH // n_head * C, T]
        embed_dim = Q.shape[2]

        atten_mat = torch.matmul(Q.transpose(-2, -1).contiguous(), K) / (
            embed_dim**0.5
        )  # [N, n_head, T, T]

        if self.attent_range is None:
            mask = torch.empty(Q.shape[-1], Q.shape[-1]).fill_(-np.inf).triu_(1)
        else:
            mask = torch.tril(
                torch.ones(Q.shape[-1], Q.shape[-1]), diagonal=-self.attent_range
            )
            mask = mask + torch.triu(torch.ones(Q.shape[-1], Q.shape[-1]), diagonal=1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float(0.0))
                .masked_fill(mask == 1, -np.inf)
            )

        mask = mask.to(atten_mat.device)
        atten_mat = atten_mat + mask
        atten_mat = torch.softmax(atten_mat, dim=-1)
        self_atten = torch.matmul(
            atten_mat, V.transpose(-2, -1).contiguous()
        )  # [N, n_head, T, CH // n_head * C]
        self_atten = self_atten.transpose(
            -2, -1
        ).contiguous()  # [N, n_head, CH // n_head * C, T]
        self_atten = self_atten.reshape(batch, -1, fdim, nframes)  # [N, CH, C, T]
        self_atten = self.atten_concat_proj(self_atten)  # [N, CH, C, T]

        x = x + self_atten

        return x, atten_mat


class GridBlock(nn.Module):
    def __init__(
        self,
        ch_dim: int,
        f_dim: int,
        hid_dim: int,
        kernel_t: int = 3,
        kernel_f: int = 3,
        n_head: int = 4,
        approx_qk_dim: int = 8,
        n_delay: int = 0,
        attent_range: Optional[int] = None,
    ):
        super().__init__()
        assert kernel_f % 2 != 0, "kernel size (f) {kernel_f} should be odd number."
        assert kernel_t > n_delay, "delay size must be smaller than kernel size t."
        self.n_delay = n_delay
        self.kernel_f = kernel_f
        self.kernel_t = kernel_t
        self.n_head = n_head
        self.attent_range = attent_range

        self.intra_frame_spectral = IntraSpectralLayer(
            channels=ch_dim,
            kernel_size=kernel_f,
            hidd_size=hid_dim,
        )

        self.subband_temporal = SubbandTemporalLayer(
            channels=ch_dim,
            kernel_size=kernel_t,
            hidd_size=hid_dim,
            n_delay=n_delay,
        )

        self.fullband_self_attention = FullbandSelfAttention(
            fdim=f_dim,
            channels=ch_dim,
            channels_qk=approx_qk_dim,
            n_head=n_head,
            attent_range=attent_range,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_atten_mat: bool = False,
    ):
        """
        Args:
            x (torch.Tensor): [N, CH, C, T]
            return_atten_mat (bool): if true, return the attention matrix

        Returns:
            x (torch.Tensor): output tensor has same shape as input tensor
            atten_mat (torch.Tensor): attention matrix has shape (batch, nheads, nframes, nframes)
        """
        # intra-spectral
        x = self.intra_frame_spectral(x)

        # subband-temporal
        x = self.subband_temporal(x)

        # fullband-attention
        x, mat = self.fullband_self_attention(x)

        if return_atten_mat:
            return x, mat
        else:
            return x


class TFGridNet(nn.Module):
    """
    Implement TF-grid Net

    Generic_Args:
        input_dim (int): input feature dimension

    TF_grid_Net_Args:
        channel_dim (int): channel size of input feature
        n_block (int): N TFGrid blocks
        block_delay_frames (int): how much delay frames in each block
        kernel_f_size (int): kernel size of freq axis in each Conv2d
        kernel_t_size (int): kernel size of time axis in each Conv2d
        f_stride (int): overall frequency stride number
        atten_range (int): causal attention with restricted contexture

    References:
        [1] TF-GRIDNET: MAKING TIME-FREQUENCY DOMAIN MODELS GREAT AGAIN FOR MONAURAL SPEAKER SEPARATION
        [2] MULTI-CHANNEL TARGET SPEAKER EXTRACTION WITH REFINEMENT: THE WAVLAB SUBMISSION TO THE SECOND CLARITY ENHANCEMENT CHALLENGE
    """

    def __init__(
        self,
        inp_channel_dim: int = 2,
        input_dim: int = 256,
        input_norm: str = "gLN",
        channel_dim: int = 32,
        lstm_dim: int = 128,
        n_block: int = 6,
        block_delay_frames: int = 0,
        kernel_f_size: int = 5,
        kernel_t_size: int = 5,
        f_stride: int = 4,
        n_head: int = 4,
        channel_qk: int = 4,
        attent_range: int = 100,
    ):
        super().__init__()
        assert kernel_f_size >= f_stride

        # features
        self.inp_channel_dim = inp_channel_dim
        self.num_freq = input_dim

        # model setting
        self.n_block = n_block
        self.channel_dim = channel_dim
        self.kernel_f_size = kernel_f_size
        self.kernel_t_size = kernel_t_size
        self.f_stride = f_stride
        self.lstm_dim = lstm_dim
        self.block_delay_frames = block_delay_frames
        self.n_head = n_head
        self.channel_qk = channel_qk
        self.attent_range = attent_range

        # input 3x3 convolution
        assert input_norm in ["LayerNorm2D", "gLN"]

        if input_norm == "LayerNorm2D":
            _inp_norm = LayerNorm2D(ch=channel_dim, f=input_dim // f_stride)
        else:
            _inp_norm = gLN(channel_size=channel_dim)

        # Add padding to prevent the incorrect size matching between input and output
        self.in_conv = nn.Sequential(
            nn.ZeroPad2d((2, 0, 1, 1)),
            nn.Conv2d(
                in_channels=inp_channel_dim,
                out_channels=channel_dim,
                kernel_size=(3, 3),
                stride=(f_stride, 1),
            ),
            _inp_norm,
        )

        # TF-Grid blocks
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(
                GridBlock(
                    ch_dim=channel_dim,
                    f_dim=input_dim // f_stride,
                    hid_dim=lstm_dim,
                    kernel_f=kernel_f_size,
                    kernel_t=kernel_t_size,
                    n_head=n_head,
                    approx_qk_dim=channel_qk,
                    n_delay=block_delay_frames,
                    attent_range=attent_range,
                )
            )

        # output 3x3 deconvolution
        s = f_stride
        k = 3
        p = k // 2
        op = s - k + 2 * p

        self.out_conv = nn.ConvTranspose2d(
            in_channels=channel_dim,
            out_channels=self.inp_channel_dim,
            kernel_size=(3, 3),
            stride=(f_stride, 1),
            padding=(p, 0),
            output_padding=(op, 0),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): RI-concate with shape [N, input_dim, T] or RI-stack with shape [N, 2, input_dim, T]

        Returns:
            return tf-mask as same shape of input_dim, deepfilter-coeff with shape as 2*input_dim
        """

        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, 1, C, T], extend chaneel axis

        x = self.in_conv(x)  # [N, CH, C, T]
        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)
        x = x[..., :-2]  # because out_conv kernel size is 3

        return x
