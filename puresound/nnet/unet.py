from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_tasnet import TCN, GatedTCN
from .lobe.activation import get_activation
from .lobe.norm import get_norm
from .lobe.rnn import FSMN, ConditionFSMN


class Unet(nn.Module):
    """
    Generic_Args:
        input_dim: input feature dimension
        activation_type: activation function
        norm_type: normalization function
        dropout: if not 0, add dropout in down-CNN layers

    Unet_Args:
        channels: controlled input/output channel for Unet
        kernel_t: kernel size in time axis for each down cnn layer
        kernel_f: kernel size in freq axis for each down/up cnn layer
        stride_t: stride size in time axis for each down cnn layer
        stride_f: stride size in freq axis for each down/up cnn layer
        dilation_t: dilation size in time axis for each down cnn layer
        dilation_f: dilation size in freq axis for each down/up cnn layer
        delay: add lookahead frames in each down cnn layers, if 0 means causal cnn operation
        transpose_t_size: the kernel size of ConvTranspose2d's time axis for up cnn layer
        skip_conv
    """

    def __init__(
        self,
        input_dim: int = 512,
        activation_type: str = "PReLU",
        norm_type: str = "bN2d",
        dropout: float = 0.05,
        channels: Tuple = (1, 1, 8, 8, 16, 16),
        transpose_t_size: int = 2,
        skip_conv: bool = False,
        kernel_t: Tuple = (5, 1, 9, 1, 1),
        stride_t: Tuple = (1, 1, 1, 1, 1),
        dilation_t: Tuple = (1, 1, 1, 1, 1),
        kernel_f: Tuple = (1, 5, 1, 5, 1),
        stride_f: Tuple = (1, 4, 1, 4, 1),
        dilation_f: Tuple = (1, 1, 1, 1, 1),
        delay: Tuple = (0, 0, 1, 0, 0),
        multi_output: int = 1,
    ):
        super().__init__()
        assert (
            len(kernel_t)
            == len(kernel_f)
            == len(stride_t)
            == len(stride_f)
            == len(dilation_t)
            == len(dilation_f)
        )
        self.input_dim = input_dim
        self.multi_output = multi_output
        self.activation_type = activation_type
        self.norm_type = norm_type
        self.dropout = dropout
        self.skip_conv = skip_conv

        # Structure information
        self.kernel_t = kernel_t
        self.kernel_f = kernel_f
        self.stride_t = stride_t
        self.stride_f = stride_f
        self.dilation_t = dilation_t
        self.dilation_f = dilation_f
        self.transpose_t_size = transpose_t_size

        active_cls = get_activation(activation_type.lower())
        norm_cls = get_norm(norm_type)

        self.n_cnn = len(kernel_t)
        self.channels = list(channels)
        self.kernel = list(
            zip(kernel_f, kernel_t)
        )  # each layer's kernel size (freq, time)
        self.delay = delay  # how much delay for each layer
        self.dilation = list(zip(dilation_f, dilation_t))
        self.stride = list(zip(stride_f, stride_t))
        self.t_kernel = transpose_t_size
        self.num_freq = input_dim

        # CNN-down, downsample in frequency axis
        self.cnn_down = nn.ModuleList()
        for i in range(self.n_cnn):
            encode = []
            freq_pad = (
                self.kernel[i][0] // 2 * dilation_f[i],
                self.kernel[i][0] // 2 * dilation_f[i],
            )  # center padding in frequency axis
            time_pad = ((self.kernel[i][1] - 1) * dilation_t[i] - self.delay[i], self.delay[i])
            encode += [
                nn.ZeroPad2d(time_pad + freq_pad),  # (left, right, top, down)
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=self.kernel[i],
                    stride=self.stride[i],
                    dilation=self.dilation[i],
                ),
                norm_cls(self.channels[i + 1]),
                active_cls(),
                nn.Dropout(self.dropout),
            ]

            self.cnn_down.append(nn.Sequential(*encode))

        # CNN-up, upsample in frequency axis
        self.cnn_up = nn.ModuleList()
        skip_double = 2 if not skip_conv else 1
        skip_double = [skip_double] * self.n_cnn

        for i in reversed(range(self.n_cnn)):
            s, _ = self.stride[i]
            k = self.kernel[i][0]
            p = k // 2
            op = s - k + 2 * p
            encode = []
            if i != 0:
                encode += [
                    nn.ConvTranspose2d(
                        self.channels[i + 1] * skip_double[i],
                        self.channels[i],
                        kernel_size=(k, self.t_kernel),
                        stride=self.stride[i],
                        dilation=1, #self.dilation[i],
                        padding=(p, 0),
                        output_padding=(op, 0),
                    ),
                    norm_cls(self.channels[i]),
                    active_cls(),
                ]

            else:
                # linear output
                encode += [
                    nn.ConvTranspose2d(
                        self.channels[i + 1] * skip_double[i],
                        self.channels[i] * self.multi_output,
                        kernel_size=(k, self.t_kernel),
                        stride=self.stride[i],
                        dilation=1, #self.dilation[i],
                        padding=(p, 0),
                        output_padding=(op, 0),
                    )
                ]

            self.cnn_up.append(nn.Sequential(*encode))

        if skip_conv:
            self.skip_cnn = nn.ModuleList()
            for i in reversed(range(self.n_cnn)):
                encode = []
                encode += [
                    nn.Conv2d(
                        self.channels[i + 1],
                        self.channels[i + 1],
                        kernel_size=(1, 1),
                        stride=1,
                    ),
                    active_cls(),
                ]

                self.skip_cnn.append(nn.Sequential(*encode))

    def shape_info(self):
        # input_shape = [N, ch, C, T]
        # conv-transpose output size is:
        #   (freq): (input_shape[2] -1) * stride[0] - 2*padding[0] + dilation[0] * (kernel_size[0]-1) + output_padding[0] + 1
        #   (time): (input_shape[2] -1) * stride[1] - 2*padding[1] + dilation[1] * (kernel_size[1]-1) + output_padding[1] + 1
        down_shape = [self.num_freq]
        for i in range(self.n_cnn):
            stride, _ = self.stride[i]
            if down_shape[i - 1] % stride == 0:
                _f = down_shape[-1] // stride
            else:
                _f = down_shape[-1] // stride
                _f += 1
            down_shape.append(_f)

        up_shape = [_f]
        for i in range(self.n_cnn):
            stride, _ = self.stride[-i - 1]
            kernel_size = self.kernel[-i - 1][0]
            padding = kernel_size // 2
            output_padding = stride - kernel_size + 2 * padding
            _f = (
                (up_shape[-1] - 1) * stride
                - 2 * padding
                + self.dilation[-i - 1][0] * (kernel_size - 1)
                + output_padding
                + 1
            )
            up_shape.append(_f)

        return down_shape, up_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, CH, C, T]

        Returns:
            output tensor has shape [N, CH, C, T]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, 1, C, T]

        skip = [x.clone()]

        # forward CNN-down layers
        for cnn_layer in self.cnn_down:
            x = cnn_layer(x)  # [N, ch, C, T]
            skip.append(x)

        # forward CNN-up layers
        for i, cnn_layer in enumerate(self.cnn_up):
            if self.skip_conv:
                x += self.skip_cnn[i](skip[-i - 1])
            else:
                x = torch.cat([x, skip[-i - 1]], dim=1)

            x = cnn_layer(x)
            if self.t_kernel != 1:
                x = x[
                    ..., : -(self.t_kernel - 1)
                ]  # transpose-conv with t-kernel size would increase (t-1) length

        return x

    @property
    def get_args(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "activation_type": self.activation_type,
            "norm_type": self.norm_type,
            "dropout": self.dropout,
            "channels": self.channels,
            "transpose_t_size": self.transpose_t_size,
            "skip_conv": self.skip_conv,
            "kernel_t": self.kernel_t,
            "stride_t": self.stride_t,
            "dilation_t": self.dilation_t,
            "kernel_f": self.kernel_f,
            "stride_f": self.stride_f,
            "dilation_f": self.dilation_f,
            "delay": self.delay,
            "multi_output": self.multi_output,
        }


class UnetTcn(Unet):
    """
    Improve temporal modeling ability by inserting a TCN inside an Unet model.

    Args:
        embed_dim: Embedding feature dimension.
        embed_norm: If True, applies the 2-norm on the input embedding.
    """

    def __init__(
        self,
        embed_dim: int = 0,
        embed_norm: bool = False,
        input_type: str = "RI",
        input_dim: int = 512,
        activation_type: str = "PReLU",
        norm_type: str = "bN2d",
        dropout: float = 0.05,
        channels: Tuple = (1, 1, 8, 8, 16, 16),
        transpose_t_size: int = 2,
        transpose_delay: bool = False,
        skip_conv: bool = False,
        kernel_t: Tuple = (5, 1, 9, 1, 1),
        stride_t: Tuple = (1, 1, 1, 1, 1),
        dilation_t: Tuple = (1, 1, 1, 1, 1),
        kernel_f: Tuple = (1, 5, 1, 5, 1),
        stride_f: Tuple = (1, 4, 1, 4, 1),
        dilation_f: Tuple = (1, 1, 1, 1, 1),
        delay: Tuple = (0, 0, 1, 0, 0),
        tcn_layer: str = "normal",
        tcn_kernel: int = 3,
        tcn_dim: int = 256,
        tcn_dilated_basic: int = 2,
        per_tcn_stack: int = 5,
        repeat_tcn: int = 4,
        tcn_with_embed: List = [1, 0, 0, 0, 0],
        tcn_use_film: bool = False,
        tcn_norm: str = "gLN",
        dconv_norm: str = "gGN",
        causal: bool = False,
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

        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.tcn_layer = tcn_layer
        self.tcn_dim = tcn_dim
        self.tcn_kernel = tcn_kernel
        self.per_tcn_stack = per_tcn_stack
        self.repeat_tcn = repeat_tcn
        self.tcn_dilated_basic = tcn_dilated_basic
        self.tcn_with_embed = tcn_with_embed
        self.tcn_norm = tcn_norm
        self.dconv_norm = dconv_norm
        self.tcn_use_film = tcn_use_film
        self.causal = causal
        self.transpose_delay = transpose_delay

        # TCN module's
        temporal_input_dim = self.num_freq
        for stride, _ in self.stride:
            if temporal_input_dim % stride == 0:
                temporal_input_dim //= stride
            else:
                temporal_input_dim //= stride
                temporal_input_dim += 1

        temporal_input_dim *= self.channels[-1]  # extend by channel size

        if self.tcn_layer.lower() == "normal":
            tcn_cls = TCN
        elif self.tcn_layer.lower() == "gated":
            print("GatedTCN would ignore dconv_norm configuration.")
            tcn_cls = GatedTCN
        else:
            raise NameError

        assert per_tcn_stack == len(tcn_with_embed)
        self.tcn_list = nn.ModuleList()
        for _ in range(repeat_tcn):
            _tcn = []

            for i in range(per_tcn_stack):
                if tcn_with_embed[i]:
                    if self.tcn_layer.lower() == "normal":
                        _tcn.append(
                            tcn_cls(
                                temporal_input_dim,
                                tcn_dim,
                                kernel=tcn_kernel,
                                dilation=tcn_dilated_basic ** i,
                                emb_dim=embed_dim,
                                causal=causal,
                                tcn_norm=tcn_norm,
                                dconv_norm=dconv_norm,
                            )
                        )
                    else:
                        _tcn.append(
                            tcn_cls(
                                temporal_input_dim,
                                tcn_dim,
                                kernel=tcn_kernel,
                                dilation=tcn_dilated_basic ** i,
                                emb_dim=embed_dim,
                                causal=causal,
                                tcn_norm=tcn_norm,
                                use_film=tcn_use_film,
                            )
                        )
                else:
                    if self.tcn_layer.lower() == "normal":
                        _tcn.append(
                            tcn_cls(
                                temporal_input_dim,
                                tcn_dim,
                                kernel=tcn_kernel,
                                dilation=tcn_dilated_basic ** i,
                                emb_dim=0,
                                causal=causal,
                                tcn_norm=tcn_norm,
                                dconv_norm=dconv_norm,
                            )
                        )
                    else:
                        _tcn.append(
                            tcn_cls(
                                temporal_input_dim,
                                tcn_dim,
                                kernel=tcn_kernel,
                                dilation=tcn_dilated_basic ** i,
                                emb_dim=0,
                                causal=causal,
                                tcn_norm=tcn_norm,
                                use_film=False,
                            )
                        )

            self.tcn_list.append(nn.ModuleList(_tcn))

    def forward(
        self, x: torch.Tensor, dvec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, CH, C, T]
            dvec: conditional tensor shape [N, C]

        Returns:
            output tensor has shape [N, CH, C, T]
        """
        # normalize
        if self.embed_norm and dvec is not None:
            dvec = F.normalize(dvec, p=2, dim=1)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, 1, C, T]

        skip = [x.clone()]

        # forward CNN-down layers
        for cnn_layer in self.cnn_down:
            x = cnn_layer(x)  # [N, ch, C, T]
            skip.append(x)

        # forward TCN block
        N_ori, ch, C_ori, T = x.shape
        x = x.reshape(N_ori, ch * C_ori, T)

        for r in range(self.repeat_tcn):
            for i in range(len(self.tcn_list[r])):
                if self.tcn_with_embed[i]:
                    x = self.tcn_list[r][i](x, dvec)
                else:
                    x = self.tcn_list[r][i](x)

        x = x.reshape(N_ori, ch, C_ori, T)

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

    @property
    def get_args(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "activation_type": self.activation_type,
            "norm_type": self.norm_type,
            "dropout": self.dropout,
            "channels": self.channels,
            "transpose_t_size": self.transpose_t_size,
            "transpose_delay": self.transpose_delay,
            "skip_conv": self.skip_conv,
            "kernel_t": self.kernel_t,
            "stride_t": self.stride_t,
            "dilation_t": self.dilation_t,
            "kernel_f": self.kernel_f,
            "stride_f": self.stride_f,
            "dilation_f": self.dilation_f,
            "delay": self.delay,
            "embed_dim": self.embed_dim,
            "embed_norm": self.embed_norm,
            "tcn_norm": self.tcn_norm,
            "dconv_norm": self.dconv_norm,
            "tcn_layer": self.tcn_layer,
            "tcn_dim": self.tcn_dim,
            "tcn_kernel": self.tcn_kernel,
            "tcn_dilated_basic": self.tcn_dilated_basic,
            "repeat_tcn": self.repeat_tcn,
            "per_tcn_stack": self.per_tcn_stack,
            "tcn_with_embed": self.tcn_with_embed,
            "tcn_use_film": self.tcn_use_film,
            "causal": self.causal,
        }


class UnetFsmn(Unet):
    """
    Improve temporal modeling ability by inserting a FSMN inside an Unet model.

    Args:
        embed_dim: Embedding feature dimension.
        embed_norm: If True, applies the 2-norm on the input embedding.
    """

    def __init__(
        self,
        embed_dim: int = 0,
        embed_norm: bool = False,
        input_dim: int = 512,
        activation_type: str = "PReLU",
        norm_type: str = "bN2d",
        dropout: float = 0.05,
        channels: Tuple = (1, 1, 8, 8, 16, 16),
        transpose_t_size: int = 2,
        transpose_delay: bool = False,
        skip_conv: bool = False,
        kernel_t: Tuple = (5, 1, 9, 1, 1),
        stride_t: Tuple = (1, 1, 1, 1, 1),
        dilation_t: Tuple = (1, 1, 1, 1, 1),
        kernel_f: Tuple = (1, 5, 1, 5, 1),
        stride_f: Tuple = (1, 4, 1, 4, 1),
        dilation_f: Tuple = (1, 1, 1, 1, 1),
        delay: Tuple = (0, 0, 1, 0, 0),
        fsmn_l_context: int = 3,
        fsmn_r_context: int = 0,
        fsmn_dim: int = 256,
        num_fsmn: int = 8,
        fsmn_with_embed: List = [1, 1, 1, 1, 1, 1, 1, 1],
        fsmn_norm: str = "gLN",
        use_film: bool = True,
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
        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.fsmn_l_context = fsmn_l_context
        self.fsmn_r_context = fsmn_r_context
        self.fsmn_dim = fsmn_dim
        self.num_fsmn = num_fsmn
        self.fsmn_with_embed = fsmn_with_embed
        self.fsmn_norm = fsmn_norm
        self.use_film = use_film

        # FSMN module's
        temporal_input_dim = self.num_freq
        for stride, _ in self.stride:
            if temporal_input_dim % stride == 0:
                temporal_input_dim //= stride
            else:
                temporal_input_dim //= stride
                temporal_input_dim += 1

        temporal_input_dim *= self.channels[-1]  # extend by channel size

        assert num_fsmn == len(fsmn_with_embed)
        self.fsmn_list = nn.ModuleList()

        for i in range(num_fsmn):
            if fsmn_with_embed[i]:
                self.fsmn_list.append(
                    ConditionFSMN(
                        temporal_input_dim,
                        temporal_input_dim,
                        fsmn_dim,
                        embed_dim,
                        fsmn_l_context,
                        fsmn_r_context,
                        norm_type=fsmn_norm,
                        use_film=use_film,
                    )
                )

            else:
                self.fsmn_list.append(
                    FSMN(
                        temporal_input_dim,
                        temporal_input_dim,
                        fsmn_dim,
                        fsmn_l_context,
                        fsmn_r_context,
                        norm_type=fsmn_norm,
                    )
                )

    def forward(
        self, x: torch.Tensor, dvec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor shape [N, CH, C, T]
            dvec: conditional tensor shape [N, C]

        Returns:
            output tensor has shape [N, CH, C, T]
        """
        # normalize
        if self.embed_norm and dvec is not None:
            dvec = F.normalize(dvec, p=2, dim=1)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # [N, 1, C, T]

        skip = [x.clone()]

        # forward CNN-down layers
        for cnn_layer in self.cnn_down:
            x = cnn_layer(x)  # [N, ch, C, T]
            skip.append(x)

        # forward FSMN block
        N_ori, ch, C_ori, T = x.shape
        x = x.reshape(N_ori, ch * C_ori, T)
        memory = None
        for i in range(len(self.fsmn_list)):
            if self.fsmn_with_embed[i]:
                x, memory = self.fsmn_list[i](x, dvec, memory)
            else:
                x, memory = self.fsmn_list[i](x, memory)

        x = x.reshape(N_ori, ch, C_ori, T)

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

    @property
    def get_args(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "activation_type": self.activation_type,
            "norm_type": self.norm_type,
            "dropout": self.dropout,
            "channels": self.channels,
            "transpose_t_size": self.transpose_t_size,
            "transpose_delay": self.transpose_delay,
            "skip_conv": self.skip_conv,
            "kernel_t": self.kernel_t,
            "stride_t": self.stride_t,
            "dilation_t": self.dilation_t,
            "kernel_f": self.kernel_f,
            "stride_f": self.stride_f,
            "dilation_f": self.dilation_f,
            "delay": self.delay,
            "embed_dim": self.embed_dim,
            "embed_norm": self.embed_norm,
            "fsmn_l_context": self.fsmn_l_context,
            "fsmn_r_context": self.fsmn_r_context,
            "fsmn_dim": self.fsmn_dim,
            "num_fsmn": self.num_fsmn,
            "fsmn_with_embed": self.fsmn_with_embed,
            "fsmn_norm": self.fsmn_norm,
            "use_film": self.use_film,
        }
