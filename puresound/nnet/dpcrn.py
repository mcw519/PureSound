from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .lobe.attention import MhaSelfAttenLayer
from .lobe.banding import BandBottleneck
from .lobe.heads import DistHead, IdentityHead, ProximityHead, VADHead
from .lobe.multiframe import DeepFilterResidualHead
from .lobe.rnn import SingleRNN
from .lobe.ssm import MambaInter
from .lobe.trivial import FiLM, spectral_compression
from .unet import Unet


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
        inter_type: str = "lstm",
        mamba_args: Optional[dict] = None,
        context_bottleneck: Optional[Dict] = None,
        context_freqs: Optional[int] = None,
        embedding_size: Optional[int] = None,
        fused_type: Optional[str] = None,
        intra_type: str = "lstm",
        intra_nhead: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.intra_type = intra_type
        self.fused_type = fused_type.lower() if fused_type else None
        self.context_bottleneck = None

        if inter_type == "mamba_context":
            if context_bottleneck is None or context_freqs is None:
                raise ValueError(
                    "inter_type='mamba_context' requires context_bottleneck "
                    "and context_freqs"
                )
            self.context_bottleneck = BandBottleneck(
                n_units=context_freqs, **context_bottleneck
            )

        if embedding_size is not None:
            if self.fused_type == "film":
                self.film = FiLM(
                    feats_size=input_size,
                    embed_size=embedding_size,
                    input_norm=True,
                )
            else:
                raise ValueError(
                    f"unknown fused_type {self.fused_type!r}; only 'film' is "
                    "implemented for embedding conditioning."
                )

        # intra(-frequency) path. "lstm" is the shipped default: a bidirectional
        # LSTM that walks the C frequency positions one step at a time, twice.
        # "attention" is DPARN's intra path: two self-attention layers over the
        # same positions in one matmul each. Frequency has no causal constraint,
        # so there is nothing an ordered walk buys here that attention cannot
        # see -- and the walk is 64 sequential steps per frame, which is what a
        # single-core streaming budget pays for.
        if intra_type == "lstm":
            self.intra_rnn = SingleRNN(
                "LSTM", input_size, hidden_size, bidirectional=True, dropout=dropout
            )
        elif intra_type == "attention":
            self.intra_atten1 = MhaSelfAttenLayer(
                input_size, hidden_size, nhead=intra_nhead, dropout=dropout,
                improved=False, bidirectional=False, position_encoding=True,
            )
            self.intra_atten2 = MhaSelfAttenLayer(
                input_size, hidden_size, nhead=intra_nhead, dropout=dropout,
                improved=False, bidirectional=False, position_encoding=False,
            )
            self.intra_fc = nn.Linear(input_size, input_size)
        else:
            raise ValueError(
                f"intra_type must be 'lstm' or 'attention', got {intra_type!r}"
            )
        self.intra_norm = nn.LayerNorm(input_size)

        # inter(-time) path. "lstm" is the shipped default. "mamba" REPLACES it
        # with a selective-state-space block at matched parameter budget -- but
        # a replacement re-initialises the network's only context carrier, and
        # losing the trained context carrier costs more than the change of
        # operator can win back. "lstm+mamba" keeps the
        # trained LSTM and adds the SSM as a parallel branch whose output
        # projection starts at zero: the sum equals the LSTM alone at step 0, so
        # a warm start carries no re-initialisation debt and the SSM can only
        # earn its way in. See lobe/ssm.py.
        self.inter_ssm = None
        if inter_type in ("lstm", "lstm+mamba"):
            self.inter_rnn = SingleRNN(
                "LSTM", input_size, hidden_size, bidirectional=False, dropout=dropout
            )
            if inter_type == "lstm+mamba":
                self.inter_ssm = MambaInter(
                    d_model=input_size, dropout=dropout, zero_init_out=True,
                    **(mamba_args or {})
                )
        elif inter_type in ("mamba", "mamba_context"):
            self.inter_rnn = MambaInter(
                d_model=input_size, dropout=dropout, **(mamba_args or {})
            )
        else:
            raise ValueError(
                "inter_type must be 'lstm', 'mamba', 'mamba_context' or "
                "'lstm+mamba', "
                f"got {inter_type!r}"
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
        if hasattr(self, "intra_rnn"):             # attention has no cuDNN params
            self.intra_rnn.rnn.flatten_parameters()
        if hasattr(self.inter_rnn, "rnn"):          # MambaInter has no cuDNN params
            self.inter_rnn.rnn.flatten_parameters()

        x_intra_skip = x.clone()
        N, CH, C, T = x.shape

        # intra-chunk, time independent and frequency dependent
        x = x.transpose(1, -1).reshape(
            N * T, C, CH
        )  # [N, CH, C, T] -> [N, T, C, CH] -> [N*T, C, CH]
        if self.intra_type == "attention":
            # Both layers keep the [N*T, CH, C] layout DPARN uses; the fc acts on CH.
            x = self.intra_atten1(x.permute(0, 2, 1), causal=False)  # [N*T, CH, C]
            x = self.intra_atten2(x, causal=False)
            x = self.intra_fc(x.permute(0, 2, 1))  # [N*T, C, CH]
        else:
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

        # The context variant keeps the local/intra path on the full encoder
        # grid. Only the expensive temporal model sees the reduced band grid;
        # its output is expanded as a residual, so per-bin detail bypasses the
        # lossy projection instead of having to survive a band round trip.
        context = self.context_bottleneck
        if context is not None:
            x = context.to_bands(x)

        # inter-chunk, time dependent and frequency independent
        context_freq = x.shape[2]
        x = x.permute(0, 2, 3, 1).reshape(
            N * context_freq, T, -1
        )  # [N, CH, C, T] -> [N, C, T, CH] -> [N*C, T, CH]
        x_inter = x.permute(0, 2, 1)  # [N*C, T, CH] -> [N*C, CH, T]
        x = self.inter_rnn(x_inter)
        if self.inter_ssm is not None:
            x = x + self.inter_ssm(x_inter)
        x = x.permute(0, 2, 1)  # [N*C, CH, T] -> [N*C, T, CH]
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(N, context_freq, CH, T)
        x = x.permute(0, 2, 1, 3)

        if context is not None:
            x = context.to_units(x)

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
        inter_type: str = "lstm",
        mamba_args: Optional[dict] = None,
        band_bottleneck: Optional[Dict] = None,
        intra_type: str = "lstm",
        intra_nhead: int = 4,
        mamba_context: Optional[Dict] = None,
        spectral_compress: bool = False,
        vad_head: Optional[Dict] = None,
        background_vad_head: Optional[Dict] = None,
        dist_head: Optional[Dict] = None,
        identity_head: Optional[Dict] = None,
        proximity_head: Optional[Dict] = None,
        expose_bottleneck: bool = False,
        df_head: Optional[Dict] = None,
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
        self._args = {
            "input_dim": input_dim,
            "dvec_dim": dvec_dim,
            "activation_type": activation_type,
            "norm_type": norm_type,
            "dropout": dropout,
            "channels": channels,
            "transpose_t_size": transpose_t_size,
            "transpose_delay": transpose_delay,
            "skip_conv": skip_conv,
            "kernel_t": kernel_t,
            "stride_t": stride_t,
            "dilation_t": dilation_t,
            "kernel_f": kernel_f,
            "stride_f": stride_f,
            "dilation_f": dilation_f,
            "delay": delay,
            "rnn_hidden": rnn_hidden,
            "inter_type": inter_type,
            "mamba_args": mamba_args,
            "band_bottleneck": band_bottleneck,
            "intra_type": intra_type,
            "intra_nhead": intra_nhead,
            "mamba_context": mamba_context,
            "spectral_compress": spectral_compress,
            "vad_head": vad_head,
            "background_vad_head": background_vad_head,
            "dist_head": dist_head,
            "identity_head": identity_head,
            "proximity_head": proximity_head,
            "expose_bottleneck": expose_bottleneck,
            "df_head": df_head,
        }

        self.transpose_delay = transpose_delay
        self.rnn_hidden = rnn_hidden

        # Optional perceptual banding around the recurrent blocks. The U-net, the
        # mask and every head keep seeing the strided grid; only the DPRNN sees
        # bands. Pair it with stride_f=[1,1,1] to band straight off the full
        # resolution -- banding after a uniform stride cannot recover what the
        # stride already threw away, it can only avoid throwing away more.
        self.band_bottleneck = None
        if band_bottleneck and mamba_context:
            raise ValueError(
                "band_bottleneck and mamba_context are mutually exclusive: "
                "the context path must start from the full encoder grid"
            )
        bottleneck_freqs = self.shape_info()[0][-1]
        if band_bottleneck:
            self.band_bottleneck = BandBottleneck(
                n_units=bottleneck_freqs, **band_bottleneck
            )
        if mamba_context and inter_type != "mamba_context":
            raise ValueError(
                "mamba_context requires inter_type='mamba_context'"
            )
        self.spectral_compress = spectral_compress
        self.dvec_dim = dvec_dim
        self.vad_head_args = vad_head

        # DPRNN block
        self.dprnn_block1 = DPRNNblock2D(
            input_size=channels[-1],
            inter_type=inter_type,
            mamba_args=mamba_args,
            context_bottleneck=mamba_context,
            context_freqs=bottleneck_freqs,
            intra_type=intra_type,
            intra_nhead=intra_nhead,
            hidden_size=rnn_hidden,
            dropout=dropout,
            embedding_size=dvec_dim,
            fused_type="FiLM",
        )
        self.dprnn_block2 = DPRNNblock2D(
            input_size=channels[-1],
            inter_type=inter_type,
            mamba_args=mamba_args,
            context_bottleneck=mamba_context,
            context_freqs=bottleneck_freqs,
            intra_type=intra_type,
            intra_nhead=intra_nhead,
            hidden_size=rnn_hidden,
            dropout=dropout,
            embedding_size=dvec_dim,
            fused_type="FiLM",
        )

        # Attribute names are the checkpoint keys (`backbone.vad_head.*`), so
        # they are load-bearing; what each head is built from lives with the
        # head, in `nnet/lobe/heads.py`.
        self.vad_head = VADHead.from_config(vad_head, enc_channels=channels[-1])
        self.last_vad_logits: Optional[torch.Tensor] = None

        # Companion head: "is NON-target speech present now". Same block shape,
        # separate weights; supervised by BackgroundVADHeadBCELoss against the
        # dataset's background_vad_reference.
        self.background_vad_head = VADHead.from_config(
            background_vad_head, enc_channels=channels[-1]
        )
        self.last_background_vad_logits: Optional[torch.Tensor] = None

        self.dist_head = DistHead.from_config(dist_head, enc_channels=channels[-1])
        self.last_dist_preds: Optional[torch.Tensor] = None

        # Per-frame identity / relative-proximity readouts. Same
        # training-only contract as the heads above: enabled from
        # `backbone_args`, never read by inference or by the streaming export,
        # and absent from the module (so from the checkpoint) when disabled.
        self.identity_head = IdentityHead.from_config(
            identity_head, enc_channels=channels[-1]
        )
        self.last_identity_emb: Optional[torch.Tensor] = None
        self.proximity_head = ProximityHead.from_config(
            proximity_head, enc_channels=channels[-1]
        )
        self.last_proximity: Optional[torch.Tensor] = None

        # The bottleneck itself, for readouts that are not modules -- the
        # inference-only presence gate fits a linear probe on exactly this
        # tensor. Off by default: a reference here would keep the graph alive
        # for the whole step, and training has no use for it.
        self.stash_bottleneck: bool = False
        self.last_bottleneck: Optional[torch.Tensor] = None

        # ...and the same tensor WITH its graph, frequency-pooled, for losses
        # that have to run a module of their own over the bottleneck -- the
        # identity loss's EMA teacher is a second copy of the head, so it needs
        # the features, not the head's output. Two separate switches on purpose:
        # `stash_bottleneck` is an inference flag flipped per call by
        # `EncDecMaskBase.forward` and its tensor is detached, and handing a
        # gradient path to something that expected a detached probe input would
        # silently change what it trains. This one is a build-time flag (`backbone_args.expose_bottleneck: true`), so what a
        # recipe trains is visible in the recipe.
        self.expose_bottleneck: bool = bool(expose_bottleneck)
        self.last_bottleneck_graph: Optional[torch.Tensor] = None

        # Deep-filtering residual on the low band (`df_head: {bins, order,
        # hidden}`). Unlike the heads above this one changes the OUTPUT: the
        # system adds `last_df_coefs` as a K-tap filter over the noisy spectrum
        # to the masked one (`Masker.apply_complex_mask_with_df`). It taps the
        # decoder's second-to-last map -- `channels[1]` wide, one first-stride
        # coarser than the mask -- and its last layer starts at zero, so a
        # checkpoint without it loads into a model that computes the same thing.
        self.df_head = None
        if df_head:
            if len(self.cnn_up) < 2:
                raise ValueError("df_head taps the decoder's second-to-last layer; need >= 2 up layers")
            upsample = int(stride_f[0])
            bins = int(df_head.get("bins", 128))
            if bins > input_dim:
                raise ValueError(f"df_head.bins ({bins}) exceeds the {input_dim} mask bins")
            self.df_head = DeepFilterResidualHead(
                in_channels=int(self.channels[1]),
                upsample=upsample,
                bins=bins,
                order=int(df_head.get("order", 5)),
                hidden=int(df_head.get("hidden", 32)),
            )
        self.last_df_coefs: Optional[torch.Tensor] = None

    @property
    def get_args(self) -> Dict:
        """The constructor arguments: ``DPCRN(**model.get_args)`` rebuilds it."""
        return dict(self._args)

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

        # forward dprnn, on perceptual bands when one is configured
        if self.band_bottleneck is not None:
            x = self.band_bottleneck.to_bands(x)
        x = self.dprnn_block1(x, embed=dvec)  # [N, ch, C, T]
        x = self.dprnn_block2(x, embed=dvec)  # [N, ch, C, T]
        if self.band_bottleneck is not None:
            # Back before the heads, so they and the decoder are unchanged.
            x = self.band_bottleneck.to_units(x)

        if self.vad_head is not None:
            self.last_vad_logits = self.vad_head(x)
        else:
            self.last_vad_logits = None

        if self.background_vad_head is not None:
            self.last_background_vad_logits = self.background_vad_head(x)
        else:
            self.last_background_vad_logits = None

        if self.dist_head is not None:
            self.last_dist_preds = self.dist_head(x)
        else:
            self.last_dist_preds = None

        if self.identity_head is not None:
            self.last_identity_emb = self.identity_head(x)  # [N, T, D]
        else:
            self.last_identity_emb = None

        if self.proximity_head is not None:
            self.last_proximity = self.proximity_head(x)  # [N, T]
        else:
            self.last_proximity = None

        self.last_bottleneck = x.detach() if self.stash_bottleneck else None
        # Frequency-pooled and NOT detached: `[N, C, T]`, the same convention
        # every per-frame head pools to internally.
        self.last_bottleneck_graph = x.mean(dim=2) if self.expose_bottleneck else None

        # forward CNN-up layers
        self.last_df_coefs = None
        for i, cnn_layer in enumerate(self.cnn_up):
            if self.skip_conv:
                x = x + self.skip_cnn[i](skip[-i - 1])
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
            if self.df_head is not None and i == len(self.cnn_up) - 2:
                self.last_df_coefs = self.df_head(x)

        return x
