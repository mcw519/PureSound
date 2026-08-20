"""Auxiliary prediction heads that read a backbone's bottleneck features.

These are backbone-agnostic building blocks: any model whose bottleneck is a
``[N, C, F, T]`` feature map can attach them. Checkpoint keys are determined by
the attribute name the backbone stores the head under (e.g.
``backbone.vad_head.*``), not by this module's path, so relocating or reusing a
head never invalidates a checkpoint.

Each head owns its config model and its own `from_config`, so the block's keys,
its defaults and the enabled gate all live in one place. That gate used to be
written out at the attach site against a raw dict, and `.get(key, default)`
answers a misspelled key with the default: `vad_head: {enabled: true, hiden: 64}`
built a head at the wrong width and trained a whole run that way without a word.
The models are `extra="forbid"`, which is the same reason the augmentation
blocks are.
"""

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pydantic import Field, field_validator

from puresound.config.base import StrictConfig


class VADHeadConfig(StrictConfig):
    """``backbone_args.vad_head`` (and ``background_vad_head``). ``hidden``
    defaults to the bottleneck width, which only the backbone knows, so it is
    None here rather than duplicated.

    ``ema_taus_s`` widens the head's evidence window: the presence decision was
    measured to need ~1 s of audible speech while the plain head sees
    ``kernel_t`` frames (50 ms). A bank of fixed exponential averages lets the
    head read several time scales at once -- and, more to the point, the RATIO
    between a fast and a slow average is envelope modulation depth, the DRR-fill
    cue that separates near from far where per-frame DRR cannot. Decay rates are
    fixed, not learned: a learned rate can drift to 0 (single-frame again) or 1
    (a constant), and a fixed one keeps the time scale an explicit, reportable
    hyperparameter. None (the default) builds exactly the pre-EMA architecture,
    so existing gate checkpoints load bit-identically.
    """

    enabled: bool = False
    hidden: Optional[int] = Field(default=None, gt=0)
    kernel_t: int = Field(default=5, gt=0)
    ema_taus_s: Optional[tuple[float, ...]] = None
    #: Bottleneck frames per second: sample_rate / encoder hop (16000/160).
    frame_rate: float = Field(default=100.0, gt=0.0)

    @field_validator("ema_taus_s", mode="before")
    @classmethod
    def _taus_positive(cls, v):
        # YAML hands over a list; strict mode will not coerce it to the tuple
        # annotation on its own.
        if isinstance(v, list):
            v = tuple(v)
        if v is not None:
            if len(v) == 0:
                raise ValueError("ema_taus_s: give at least one tau or omit the key")
            if any(t <= 0.0 for t in v):
                raise ValueError(f"ema_taus_s must all be > 0 seconds, got {v}")
        return v


class DistHeadConfig(StrictConfig):
    """``backbone_args.dist_head``."""

    enabled: bool = False
    hidden: int = Field(default=128, gt=0)
    n_out: int = Field(default=3, gt=0)


class VADHead(nn.Module):
    """Frame-level speech-activity logits from the bottleneck. Causal in time."""

    @classmethod
    def from_config(
        cls, config: Optional[Mapping[str, Any]], *, enc_channels: int
    ) -> Optional["VADHead"]:
        """Build from a recipe block, or None when it is absent or disabled."""
        parsed = VADHeadConfig.model_validate(dict(config or {}))
        if not parsed.enabled:
            return None
        return cls(
            enc_channels=enc_channels,
            hidden=enc_channels if parsed.hidden is None else parsed.hidden,
            kernel_t=parsed.kernel_t,
            ema_taus_s=parsed.ema_taus_s,
            frame_rate=parsed.frame_rate,
        )

    def __init__(
        self,
        enc_channels: int,
        hidden: int,
        kernel_t: int,
        ema_taus_s: Optional[Sequence[float]] = None,
        frame_rate: float = 100.0,
    ):
        super().__init__()
        self.kernel_t = kernel_t
        self.ema_taus_s = tuple(float(t) for t in ema_taus_s) if ema_taus_s else ()
        self.frame_rate = float(frame_rate)
        # One decay per tau; alpha converts "seconds" into a per-frame step the
        # same way everywhere in this repo: a = 1 - exp(-1/(tau * fps)).
        self._alphas = [
            1.0 - float(torch.exp(torch.tensor(-1.0 / (t * self.frame_rate))))
            for t in self.ema_taus_s
        ]
        in_features = enc_channels * (1 + len(self.ema_taus_s))
        self.proj = nn.Linear(in_features, hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_t, padding=0, groups=1)
        self.act = nn.SiLU()
        self.out = nn.Conv1d(hidden, 1, 1)

    def _ema_bank(self, h: torch.Tensor) -> torch.Tensor:
        """Debiased exponential averages of the pooled bottleneck, one per tau.

        The recurrence runs in float32 whatever the autocast dtype: with a 4 s
        tau at 100 fps the step is a = 0.0025, and accumulating x*0.0025 into a
        bf16 state loses the increment entirely -- the slow averages would
        silently freeze.

        ``lfilter`` hard-clips to [-1, 1] BY DEFAULT and bottleneck features are
        not bounded by 1; ``clamp=False`` is load-bearing here, the same silent
        saturation this repo dug out of six device-chain stages (9c56e02).

        Debiasing: with zero initial state the EMA underestimates until it has
        seen ~tau of input. The normalizer is closed-form
        ``1 - (1-a)^(t+1)``, so dividing by it makes every average an average
        *of what has been seen so far* from the first frame -- no warm-up
        transient for the head to learn around, and the same semantics a
        streaming implementation gets by carrying (state, normalizer).
        """
        n, c, t = h.shape
        # autocast(enabled=False), not just .float(): under bf16-mixed training
        # autocast re-casts lfilter's internals back to bf16, which both loses
        # the small-step increments AND trips the CUDA kernel's fp32/fp64
        # assertion. Disabling the context makes the .float() actually stick.
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = h.float()
            steps = torch.arange(1, t + 1, device=h.device, dtype=torch.float32)
            out = []
            for a in self._alphas:
                y = torchaudio.functional.lfilter(
                    x,
                    a_coeffs=x.new_tensor([1.0, -(1.0 - a)]),
                    b_coeffs=x.new_tensor([a, 0.0]),
                    clamp=False,
                )
                norm = 1.0 - (1.0 - a) ** steps  # [T], debias for the zero init
                out.append((y / norm).to(h.dtype))
        return torch.cat([h] + out, dim=1)  # [N, (K+1)C, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, F, T] -> pool over F -> [N, C, T]
        h = x.mean(dim=2)  # [N, C, T]
        if self._alphas:
            h = self._ema_bank(h)  # [N, (K+1)C, T]
        h = self.proj(h.transpose(1, 2)).transpose(1, 2)  # [N, hidden, T]
        h = F.pad(h, (self.kernel_t - 1, 0))  # causal
        h = self.act(self.dwconv(h))
        return self.out(h).squeeze(1)  # [N, T]

    # ---------------------------------------------------------------- streaming

    def initial_stream_state(self, batch_size: int = 1, device="cpu", dtype=None):
        """(ema_state, conv_cache, count) matching what `forward` starts from.

        ``ema_state`` is [N, K, C] of exponential averages, ``conv_cache`` the
        causal dwconv's [N, hidden, kernel_t-1] left context, ``count`` the frames
        seen so far (the debias normalizer needs it, and a wrong count is a
        silent level error rather than a crash).
        """
        dtype = dtype or next(self.parameters()).dtype
        c = self.proj.in_features // (1 + len(self._alphas))
        return (
            torch.zeros(batch_size, len(self._alphas), c, dtype=torch.float32, device=device),
            torch.zeros(batch_size, self.dwconv.in_channels, self.kernel_t - 1,
                        dtype=dtype, device=device),
            torch.zeros(batch_size, 1, dtype=torch.float32, device=device),
        )

    def step(self, x: torch.Tensor, state):
        """One frame. ``x`` is [N, C, F, 1]; returns (logit [N, 1], new_state).

        Mirrors `forward` exactly: the EMA recurrence in float32, the same
        closed-form debias, and the dwconv fed from a left-context cache instead
        of zero padding.
        """
        ema, cache, count = state
        # Match the parameters, not the caller: an exported graph may hand over
        # bf16 activations while the weights stay float32, and Linear refuses
        # the mix. The EMA recurrence below is float32 regardless.
        h = x.mean(dim=2).to(self.proj.weight.dtype)         # [N, C, 1]
        if self._alphas:
            cur = h[..., 0].float()                          # [N, C]
            n = count + 1.0
            new_ema, cols = [], []
            for k, a in enumerate(self._alphas):
                s_k = ema[:, k] + a * (cur - ema[:, k])
                new_ema.append(s_k)
                # 1 - (1-a)^n, the same normalizer forward divides by
                cols.append(s_k / (1.0 - (1.0 - a) ** n))
            ema = torch.stack(new_ema, dim=1)
            count = n
            h = torch.cat([h] + [c.unsqueeze(-1).to(h.dtype) for c in cols], dim=1)
        h = self.proj(h.transpose(1, 2)).transpose(1, 2)     # [N, hidden, 1]
        window = torch.cat([cache.to(h.dtype), h], dim=-1)   # [N, hidden, kernel_t]
        logit = self.out(self.act(self.dwconv(window))).squeeze(1)   # [N, 1]
        return logit, (ema, window[..., 1:], count)


class DistHead(nn.Module):
    """Utterance-level distance/DRR regression from the bottleneck.

    Auxiliary multi-task pressure that makes the bottleneck encode the physical
    proximity cues (DRR / source distance) rather than the capture-chain
    signature of the training near-field rows. Predicts
    ``[fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]``;
    supervision comes from the dataset's free scalar labels and is NaN-masked
    (see DistHeadRegressionLoss). Training-only: inference never reads it and
    the streaming export is untouched."""

    @classmethod
    def from_config(
        cls, config: Optional[Mapping[str, Any]], *, enc_channels: int
    ) -> Optional["DistHead"]:
        """Build from a recipe block, or None when it is absent or disabled."""
        parsed = DistHeadConfig.model_validate(dict(config or {}))
        if not parsed.enabled:
            return None
        return cls(enc_channels=enc_channels, hidden=parsed.hidden, n_out=parsed.n_out)

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
