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

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from puresound.config.base import StrictConfig


class VADHeadConfig(StrictConfig):
    """``backbone_args.vad_head``. ``hidden`` defaults to the bottleneck width,
    which only the backbone knows, so it is None here rather than duplicated."""

    enabled: bool = False
    hidden: Optional[int] = Field(default=None, gt=0)
    kernel_t: int = Field(default=5, gt=0)


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
        )

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
