from typing import Dict, Optional

import torch
import torch.nn as nn


class DistHeadRegressionLoss(nn.Module):
    """NaN-masked regression on the DPCRN DistHead's utterance-level outputs.

    Auxiliary multi-task supervision: the bottleneck is regressed against the
    physical proximity labels the dataset already emits for free, pushing the
    near/far decision toward distance/DRR cues rather than the capture-chain
    signature of the training near-field rows.

    Targets, in the head's output order:
        0. ``foreground_drr`` / drr_scale        (synthetic rows only)
        1. ``log10(foreground_distance)``        (synthetic + real-near rows)
        2. ``log10(nearest_interferer_distance)``(synthetic + real-far rows)

    Every label can be NaN for any given row (e.g. real recordings carry a
    distance but no DRR; no-interferer rows carry no interferer distance);
    masking is entrywise, and a batch with zero valid entries contributes a
    zero loss that still carries a graph edge so DDP never sees an unused head.
    """

    uses_dist_preds = True

    def __init__(
        self,
        drr_scale: float = 10.0,
        min_dist: float = 0.1,
        max_dist: float = 30.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.drr_scale = float(drr_scale)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)
        self.beta = float(beta)

    def _log_dist(self, d: torch.Tensor) -> torch.Tensor:
        return torch.log10(d.clamp(self.min_dist, self.max_dist))

    def forward(self, dist_preds: Optional[torch.Tensor], batch: Dict) -> torch.Tensor:
        if dist_preds is None:
            raise ValueError(
                "DistHeadRegressionLoss is configured but the backbone produced no "
                "last_dist_preds -- enable model.backbone.backbone_args.dist_head."
            )
        n = dist_preds.shape[0]
        device = dist_preds.device
        nan = torch.full((n,), float("nan"), device=device)

        def _col(key: str) -> torch.Tensor:
            v = batch.get(key)
            if v is None:
                return nan
            return v.to(device).view(-1).float()

        target = torch.stack(
            [
                _col("foreground_drr") / self.drr_scale,
                self._log_dist(_col("foreground_distance")),
                self._log_dist(_col("nearest_interferer_distance")),
            ],
            dim=1,
        )  # [N, 3]

        valid = torch.isfinite(target)
        if not bool(valid.any()):
            return dist_preds.sum() * 0.0
        return nn.functional.smooth_l1_loss(
            dist_preds[valid], target[valid], beta=self.beta
        )
