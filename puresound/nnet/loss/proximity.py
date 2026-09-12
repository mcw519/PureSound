"""Relative proximity supervised by rendered turn distances, not speaker roles.

``turn_distance`` contains metres for each rendered turn, including user seat
changes. Unknown distances are NaN. Only sufficiently separated user/bystander
turns are compared; either role may be physically nearer. Missing distances
never fall back to the assumption that the user is nearer.

The baseline uses unbounded readouts and softplus(margin - signed gap).
``scale_free=True`` is an explicit bounded-readout ablation, never an implicit
change to an old recipe. Explicit second views are evaluated separately by
``paired_consistency``: no duplicated ordering or separation example. Matched
turn contrasts are compared individually so opposite errors cannot cancel.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity import align_turn_frames, eligible_turns, pool_turn_means


class RelativeProximityLoss(nn.Module):
    """Ordering in head units; eligibility in metres. Bookkeeping is detached."""

    required_inputs = ("proximity", "batch")
    paired_output = "proximity"

    def __init__(self, margin: float = 1.0, consistency_weight: float = 1.0,
                 min_turn_frames: int = 1, min_distance_gap_m: float = 0.25,
                 scale_free: bool = False, temperature: float = 1.0,
                 pair_selection: str = "cross_role"):
        super().__init__()
        for name, value in (("margin", margin), ("temperature", temperature),
                            ("min_distance_gap_m", min_distance_gap_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("consistency_weight must be finite and non-negative")
        if min_turn_frames < 1:
            raise ValueError("min_turn_frames must be positive")
        self.margin = float(margin)
        self.consistency_weight = float(consistency_weight)
        self.min_turn_frames = int(min_turn_frames)
        self.min_distance_gap_m = float(min_distance_gap_m)
        self.scale_free = bool(scale_free)
        self.temperature = float(temperature)
        if pair_selection not in ("cross_role", "all"):
            raise ValueError("pair_selection must be cross_role or all")
        self.pair_selection = pair_selection
        self.last_stats = {}

    @property
    def paired_weight(self):
        return self.consistency_weight

    def _pairs(self, proximity: torch.Tensor, batch: Dict):
        role, distance = batch.get("turn_role"), batch.get("turn_distance")
        if role is None or distance is None or batch.get("turn_id") is None:
            return None
        if role.shape[1] == 0:
            return None
        if distance.shape != role.shape:
            raise ValueError("turn_distance must have the same [B, K] shape as turn_role")
        values = proximity.float()
        if self.scale_free:
            values = values.tanh()
        aligned = align_turn_frames([values.unsqueeze(-1)], batch, proximity.device)
        if aligned is None:
            return None
        (frames,), turn_id, exclude = aligned
        means, counts = pool_turn_means(frames, turn_id, role.shape[1], exclude)
        means = means.squeeze(-1)
        role = role.to(proximity.device)
        distance = distance.to(device=proximity.device, dtype=torch.float32)
        eligible = eligible_turns(counts, batch, proximity.device, self.min_turn_frames)
        eligible = eligible & torch.isfinite(distance) & (distance > 0)
        delta = distance.unsqueeze(1) - distance.unsqueeze(2)  # d_right - d_left
        mask = eligible.unsqueeze(2) & eligible.unsqueeze(1)
        if self.pair_selection == "cross_role":
            mask = mask & (role.unsqueeze(2) != role.unsqueeze(1))
        # Count each unordered turn pair once; role numbers have no semantic
        # meaning here beyond padding=0 and optional cross-role selection.
        mask = mask & torch.ones_like(mask).triu(diagonal=1)
        mask = mask & (delta.abs() >= self.min_distance_gap_m)
        direction = torch.nan_to_num(delta).sign()
        gaps = (means.unsqueeze(2) - means.unsqueeze(1)) * direction
        return gaps, mask

    def forward(self, proximity: Optional[torch.Tensor], batch: Dict) -> torch.Tensor:
        if proximity is None:
            raise ValueError("RelativeProximityLoss requires an enabled proximity_head")
        zero = proximity.sum() * 0.0
        self.last_stats = {key: zero.detach() for key in (
            "ordering_pairs", "ordering_correct", "ordering_loss",
            "within_batch_pairs", "consistency_loss",
        )}
        pairs = self._pairs(proximity, batch)
        if pairs is None:
            return zero
        gaps, mask = pairs
        selected = gaps[mask]
        self.last_stats["ordering_pairs"] = mask.sum().detach()
        self.last_stats["ordering_correct"] = (selected > 0).sum().detach()
        ordering = zero
        if selected.numel():
            tau = self.temperature if self.scale_free else 1.0
            ordering = (tau * F.softplus((self.margin - selected) / tau)).mean()
        self.last_stats["ordering_loss"] = ordering.detach()

        # Legacy explicitly collated source twins. The new recipe disables
        # source-pool collisions and instead uses the additional paired view.
        terms = []
        sources, chains = batch.get("row_source_id"), batch.get("turn_chain")
        if sources is not None and chains is not None and self.consistency_weight:
            ids = sources.detach().cpu().tolist()
            chains = chains.to(proximity.device)
            for left in range(len(ids)):
                for right in range(left + 1, len(ids)):
                    if ids[left] < 0 or ids[left] != ids[right]:
                        continue
                    valid = mask[left] & mask[right]
                    involved = valid.any(0) | valid.any(1)
                    if not bool(valid.any()) or not bool((chains[left][involved] != chains[right][involved]).any()):
                        continue
                    terms.append((gaps[left][valid] - gaps[right][valid]).abs().mean())
        consistency = torch.stack(terms).mean() if terms else zero
        self.last_stats["within_batch_pairs"] = zero.detach().new_tensor(len(terms))
        self.last_stats["consistency_loss"] = consistency.detach()
        return ordering + self.consistency_weight * consistency

    def paired_consistency(self, proximity: torch.Tensor, second: torch.Tensor,
                           batch: Dict, view: Dict):
        """Return (unweighted consistency, effective view pairs, turn pairs).

        Inputs: original B-row readout, K-row auxiliary readout. Source indices
        establish exact provenance, without random collisions or stale teachers.
        """
        zero = (proximity.sum() + second.sum()) * 0.0
        indices = view["source_indices"].to(proximity.device).long()
        if indices.numel() == 0:
            return zero, 0, 0
        if "row_source_id" not in batch or "row_source_id" not in view:
            raise ValueError("paired proximity views require row_source_id provenance")
        if second.shape[0] != indices.numel():
            raise ValueError("paired readout and source_indices have different row counts")
        if bool(((indices < 0) | (indices >= proximity.shape[0])).any()):
            raise ValueError("paired source_indices are outside the primary batch")
        labels = {key: value.index_select(0, indices.to(value.device))
                  for key, value in batch.items()
                  if torch.is_tensor(value) and value.ndim and value.shape[0] == proximity.shape[0]}
        a = self._pairs(proximity.index_select(0, indices), labels)
        b = self._pairs(second, labels)
        if a is None or b is None:
            return zero, 0, 0
        gaps_a, mask_a = a
        gaps_b, mask_b = b
        terms, turn_pairs = [], 0
        for row in range(indices.numel()):
            source = labels["row_source_id"][row]
            if source < 0 or source != view["row_source_id"][row]:
                raise ValueError("paired view does not share the original row_source_id")
            valid = mask_a[row] & mask_b[row]
            involved = valid.any(0) | valid.any(1)
            if not bool(valid.any()):
                continue
            # The paired collate may have fewer padded turns than the primary.
            if "turn_chain" not in labels or "turn_chain" not in view:
                raise ValueError("paired proximity views require turn_chain provenance")
            other_chain = view["turn_chain"][row].to(proximity.device)
            other_chain = F.pad(other_chain, (0, involved.numel() - other_chain.numel()))
            if not bool((labels["turn_chain"][row][involved] != other_chain[involved]).any()):
                continue
            terms.append((gaps_a[row][valid] - gaps_b[row][valid]).abs().mean())
            turn_pairs += int(valid.sum())
        return (torch.stack(terms).mean() if terms else zero), len(terms), turn_pairs
