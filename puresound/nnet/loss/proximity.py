"""Relative-proximity loss on the per-frame `ProximityHead` (v20 R1a).

Two terms, neither of which ever names a distance:

1. **Ordering.** Within a row, every (user turn u, bystander turn b) pair must
   satisfy ``p̄_u - p̄_b >= m`` in the head's own units:
   ``softplus(m - (p̄_u - p̄_b))``.
2. **Cross-chain consistency.** Two rows rendered from the same source material
   through *different* device chains must produce the same contrast:
   ``|Δ(p̄_u - p̄_b)|``.

Absolute readouts died three times in this repo (fixed dB thresholds and a
metres regression, each on a chain offset or a checkpoint drift), which is why
this loss supervises differences only and why nothing downstream reads the
head's value. `DistHead`'s metres regression stays as an auxiliary.

Batch contract (produced by the session-row generator; every key optional --
rows without them make this loss a graph-carrying zero, so old recipes are
unaffected). All on the 100 fps frame grid of ``vad_target`` (hop 160 at
16 kHz):

- ``user_active`` float [B, T] ∈ {0,1}; ``bystander_active`` float [B, T];
  overlap = both 1.
- ``turn_id`` long [B, T]: unique id (1..K) per contiguous single-talker turn,
  0 = no turn / overlap.
- ``turn_role`` long [B, K_max]: 1 = user, 2 = bystander, 0 = pad;
  ``turn_speaker`` long [B, K_max]: global speaker id (−1 pad); ``turn_chain``
  long [B, K_max]: id of the device-chain draw of the row (same for all turns of
  a row).
- ``row_source_id`` long [B]: the source material a row was rendered from, −1 =
  none. Only rows that share a non-negative id and differ in ``turn_chain``
  enter the consistency term.
- existing keys: ``foreground_distance``, ``foreground_drr``,
  ``nearest_interferer_distance``, ``n_interferers``, ``consistency_noise``,
  ``vad_target``, ``background_vad_target``.

Frame-grid alignment is `identity.align_turn_frames`, shared with the identity
loss so "which grid do turns pool on" has one implementation: the label grid
runs a frame longer than the head's (400-sample labeler window against the
encoder's 512), and both are truncated to the common prefix.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity import align_turn_frames, eligible_turns, pool_turn_means

#: ``turn_role`` values in the batch contract.
ROLE_PAD = 0
ROLE_USER = 1
ROLE_BYSTANDER = 2


class RelativeProximityLoss(nn.Module):
    """Margin on the user-minus-bystander proximity contrast, plus cross-chain
    consistency of that contrast.

    ``p̄_k`` is the mean of the head's per-frame scalar over turn ``k``'s
    non-overlap frames, pooled exactly as the identity loss pools embeddings.

    Args:
        margin: ``m``, how far above a bystander's turn the user's must read.
            In the head's own units -- the head is free to choose its scale, and
            only ``m`` fixes what "clearly nearer" means relative to it.
        consistency_weight: multiplier on term 2. ``0.0`` leaves the ordering
            term alone, which is the ablation the round is designed to be able
            to run.
        min_turn_frames: turns shorter than this are ignored (a two-frame mean
            is noise).

    Zero-contribution rules, all graph-carrying (`dist.py`'s idiom, so DDP never
    sees an unused head): a batch with no ``turn_id`` / ``turn_role``, no row
    holding both a user turn and a bystander turn, and no cross-chain row pair
    returns ``proximity.sum() * 0``.
    """

    required_inputs = ("proximity", "batch")

    def __init__(
        self,
        margin: float = 1.0,
        consistency_weight: float = 1.0,
        min_turn_frames: int = 1,
    ):
        super().__init__()
        self.margin = float(margin)
        self.consistency_weight = float(consistency_weight)
        self.min_turn_frames = int(min_turn_frames)

    def forward(
        self, proximity: Optional[torch.Tensor], batch: Dict
    ) -> torch.Tensor:
        if proximity is None:
            raise ValueError(
                "RelativeProximityLoss requires the backbone to expose "
                "`last_proximity`; enable a proximity_head in the backbone config "
                "(model.backbone.backbone_args.proximity_head)."
            )

        zero = proximity.sum() * 0.0
        turn_role = batch.get("turn_role")
        if turn_role is None or batch.get("turn_id") is None:
            return zero

        device = proximity.device
        turn_role = turn_role.to(device=device).long()
        n_turns = int(turn_role.shape[1])
        if n_turns == 0:
            return zero

        aligned = align_turn_frames([proximity.unsqueeze(-1)], batch, device)
        if aligned is None:
            return zero
        (frames,), turn_id, exclude = aligned

        means, counts = pool_turn_means(frames, turn_id, n_turns, exclude)
        means = means.squeeze(-1)  # [B, K]
        eligible = eligible_turns(counts, batch, device, self.min_turn_frames)

        users = eligible & (turn_role == ROLE_USER)
        bystanders = eligible & (turn_role == ROLE_BYSTANDER)
        # One device-to-host transfer for the whole row loop rather than two per
        # row: on a GPU each `bool(tensor)` is a synchronisation.
        pairable = (users.any(dim=1) & bystanders.any(dim=1)).tolist()

        ordering: List[torch.Tensor] = []
        contrasts: Dict[int, torch.Tensor] = {}
        for row, has_pair in enumerate(pairable):
            if not has_pair:
                continue
            # [n_user, n_bystander] of p̄_u - p̄_b: every pair, because the
            # ordering has to hold for the user's quiet turns too, not only for
            # whichever pair happens to be easiest.
            gaps = (
                means[row][users[row]].unsqueeze(1)
                - means[row][bystanders[row]].unsqueeze(0)
            )
            ordering.append(F.softplus(self.margin - gaps).reshape(-1))
            contrasts[row] = gaps.mean()

        consistency: List[torch.Tensor] = []
        chains = self._row_chains(batch, eligible)
        source_ids = batch.get("row_source_id")
        if source_ids is not None and chains is not None and self.consistency_weight:
            sources = source_ids.long().view(-1).tolist()
            paired_rows = sorted(contrasts)
            for i, left in enumerate(paired_rows):
                for right in paired_rows[i + 1 :]:
                    if sources[left] < 0 or sources[left] != sources[right]:
                        continue
                    if chains[left] == chains[right]:
                        # Same chain: this pair says nothing about chain
                        # robustness, and the ordering term already covers it.
                        continue
                    consistency.append(
                        (contrasts[left] - contrasts[right]).abs().reshape(1)
                    )

        total = None
        if ordering:
            total = torch.cat(ordering).mean()
        if consistency:
            term = self.consistency_weight * torch.cat(consistency).mean()
            total = term if total is None else total + term
        return zero if total is None else total

    @staticmethod
    def _row_chains(batch: Dict, eligible: torch.Tensor) -> Optional[List[int]]:
        """Device-chain id per row, read off the row's first eligible turn.

        The contract says one chain draw per row (``ns.py`` applies the chain
        once, jointly to mixture and target -- a per-talker draw would break
        mixture/target consistency, v20 review item 6), so any eligible turn
        answers for the row; the first is taken to keep it deterministic.
        """
        chain = batch.get("turn_chain")
        if chain is None:
            return None
        chain = chain.to(device=eligible.device).long()
        first = torch.argmax(eligible.to(torch.int8), dim=1)  # 0 if none eligible
        return chain.gather(1, first.unsqueeze(1)).squeeze(1).tolist()
