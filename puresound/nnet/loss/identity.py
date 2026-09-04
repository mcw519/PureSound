"""Speaker-identity contrastive loss on the per-frame `IdentityHead` (v20 R1a).

Turn embeddings are mean-pooled per single-talker turn and contrasted against a
stop-gradient EMA teacher's turn embeddings across the whole batch, so a turn's
positives are the *same speaker rendered through a different capture chain* and
its negatives include bystanders drawn at the user's own distance. That pairing
is the point: with positives all near and negatives all far, a near/far scalar
satisfies the objective and no identity is learned (v20 review item 4).

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
  none. Read by `RelativeProximityLoss`, not here.
- existing keys: ``foreground_distance``, ``foreground_drr``,
  ``nearest_interferer_distance``, ``n_interferers``, ``consistency_noise``,
  ``vad_target``, ``background_vad_target``.

The frame grid is *nominally* the same 100 fps for the labels and for the
bottleneck, but the two are framed differently -- the labeler's analysis window
is 400 samples (`puresound.audio.vad.frame_count`) and the encoder's is 512 --
so at 16 kHz the label grid runs one frame longer than the head's. Everything
here truncates to the common prefix, which is the alignment rule
`VADHeadBCELoss` already uses; frame 0 is frame 0 on both sides.
"""

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: ``turn_id`` value for a frame that carries no single-talker turn: silence, or
#: overlap the generator refused to attribute to either talker.
NO_TURN = 0


def align_frames(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Truncate every argument to the shortest frame axis (always axis 1).

    Shared by both v20 R1a losses so "which grid do turns pool on" has one
    answer. See the module docstring for why the two grids differ by a frame.
    """
    n = min(int(t.shape[1]) for t in tensors)
    return tuple(t[:, :n] for t in tensors)


def align_turn_frames(
    values: Sequence[torch.Tensor], batch: Dict, device: torch.device
) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]]:
    """Put per-frame model outputs and the batch's turn labels on one grid.

    Args:
        values: ``[B, T, ...]`` per-frame tensors (the head outputs).
        batch: the training batch; read for ``turn_id`` and, when present, the
            two activity tracks that mark overlap.
        device: where the model's tensors live.

    Returns:
        ``(values, turn_id, exclude)`` truncated to the common frame prefix, or
        None when the batch carries no ``turn_id`` at all (today's rows). The
        ``exclude`` mask is ``user_active AND bystander_active``; ``turn_id ==
        NO_TURN`` already marks overlap per the contract, so it is a second lock
        on the same door -- cheap, and the one that stops a generator bug from
        quietly training identity on mixed frames.
    """
    turn_id = batch.get("turn_id")
    if turn_id is None:
        return None
    turn_id = turn_id.to(device=device).long()

    user = batch.get("user_active")
    bystander = batch.get("bystander_active")
    extras: List[torch.Tensor] = []
    if user is not None and bystander is not None:
        extras = [user.to(device=device), bystander.to(device=device)]

    aligned = align_frames(*values, turn_id, *extras)
    n = len(values)
    exclude = None
    if extras:
        exclude = (aligned[n + 1] > 0.5) & (aligned[n + 2] > 0.5)
    return list(aligned[:n]), aligned[n], exclude


def pool_turn_means(
    values: torch.Tensor,
    turn_id: torch.Tensor,
    n_turns: int,
    exclude: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean of ``values`` over the frames of each turn.

    Args:
        values: [B, T, D] per-frame quantity (D = 1 for a scalar readout).
        turn_id: [B, T] long, 0 = no turn, 1..n_turns = that turn's slot.
        n_turns: K_max, the width of the per-turn label tensors.
        exclude: [B, T] bool, frames to drop (overlap).

    Returns:
        ``(means, counts)`` -- [B, K, D] float32 and [B, K] long. A turn with no
        frames gets a zero mean and count 0; the caller drops it on the count,
        which is what keeps the division NaN-free rather than a masked-mean
        dance.

    The sum and the frame count run in **float32 whatever the autocast dtype**.
    bf16 carries 8 mantissa bits, so a 12 s turn's frame count (1197) is not
    even representable -- it rounds to 1200 -- and the sum of 1197 frames is
    wrong by the same 0.25 %: the turn mean would be silently mis-scaled, worst
    for the longest turns. ``autocast(enabled=False)``, not just ``.float()``,
    for the reason `VADHead._ema_bank` records: inside an autocast region a bare
    cast is undone again by the next op.
    """
    if turn_id.numel() and int(turn_id.max()) > n_turns:
        # one_hot's own message ("Class values must be smaller than num_classes")
        # names neither tensor; the generator writing these labels deserves better.
        raise ValueError(
            f"turn_id reaches {int(turn_id.max())} but the per-turn label tensors "
            f"are only {n_turns} wide: turn_id indexes 1..K_max, 0 = no turn."
        )
    keep = turn_id != NO_TURN
    if exclude is not None:
        keep = keep & ~exclude
    with torch.autocast(device_type=values.device.type, enabled=False):
        slots = F.one_hot(turn_id.clamp(min=0), num_classes=n_turns + 1)[..., 1:]
        slots = slots.float() * keep.float().unsqueeze(-1)  # [B, T, K]
        counts = slots.sum(dim=1)  # [B, K]
        sums = torch.einsum("btk,btd->bkd", slots, values.float())  # [B, K, D]
        means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return means, counts.long()


def eligible_turns(
    counts: torch.Tensor, batch: Dict, device: torch.device, min_frames: int = 1
) -> torch.Tensor:
    """[B, K] bool: turns with enough frames and a non-pad label row."""
    keep = counts >= max(1, int(min_frames))
    role = batch.get("turn_role")
    if role is not None:
        keep = keep & (role.to(device=device).long() != 0)
    return keep


class IdentityContrastiveLoss(nn.Module):
    """InfoNCE over turn embeddings, against a stop-gradient EMA teacher.

    ``e_k`` = L2-normalised mean of the head's per-frame embedding over turn
    ``k``'s non-overlap frames. Candidates are every *other* eligible turn in
    the batch, embedded by the teacher; positives are the ones with the same
    ``turn_speaker`` -- in the same row (a later turn, possibly after the RIR
    move) and in other rows, where that speaker was rendered through a different
    device chain and possibly in the bystander role::

        L_id = mean_k  -log  Σ_pos exp(cos(e_k, ē_p)/τ)
                             ---------------------------------
                             Σ_{pos ∪ neg} exp(cos(e_k, ē_·)/τ)

    **The teacher.** A deep copy of the live head, kept inside this module and
    advanced as ``p_t <- m*p_t + (1-m)*p_s`` (m = ``momentum``, default 0.99) on
    every *training* call. It is what makes the targets stop-gradient: with both
    sides trainable the two collapse onto each other by co-adaptation and the
    loss falls having learned nothing (v20 §4.2(i)). Three deliberate
    properties:

    * It is **not** a submodule -- it lives in a list, the trick
      `register_gpu_vad_labeler` uses -- so it never reaches a checkpoint. A
      lazily-created submodule would change ``state_dict`` after step 1 and
      break its own resume, and non-trained weights in a checkpoint are weights
      someone eventually loads by accident. After a resume the teacher restarts
      as a copy of the student, which is exactly what it is at initialisation.
    * It runs under ``no_grad`` and its parameters carry
      ``requires_grad = False``, so the gradient really does stop and DDP never
      sees them.
    * It reads the **bottleneck**, not the student's output, so this loss needs
      the graph-carrying ``bottleneck`` provider: build the backbone with
      ``backbone_args.expose_bottleneck: true``. It uses those features only to
      run its own copy of the head over them, under ``no_grad``.

    Zero-contribution rules, all graph-carrying (`dist.py`'s idiom, so DDP never
    sees an unused head): a batch with no ``turn_id`` / ``turn_speaker``, fewer
    than two eligible turns, or no turn whose speaker appears anywhere else in
    the batch returns ``identity_emb.sum() * 0``.
    """

    required_inputs = ("identity_emb", "identity_head", "bottleneck", "batch")

    def __init__(
        self,
        temperature: float = 0.1,
        momentum: float = 0.99,
        min_turn_frames: int = 1,
    ):
        super().__init__()
        if not temperature > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.min_turn_frames = int(min_turn_frames)
        # List-wrapped so nn.Module does not register it; see the class docstring.
        self._teacher: List[nn.Module] = []

    # ------------------------------------------------------------------ teacher

    @property
    def teacher(self) -> Optional[nn.Module]:
        """The EMA copy of the head, or None before the first call."""
        return self._teacher[0] if self._teacher else None

    @torch.no_grad()
    def _sync_teacher(self, head: nn.Module) -> nn.Module:
        if not self._teacher:
            teacher = copy.deepcopy(head)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            teacher.eval()
            self._teacher = [teacher]
            return teacher

        teacher = self._teacher[0]
        student_params = list(head.parameters())
        teacher_params = list(teacher.parameters())
        if len(student_params) != len(teacher_params) or any(
            s.shape != t.shape for s, t in zip(student_params, teacher_params)
        ):
            raise ValueError(
                "IdentityContrastiveLoss's EMA teacher was built from a head with a "
                "different shape than the one it is now given; one loss instance "
                "tracks one head."
            )
        # Follow the head across a .to(device)/.to(dtype) taken after the copy.
        if teacher_params and (
            teacher_params[0].device != student_params[0].device
            or teacher_params[0].dtype != student_params[0].dtype
        ):
            teacher.to(device=student_params[0].device, dtype=student_params[0].dtype)
            teacher_params = list(teacher.parameters())
        # Only while training: advancing the teacher inside validation_step would
        # fold held-out batches into the targets.
        if self.training:
            for target, source in zip(teacher_params, student_params):
                target.mul_(self.momentum).add_(
                    source.detach(), alpha=1.0 - self.momentum
                )
            for target, source in zip(teacher.buffers(), head.buffers()):
                target.copy_(source.detach())
        return teacher

    # --------------------------------------------------------------------- loss

    def forward(
        self,
        identity_emb: Optional[torch.Tensor],
        identity_head: Optional[nn.Module],
        bottleneck: Optional[torch.Tensor],
        batch: Dict,
    ) -> torch.Tensor:
        if identity_emb is None or identity_head is None:
            raise ValueError(
                "IdentityContrastiveLoss requires the backbone to expose "
                "`last_identity_emb`; enable an identity_head in the backbone "
                "config (model.backbone.backbone_args.identity_head)."
            )
        if bottleneck is None:
            raise ValueError(
                "IdentityContrastiveLoss needs the graph-carrying bottleneck to run "
                "its EMA teacher; build the backbone with "
                "model.backbone.backbone_args.expose_bottleneck: true."
            )

        zero = identity_emb.sum() * 0.0
        turn_speaker = batch.get("turn_speaker")
        if turn_speaker is None or batch.get("turn_id") is None:
            return zero

        device = identity_emb.device
        turn_speaker = turn_speaker.to(device=device).long()
        n_turns = int(turn_speaker.shape[1])
        if n_turns == 0:
            return zero

        teacher = self._sync_teacher(identity_head)
        with torch.no_grad():
            # `bottleneck` is the frequency-pooled [B, C, T]; the head takes an
            # already-pooled tensor unchanged.
            teacher_emb = teacher(bottleneck.detach()).float()

        aligned = align_turn_frames([identity_emb, teacher_emb], batch, device)
        if aligned is None:
            return zero
        (student_emb, teacher_emb), turn_id, exclude = aligned

        student_turns, counts = pool_turn_means(student_emb, turn_id, n_turns, exclude)
        with torch.no_grad():
            teacher_turns, _ = pool_turn_means(teacher_emb, turn_id, n_turns, exclude)

        eligible = eligible_turns(counts, batch, device, self.min_turn_frames)
        eligible = eligible & (turn_speaker >= 0)
        if int(eligible.sum()) < 2:
            return zero

        rows, slots = torch.nonzero(eligible, as_tuple=True)
        speakers = turn_speaker[rows, slots]  # [M]
        self_pair = torch.eye(speakers.shape[0], dtype=torch.bool, device=device)
        positive = (speakers.unsqueeze(1) == speakers.unsqueeze(0)) & ~self_pair
        has_positive = positive.any(dim=1)
        if not bool(has_positive.any()):
            # Speaker ids present but nobody appears twice: nothing to contrast.
            return zero

        # float32 throughout, autocast off: [M, M] is a handful of turns, and a
        # bf16 cosine carries ~3 decimal digits, which at tau = 0.1 is a
        # third of a logit.
        with torch.autocast(device_type=device.type, enabled=False):
            anchors = F.normalize(student_turns[rows, slots].float(), dim=-1, p=2)
            targets = F.normalize(teacher_turns[rows, slots].float(), dim=-1, p=2)
            logits = (anchors @ targets.transpose(0, 1)) / self.temperature  # [M, M]
            # masked_fill rather than indexing: its backward zeroes the masked
            # entries, so the excluded self-similarity contributes neither
            # gradient nor NaN. finfo.min rather than -inf so a row that happens
            # to be fully masked stays finite instead of poisoning the mean.
            neg_inf = torch.finfo(logits.dtype).min
            denominator = torch.logsumexp(logits.masked_fill(self_pair, neg_inf), dim=1)
            numerator = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
            return (denominator - numerator)[has_positive].mean()
