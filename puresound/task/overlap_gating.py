"""Who is talking when.

Left alone, a synthesised row is two people talking continuously for its whole
length: both the target clip and the interferer clip are unbroken speech, so
better than 80% overlap dominates the training distribution and the model never
sees the cases that decide real behaviour -- an interferer that talks in the
target's silences, or a far voice holding the floor alone with no near speaker
anywhere in the row.

Two ways to break that up, and they are not variants of each other:

* **Per-frame Bernoulli.** Each interferer draws its own overlap regime (none /
  mid / high) and is gated frame by frame, so one row can carry an easy
  interferer and a hard one at once. This is the default and it produces the
  spread.
* **Turn-taking.** Near and far alternate in long conversational blocks, and
  half the rows open on a *far* turn. That is the one shape the Bernoulli fill
  can never produce -- a multi-second far monologue with no preceding near
  anchor -- and it is the regime where real far speech was measured passing
  through untouched.

Contracts, the same ones `DeviceChain` holds and for the same reason:

**RNG order is load-bearing.** Every branch here draws from the shared stream,
so moving a draw changes what a seeded recipe produces even when the arithmetic
is unchanged.

**A row that gates nothing draws nothing.** Gating disabled, no labeler, no
interferers, a target with no active frames, or `turn_taking_prob` at zero --
each returns before its draw, so turning gating off leaves every later stage
seeing exactly what it saw before.

**In turn-taking the target's label and the target's contribution to the mixture
are gated by the same envelope.** Gate one without the other and the row claims
the near speaker was talking during frames where the mixture has silence.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class GatingResult(NamedTuple):
    """The gated signals, plus what the row should report about itself.

    ``overlap_fraction`` is the *realized* share of target-active frames that an
    interferer also covers -- the drawn probabilities are not it, because the
    target's own silences move the outcome. Eval buckets by it to read whether a
    checkpoint is overlap-limited without a separate no-overlap run. NaN when
    gating did not run. ``turn_taking`` is 1.0 on rows that took that branch.
    """

    target: torch.Tensor
    target_mix: Optional[torch.Tensor]
    interferers: list
    overlap_fraction: float
    turn_taking: float


class _Envelope:
    """Turns a frame mask into a sample-rate gain curve.

    A frame mask flips between 0 and 1 in a single sample, and a step
    discontinuity in a speech waveform is a click. The mask is upsampled by the
    hop and convolved with a normalized Hann taper so every gate opens and
    closes over ``fade_samples`` instead. Consumes no randomness.
    """

    def __init__(self, hop: int, fade_samples: int):
        self.hop = hop
        self.fade_samples = fade_samples
        kernel = torch.hann_window(fade_samples * 2 + 1)
        self.kernel = (kernel / kernel.sum().clamp_min(1e-6)).view(1, 1, -1)

    def curve(self, frames_mask: torch.Tensor, length: int) -> torch.Tensor:
        env = frames_mask.float().repeat_interleave(self.hop)
        if env.shape[0] < length:
            env = torch.nn.functional.pad(env, (0, length - env.shape[0]))
        else:
            env = env[:length]
        return (
            torch.nn.functional.conv1d(
                env.view(1, 1, -1), self.kernel, padding=self.fade_samples
            )
            .view(-1)
            .clamp(0.0, 1.0)
        )

    def gate(self, sig: torch.Tensor, frames_mask: torch.Tensor) -> torch.Tensor:
        curve = self.curve(frames_mask, sig.shape[-1])
        return sig * curve.view(*([1] * (sig.dim() - 1)), -1)


class OverlapGating:
    """Applies one of the two gating regimes to a row.

    Built once per dataset from the already-validated ``overlap_control`` block
    and the cheap activity labeler; ``sr`` is per call because it follows the
    item when a recipe has no ``target_sample_rate``.
    """

    def __init__(self, config, vad_labeler):
        self.config = config
        # The coarse energy labeler, not the loss label's: this runs inside the
        # DataLoader workers on every row.
        self.vad_labeler = vad_labeler

    @property
    def enabled(self) -> bool:
        return bool(
            self.config is not None
            and self.config.used
            and self.vad_labeler is not None
        )

    def apply(
        self,
        target: torch.Tensor,
        interferers: list,
        *,
        sr: int,
        target_mix: Optional[torch.Tensor] = None,
        allow_turn_taking: bool = True,
        turn_taking_prob: Optional[float] = None,
    ) -> GatingResult:
        """Gate interferer activity, and in turn-taking mode the target too.

        ``allow_turn_taking=False`` is for target-absent rows: the foreground is
        subtracted from the mixture later using a pre-gating snapshot, so gating
        it here would leave a residual of exactly the voice the row claims is
        not there.

        ``turn_taking_prob`` overrides the block's own rate for one row. Row
        types that want more far-solo stretches ask for them here rather than in
        the config, which keeps every row type that does not override it
        bit-identical.
        """
        untouched = GatingResult(target, target_mix, interferers, float("nan"), 0.0)
        if not self.enabled or not interferers:
            return untouched

        target_wav = target.squeeze(0) if target.dim() == 2 else target
        target_active = self.vad_labeler(target_wav, sample_rate=sr).bool()
        if int(target_active.sum().item()) == 0:
            # Nothing is talking, so "overlap" has no meaning on this row.
            return untouched

        envelope = _Envelope(self.vad_labeler.hop_length, self.config.fade_samples)
        prob = (
            self.config.turn_taking_prob
            if turn_taking_prob is None
            else float(turn_taking_prob)
        )
        if allow_turn_taking and prob > 0.0 and torch.rand(1).item() < prob:
            return self._turn_taking(
                target, target_mix, interferers, target_active, sr, envelope
            )
        return self._bernoulli(
            target, target_mix, interferers, target_active, envelope
        )

    # ------------------------------------------------------------------ #

    def _turn_taking(
        self, target, target_mix, interferers, target_active, sr, envelope
    ) -> GatingResult:
        """Long alternating blocks: near speaks, then far, then near."""
        n_frames = target_active.shape[0]
        near_mask, far_mask = self._turn_script(n_frames, envelope.hop, sr)
        target = envelope.gate(target, near_mask)
        if target_mix is not None:
            target_mix = envelope.gate(target_mix, near_mask)
        gated = [envelope.gate(interferer, far_mask) for interferer in interferers]

        # Against the target's *gated* activity, not its original: the near
        # speaker is silent through the far turns by construction, and counting
        # those frames as target-active would report overlap that is not there.
        active_gated = target_active & near_mask
        active = int(active_gated.sum().item())
        overlap = (
            float(int((far_mask & active_gated).sum().item()) / active)
            if active > 0
            else 0.0
        )
        return GatingResult(target, target_mix, gated, overlap, 1.0)

    def _turn_script(self, n_frames: int, hop: int, sr: int) -> tuple:
        """Sample a conversational turn script on the VAD frame grid.

        Alternating near/far turns like a real exchange, with a small gap or a
        slight boundary overlap between them. ``far_first_prob`` of rows open on
        a FAR turn -- the hardest streaming case, a far monologue with no
        preceding near anchor. Returns (near_frames, far_frames) boolean masks.
        """
        cfg = self.config
        near_len = cfg.turn_near_seconds
        far_len = cfg.turn_far_seconds
        gap_rng = cfg.turn_gap_seconds
        ovl_rng = cfg.turn_overlap_seconds
        far_first = torch.rand(1).item() < cfg.far_first_prob

        fps = float(sr) / float(hop)
        near_mask = torch.zeros(n_frames, dtype=torch.bool)
        far_mask = torch.zeros(n_frames, dtype=torch.bool)
        pos = 0
        turn_is_far = far_first
        while pos < n_frames:
            rng = far_len if turn_is_far else near_len
            turn_s = float(torch.empty(1).uniform_(float(rng[0]), float(rng[1])))
            turn_f = max(1, int(round(turn_s * fps)))
            end = min(pos + turn_f, n_frames)
            (far_mask if turn_is_far else near_mask)[pos:end] = True
            if torch.rand(1).item() < 0.5:  # gap between turns
                step_s = float(
                    torch.empty(1).uniform_(float(gap_rng[0]), float(gap_rng[1]))
                )
                pos = end + int(round(step_s * fps))
            else:  # slight boundary overlap (next turn starts before this ends)
                step_s = float(
                    torch.empty(1).uniform_(float(ovl_rng[0]), float(ovl_rng[1]))
                )
                pos = max(pos + 1, end - int(round(step_s * fps)))
            turn_is_far = not turn_is_far
        return near_mask, far_mask

    # ------------------------------------------------------------------ #

    def _bernoulli(
        self, target, target_mix, interferers, target_active, envelope
    ) -> GatingResult:
        """Per-frame coin flips, at a different rate per interferer.

        The target is not gated here -- only who talks *over* it changes.
        """
        cfg = self.config
        n_frames = target_active.shape[0]
        gated = []
        union_wanted = torch.zeros(n_frames, dtype=torch.bool)
        for interferer in interferers:
            overlap_p = self._draw_overlap_rate()
            # A separate, usually higher rate for the target's silences: an
            # interferer that only ever speaks *during* the target is a much
            # narrower distribution than the one real conversations produce.
            fill_p = float(
                torch.empty(1).uniform_(
                    float(cfg.fill_on_silent_range[0]),
                    float(cfg.fill_on_silent_range[1]),
                )
            )
            wanted = torch.zeros(n_frames)
            draws = torch.rand(n_frames)
            wanted[target_active] = (draws[target_active] < overlap_p).float()
            wanted[~target_active] = (draws[~target_active] < fill_p).float()
            union_wanted |= wanted.bool()
            gated.append(envelope.gate(interferer, wanted.bool()))

        # `apply` already returned on a silent target, so this cannot divide by
        # zero -- unlike the turn-taking branch, where the near envelope can
        # legitimately miss every active frame.
        active = int(target_active.sum().item())
        overlap = float(int((union_wanted & target_active).sum().item()) / active)
        return GatingResult(target, target_mix, gated, overlap, 0.0)

    def _draw_overlap_rate(self) -> float:
        """One of three regimes, so a batch spans the whole spread.

        One roll against two cumulative thresholds picks the regime, and only
        the two non-zero regimes then draw a rate inside their range -- so the
        number of values an interferer consumes depends on which regime it drew.
        That is fine within a row but it is why the regime thresholds cannot be
        reordered: swap `no_overlap_prob` and `high_overlap_prob` and the same
        seed produces a different row, not merely a differently-weighted one.
        """
        cfg = self.config
        roll = torch.rand(1).item()
        if roll < cfg.no_overlap_prob:
            return 0.0
        if roll < cfg.no_overlap_prob + cfg.high_overlap_prob:
            lo, hi = cfg.high_overlap_range
        else:
            lo, hi = cfg.mid_overlap_range
        return float(torch.empty(1).uniform_(float(lo), float(hi)))
