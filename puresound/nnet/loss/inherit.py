"""Anchor-inheritance hinge: the next talker must not inherit the last one's gain.

The failure this is written against: after a few seconds of a *different* near
talker, the user's own first second comes out quieter than it does with no
prefix at all. The model treats "whoever spoke last" as the foreground and hands
the next voice the gain it had decided on for the previous one -- it *inherits
the anchor*.

What this term says, and nothing else:

    over the user's onset window, the gain achieved ON THE USER must sit at
    least ``margin_db`` above the gain the model applied to WHAT WAS THERE
    BEFORE the user arrived.

Three design properties:

* **Only a difference appears.** No absolute dB threshold anywhere: an absolute
  calibration does not carry across capture chains or across checkpoints of
  one run, while a relative, self-calibrated form does.
* **The pre-onset term is detached.** The hinge therefore cannot be satisfied
  by suppressing the bystander *less* -- that would buy onset keep with far
  suppression, the trade an inference-time ``OnsetGuard`` makes and this term
  exists to avoid.
* **It is a hinge on an onset window, not a row mean.** The row-wide contrast
  is already large on most rows and a 1 s event barely moves a row mean, so a
  row-mean form carries almost no gradient.

Scores, per frame, on the ``EnergyVADLabeler`` grid (400 / 160 = 100 fps, the
grid ``vad_target`` is on)::

    a_t = <E_t, R_t> / max(<R_t, R_t>, 1e-10)     # achieved gain on the user
    q_t = <E_t, N_t> / max(<N_t, N_t>, 1e-10)     # achieved gain on what was there before
    pos(z) = softplus(16 z) / 16
    P_t = clamp(10 log10(pos(a_t)^2 + 1e-12), -30, 0)
    Q_t = same, on q_t

``E`` = enhanced, ``R`` = ``target`` (the user), ``N`` =
``batch["consistency_noise"]`` = ``noisy - target``. No per-talker plumbing is
needed and none is added: during the bystander's solo run the user is silent by
construction, so ``noisy - target`` over the pre-onset frames *is* the bystander
plus noise, already in every batch and already on the post-speed-perturbation
grid (``task/ns.py``: ``consistency_noise`` is built from the final waveforms and
``vad_reference`` is cloned after the speed block). It also avoids
``background_speech_reference``, a pre-speed snapshot that is misaligned by the
speed factor.

**The -30 dB floor is load-bearing, not cosmetic.** Unfloored, the pooled hinge
is dominated by a small tail of onsets whose output is anti-correlated with the
reference, so the statistic detects phase / anti-correlation tails as much as
deletion. ``softplus(16 .)`` does not remove the dead zone either, it moves it: the
unclamped envelope's slope at ``a = -1`` is 0.0069 dB per unit ``a``, and the
clamp is what actually zeroes the gradient, from ``a <= -0.0261`` down. Report
medians, never bare means.

Row eligibility (evaluated per row; no dataset change, no new provider):

* ``batch["n_interferers"] >= 1``,
* at least ``min_pre_interferer_frames`` frames of interferer speech strictly
  before the user's first active frame. This is the **only** use of
  ``background_vad_target``, and it is deliberately a row-level 1 s test: that
  target comes from a pre-speed snapshot, and a +-5% timing error cannot flip it,
* the user's first active frame is at least ``min_onset_frame`` into the row,
* at least ``onset_frames`` frames of user activity from that frame on, so the
  onset window is full rather than partial.

Frame sets, both from aligned tensors:

* ``O`` = the first ``onset_frames`` *active* frames of ``vad_target`` at the
  onset (0.5 s of speech at the defaults).
* ``B`` = frames before the onset with ``vad_target == 0`` **and**
  ``consistency_noise`` frame energy within ``pre_energy_window_db`` of its own
  maximum over that prefix, so a noise-only prefix cannot make the hinge
  trivially satisfiable.

The hinge::

    Pbar_on  = sum_O  w_t  P_t / sum_O  w_t          w_t  = <R_t, R_t>
    Qbar_pre = detach( sum_B w'_t Q_t / sum_B w'_t ) w'_t = <N_t, N_t>
    L        = mean over eligible rows of relu(margin_db - (Pbar_on - Qbar_pre))

A batch with no eligible row returns a **graph-carrying zero**
(``enhanced.sum() * 0.0``, the ``dist.py`` idiom). A ``None`` or a NaN here
would sever a DDP job.

The loss has no parameters, so it changes neither the checkpoint layout of a
warm start nor the streaming export.

``row_scores`` is the same arithmetic with nothing reduced away: it is what a
pre-flight check or a validation-time monitor reads, so the number that is
monitored and the number that is trained come from one implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorInheritanceLoss(nn.Module):
    """See the module docstring.

    Args:
        margin_db: ``M``, how far above the pre-onset background's achieved gain
            the user's onset gain must sit.
        frame_length / hop_length: the ``EnergyVADLabeler`` grid that produced
            ``vad_target``. Changing these desynchronises the loss from its own
            labels, so they are arguments only so a test can shrink them.
        onset_frames: ``|O|`` -- active frames pooled at the onset.
        min_onset_frame: the user's first active frame must be at least this far
            into the row (1 s), else there is no "before" to inherit from.
        min_pre_interferer_frames: frames of interferer speech required strictly
            before the onset (1 s; a shorter prefix sets a much weaker anchor).
        fallback_pre_interferer_frames: a looser rule (0.5 s), reported by
            ``row_scores`` but never used by ``forward``.
        pre_energy_window_db: a pre-onset frame joins ``B`` only if its
            ``consistency_noise`` energy is within this of the prefix's own peak
            frame.
        floor_db: the load-bearing floor on ``P`` and ``Q``.
        softplus_beta: the ``pos()`` sharpness (16 -> ``max|dP/da|`` ~ 139).
    """

    required_inputs = ("enhanced", "target", "batch", "vad_target")

    def __init__(
        self,
        margin_db: float = 10.0,
        frame_length: int = 400,
        hop_length: int = 160,
        onset_frames: int = 50,
        min_onset_frame: int = 100,
        min_pre_interferer_frames: int = 100,
        fallback_pre_interferer_frames: int = 50,
        pre_energy_window_db: float = 25.0,
        floor_db: float = -30.0,
        softplus_beta: float = 16.0,
        eps_ratio: float = 1e-10,
        eps_db: float = 1e-12,
        reference_key: str = "consistency_noise",
    ):
        super().__init__()
        self.margin_db = float(margin_db)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.onset_frames = int(onset_frames)
        self.min_onset_frame = int(min_onset_frame)
        self.min_pre_interferer_frames = int(min_pre_interferer_frames)
        self.fallback_pre_interferer_frames = int(fallback_pre_interferer_frames)
        self.pre_energy_window_db = float(pre_energy_window_db)
        self.floor_db = float(floor_db)
        self.softplus_beta = float(softplus_beta)
        self.eps_ratio = float(eps_ratio)
        self.eps_db = float(eps_db)
        self.reference_key = str(reference_key)
        if self.onset_frames <= 0:
            raise ValueError(f"onset_frames must be > 0, got {onset_frames}")
        if self.min_pre_interferer_frames < self.fallback_pre_interferer_frames:
            raise ValueError(
                "fallback_pre_interferer_frames is the *looser* rule and must not "
                f"exceed min_pre_interferer_frames "
                f"({fallback_pre_interferer_frames} > {min_pre_interferer_frames})"
            )
        if not self.floor_db < 0.0:
            raise ValueError(
                f"floor_db must be < 0 dB, got {floor_db}: at or above 0 the floor "
                "and the ceiling meet and P carries no gradient at all"
            )

    # ------------------------------------------------------------------ #
    # the floored envelope

    def floored_gain_db(self, ratio: torch.Tensor) -> torch.Tensor:
        """``clamp(10 log10(pos(ratio)^2 + eps), floor_db, 0)``, ``pos`` softplus.

        Separate and public because a change here changes what "deletion"
        means, so a test pins a table of its values rather than the
        implementation.
        """
        pos = F.softplus(ratio, beta=self.softplus_beta)
        return (10.0 * torch.log10(pos.square() + self.eps_db)).clamp(
            min=self.floor_db, max=0.0
        )

    # ------------------------------------------------------------------ #
    # framing

    @staticmethod
    def _as_batch_waveform(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3:
            wav = wav.mean(dim=1)
        if wav.dim() != 2:
            raise ValueError(f"expected [B, T] or [B, C, T], got {tuple(wav.shape)}")
        return wav

    def _frames(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, F, frame_length] on the labeler's grid."""
        if wav.shape[-1] < self.frame_length:
            wav = F.pad(wav, (0, self.frame_length - wav.shape[-1]))
        return wav.unfold(-1, self.frame_length, self.hop_length)

    def _align_labels(
        self, labels: Optional[torch.Tensor], n_rows: int, n_frames: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Frame labels on the score grid; a missing target is all-silent.

        An all-silent background target is a *valid* signal, not a crash: when
        no row in the batch carries background speech the collate emits no
        ``background_vad_target`` at all (`VoiceIsolationCollateFunc`), and such
        a batch simply has no eligible row.
        """
        if labels is None:
            return torch.zeros(n_rows, n_frames, device=device, dtype=dtype)
        out = labels.to(device=device, dtype=dtype)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] > n_frames:
            out = out[..., :n_frames]
        elif out.shape[-1] < n_frames:
            out = F.pad(out, (0, n_frames - out.shape[-1]))
        return (out > 0.5).to(dtype)

    # ------------------------------------------------------------------ #
    # the arithmetic, nothing reduced away

    def row_scores(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        batch: Dict,
        vad_target: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per-row hinge ingredients and both eligibility rules.

        Every returned tensor is [B]-shaped. ``hinge`` carries the graph;
        ``qbar_pre`` does not, by construction. ``hinge``, ``pbar_on`` and
        ``qbar_pre`` are only meaningful where ``eligible`` (or, for the
        looser rule, ``eligible_fallback``) is true -- an
        ineligible row's pooling denominators are empty and it reads
        ``hinge == margin_db``. Mask before you take a median.
        """
        if vad_target is None:
            raise ValueError(
                "AnchorInheritanceLoss requires `vad_target`; enable `vad_label` "
                "in the recipe (the energy backend is enough -- it is the grid "
                "this loss frames on)."
            )
        if self.reference_key not in batch:
            raise KeyError(
                f"AnchorInheritanceLoss requires batch['{self.reference_key}'] "
                "(= noisy_speech - clean_speech), which the voice-isolation "
                "dataloader emits for every row."
            )
        if "n_interferers" not in batch:
            raise KeyError(
                "AnchorInheritanceLoss requires batch['n_interferers']; it is "
                "emitted by VoiceIsolationCollateFunc for every row. Without it "
                "the eligibility rule cannot be evaluated and the term would "
                "silently contribute nothing."
            )

        # float32 throughout: under bf16-mixed a 400-sample inner product loses
        # the precision the -30..0 dB envelope is read at.
        enh = self._as_batch_waveform(enhanced).float()
        ref = self._as_batch_waveform(
            target.to(device=enh.device)
        ).float()
        noise = self._as_batch_waveform(
            batch[self.reference_key].to(device=enh.device)
        ).float()

        length = min(enh.shape[-1], ref.shape[-1], noise.shape[-1])
        enh, ref, noise = enh[..., :length], ref[..., :length], noise[..., :length]

        e_fr, r_fr, n_fr = self._frames(enh), self._frames(ref), self._frames(noise)
        n_rows, n_frames = e_fr.shape[0], e_fr.shape[1]
        device = e_fr.device

        w_ref = r_fr.square().sum(-1)                       # <R,R>  [B, F]
        w_noise = n_fr.square().sum(-1)                     # <N,N>
        a = (e_fr * r_fr).sum(-1) / w_ref.detach().clamp_min(self.eps_ratio)
        q = (e_fr * n_fr).sum(-1) / w_noise.detach().clamp_min(self.eps_ratio)
        p_db = self.floored_gain_db(a)
        q_db = self.floored_gain_db(q)

        vad = self._align_labels(vad_target, n_rows, n_frames, device, e_fr.dtype)
        bg = self._align_labels(
            batch.get("background_vad_target"), n_rows, n_frames, device, e_fr.dtype
        )

        idx = torch.arange(n_frames, device=device).unsqueeze(0)
        has_active = vad.sum(-1) > 0
        onset = vad.argmax(-1)                              # first active frame
        pre = (idx < onset.unsqueeze(1)) & has_active.unsqueeze(1)

        # --- frame set O: the first `onset_frames` ACTIVE frames at the onset --
        active_from_onset = vad * (idx >= onset.unsqueeze(1)).to(vad.dtype)
        rank = active_from_onset.cumsum(-1)
        mask_o = (active_from_onset > 0) & (rank <= self.onset_frames)

        # --- frame set B: quiet-for-the-user, loud-for-the-background prefix ---
        pre_noise = torch.where(pre, w_noise, torch.zeros_like(w_noise))
        pre_peak = pre_noise.amax(-1)
        window = 10.0 ** (-self.pre_energy_window_db / 10.0)
        mask_b = (
            pre
            & (vad < 0.5)
            & (w_noise >= pre_peak.unsqueeze(1) * window)
            & (pre_peak.unsqueeze(1) > 0)
        )

        tiny = torch.finfo(e_fr.dtype).tiny
        w_o = w_ref * mask_o.to(w_ref.dtype)
        w_b = (w_noise * mask_b.to(w_noise.dtype)).detach()
        pbar_on = (w_o * p_db).sum(-1) / w_o.sum(-1).clamp_min(tiny)
        qbar_pre = ((w_b * q_db.detach()).sum(-1) / w_b.sum(-1).clamp_min(tiny)).detach()
        hinge = F.relu(self.margin_db - (pbar_on - qbar_pre))

        # --- eligibility ------------------------------------------------------
        n_itf = batch["n_interferers"].to(device).reshape(-1).float()
        pre_itf = (bg * pre.to(bg.dtype)).sum(-1)
        n_active_after = active_from_onset.sum(-1)
        base = (
            (n_itf >= 1.0)
            & has_active
            & (onset >= self.min_onset_frame)
            & (n_active_after >= float(self.onset_frames))
            & (mask_b.sum(-1) > 0)
        )
        eligible = base & (pre_itf >= float(self.min_pre_interferer_frames))
        fallback = base & (pre_itf >= float(self.fallback_pre_interferer_frames))

        return {
            "hinge": hinge,
            "pbar_on": pbar_on,
            "qbar_pre": qbar_pre,
            "contrast": (pbar_on - qbar_pre),
            "eligible": eligible,
            "eligible_fallback": fallback,
            "onset_frame": onset,
            "pre_interferer_frames": pre_itf,
            "n_onset_frames": mask_o.sum(-1),
            "n_pre_frames": mask_b.sum(-1),
            "n_active_after_onset": n_active_after,
            "n_interferers": n_itf,
            "n_frames": torch.full((n_rows,), n_frames, device=device),
        }

    # ------------------------------------------------------------------ #

    def forward(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        batch: Dict,
        vad_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.row_scores(enhanced, target, batch, vad_target)
        eligible = scores["eligible"]
        if not bool(eligible.any()):
            # Graph-carrying zero, not a None and not a bare scalar: DDP needs
            # every rank's autograd graph to reach the same parameters.
            return (enhanced.sum() * 0.0).float()
        return scores["hinge"][eligible].mean()
