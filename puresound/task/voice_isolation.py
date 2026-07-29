import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise
from puresound.task.ns import (
    NoiseSuppressionCollateFunc,
    NoiseSuppressionDataset,
    RowPlan,
    if_none_else,
)


VOICE_ISOLATION_SCALAR_KEYS = (
    "foreground_distance",
    "foreground_drr",
    "nearest_interferer_distance",
    "strongest_interferer_drr",
    "drr_gap",
    "rt60",
    "n_interferers",
    "target_absent",
    "target_present",
    "has_background_speech",
    # Parent composition + mixing diagnostics.
    "near_count",
    "far_count",
    "mix_mode",
    "realized_speech_sir",
    "noise_snr",
    "overlap_fraction",
    # Phase-1 boundary experiment: 1.0 when the row used the conversational
    # turn-taking gating (long alternating near/far turns) instead of the
    # per-frame Bernoulli overlap gating.
    "turn_taking",
)

# mix_mode is emitted as a float code so it collates with the other scalars.
MIX_MODE_CODES = {
    "none": 0.0,
    "legacy": 1.0,
    "physical": 2.0,
    "moderate": 3.0,
    "counter_level": 4.0,
}


@dataclass
class VoiceIsolationRowPlan(RowPlan):
    """RowPlan extended with the near/far row types this task trains on."""

    use_realnear: bool = False
    use_realfar: bool = False
    realnear_room: Optional[str] = None
    realnear_speaker: Optional[str] = None
    realnear_fg_metadata: Optional[dict] = None


class VoiceIsolationDataset(NoiseSuppressionDataset):
    """Near-field foreground voice isolation dataset.

    Reuses the synthesis skeleton from NoiseSuppressionDataset and adds, via the
    row-type hooks, everything specific to the near/far decision:

    * **real-far interferer rows** -- the far channel is a finished
      loudspeaker->air->mic recording drawn from a pool manifest and inserted
      with no RIR applied. A convolved far channel only carries the linear
      time-invariant part of a capture chain; a real recording also carries its
      level, spectral tilt, transducer non-linearity and noise floor.
    * **real-near keep rows** -- the foreground is a genuine close-mic recording
      (target = itself), mixed with the same interferers and noise as any other
      row, so real captured speech sits on the KEEP side of the same mixtures
      whose far side is real and the keep/suppress boundary stays on proximity
      cues instead of capture-chain identity.
    * **row-level turn-taking rates** -- real rows may carry their own
      turn-taking probability, so far-solo stretches supervise the absolute
      "lone far voice = suppress" decision while near speech stays elsewhere in
      the row.
    * **mix_mode** -- explicit foreground/interferer level relationships
      ('physical' preserves the natural post-RIR ratio so the DRR/proximity cue
      survives; rescale modes draw a mode-specific SIR range).

    It also emits the task's scalar labels (DRR separability, distances,
    target-present flags, background-speech activity) for auxiliary heads.

    Every block is knob-gated and absent/disabled blocks never touch the RNG
    stream, so plain noise-suppression recipes regenerate bit-identically.
    """

    def __init__(
        self,
        *args,
        augmentation_realfar_args: Optional[Dict] = None,
        augmentation_realnear_args: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.augmentation_realfar_args = augmentation_realfar_args
        self._realfar_pool = self._load_real_pool(augmentation_realfar_args)
        self.augmentation_realnear_args = augmentation_realnear_args
        self._realnear_pool = self._load_real_pool(augmentation_realnear_args)

    # ------------------------------------------------------------------ #
    # real-recording pools
    # ------------------------------------------------------------------ #
    def _load_real_pool(self, cfg: Optional[Dict]) -> List[Dict]:
        """Load a real-recording pool manifest (one JSON object per line; see
        egs/voice_isolate/scripts/build_real_recording_pool.py). Returns [] when
        the block is absent/disabled so the row hooks are no-ops."""
        if not (cfg and cfg.get("used")):
            return []
        manifest = cfg["pool_manifest"]
        pool: List[Dict] = []
        with open(manifest, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    pool.append(
                        {
                            "wav_path": e["wav_path"],
                            "distance_m": e.get("distance_m"),
                            "room": e.get("room"),
                            "speaker": e.get("speaker"),
                        }
                    )
        if not pool:
            raise ValueError(f"real pool manifest has no entries: {manifest}")
        return pool

    def _sample_realfar_interferers(
        self,
        n: int,
        sr: int,
        prefer_room: Optional[str] = None,
        exclude_speaker: Optional[str] = None,
    ) -> Tuple[List[torch.Tensor], List[dict]]:
        """Draw ``n`` finished far-field recordings from the real-far pool.

        No RIR is applied -- these already carry the full recording chain.
        Loudness is normalized like every other source (the SIR mixing downstream
        sets the level anyway); the measured distance rides along as metadata so
        eval can bucket by it, tagged origin=real.

        ``prefer_room``: draw from the same room as the near foreground when it
        has enough entries, so near and far differ mainly in distance.
        ``exclude_speaker`` keeps the far interferer from being the same speaker
        as the near foreground."""
        candidates = self._realfar_pool
        if exclude_speaker is not None:
            filtered = [p for p in candidates if p.get("speaker") != exclude_speaker]
            if filtered:
                candidates = filtered
        if prefer_room is not None:
            same_room = [p for p in candidates if p.get("room") == prefer_room]
            if len(same_room) >= n:
                candidates = same_room
        n = max(1, min(int(n), len(candidates)))
        picks = random.sample(candidates, k=n)
        wavs: List[torch.Tensor] = []
        metas: List[dict] = []
        for p in picks:
            wav, _ = AudioIO.open(
                f_path=p["wav_path"],
                target_lvl=self.audio_gain_normalized_to,
                resample_to=sr,
            )
            wavs.append(wav[0].reshape(1, -1))
            metas.append(
                {"source_receiver_distance": p.get("distance_m"), "origin": "real"}
            )
        return wavs, metas

    # ------------------------------------------------------------------ #
    # row-type hooks (see NoiseSuppressionDataset for the contract)
    # ------------------------------------------------------------------ #
    def _plan_row(
        self, target_speech: torch.Tensor
    ) -> Tuple[VoiceIsolationRowPlan, torch.Tensor]:
        plan = VoiceIsolationRowPlan()

        # Real-NEAR keep row? Decide FIRST (guarded: an absent/disabled block
        # never touches the RNG stream). On these rows the foreground becomes a
        # genuine close-mic recording and the target is that recording itself.
        # The corpus utterance drawn upstream is discarded; batching/speaker
        # semantics stay untouched.
        realnear_cfg = self.augmentation_realnear_args
        if realnear_cfg is not None and realnear_cfg.get("used", False) and self._realnear_pool:
            plan.use_realnear = torch.rand(1).item() < float(realnear_cfg.get("prob", 0.0))
        if plan.use_realnear:
            near_pick = random.choice(self._realnear_pool)
            plan.realnear_room = near_pick.get("room")
            plan.realnear_speaker = near_pick.get("speaker")
            near_wav, _ = AudioIO.open(
                f_path=near_pick["wav_path"],
                target_lvl=self.audio_gain_normalized_to,
                resample_to=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            target_speech = self.align_audio_list(
                wav_list=[near_wav[0].reshape(1, -1)],
                length=if_none_else(
                    self.training_sample_length,
                    int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                ),
            )[0]
            plan.realnear_fg_metadata = {
                "source_receiver_distance": near_pick.get("distance_m"),
                "origin": "real",
            }
            plan.skip_whole_mix_reverb = True

        # Real-far row? Decide next so it can own the target-absent draw.
        # Guarded so an absent/disabled block never touches the RNG stream.
        realfar_cfg = self.augmentation_realfar_args
        if plan.use_realnear:
            # Keep row: interferers (when the speech-aug gate fires) come from
            # the real-far pool; the target is always present -- lone-far never
            # applies.
            plan.use_realfar = bool(self._realfar_pool)
        else:
            if realfar_cfg is not None and realfar_cfg.get("used", False) and self._realfar_pool:
                plan.use_realfar = torch.rand(1).item() < float(realfar_cfg.get("prob", 0.0))
            if plan.use_realfar:
                # Real-far rows decide lone-far (target-absent) by their OWN
                # prob, independently of the synthetic target-absent rate: the
                # two far-field sources need separate pressure to be tuned
                # separately.
                plan.target_absent = (
                    torch.rand(1).item() < float(realfar_cfg.get("lone_far_prob", 0.0))
                )
                plan.force_interferer = plan.target_absent
            else:
                base_plan, target_speech = super()._plan_row(target_speech)
                plan.target_absent = base_plan.target_absent
                plan.force_interferer = base_plan.force_interferer
        plan.force_speech_interferers = plan.use_realfar
        return plan, target_speech

    def _prepare_foreground(self, target_speech, plan):
        if plan.use_realnear:
            # The real near recording already carries its full end-to-end
            # channel; no synthetic room is simulated on these rows.
            return False, None, plan.realnear_fg_metadata, target_speech.clone(), target_speech
        return super()._prepare_foreground(target_speech, plan)

    def _sample_interferers(
        self, target_speaker, target_speech, room_scene, source_level_reverb, plan
    ):
        if not plan.use_realfar:
            return super()._sample_interferers(
                target_speaker, target_speech, room_scene, source_level_reverb, plan
            )
        # Real far interferers: finished loudspeaker->air->mic recordings
        # inserted directly, with no RIR applied. The near foreground still
        # carries its channel from the foreground-preparation step.
        add_n_cases_cfg = self.augmentation_speech_args["add_n_cases"]
        if isinstance(add_n_cases_cfg, (list, tuple)):
            n_interferers = random.randint(int(add_n_cases_cfg[0]), int(add_n_cases_cfg[1]))
        else:
            n_interferers = int(add_n_cases_cfg)
        interfered_speech, realfar_meta = self._sample_realfar_interferers(
            n_interferers,
            if_none_else(self.target_sr, self.ori_audio_sr),
            prefer_room=plan.realnear_room,
            exclude_speaker=plan.realnear_speaker,
        )
        interfered_speech = self.align_audio_list(
            wav_list=interfered_speech,
            length=if_none_else(
                self.training_sample_length,
                int(self.ori_audio_sr * self.training_sample_length_in_seconds),
            ),
            padding_type="zero",
        )
        return target_speech, interfered_speech, realfar_meta

    def _turn_taking_override(self, plan) -> Optional[float]:
        # Real rows may carry their own turn-taking rate: their far-solo
        # stretches supply absolute-suppress supervision for real far voices
        # while the near foreground stays present elsewhere in the row.
        if plan.use_realnear and self.augmentation_realnear_args is not None:
            return self.augmentation_realnear_args.get("turn_taking_prob")
        if plan.use_realfar and self.augmentation_realfar_args is not None:
            return self.augmentation_realfar_args.get("turn_taking_prob")
        return None

    def _mix_foreground_with_interferers(self, fg_wav, interfered_speech, plan):
        # mix_mode replaces the single hard SIR draw with explicit level
        # relationships: 'physical' keeps the natural post-RIR ratio so the
        # real DRR/proximity cue survives; the rescale modes draw a
        # mode-specific SIR range. Real-far rows skip it -- the synthetic-near
        # vs real-recorded-far level ratio is not physically meaningful, so
        # they use the controlled hard SIR draw instead.
        mm_cfg = (self.augmentation_speech_args or {}).get("mix_mode")
        if not (mm_cfg and mm_cfg.get("used", False) and not plan.use_realfar):
            return super()._mix_foreground_with_interferers(fg_wav, interfered_speech, plan)
        mode = self._sample_mix_mode(mm_cfg)
        mix_mode_name = mode.get("name", "physical")
        if mode.get("physical", False):
            # No relative rescale: sum at natural post-RIR levels.
            noisy_speech = fg_wav + interfered_speech
            background_speech_reference = interfered_speech
            fg_p = float(fg_wav.pow(2).sum())
            itf_p = float(interfered_speech.pow(2).sum())
            realized_speech_sir = float(10.0 * np.log10((fg_p + 1e-8) / (itf_p + 1e-8)))
        else:
            lo, hi = mode["sir_range"]
            sir = float(torch.empty(1).uniform_(float(lo), float(hi)).item())
            noisy_speech, interfered_speech = add_bg_noise(
                wav=fg_wav, noise=[interfered_speech], snr_list=[sir],
            )
            noisy_speech = noisy_speech[0]
            background_speech_reference = interfered_speech[0]
            realized_speech_sir = sir
        return noisy_speech, background_speech_reference, mix_mode_name, realized_speech_sir

    def _sample_mix_mode(self, mm_cfg: Dict) -> Dict:
        """Pick one foreground/interferer mixing mode by its prob weight.

        Modes come from ``augmentation_speech.mix_mode.modes``: a ``physical``
        mode keeps the natural post-RIR level ratio, the others carry an
        explicit ``sir_range``. Probs need not sum to 1.
        """
        modes = mm_cfg.get("modes") or []
        if not modes:
            return {"name": "physical", "physical": True}
        probs = [max(0.0, float(m.get("prob", 0.0))) for m in modes]
        total = sum(probs)
        if total <= 0.0:
            return dict(modes[0])
        r = float(torch.empty(1).uniform_(0.0, total).item())
        acc = 0.0
        for m, pr in zip(modes, probs):
            acc += pr
            if r <= acc:
                return dict(m)
        return dict(modes[-1])

    def _emit_task_metadata(
        self,
        sample: Dict,
        *,
        foreground_metadata: Optional[dict],
        interferer_metadata: List[dict],
        target_absent: bool,
        background_speech_reference: Optional[torch.Tensor],
        near_count: int = 1,
        far_count: int = 0,
        mix_mode: str = "none",
        realized_speech_sir: float = float("nan"),
        noise_snr: float = float("nan"),
        overlap_fraction: float = float("nan"),
        turn_taking: float = 0.0,
    ) -> None:
        sample.update(
            self._build_voice_isolation_metadata(
                foreground_metadata=foreground_metadata,
                interferer_metadata=interferer_metadata,
                target_absent=target_absent,
                background_speech_reference=background_speech_reference,
                near_count=near_count,
                far_count=far_count,
                mix_mode=mix_mode,
                realized_speech_sir=realized_speech_sir,
                noise_snr=noise_snr,
                overlap_fraction=overlap_fraction,
                turn_taking=turn_taking,
            )
        )

    def _build_voice_isolation_metadata(
        self,
        foreground_metadata: Optional[dict],
        interferer_metadata: List[dict],
        target_absent: bool,
        background_speech_reference: Optional[torch.Tensor],
        near_count: int = 1,
        far_count: int = 0,
        mix_mode: str = "none",
        realized_speech_sir: float = float("nan"),
        noise_snr: float = float("nan"),
        overlap_fraction: float = float("nan"),
        turn_taking: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Create scalar labels useful for distance-cued voice isolation.

        These labels are generated from simulation metadata only. They are not
        inference inputs; auxiliary heads can use them as supervision so the
        model learns explicit distance/DRR confidence instead of relying only on
        the enhancement loss.
        """
        nan = float("nan")

        def _meta_float(meta: Optional[dict], key: str) -> float:
            if not meta:
                return nan
            value = meta.get(key)
            return float(value) if value is not None else nan

        fg_dist = _meta_float(foreground_metadata, "source_receiver_distance")
        fg_drr = _meta_float(foreground_metadata, "drr_db")
        rt60 = _meta_float(foreground_metadata, "rt60")
        itf_distances = [
            float(m["source_receiver_distance"])
            for m in interferer_metadata
            if m.get("source_receiver_distance") is not None
        ]
        itf_drrs = [
            float(m["drr_db"]) for m in interferer_metadata if m.get("drr_db") is not None
        ]
        min_itf_dist = min(itf_distances) if itf_distances else nan
        max_itf_drr = max(itf_drrs) if itf_drrs else nan
        drr_gap = fg_drr - max_itf_drr if not np.isnan(fg_drr) and not np.isnan(max_itf_drr) else nan
        has_background_speech = (
            background_speech_reference is not None
            and background_speech_reference.abs().amax().item() > 0.0
        )
        target_present = not target_absent

        values = {
            "foreground_distance": fg_dist,
            "foreground_drr": fg_drr,
            "nearest_interferer_distance": min_itf_dist,
            "strongest_interferer_drr": max_itf_drr,
            "drr_gap": drr_gap,
            "rt60": rt60,
            "n_interferers": float(len(interferer_metadata)),
            "target_absent": float(target_absent),
            "target_present": float(target_present),
            "has_background_speech": float(has_background_speech),
            "near_count": float(near_count),
            "far_count": float(far_count),
            "mix_mode": MIX_MODE_CODES.get(mix_mode, float("nan")),
            "realized_speech_sir": float(realized_speech_sir),
            "noise_snr": float(noise_snr),
            "overlap_fraction": float(overlap_fraction),
            "turn_taking": float(turn_taking),
        }
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in values.items()
        }


class VoiceIsolationCollateFunc(NoiseSuppressionCollateFunc):
    """Collate waveform batches plus voice-isolation scalar/frame labels."""

    def __call__(self, batch: Dict):
        out = super().__call__(batch)

        for key in VOICE_ISOLATION_SCALAR_KEYS:
            values = [b[key].view(-1) for b in batch if key in b]
            if values:
                out[key] = torch.cat(values, dim=0)

        # Far-parent target waveform (P1): pad like the other waveforms so an
        # optional far decoder loss can read batch["far_target"].
        if any("far_target" in b for b in batch):
            far = [
                (b["far_target"] if "far_target" in b else torch.zeros_like(b["noisy_speech"])).squeeze()
                for b in batch
            ]
            out["far_target"] = pad_sequence(far, batch_first=True)

        has_background_vad = any("background_vad_target" in b for b in batch)
        has_background_vad_ref = any("background_vad_reference" in b for b in batch)
        background_vad = []
        background_vad_ref = []
        for b in batch:
            if has_background_vad:
                if "background_vad_target" not in b:
                    template = b.get("vad_target")
                    if template is None:
                        continue
                    background_vad.append(torch.zeros_like(template).squeeze())
                else:
                    background_vad.append(b["background_vad_target"].squeeze())
            if has_background_vad_ref:
                if "background_vad_reference" not in b:
                    background_vad_ref.append(torch.zeros_like(b["noisy_speech"]).squeeze())
                else:
                    background_vad_ref.append(b["background_vad_reference"].squeeze())

        if background_vad:
            out["background_vad_target"] = pad_sequence(background_vad, batch_first=True)
        if background_vad_ref:
            out["background_vad_reference"] = pad_sequence(
                background_vad_ref,
                batch_first=True,
            )
        return out
