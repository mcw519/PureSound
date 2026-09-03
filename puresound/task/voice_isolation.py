import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise
from puresound.config.augmentation import (
    MixModeConfig,
    MixModeEntry,
    RealFarAugmentation,
    RealNearAugmentation,
)
from puresound.task.ns import (
    NoiseSuppressionCollateFunc,
    NoiseSuppressionDataset,
    RowPlan,
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
    "distance_level": 5.0,
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
      ('physical' sums without rescaling -- a near-0 dB ratio in practice, since
      per-channel RIR peak normalization removes the 1/r level cue; the distance
      information that survives is DRR / tail shape / tilt. Rescale modes draw a
      mode-specific SIR range).

    It also emits the task's scalar labels (DRR separability, distances,
    target-present flags, background-speech activity) for auxiliary heads.

    Every block is knob-gated and absent/disabled blocks never touch the RNG
    stream, so plain noise-suppression recipes regenerate bit-identically.
    """

    #: The two real-recording row types, on top of the noise-suppression set.
    AUGMENTATION_BLOCKS = {
        **NoiseSuppressionDataset.AUGMENTATION_BLOCKS,
        "augmentation_realfar_args": RealFarAugmentation,
        "augmentation_realnear_args": RealNearAugmentation,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The blocks themselves are set by the base from the registry; what is
        # task-specific is loading the manifests they name.
        self._realfar_pool = self._load_real_pool(self.augmentation_realfar_args)
        self._realnear_pool = self._load_real_pool(self.augmentation_realnear_args)
        # Only built when a block asks to stitch -- an index over 150k entries is
        # not worth the startup cost for recipes that never look at it.
        self._realfar_index = (
            self._channel_index(self._realfar_pool)
            if getattr(self.augmentation_realfar_args, "stitch_to_length", False) else {})
        self._realnear_index = (
            self._channel_index(self._realnear_pool)
            if getattr(self.augmentation_realnear_args, "stitch_to_length", False) else {})

    # ------------------------------------------------------------------ #
    # real-recording pools
    # ------------------------------------------------------------------ #
    def _load_real_pool(self, cfg: Optional[Dict]) -> List[Dict]:
        """Load a real-recording pool manifest (one JSON object per line; see
        egs/voice_isolate/scripts/build_real_recording_pool.py). Returns [] when
        the block is absent/disabled so the row hooks are no-ops."""
        if cfg is None or not cfg.used:
            return []
        manifest = cfg.pool_manifest
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
                            "mic": e.get("mic"),
                        }
                    )
        if not pool:
            raise ValueError(f"real pool manifest has no entries: {manifest}")
        return pool

    @staticmethod
    def _channel_key(entry: Dict):
        """What makes two takes the SAME recording chain and talker."""
        return (entry.get("speaker"), entry.get("room"), entry.get("mic"))

    def _channel_index(self, pool: List[Dict]) -> Dict:
        index: Dict = {}
        for entry in pool:
            index.setdefault(self._channel_key(entry), []).append(entry)
        return index

    def _open_pool_wav(self, pick: Dict, index: Dict, stitch: bool) -> torch.Tensor:
        """One pool recording, covering the row length when asked to.

        A pool take is one utterance (VOiCES: ~16 s). Longer than the row it is
        cropped downstream, which is fine; SHORTER, the aligner zero-pads -- and
        on a real-near KEEP row that makes the target half digital silence, which
        is precisely the lesson this recipe must never teach. With ``stitch`` on,
        further takes from the same (speaker, room, mic) are appended until the
        row is covered: same talker, same chain, so nothing about the row's
        identity changes, only its length.
        """
        def _open(entry):
            wav, _ = AudioIO.open(
                f_path=entry["wav_path"],
                target_lvl=self.audio_gain_normalized_to,
                resample_to=self.audio_sr,
            )
            return wav[0].reshape(1, -1)

        wav = _open(pick)
        if not stitch or wav.shape[-1] >= self.sample_length:
            return wav
        siblings = [e for e in index.get(self._channel_key(pick), ())
                    if e["wav_path"] != pick["wav_path"]]
        random.shuffle(siblings)
        parts, total = [wav], wav.shape[-1]
        for entry in siblings:
            if total >= self.sample_length:
                break
            extra = _open(entry)
            parts.append(extra)
            total += extra.shape[-1]
        return torch.cat(parts, dim=-1) if len(parts) > 1 else wav

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
        cfg = self.augmentation_realfar_args
        stitch = bool(cfg is not None and getattr(cfg, "stitch_to_length", False))
        for p in picks:
            wavs.append(self._open_pool_wav(p, self._realfar_index, stitch))
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
        if realnear_cfg is not None and realnear_cfg.used and self._realnear_pool:
            plan.use_realnear = torch.rand(1).item() < realnear_cfg.prob
        if plan.use_realnear:
            near_pick = random.choice(self._realnear_pool)
            plan.realnear_room = near_pick.get("room")
            plan.realnear_speaker = near_pick.get("speaker")
            near_wav = self._open_pool_wav(
                near_pick, self._realnear_index,
                bool(getattr(realnear_cfg, "stitch_to_length", False)),
            )
            target_speech = self.align_audio_list(
                wav_list=[near_wav],
                length=self.sample_length,
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
            if realfar_cfg is not None and realfar_cfg.used and self._realfar_pool:
                plan.use_realfar = torch.rand(1).item() < realfar_cfg.prob
            if plan.use_realfar:
                # Real-far rows decide lone-far (target-absent) by their OWN
                # prob, independently of the synthetic target-absent rate: the
                # two far-field sources need separate pressure to be tuned
                # separately.
                plan.target_absent = (
                    torch.rand(1).item() < realfar_cfg.lone_far_prob
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
        add_n_cases_cfg = self.augmentation_speech_args.add_n_cases
        if isinstance(add_n_cases_cfg, (list, tuple)):
            n_interferers = random.randint(int(add_n_cases_cfg[0]), int(add_n_cases_cfg[1]))
        else:
            n_interferers = int(add_n_cases_cfg)
        interfered_speech, realfar_meta = self._sample_realfar_interferers(
            n_interferers,
            self.audio_sr,
            prefer_room=plan.realnear_room,
            exclude_speaker=plan.realnear_speaker,
        )
        interfered_speech = self.align_audio_list(
            wav_list=interfered_speech,
            length=self.sample_length,
            padding_type="zero",
        )
        return target_speech, interfered_speech, realfar_meta

    def _turn_taking_override(self, plan) -> Optional[float]:
        # Real rows may carry their own turn-taking rate: their far-solo
        # stretches supply absolute-suppress supervision for real far voices
        # while the near foreground stays present elsewhere in the row.
        if plan.use_realnear and self.augmentation_realnear_args is not None:
            return self.augmentation_realnear_args.turn_taking_prob
        if plan.use_realfar and self.augmentation_realfar_args is not None:
            return self.augmentation_realfar_args.turn_taking_prob
        return None

    def _mix_foreground_with_interferers(
        self, fg_wav, interfered_speech, plan, fg_metadata=None, interferer_metadata=None
    ):
        # mix_mode replaces the single hard SIR draw with explicit level
        # relationships. NOTE on 'physical': it applies no additional rescale,
        # but it does NOT deliver a 1/r level law -- sources are RMS-normalized
        # at load and every RIR is peak-normalized per channel at convolution
        # time (wav_apply_rir), so the summed ratio lands near 0 dB and the
        # surviving distance cues are DRR / tail shape / spectral tilt, not
        # level. The 'distance_level' mode reinstates the level cue explicitly
        # (SIR from the inverse-distance law on the scene's actual geometry,
        # plus jitter); the other rescale modes draw a mode-specific SIR range.
        # Real-far rows skip mix_mode -- the simulated-near vs real-recorded-far
        # level ratio is not physically meaningful -- and use the hard SIR draw.
        mm_cfg = (
            self.augmentation_speech_args.mix_mode
            if self.augmentation_speech_args
            else None
        )
        if not (mm_cfg and mm_cfg.used and not plan.use_realfar):
            return super()._mix_foreground_with_interferers(fg_wav, interfered_speech, plan)
        mode = self._sample_mix_mode(mm_cfg)
        mix_mode_name = mode.name
        if mode.distance_level:
            sir = self._distance_level_sir(mode, fg_metadata, interferer_metadata)
            if sir is None:
                # geometry unavailable on this row -- fall back to the legacy
                # hard-SIR draw so the row still trains
                return super()._mix_foreground_with_interferers(
                    fg_wav, interfered_speech, plan
                )
            noisy_speech, interfered_speech = add_bg_noise(
                wav=fg_wav, noise=[interfered_speech], snr_list=[sir],
            )
            return noisy_speech[0], interfered_speech[0], "distance_level", sir
        if mode.physical:
            # No relative rescale: sum at natural post-RIR levels.
            noisy_speech = fg_wav + interfered_speech
            background_speech_reference = interfered_speech
            fg_p = float(fg_wav.pow(2).sum())
            itf_p = float(interfered_speech.pow(2).sum())
            realized_speech_sir = float(10.0 * np.log10((fg_p + 1e-8) / (itf_p + 1e-8)))
        else:
            lo, hi = mode.sir_range
            sir = float(torch.empty(1).uniform_(float(lo), float(hi)).item())
            noisy_speech, interfered_speech = add_bg_noise(
                wav=fg_wav, noise=[interfered_speech], snr_list=[sir],
            )
            noisy_speech = noisy_speech[0]
            background_speech_reference = interfered_speech[0]
            realized_speech_sir = sir
        return noisy_speech, background_speech_reference, mix_mode_name, realized_speech_sir

    def _sample_mix_mode(self, mm_cfg: MixModeConfig) -> MixModeEntry:
        """Pick one foreground/interferer mixing mode by its prob weight.

        Modes come from ``augmentation_speech.mix_mode.modes``: a ``physical``
        mode sums the post-RIR signals without rescaling (which, given the
        per-source RMS normalization and per-channel RIR peak normalization,
        yields a near-0 dB ratio -- see _mix_foreground_with_interferers); the
        others carry an explicit ``sir_range``. Probs need not sum to 1.
        """
        modes = mm_cfg.modes
        if not modes:
            return MixModeEntry(name="physical", physical=True)
        probs = [max(0.0, m.prob) for m in modes]
        total = sum(probs)
        if total <= 0.0:
            return modes[0]
        r = float(torch.empty(1).uniform_(0.0, total).item())
        acc = 0.0
        for m, pr in zip(modes, probs):
            acc += pr
            if r <= acc:
                return m
        return modes[-1]

    def _distance_level_sir(
        self, mode: MixModeEntry, fg_metadata, interferer_metadata
    ) -> Optional[float]:
        """SIR implied by the scene's geometry: 20*log10(d_itf / d_fg) + jitter.

        Reinstates the distance level cue the pipeline otherwise removes (see
        _mix_foreground_with_interferers): a talker at 3 m really is ~15 dB
        quieter at the mic than one at 0.5 m. Uses the NEAREST interferer (the
        loudest under the 1/r law). Returns None when either distance is
        unknown so the caller can fall back.
        """

        def _dist(meta) -> Optional[float]:
            if not meta:
                return None
            value = meta.get("source_receiver_distance")
            return float(value) if value is not None else None

        d_fg = _dist(fg_metadata)
        itf = [
            _dist(m) for m in (interferer_metadata or []) if _dist(m) is not None
        ]
        if d_fg is None or d_fg <= 0 or not itf:
            return None
        d_itf = min(itf)
        sir = 20.0 * float(np.log10(d_itf / max(d_fg, 1e-3)))
        lo, hi = mode.jitter_db
        sir += float(torch.empty(1).uniform_(float(lo), float(hi)).item())
        return sir

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
        super()._emit_task_metadata(
            sample,
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
