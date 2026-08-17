import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.dsp import wav_resampling
from puresound.audio.noise import add_bg_noise
from puresound.config.augmentation import (
    CodecAugmentation,
    OverlapControlConfig,
    PacketLossAugmentation,
    TargetAbsentAugmentation,
)
from puresound.dataset.dynamic_base import (
    AugmentationArg,
    DynamicBaseDataset,
    as_block,
)


RIR_PROVENANCE_KEYS = (
    "rir_release_id",
    "rir_release_sha256",
    "rir_recipe_id",
    "rir_variant_id",
    "rir_split",
    "rir_origin",
    "rir_renderer_profile_id",
    "rir_production_certificate_sha256",
    "rir_interferer_variant_ids",
)


@dataclass
class RowPlan:
    """Per-item decisions a task subclass makes before synthesis starts.

    The base dataset only decides ``target_absent`` (and whether that forces an
    interferer). Subclasses return a subclass of this plan from ``_plan_row`` to
    drive their own row types through the shared synthesis skeleton without the
    skeleton knowing about them:

    * ``force_speech_interferers`` makes the interferer block fire without
      consuming the speech-augmentation probability draw.
    * ``skip_whole_mix_reverb`` keeps the whole-mix RIR off rows whose channel
      must stay exactly as the foreground provided it.
    """

    target_absent: bool = False
    force_interferer: bool = False
    force_speech_interferers: bool = False
    skip_whole_mix_reverb: bool = False


class NoiseSuppressionDataset(DynamicBaseDataset):
    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        audio_gain_normalized_to: Optional[int] = None,
        augmentation_speech_args: AugmentationArg = None,
        augmentation_noise_args: AugmentationArg = None,
        augmentation_reverb_args: AugmentationArg = None,
        augmentation_speed_args: AugmentationArg = None,
        augmentation_ir_response_args: AugmentationArg = None,
        augmentation_src_args: AugmentationArg = None,
        augmentation_hpf_args: AugmentationArg = None,
        augmentation_volume_args: AugmentationArg = None,
        augmentation_codec_args: AugmentationArg = None,
        augmentation_packet_loss_args: AugmentationArg = None,
        augmentation_target_absent_args: AugmentationArg = None,
        vad_label_args: AugmentationArg = None,
        dataset_role: str = "train",
        pipeline_role: str | None = None,
    ):
        super().__init__(
            metafile_path=metafile_path,
            min_utt_length_in_seconds=min_utt_length_in_seconds,
            min_utts_in_each_speaker=min_utts_in_each_speaker,
            target_sr=target_sr,
            training_sample_length_in_seconds=training_sample_length_in_seconds,
            audio_gain_normalized_to=audio_gain_normalized_to,
            augmentation_speech_args=augmentation_speech_args,
            augmentation_noise_args=augmentation_noise_args,
            augmentation_reverb_args=augmentation_reverb_args,
            augmentation_speed_args=augmentation_speed_args,
            augmentation_ir_response_args=augmentation_ir_response_args,
            augmentation_src_args=augmentation_src_args,
            augmentation_hpf_args=augmentation_hpf_args,
            augmentation_volume_args=augmentation_volume_args,
            vad_label_args=vad_label_args,
            dataset_role=dataset_role,
            pipeline_role=pipeline_role,
        )
        self.augmentation_codec_args = as_block(
            augmentation_codec_args, CodecAugmentation
        )
        self.augmentation_packet_loss_args = as_block(
            augmentation_packet_loss_args, PacketLossAugmentation
        )
        self.augmentation_target_absent_args = as_block(
            augmentation_target_absent_args, TargetAbsentAugmentation
        )
        # Fail fast instead of silently ignoring a task-specific block: mix_mode
        # describes near/far level relationships, which only the voice-isolation
        # dataset implements.
        mm_cfg = (
            self.augmentation_speech_args.mix_mode
            if self.augmentation_speech_args
            else None
        )
        if type(self) is NoiseSuppressionDataset and mm_cfg and mm_cfg.used:
            raise ValueError(
                "augmentation_speech.mix_mode needs dataset.task: voice_isolation"
            )

    def _build_synthetic_interferers(
        self,
        target_speaker,
        target_speech: torch.Tensor,
        room_scene: Optional[dict],
        source_level_reverb: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[dict]]:
        """Synthetic interferer path (extracted verbatim from __getitem__): sample
        clean speakers, optionally color some as media, then reverberate with the
        room's far-field RIR channels (source-level) or align (non-source-level).
        Returns the (possibly re-aligned) target, the interferer list, and the RIR
        metadata. This is the clean-speech-convolved-with-RIR far channel the
        real-far branch replaces with genuine end-to-end recordings."""
        interferer_rir_metadata: List[dict] = []
        interfered_speech: List[torch.Tensor] = []
        # Samples speakers from overall speaker pool
        if self.target_sr is not None:
            spk_pool = deepcopy(self.total_spks)
        # Samples speakers from same SR conditions
        else:
            spk_pool = deepcopy(list(self.sr_meta[self.ori_audio_sr].keys()))

        spk_pool = set(spk_pool)
        spk_pool.remove(target_speaker)
        add_n_cases_cfg = self.augmentation_speech_args.add_n_cases
        if isinstance(add_n_cases_cfg, (list, tuple)):
            lo, hi = int(add_n_cases_cfg[0]), int(add_n_cases_cfg[1])
            n_interferers = random.randint(lo, hi)
        else:
            n_interferers = int(add_n_cases_cfg)
        if n_interferers > len(spk_pool):
            n_interferers = len(spk_pool)
        if n_interferers < 1:
            n_interferers = 1
        interference_spk_list = random.sample(sorted(spk_pool), k=n_interferers)
        interference_sr = None if self.target_sr is not None else self.ori_audio_sr
        for spk in interference_spk_list:
            _speech, _sr, _ = self.choose_an_utterance_by_speaker_name(
                target_speaker_name=spk,
                select_channel=0,
                select_with_sr_as_key=interference_sr,
            )
            # Interfered speech has to be same sample rate of target speech.
            # BOUND the redraw: a speaker with no utterance at ori_audio_sr
            # would otherwise loop forever -- a stalled DataLoader worker then
            # deadlocks DDP. Give up after a few tries and skip this interferer
            # (the interferer count is already random) rather than hang.
            sr_retry = 0
            while _sr != self.ori_audio_sr and sr_retry < 5:
                sr_retry += 1
                _speech, _sr, _ = self.choose_an_utterance_by_speaker_name(
                    target_speaker_name=spk,
                    select_channel=0,
                    select_with_sr_as_key=interference_sr,
                )
            if _sr != self.ori_audio_sr:
                continue
            interfered_speech.append(_speech)

        # Some interferers become "media-device" speech (TV / loudspeaker
        # playback): band-limited + lightly compressed before its RIR. With a
        # pre-generated bank the "media" role draws from the same far pool as a
        # plain interferer; only the on-the-fly room simulator places media
        # sources wall-adjacent.
        media_cfg = self.augmentation_speech_args.media_voice
        media_flags = [
            bool(
                media_cfg
                and media_cfg.used
                and torch.rand(1).item() < media_cfg.prob
            )
            for _ in interfered_speech
        ]
        if any(media_flags):
            hp_lo, hp_hi = media_cfg.hp_cutoff_range
            lp_lo, lp_hi = media_cfg.lp_cutoff_range
            cp_lo, cp_hi = media_cfg.compress_power_range
            for idx, is_media in enumerate(media_flags):
                if not is_media:
                    continue
                interfered_speech[idx], _ = self.augmentor.apply_media_coloring(
                    wav=interfered_speech[idx],
                    sr=self.audio_sr,
                    hp_cutoff=torch.empty(1).uniform_(hp_lo, hp_hi).item(),
                    lp_cutoff=torch.empty(1).uniform_(lp_lo, lp_hi).item(),
                    compress_power=torch.empty(1).uniform_(cp_lo, cp_hi).item(),
                )

        if source_level_reverb:
            interfered_speech = self.align_audio_list(
                wav_list=interfered_speech,
                length=self.sample_length,
                padding_type="zero",
            )
            reverb_interferers = []
            for speech, is_media in zip(interfered_speech, media_flags):
                reverbed = self.apply_source_level_interferer_reverb(
                    wav=speech,
                    sr=self.audio_sr,
                    room_scene=room_scene,
                    source_role="media" if is_media else "interferer",
                )
                reverb_interferers.append(reverbed.wav)
                if reverbed.metadata is not None:
                    interferer_rir_metadata.append(dict(reverbed.metadata))
            interfered_speech = reverb_interferers
        else:
            # Aligned and Mixing
            clips_wav = [target_speech] + interfered_speech
            clips_wav = self.align_audio_list(
                wav_list=clips_wav,
                length=self.sample_length,
                padding_type="zero",
            )
            target_speech = clips_wav[0]
            interfered_speech = clips_wav[1:]

        return target_speech, interfered_speech, interferer_rir_metadata

    # ------------------------------------------------------------------ #
    # Row-type hooks. The synthesis skeleton in __getitem__ calls these at
    # every point where a task subclass may substitute its own row types.
    # Each base implementation IS the generic noise-suppression behaviour,
    # and none of them touches the RNG stream beyond what the equivalent
    # inline code always drew, so recipes regenerate bit-identically.
    # ------------------------------------------------------------------ #
    def _plan_row(self, target_speech: torch.Tensor) -> Tuple[RowPlan, torch.Tensor]:
        """Decide the row type; may replace the foreground waveform."""
        cfg = self.augmentation_target_absent_args
        target_absent = (
            cfg is not None
            and cfg.used
            and torch.rand(1).item() < cfg.prob
        )
        force_interferer = bool(target_absent and cfg is not None and cfg.force_interferer)
        return (
            RowPlan(target_absent=target_absent, force_interferer=force_interferer),
            target_speech,
        )

    def _prepare_foreground(
        self, target_speech: torch.Tensor, plan: RowPlan
    ) -> Tuple[bool, Optional[dict], Optional[dict], torch.Tensor, torch.Tensor]:
        """Give the foreground its channel; returns
        (source_level_reverb, room_scene, fg_metadata, noisy_speech, target_speech)."""
        source_level_reverb = self.should_apply_source_level_reverb()
        room_scene = self.augmentor.sample_room_scene() if source_level_reverb else None
        fg_rir_metadata = None
        if source_level_reverb:
            noisy_speech, target_speech, fg_rir_metadata = (
                self.apply_source_level_target_reverb(
                    wav=target_speech,
                    sr=self.audio_sr,
                    room_scene=room_scene,
                )
            )
        else:
            noisy_speech = target_speech.clone()
        return source_level_reverb, room_scene, fg_rir_metadata, noisy_speech, target_speech

    def _sample_interferers(
        self,
        target_speaker,
        target_speech: torch.Tensor,
        room_scene: Optional[dict],
        source_level_reverb: bool,
        plan: RowPlan,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[dict]]:
        """Source the interfering speech; returns (target, interferers, metadata)."""
        return self._build_synthetic_interferers(
            target_speaker=target_speaker,
            target_speech=target_speech,
            room_scene=room_scene,
            source_level_reverb=source_level_reverb,
        )

    def _turn_taking_override(self, plan: RowPlan) -> Optional[float]:
        """Row-level turn-taking rate; None = use the overlap_control value."""
        return None

    def _mix_foreground_with_interferers(
        self,
        fg_wav: torch.Tensor,
        interfered_speech: torch.Tensor,
        plan: RowPlan,
        fg_metadata: Optional[dict] = None,
        interferer_metadata: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, float]:
        """Combine foreground and summed interferers; returns
        (noisy_speech, background_speech_reference, mix_mode_name, realized_sir)."""
        sir = (
            torch.FloatTensor(1)
            .uniform_(
                self.augmentation_speech_args.snr_range[0],
                self.augmentation_speech_args.snr_range[1],
            )
            .item()
        )

        # Mixing with SIR
        noisy_speech, interfered_speech = add_bg_noise(
            wav=fg_wav,
            noise=[interfered_speech],
            snr_list=[sir],
        )
        return noisy_speech[0], interfered_speech[0], "legacy", sir

    def __getitem__(self, target_speaker: Tuple[str, int | None] | Tuple[str, int | None, int]):
        # A 3-tuple carries a per-item seed from a seeded SpeakerSampler
        # (deterministic validation): reseed every RNG the synthesis path uses
        # (utterance pick / room sim / augmentation draws) so the same item
        # regenerates bit-exact across epochs, runs and worker layouts.
        if len(target_speaker) == 3:
            target_speaker, batch_sr, item_seed = target_speaker
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
        else:
            target_speaker, batch_sr = target_speaker
        batch_sr = int(batch_sr) if batch_sr is not None else batch_sr
        target_speech, self.ori_audio_sr, (_, _) = (
            self.choose_an_utterance_by_speaker_name(
                target_speaker_name=target_speaker, select_channel=0, select_with_sr_as_key=batch_sr
            )
        )
        # Snipts first
        target_speech = self.align_audio_list(
            wav_list=[target_speech],
            length=self.sample_length,
        )[0]
        # Decide once per sample what row this is (target-absent and any
        # task-specific row type). The plan hook may replace the foreground
        # entirely; concrete target-absent gating happens after the
        # augmentation_speech block so SIR sampling and overlap-control still
        # use the real target as a reference; we subtract target_in_mix at
        # the end.
        plan, target_speech = self._plan_row(target_speech)
        target_absent = plan.target_absent
        force_interferer = plan.force_interferer

        interferer_rir_metadata: List[dict] = []
        background_speech_reference = None

        source_level_reverb, room_scene, fg_rir_metadata, noisy_speech, target_speech = (
            self._prepare_foreground(target_speech, plan)
        )

        # Snapshot the target's contribution to the mixture; subtracted later
        # for target-absent samples so noisy_speech keeps only interferer+noise.
        target_in_mix = noisy_speech.clone()

        # Diagnostic scalars emitted as per-sample metadata, so eval can bucket
        # samples by geometry. Defaults cover the no-interferer / no-noise paths.
        far_count = 0
        mix_mode_name = "none"
        realized_speech_sir = float("nan")
        noise_snr = float("nan")
        self._last_overlap_fraction = float("nan")
        self._last_turn_taking = 0.0

        # Add interference speech from other speakers
        interfered_speech = []
        if (
            self.augmentation_speech_args
            and self.augmentation_speech_args.used
            and (
                force_interferer
                or plan.force_speech_interferers
                or torch.rand(1) < self.augmentation_speech_args.prob
            )
        ):
            target_speech, interfered_speech, itf_meta = self._sample_interferers(
                target_speaker=target_speaker,
                target_speech=target_speech,
                room_scene=room_scene,
                source_level_reverb=source_level_reverb,
                plan=plan,
            )
            interferer_rir_metadata.extend(itf_meta)

            tt_override = self._turn_taking_override(plan)
            target_speech, noisy_speech, interfered_speech = self._apply_overlap_gating(
                target_speech=target_speech,
                interferers=interfered_speech,
                sr=self.audio_sr,
                target_mix=noisy_speech,
                allow_turn_taking=not target_absent,
                turn_taking_prob_override=tt_override,
            )

            far_count = len(interfered_speech)
            interfered_speech = (
                torch.cat(interfered_speech, dim=0).sum(dim=0).reshape(1, -1)
            )

            # Foreground-vs-interferer mixing (task subclasses may add richer
            # level relationships; the base draw is one hard SIR from
            # augmentation_speech.snr_range).
            fg_wav = noisy_speech if source_level_reverb else target_speech
            noisy_speech, background_speech_reference, mix_mode_name, realized_speech_sir = (
                self._mix_foreground_with_interferers(
                    fg_wav,
                    interfered_speech,
                    plan,
                    fg_metadata=fg_rir_metadata,
                    interferer_metadata=interferer_rir_metadata,
                )
            )

            # Treating all speech clips as target speech
            if self.augmentation_speech_args.is_target:
                target_speech = noisy_speech.clone()

        # Target-absent gating: strip the foreground contribution from the
        # mixture and zero the reference so VAD / SDR / consistency_noise
        # downstream all reflect "no near-field speaker". Done before any
        # rescaling so the leftover mixture levels stay self-consistent.
        if target_absent:
            noisy_speech = noisy_speech - target_in_mix
            target_speech = torch.zeros_like(target_speech)

        # Residual playback echo: the device's own loudspeaker leaks into the
        # mic. Modeled as another utterance through a channel of the same room,
        # added at the residual level an upstream AEC would leave
        # (erle_db_range below the mixture). Never in the target; requires
        # source-level reverb for its own RIR. NOTE: with a pre-generated bank
        # the channel comes from the FAR pool (distance_range_override cannot
        # be honored there; the bank picks the far channel closest to the
        # requested band); a true near-field echo channel needs the on-the-fly
        # simulator.
        echo_cfg = (
            self.augmentation_speech_args.echo_playback
            if self.augmentation_speech_args
            else None
        )
        if (
            echo_cfg
            and echo_cfg.used
            and source_level_reverb
            and torch.rand(1).item() < echo_cfg.prob
        ):
            if self.target_sr is not None:
                echo_pool = set(self.total_spks)
            else:
                echo_pool = set(self.sr_meta[self.ori_audio_sr].keys())
            echo_pool.discard(target_speaker)
            echo_spk = random.choice(sorted(echo_pool))
            echo_speech, _, _ = self.choose_an_utterance_by_speaker_name(
                target_speaker_name=echo_spk,
                select_channel=0,
                select_with_sr_as_key=None if self.target_sr is not None else self.ori_audio_sr,
            )
            echo_speech = self.align_audio_list(
                wav_list=[echo_speech],
                length=self.sample_length,
                padding_type="zero",
            )[0]
            # Echo is never part of the target and never reported in the RIR
            # lineage, so its channel metadata is deliberately dropped here.
            echo_speech = self.apply_source_level_interferer_reverb(
                wav=echo_speech,
                sr=self.audio_sr,
                room_scene=room_scene,
                distance_range_override=list(echo_cfg.distance_range),
            ).wav
            erle_lo, erle_hi = echo_cfg.erle_db_range
            erle_db = torch.empty(1).uniform_(float(erle_lo), float(erle_hi)).item()
            noisy_speech, _ = add_bg_noise(
                wav=noisy_speech, noise=[echo_speech], snr_list=[erle_db]
            )
            noisy_speech = noisy_speech[0]

        # Avoiding clipping issue
        [noisy_speech, target_speech] = self.avoid_audio_clipping(
            wav_list=[noisy_speech, target_speech]
        )

        # Speed Perturbation
        if (
            self.augmentation_speed_args
            and self.augmentation_speed_args.used
            and torch.rand(1) < self.augmentation_speed_args.prob
        ):
            # include the range top: a bare arange(lo, hi, step) excludes hi,
            # which silently removed the speed-up half of the perturbation
            speed = torch.arange(
                self.augmentation_speed_args.speed_range[0],
                self.augmentation_speed_args.speed_range[1] + 0.025,
                0.05,
            )
            speed = random.choice(speed)
            noisy_speech, (speed) = self.augmentor.sox_speed_perturbed(
                wav=noisy_speech,
                speed=speed.item(),
                sr=self.audio_sr,
            )
            target_speech, _ = self.augmentor.sox_speed_perturbed(
                wav=target_speech,
                speed=speed,
                sr=self.audio_sr,
            )

        # Reverb (whole-mix folder RIR; skipped when the row plan says the
        # channel must stay exactly as the foreground provided it)
        if (
            self.augmentation_reverb_args
            and self.augmentation_reverb_args.used
            and not source_level_reverb
            and not plan.skip_whole_mix_reverb
            and torch.rand(1) < self.augmentation_reverb_args.prob
        ):
            # RIR's target for noisy is full
            noisy_speech, (rir_id, _) = self.augmentor.apply_rir(
                wav=noisy_speech,
                rir_mode="full",
                sr=self.audio_sr,
            )
            # Warping target speech for same RIR but different rir mode
            if self.augmentation_reverb_args.target_rir_type != "anechoic":
                target_speech, _ = self.augmentor.apply_rir(
                    wav=target_speech,
                    rir_id=rir_id,
                    rir_mode=self.augmentation_reverb_args.target_rir_type,
                    sr=self.audio_sr,
                )

            if noisy_speech.shape[0] != 1:
                noisy_speech = noisy_speech[0].view(1, -1)
                target_speech = target_speech[0].view(1, -1)

        # Noise
        if (
            self.augmentation_noise_args
            and self.augmentation_noise_args.used
            and torch.rand(1) < self.augmentation_noise_args.prob
        ):
            dynamic_type = False
            snr = (
                torch.FloatTensor(1)
                .uniform_(
                    self.augmentation_noise_args.snr_range[0],
                    self.augmentation_noise_args.snr_range[1],
                )
                .item()
            )
            # Record before the white-noise branch below re-draws ``snr``.
            noise_snr = snr

            # 1 / 4 cases add dynamic noise type
            if torch.rand(1) < self.augmentation_noise_args.prob / 4:
                dynamic_type = True

            # Room coloring: give the noise a channel of the SAME room the
            # speech was rendered in, so the noise floor is spatially coherent
            # with the mixture instead of arriving dry from nowhere. Guarded:
            # an absent/disabled block never touches the RNG stream.
            noise_transform = None
            room_cfg = self.augmentation_noise_args.room_coloring
            if (
                room_cfg
                and room_cfg.used
                and room_scene is not None
                and torch.rand(1).item() < room_cfg.prob
            ):
                mix_sr = self.audio_sr

                def noise_transform(n, _sr=mix_sr, _scene=room_scene):
                    reverbed, _ = self.augmentor.apply_rir(
                        wav=n,
                        rir_mode="full",
                        sr=_sr,
                        room_scene=_scene,
                        source_role="interferer",
                    )
                    return reverbed

            noisy_speech, _ = self.augmentor.add_bg_noise(
                wav=noisy_speech,
                snr_list=[snr],
                dynamic_type=dynamic_type,
                sr=self.audio_sr,
                noise_transform=noise_transform,
            )
            # unwrap list
            noisy_speech = noisy_speech[0]

            # if dynamic is False, 1 / 4 add white noise
            if (
                not dynamic_type
                and torch.rand(1) < self.augmentation_noise_args.prob_white_noise
            ):
                snr = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.augmentation_noise_args.white_noise_snr_range[0],
                        self.augmentation_noise_args.white_noise_snr_range[1],
                    )
                    .item()
                )
                noisy_speech, _ = self.augmentor.add_bg_white_noise(
                    wav=noisy_speech, snr_list=[snr]
                )

        if isinstance(noisy_speech, list):
            noisy_speech = noisy_speech[0]

        # Absolute capture floor: a noise floor anchored to digital full scale,
        # NOT to the mixture level -- a deployed mic's self-noise + room tone
        # sit at a fixed level whoever is speaking. The SNR-relative noise
        # above cannot express this (it scales with the speech). Added to the
        # mixture only, before the device chain, so it inherits the device
        # response like real capsule noise. Guarded: absent/disabled block
        # never touches the RNG stream.
        floor_cfg = (
            self.augmentation_noise_args.absolute_floor
            if self.augmentation_noise_args
            else None
        )
        if (
            floor_cfg
            and floor_cfg.used
            and torch.rand(1).item() < floor_cfg.prob
        ):
            lo, hi = floor_cfg.level_dbfs_range
            floor_dbfs = torch.empty(1).uniform_(float(lo), float(hi)).item()
            floor = torch.randn_like(noisy_speech) * (10.0 ** (floor_dbfs / 20.0))
            noisy_speech = noisy_speech + floor

        # Snapshot the clean target for VAD labeling before the downstream
        # distortion chain (SRC / IIR / HPF / volume / clipping). Silero VAD
        # gets unreliable on heavily distorted speech, so we label activity on
        # the early-reverb clean signal (post speed-perturb so timing matches).
        vad_reference = target_speech.clone()

        # SRC
        if (
            self.augmentation_src_args
            and self.augmentation_src_args.used
            and torch.rand(1) < self.augmentation_src_args.prob
        ):
            src_target = random.choices(
                self.augmentation_src_args.src_range,
                weights=self.augmentation_src_args.prob_each,
            )[0]

            if torch.rand(1) < 0.5:
                src_backend = "sox"
            else:
                src_backend = "torchaudio"

            noisy_speech, src_info = self.augmentor.apply_src_effect(
                wav=noisy_speech,
                sr=self.audio_sr,
                src_sr=src_target,
                src_backend=src_backend,
            )

            # Wrap target speech to same SRC effect
            if src_backend == "sox":
                target_speech, _ = wav_resampling(
                    wav=target_speech,
                    origin_sr=self.audio_sr,
                    target_sr=src_target,
                    backend="sox",
                )
                target_speech, _ = wav_resampling(
                    wav=target_speech,
                    origin_sr=src_target,
                    target_sr=self.audio_sr,
                    backend="sox",
                )
            else:
                target_speech, *src_info = wav_resampling(
                    wav=target_speech,
                    origin_sr=self.audio_sr,
                    target_sr=src_target,
                    backend="torchaudio",
                    torch_backend_params=src_info[-1],
                )
                target_speech, *src_info = wav_resampling(
                    wav=target_speech,
                    origin_sr=src_target,
                    target_sr=self.audio_sr,
                    backend="torchaudio",
                    torch_backend_params=src_info[-1],
                )

        # 2nd-IIR response
        if (
            self.augmentation_ir_response_args
            and self.augmentation_ir_response_args.used
            and torch.rand(1) < self.augmentation_ir_response_args.prob
        ):
            noisy_speech, (a_coeffs, b_coeffs) = self.augmentor.apply_2nd_iir_response(
                wav=noisy_speech
            )
            target_speech, _ = self.augmentor.apply_2nd_iir_response(
                wav=target_speech, a_coeffs=a_coeffs, b_coeffs=b_coeffs
            )

        # HPF effects
        if (
            self.augmentation_hpf_args
            and self.augmentation_hpf_args.used
            and torch.rand(1) < self.augmentation_hpf_args.prob
        ):
            hpf_cutoff = random.choices(
                self.augmentation_hpf_args.cutoff,
                weights=self.augmentation_hpf_args.prob_each,
            )[0]
            q_factor = torch.FloatTensor(1).normal_(mean=0.707, std=0.1).clip(0.3, 1.3)
            noisy_speech, _ = self.augmentor.apply_hpf(
                wav=noisy_speech,
                sr=self.audio_sr,
                cutoff_freq=hpf_cutoff,
                q_factor=q_factor,
            )
            target_speech, _ = self.augmentor.apply_hpf(
                wav=target_speech,
                sr=self.audio_sr,
                cutoff_freq=hpf_cutoff,
                q_factor=q_factor,
            )

        # Volume perturbed
        if (
            self.augmentation_volume_args
            and self.augmentation_volume_args.used
            and torch.rand(1) < self.augmentation_volume_args.prob
        ):
            vol_ratio = None
            min_quantile = None
            max_quantile = None
            if torch.rand(1) < self.augmentation_volume_args.clipping_prob:
                min_q = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args.clipping_range.min[0],
                    self.augmentation_volume_args.clipping_range.min[1],
                )
                max_q = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args.clipping_range.max[0],
                    self.augmentation_volume_args.clipping_range.max[1],
                )
                noisy_speech, (min_quantile, max_quantile) = (
                    self.augmentor.apply_clipping_distortion(
                        wav=noisy_speech, min_quantile=min_q, max_quantile=max_q
                    )
                )
                target_speech, (_, _) = self.augmentor.apply_clipping_distortion(
                    wav=target_speech,
                    min_quantile=min_quantile,
                    max_quantile=max_quantile,
                )

            else:
                gain = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.augmentation_volume_args.perturbed_range[0],
                        self.augmentation_volume_args.perturbed_range[1],
                    )
                    .item()
                )
                noisy_speech, (vol_ratio) = self.augmentor.sox_volume_perturbed(
                    wav=noisy_speech,
                    vol_ratio=gain,
                    sr=self.audio_sr,
                )
                target_speech, (vol_ratio) = self.augmentor.sox_volume_perturbed(
                    wav=target_speech,
                    vol_ratio=vol_ratio,
                    sr=self.audio_sr,
                )

        # Codec round-trip (channel-side artifact: VoIP/PSTN compression).
        # Applied to noisy_speech only -- target_speech is the clean reference.
        if (
            self.augmentation_codec_args
            and self.augmentation_codec_args.used
            and torch.rand(1) < self.augmentation_codec_args.prob
        ):
            codecs = self.augmentation_codec_args.codecs
            prob_each = self.augmentation_codec_args.prob_each
            if prob_each:
                codec_name = random.choices(codecs, weights=prob_each, k=1)[0]
            else:
                codec_name = random.choice(codecs)
            bitrate_range = self.augmentation_codec_args.bitrate_range.get(codec_name)
            bit_rate = (
                random.randint(int(bitrate_range[0]), int(bitrate_range[1]))
                if bitrate_range
                else None
            )
            noisy_speech, _ = self.augmentor.apply_codec(
                wav=noisy_speech,
                sr=self.audio_sr,
                codec_name=codec_name,
                bit_rate=bit_rate,
            )

        # Packet loss (VoIP transmission artifact).
        if (
            self.augmentation_packet_loss_args
            and self.augmentation_packet_loss_args.used
            and torch.rand(1) < self.augmentation_packet_loss_args.prob
        ):
            packet_ms = random.choice(
                self.augmentation_packet_loss_args.packet_ms_choices
            )
            lo, hi = self.augmentation_packet_loss_args.loss_rate_range
            loss_rate = random.uniform(float(lo), float(hi))
            noisy_speech, _ = self.augmentor.apply_packet_loss(
                wav=noisy_speech,
                sr=self.audio_sr,
                packet_ms=int(packet_ms),
                loss_rate=loss_rate,
            )

        # Final overload guard. The earlier avoid_audio_clipping runs before
        # noise / volume / IIR, any of which can push the mixture past +-1
        # while the model's output is clamped to [-1, 1] -- rescale noisy and
        # target together (and the noise bookkeeping) so the pair stays
        # consistent and inside the representable range.
        peak = float(
            torch.maximum(noisy_speech.abs().amax(), target_speech.abs().amax())
        )
        if peak > 1.0:
            noisy_speech = noisy_speech / peak
            target_speech = target_speech / peak

        # Snipts to training target sample length
        noisy_speech = noisy_speech[..., : self.sample_length]
        target_speech = target_speech[..., : self.sample_length]

        audio_sr = self.audio_sr
        vad_reference = vad_reference[..., : noisy_speech.shape[-1]]
        vad_target = None
        if not self.defer_vad_to_gpu:
            vad_target = self.create_vad_target(vad_reference, sample_rate=audio_sr)
        background_vad_target = None
        if background_speech_reference is not None:
            background_speech_reference = background_speech_reference[
                ..., : noisy_speech.shape[-1]
            ]
            if not self.defer_vad_to_gpu:
                background_vad_target = self.create_vad_target(
                    background_speech_reference,
                    sample_rate=audio_sr,
                )

        # Far parent target: the summed full-RIR interferer speech (post-SIR),
        # zeros when no interferer is present.
        #
        # No loss consumes this. It was added for a far decoder that was never
        # written -- the comment here used to name a `FarReconstructionLoss` that
        # does not exist. It stays because it IS read: `scripts/eval_indomain.py`
        # measures far-speech leakage against it, which is the near/far axis's
        # main diagnostic. Treat it as an eval output, not a training target, and
        # if you add that decoder, say so here.
        far_target = (
            background_speech_reference
            if background_speech_reference is not None
            else torch.zeros_like(noisy_speech)
        )
        sample = {
            "noisy_speech": noisy_speech,
            "clean_speech": target_speech,
            "consistency_noise": noisy_speech - target_speech,
            "far_target": far_target,
            "speaker_id": self.spk2idx[target_speaker],
            "audio_sr": audio_sr,
            "audio_length": noisy_speech.shape[-1],
        }
        if vad_target is not None:
            sample["vad_target"] = vad_target
        elif self.defer_vad_to_gpu:
            # Clean (pre-distortion) reference for the batched GPU VAD labeler.
            sample["vad_reference"] = vad_reference
        if background_vad_target is not None:
            sample["background_vad_target"] = background_vad_target
        elif self.defer_vad_to_gpu and background_speech_reference is not None:
            sample["background_vad_reference"] = background_speech_reference
        self._emit_task_metadata(
            sample,
            foreground_metadata=fg_rir_metadata,
            interferer_metadata=interferer_rir_metadata,
            target_absent=target_absent,
            background_speech_reference=background_speech_reference,
            near_count=0 if target_absent else 1,
            far_count=far_count,
            mix_mode=mix_mode_name,
            realized_speech_sir=realized_speech_sir,
            noise_snr=noise_snr,
            overlap_fraction=getattr(self, "_last_overlap_fraction", float("nan")),
            turn_taking=getattr(self, "_last_turn_taking", 0.0),
        )
        return sample

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
        """Attach bank lineage; subclasses may add task-specific labels."""

        primary = foreground_metadata or (
            interferer_metadata[0] if interferer_metadata else {}
        )

        def _text(metadata: Optional[dict], key: str) -> str:
            if not metadata:
                return ""
            value = metadata.get(key)
            return "" if value is None else str(value)

        sample.update(
            {
                "rir_release_id": _text(primary, "release_id"),
                "rir_release_sha256": _text(primary, "release_sha256"),
                "rir_recipe_id": _text(primary, "release_recipe_id"),
                "rir_variant_id": _text(primary, "release_variant_id"),
                "rir_split": _text(primary, "split"),
                "rir_origin": _text(primary, "origin")
                or _text(primary, "release_origin"),
                "rir_renderer_profile_id": _text(
                    primary,
                    "renderer_profile_id",
                ),
                "rir_production_certificate_sha256": _text(
                    primary,
                    "production_certificate_sha256",
                ),
                "rir_interferer_variant_ids": tuple(
                    _text(metadata, "release_variant_id")
                    for metadata in interferer_metadata
                ),
            }
        )

    def _sample_turn_script(
        self, n_frames: int, hop: int, sr: int, overlap_cfg: OverlapControlConfig
    ) -> tuple:
        """Sample a conversational turn script on the VAD frame grid.

        Alternating near/far turns like a real exchange: the near (target)
        speaker and the far interferer take turns, with small gaps or slight
        boundary overlaps between turns. 50% of rows start with a FAR turn --
        the hardest streaming case (a far monologue with no preceding near
        anchor), which the per-frame Bernoulli fill can never produce.
        Returns (near_frames, far_frames) boolean masks over the frame grid.
        """
        near_len = overlap_cfg.turn_near_seconds
        far_len = overlap_cfg.turn_far_seconds
        gap_rng = overlap_cfg.turn_gap_seconds
        ovl_rng = overlap_cfg.turn_overlap_seconds
        far_first = torch.rand(1).item() < overlap_cfg.far_first_prob

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
                step_s = float(torch.empty(1).uniform_(float(gap_rng[0]), float(gap_rng[1])))
                pos = end + int(round(step_s * fps))
            else:  # slight boundary overlap (next turn starts before this ends)
                step_s = float(torch.empty(1).uniform_(float(ovl_rng[0]), float(ovl_rng[1])))
                pos = max(pos + 1, end - int(round(step_s * fps)))
            turn_is_far = not turn_is_far
        return near_mask, far_mask

    def _apply_overlap_gating(
        self,
        target_speech: torch.Tensor,
        interferers: list,
        sr: int,
        target_mix: Optional[torch.Tensor] = None,
        allow_turn_taking: bool = True,
        turn_taking_prob_override: Optional[float] = None,
    ) -> tuple:
        """Gate interferer (and, in turn-taking mode, target) activity.

        Default path: gate each interferer's activity so it covers a sampled
        fraction of the target's active frames. Without this, both target and
        interferer clips are continuous speech and >80% overlap dominates the
        training distribution. We sample each interferer independently into one
        of three regimes (no/mid/high overlap) so the model sees the full
        spread, including the easier cases where the interferer talks during
        the target's silences.

        Turn-taking path (``turn_taking_prob`` > 0, sampled per ROW): instead
        of per-frame Bernoulli gating, build complementary block envelopes so
        near and far speakers alternate in long conversational turns -- the far
        side gets continuous multi-second solo stretches (incl. row-initial),
        the structure the Bernoulli fill never produces and the regime where
        real far speech was observed to pass through untouched. The target's
        label AND its contribution to the mix are gated with the same near
        envelope so mix and label stay consistent. ``allow_turn_taking=False``
        (target-absent rows) skips target gating: the foreground is subtracted
        from the mix later using a pre-gating snapshot, so gating it here would
        leave a residual.

        Only runs when ``augmentation_speech.overlap_control.used`` is True
        and a VAD labeler is configured.

        Returns ``(target_speech, target_mix, interferers)``.
        """
        overlap_cfg = (
            self.augmentation_speech_args.overlap_control
            if self.augmentation_speech_args
            else None
        )
        if overlap_cfg is None or not overlap_cfg.used:
            return target_speech, target_mix, interferers
        if self.gating_vad_labeler is None or not interferers:
            return target_speech, target_mix, interferers

        target_wav = target_speech.squeeze(0) if target_speech.dim() == 2 else target_speech
        target_vad = self.gating_vad_labeler(target_wav, sample_rate=sr)
        target_active = target_vad.bool()
        if int(target_active.sum().item()) == 0:
            # No active target frames -> no meaningful overlap concept; pass through.
            return target_speech, target_mix, interferers
        hop = self.gating_vad_labeler.hop_length
        n_frames = target_vad.shape[0]

        fade_samples = overlap_cfg.fade_samples
        fade_kernel = torch.hann_window(fade_samples * 2 + 1)
        fade_kernel = fade_kernel / fade_kernel.sum().clamp_min(1e-6)
        fade_kernel = fade_kernel.view(1, 1, -1)

        def _smooth_env(frames_mask: torch.Tensor, length: int) -> torch.Tensor:
            env = frames_mask.float().repeat_interleave(hop)
            if env.shape[0] < length:
                env = torch.nn.functional.pad(env, (0, length - env.shape[0]))
            else:
                env = env[:length]
            return (
                torch.nn.functional.conv1d(
                    env.view(1, 1, -1), fade_kernel, padding=fade_samples
                )
                .view(-1)
                .clamp(0.0, 1.0)
            )

        def _gate(sig: torch.Tensor, frames_mask: torch.Tensor) -> torch.Tensor:
            env = _smooth_env(frames_mask, sig.shape[-1])
            return sig * env.view(*([1] * (sig.dim() - 1)), -1)

        # Row-type override: individual row types can request their own
        # turn-taking rate -- far-solo stretches teach the absolute "lone far
        # voice = suppress" decision while the same row still contains near-field
        # keep segments. None = use the overlap_control value (bit-identical for
        # every row type that does not override it).
        if turn_taking_prob_override is not None:
            turn_taking_prob = float(turn_taking_prob_override)
        else:
            turn_taking_prob = overlap_cfg.turn_taking_prob
        if (
            allow_turn_taking
            and turn_taking_prob > 0.0
            and torch.rand(1).item() < turn_taking_prob
        ):
            near_mask, far_mask = self._sample_turn_script(n_frames, hop, sr, overlap_cfg)
            target_speech = _gate(target_speech, near_mask)
            if target_mix is not None:
                target_mix = _gate(target_mix, near_mask)
            gated = [_gate(intf, far_mask) for intf in interferers]
            active_gated = target_active & near_mask
            active = int(active_gated.sum().item())
            self._last_overlap_fraction = (
                float(int((far_mask & active_gated).sum().item()) / active)
                if active > 0
                else 0.0
            )
            self._last_turn_taking = 1.0
            return target_speech, target_mix, gated

        no_overlap_prob = overlap_cfg.no_overlap_prob
        high_overlap_prob = overlap_cfg.high_overlap_prob
        mid_range = overlap_cfg.mid_overlap_range
        high_range = overlap_cfg.high_overlap_range
        fill_range = overlap_cfg.fill_on_silent_range

        gated = []
        union_wanted = torch.zeros(n_frames, dtype=torch.bool)
        for intf in interferers:
            r = torch.rand(1).item()
            if r < no_overlap_prob:
                overlap_p = 0.0
            elif r < no_overlap_prob + high_overlap_prob:
                overlap_p = float(
                    torch.empty(1).uniform_(float(high_range[0]), float(high_range[1]))
                )
            else:
                overlap_p = float(
                    torch.empty(1).uniform_(float(mid_range[0]), float(mid_range[1]))
                )
            fill_p = float(
                torch.empty(1).uniform_(float(fill_range[0]), float(fill_range[1]))
            )

            wanted = torch.zeros(n_frames)
            rand_vals = torch.rand(n_frames)
            wanted[target_active] = (rand_vals[target_active] < overlap_p).float()
            wanted[~target_active] = (rand_vals[~target_active] < fill_p).float()
            union_wanted |= wanted.bool()
            gated.append(_gate(intf, wanted.bool()))
        # Realized overlap = fraction of target-active frames also covered by some
        # interferer. Emitted as metadata so eval can bucket by overlap and read
        # whether the model is overlap-limited without a separate no-overlap run.
        active = int(target_active.sum().item())
        if active > 0:
            self._last_overlap_fraction = float(
                int((union_wanted & target_active).sum().item()) / active
            )
        return target_speech, target_mix, gated


class NoiseSuppressionCollateFunc:
    """Collate functino used in Dataloader."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict):
        col_noisy = []
        col_clean = []
        col_consistency = []
        col_spkid = []
        col_sr = []
        col_length = []
        col_vad = []
        col_vad_ref = []

        for b in batch:
            """
            one batch -- (dict) -- {
                "noisy_speech",
                "clean_speech",
                "consistency_noise",
                "speaker_id",
                "audio_sr",
                "audio_length", }
            wav file each with shape [1, L]
            """
            col_clean.append(b["clean_speech"].squeeze())
            col_noisy.append(b["noisy_speech"].squeeze())
            col_consistency.append(b["consistency_noise"].squeeze())
            col_spkid.append(b["speaker_id"])
            col_sr.append(b["audio_sr"])
            col_length.append(b["audio_length"])
            if "vad_target" in b:
                col_vad.append(b["vad_target"].squeeze())
            if "vad_reference" in b:
                col_vad_ref.append(b["vad_reference"].squeeze())

        padded_clean = pad_sequence(col_clean, batch_first=True)  # [N, L]
        padded_noisy = pad_sequence(col_noisy, batch_first=True)  # [N, L]
        padded_consistency = pad_sequence(col_consistency, batch_first=True)  # [N, L]

        out = {
            "clean_speech": padded_clean,
            "noisy_speech": padded_noisy,
            "consistency_noise": padded_consistency,
            "spkid": torch.Tensor(col_spkid),
            "sr": torch.Tensor(col_sr),
            "length": torch.Tensor(col_length),
        }
        if col_vad:
            out["vad_target"] = pad_sequence(col_vad, batch_first=True)
        if col_vad_ref:
            out["vad_reference"] = pad_sequence(col_vad_ref, batch_first=True)
        for key in RIR_PROVENANCE_KEYS:
            if any(key in item for item in batch):
                out[key] = [item.get(key, "") for item in batch]
        return out
