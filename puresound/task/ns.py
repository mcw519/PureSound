import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.noise import add_bg_noise
from puresound.config.augmentation import (
    CodecAugmentation,
    PacketLossAugmentation,
    TargetAbsentAugmentation,
)
from puresound.dataset.dynamic_base import (
    DynamicBaseDataset,
)
from puresound.task.device_chain import (
    DEVICE_CHAIN_SCALARS,
    device_chain_from_blocks,
)
from puresound.task.overlap_gating import OverlapGating


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
    #: Transmission damage and the target-absent row type on top of the shared
    #: capture blocks.
    AUGMENTATION_BLOCKS = {
        **DynamicBaseDataset.AUGMENTATION_BLOCKS,
        "augmentation_codec_args": CodecAugmentation,
        "augmentation_packet_loss_args": PacketLossAugmentation,
        "augmentation_target_absent_args": TargetAbsentAugmentation,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.device_chain = device_chain_from_blocks(self.augmentor, self)
        self.overlap_gating = OverlapGating(
            self.augmentation_speech_args.overlap_control
            if self.augmentation_speech_args
            else None,
            self.gating_vad_labeler,
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
        overlap_fraction = float("nan")
        turn_taking = 0.0

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

            gating = self.overlap_gating.apply(
                target_speech,
                interfered_speech,
                sr=self.audio_sr,
                target_mix=noisy_speech,
                allow_turn_taking=not target_absent,
                turn_taking_prob=self._turn_taking_override(plan),
            )
            target_speech = gating.target
            noisy_speech = gating.target_mix
            interfered_speech = gating.interferers
            overlap_fraction = gating.overlap_fraction
            turn_taking = gating.turn_taking

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

        # Capture noise floor: a level that does NOT scale with the speech.
        # A deployed mic's self-noise and the room's own tone sit where they sit
        # whoever is talking, and the SNR-relative noise above cannot express
        # that. Mixture only, and before the device chain deliberately, so it
        # picks up the device response the way capsule noise does.
        #
        # "dBFS" is the level as *drawn*, not as delivered. The converter at the
        # end of the chain gain-stages the whole row, so a floor drawn at -45
        # arrives lower by however much that row was turned down -- on the
        # shipped recipe, the 29% of rows it touches move by a median of 1.0 dB
        # and 6.3 dB at p5. That is correct for capsule and room noise: both sit
        # upstream of the preamp and both follow it. A converter's *own*
        # electronic noise would not, and would have to be added after
        # `_analogue_to_digital` -- at roughly -90 dBFS it is 40 dB below
        # anything this range draws, which is why there is no such stage.
        #
        # Guarded: absent/disabled block never touches the RNG stream.
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

        # Capture and transmission chain: the analogue path (SRC, IIR, HPF,
        # volume), the converter, then digital transmission (codec, packet
        # loss). Order, linearity and RNG discipline are the chain's contract --
        # see puresound/task/device_chain.py.
        chain = self.device_chain.apply(
            noisy_speech, target_speech, sample_rate=self.audio_sr
        )
        noisy_speech, target_speech = chain.noisy, chain.target

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
        sample.update(
            {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in chain.applied.items()
            }
        )
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
            overlap_fraction=overlap_fraction,
            turn_taking=turn_taking,
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
        # Which channel each row went through -- see DEVICE_CHAIN_SCALARS. Every
        # row carries every key, so one `cat` per key is well defined.
        for key in DEVICE_CHAIN_SCALARS:
            values = [b[key].view(-1) for b in batch if key in b]
            if values:
                out[key] = torch.cat(values, dim=0)
        for key in RIR_PROVENANCE_KEYS:
            if any(key in item for item in batch):
                out[key] = [item.get(key, "") for item in batch]
        return out
