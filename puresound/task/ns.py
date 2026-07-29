import json
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.dsp import wav_resampling
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise
from puresound.dataset.dynamic_base import DynamicBaseDataset


def if_none_else(a, b):
    if a is not None:
        return a
    else:
        return b


class NoiseSuppressionDataset(DynamicBaseDataset):
    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        audio_gain_nomalized_to: Optional[int] = None,
        augmentation_speech_args: Optional[int] = None,
        augmentation_noise_args: Optional[Dict] = None,
        augmentation_reverb_args: Optional[Dict] = None,
        augmentation_speed_args: Optional[Dict] = None,
        augmentation_ir_response_args: Optional[Dict] = None,
        augmentation_src_args: Optional[Dict] = None,
        augmentation_hpf_args: Optional[Dict] = None,
        augmentation_volume_args: Optional[Dict] = None,
        augmentation_codec_args: Optional[Dict] = None,
        augmentation_packet_loss_args: Optional[Dict] = None,
        augmentation_target_absent_args: Optional[Dict] = None,
        augmentation_realfar_args: Optional[Dict] = None,
        augmentation_realnear_args: Optional[Dict] = None,
        vad_label_args: Optional[Dict] = None,
    ):
        super().__init__(
            metafile_path=metafile_path,
            min_utt_length_in_seconds=min_utt_length_in_seconds,
            min_utts_in_each_speaker=min_utts_in_each_speaker,
            target_sr=target_sr,
            training_sample_length_in_seconds=training_sample_length_in_seconds,
            audio_gain_nomalized_to=audio_gain_nomalized_to,
            augmentation_speech_args=augmentation_speech_args,
            augmentation_noise_args=augmentation_noise_args,
            augmentation_reverb_args=augmentation_reverb_args,
            augmentation_speed_args=augmentation_speed_args,
            augmentation_ir_response_args=augmentation_ir_response_args,
            augmentation_src_args=augmentation_src_args,
            augmentation_hpf_args=augmentation_hpf_args,
            augmentation_volume_args=augmentation_volume_args,
            vad_label_args=vad_label_args,
        )
        self.augmentation_codec_args = augmentation_codec_args
        self.augmentation_packet_loss_args = augmentation_packet_loss_args
        self.augmentation_target_absent_args = augmentation_target_absent_args
        # Pool of real far-field recordings used directly as interferers, i.e.
        # finished loudspeaker->air->mic waveforms inserted WITHOUT RIR
        # convolution. A convolved far channel only carries the linear
        # time-invariant part of a capture chain; a real recording also carries
        # its level, spectral tilt, transducer non-linearity and noise floor.
        self.augmentation_realfar_args = augmentation_realfar_args
        self._realfar_pool = self._load_real_pool(augmentation_realfar_args)
        # Pool of real near-field recordings used as the KEEP side: rows whose
        # foreground is a genuine close-mic recording (target = itself), mixed
        # with the same interferers and noise as any other row. Putting real
        # recordings on both sides of the decision keeps the keep/suppress
        # boundary on proximity cues instead of on capture-chain identity.
        self.augmentation_realnear_args = augmentation_realnear_args
        self._realnear_pool = self._load_real_pool(augmentation_realnear_args)

    def _load_real_pool(self, cfg: Optional[Dict]) -> List[Dict]:
        """Load a real-recording pool manifest (one JSON object per line; see
        egs/voice_isolate/scripts/build_real_recording_pool.py). Returns [] when the
        block is absent/disabled so the __getitem__ branches are no-ops and
        existing recipes stay bit-identical."""
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
                target_lvl=self.audio_gain_nomalized_to,
                resample_to=sr,
            )
            wavs.append(wav[0].reshape(1, -1))
            metas.append(
                {"source_receiver_distance": p.get("distance_m"), "origin": "real"}
            )
        return wavs, metas

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
        add_n_cases_cfg = self.augmentation_speech_args["add_n_cases"]
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
        # playback): band-limited + lightly compressed before its RIR, and
        # placed wall-adjacent by the room simulator (source_role="media").
        media_cfg = self.augmentation_speech_args.get("media_voice")
        media_flags = [
            bool(
                media_cfg
                and media_cfg.get("used", False)
                and torch.rand(1).item() < float(media_cfg.get("prob", 0.0))
            )
            for _ in interfered_speech
        ]
        if any(media_flags):
            hp_lo, hp_hi = media_cfg.get("hp_cutoff_range", [200, 400])
            lp_lo, lp_hi = media_cfg.get("lp_cutoff_range", [3500, 7000])
            cp_lo, cp_hi = media_cfg.get("compress_power_range", [0.6, 0.9])
            for idx, is_media in enumerate(media_flags):
                if not is_media:
                    continue
                interfered_speech[idx], _ = self.augmentor.apply_media_coloring(
                    wav=interfered_speech[idx],
                    sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    hp_cutoff=torch.empty(1).uniform_(hp_lo, hp_hi).item(),
                    lp_cutoff=torch.empty(1).uniform_(lp_lo, lp_hi).item(),
                    compress_power=torch.empty(1).uniform_(cp_lo, cp_hi).item(),
                )

        if source_level_reverb:
            interfered_speech = self.align_audio_list(
                wav_list=interfered_speech,
                length=if_none_else(
                    self.training_sample_length,
                    int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                ),
                padding_type="zero",
            )
            reverb_interferers = []
            for speech, is_media in zip(interfered_speech, media_flags):
                reverb_interferers.append(
                    self.apply_source_level_interferer_reverb(
                        wav=speech,
                        sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        room_scene=room_scene,
                        source_role="media" if is_media else "interferer",
                    )
                )
                rir_meta = getattr(self.augmentor, "_last_rir_meta", None)
                if rir_meta is not None:
                    interferer_rir_metadata.append(dict(rir_meta))
            interfered_speech = reverb_interferers
        else:
            # Aligned and Mixing
            clips_wav = [target_speech] + interfered_speech
            clips_wav = self.align_audio_list(
                wav_list=clips_wav,
                length=if_none_else(
                    self.training_sample_length,
                    int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                ),
                padding_type="zero",
            )
            target_speech = clips_wav[0]
            interfered_speech = clips_wav[1:]

        return target_speech, interfered_speech, interferer_rir_metadata

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
            length=if_none_else(
                self.training_sample_length,
                int(self.ori_audio_sr * self.training_sample_length_in_seconds),
            ),
        )[0]
        # Decide once per sample whether this is a "target-absent" case
        # (near-field foreground is silent, only far-field interferers + noise
        # remain). Concrete gating happens after the augmentation_speech block
        # so SIR sampling and overlap-control still use the real target as a
        # reference; we subtract target_in_mix at the end.
        # Real-NEAR keep row? Decide FIRST (guarded: an absent/disabled block never
        # touches the RNG stream). On these rows the foreground becomes a genuine
        # close-mic recording and the target is that recording itself, so real
        # captured speech appears on the KEEP side of the same mixtures whose far
        # side is real. The corpus utterance drawn above is discarded;
        # batching/speaker semantics stay untouched.
        realnear_cfg = self.augmentation_realnear_args
        use_realnear = False
        realnear_room: Optional[str] = None
        realnear_speaker: Optional[str] = None
        realnear_fg_metadata: Optional[dict] = None
        if realnear_cfg is not None and realnear_cfg.get("used", False) and self._realnear_pool:
            use_realnear = torch.rand(1).item() < float(realnear_cfg.get("prob", 0.0))
        if use_realnear:
            near_pick = random.choice(self._realnear_pool)
            realnear_room = near_pick.get("room")
            realnear_speaker = near_pick.get("speaker")
            near_wav, _ = AudioIO.open(
                f_path=near_pick["wav_path"],
                target_lvl=self.audio_gain_nomalized_to,
                resample_to=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            target_speech = self.align_audio_list(
                wav_list=[near_wav[0].reshape(1, -1)],
                length=if_none_else(
                    self.training_sample_length,
                    int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                ),
            )[0]
            realnear_fg_metadata = {
                "source_receiver_distance": near_pick.get("distance_m"),
                "origin": "real",
            }

        # Real-far row? Decide next so it can own the target-absent draw. Guarded
        # so an absent/disabled block never touches the RNG stream -> recipes
        # without augmentation_realfar regenerate bit-identically.
        realfar_cfg = self.augmentation_realfar_args
        target_absent_cfg = self.augmentation_target_absent_args
        use_realfar = False
        if use_realnear:
            # Keep row: interferers (when the speech-aug gate fires) come from the
            # real-far pool; the target is always present -- lone-far never applies.
            use_realfar = bool(self._realfar_pool)
            target_absent = False
            force_interferer = False
        else:
            if realfar_cfg is not None and realfar_cfg.get("used", False) and self._realfar_pool:
                use_realfar = torch.rand(1).item() < float(realfar_cfg.get("prob", 0.0))
            if use_realfar:
                # Real-far rows decide lone-far (target-absent) by their OWN prob,
                # independently of the synthetic target-absent rate: the two
                # far-field sources need separate pressure to be tuned separately.
                target_absent = torch.rand(1).item() < float(realfar_cfg.get("lone_far_prob", 0.0))
                force_interferer = target_absent
            else:
                target_absent = (
                    target_absent_cfg is not None
                    and target_absent_cfg.get("used", False)
                    and torch.rand(1).item() < float(target_absent_cfg.get("prob", 0.0))
                )
                force_interferer = bool(
                    target_absent
                    and target_absent_cfg is not None
                    and target_absent_cfg.get("force_interferer", False)
                )

        interferer_rir_metadata: List[dict] = []
        background_speech_reference = None

        if use_realnear:
            # The real near recording already carries its full end-to-end channel;
            # no synthetic room is simulated on these rows.
            source_level_reverb = False
            room_scene = None
            fg_rir_metadata = realnear_fg_metadata
            noisy_speech = target_speech.clone()
        else:
            source_level_reverb = self.should_apply_source_level_reverb()
            room_scene = self.augmentor.sample_room_scene() if source_level_reverb else None
            fg_rir_metadata = None
            if source_level_reverb:
                noisy_speech, target_speech, fg_rir_metadata = (
                    self.apply_source_level_target_reverb(
                        wav=target_speech,
                        sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        room_scene=room_scene,
                    )
                )
            else:
                noisy_speech = target_speech.clone()

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
            and self.augmentation_speech_args["used"]
            and (
                force_interferer
                or use_realfar
                or torch.rand(1) < self.augmentation_speech_args["prob"]
            )
        ):
            if use_realfar:
                # Real far interferers: finished loudspeaker->air->mic recordings
                # inserted directly, with no RIR applied. The near foreground
                # still carries its simulated near-field RIR from the
                # source-level-reverb block above.
                add_n_cases_cfg = self.augmentation_speech_args["add_n_cases"]
                if isinstance(add_n_cases_cfg, (list, tuple)):
                    n_interferers = random.randint(
                        int(add_n_cases_cfg[0]), int(add_n_cases_cfg[1])
                    )
                else:
                    n_interferers = int(add_n_cases_cfg)
                interfered_speech, realfar_meta = self._sample_realfar_interferers(
                    n_interferers,
                    if_none_else(self.target_sr, self.ori_audio_sr),
                    prefer_room=realnear_room,
                    exclude_speaker=realnear_speaker,
                )
                interferer_rir_metadata.extend(realfar_meta)
                interfered_speech = self.align_audio_list(
                    wav_list=interfered_speech,
                    length=if_none_else(
                        self.training_sample_length,
                        int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                    ),
                    padding_type="zero",
                )
            else:
                target_speech, interfered_speech, syn_meta = (
                    self._build_synthetic_interferers(
                        target_speaker=target_speaker,
                        target_speech=target_speech,
                        room_scene=room_scene,
                        source_level_reverb=source_level_reverb,
                    )
                )
                interferer_rir_metadata.extend(syn_meta)

            # Real rows may carry their own turn-taking rate: their far-solo
            # stretches supply absolute-suppress supervision for real far voices
            # while the near foreground stays present elsewhere in the row.
            tt_override = None
            if use_realnear and realnear_cfg is not None:
                tt_override = realnear_cfg.get("turn_taking_prob")
            elif use_realfar and realfar_cfg is not None:
                tt_override = realfar_cfg.get("turn_taking_prob")
            target_speech, noisy_speech, interfered_speech = self._apply_overlap_gating(
                target_speech=target_speech,
                interferers=interfered_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                target_mix=noisy_speech,
                allow_turn_taking=not target_absent,
                turn_taking_prob_override=tt_override,
            )

            far_count = len(interfered_speech)
            interfered_speech = (
                torch.cat(interfered_speech, dim=0).sum(dim=0).reshape(1, -1)
            )

            # Foreground-vs-interferer mixing. When mix_mode is enabled it
            # replaces the single hard SIR draw with explicit modes:
            # 'physical' keeps the natural post-RIR level
            # ratio so the real DRR/proximity cue survives; the rescaling modes
            # draw a mode-specific SIR range. Falls back bit-exact to the legacy
            # snr_range path when mix_mode is off / absent.
            fg_wav = noisy_speech if source_level_reverb else target_speech
            mm_cfg = (self.augmentation_speech_args or {}).get("mix_mode")
            # Real-far rows skip mix_mode: the synthetic-near vs real-recorded-far
            # level ratio is not physically meaningful, so use the controlled hard
            # SIR draw (augmentation_speech.snr_range) instead of 'physical' sum.
            if mm_cfg and mm_cfg.get("used", False) and not use_realfar:
                mode = self._sample_mix_mode(mm_cfg)
                mix_mode_name = mode.get("name", "physical")
                if mode.get("physical", False):
                    # No relative rescale: sum at natural post-RIR levels.
                    noisy_speech = fg_wav + interfered_speech
                    background_speech_reference = interfered_speech
                    fg_p = float(fg_wav.pow(2).sum())
                    itf_p = float(interfered_speech.pow(2).sum())
                    realized_speech_sir = float(
                        10.0 * np.log10((fg_p + 1e-8) / (itf_p + 1e-8))
                    )
                else:
                    lo, hi = mode["sir_range"]
                    sir = float(torch.empty(1).uniform_(float(lo), float(hi)).item())
                    noisy_speech, interfered_speech = add_bg_noise(
                        wav=fg_wav, noise=[interfered_speech], snr_list=[sir],
                    )
                    noisy_speech = noisy_speech[0]
                    background_speech_reference = interfered_speech[0]
                    realized_speech_sir = sir
            else:
                sir = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.augmentation_speech_args["snr_range"][0],
                        self.augmentation_speech_args["snr_range"][1],
                    )
                    .item()
                )

                # Mixing with SIR
                noisy_speech, interfered_speech = add_bg_noise(
                    wav=fg_wav,
                    noise=[interfered_speech],
                    snr_list=[sir],
                )
                noisy_speech = noisy_speech[0]
                background_speech_reference = interfered_speech[0]
                mix_mode_name = "legacy"
                realized_speech_sir = sir

            # Treating all speech clips as target speech
            if self.augmentation_speech_args["is_target"]:
                target_speech = noisy_speech.clone()

        # Target-absent gating: strip the foreground contribution from the
        # mixture and zero the reference so VAD / SDR / consistency_noise
        # downstream all reflect "no near-field speaker". Done before any
        # rescaling so the leftover mixture levels stay self-consistent.
        if target_absent:
            noisy_speech = noisy_speech - target_in_mix
            target_speech = torch.zeros_like(target_speech)

        # Residual TTS-playback echo (plan section 9.5 stage 5): the robot's own
        # loudspeaker leaks into the mic. Modeled as another utterance through a
        # near-field RIR from the same room, added at the residual level an
        # upstream AEC would leave (erle_db_range below the mixture). Never in
        # the target; requires source-level reverb for its own RIR.
        echo_cfg = (
            self.augmentation_speech_args.get("echo_playback")
            if self.augmentation_speech_args
            else None
        )
        if (
            echo_cfg
            and echo_cfg.get("used", False)
            and source_level_reverb
            and torch.rand(1).item() < float(echo_cfg.get("prob", 0.0))
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
                length=if_none_else(
                    self.training_sample_length,
                    int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                ),
                padding_type="zero",
            )[0]
            echo_speech = self.apply_source_level_interferer_reverb(
                wav=echo_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                room_scene=room_scene,
                distance_range_override=echo_cfg.get("distance_range", [0.2, 1.0]),
            )
            erle_lo, erle_hi = echo_cfg.get("erle_db_range", [20.0, 35.0])
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
            and self.augmentation_speed_args["used"]
            and torch.rand(1) < self.augmentation_speed_args["prob"]
        ):
            speed = torch.arange(
                self.augmentation_speed_args["speed_range"][0],
                self.augmentation_speed_args["speed_range"][1],
                0.05,
            )
            speed = random.choice(speed)
            noisy_speech, (speed) = self.augmentor.sox_speed_perturbed(
                wav=noisy_speech,
                speed=speed.item(),
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            target_speech, _ = self.augmentor.sox_speed_perturbed(
                wav=target_speech,
                speed=speed,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )

        # Reverb (whole-mix folder RIR; never on real-near rows -- their channel
        # is the real recording itself, and source_level_reverb is forced False
        # there so this block would otherwise fire)
        if (
            self.augmentation_reverb_args
            and self.augmentation_reverb_args["used"]
            and not source_level_reverb
            and not use_realnear
            and torch.rand(1) < self.augmentation_reverb_args["prob"]
        ):
            # RIR's target for noisy is full
            noisy_speech, (rir_id, _) = self.augmentor.apply_rir(
                wav=noisy_speech,
                rir_mode="full",
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            # Warping target speech for same RIR but different rir mode
            if self.augmentation_reverb_args["target_rir_type"] != "anechoic":
                target_speech, _ = self.augmentor.apply_rir(
                    wav=target_speech,
                    rir_id=rir_id,
                    rir_mode=self.augmentation_reverb_args["target_rir_type"],
                    sr=if_none_else(self.target_sr, self.ori_audio_sr),
                )

            if noisy_speech.shape[0] != 1:
                noisy_speech = noisy_speech[0].view(1, -1)
                target_speech = target_speech[0].view(1, -1)

        # Noise
        # We collect added noises for if we need to use high SNR noisy speech as ground truth
        added_noise = None
        if (
            self.augmentation_noise_args
            and self.augmentation_noise_args["used"]
            and torch.rand(1) < self.augmentation_noise_args["prob"]
        ):
            dynamic_type = False
            snr = (
                torch.FloatTensor(1)
                .uniform_(
                    self.augmentation_noise_args["snr_range"][0],
                    self.augmentation_noise_args["snr_range"][1],
                )
                .item()
            )
            # Record before the white-noise branch below re-draws ``snr``.
            noise_snr = snr

            # 1 / 4 cases add dynamic noise type
            if torch.rand(1) < self.augmentation_noise_args["prob"] / 4:
                dynamic_type = True

            noisy_speech, (added_noise, _, _) = self.augmentor.add_bg_noise(
                wav=noisy_speech,
                snr_list=[snr],
                dynamic_type=dynamic_type,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            added_noise = added_noise[0]

            # unwrap list
            noisy_speech = noisy_speech[0]

            # if dynamic is False, 1 / 4 add white noise
            if (
                not dynamic_type
                and torch.rand(1) < self.augmentation_noise_args["prob_white_noise"]
            ):
                snr = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.augmentation_noise_args["white_noise_snr_range"][0],
                        self.augmentation_noise_args["white_noise_snr_range"][1],
                    )
                    .item()
                )
                noisy_speech, (added_white_noise, _) = (
                    self.augmentor.add_bg_white_noise(wav=noisy_speech, snr_list=[snr])
                )

                # Mixing noise for later using
                added_noise += added_white_noise[0]

        if isinstance(noisy_speech, list):
            noisy_speech = noisy_speech[0]

        # Snapshot the clean target for VAD labeling before the downstream
        # distortion chain (SRC / IIR / HPF / volume / clipping). Silero VAD
        # gets unreliable on heavily distorted speech, so we label activity on
        # the early-reverb clean signal (post speed-perturb so timing matches).
        vad_reference = target_speech.clone()

        # SRC
        flag_src = False
        if (
            self.augmentation_src_args
            and self.augmentation_src_args["used"]
            and torch.rand(1) < self.augmentation_src_args["prob"]
        ):
            flag_src = True
            src_target = random.choices(
                self.augmentation_src_args["src_range"],
                weights=self.augmentation_src_args["prob_each"],
            )[0]

            if torch.rand(1) < 0.5:
                src_backend = "sox"
            else:
                src_backend = "torchaudio"

            noisy_speech, src_info = self.augmentor.apply_src_effect(
                wav=noisy_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                src_sr=src_target,
                src_backend=src_backend,
            )

            # Wrap target speech to same SRC effect
            if src_backend == "sox":
                target_speech, _ = wav_resampling(
                    wav=target_speech,
                    origin_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    target_sr=src_target,
                    backend="sox",
                )
                target_speech, _ = wav_resampling(
                    wav=target_speech,
                    origin_sr=src_target,
                    target_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    backend="sox",
                )
            else:
                target_speech, *src_info = wav_resampling(
                    wav=target_speech,
                    origin_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    target_sr=src_target,
                    backend="torchaudio",
                    torch_backend_params=src_info[-1],
                )
                target_speech, *src_info = wav_resampling(
                    wav=target_speech,
                    origin_sr=src_target,
                    target_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    backend="torchaudio",
                    torch_backend_params=src_info[-1],
                )

        # 2nd-IIR response
        flag_iir = False
        if (
            self.augmentation_ir_response_args
            and self.augmentation_ir_response_args["used"]
            and torch.rand(1) < self.augmentation_ir_response_args["prob"]
        ):
            flag_iir = True
            noisy_speech, (a_coeffs, b_coeffs) = self.augmentor.apply_2nd_iir_response(
                wav=noisy_speech
            )
            target_speech, _ = self.augmentor.apply_2nd_iir_response(
                wav=target_speech, a_coeffs=a_coeffs, b_coeffs=b_coeffs
            )

        # HPF effects
        flag_hpf = False
        if (
            self.augmentation_hpf_args
            and self.augmentation_hpf_args["used"]
            and torch.rand(1) < self.augmentation_hpf_args["prob"]
        ):
            flag_hpf = True
            hpf_cutoff = random.choices(
                self.augmentation_hpf_args["cutoff"],
                weights=self.augmentation_hpf_args["prob_each"],
            )[0]
            q_factor = torch.FloatTensor(1).normal_(mean=0.707, std=0.1).clip(0.3, 1.3)
            noisy_speech, _ = self.augmentor.apply_hpf(
                wav=noisy_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                cutoff_freq=hpf_cutoff,
                q_factor=q_factor,
            )
            target_speech, _ = self.augmentor.apply_hpf(
                wav=target_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                cutoff_freq=hpf_cutoff,
                q_factor=q_factor,
            )

        # Volume perturbed
        flag_volume = False
        if (
            self.augmentation_volume_args
            and self.augmentation_volume_args["used"]
            and torch.rand(1) < self.augmentation_volume_args["prob"]
        ):
            flag_volume = True
            vol_ratio = None
            min_quantile = None
            max_quantile = None
            if torch.rand(1) < self.augmentation_volume_args["clipping_prob"]:
                min = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args["clipping_range"]["min"][0],
                    self.augmentation_volume_args["clipping_range"]["min"][1],
                )
                max = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args["clipping_range"]["max"][0],
                    self.augmentation_volume_args["clipping_range"]["max"][1],
                )
                noisy_speech, (min_quantile, max_quantile) = (
                    self.augmentor.apply_clipping_distortion(
                        wav=noisy_speech, min_quantile=min, max_quantile=max
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
                        self.augmentation_volume_args["perturbed_range"][0],
                        self.augmentation_volume_args["perturbed_range"][1],
                    )
                    .item()
                )
                noisy_speech, (vol_ratio) = self.augmentor.sox_volume_perturbed(
                    wav=noisy_speech,
                    vol_ratio=gain,
                    sr=if_none_else(self.target_sr, self.ori_audio_sr),
                )
                target_speech, (vol_ratio) = self.augmentor.sox_volume_perturbed(
                    wav=target_speech,
                    vol_ratio=vol_ratio,
                    sr=if_none_else(self.target_sr, self.ori_audio_sr),
                )

        # Codec round-trip (channel-side artifact: VoIP/PSTN compression).
        # Applied to noisy_speech only -- target_speech is the clean reference.
        if (
            self.augmentation_codec_args
            and self.augmentation_codec_args.get("used")
            and torch.rand(1) < self.augmentation_codec_args["prob"]
        ):
            codecs = self.augmentation_codec_args["codecs"]
            prob_each = self.augmentation_codec_args.get("prob_each")
            if prob_each:
                codec_name = random.choices(codecs, weights=prob_each, k=1)[0]
            else:
                codec_name = random.choice(codecs)
            bitrate_range = self.augmentation_codec_args.get("bitrate_range", {}).get(
                codec_name
            )
            bit_rate = (
                random.randint(int(bitrate_range[0]), int(bitrate_range[1]))
                if bitrate_range
                else None
            )
            noisy_speech, _ = self.augmentor.apply_codec(
                wav=noisy_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                codec_name=codec_name,
                bit_rate=bit_rate,
            )

        # Packet loss (VoIP transmission artifact).
        if (
            self.augmentation_packet_loss_args
            and self.augmentation_packet_loss_args.get("used")
            and torch.rand(1) < self.augmentation_packet_loss_args["prob"]
        ):
            packet_ms = random.choice(
                self.augmentation_packet_loss_args["packet_ms_choices"]
            )
            lo, hi = self.augmentation_packet_loss_args["loss_rate_range"]
            loss_rate = random.uniform(float(lo), float(hi))
            noisy_speech, _ = self.augmentor.apply_packet_loss(
                wav=noisy_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                packet_ms=int(packet_ms),
                loss_rate=loss_rate,
            )

        # Snipts to training target sample length
        noisy_speech = noisy_speech[
            ...,
            : if_none_else(
                self.training_sample_length,
                int(self.ori_audio_sr * self.training_sample_length_in_seconds),
            ),
        ]
        target_speech = target_speech[
            ...,
            : if_none_else(
                self.training_sample_length,
                int(self.ori_audio_sr * self.training_sample_length_in_seconds),
            ),
        ]

        # Wrap added_noise
        if added_noise is not None:
            if flag_src:
                if src_backend == "sox":
                    added_noise, _ = wav_resampling(
                        wav=added_noise,
                        origin_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        target_sr=src_target,
                        backend="sox",
                    )
                    added_noise, _ = wav_resampling(
                        wav=added_noise,
                        origin_sr=src_target,
                        target_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        backend="sox",
                    )
                else:
                    added_noise, *src_info = wav_resampling(
                        wav=added_noise,
                        origin_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        target_sr=src_target,
                        backend="torchaudio",
                        torch_backend_params=src_info[-1],
                    )
                    added_noise, *src_info = wav_resampling(
                        wav=added_noise,
                        origin_sr=src_target,
                        target_sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        backend="torchaudio",
                        torch_backend_params=src_info[-1],
                    )

            if flag_iir:
                added_noise, _ = self.augmentor.apply_2nd_iir_response(
                    wav=added_noise, a_coeffs=a_coeffs, b_coeffs=b_coeffs
                )

            if flag_hpf:
                added_noise, _ = self.augmentor.apply_hpf(
                    wav=added_noise,
                    sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    cutoff_freq=hpf_cutoff,
                    q_factor=q_factor,
                )

            if flag_volume:
                if vol_ratio is not None:
                    added_noise, _ = self.augmentor.sox_volume_perturbed(
                        wav=added_noise,
                        vol_ratio=vol_ratio,
                        sr=if_none_else(self.target_sr, self.ori_audio_sr),
                    )
                else:
                    added_noise, (_, _) = self.augmentor.apply_clipping_distortion(
                        wav=added_noise,
                        min_quantile=min_quantile,
                        max_quantile=max_quantile,
                    )

            added_noise = added_noise[..., : self.training_sample_length]

        audio_sr = if_none_else(self.target_sr, self.ori_audio_sr)
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

        # Far parent target: the summed full-RIR interferer
        # speech (post-SIR), zeros when no interferer is present. Used only by an
        # optional far decoder + FarReconstructionLoss; harmless otherwise.
        far_target = (
            background_speech_reference
            if background_speech_reference is not None
            else torch.zeros_like(noisy_speech)
        )
        sample = {
            "noisy_speech": noisy_speech,
            "clean_speech": target_speech,
            "added_noise": added_noise,
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
        """Hook for task subclasses to attach extra per-sample labels.

        No-op in the base noise-suppression dataset. Task subclasses (e.g. the
        distance-cued voice isolation dataset) override this to derive auxiliary
        labels from the simulation metadata and ``sample.update`` them in. The
        base dataset stays unaware of any derived task's label schema.
        """
        return

    def _sample_mix_mode(self, mm_cfg: Dict) -> Dict:
        """Pick one foreground/interferer mixing mode by its prob weight.

        Modes come from ``augmentation_speech.mix_mode.modes``: a ``physical``
        mode keeps the natural post-RIR level ratio,
        the others carry an explicit ``sir_range``. Probs need not sum to 1.
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

    def _sample_turn_script(
        self, n_frames: int, hop: int, sr: int, overlap_cfg: dict
    ) -> tuple:
        """Sample a conversational turn script on the VAD frame grid.

        Alternating near/far turns like a real exchange: the near (target)
        speaker and the far interferer take turns, with small gaps or slight
        boundary overlaps between turns. 50% of rows start with a FAR turn --
        the hardest streaming case (a far monologue with no preceding near
        anchor), which the per-frame Bernoulli fill can never produce.
        Returns (near_frames, far_frames) boolean masks over the frame grid.
        """
        near_len = overlap_cfg.get("turn_near_seconds", [1.5, 3.0])
        far_len = overlap_cfg.get("turn_far_seconds", [2.0, 4.5])
        gap_rng = overlap_cfg.get("turn_gap_seconds", [0.0, 0.4])
        ovl_rng = overlap_cfg.get("turn_overlap_seconds", [0.0, 0.3])
        far_first = torch.rand(1).item() < float(overlap_cfg.get("far_first_prob", 0.5))

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
        overlap_cfg = (self.augmentation_speech_args or {}).get("overlap_control")
        if not overlap_cfg or not overlap_cfg.get("used"):
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

        fade_samples = int(overlap_cfg.get("fade_samples", 400))
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
            turn_taking_prob = float(overlap_cfg.get("turn_taking_prob", 0.0))
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

        no_overlap_prob = float(overlap_cfg.get("no_overlap_prob", 0.25))
        high_overlap_prob = float(overlap_cfg.get("high_overlap_prob", 0.25))
        mid_range = overlap_cfg.get("mid_overlap_range", [0.1, 0.5])
        high_range = overlap_cfg.get("high_overlap_range", [0.5, 1.0])
        fill_range = overlap_cfg.get("fill_on_silent_range", [0.3, 0.5])

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
                "added_noise",
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
        return out
