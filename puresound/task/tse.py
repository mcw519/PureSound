import random
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.augmentaion import AudioEffectAugmentor
from puresound.audio.dsp import wav_resampling
from puresound.audio.noise import add_bg_noise
from puresound.audio.volume import rescale_waveform
from puresound.dataset.dynamic_base import DynamicBaseDataset


def if_none_else(a, b):
    if a is not None:
        return a
    else:
        return b


class TargetSpeakerExtractDataset(DynamicBaseDataset):
    def __init__(
        self,
        metafile_path: str,
        min_utt_length_in_seconds: float = 3.0,
        min_utts_in_each_speaker: int = 5,
        target_sr: Optional[int] = None,
        training_sample_length_in_seconds: float = 6.0,
        enroll_speech_args: Optional[Dict] = None,
        audio_gain_nomalized_to: Optional[int] = None,
        augmentation_speech_args: Optional[int] = None,
        augmentation_noise_args: Optional[Dict] = None,
        augmentation_reverb_args: Optional[Dict] = None,
        augmentation_speed_args: Optional[Dict] = None,
        augmentation_ir_response_args: Optional[Dict] = None,
        augmentation_src_args: Optional[Dict] = None,
        augmentation_hpf_args: Optional[Dict] = None,
        augmentation_volume_args: Optional[Dict] = None,
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
        self.enroll_speech_args = enroll_speech_args
        self.init_enroll_augmentor()

    def init_enroll_augmentor(self):
        self.enroll_augmentor = AudioEffectAugmentor()
        if self.enroll_speech_args["add_noise"]["used"]:
            self.enroll_augmentor.load_bg_noise_from_folder(
                self.enroll_speech_args["add_noise"]["noise_folder"]
            )
            print(
                f"Enroll-Augmentor finished load {len(self.enroll_augmentor.bg_noise.keys())} noises"
            )

        if self.enroll_speech_args["add_reverb"]["used"]:
            simulator_args = self.enroll_speech_args["add_reverb"].get("simulator")
            if simulator_args and simulator_args.get("used"):
                self.enroll_augmentor.init_room_simulator(simulator_args)
                print("Enroll-Augmentor initialized physics-based room simulator")
            else:
                self.enroll_augmentor.load_rir_from_folder(
                    self.enroll_speech_args["add_reverb"]["rir_folder"]
                )
                print(
                    f"Enroll-Augmentor finished load {len(self.enroll_augmentor.rir.keys())} rirs"
                )

        print("----" * 30)

    def get_enroll_speech(self, target_speaker, batch_sr):
        # Enrollment speech
        enroll_speech, ori_enroll_audio_sr, (_, _) = (
            self.choose_an_utterance_by_speaker_name(
                target_speaker_name=target_speaker,
                select_channel=0,
                select_with_sr_as_key=batch_sr,
            )
        )
        enroll_speech = self.align_audio_list(
            wav_list=[enroll_speech],
            length=if_none_else(
                None,
                int(
                    ori_enroll_audio_sr
                    * self.enroll_speech_args["enroll_length_seconds"]
                ),
            ),
        )[0]

        if self.enroll_speech_args["gain_nomalized_to"]:
            enroll_speech = rescale_waveform(
                enroll_speech,
                target_lvl=self.enroll_speech_args["gain_nomalized_to"],
                scale="dB",
            )

        if self.enroll_speech_args["add_reverb"]["used"]:
            if torch.rand(1) < self.enroll_speech_args["add_reverb"]["prob"]:
                enroll_speech, (rir_id, _) = self.enroll_augmentor.apply_rir(
                    wav=enroll_speech,
                    rir_mode=self.enroll_speech_args["add_reverb"]["target_rir_type"],
                    sr=if_none_else(self.target_sr, ori_enroll_audio_sr),
                )

                if enroll_speech.shape[0] != 1:
                    enroll_speech = enroll_speech[0].view(1, -1)

        if self.enroll_speech_args["add_noise"]["used"]:
            if torch.rand(1) < self.enroll_speech_args["add_noise"]["prob"]:
                snr = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.enroll_speech_args["add_noise"]["snr_range"][0],
                        self.enroll_speech_args["add_noise"]["snr_range"][1],
                    )
                    .item()
                )

                enroll_speech, (added_noise, _, _) = self.enroll_augmentor.add_bg_noise(
                    wav=enroll_speech,
                    snr_list=[snr],
                    dynamic_type=False,
                    sr=if_none_else(self.target_sr, ori_enroll_audio_sr),
                )

                # unwrap list
                enroll_speech = enroll_speech[0]

            if torch.rand(1) < self.enroll_speech_args["add_noise"]["prob_white_noise"]:
                snr = (
                    torch.FloatTensor(1)
                    .uniform_(
                        self.enroll_speech_args["add_noise"]["white_noise_snr_range"][
                            0
                        ],
                        self.enroll_speech_args["add_noise"]["white_noise_snr_range"][
                            1
                        ],
                    )
                    .item()
                )
                enroll_speech, (added_white_noise, _) = (
                    self.enroll_augmentor.add_bg_white_noise(
                        wav=enroll_speech, snr_list=[snr]
                    )
                )

                # unwrap list
                enroll_speech = enroll_speech[0]

            if isinstance(enroll_speech, list):
                enroll_speech = enroll_speech[0]

        if self.enroll_speech_args["add_volume"]["used"]:
            if torch.rand(1) < self.enroll_speech_args["add_volume"]["prob"]:
                if (
                    torch.rand(1)
                    < self.enroll_speech_args["add_volume"]["clipping_prob"]
                ):
                    min = torch.FloatTensor(1).uniform_(
                        self.enroll_speech_args["add_volume"]["clipping_range"]["min"][
                            0
                        ],
                        self.enroll_speech_args["add_volume"]["clipping_range"]["min"][
                            1
                        ],
                    )
                    max = torch.FloatTensor(1).uniform_(
                        self.enroll_speech_args["add_volume"]["clipping_range"]["max"][
                            0
                        ],
                        self.enroll_speech_args["add_volume"]["clipping_range"]["max"][
                            1
                        ],
                    )
                    enroll_speech, (min_quantile, max_quantile) = (
                        self.enroll_augmentor.apply_clipping_distortion(
                            wav=enroll_speech, min_quantile=min, max_quantile=max
                        )
                    )

                else:
                    gain = (
                        torch.FloatTensor(1)
                        .uniform_(
                            self.enroll_speech_args["add_volume"]["perturbed_range"][0],
                            self.enroll_speech_args["add_volume"]["perturbed_range"][1],
                        )
                        .item()
                    )
                    enroll_speech, (vol_ratio) = (
                        self.enroll_augmentor.sox_volume_perturbed(
                            wav=enroll_speech,
                            vol_ratio=gain,
                            sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        )
                    )

        return enroll_speech

    def __getitem__(self, target_speaker: Tuple[str, int | None]):
        target_speaker, batch_sr = target_speaker
        batch_sr = int(batch_sr) if batch_sr is not None else batch_sr

        # Get enroll speech here
        enroll_speech = self.get_enroll_speech(
            target_speaker=target_speaker, batch_sr=batch_sr
        )

        target_speech, self.ori_audio_sr, (_, _) = (
            self.choose_an_utterance_by_speaker_name(
                target_speaker_name=target_speaker,
                select_channel=0,
                select_with_sr_as_key=batch_sr,
            )
        )

        # Replace target speaker for inactive target cases
        inactive_target_speaker = None
        if (
            self.enroll_speech_args["add_inactive_target"]["used"]
            and torch.rand(1) < self.enroll_speech_args["add_inactive_target"]["prob"]
        ):
            # Samples speakers from overall speaker pool
            if self.target_sr is not None:
                spk_pool = deepcopy(self.total_spks)
            # Samples speakers from same SR conditions
            else:
                spk_pool = deepcopy(list(self.sr_meta[self.ori_audio_sr].keys()))

            spk_pool = set(spk_pool)
            spk_pool.remove(target_speaker)

            inactive_target_speaker = random.sample(sorted(spk_pool), k=1)[0]
            target_speech, self.ori_audio_sr, (_, _) = (
                self.choose_an_utterance_by_speaker_name(
                    target_speaker_name=inactive_target_speaker,
                    select_channel=0,
                    select_with_sr_as_key=batch_sr,
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
        source_level_reverb = self.should_apply_source_level_reverb()
        room_scene = self.augmentor.sample_room_scene() if source_level_reverb else None
        if source_level_reverb:
            noisy_speech, target_speech, _ = self.apply_source_level_target_reverb(
                wav=target_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                room_scene=room_scene,
            )
        else:
            noisy_speech = target_speech.clone()

        # Add interference speech from other speakers
        interfered_speech = []
        if (
            self.augmentation_speech_args
            and self.augmentation_speech_args["used"]
            and torch.rand(1) < self.augmentation_speech_args["prob"]
        ):
            # Samples speakers from overall speaker pool
            if self.target_sr is not None:
                spk_pool = deepcopy(self.total_spks)
            # Samples speakers from same SR conditions
            else:
                spk_pool = deepcopy(list(self.sr_meta[self.ori_audio_sr].keys()))

            spk_pool = set(spk_pool)
            spk_pool.remove(target_speaker)

            if inactive_target_speaker is not None:
                spk_pool.remove(inactive_target_speaker)

            interference_spk_list = random.sample(
                sorted(spk_pool), k=self.augmentation_speech_args["add_n_cases"]
            )
            interference_sr = None if self.target_sr is not None else self.ori_audio_sr
            for spk in interference_spk_list:
                _speech, _sr, _ = self.choose_an_utterance_by_speaker_name(
                    target_speaker_name=spk,
                    select_channel=0,
                    select_with_sr_as_key=interference_sr,
                )
                # Interfered speech has to be same sample rate of target speech
                while _sr != self.ori_audio_sr:
                    _speech, _sr, _ = self.choose_an_utterance_by_speaker_name(
                        target_speaker_name=spk,
                        select_channel=0,
                        select_with_sr_as_key=interference_sr,
                    )
                interfered_speech.append(_speech)

            if source_level_reverb:
                interfered_speech = self.align_audio_list(
                    wav_list=interfered_speech,
                    length=if_none_else(
                        self.training_sample_length,
                        int(self.ori_audio_sr * self.training_sample_length_in_seconds),
                    ),
                    padding_type="zero",
                )
                interfered_speech = [
                    self.apply_source_level_interferer_reverb(
                        wav=speech,
                        sr=if_none_else(self.target_sr, self.ori_audio_sr),
                        room_scene=room_scene,
                    )
                    for speech in interfered_speech
                ]
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
            interfered_speech = (
                torch.cat(interfered_speech, dim=0).sum(dim=0).reshape(1, -1)
            )
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
                wav=noisy_speech if source_level_reverb else target_speech,
                noise=[interfered_speech],
                snr_list=[sir],
            )
            noisy_speech = noisy_speech[0]

            # Treating all speech clips as target speech
            if self.augmentation_speech_args["is_target"]:
                target_speech = noisy_speech.clone()

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

        # Reverb
        if (
            self.augmentation_reverb_args
            and self.augmentation_reverb_args["used"]
            and not source_level_reverb
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

        # Warp target speech to zeros
        if inactive_target_speaker is not None:
            target_speaker = torch.zeros_like(target_speech)

        audio_sr = if_none_else(self.target_sr, self.ori_audio_sr)
        if inactive_target_speaker is not None:
            vad_target = self.create_empty_vad_target(target_speech)
        else:
            vad_target = self.create_vad_target(target_speech, sample_rate=audio_sr)

        sample = {
            "noisy_speech": noisy_speech,
            "clean_speech": target_speech,
            "enroll_speech": enroll_speech,
            "added_noise": added_noise,
            "consistency_noise": noisy_speech - target_speech,
            "speaker_id": self.spk2idx[target_speaker],
            "audio_sr": audio_sr,
            "audio_length": noisy_speech.shape[-1],
        }
        if vad_target is not None:
            sample["vad_target"] = vad_target
        return sample


class TargetSpeakerExtractCollateFunc:
    """Collate functino used in Dataloader."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict):
        col_noisy = []
        col_clean = []
        col_enroll = []
        col_consistency = []
        col_spkid = []
        col_sr = []
        col_length = []
        col_vad = []

        for b in batch:
            """
            one batch -- (dict) -- {
                "noisy_speech",
                "clean_speech",
                "enroll_speech",
                "added_noise",
                "consistency_noise",
                "speaker_id",
                "audio_sr",
                "audio_length", }
            wav file each with shape [1, L]
            """
            col_clean.append(b["clean_speech"].squeeze())
            col_noisy.append(b["noisy_speech"].squeeze())
            col_enroll.append(b["enroll_speech"].squeeze())
            col_consistency.append(b["consistency_noise"].squeeze())
            col_spkid.append(b["speaker_id"])
            col_sr.append(b["audio_sr"])
            col_length.append(b["audio_length"])
            if "vad_target" in b:
                col_vad.append(b["vad_target"].squeeze())

        padded_clean = pad_sequence(col_clean, batch_first=True)  # [N, L]
        padded_noisy = pad_sequence(col_noisy, batch_first=True)  # [N, L]
        padded_enroll = pad_sequence(col_enroll, batch_first=True)  # [N, L]
        padded_consistency = pad_sequence(col_consistency, batch_first=True)  # [N, L]

        out = {
            "clean_speech": padded_clean,
            "noisy_speech": padded_noisy,
            "conditional_speech": padded_enroll,
            "consistency_noise": padded_consistency,
            "target": torch.Tensor(col_spkid).type(torch.int64),
            "sr": torch.Tensor(col_sr),
            "length": torch.Tensor(col_length),
        }
        if col_vad:
            out["vad_target"] = pad_sequence(col_vad, batch_first=True)
        return out
