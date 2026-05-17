import random
from copy import deepcopy
from typing import Dict, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from puresound.audio.noise import add_bg_noise
from puresound.dataset.dynamic_base import DynamicBaseDataset


def if_none_else(a, b):
    if a is not None:
        return a
    else:
        return b


class SpeakerEmbeddingDataset(DynamicBaseDataset):
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
        )

    @property
    def total_speakers(self):
        return len(self.total_spks)

    def __getitem__(self, target_speaker):
        target_speaker, batch_sr = target_speaker
        batch_sr = int(batch_sr) if batch_sr is not None else batch_sr
        target_speech, self.ori_audio_sr, (_, _) = (
            self.choose_an_utterance_by_speaker_name(
                target_speaker_name=target_speaker,
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
            # Here, we assume all clips has same shape equal to [1, L]
            # For multi channel simulation should consider next time. TODO
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
                wav=target_speech, noise=[interfered_speech], snr_list=[sir]
            )
            noisy_speech = noisy_speech[0]

        # Avoiding clipping issue
        [noisy_speech] = self.avoid_audio_clipping(wav_list=[noisy_speech])

        # Speed Perturbation
        sp_idx = None
        if (
            self.augmentation_speed_args
            and self.augmentation_speed_args["used"]
            and torch.rand(1) < self.augmentation_speed_args["prob"]
        ):
            speed = torch.arange(len(self.augmentation_speed_args["speed_change"]))
            sp_idx = random.choice(speed)
            speed = self.augmentation_speed_args["speed_change"][sp_idx]
            noisy_speech, (speed) = self.augmentor.sox_speed_perturbed(
                wav=noisy_speech,
                speed=speed,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )

        # Reverb
        if (
            self.augmentation_reverb_args
            and self.augmentation_reverb_args["used"]
            and torch.rand(1) < self.augmentation_reverb_args["prob"]
        ):
            # RIR's target for noisy is full
            noisy_speech, (rir_id, _) = self.augmentor.apply_rir(
                wav=noisy_speech,
                rir_mode="full",
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
            )
            if noisy_speech.shape[0] != 1:
                noisy_speech = noisy_speech[0].view(1, -1)

        # Noise
        # We collect added noises for if we need to use high SNR noisy speech as ground truth
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

            # unwrap list
            noisy_speech = noisy_speech[0]

            # if dynamic is False, 1 / 4 add white noise
            if (
                dynamic_type == False
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
        if (
            self.augmentation_src_args
            and self.augmentation_src_args["used"]
            and torch.rand(1) < self.augmentation_src_args["prob"]
        ):
            src_target = random.choices(
                self.augmentation_src_args["src_range"],
                weights=self.augmentation_src_args["prob_each"],
            )[0]

            if torch.rand(1) < 0.5:
                src_backend = "sox"
            else:
                src_backend = "torchaudio"

            noisy_speech, _ = self.augmentor.apply_src_effect(
                wav=noisy_speech,
                sr=if_none_else(self.target_sr, self.ori_audio_sr),
                src_sr=src_target,
                src_backend=src_backend,
            )

        # 2nd-IIR response
        if (
            self.augmentation_ir_response_args
            and self.augmentation_ir_response_args["used"]
            and torch.rand(1) < self.augmentation_ir_response_args["prob"]
        ):
            noisy_speech, _ = self.augmentor.apply_2nd_iir_response(wav=noisy_speech)

        # HPF effects
        if (
            self.augmentation_hpf_args
            and self.augmentation_hpf_args["used"]
            and torch.rand(1) < self.augmentation_hpf_args["prob"]
        ):
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

        # Volume perturbed
        if (
            self.augmentation_volume_args
            and self.augmentation_volume_args["used"]
            and torch.rand(1) < self.augmentation_volume_args["prob"]
        ):
            if torch.rand(1) < self.augmentation_volume_args["clipping_prob"]:
                min = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args["clipping_range"]["min"][0],
                    self.augmentation_volume_args["clipping_range"]["min"][1],
                )
                max = torch.FloatTensor(1).uniform_(
                    self.augmentation_volume_args["clipping_range"]["max"][0],
                    self.augmentation_volume_args["clipping_range"]["max"][1],
                )
                noisy_speech, _ = self.augmentor.apply_clipping_distortion(
                    wav=noisy_speech, min_quantile=min, max_quantile=max
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
                noisy_speech, _ = self.augmentor.sox_volume_perturbed(
                    wav=noisy_speech,
                    vol_ratio=gain,
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

        if sp_idx is not None and self.augmentation_speed_args["treat_as_new_speaker"]:
            spk2idx_shift = (sp_idx + 1) * len(self.spk2idx)
        else:
            spk2idx_shift = 0

        return {
            "noisy_speech": noisy_speech,
            "speaker_id": self.spk2idx[target_speaker] + spk2idx_shift,
        }


class SpeakerEmbeddingCollateFunc:
    """Collate functino used in Dataloader."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict):
        col_noisy = []
        col_spkid = []

        for b in batch:
            """
            one batch -- (dict) -- {
                "noisy_speech",
                "speaker_id",
                "audio_sr",
                "audio_length", }
            wav file each with shape [1, L]
            """
            col_noisy.append(b["noisy_speech"].squeeze())
            col_spkid.append(b["speaker_id"])

        return {
            "noisy_speech": pad_sequence(col_noisy, batch_first=True),
            "target": torch.Tensor(col_spkid).type(torch.int64),
        }
