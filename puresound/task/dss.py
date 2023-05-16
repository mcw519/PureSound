import random
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from puresound.src.audio import AudioAugmentor, AudioIO
from puresound.src.utils import load_text_as_dict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .base import BaseTrainer, TaskDataset


class DssCollateFunc:
    """Collate functino used in Dataloader."""

    def __init__(self):
        pass

    def __call__(self, batch: Any) -> Dict:
        col_key = []
        col_near = []
        col_far = []
        col_process = []
        col_inactive_nearend = []
        col_inactive_farfield = []

        for b in batch:
            """
            one batch -- (dict) -- {'near_wav': near_wav, 'far_wav': far_wav, 'process_wav': process_wav, \
                'inactive_nearend': inactive_nearend, 'inactive_farfield': inactive_farfield}
            wav file each with shape [1, L]
            """
            col_key.append(b["uttid"])
            col_near.append(b["near_wav"].squeeze())
            col_far.append(b["far_wav"].squeeze())
            col_process.append(b["process_wav"].squeeze())
            col_inactive_nearend.append(b["inactive_nearend"])
            col_inactive_farfield.append(b["inactive_farfield"])

        padded_near = pad_sequence(col_near, batch_first=True)  # [N, L]
        padded_far = pad_sequence(col_far, batch_first=True)  # [N, L]
        padded_process = pad_sequence(col_process, batch_first=True)  # [N, L]
        col_inactive_nearend = torch.Tensor(col_inactive_nearend)
        col_inactive_farfield = torch.Tensor(col_inactive_farfield)

        return {
            "uttid": col_key,
            "near_wav": padded_near,
            "far_wav": padded_far,
            "process_wav": padded_process,
            "inactive_nearend": col_inactive_nearend,
            "inactive_farfield": col_inactive_farfield,
        }


class DssDataset(TaskDataset):
    """
    Distance-based target speech separation.
    Online dataset should implement wave_process() to generate parallel data for training.

    Args:
        resample_to: open waveform then resample it.
        max_length: cut each waveform until max_length(seconds).
    """

    def __init__(
        self,
        folder: str,
        resample_to: int,
        max_length: Optional[int] = None,
        noise_folder: Optional[str] = None,
        vol_perturbed: Optional[tuple] = None,
        speed_perturbed: bool = False,
    ):

        self.max_length = max_length
        self.noise_folder = noise_folder
        self.speed_perturbed = speed_perturbed
        self.vol_perturbed = vol_perturbed
        super().__init__(folder, resample_to=resample_to)

        if (
            self.noise_folder is not None
            or self.speed_perturbed
            or self.vol_perturbed is not None
        ):
            self.create_augmentor()
        else:
            self.augmentor = None

    @property
    def folder_content(self):
        _content = {
            "wav2scp": "wav2scp.txt",  # clean wav path
            "ref2near": "ref2near.txt",  # near-end wav path
            "ref2far": "ref2far.txt",  # far-field wav path
        }

        return _content

    def __getitem__(self, index: int) -> Dict:
        key = self.idx_df[index]
        feats = self.get_feature(key)
        process_wav = feats["process_wav"].view(1, -1)
        near_wav = feats["near_wav"].view(1, -1)
        far_wav = feats["far_wav"].view(1, -1)
        inactive_nearend = feats["inactive_nearend"]
        inactive_farfield = feats["inactive_farfield"]
        return {
            "uttid": key,
            "process_wav": process_wav,
            "near_wav": near_wav,
            "far_wav": far_wav,
            "inactive_nearend": inactive_nearend,
            "inactive_farfield": inactive_farfield,
        }

    def get_feature(self, key: str) -> Dict:
        """noisy_wav(2 speaker mixed) -> speed perturbed -> noise inject"""
        wav, sr = AudioIO.open(f_path=self.df[key]["wav2scp"])
        if sr != self.resample_to:
            wav = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.resample_to
            )(wav)

        if wav.shape[0] != 1:
            wav = wav[0].view(1, -1)  # ignore multi-channel

        near_wav, sr = AudioIO.open(f_path=self.df[key]["ref2near"])
        if sr != self.resample_to:
            near_wav = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.resample_to
            )(near_wav)

        if near_wav.shape[0] != 1:
            near_wav = near_wav[0].view(1, -1)  # ignore multi-channel

        far_wav, sr = AudioIO.open(f_path=self.df[key]["ref2far"])
        if sr != self.resample_to:
            far_wav = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.resample_to
            )(far_wav)

        if far_wav.shape[0] != 1:
            far_wav = far_wav[0].view(1, -1)  # ignore multi-channel

        if self.max_length is not None:
            # only using segmented audio
            target_len = sr * self.max_length
            if wav.shape[-1] > target_len:
                offset = random.randint(0, int(wav.shape[-1]) - target_len)
                wav = wav[:, offset : offset + target_len]
                near_wav = near_wav[:, offset : offset + target_len]
                far_wav = far_wav[:, offset : offset + target_len]
            else:
                pad_zero = torch.zeros(1, target_len - wav.shape[-1])
                wav = torch.cat([wav, pad_zero], dim=-1)
                pad_zero = torch.zeros(1, target_len - near_wav.shape[-1])
                near_wav = torch.cat([near_wav, pad_zero], dim=-1)
                pad_zero = torch.zeros(1, target_len - far_wav.shape[-1])
                far_wav = torch.cat([far_wav, pad_zero], dim=-1)
        else:
            target_len = wav.shape[1]  # wav is a tensor with shape [channel, N_sample]

        # Start audio augmentation
        if self.augmentor:
            process_wav, (speed, _) = self.wave_process(wav)
        else:
            process_wav, speed = wav, None

        # warp clean_wav with same speed perturbed
        if speed is not None:
            near_wav, _ = self.augmentor.sox_speed_perturbed(near_wav, speed)
            far_wav, _ = self.augmentor.sox_speed_perturbed(far_wav, speed)

        # random adjust volumn on both clean and process wav
        if self.vol_perturbed is not None:
            if not isinstance(self.vol_perturbed, tuple):
                min_ratio = float(self.vol_perturbed.strip().split(",")[0])
                max_ratio = float(self.vol_perturbed.strip().split(",")[1])
            else:
                min_ratio, max_ratio = self.vol_perturbed

            perturbed_ratio = torch.FloatTensor(1).uniform_(min_ratio, max_ratio).item()
            near_wav = self.augmentor.sox_volumn_perturbed(near_wav, perturbed_ratio)
            near_wav = torch.clamp(near_wav, min=-1, max=1)
            far_wav = self.augmentor.sox_volumn_perturbed(far_wav, perturbed_ratio)
            far_wav = torch.clamp(far_wav, min=-1, max=1)
            process_wav = self.augmentor.sox_volumn_perturbed(
                process_wav, perturbed_ratio
            )
            process_wav = torch.clamp(process_wav, min=-1, max=1)

        inactive_nearend = True if near_wav.sum() == 0 else False
        if inactive_nearend:
            near_wav = process_wav.clone()

        inactive_farfield = True if far_wav.sum() == 0 else False
        if inactive_farfield:
            far_wav = process_wav.clone()

        return {
            "near_wav": near_wav,
            "far_wav": far_wav,
            "process_wav": process_wav,
            "inactive_nearend": inactive_nearend,
            "inactive_farfield": inactive_farfield,
        }

    def create_augmentor(self) -> None:
        self.augmentor = AudioAugmentor(
            sample_rate=self.resample_to, convolve_mode="fft"
        )
        if self.noise_folder:
            self.augmentor.load_bg_noise_from_folder(self.noise_folder)
            print(f"Finished load {len(self.augmentor.bg_noise.keys())} noises")

    def wave_process(self, x: torch.Tensor) -> Tuple:
        speed, snr = None, None
        backup = x.clone()

        # speed perturbed
        if self.speed_perturbed and torch.rand(1) < 0.5:
            speed = float(torch.FloatTensor(1).uniform_(0.9, 1.1))
            x, _ = self.augmentor.sox_speed_perturbed(x, speed)

        # noise inject
        if self.noise_folder is not None and torch.rand(1) < 0.8:
            snr = float(torch.FloatTensor(1).uniform_(-5, 15))
            x = self.augmentor.add_bg_noise(x, [snr])[0]

        # error handling
        if torch.isnan(x).any():
            print(f"warning this augment has nan, snr={snr}, speed={speed}")
            x, speed = backup, None

        return x, (speed, snr)


class DssTask(BaseTrainer):
    def __init__(self, hparam, device_backend, train_dataloader, dev_dataloader):
        super().__init__(hparam, device_backend)
        self.overall_step = 0
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

    def train_one_epoch(self, current_epoch):
        step = 0
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
            self.overall_step += 1
            step += 1
            near_wav = batch["near_wav"].to(self.device)  # [N, L]
            far_wav = batch["far_wav"].to(self.device)  # [N, L]
            noisy_wav = batch["process_wav"].to(self.device)  # [N, L]
            inactive_nearend = batch["inactive_nearend"]  # [N]
            inactive_farfield = batch["inactive_farfield"]  # [N]

            self.optimizer.zero_grad()

            # Model forward
            inactive_inp = torch.stack([inactive_nearend, inactive_farfield], dim=1)
            clean_wav = torch.stack([near_wav, far_wav], dim=1)
            loss = self.model(
                noisy=noisy_wav, ref_clean=clean_wav, inactive_labels=inactive_inp
            )
            loss = torch.mean(loss, dim=0)  # aggregate loss from each device
            print(f"epoch: {current_epoch}, iter: {batch_idx+1}, batch_loss: {loss}")
            total_loss += loss.item()
            loss.backward()

            if self.hparam["OPTIMIZER"]["gradiend_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hparam["OPTIMIZER"]["gradiend_clip"]
                )

            self.optimizer.step()

            if self.tf_writer:
                _log_name = "train/batch_loss"
                self.tf_writer.update_step_loss(_log_name, loss, self.overall_step)

        return {"total_loss": total_loss / step}

    def compute_dev_loss(self, current_epoch):
        step = 0
        dev_total_loss = 0.0

        for _, batch in enumerate(tqdm(self.dev_dataloader)):
            step += 1
            near_wav = batch["near_wav"].to(self.device)  # [N, L]
            far_wav = batch["far_wav"].to(self.device)  # [N, L]
            noisy_wav = batch["process_wav"].to(self.device)  # [N, L]
            inactive_nearend = batch["inactive_nearend"]  # [N]
            inactive_farfield = batch["inactive_farfield"]  # [N]

            with torch.no_grad():
                inactive_inp = torch.stack([inactive_nearend, inactive_farfield], dim=1)
                clean_wav = torch.stack([near_wav, far_wav], dim=1)
                loss = self.model(
                    noisy=noisy_wav, ref_clean=clean_wav, inactive_labels=inactive_inp
                )
                loss = torch.mean(loss, dim=0)  # aggregate loss from each device
                dev_total_loss += loss.item()

        print(f"dev average loss: {dev_total_loss / step}")
        return {"total_loss": dev_total_loss / step}

    def gen_logging(self, epoch: int, prefix: str):
        """
        Generate samples on tensorboard for loggin
        """
        test_audio_dct = load_text_as_dict(
            f"{self.hparam['DATASET']['eval']}/wav2scp.txt"
        )
        resample_to = self.hparam["DATASET"]["sample_rate"]

        for _, key in enumerate(test_audio_dct.keys()):
            uttid = key
            print(f"Running inference: {uttid}")
            wav, sr = AudioIO.open(f_path=test_audio_dct[key][0])
            if sr != resample_to:
                wav = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=resample_to
                )(wav)

            wav = wav.to(self.device)

            if isinstance(self.model, torch.nn.DataParallel):
                enh_wav = self.model.module.inference(noisy=wav)
            else:
                enh_wav = self.model.inference(noisy=wav)

            if self.tf_writer:
                self.tf_writer.add_ep_audio(
                    f"{prefix}{uttid}_near.wav", enh_wav[:, 0, :], epoch, resample_to
                )
                self.tf_writer.add_ep_audio(
                    f"{prefix}{uttid}_far.wav", enh_wav[:, 1, :], epoch, resample_to
                )
