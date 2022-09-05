import random
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from puresound.src.audio import AudioAugmentor, AudioIO
from puresound.src.utils import load_text_as_dict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .base import BaseTrainer, TaskDataset


class NsCollateFunc:
    """Collate functino used in Dataloader."""
    def __init__(self):
        pass

    def __call__(self, batch: Any) -> Dict:
        col_key = []
        col_clean = []
        col_process = []

        for b in batch:
            """
            one batch -- (dict) -- {'uttid': key, 'process_wav': process_wav, 'clean_wav': clean_wav}
            wav file each with shape [1, L]
            """
            col_key.append(b['uttid'])
            col_clean.append(b['clean_wav'].squeeze())
            col_process.append(b['process_wav'].squeeze())
                               
        padded_clean = pad_sequence(col_clean, batch_first=True) # [N, L]
        padded_process = pad_sequence(col_process, batch_first=True) # [N, L]
        
        return {'uttid': col_key, 'clean_wav': padded_clean, 'process_wav': padded_process}


class NsDataset(TaskDataset):
    """
    Noise suppression dataset.
    Online dataset should implement wave_process() to generate parallel data for training.

    Args:
        resample_to: open waveform then resample it.
        max_length: cut each waveform until max_length(seconds).
    """
    def __init__(self,
                folder: str,
                resample_to: int,
                max_length: Optional[int] = None,
                noise_folder: Optional[str] = None,
                rir_folder: Optional[str] = None,
                rir_mode: str = 'image',
                vol_perturbed: Optional[tuple] = None,
                speed_perturbed: bool = False):
        
        self.max_length = max_length
        self.noise_folder = noise_folder
        self.rir_folder = rir_folder
        self.rir_mode = rir_mode
        self.speed_perturbed = speed_perturbed
        self.vol_perturbed = vol_perturbed
        super().__init__(folder, resample_to=resample_to)

        if self.noise_folder is not None or self.rir_folder is not None or self.speed_perturbed or self.vol_perturbed is not None:
            self.create_augmentor()
        else:
            self.augmentor = None

    @property
    def folder_content(self):
        _content = {
            'wav2scp': 'wav2scp.txt', # clean wav path
            'wav2ref': 'wav2ref.txt', # clean wav path
            }
        
        return _content

    def __getitem__(self, index: int) -> Dict:
        key = self.idx_df[index]
        feats = self.get_feature(key)
        process_wav = feats['process_wav'].view(1, -1)
        clean_wav = feats['clean_wav'].view(1, -1)
        return {'uttid': key, 'process_wav': process_wav, 'clean_wav': clean_wav}

    def get_feature(self, key: str) -> Dict:
        """noisy_wav(2 speaker mixed) -> speed perturbed -> rir reverb -> noise inject"""
        wav, sr = AudioIO.open(f_path=self.df[key]['wav2scp'])
        if sr != self.resample_to:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(wav)

        if wav.shape[0] != 1: wav = wav[0].view(1, -1) # ignore multi-channel

        clean_wav, sr = AudioIO.open(f_path=self.df[key]['wav2ref'])
        if sr != self.resample_to:
            clean_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(clean_wav)
        
        if clean_wav.shape[0] != 1: clean_wav = clean_wav[0].view(1, -1) # ignore multi-channel
        
        if self.max_length is not None:
            # only using segmented audio
            target_len = sr * self.max_length
            if wav.shape[-1] > target_len:
                offset = random.randint(0, int(wav.shape[-1]) - target_len)
                # Avoid choice the zero tensor as target
                while clean_wav[:, offset : offset + target_len].sum() == 0:
                    offset = random.randint(0, int(wav.shape[-1]) - target_len)
                    if clean_wav[:, offset : offset + target_len].sum() != 0: break
                wav = wav[:, offset : offset + target_len]
                clean_wav = clean_wav[:, offset : offset + target_len]
            else:
                pad_zero = torch.zeros(1, target_len - wav.shape[-1])
                wav = torch.cat([wav, pad_zero], dim=-1)
                pad_zero = torch.zeros(1, target_len - clean_wav.shape[-1])
                clean_wav = torch.cat([clean_wav, pad_zero], dim=-1)
        else:
            target_len = wav.shape[1] # wav is a tensor with shape [channel, N_sample]

        # Start audio augmentation
        if self.augmentor:
            process_wav, (speed, _, rir_id, rir_ch) = self.wave_process(wav)
        else:
            process_wav, speed, rir_id, rir_ch = wav, None, None, None
        
        # warp clean_wav with same speed perturbed
        if speed is not None:
            clean_wav, _ = self.augmentor.sox_speed_perturbed(clean_wav, speed)
        
        # warp clean_wav with same rir reverb type, but different reverb mode
        # 1. target image: warp same rir channel impaulse
        # 2. target direct: warp same rir channel impaulse with 6ms center from peak
        # 3. target early: warp same rir channel impaulse with 50ms center from peak
        if rir_id is not None and self.rir_mode != 'anechoic':
            clean_wav = self.augmentor.apply_rir_by_key(clean_wav, rir_id, choose_ch=rir_ch, rir_mode=self.rir_mode)
        
        # random adjust volumn on both clean and process wav
        if self.vol_perturbed is not None:
            if not isinstance(self.vol_perturbed, tuple):
                min_ratio = float(self.vol_perturbed.strip().split(',')[0])
                max_ratio = float(self.vol_perturbed.strip().split(',')[1])
            else:
                min_ratio, max_ratio = self.vol_perturbed
            perturbed_ratio = torch.FloatTensor(1).uniform_(min_ratio, max_ratio).item()
            clean_wav = self.augmentor.sox_volumn_perturbed(clean_wav, perturbed_ratio)
            clean_wav = torch.clamp(clean_wav, min=-1, max=1)
            process_wav = self.augmentor.sox_volumn_perturbed(process_wav, perturbed_ratio)
            process_wav = torch.clamp(process_wav, min=-1, max=1)
        
        return {'clean_wav': clean_wav, 'process_wav': process_wav}
    
    def create_augmentor(self) -> None:
        self.augmentor = AudioAugmentor(sample_rate=self.resample_to, convolve_mode='fft')
        if self.noise_folder:
            self.augmentor.load_bg_noise_from_folder(self.noise_folder)
            print(f"Finished load {len(self.augmentor.bg_noise.keys())} noises")
        
        if self.rir_folder:
            self.augmentor.load_rir_from_folder(self.rir_folder)
            print(f"Finished load {len(self.augmentor.rir.keys())} rirs")

    def wave_process(self, x: torch.Tensor) -> Tuple:
        speed, snr, rir_id, rir_ch = None, None, None, None
        backup = x.clone()

        # speed perturbed
        if self.speed_perturbed and torch.rand(1) < 0.5:
            speed = float(torch.FloatTensor(1).uniform_(0.9, 1.1))
            x, _ = self.augmentor.sox_speed_perturbed(x, speed)

        # rir inject
        if self.rir_folder is not None and torch.rand(1) < 0.8:
            x, rir_id, rir_ch = self.augmentor.apply_rir(x)

        # noise inject
        if self.noise_folder is not None and torch.rand(1) < 0.8:
            snr = float(torch.FloatTensor(1).uniform_(-5, 15))
            x = self.augmentor.add_bg_noise(x, [snr])[0]
        
        # error handling
        if torch.isnan(x).any():
            print(f'warning this augment has nan, snr={snr}, speed={speed}, rir_id={rir_id}')
            x, speed, rir_id = backup, None, None
        
        return x, (speed, snr, rir_id, rir_ch)


class NsTask(BaseTrainer):
    def __init__(self, hparam, device_backend, train_dataloader, dev_dataloader):
        super().__init__(hparam, device_backend)
        self.overall_step = 0
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
    
    def train_one_epoch(self, current_epoch):
        step = 0
        total_loss = 0.
        
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
            self.overall_step += 1
            step += 1
            clean_wav = batch['clean_wav'].to(self.device) # [N, L]
            noisy_wav = batch['process_wav'].to(self.device) # [N, L]
            
            self.optimizer.zero_grad()
            
            # Model forward
            loss = self.model(noisy=noisy_wav, enroll=None, ref_clean=clean_wav)
            loss = torch.mean(loss, dim=0) # aggregate loss from each device
            print(f"epoch: {current_epoch}, iter: {batch_idx+1}, batch_loss: {loss}")
            total_loss += loss.item()
            loss.backward()

            if self.hparam['OPTIMIZER']['gradiend_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparam['OPTIMIZER']['gradiend_clip'])
            
            self.optimizer.step()
            
            if self.tf_writer:
                _log_name = 'train/batch_loss'
                self.tf_writer.update_step_loss(_log_name, loss, self.overall_step)
                
        return {'total_loss': total_loss / step}
    
    def compute_dev_loss(self, current_epoch):
        step = 0
        dev_total_loss = 0.
        
        for _, batch in enumerate(tqdm(self.dev_dataloader)):
            step += 1
            clean_wav = batch['clean_wav'].to(self.device) # [N, L]
            noisy_wav = batch['process_wav'].to(self.device) # [N, L]
            
            with torch.no_grad():
                loss = self.model(noisy=noisy_wav, enroll=None, ref_clean=clean_wav)
                loss = torch.mean(loss, dim=0) # aggregate loss from each device
                dev_total_loss += loss.item()

        print(f"dev average loss: {dev_total_loss / step}")
        return {'total_loss': dev_total_loss / step}
    
    def gen_logging(self, epoch: int, prefix: str):
        """
        Generate samples on tensorboard for loggin
        """
        test_audio_dct = load_text_as_dict(f"{self.hparam['DATASET']['eval']}/wav2scp.txt")
        resample_to = self.hparam['DATASET']['sample_rate']

        for _, key in enumerate(test_audio_dct.keys()):
            uttid = key
            print(f"Running inference: {uttid}")
            wav, sr = AudioIO.open(f_path=test_audio_dct[key][0])
            if sr != resample_to:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_to)(wav)

            wav = wav.to(self.device)

            if isinstance(self.model, torch.nn.DataParallel):
                enh_wav = self.model.module.inference(noisy=wav, enroll=None)
            else:
                enh_wav = self.model.inference(noisy=wav, enroll=None)
        
            if self.tf_writer:
                self.tf_writer.add_ep_audio(f"{prefix}{uttid}.wav", enh_wav, epoch, resample_to)
