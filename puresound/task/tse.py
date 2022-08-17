import random
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from puresound.src.audio import AudioAugmentor, AudioIO
from puresound.src.utils import load_text_as_dict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .base import BaseTrainer, TaskDataset


class TseCollateFunc:
    """Collate functino used in Dataloader."""
    def __init__(self):
        pass

    def __call__(self, batch: Any) -> Dict:
        col_key = []
        col_clean = []
        col_process = []
        col_spks = []
        col_enroll = []
        col_inactive_utts = []

        for b in batch:
            """
            one batch -- (dict) -- {'uttid': key, 'process_wav': process_wav, 'clean_wav': clean_wav, 'enroll_wav': enroll_wav, 'inactive': inactive}
            wav file each with shape [1, L]
            """
            col_key.append(b['uttid'])
            col_clean.append(b['clean_wav'].squeeze())
            col_process.append(b['process_wav'].squeeze())
            col_enroll.append(b['enroll_wav'].squeeze())
            col_spks.append(b['spk_label'])
            col_inactive_utts.append(b['inactive'])
                   
        padded_clean = pad_sequence(col_clean, batch_first=True) # [N, L]
        padded_process = pad_sequence(col_process, batch_first=True) # [N, L]
        padded_enroll = pad_sequence(col_enroll, batch_first=True) # [N, L]
        col_spks = torch.LongTensor(col_spks)
        col_inactive_utts = torch.Tensor(col_inactive_utts)

        return {'uttid': col_key, 'clean_wav': padded_clean, 'process_wav': padded_process, 'enroll_wav': padded_enroll, 'spk_label': col_spks, 'inactive_utts': col_inactive_utts}


class TseDataset(TaskDataset):
    """
    Target speech extraction or target speech activity dataset.
    Online dataset should implement wave_process() to generate parallel data for training.

    Args:
        resample_to: open waveform then resample it.
        max_length: cut each waveform until max_length(seconds).
        enroll_rule: longest, shortest, fixed_length of full.
        enroll_augment: same data augmentation methods using between both separation and speaker embedding training.
        rir_folder: path of rir corpus.
        rir_mode: method of anechoic, image, direct or early reverberation
        vol_perturbed: volume data augmentation
        speed_perturbed: speed up or slow down augmentation
        single_spk_pb: probability of how much single speech cases in the on-the-fly dataset.
        inactive_training: probability of how much inactive speech cases in the on-the-fly dataset.
        is_vad_dataset: if true, replace the target as VAD labels.
    """
    def __init__(self,
                folder: str,
                resample_to: int,
                max_length: Optional[int] = None,
                enroll_rule: Optional[str] = None,
                enroll_augment: bool = False,
                noise_folder: Optional[str] = None,
                rir_folder: Optional[str] = None,
                rir_mode: str = 'image',
                vol_perturbed: Optional[tuple] = None,
                speed_perturbed: bool = False,
                single_spk_pb: float = 0.,
                inactive_training: float = 0.,
                is_vad_dataset: bool = False
                ):
        self.max_length = max_length
        self.noise_folder = noise_folder
        self.rir_folder = rir_folder
        self.rir_mode = rir_mode
        self.speed_perturbed = speed_perturbed
        self.vol_perturbed = vol_perturbed
        self.single_spk_pb = single_spk_pb
        self.inactive_training = inactive_training
        self.enroll_rule = enroll_rule
        self.enroll_augment = enroll_augment
        self.is_vad_dataset = is_vad_dataset
        super().__init__(folder, resample_to=resample_to)

        if self.noise_folder is not None or self.rir_folder is not None or self.speed_perturbed or self.vol_perturbed is not None:
            self.create_augmentor()
        else:
            self.augmentor = None
        
        self.create_df2spk()

    @property
    def folder_content(self):
        _content = {
            'wav2scp': 'wav2scp.txt', # noisy wav path
            'wav2ref': 'wav2ref.txt', # clean wav path
            'ref2list': 'ref2list.txt', # target enrollment speech list
            'ref2spk': 'ref2spk.txt', # target speaker id
            'wav2spk': 'wav2spk.txt', # speakers in mixture
            }
        
        if self.is_vad_dataset:
            _content.update({'ref2vad': 'ref2vad.txt'})
        
        return _content

    def __getitem__(self, index: int) -> Dict:
        key = self.idx_df[index]
        feats = self.get_feature(key)
        process_wav = feats['process_wav'].view(1, -1)
        clean_wav = feats['clean_wav'].view(1, -1)
        enroll_wav = feats['enroll_wav'].view(1, -1)
        spk_label = feats['spk_label']
        inactive = feats['inactive']
        return {'uttid': key, 'process_wav': process_wav, 'clean_wav': clean_wav, 'enroll_wav': enroll_wav, 'spk_label': spk_label, 'inactive': inactive}

    def get_feature(self, key: str) -> Dict:
        """noisy_wav(2 speaker mixed) -> speed perturbed -> rir reverb -> noise inject"""
        spk_label = self.ref2spk[self.df[key]['ref2spk']]
        wav, sr = AudioIO.open(f_path=self.df[key]['wav2scp'])
        if sr != self.resample_to:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(wav)

        if wav.shape[0] != 1: wav = wav[0].view(1, -1) # ignore multi-channel

        clean_wav, sr = AudioIO.open(f_path=self.df[key]['wav2ref']) if not self.is_vad_dataset else AudioIO.open(f_path=self.df[key]['ref2vad'])
        if sr != self.resample_to:
            clean_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(clean_wav)
        
        if clean_wav.shape[0] != 1: clean_wav = clean_wav[0].view(1, -1) # ignore multi-channel

        # Single target speaker speech cases
        if torch.rand(1) < self.single_spk_pb:
            if not self.is_vad_dataset:
                # TSE
                wav = clean_wav.clone()
            else:
                # PVAD
                wav, sr = AudioIO.open(f_path=self.df[key]['wav2ref']) # replaced input waveform by single target cases
                if sr != self.resample_to:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(wav)

        # Random pick another speaker to replace the mixed speech, and also replace reference speech by mixture
        # Format of wav2spk: "corpus_uttid_source s1-s2-s3"
        if torch.rand(1) < self.inactive_training:
            current_spks = self.df[key]['wav2spk'].split('-')
            pick_key = random.sample(list(self.df.keys()), 1)[0]
            pick_sid = int(pick_key.strip().split('_')[-1][-1]) - 1 # s1, s2 or s3
            pick_spk = self.df[pick_key]['wav2spk'].split('-')[pick_sid]
            while pick_spk in current_spks:
                pick_key = random.sample(list(self.df.keys()), 1)[0]
                pick_sid = int(pick_key.strip().split('_')[-1][-1]) - 1 # s1, s2 or s3
                pick_spk = self.df[pick_key]['wav2spk'].split('-')[pick_sid]
                if pick_spk not in current_spks:
                    break
            
            # replace noisy mixture, keeping speaker embedding for training speaker net
            enroll_wav = self.load_enroll(key, mode=self.enroll_rule)

            if torch.rand(1) > 0.5:
                # double talk interference
                wav, sr = AudioIO.open(f_path=self.df[pick_key]['wav2scp'])
            else:
                # single talk interference
                wav, sr = AudioIO.open(f_path=self.df[pick_key]['wav2ref'])

            if sr != self.resample_to:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(wav)

            if wav.shape[0] != 1: wav = wav[0].view(1, -1) # ignore multi-channel

            if not self.is_vad_dataset:
                clean_wav = wav.clone()
            else:
                clean_wav = torch.zeros_like(wav)

            inactive = True
            
        else:
            enroll_wav = self.load_enroll(key, mode=self.enroll_rule)
            inactive = False

        if self.resample_to is not None: assert sr == self.resample_to
        
        if self.max_length is not None:
            # only using segmented audio
            target_len = sr * self.max_length
            if wav.shape[-1] > target_len:
                offset = random.randint(0, int(wav.shape[-1]) - target_len)
                # Avoid choice the zero tensor as target
                while clean_wav[:, offset : offset + target_len].sum() == 0 and not self.is_vad_dataset:
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
        if speed is not None and not self.is_vad_dataset:
            clean_wav, _ = self.augmentor.sox_speed_perturbed(clean_wav, speed)
        
        # warp clean_wav with same rir reverb type, but different reverb mode
        # 1. target image: warp same rir channel impaulse
        # 2. target direct: warp same rir channel impaulse with 6ms center from peak
        # 3. target early: warp same rir channel impaulse with 50ms center from peak
        if rir_id is not None and self.rir_mode != 'anechoic' and not self.is_vad_dataset:
            clean_wav = self.augmentor.apply_rir_by_key(clean_wav, rir_id, choose_ch=rir_ch, rir_mode=self.rir_mode)
        
        # random adjust volumn on both clean and process wav
        if self.vol_perturbed is not None:
            if not isinstance(self.vol_perturbed, tuple):
                min_ratio = float(self.vol_perturbed.strip().split(',')[0])
                max_ratio = float(self.vol_perturbed.strip().split(',')[1])
            else:
                min_ratio, max_ratio = self.vol_perturbed
            perturbed_ratio = torch.FloatTensor(1).uniform_(min_ratio, max_ratio).item()
            if not self.is_vad_dataset:
                clean_wav = self.augmentor.sox_volumn_perturbed(clean_wav, perturbed_ratio)
                clean_wav = torch.clamp(clean_wav, min=-1, max=1)
            process_wav = self.augmentor.sox_volumn_perturbed(process_wav, perturbed_ratio)
            process_wav = torch.clamp(process_wav, min=-1, max=1)
            enroll_wav = self.augmentor.sox_volumn_perturbed(enroll_wav, perturbed_ratio)
            enroll_wav = torch.clamp(enroll_wav, min=-1, max=1)
        
        if inactive:
            # clean_wav in IS case is input mixture
            clean_wav = process_wav.clone() if not self.is_vad_dataset else torch.zeros_like(process_wav)
            # spk_label = self.ref2spk[self.df[pick_key]['ref2spk']]
        
        return {'clean_wav': clean_wav, 'process_wav': process_wav, 'enroll_wav': enroll_wav, 'spk_label': spk_label, 'inactive': inactive}
    
    def load_enroll(self, key: Any, mode: Optional[str] = None) -> torch.Tensor:
        min_length = self.resample_to * 1
        max_length = self.resample_to * 15
        enroll_list = self.df[key]['ref2list']
        
        if not isinstance(enroll_list, list): enroll_list = [enroll_list] # Handling type error

        target_lvl = torch.normal(mean=torch.Tensor([-28]), std=torch.Tensor([10]).sqrt())
        target_lvl = round(target_lvl.item(), 1)

        if mode is None:
            pick_id = random.sample(list(range(len(enroll_list))), 1)[0]
            enroll_wav, sr = AudioIO.open(f_path=enroll_list[pick_id], target_lvl=target_lvl)
            if sr != self.resample_to:
                enroll_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(enroll_wav)
            
            while enroll_wav.shape[-1] < min_length:
                del enroll_list[pick_id]
                
                if enroll_list == []: break
                pick_id = random.sample(list(range(len(enroll_list))), 1)[0]
                temp_wav, sr = AudioIO.open(f_path=enroll_list[pick_id], target_lvl=target_lvl)
                
                if sr != self.resample_to:
                    temp_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(temp_wav)
                enroll_wav = torch.cat([enroll_wav, temp_wav], dim=-1)

                if enroll_wav.shape[-1] > min_length: break
        
        elif mode == 'longest' or mode == 'shortest':
            enroll_length = []
            for fpath in enroll_list:
                enroll_length.append(AudioIO.audio_info(fpath)[1])
            
            pick_id = torch.argmax(torch.Tensor(enroll_length)) if mode == 'longest' else torch.argmin(torch.Tensor(enroll_length))
            enroll_wav, sr = AudioIO.open(f_path=enroll_list[pick_id], target_lvl=target_lvl)
        
        elif mode == 'fixed_length':
            enroll_len = self.resample_to * 5
            pick_id = random.sample(list(range(len(enroll_list))), 1)[0]
            enroll_wav, sr = AudioIO.open(f_path=enroll_list[pick_id], target_lvl=target_lvl)
            if sr != self.resample_to:
                enroll_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(enroll_wav)
            
            if enroll_wav.shape[-1] > enroll_len:
                offset = random.randint(0, int(enroll_wav.shape[-1]) - enroll_len)
                enroll_wav = enroll_wav[:, offset : offset + enroll_len]
        
        elif mode == 'full':
            enroll_wav_list = []
            for idx in range(len(enroll_list)):
                enroll_wav, sr = AudioIO.open(f_path=enroll_list[idx], target_lvl=target_lvl)
                if sr != self.resample_to:
                    enroll_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample_to)(enroll_wav)

                enroll_wav_list.append(enroll_wav)
            enroll_wav = torch.cat(enroll_wav_list, dim=-1)
            
        else:
            raise NameError

        if self.augmentor and self.enroll_augment:
            backup = enroll_wav.clone()
            # rir inject
            if self.rir_folder is not None and torch.rand(1) < 0.5: # ??% add RIRs
                enroll_wav, rir_id, rir_ch = self.augmentor.apply_rir(enroll_wav)

            # noise inject
            if self.noise_folder is not None and torch.rand(1) < 0.5: # ??% add noise
                snr = float(torch.FloatTensor(1).uniform_(5, 15))
                enroll_wav = self.augmentor.add_bg_noise(enroll_wav, [snr])[0]
            
            # error handling
            if torch.isnan(enroll_wav).any():
                print(f'Enroll augmentation warning: this augment has nan, snr={snr}, rir_id={rir_id}')
                enroll_wav = backup
        
        return enroll_wav[:, :max_length]
        
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
            snr = float(torch.FloatTensor(1).uniform_(5, 15))
            x = self.augmentor.add_bg_noise(x, [snr])[0]
        
        # error handling
        if torch.isnan(x).any():
            print(f'warning this augment has nan, snr={snr}, speed={speed}, rir_id={rir_id}')
            x, speed, rir_id = backup, None, None
        
        return x, (speed, snr, rir_id, rir_ch)
    
    def create_df2spk(self):
        total_spkid = set([self.df[key]['ref2spk'] for key in self.df.keys()])
        self.ref2spk = {}
        for idx, spkid in enumerate(sorted(total_spkid)):
            self.ref2spk[spkid] = idx

    def sampler_meta(self):
        spk2utt_dct = {}
        for idx in range(len(self.df)):
            key = self.idx_df[idx]
            spk = self.df[key]['ref2spk']
            if spk not in list(spk2utt_dct.keys()):
                spk2utt_dct[spk] = [idx]
            
            else:
                spk2utt_dct[spk].append(idx)
    
        return spk2utt_dct


class TseTask(BaseTrainer):
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
            enroll_wav = batch['enroll_wav'].to(self.device) # [N, L]
            inactive_utts = batch['inactive_utts'] # [N]
            target_spk_class = batch['spk_label'].to(self.device) # [N]
            
            self.optimizer.zero_grad()
            
            # Model forward
            loss, loss_detail = self.model(noisy=noisy_wav, enroll=enroll_wav, ref_clean=clean_wav, spk_class=target_spk_class, inactive_labels=inactive_utts,
                                    alpha=self.hparam['LOSS']['alpha'], return_loss_detail=True)
            loss = torch.mean(loss, dim=0) # aggregate loss from each device
            signal_loss = torch.mean(loss_detail[0], dim=0)
            class_loss = torch.mean(loss_detail[1], dim=0)
            print(f"epoch: {current_epoch}, iter: {batch_idx+1}, batch_loss: {loss}, signal_loss: {signal_loss}, class_loss: {class_loss}")
            total_loss += loss.item()
            loss.backward()

            if self.hparam['OPTIMIZER']['gradiend_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparam['OPTIMIZER']['gradiend_clip'])
            
            self.optimizer.step()
            
            if self.tf_writer:
                _log_name = 'train/batch_loss'
                self.tf_writer.update_step_loss(_log_name, loss, self.overall_step)
                _log_name = 'train/batch_signal_loss'
                self.tf_writer.update_step_loss(_log_name, signal_loss, self.overall_step)
                _log_name = 'train/batch_class_loss'
                self.tf_writer.update_step_loss(_log_name, class_loss, self.overall_step)
            
        return {'total_loss': total_loss / step}
    
    def compute_dev_loss(self, current_epoch):
        step = 0
        dev_total_loss = 0.
        
        for _, batch in enumerate(tqdm(self.dev_dataloader)):
            step += 1
            clean_wav = batch['clean_wav'].to(self.device) # [N, L]
            noisy_wav = batch['process_wav'].to(self.device) # [N, L]
            enroll_wav = batch['enroll_wav'].to(self.device) # [N, L]
            inactive_utts = batch['inactive_utts'] # [N]
            target_spk_class = batch['spk_label'].to(self.device) # [N]
            
            with torch.no_grad():
                if self.hparam['TRAIN']['contrastive_learning']:
                    loss = self.model(noisy=noisy_wav, enroll=enroll_wav, ref_clean=clean_wav, spk_class=target_spk_class, alpha=self.hparam['LOSS']['alpha'],
                                    inactive_labels=inactive_utts, return_loss_detail=False)
                else:
                    loss = self.model(noisy=noisy_wav, enroll=enroll_wav, ref_clean=clean_wav, spk_class=None, alpha=self.hparam['LOSS']['alpha'],
                                    inactive_labels=inactive_utts, return_loss_detail=False)
                loss = torch.mean(loss, dim=0) # aggregate loss from each device
                dev_total_loss += loss.item()

        print(f"dev average loss: {dev_total_loss / step}")
        return {'total_loss': dev_total_loss / step}
    
    def gen_logging(self, epoch: int, prefix: str):
        """
        Generate samples on tensorboard for loggin
        """
        test_audio_dct = load_text_as_dict(f"{self.hparam['DATASET']['eval']}/wav2scp.txt")
        test_enroll_dct = load_text_as_dict(f"{self.hparam['DATASET']['eval']}/ref2list.txt")
        resample_to = self.hparam['DATASET']['sample_rate']

        for _, key in enumerate(test_audio_dct.keys()):
            uttid = key
            print(f"Running inference: {uttid}")
            wav, sr = AudioIO.open(f_path=test_audio_dct[key][0])
            if sr != resample_to:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_to)(wav)

            enroll_wav, sr = AudioIO.open(f_path=test_enroll_dct[key][0], target_lvl=-28)
            if sr != resample_to:
                enroll_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_to)(enroll_wav)

            wav = wav.to(self.device)
            enroll_wav = enroll_wav.to(self.device)

            if isinstance(self.model, torch.nn.DataParallel):
                enh_wav = self.model.module.inference(noisy=wav, enroll=enroll_wav)
            else:
                enh_wav = self.model.inference(noisy=wav, enroll=enroll_wav)
        
            if self.tf_writer:
                self.tf_writer.add_ep_audio(f"{prefix}{uttid}.wav", enh_wav, epoch, resample_to)
