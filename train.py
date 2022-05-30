import torch
import torchaudio
from tqdm import tqdm

from model import init_loss, init_model
from src.audio import AudioIO
from src.trainer import BaseTrainer
from src.utils import load_text_as_dict


class TseTrainer(BaseTrainer):
    def __init__(self, hparam, device_backend, train_dataloader, dev_dataloader):
        super().__init__(hparam, device_backend)
        self.overall_step = 0
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
    
    def build_model(self):
        sig_loss, cls_loss = init_loss(self.hparam)
        self.model = init_model(self.hparam['MODEL']['type'], sig_loss, cls_loss, verbose=True)

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
            loss, loss_detail = self.model(noisy=noisy_wav, enroll=enroll_wav, ref_clean=clean_wav, spk_class=target_spk_class,
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
            
            with torch.no_grad():
                loss = self.model(noisy=noisy_wav, enroll=enroll_wav, ref_clean=clean_wav, spk_class=None, alpha=10, return_loss_detail=False)
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
