import argparse
from typing import Any, Tuple

import torch
import torchaudio
from puresound.src.audio import AudioIO
from puresound.src.metrics import Metrics
from puresound.src.utils import create_folder, load_hparam, load_text_as_dict
from puresound.task.ns import NsCollateFunc, NsDataset, NsTask
from tqdm import tqdm

from model import init_loss, init_model


class NsTrainer(NsTask):
    """Build NS-task trainer with loss/model init function."""
    def __init__(self, hparam, device_backend, train_dataloader, dev_dataloader):
        super().__init__(hparam, device_backend, train_dataloader, dev_dataloader)
    
    def build_model(self):
        sig_loss = init_loss(self.hparam)
        self.model = init_model(self.hparam['MODEL']['type'], sig_loss, verbose=True)


def init_dataloader(hparam: Any) -> Tuple:
    train_dataset = NsDataset(
        folder=hparam['DATASET']['train'],
        resample_to=hparam['DATASET']['sample_rate'],
        max_length=hparam['DATASET']['max_length'],
        noise_folder=hparam['DATASET']['noise_folder'],
        rir_folder=hparam['DATASET']['rir_folder'],
        rir_mode=hparam['DATASET']['rir_mode'],
        speed_perturbed=hparam['DATASET']['speed_perturbed'],
        vol_perturbed=hparam['DATASET']['vol_perturbed'])
    
    dev_dataset = NsDataset(
        folder=hparam['DATASET']['dev'],
        resample_to=hparam['DATASET']['sample_rate'],
        max_length=hparam['DATASET']['max_length'],
        noise_folder=hparam['DATASET']['noise_folder'],
        rir_folder=hparam['DATASET']['rir_folder'],
        rir_mode=hparam['DATASET']['rir_mode'],
        speed_perturbed=hparam['DATASET']['speed_perturbed'],
        vol_perturbed=hparam['DATASET']['vol_perturbed'])

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparam['TRAIN']['batch_size'], shuffle=True,
        num_workers=hparam['TRAIN']['num_workers'], collate_fn=NsCollateFunc())
    dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=hparam['TRAIN']['batch_size'], shuffle=True,
        num_workers=hparam['TRAIN']['num_workers'], collate_fn=NsCollateFunc())

    return train_dataloader, dev_dataloader


def main(config):
    """
    config is the command-line args
    hparam is the YAML configuration
    """
    hparam = load_hparam(config.config_path)
    create_folder(hparam['TRAIN']['model_save_dir'])
    
    if config.action == 'train':
        train_dataloader, dev_dataloader = init_dataloader(hparam)
        trainer = NsTrainer(hparam, device_backend=config.backend, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)
        trainer.train()
    
    elif config.action == 'dev':
        dev_dataset = NsDataset(
            folder=hparam['DATASET']['dev'],
            resample_to=hparam['DATASET']['sample_rate'],
            max_length=None,
            noise_folder=None,
            rir_folder=None,
            rir_mode=hparam['DATASET']['rir_mode'],
            speed_perturbed=None,
            vol_perturbed=None)

        dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=NsCollateFunc())
        scorePESQ, scoreSTOI, scoreSDR, scoreSISNR, scoreSISNRi = [], [], [], [], []
        model = init_model(hparam['MODEL']['type'], verbose=False)
        ckpt = torch.load(f"{hparam['TRAIN']['model_save_dir']}/{config.ckpt}", map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False) # ignore loss's weight
        model = model.to(config.backend)
        model.eval()
        print("uttid, PESQ, STOI, SDR, SISNR, SISNRi")

        for _, batch in enumerate(tqdm(dev_dataloader)):
            uttid = batch['uttid']
            clean_wav = batch['clean_wav'] # [1, L]
            noisy_wav = batch['process_wav'].to(config.backend) # [1, L]
            enh_wav = model.inference(noisy_wav) # [1, L]
            enh_wav = enh_wav.detach().cpu()
            noisy_wav = noisy_wav.detach().cpu()

            _sisnr = Metrics.sisnr(clean_wav, enh_wav)
            _sisnri = Metrics.sisnr_imp(clean_wav, enh_wav, noisy_wav)
            scoreSISNR.append(_sisnr)
            scoreSISNRi.append(_sisnri)

            if config.metrics == 'detail':
                scorePESQ.append(Metrics.pesq_wb(clean_wav, enh_wav))
                scoreSTOI.append(Metrics.stoi(clean_wav, enh_wav))
                scoreSDR.append(Metrics.bss_sdr(clean_wav, enh_wav))

            else:
                scorePESQ.append(0)
                scoreSTOI.append(0)
                scoreSDR.append(0)
            
            print(f"{uttid[0]}, {scorePESQ[-1]}, {scoreSTOI[-1]}, {scoreSDR[-1]}, {scoreSISNR[-1]}, {scoreSISNRi[-1]}")
        
        print(f"PESQ: {torch.Tensor(scorePESQ).mean()}")
        print(f"STOI: {torch.Tensor(scoreSTOI).mean()}")
        print(f"SDR: {torch.Tensor(scoreSDR).mean()}")
        print(f"SiSNR: {torch.Tensor(scoreSISNR).mean()}")
        print(f"SiSNRi: {torch.Tensor(scoreSISNRi).mean()}")

    elif config.action == 'eval':
        # Evaluation block
        create_folder(f"{hparam['TRAIN']['model_save_dir']}/eval_audio")
        ckpt = torch.load(f"{hparam['TRAIN']['model_save_dir']}/{config.ckpt}", map_location='cpu')
        model = init_model(hparam['MODEL']['type'], verbose=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.eval()
        test_audio_dct = load_text_as_dict(f"{hparam['DATASET']['eval']}/wav2scp.txt")
        sr = hparam['DATASET']['sample_rate']

        with torch.no_grad():
            for _, key in enumerate(test_audio_dct.keys()):
                uttid = key
                print(f"Running inference: {uttid}")
                noisy_wav, wav_sr = AudioIO.open(f_path=test_audio_dct[key][0])
                if wav_sr != sr: noisy_wav = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=sr)(noisy_wav)
                enh_wav = model.inference(noisy_wav) # [1, L]
                enh_wav = enh_wav.detach().cpu()
                if enh_wav.dim() == 3: enh_wav = enh_wav.squeeze(0)
                AudioIO.save(enh_wav, f"{hparam['TRAIN']['model_save_dir']}/eval_audio/{uttid}.wav", sr)

    else:
        raise NameError('Unrecognize action.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--action", type=str, default='train', choices=['train', 'dev', 'eval'])
    parser.add_argument("--backend", type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument("--metrics", type=str, default='simple', choices=['simple', 'detail'])
    parser.add_argument("--ckpt", type=str, default=None)
    config = parser.parse_args()
    main(config)
