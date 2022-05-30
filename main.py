import argparse
from typing import Any, Tuple

import torch
import torchaudio
from tqdm import tqdm

from model import init_model
from src.audio import AudioIO
from src.dataset import TseCollateFunc, TseDataset
from src.metrics import Metrics
from src.sampler import SpeakerSampler
from src.utils import create_folder, load_hparam, load_text_as_dict
from train import TseTrainer


def init_dataloader(hparam: Any) -> Tuple:
    if hparam['DATASET']['type'].lower() == 'tse':
        train_dataset = TseDataset(
            folder=hparam['DATASET']['train'],
            resample_to=hparam['DATASET']['sample_rate'],
            max_length=hparam['DATASET']['max_length'],
            noise_folder=hparam['DATASET']['noise_folder'],
            rir_folder=hparam['DATASET']['rir_folder'],
            rir_mode=hparam['DATASET']['rir_mode'],
            speed_perturbed=hparam['DATASET']['speed_perturbed'],
            vol_perturbed=hparam['DATASET']['vol_perturbed'],
            single_spk_pb=hparam['DATASET']['single_spk_prob'],
            inactive_training=hparam['DATASET']['inactive_training'],
            enroll_augment=hparam['DATASET']['enroll_augment'],
            enroll_rule=hparam['DATASET']['enroll_rule']
            )
        
        dev_dataset = TseDataset(
            folder=hparam['DATASET']['dev'],
            resample_to=hparam['DATASET']['sample_rate'],
            max_length=hparam['DATASET']['max_length'],
            noise_folder=hparam['DATASET']['noise_folder'],
            rir_folder=hparam['DATASET']['rir_folder'],
            rir_mode=hparam['DATASET']['rir_mode'],
            speed_perturbed=hparam['DATASET']['speed_perturbed'],
            vol_perturbed=hparam['DATASET']['vol_perturbed'],
            single_spk_pb=0.,
            inactive_training=0.,
            enroll_augment=hparam['DATASET']['enroll_augment'],
            enroll_rule=hparam['DATASET']['enroll_rule']
            )
    
    else:
        raise NameError
    
    if hparam['TRAIN']['contrastive_learning']:
        train_len = len(train_dataset) // (hparam['TRAIN']['p_spks'] * hparam['TRAIN']['p_utts'])
        train_meta = train_dataset.sampler_meta()
        dev_len = len(dev_dataset) // (hparam['TRAIN']['p_spks'] * hparam['TRAIN']['p_utts'])
        dev_meta = dev_dataset.sampler_meta()
        train_sampler = SpeakerSampler(train_meta, train_len, n_spks=hparam['TRAIN']['p_spks'], n_per=hparam['TRAIN']['p_utts'])
        dev_sampler = SpeakerSampler(dev_meta, dev_len, n_spks=hparam['TRAIN']['p_spks'], n_per=hparam['TRAIN']['p_utts'])
    
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
            num_workers=hparam['TRAIN']['num_workers'], collate_fn=TseCollateFunc())
        dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_sampler=dev_sampler,
            num_workers=hparam['TRAIN']['num_workers'], collate_fn=TseCollateFunc())
    
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparam['TRAIN']['batch_size'], shuffle=True,
            num_workers=hparam['TRAIN']['num_workers'], collate_fn=TseCollateFunc())
        dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=hparam['TRAIN']['batch_size'], shuffle=True,
            num_workers=hparam['TRAIN']['num_workers'], collate_fn=TseCollateFunc())

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
        
        if config.task == 'TSE':
            assert hparam['DATASET']['type'].lower() == 'tse', f"Wrong dataset type."
            trainer = TseTrainer(hparam, device_backend=config.backend, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)
        
        else:
            raise NameError
        
        trainer.train()
    
    elif config.action == 'dev':
        """
        All of tasks, in dev-mode ......
        """
        #TODO: SE dataset init ???
        if config.task.lower() == 'tse':
            dev_dataset = TseDataset(
                folder=hparam['DATASET']['dev'],
                resample_to=hparam['DATASET']['sample_rate'],
                max_length=None,
                noise_folder=None,
                rir_folder=None,
                rir_mode=hparam['DATASET']['rir_mode'],
                speed_perturbed=None,
                vol_perturbed=None,
                single_spk_pb=0.,
                inactive_training=0.,
                enroll_augment=None,
                enroll_rule=hparam['DATASET']['enroll_rule']
                )
        else:
            raise NameError

        dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=TseCollateFunc())
        scorePESQ, scoreSTOI, scoreSDR, scoreSISNR, scoreSISNRi, scoreNSR, scoreNSR_neg = [], [], [], [], [], [], []
        model = init_model(hparam['MODEL']['type'], verbose=False)
        ckpt = torch.load(f"{hparam['TRAIN']['model_save_dir']}/{config.ckpt}", map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False) # ignore loss's weight
        model = model.to(config.backend)
        model.eval()
        print("uttid, PESQ, STOI, SDR, SISNR, SISNRi, NSR")

        for _, batch in enumerate(tqdm(dev_dataloader)):
            uttid = batch['uttid']
            clean_wav = batch['clean_wav'] # [1, L]
            noisy_wav = batch['process_wav'].to(config.backend) # [1, L]
            enroll_wav = batch['enroll_wav'].to(config.backend) # [1, L]
            enh_wav = model.inference(noisy_wav, enroll_wav) # [1, L]
            enh_wav = enh_wav.detach().cpu()
            noisy_wav = noisy_wav.detach().cpu()

            _sisnr = Metrics.sisnr(clean_wav, enh_wav)
            _sisnri = Metrics.sisnr_imp(clean_wav, enh_wav, noisy_wav)
            # define NSR is the count of negative SiSNRi and SiSNR lower than 30
            if _sisnri < 0 and _sisnr < 30:
                _nsr = 1
                _nsr_neg = 1 if _sisnr < 0 else 0
            else:
                _nsr = 0
                _nsr_neg = 0
            
            scoreSISNR.append(_sisnr)
            scoreSISNRi.append(_sisnri)
            scoreNSR.append(_nsr)
            scoreNSR_neg.append(_nsr_neg)

            if config.metrics == 'detail':
                scorePESQ.append(Metrics.pesq_wb(clean_wav, enh_wav))
                scoreSTOI.append(Metrics.stoi(clean_wav, enh_wav))
                scoreSDR.append(Metrics.bss_sdr(clean_wav, enh_wav))

            else:
                scorePESQ.append(0)
                scoreSTOI.append(0)
                scoreSDR.append(0)
            
            print(f"{uttid[0]}, {scorePESQ[-1]}, {scoreSTOI[-1]}, {scoreSDR[-1]}, {scoreSISNR[-1]}, {scoreSISNRi[-1]}, {scoreNSR[-1]}")
        
        print(f"PESQ: {torch.Tensor(scorePESQ).mean()}")
        print(f"STOI: {torch.Tensor(scoreSTOI).mean()}")
        print(f"SDR: {torch.Tensor(scoreSDR).mean()}")
        print(f"SiSNR: {torch.Tensor(scoreSISNR).mean()}")
        print(f"SiSNRi: {torch.Tensor(scoreSISNRi).mean()}")
        print(f"NSR: {torch.Tensor(scoreNSR).mean()}")
        print(f"NSR-negative: {torch.Tensor(scoreNSR_neg).mean()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--action", type=str, default='train', choices=['train', 'dev', 'eval', 'tSNE'])
    parser.add_argument("--task", type=str, default='TSE', choices=['TSE', 'SS', 'SE'])
    parser.add_argument("--backend", type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument("--metrics", type=str, default='simple', choices=['simple', 'detail'])
    parser.add_argument("--ckpt", type=str, default=None)
    config = parser.parse_args()
    main(config)
