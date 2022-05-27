import argparse
from typing import Any, Tuple

import torch
import torchaudio
from tqdm import tqdm

from model import get_model
from src.audio import AudioIO
from src.dataset import TseCollateFunc, TseDataset
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

        dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=0, shuffle=True, num_workers=0, collate_fn=TseCollateFunc())

        model = get_model(hparam['MODEL']['type'])
        ckpt = torch.load(f"{hparam['TRAIN']['model_save_dir']}/{config.ckpt}", map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(config.backend)
        model.eval()
        
        #TODO: calculate metric score
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--action", type=str, default='train', choices=['train', 'dev', 'eval', 'tSNE'])
    parser.add_argument("--task", type=str, default='TSE', choices=['TSE', 'SS', 'SE'])
    parser.add_argument("--backend", type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    config = parser.parse_args()
    main(config)
