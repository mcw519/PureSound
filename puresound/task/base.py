import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from puresound.src.utils import create_folder, load_text_as_dict
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter():
    """This is tensorboard logger"""
    def __init__(self, logging_path: str):
        self.tf_writer = SummaryWriter(log_dir=logging_path)
    
    def update_ep_lr(self, lr: float, epoch: int) -> None:
        """update learning rate for epoch change"""
        self.tf_writer.add_scalar('learning_rate', lr, epoch)

    def update_ep_loss(self, name: str, loss: Any, epoch: int) -> None:
        """self.tf_writer.add_scalar('train/avg_loss', loss['total_loss'], epoch)"""
        self.tf_writer.add_scalar(name, loss, epoch)
    
    def update_step_loss(self, name: str, loss: Any, step: int) -> None:
        """self.tf_writer.add_scalar('train/batch_loss', loss, self.overall_step)"""
        self.tf_writer.add_scalar(name, loss, step)
    
    def add_ep_picture(self, name: str, pic: Any, epoch: int, log: bool = True) -> None:
        """
        add image logging after epoch finished
        pic -- [C, T] or [1, C, T]
        """
        if log:
            pic = pic.log10()
        
        if pic.dim() == 3:
            # remove batch dim
            pic = pic.squeeze(0)
        
        pic = pic.numpy()
        fig = plt.figure()
        plt.imshow(pic, origin='lower')
        self.tf_writer.add_figure(name, fig, epoch)
        plt.close()

    def add_ep_audio(self, name: str, audio: Any, epoch: int, sr: int = 16000) -> None:
        """
        add audio log after epoch finished
        audio -- [1, L] or [1, 1, L]
        """
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        self.tf_writer.add_audio(name, audio, epoch, sr)
    
    def update_ep_metric(self, **kwargs) -> None:
        """
        interface for user to save you wanted.
        **kwargs -- tag='metric name', scalar_value=value, global_step=step
        """
        self.tf_writer.add_scalar(**kwargs)


class LearningRateScheduler():
    """
    Class for learning rate scheduler
    **kwargs:
        mode: choice=['min', 'max'], only used when scheduler type is Plateau
        gamma: how to reduce learning rate
        wait: how much step to reduce learning rate [stepLR] or how much patience to not reduce [Plateau].
    """
    def __init__(self, type: str, optimizer: Any, **kwargs):
        self.type = type
        gamma = 0.5 if not kwargs else kwargs['gamma']
        wait = 3 if not kwargs else kwargs['patience']
        if type == 'stepLR':
             self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=wait, gamma=gamma)
        
        elif type == 'Plateau':
            mode = 'min' if not kwargs else kwargs['mode']
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=gamma, patience=wait)
        
        else:
            raise NotImplementedError

    def step(self, loss: Any = None) -> None:
        if self.type == 'stepLR':
            self.scheduler.step()
        
        elif self.type == 'Plateau':
            self.scheduler.step(loss)
        
        else:
            raise NotImplementedError


class TaskDataset(torch.utils.data.Dataset):
    """
    Basic dataset follow Kaldi data preparation *.scp format.\n
    handcraft data folder must include:
        -- wav2scp.txt: audio path file
        -- [options] wav2ref.txt: audio for which reference audio, used in noisy to clean mapping
    
    Include:
        df: data frame
        idx_df: mapping idx to an unique df's key
    
    Args:
        folder: manifest folder
        resample_to: if not None, open waveform will resample to this value
    """
    def __init__(self, folder, resample_to: Optional[int] = None):
        super().__init__()
        self.folder = folder
        self.resample_to = resample_to
        self.df = self._load_df(self.folder)
        self.idx_df = self._idx2key(self.df)
    
    def __len__(self):
        return len(self.idx_df)
    
    def __getitem__(self, index: Any):
        raise NotImplementedError
    
    def get_feature(self, key: str):
        raise NotImplementedError

    @property
    def folder_content(self):
        """
        Set like:
            'wav2scp': wav2scp.txt
            'wav2class': wav2class.txt
            etc.
        """
        return {'wav2scp': 'wav2scp.txt'}

    def _load_df(self, folder: str) -> Dict:
        """method about loading manifest information."""
        _df = {}
        load_dct = self.folder_content

        # check file, wav2scp is must needed
        if not os.path.isfile(f"{folder}/{self.folder_content['wav2scp']}"):
            raise FileNotFoundError(f"{self.folder_content['wav2scp']} is not found")
        
        else:
            _wav2scp = load_text_as_dict(f"{folder}/wav2scp.txt")
            for key in sorted(_wav2scp.keys()):
                _df[key] = {'wav2scp': _wav2scp[key][0]}
                        
            del load_dct['wav2scp']

        if load_dct.keys != {}:
            for f in load_dct.keys():
                if not os.path.isfile(f"{folder}/{load_dct[f]}"):
                    raise FileNotFoundError(f"{load_dct[f]} is not found")
                else:
                    _temp = load_text_as_dict(f"{folder}/{load_dct[f]}")
                    for key in sorted(_temp.keys()):
                        try:
                            if len(_temp[key]) != 1:
                                _df[key].update({f: _temp[key][:]})
                            else:
                                _df[key].update({f: _temp[key][0]})
                        except KeyError:
                            print(f"Non match key {key}")

        return _df
    
    def _idx2key(self, df) -> Dict:
        """mapping df.keys to idx."""
        _idx_key = {}
        idx = 0
        for key in df.keys():
            _idx_key[idx] = key
            idx += 1
        return _idx_key

    def _to_onehot(self, y: int, num_classes: int) -> torch.Tensor:
        """Function to convert label(int) to onehot vector."""
        target = torch.zeros(num_classes, dtype=torch.float)
        target[y] = 1.
        return target


class BaseTrainer():
    """
    Basic class for NN training.\n

    Args:
        device_backend

    Inherit this class must implement:
        -- build_model(): init NN model
        -- loss_func(): define compute loss step
        -- train_one_epoch(): define train step
        -- compute_dev_loss(): define dev step
        -- gen_logging(): define logging step
    """
    def __init__(self, hparam: Dict, device_backend: str = 'cuda'):
        self.hparam = hparam
        self.best_loss = torch.inf
        self.best_epoch = torch.inf

        if device_backend.lower() == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')

        elif device_backend.lower() == 'mps':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Model & Optimizer
        if self.hparam['TRAIN']['multi_gpu'] and device_backend != 'cpu':
            print(f"DP: using single-machine && multi-gpus")
            self.build_model()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        
        else:
            self.build_model()
            self.model.to(self.device)
        
        self.build_optim()
        
        # Logging
        if self.hparam['TRAIN']['use_tensorboard']:
            create_folder(self.hparam['TRAIN']['log_dir'])
            self.tf_writer = TensorboardWriter(logging_path=self.hparam['TRAIN']['log_dir'])

        else:
            self.tf_writer = None

    def build_model(self) -> Any:
        """init model."""
        raise NotImplementedError

    def build_optim(self) -> None:
        """init optimizer."""
        lr = self.hparam['OPTIMIZER']['lr']
        if self.hparam['TRAIN']['resume_epoch']:
            print(f"***** Start from {self.hparam['TRAIN']['resume_epoch']} epoch")
            _, lr, _ = self.load_ckpt(f"{self.hparam['TRAIN']['model_save_dir']}/epoch_{self.hparam['TRAIN']['resume_epoch'] -1}.ckpt", self.model)
            self.lr = lr
    
        beta1 = self.hparam['OPTIMIZER']['beta1']
        beta2 = self.hparam['OPTIMIZER']['beta2']
        weight_decay = self.hparam['OPTIMIZER']['weight_decay']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, [beta1, beta2], weight_decay=weight_decay)
        self.scheduler = LearningRateScheduler(self.hparam['OPTIMIZER']['lr_scheduler'], self.optimizer, gamma=self.hparam['OPTIMIZER']['gamma'], patience=self.hparam['OPTIMIZER']['patience'], mode=self.hparam['OPTIMIZER']['mode'])
    
    def save_ckpt(self, filename: str, model: Any, epoch: int, learning_rate: Any, loss: Any) -> None:
        """function to save model's parameters."""
        if not self.hparam['TRAIN']['multi_gpu']:
            ckpt = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "learning_rate": learning_rate,
                "loss": loss['total_loss'],
                "best_epoch": self.best_loss,
                "best_epoch": self.best_epoch,}
        else:
            ckpt = {
                "state_dict": model.module.state_dict(),
                "epoch": epoch,
                "learning_rate": learning_rate,
                "loss": loss['total_loss'],
                "best_epoch": self.best_loss,
                "best_epoch": self.best_epoch,}
        
        torch.save(ckpt, filename)
    
    def save_ckpt_info(self, filename: str, epoch: int, learning_rate: Any, loss: Any) -> None:
        """function to save model's training information."""
        with open(filename, 'w') as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"lr: {learning_rate}\n")
            f.write(f"loss: {loss['total_loss']}\n")
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"best_loss: {self.best_loss}\n")
    
    def load_ckpt(self, filename: str, model: Any) -> Tuple:
        """function to load pre-trained model."""
        ckpt = torch.load(filename, map_location='cpu')
        epoch = ckpt['epoch']
        lr = ckpt['learning_rate']
        loss = ckpt['loss']
        try:
            self.best_epoch = ckpt['best_epoch']
            self.best_loss = ckpt['best_loss']
        except:
            self.best_epoch = torch.inf
            self.best_loss = torch.inf

        if self.hparam['TRAIN']['multi_gpu']:
            model.module.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt['state_dict'])

        return epoch, lr, loss
    
    def train_one_epoch(self):
        raise NotImplementedError
    
    def compute_dev_loss(self):
        raise NotImplementedError
    
    def gen_logging(self, epoch: Optional[int] = None, prefix: str = ""):
        raise NotImplementedError
    
    def train(self):
        """function to start train a model."""
        self.num_epochs = self.hparam['TRAIN']['num_epochs']
        start_epoch = 0 if self.hparam['TRAIN']['resume_epoch'] == None else self.hparam['TRAIN']['resume_epoch']

        for epoch in range(start_epoch, self.num_epochs):
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            
            self.model.train()
            loss = self.train_one_epoch(current_epoch=epoch)
            if loss <= self.best_loss:
                self.best_loss = loss
                self.best_epoch = epoch

            self.model.eval()
            dev_loss = self.compute_dev_loss(current_epoch=epoch)

            if self.tf_writer:
                self.tf_writer.update_ep_lr(learning_rate, epoch)
                self.tf_writer.update_ep_loss('train/avg_loss', loss['total_loss'], epoch)
                self.tf_writer.update_ep_loss('train/avg_dev_loss', dev_loss['total_loss'], epoch)

            model_path = os.path.join(self.hparam['TRAIN']['model_save_dir'], f"epoch_{epoch}.ckpt")
            info_path = os.path.join(self.hparam['TRAIN']['model_save_dir'], f"epoch_{epoch}.info")

            self.save_ckpt(model_path, self.model, epoch, learning_rate, loss)
            self.save_ckpt_info(info_path, epoch, learning_rate, loss)

            if epoch >= self.hparam['OPTIMIZER']['num_epochs_decay']:
                if self.hparam['OPTIMIZER']['lr_scheduler'] == 'Plateau':
                    self.scheduler.step(dev_loss['total_loss'])
                else:
                    self.scheduler.step()

            self.gen_logging(epoch=epoch, prefix="")
