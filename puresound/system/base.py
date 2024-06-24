from typing import Any, List

from lightning.pytorch import LightningModule

from .logger import Logging


class BaseLightningModule(LightningModule):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self._optimizer = None
        self._scheduler = None
        self.warmup_step = 1
        self.puresound_logging = Logging()

    def forward(self):
        """Forward to get results"""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def register_loss_func(self, loss_func_list: List):
        # loss_func_list looks like [[loss_1, weighted_1], [loss_2, weighted_2], ...]
        self.loss_func_list = loss_func_list

    def register_optimizer(self, optimizer: Any):
        self._optimizer = optimizer

    def register_scheduler(self, scheduler: Any):
        self._scheduler = scheduler

    def register_metrics_func(self, metrics: Any):
        self._metrics_func = metrics

    def register_proc_output_folder(self, fpath: str):
        self.eval_output_folder_path = fpath

    def register_warmup_step(self, warmup_step: int):
        self.warmup_step = warmup_step

    def configure_optimizers(self):
        if self._scheduler is not None:
            return [self._optimizer], [self._scheduler]
        else:
            return self._optimizer

    def on_train_epoch_end(self):
        """
        Show the epoch training loss.
        We defined the two keys here:
            epoch_train_loss: register each iterarions validation loss inside an epoch
        """
        scores = self.puresound_logging.average(key="epoch_train_loss")
        self.log(
            "epoch_train_loss",
            scores,
            prog_bar=True,
            sync_dist=False,
        )
        self.puresound_logging.clear(key="epoch_train_loss")

    def on_test_epoch_end(self):
        """Show the average metric scores."""
        scores = self.puresound_logging.average()
        for key in scores:
            print(key, scores[key].item())
            self.puresound_logging.clear(key=key)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        # For warmup used
        if self.trainer.global_step == 0:
            self.pg_lr = []
            for pg in optimizer.param_groups:
                self.pg_lr.append(pg["lr"])

        if self.trainer.global_step < self.warmup_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_step)
            for idx, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_scale * self.pg_lr[idx]

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad(set_to_none=True)
