import logging
from typing import Any, Callable, List, Optional

import torch.nn as nn
from lightning.pytorch import LightningModule

from .logger import Logging


logger = logging.getLogger(__name__)


#: What a loss is called with when it declares nothing. Every loss in this
#: package takes the enhanced/reference pair as its first two positionals.
DEFAULT_LOSS_INPUTS = ("enhanced", "target")


def invoke_loss(loss_func: nn.Module, providers: dict) -> Any:
    """Call one loss with exactly the inputs it asked for, in its own order.

    A loss declares ``required_inputs`` -- a tuple of provider names matching
    its ``forward`` signature -- and a module declares what it can provide.
    Nothing else decides the call, which is the point.

    What this replaces was a chain of ``if getattr(loss, "uses_X", False)``
    branches, and in a chain the *order* decides. A loss that wanted a later
    input had to switch every earlier flag off by hand:
    ``BackgroundVADHeadBCELoss`` inherits ``uses_vad_logits = True`` from
    ``VADHeadBCELoss``, so it carried ``uses_vad_logits = False`` purely to fall
    past the first branch. Forget that line and it trains against the foreground
    head's logits and the foreground target, silently, forever. A declaration
    replaces rather than negates, so a subclass cannot be wrong that way.

    An unknown name is an error rather than a fallback. The chain ended in
    ``return loss_func(enhanced, target)``, so a module that could not satisfy a
    loss called it with the waveform pair instead -- measured on MISO, asking it
    for ``VADHeadBCELoss`` produced 0.8132 from the enhanced waveform read as
    logits, against 0.7727 from the real ones. A number, not a crash.

    Providers are callables so a side output nothing asked for is never read,
    and the ones that synthesize a missing target do not run on every batch.
    """
    names = getattr(loss_func, "required_inputs", DEFAULT_LOSS_INPUTS)
    missing = [name for name in names if name not in providers]
    if missing:
        raise TypeError(
            f"{type(loss_func).__name__} requires {list(names)}, and "
            f"{missing} is not available here. This module provides "
            f"{sorted(providers)}; a loss that needs a backbone side output "
            "belongs on a module that produces one."
        )
    return loss_func(*(providers[name]() for name in names))


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

    def register_loss_func(
        self, loss_func_list: nn.ModuleList, loss_func_list_weights: List
    ):
        self.loss_func_list = loss_func_list
        self.loss_func_list_w = loss_func_list_weights

    def reduce_losses(
        self,
        invoke: Callable[[nn.Module], Any],
        loss_funcs: Optional[nn.ModuleList] = None,
        weights: Optional[List] = None,
    ):
        """Weighted sum of the registered losses, plus each one's scalar value.

        ``invoke(loss_func)`` returns that loss's unweighted tensor. That call is
        the only part that differs between modules -- what each can provide, and
        so which losses it can host -- so it is the only part left to the caller.
        See `invoke_loss`, which is what every module's `invoke` delegates to.

        ``loss_funcs`` / ``weights`` default to the registered pair; pass them
        explicitly for a secondary set (e.g. MISO's conditional-branch losses).

        Returns ``(total, per_loss_values)``. ``total`` is None for an empty
        loss list, which is a misconfigured recipe either way.
        """
        loss_funcs = self.loss_func_list if loss_funcs is None else loss_funcs
        weights = self.loss_func_list_w if weights is None else weights

        total = None
        values = []
        for loss_func, weight in zip(loss_funcs, weights):
            weighted_loss = weight * invoke(loss_func)
            values.append(weighted_loss.item())
            # Out of place: the loop this replaces accumulated with `+=` onto
            # the first weighted tensor, mutating it. Same value and same
            # gradient, one less in-place op on a graph tensor.
            total = weighted_loss if total is None else total + weighted_loss
        return total, values

    def register_gpu_vad_labeler(self, labeler: Any):
        """Attach a batched GPU VAD labeler used to build ``vad_target`` from
        the per-batch ``vad_reference`` waveform after device transfer.

        Stored inside a list so ``nn.Module`` does not register the Silero
        TorchScript model as a submodule (it must stay out of state_dict /
        checkpoints -- it is an auxiliary labeler, not trained weights).
        """
        self._gpu_vad_labeler = [labeler]

    def ensure_vad_targets(self, batch: Any):
        """Materialize deferred VAD labels if a batch still carries references.

        Lightning normally calls ``on_after_batch_transfer`` before
        ``training_step``/``validation_step``. Keeping the conversion in this
        helper lets the step methods call it defensively as well, which is
        useful across strategy/DDP versions and for direct unit-test calls.
        """
        labeler = getattr(self, "_gpu_vad_labeler", None)
        if not labeler or not isinstance(batch, dict):
            return batch

        if "sr" in batch and batch["sr"].numel() > 0:
            sample_rate = int(batch["sr"].reshape(-1)[0].item())
        else:
            sample_rate = labeler[0].model_sample_rate

        if "vad_reference" in batch and "vad_target" not in batch:
            ref = batch.pop("vad_reference")
            batch["vad_target"] = labeler[0](ref, sample_rate=sample_rate)

        if "background_vad_reference" in batch and "background_vad_target" not in batch:
            ref = batch.pop("background_vad_reference")
            batch["background_vad_target"] = labeler[0](
                ref,
                sample_rate=sample_rate,
            )
        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        """Label VAD activity on GPU for the whole batch at once.

        Silero is lifted out of the DataLoader workers: the dataset emits the
        clean reference waveform as ``vad_reference`` and the labeling happens
        here, batched, on the same device as the batch.
        """
        return self.ensure_vad_targets(batch)

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
        """Show the average metric scores.

        `print`, not the logger, and deliberately the only one left in the
        library: these numbers are what `--scoring` was RUN for, not a progress
        note about it. Routing them through logging would let someone silence
        the chatter (`setLevel(WARNING)`) and lose their results with it.
        """
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

    def reload_checkpoint(self, loaded_state, load_loss_func: bool = True):
        self_state = self.state_dict()
        check_key = list(self_state.keys())

        for name, param in loaded_state.items():
            if name not in self_state:
                logger.warning("%s is not in the model.", name)
                continue

            if "loss_func_list" in name and not load_loss_func:
                logger.info("Not loading %s because load_loss_func=%s", name, load_loss_func)
                check_key.remove(name)
                continue

            self_state[name].copy_(param)
            check_key.remove(name)

        if check_key == []:
            logger.info("Loaded params is ok.")
        else:
            logger.warning("Needed param name but missing: %s", check_key)
