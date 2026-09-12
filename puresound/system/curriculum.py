"""The training-process half of a recipe's ``curriculum`` block.

The dataset half rides along with each item (the sampler attaches the epoch, and
the workers move their own knobs). Loss weights are not in a worker: they live on
the module, so they are set here, once per epoch, from the same schedule.

This callback is also where a run says out loud what recipe it is training under
at each epoch -- a schedule that only exists in a config file is exactly the kind
of thing a log has to carry, because a checkpoint does not.
"""

from __future__ import annotations

import logging
from typing import Mapping

import lightning as L

from puresound.config.curriculum import CurriculumConfig


logger = logging.getLogger(__name__)


class CurriculumCallback(L.Callback):
    """Apply the epoch's scheduled loss weights, and log every scheduled knob."""

    def __init__(self, curriculum: CurriculumConfig, loss_indices: Mapping[str, int]):
        super().__init__()
        self.curriculum = curriculum
        self.loss_indices = dict(loss_indices)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = int(trainer.current_epoch)
        values = self.curriculum.resolve(epoch)

        if values.loss_weights:
            weights = getattr(pl_module, "loss_func_list_w", None)
            if weights is None:
                raise RuntimeError(
                    "curriculum schedules loss weights but no loss list is "
                    "registered on the module"
                )
            for reference, weight in values.loss_weights.items():
                weights[self.loss_indices[reference]] = weight

        # The sampler is normally told the epoch by Lightning itself
        # (`_set_sampler_epoch`, before the epoch's iterator is consumed). Repeat
        # it here so the data side still follows the schedule if that hook ever
        # stops reaching a batch sampler; it is idempotent.
        sampler = getattr(
            getattr(trainer, "train_dataloader", None), "batch_sampler", None
        )
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

        if trainer.is_global_zero:
            logger.info("curriculum epoch %d: %s", epoch, values.describe())
        pl_module.log_dict(
            {
                **{f"curriculum/aug.{k}": v for k, v in values.augmentation.items()},
                **{f"curriculum/bank.{k}": v for k, v in values.bank_weights.items()},
                **{f"curriculum/loss.{k}": v for k, v in values.loss_weights.items()},
            },
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )
