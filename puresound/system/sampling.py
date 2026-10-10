"""Carry a `CoverageSampler` walk across runs through the checkpoint.

Every checkpoint records the walk under ``checkpoint["coverage_sampler"]``; the
next run reads it back:

  * ``--ckpt_path`` (resume): the same stage at an epoch boundary;
  * ``--pretrained_ckpt_path`` (warm start): a new stage, from where that one ended;
  * neither, or a warm-start checkpoint with no record: a new walk at ``--set_seed``.

Completed batches determine the saved position, including early ``max_steps``
stops. A partial-epoch checkpoint can warm start, but cannot resume a same-stage
Lightning run with a non-stateful dataloader.
"""

from __future__ import annotations

import logging
from typing import Optional

import lightning as L
import torch

from puresound.task.sampler import CoverageSampler


logger = logging.getLogger(__name__)

STATE_KEY = "coverage_sampler"


def _recorded_walk(path: str) -> Optional[dict]:
    return torch.load(path, map_location="cpu", weights_only=False).get(STATE_KEY)


def attach_coverage_sampler(sampler: CoverageSampler, args) -> "CoverageSamplerCheckpoint":
    """Place the walk for this run and return the callback that records it."""
    set_seed = getattr(args, "set_seed", None)
    resume, warm = getattr(args, "ckpt_path", None), getattr(args, "pretrained_ckpt_path", None)
    seed = 0 if set_seed is None else int(set_seed)
    resume_epoch = None
    sampler.set_seed(seed)
    if resume:
        state = _recorded_walk(resume)
        if state is None:
            raise ValueError(f"{resume} records no coverage-sampler walk to resume")
        sampler.start_from(state, resume=True)
        resume_epoch = (state["next"]["batch"] - state["stage_start"]["batch"]) // len(sampler)
        # DataLoader workers can prefetch as soon as Lightning constructs its
        # iterator, before Lightning's first set_epoch call. Place the sampler
        # now so those rows also come from the resumed epoch.
        sampler.set_epoch(resume_epoch)
        logger.info("coverage sampler: resuming the stage that began at batch %d",
                    state["stage_start"]["batch"])
    elif warm:
        state = _recorded_walk(warm)
        if state is None:
            logger.warning(
                "coverage sampler: %s records no walk; new walk at seed %d", warm, seed,
            )
        else:
            sampler.start_from(state, resume=False)
            logger.info("coverage sampler: continuing the walk at batch %d, slot %d",
                        state["next"]["batch"], state["next"]["slot"])
    else:
        logger.info("coverage sampler: new walk at seed %d", seed)
    return CoverageSamplerCheckpoint(sampler, resume_epoch=resume_epoch)


class CoverageSamplerCheckpoint(L.Callback):
    """Write the walk's position into every checkpoint."""

    def __init__(self, sampler: CoverageSampler, *, resume_epoch: Optional[int] = None):
        super().__init__()
        self.sampler = sampler
        self._epoch = None
        self._batches_done = 0
        self._resume_epoch = resume_epoch

    def on_train_start(self, trainer: L.Trainer, pl_module) -> None:
        # Lightning's iteration-based loop can restore the preceding epoch even
        # for an end-of-epoch checkpoint. Check the public epoch before any row
        # is consumed; replaying that epoch would corrupt the saved walk.
        if self._resume_epoch is not None and int(trainer.current_epoch) != self._resume_epoch:
            raise ValueError(
                "coverage sampler restored epoch does not match the checkpoint position; "
                "use epoch-based training or a warm start"
            )
        self._epoch = int(trainer.current_epoch)
        self._batches_done = 0

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module) -> None:
        self._epoch = int(trainer.current_epoch)
        self._batches_done = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._batches_done = int(batch_idx) + 1

    def on_save_checkpoint(self, trainer: L.Trainer, pl_module, checkpoint: dict) -> None:
        # A max_steps stop can end an epoch early; sampler prefetch can be ahead
        # of the model. Count completed training batches from the callback.
        checkpoint[STATE_KEY] = self.sampler.checkpoint_state(
            epochs_done=self._epoch if self._epoch is not None else 0,
            batches_done=self._batches_done,
        )
