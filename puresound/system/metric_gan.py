"""MetricGAN for an encoder/decoder enhancer: a learned PESQ critic, trained
alongside the model, whose verdict the model is trained to raise.

Two terms per step, added to the ordinary loss:

* **discriminator** -- ``D(clean, clean) -> 1`` on the current batch, and
  ``D(clean, enhanced) -> normalised PESQ`` on rows REPLAYED from earlier steps.
  PESQ runs on the CPU and a batch's worth would dominate the step time, so it
  is computed by a background process pool and the discriminator learns from
  whatever has come back. MetricGAN+ (Fu et al., 2021) keeps a replay buffer of
  past enhancements on purpose; here it is also what takes PESQ off the
  critical path.
* **generator** -- ``weight * (D(clean, enhanced) - 1)^2`` on the current batch,
  with the discriminator's parameters frozen for that forward so the gradient
  reaches the enhancer only. Off until ``warmup_steps``: a discriminator that
  has not yet learned anything gives the enhancer noise to climb.

Both terms go into the one loss the one optimizer steps, and each reaches only
its own side: the generator term cannot move the discriminator (its parameters
do not require grad during that forward) and the discriminator term cannot move
the enhancer (it reads detached, replayed audio and the clean target). The real
term runs every step, so every discriminator parameter has a gradient on every
rank -- which is what DDP needs from a module that is otherwise used a
different number of times per rank.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
from pydantic import Field, StrictBool

from puresound.config.base import StrictConfig


class MetricGanConfig(StrictConfig):
    enabled: StrictBool = False
    #: generator term weight; 0 trains the discriminator alone
    weight: float = Field(default=0.0, ge=0)
    #: optimizer steps before the generator term switches on
    warmup_steps: int = Field(default=1000, ge=0)
    n_fft: int = Field(default=400, gt=0)
    hop: int = Field(default=100, gt=0)
    ndf: int = Field(default=16, gt=0)
    lr_factor: float = Field(default=1.0, gt=0)
    #: PESQ worker processes per rank; 0 computes inline (tests)
    pesq_workers: int = Field(default=4, ge=0)
    #: rows of each batch sent for scoring
    rows_per_step: int = Field(default=12, gt=0)
    #: replay capacity in rows (6 s rows are ~0.8 MB a pair)
    buffer_rows: int = Field(default=256, gt=0)
    #: replayed rows per discriminator update
    d_rows: int = Field(default=12, gt=0)
    sample_rate: int = Field(default=16000, gt=0)


def normalised_pesq(values: np.ndarray) -> np.ndarray:
    """PESQ-WB to the discriminator's ``[0, 1]`` target: ``(pesq - 1) / 3.5``."""
    return np.clip((np.asarray(values, dtype=np.float64) - 1.0) / 3.5, 0.0, 1.0)


def _score(args):
    """One PESQ-WB, or NaN where PESQ has no answer (a silent reference)."""
    from pesq import PesqError, pesq

    sample_rate, reference, estimate = args
    value = pesq(sample_rate, reference, estimate, "wb", on_error=PesqError.RETURN_VALUES)
    return float(value) if value >= 1.0 else float("nan")


class PesqReplay:
    """Asynchronous PESQ labels and a replay buffer of labelled pairs.

    ``submit`` sends rows for scoring and returns at once; ``harvest`` moves
    finished results into the buffer; ``sample`` draws labelled pairs from it.
    The pool is created on first use and uses ``spawn``: forking a process that
    holds CUDA and NCCL threads is how a worker inherits a lock nobody will
    release.
    """

    def __init__(self, config: MetricGanConfig, seed: int = 0):
        self.config = config
        self.buffer: deque = deque(maxlen=config.buffer_rows)
        self.pending: deque = deque()
        self.rng = random.Random(seed)
        self._pool = None
        self.submitted = 0
        self.scored = 0
        self.rejected = 0

    def _get_pool(self):
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(self.config.pesq_workers)
        return self._pool

    def submit(self, reference: torch.Tensor, estimate: torch.Tensor) -> None:
        """reference, estimate: ``[B, T]``; at most ``rows_per_step`` rows go."""
        rows = min(self.config.rows_per_step, reference.shape[0])
        if rows == 0:
            return
        ref = reference[:rows].detach().float().cpu()
        est = estimate[:rows].detach().float().cpu()
        jobs = [(self.config.sample_rate, r.numpy(), e.numpy()) for r, e in zip(ref, est)]
        self.submitted += rows
        if self.config.pesq_workers == 0:
            self._store(ref, est, [_score(j) for j in jobs])
            return
        # A pool that has fallen behind blocks here rather than queueing without
        # bound: the replay stays at most a few steps stale.
        while len(self.pending) >= 4 * self.config.pesq_workers:
            self._collect(self.pending.popleft(), block=True)
        self.pending.append((self._get_pool().map_async(_score, jobs), ref, est))

    def harvest(self) -> None:
        while self.pending and self.pending[0][0].ready():
            self._collect(self.pending.popleft(), block=False)

    def _collect(self, item, block: bool) -> None:
        result, ref, est = item
        self._store(ref, est, result.get() if block else result.get(timeout=0))

    def _store(self, ref: torch.Tensor, est: torch.Tensor, values) -> None:
        labels = normalised_pesq(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=-1.0))
        for r, e, v, y in zip(ref, est, values, labels):
            if np.isnan(v):
                self.rejected += 1
                continue
            self.buffer.append((r, e, float(y)))
            self.scored += 1

    def sample(self, rows: int, device) -> Optional[tuple]:
        if not self.buffer:
            return None
        picks = [self.buffer[self.rng.randrange(len(self.buffer))] for _ in range(rows)]
        n = min(p[0].shape[-1] for p in picks)
        ref = torch.stack([p[0][:n] for p in picks]).to(device)
        est = torch.stack([p[1][:n] for p in picks]).to(device)
        target = torch.tensor([p[2] for p in picks], dtype=torch.float32, device=device)
        return ref, est, target

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self.pending.clear()


def discriminator_loss(disc, replay: PesqReplay, reference: torch.Tensor, rows: int):
    """Real term on the current clean rows, plus the replayed fake term."""
    real = reference[:rows].detach()
    loss = ((disc(real, real).float() - 1.0) ** 2).mean()
    fake_mse = None
    replayed = replay.sample(rows, reference.device)
    if replayed is not None:
        ref, est, target = replayed
        fake_mse = ((disc(ref, est).float() - target) ** 2).mean()
        loss = loss + fake_mse
    return loss, fake_mse


def generator_loss(disc, reference: torch.Tensor, estimate: torch.Tensor):
    """``(D(clean, enhanced) - 1)^2`` with D's parameters frozen for the forward."""
    flags = [p.requires_grad for p in disc.parameters()]
    for p in disc.parameters():
        p.requires_grad_(False)
    try:
        score = disc(reference.detach(), estimate).float()
    finally:
        for p, flag in zip(disc.parameters(), flags):
            p.requires_grad_(flag)
    return ((score - 1.0) ** 2).mean(), score.detach().mean()
