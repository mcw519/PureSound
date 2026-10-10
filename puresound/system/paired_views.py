"""Auxiliary-view loss dispatch for encoder/decoder systems, independent of task.

A registered loss opts in with ``paired_output`` (a loss-provider name),
``paired_weight`` (optional, defaults to 1), and a ``paired_consistency`` method.
The method receives primary/auxiliary tensors, the primary batch and auxiliary
view metadata; it returns a mean loss, effective example count and comparison
count. Extra views never pass through ordinary loss reduction.
"""
from __future__ import annotations

import torch
from pydantic import Field, StrictBool
from puresound.config.base import StrictConfig


class PairedViewConsistencyConfig(StrictConfig):
    enabled: StrictBool = False
    max_rows: int = Field(default=1, gt=0)


def paired_view_loss(module, batch, enhanced, config):
    """At most one extra forward, even when several objectives consume it.

    The collective maximum determines the forward batch size on every rank.
    Empty ranks retain a zero-valued gradient path through the same readouts,
    which also keeps synchronized-normalization backward collectives aligned.
    Each mean is weighted by the global number of effective examples rather
    than by ranks, so empty ranks do not dilute observations.
    """
    objectives = [(i, loss, weight) for i, (loss, weight) in enumerate(
        zip(module.loss_func_list, module.loss_func_list_w))
        if getattr(loss, 'paired_output', None) is not None]
    if not objectives:
        raise ValueError('paired_view_consistency needs a loss declaring paired_output')

    def providers(output, labels):
        target = labels.get('clean_speech')
        return module._loss_providers(
            enhanced=output, target=target, vad_target=labels.get('vad_target'),
            batch=labels, inactive_labels=None if target is None else target.abs().amax(-1) == 0,
        )

    def read(table, loss):
        name = loss.paired_output
        if name not in table or not callable(getattr(loss, 'paired_consistency', None)):
            raise ValueError(f'invalid paired output contract for {type(loss).__name__}: {name}')
        value = table[name]()
        if not torch.is_tensor(value):
            raise ValueError(f'paired output {name!r} is unavailable; enable its provider')
        return value

    original = providers(enhanced, batch)
    primary = {i: read(original, loss) for i, loss, _ in objectives}
    noisy, view = batch['noisy_speech'], batch.get('paired_view')
    count = min(config.max_rows, len(view['source_indices'])) if view else 0
    size = torch.tensor(count, dtype=torch.long, device=noisy.device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(size, op=torch.distributed.ReduceOp.MAX)
    global_size = int(size)
    secondary = {}
    if global_size:
        extra = noisy[:1].expand(global_size, *noisy.shape[1:]).clone()
        if count:
            extra[:count] = view['noisy_speech'][:count]
        # Preserve the primary side outputs for any existing regularizer that
        # runs afterwards. Readout names belong to providers, not this helper.
        saved_mask = getattr(module, 'last_mask', None)
        saved_side = {name: value for name, value in vars(module.backbone).items()
                      if name.startswith('last_')}
        try:
            extra_output = module.forward(extra)
            table = providers(extra_output, view or {})
            secondary = {i: read(table, loss) for i, loss, _ in objectives}
        finally:
            module.last_mask = saved_mask
            for name, value in saved_side.items():
                setattr(module.backbone, name, value)

    total = enhanced.sum() * 0.0
    world_size = torch.distributed.get_world_size() if distributed else 1
    selected = {key: value[:count] for key, value in view.items()} if count else {}
    for i, loss, weight in objectives:
        first = primary[i]
        raw, effective, comparisons = first.sum() * 0.0, 0, 0
        if count:
            raw, effective, comparisons = loss.paired_consistency(
                first, secondary[i][:count], batch, selected,
            )
        if global_size:
            raw = raw + secondary[i].sum() * 0.0
        counts = torch.tensor([effective, comparisons], device=noisy.device, dtype=torch.float32)
        if distributed:
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        n = float(counts[0])
        coefficient = weight * float(getattr(loss, 'paired_weight', 1.0))
        total = total + coefficient * raw * (world_size * effective / n if n else 0.0)
        logged = raw.detach().float() * effective
        if distributed:
            torch.distributed.all_reduce(logged, op=torch.distributed.ReduceOp.SUM)
        logged = logged / max(n, 1.0)
        prefix = f'train_paired_{i}_{type(loss).__name__}'
        for suffix, value in [('count', counts[0]), ('comparisons', counts[1]),
                              ('loss', logged), ('weighted', coefficient * logged)]:
            module.log(f'{prefix}_{suffix}', value, on_step=True, sync_dist=False)
        for name, value in getattr(loss, 'last_stats', {}).items():
            module.log(f'train_loss_{i}_{name}', value, on_step=True, sync_dist=False)
    return total
