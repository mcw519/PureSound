"""Warm start: new params stay at init, a changed shape is refused unless asked
for -- the checkpoint preflight applies the same rule before a run starts -- the
training driver builds a trainer for the devices the recipe names, and every rank's
DataLoader workers draw their own augmentation on one thread per pool."""

import os
import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from lightning.fabric.utilities import seed as lightning_seed

from egs.voice_isolate.scripts import preflight_ckpt_recipe as preflight
from puresound.config.loader import _parse_recipe as parse_recipe
from puresound.system import runner
from puresound.system.base import BaseLightningModule
from puresound.system.runner import load_warm_start


def _model(width):
    return torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.Linear(width, 2))


def test_a_param_the_checkpoint_lacks_is_kept_at_init():
    source = torch.nn.Sequential(torch.nn.Linear(4, 8))
    target = _model(8)
    missing, unexpected, reshaped = load_warm_start(target, source.state_dict())
    assert set(missing) == {"1.weight", "1.bias"} and not unexpected and not reshaped
    assert torch.equal(target[0].weight, source[0].weight)


def test_a_changed_shape_is_refused_unless_it_is_deliberate():
    with pytest.raises(ValueError, match="do not match this model's shape"):
        load_warm_start(_model(16), _model(8).state_dict())

    source, target = _model(8), _model(16)
    missing, _, reshaped = load_warm_start(target, source.state_dict(), allow_reshaped=True)
    assert set(reshaped) == {"0.weight", "0.bias", "1.weight"} and set(missing) == set(reshaped)
    assert torch.equal(target[1].bias, source[1].bias)


@pytest.mark.parametrize(
    "allowed, mismatch, expected",
    [
        ([], False, 1),
        (["vad_head"], False, 1),
        (["vad_head", "proximity_head"], False, 0),
        (["vad_head", "proximity_head"], True, 1),
    ],
)
def test_preflight_requires_each_missing_head_to_be_named(monkeypatch, allowed, mismatch, expected):
    state = {"backbone.encoder.weight": torch.zeros(2),
             "backbone.vad_head.weight": torch.zeros(2),
             "backbone.proximity_head.weight": torch.zeros(2)}
    checkpoint = {"backbone.encoder.weight": torch.zeros(3 if mismatch else 2)}
    argv = ["preflight", "--ckpt", "checkpoint", "recipe"]
    for head in allowed:
        argv += ["--allow-missing-head", head]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(preflight, "load_recipe", lambda *a, **kw: SimpleNamespace(model={}))
    monkeypatch.setattr(preflight, "init_siso_model",
                        lambda *a: SimpleNamespace(state_dict=lambda: state))
    monkeypatch.setattr(preflight.torch, "load", lambda *a, **kw: checkpoint)
    assert preflight.main() == expected


class _Tiny(BaseLightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch).square().mean()
        self.puresound_logging.update({"epoch_train_loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        return None

    def get_total_param_groups(self):
        return {"all": {"params": self.layer.parameters(), "lr_factor": 1.0}}


def test_a_recipe_without_gpus_trains_on_the_cpu(tmp_path, monkeypatch):
    """``num_gpus: 0`` is the CPU run; Lightning refuses ``devices=0``."""
    monkeypatch.setattr(runner, "init_loss_func", lambda config: (torch.nn.ModuleList(), []))
    recipe = SimpleNamespace(
        task="noise_suppression", model=None, loss_func=None, vad_label=None, curriculum=None,
        optimizer=SimpleNamespace(type="SGD", learning_rate=0.1, args={}),
        scheduler=SimpleNamespace(type="StepLR", args={"step_size": 10}, warmup_step=1),
        trainer=SimpleNamespace(
            num_gpus=0, find_unused_parameters=False, train_iter_per_epoch=2,
            valid_iter_per_epoch=1, work_folder=str(tmp_path),
            lightning_trainer_args={"max_epochs": 1, "enable_progress_bar": False,
                                    "enable_model_summary": False},
        ),
    )
    loader = torch.utils.data.DataLoader(torch.ones(4, 1, 1), batch_size=None)
    args = SimpleNamespace(pretrained_ckpt_path=None, ckpt_path=None)

    runner.run_training(args, recipe, loader, loader, init_model=lambda config: _Tiny())

    assert list(tmp_path.glob("lightning_logs/version_0/checkpoints/*.ckpt"))


def test_dumping_samples_from_a_short_epoch_writes_what_there_is(tmp_path):
    loader = torch.utils.data.DataLoader([{"x": 1}, {"x": 2}], batch_size=None)
    written = []

    runner.dump_training_samples(
        loader, out_folder=str(tmp_path), n_batches=3,
        write_batch=lambda batch, name: written.append((batch["x"], name)),
    )

    assert written == [(1, f"{tmp_path}/batch_00"), (2, f"{tmp_path}/batch_01")]


class _UnseededDraws(torch.utils.data.Dataset):
    """What the synthesis path does: unseeded draws from the worker's own streams."""

    def __len__(self):
        return 4

    def __getitem__(self, index):
        import threadpoolctl

        pools = [pool["num_threads"] for pool in threadpoolctl.threadpool_info()]
        draws = (float(torch.rand(1)), random.random(), float(np.random.rand()))
        return draws, torch.get_num_threads(), pools


def _worker_draws_on_rank(rank, monkeypatch):
    # seed_everything leaves the same main-process seed on every rank.
    monkeypatch.setattr(lightning_seed.rank_zero_only, "rank", rank)
    torch.manual_seed(1234)
    loader = torch.utils.data.DataLoader(
        _UnseededDraws(), batch_size=None, num_workers=2, worker_init_fn=runner.seed_worker
    )
    return list(loader)


@pytest.mark.parametrize("seed_workers, ranks_differ", [("1", True), ("0", False)])
def test_ranks_draw_their_own_augmentation_only_when_workers_are_seeded(
    monkeypatch, seed_workers, ranks_differ
):
    pytest.importorskip("threadpoolctl")
    monkeypatch.setenv("PL_SEED_WORKERS", seed_workers)
    rank0 = _worker_draws_on_rank(0, monkeypatch)
    rank1 = _worker_draws_on_rank(1, monkeypatch)

    assert all((a[0] != b[0]) == ranks_differ for a, b in zip(rank0, rank1))
    for _, torch_threads, pools in rank0 + rank1:
        assert torch_threads == 1
        assert pools, "no thread pools reported; the check would pass vacuously"
        assert all(count == 1 for count in pools), pools


def test_both_training_loaders_initialise_their_workers():
    root = pathlib.Path(__file__).resolve().parents[2]
    recipe = parse_recipe(
        yaml.safe_load((root / "egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml").read_text())
    )

    class _Stub:
        meta = {f"spk{i}": {"utts": {}} for i in range(8)}

        def __init__(self, **kwargs):
            pass

    loaders = runner.build_dataloaders(dataset_cls=_Stub, collate_fn=lambda b: b, recipe=recipe)
    assert [loader.worker_init_fn for loader in loaders] == [runner.seed_worker] * 2


def test_a_seeded_run_asks_lightning_to_seed_every_worker(monkeypatch):
    for name in ("PL_SEED_WORKERS", "PL_GLOBAL_SEED"):
        monkeypatch.setenv(name, "")
    runner.seed_run(None)
    assert os.environ["PL_SEED_WORKERS"] == ""
    runner.seed_run(7)
    assert (os.environ["PL_GLOBAL_SEED"], os.environ["PL_SEED_WORKERS"]) == ("7", "1")
