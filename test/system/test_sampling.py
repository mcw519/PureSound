"""The coverage sampler's walk carried through checkpoints, and its runner wiring."""
import pathlib
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import pytest
import torch
import yaml

from puresound.config.loader import _parse_recipe as parse_recipe
from puresound.system import runner
from puresound.system.sampling import STATE_KEY, CoverageSamplerCheckpoint, attach_coverage_sampler
from puresound.task.sampler import CoverageSampler

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
NS_RECIPE = REPO_ROOT / "egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml"
SR = 16000


def _meta(n=20):
    return {f"spk{i}": {"utts": {f"spk{i}_{j}": {"sr": SR, "length": 12 * SR} for j in range(3)}}
            for i in range(n)}


def _sampler():
    return CoverageSampler(data=_meta(), total_batch=4, n_items=3, default_seconds=4.0)


def _items(sampler, epoch):
    sampler.set_epoch(epoch)
    return [item for batch in sampler for item in batch]


def _args(**kw):
    return SimpleNamespace(**{"set_seed": 5, "ckpt_path": None, "pretrained_ckpt_path": None, **kw})


def _checkpoint(tmp_path, sampler, epoch):
    """What Lightning writes after ``epoch`` through the callback."""
    checkpoint = {"epoch": epoch}
    callback = CoverageSamplerCheckpoint(sampler)
    trainer = SimpleNamespace(current_epoch=epoch)
    callback.on_train_epoch_start(trainer, None)
    for batch_idx in range(len(sampler)):
        callback.on_train_batch_end(trainer, None, None, None, batch_idx)
    callback.on_save_checkpoint(trainer, None, checkpoint)
    path = tmp_path / f"epoch={epoch}.ckpt"
    torch.save(checkpoint, path)
    return path


def test_a_warm_start_continues_the_walk_and_a_resume_replaces_it(tmp_path):
    first = _sampler()
    attach_coverage_sampler(first, _args())
    walked = [_items(first, e) for e in range(3)]
    path = _checkpoint(tmp_path, first, epoch=1)
    assert torch.load(path, weights_only=False)[STATE_KEY]["next"]["batch"] == 8

    resumed = _sampler()
    attach_coverage_sampler(resumed, _args(ckpt_path=str(path), set_seed=99))
    assert _items(resumed, 2) == walked[2]

    warm = _sampler()
    attach_coverage_sampler(warm, _args(pretrained_ckpt_path=str(path), set_seed=99))
    assert _items(warm, 0) == walked[2]


def test_a_checkpoint_without_a_walk_cannot_be_resumed_but_can_seed_a_new_one(tmp_path):
    path = tmp_path / "other.ckpt"
    torch.save({"epoch": 3}, path)
    with pytest.raises(ValueError, match="no coverage-sampler walk"):
        attach_coverage_sampler(_sampler(), _args(ckpt_path=str(path)))
    warm = _sampler()
    attach_coverage_sampler(warm, _args(pretrained_ckpt_path=str(path)))
    fresh = _sampler()
    attach_coverage_sampler(fresh, _args())
    assert _items(warm, 0) == _items(fresh, 0)


def test_the_recipe_selects_the_coverage_sampler_for_training_only():
    config = yaml.safe_load(NS_RECIPE.read_text())
    config["trainer"].update(
        train_sampler="coverage", n_utt_per_speaker=1, n_spk_per_batch=6,
        length_schedule=[{"seconds": 4.0, "n_spk": 6, "prob": 0.5},
                         {"seconds": 10.0, "n_spk": 2, "prob": 0.5}],
    )
    config["trainer"].pop("speaker_source_weights", None)
    recipe = parse_recipe(config)

    class _Stub:
        meta = _meta()

        def __init__(self, **kwargs):
            pass

    train, valid = runner.build_dataloaders(dataset_cls=_Stub, collate_fn=lambda b: b, recipe=recipe)
    assert isinstance(train.batch_sampler, CoverageSampler)
    assert not isinstance(valid.batch_sampler, CoverageSampler)
    assert {b[0][3] for b in train.batch_sampler} == {4.0, 10.0}


class _Rows(torch.utils.data.Dataset):
    def __getitem__(self, key):
        return torch.tensor([int(key[0][3:]), key[2]])


class _TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.seen = 0
        self.seen_seeds = []

    def training_step(self, batch, batch_idx):
        self.seen += 1
        self.seen_seeds.extend(batch[:, 1].tolist())
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


def test_max_steps_checkpoint_counts_consumed_batches_instead_of_a_full_epoch(tmp_path):
    sampler = _sampler()
    attach_coverage_sampler(sampler, _args())
    monitor = ModelCheckpoint(dirpath=tmp_path, save_on_train_epoch_end=True,
                              every_n_epochs=1, save_top_k=-1)
    model = _TinyModel()
    trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=2, max_epochs=3,
                        logger=False, enable_progress_bar=False, enable_model_summary=False,
                        callbacks=[monitor, CoverageSamplerCheckpoint(sampler)])
    trainer.fit(model, torch.utils.data.DataLoader(_Rows(), batch_sampler=sampler, num_workers=1))
    state = torch.load(monitor.best_model_path, weights_only=False)[STATE_KEY]
    assert model.seen == 2
    assert state["next"]["batch"] == 2
    assert state["next"]["slot"] == 6
    assert not state["epoch_boundary"]

    with pytest.raises(ValueError, match="epoch-boundary checkpoint"):
        attach_coverage_sampler(_sampler(), _args(ckpt_path=monitor.best_model_path))
    warm = _sampler()
    attach_coverage_sampler(warm, _args(pretrained_ckpt_path=monitor.best_model_path))
    expected = _items(_sampler_with_seed(5), 0)[6:]
    assert _items(warm, 0)[:6] == expected


def _sampler_with_seed(seed):
    sampler = _sampler()
    sampler.set_seed(seed)
    return sampler


@pytest.mark.parametrize("num_workers", [0, 1])
def test_lightning_resume_at_epoch_boundary_consumes_the_next_epoch(tmp_path, num_workers):
    def fit(sampler, max_epochs, ckpt_path=None):
        callback = attach_coverage_sampler(sampler, _args(ckpt_path=ckpt_path))
        monitor = ModelCheckpoint(dirpath=tmp_path / str(max_epochs),
                                  save_on_train_epoch_end=True, every_n_epochs=1,
                                  save_top_k=-1)
        model = _TinyModel()
        trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs,
                            logger=False, enable_progress_bar=False,
                            enable_model_summary=False, callbacks=[monitor, callback])
        trainer.fit(model, torch.utils.data.DataLoader(_Rows(), batch_sampler=sampler,
                                                     num_workers=num_workers),
                    ckpt_path=ckpt_path)
        return model, monitor.best_model_path

    first, path = fit(_sampler(), 1)
    state = torch.load(path, weights_only=False)[STATE_KEY]
    assert state["epoch_boundary"] and state["next"]["batch"] == 4
    resumed, resumed_path = fit(_sampler(), 2, ckpt_path=path)
    expected = _sampler_with_seed(5)
    assert first.seen_seeds == [item[2] for item in _items(expected, 0)]
    assert resumed.seen_seeds == [item[2] for item in _items(expected, 1)]
    assert torch.load(resumed_path, weights_only=False)[STATE_KEY]["next"]["batch"] == 8

    # Iteration-based Lightning resumes can retain the previous epoch index.
    sampler = _sampler()
    callback = attach_coverage_sampler(sampler, _args(ckpt_path=path))
    model = _TinyModel()
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=2, max_steps=8,
                        logger=False, enable_progress_bar=False, enable_model_summary=False,
                        enable_checkpointing=False, callbacks=[callback])
    with pytest.raises(ValueError, match="restored epoch does not match"):
        trainer.fit(model, torch.utils.data.DataLoader(_Rows(), batch_sampler=sampler,
                                                     num_workers=num_workers),
                    ckpt_path=path)
    assert model.seen == 0
