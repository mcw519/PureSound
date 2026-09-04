"""Shared plumbing for the v19c pre-flight (Decision point B).

Nothing here measures anything; it builds the exact training distribution and the
exact checkpoint the round would warm-start from, so that P0-P4 all read the same
model on the same rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = str(RECIPE_DIR / "config/exp/train_dpcrn_v19c_inherit.yaml")
V16_EP19 = (
    "/work/any_exp_link/puresound_exp/dpcrn_v16_lengthmix/lightning_logs/"
    "version_0/checkpoints/epoch=19-step=10000.ckpt"
)

SR = 16000
FRAME = 400
HOP = 160
FPS = SR / HOP  # 100


def seed_everything(seed: int) -> None:
    """The audit's own seeding: torch drives the synthesis, so all three matter."""
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class _BlockFilteringRecipe:
    """A recipe proxy that drops augmentation blocks the dataset cannot accept.

    Why this exists: ``BaseRecipe.augmentation_kwargs`` forwards one ``*_args``
    keyword per declared block, and ``DynamicBaseDataset`` rejects any keyword
    outside its own ``AUGMENTATION_BLOCKS``. While a new recipe block is landing
    ahead of its dataset side (``session_rows`` was mid-flight in the working
    tree when this pre-flight ran) every voice-isolation dataloader raises
    ``TypeError`` on construction, including ``scripts/check_training_data.py``.

    A block that is *configured* is never dropped -- that would silently change
    the distribution being measured. Only ``None`` blocks are filtered, which is
    a no-op once the dataset accepts them.
    """

    def __init__(self, recipe, accepted: set[str]):
        self._recipe = recipe
        self._accepted = accepted

    def __getattr__(self, name):
        return getattr(self._recipe, name)

    def augmentation_kwargs(self) -> dict:
        kwargs = self._recipe.augmentation_kwargs()
        dropped = {k: v for k, v in kwargs.items() if k not in self._accepted}
        configured = {k: v for k, v in dropped.items() if v is not None}
        if configured:
            raise TypeError(
                f"the dataset does not accept configured blocks {sorted(configured)}; "
                "refusing to measure a distribution that is not the recipe's"
            )
        if dropped:
            print(f"[loader] dropped unsupported empty blocks: {sorted(dropped)}",
                  flush=True)
        return {k: v for k, v in kwargs.items() if k in self._accepted}


def build_loaders(config_path: str = CONFIG, num_workers: int = 8):
    """(train, valid, recipe) for the v19c recipe, through the recipe's own entry
    point (`egs/voice_isolate/main.init_dataloader`)."""
    import egs.voice_isolate.main as recipe_main
    from puresound.config import load_recipe, with_overrides
    from puresound.task.voice_isolation import VoiceIsolationDataset

    recipe = load_recipe(
        config_path, expected_task="voice_isolation", expected_purpose="train"
    )
    recipe = with_overrides(recipe, trainer={"num_workers": num_workers})
    accepted = set(VoiceIsolationDataset.AUGMENTATION_BLOCKS)
    train, valid = recipe_main.init_dataloader(
        _BlockFilteringRecipe(recipe, accepted)
    )
    return train, valid, recipe


def load_model(config_path: str = CONFIG, ckpt: str = V16_EP19,
               device: str = "cuda:0", with_losses: bool = False):
    """The v16 ep19 weights inside the v19c recipe's model.

    ``with_losses`` also registers the recipe's loss list on the module, which is
    what P2/P3 need (`compute_loss` reads `self.loss_func_list`).
    """
    from puresound.config import load_recipe
    from puresound.recipes import init_loss_func, init_siso_model

    recipe = load_recipe(
        config_path, expected_task="voice_isolation", expected_purpose="train"
    )
    model = init_siso_model(recipe.model)
    blob = torch.load(ckpt, map_location="cpu")
    state = blob.get("state_dict", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} ckpt={ckpt}",
          flush=True)
    if missing:
        print(f"[load] MISSING: {sorted(missing)[:8]}", flush=True)
    if with_losses:
        losses, weights = init_loss_func(recipe.loss_func)
        model.register_loss_func(losses, weights)
        names = [type(loss).__name__ for loss in losses]
        print(f"[load] losses={list(zip(names, weights))}", flush=True)
    dev = torch.device(device)
    return model.to(dev).eval(), recipe, dev


# --------------------------------------------------------------------------- #
# row typing

#: `mix_mode` is emitted as a float code (see task/voice_isolation.py).
MIX_MODE_NAMES = {0.0: "none", 1.0: "legacy", 2.0: "physical", 3.0: "moderate",
                  4.0: "counter_level", 5.0: "distance_level"}


def scalar(batch: dict, key: str, row: int):
    value = batch.get(key)
    if value is None:
        return None
    x = value.reshape(-1)[row]
    return float(x) if torch.isfinite(x) else None


def row_type(batch: dict, row: int) -> str:
    """Which of the recipe's three row shapes this row is, from emitted scalars.

    `RowPlan.use_realfar` / `use_realnear` are *not* emitted, so this is read off
    what is (verified against a sampled batch): real-near rows have no simulated
    foreground RIR, so `foreground_drr` is NaN; real-far rows skip `mix_mode`
    (the simulated-near / real-recorded-far level ratio is not physical) and
    have no interferer RIR, so `strongest_interferer_drr` is NaN while
    `mix_mode` reads the base path's `legacy`. Turn-taking cross-cuts all three
    and is `is_turn_taking`, not a fourth value here.
    """
    fg_drr = scalar(batch, "foreground_drr", row)
    itf_drr = scalar(batch, "strongest_interferer_drr", row)
    n_itf = scalar(batch, "n_interferers", row) or 0.0
    if fg_drr is None:
        return "real-near"
    if n_itf >= 1 and itf_drr is None:
        return "real-far"
    return "synthetic"


def is_turn_taking(batch: dict, row: int) -> bool:
    """1.0 when the row used the conversational turn-taking gating (long
    alternating near/far turns) rather than per-frame Bernoulli overlap."""
    return bool(scalar(batch, "turn_taking", row))


def bucket(batch: dict, row: int) -> float:
    """The length bucket, in seconds, this row was drawn at."""
    return round(float(batch["length"].reshape(-1)[row].item()) / SR, 2)


# --------------------------------------------------------------------------- #
# reporting


def describe(values, label: str = "") -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    # winsorised mean beside the median: the round's metric policy forbids a bare
    # mean, because the unfloored form of this statistic is a tail detector.
    lo, hi = np.percentile(arr, [5, 95])
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "winsor_mean": float(np.clip(arr, lo, hi).mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "label": label,
    }


def fmt(stats: dict) -> str:
    if not stats.get("n"):
        return "n=0"
    return (f"n={stats['n']:<5d} med={stats['median']:+8.3f} "
            f"p25={stats['p25']:+8.3f} p75={stats['p75']:+8.3f} "
            f"wmean={stats['winsor_mean']:+8.3f}")
