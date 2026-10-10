"""What a benchmark stage runs its audio through.

Every stage scores two systems, not one: the candidate, and **doing nothing**. That
is not a formality. An absolute score says nothing on its own -- a set where the
unprocessed mixture already scores well has no headroom to win, and a large
improvement on a terrible starting point is not a good end state. The paired
difference against :class:`Passthrough` is what a verdict is read from.

It also means the whole benchmark can run before any model exists.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import torch

from puresound.audio.io import AudioIO


class System(Protocol):
    """Something that turns noisy audio into (hopefully) less noisy audio."""

    name: str

    def process(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """``wav`` is ``[1, T]`` float in [-1, 1]; the result has the same shape."""
        ...

    def describe(self) -> dict[str, Any]:
        """What a record needs to reproduce this system."""
        ...


@dataclass
class Passthrough:
    """The do-nothing baseline. Returns its input unchanged."""

    name: str = "unprocessed"

    def process(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return wav

    def describe(self) -> dict[str, Any]:
        return {"system": "passthrough"}


def run_system(system: "System", key: str, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Run one system on one item, whichever kind it is.

    A model takes the waveform. A system whose output already exists on disk
    (:class:`PrecomputedSystem`) needs to know *which* item this is, and the
    waveform tells it nothing -- so the scoring tools hand every system the item's
    key as well, and this is the one place that decides which call to make.
    ``key`` is the input file's stem (``p232_001_mix``, ``fileid_123``).
    """
    if getattr(system, "wants_item", False):
        return system.process_item(key, wav, sample_rate)
    return system.process(wav, sample_rate)


@dataclass
class PrecomputedSystem:
    """Audio somebody else already enhanced, scored through the same stages.

    Puts a third-party model on the same axis as our checkpoints and doing
    nothing: run the other system offline over the same input files, point this
    at the output directory, and every stage reads its output where it would
    have read ours. RTF does not apply -- there is no model here to time -- and the
    gate driver skips that stage when given a directory.

    Files are found by the input file's stem: ``<dir>/<stem><suffix>`` first, then
    a single ``<stem>_*<suffix>`` (tools like DeepFilterNet append their model name
    to the output filename). Two candidates is an error rather than a guess --
    scoring the wrong file is the failure that would look like a result.
    """

    directory: Path
    name: str
    suffix: str = ".wav"
    wants_item: ClassVar[bool] = True

    def __post_init__(self):
        self.directory = Path(self.directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f"precomputed output directory not found: {self.directory}")

    def resolve(self, key: str) -> Path:
        exact = self.directory / f"{key}{self.suffix}"
        if exact.exists():
            return exact
        matches = sorted(self.directory.glob(f"{glob.escape(key)}_*{glob.escape(self.suffix)}"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"{self.name}: no output for {key!r} in {self.directory} "
                f"(looked for {exact.name} and {key}_*{self.suffix})"
            )
        raise ValueError(
            f"{self.name}: {len(matches)} outputs match {key!r} in {self.directory}: "
            f"{[m.name for m in matches]}. Name them so the stem is unambiguous."
        )

    def process(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise TypeError(
            "PrecomputedSystem needs the item key; call it through run_system()"
        )

    def process_item(self, key: str, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        out, _ = AudioIO.open(str(self.resolve(key)), resample_to=sample_rate)
        return out.reshape(1, -1)

    def describe(self) -> dict[str, Any]:
        return {"system": "precomputed", "directory": str(self.directory), "name": self.name}


@dataclass
class CheckpointSystem:
    """A trained model, at a stated operating point.

    ``chunk_seconds`` bounds memory on long files. It is off by default because
    chunking is not free: a model whose suppression depends on context behaves
    differently in 10 s windows than on a whole session, and the difference can
    be large. Turn it on knowingly.
    """

    model: torch.nn.Module
    device: torch.device
    name: str = "model"
    checkpoint: str = ""
    recipe: str = ""
    dry_blend: float = 1.0
    chunk_seconds: float | None = None
    unused_checkpoint_params: tuple[str, ...] = ()

    @torch.no_grad()
    def process(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = wav.to(self.device)
        if self.chunk_seconds is None:
            return self._forward(wav).cpu()

        span = int(self.chunk_seconds * sample_rate)
        pieces = [
            self._forward(wav[..., start : start + span])
            for start in range(0, wav.shape[-1], span)
        ]
        return torch.cat(pieces, dim=-1).cpu()

    def _forward(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.model(
            wav, dry_blend=self.dry_blend
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(1, -1)

    def describe(self) -> dict[str, Any]:
        return {
            "system": "checkpoint",
            "checkpoint": self.checkpoint,
            "recipe": self.recipe,
            "dry_blend": self.dry_blend,
            "chunk_seconds": self.chunk_seconds,
            "unused_checkpoint_params": list(self.unused_checkpoint_params),
        }


#: Checkpoint keys that belong to training, not to the model. A checkpoint carries
#: its loss modules' buffers; an inference model has nowhere to put them, and their
#: absence says nothing about whether the weights loaded.
NON_MODEL_PREFIXES: tuple[str, ...] = ("loss_func_list.", "loss_func.")


def load_checkpoint_system(
    recipe_path: str | Path,
    checkpoint_path: str | Path,
    *,
    task: str = "noise_suppression",
    device: str | torch.device = "cpu",
    dry_blend: float = 1.0,
    chunk_seconds: float | None = None,
    name: str = "model",
) -> CheckpointSystem:
    """Build the model from its recipe and load the checkpoint **whole**.

    A recipe and a checkpoint that disagree load partially and still produce
    numbers, so the mismatch is reported here rather than discovered as an
    unexplained regression three stages later.
    """
    from puresound.config import load_recipe
    from puresound.recipes import init_model_for_task

    device = torch.device(device)
    recipe = load_recipe(str(recipe_path), expected_task=task)
    model = init_model_for_task(task)(recipe.model)

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Training state that inference has no model to put back. Dropping it is
    # correct; dropping an actual weight is not, so the two are judged apart.
    unexpected = [
        key
        for key in unexpected
        if not any(key.startswith(prefix) for prefix in NON_MODEL_PREFIXES)
    ]
    # A weight the model needs and the checkpoint does not supply stays at its
    # random init, and the run still produces numbers. This is also how a renamed
    # parameter shows up -- as missing on one side and unexpected on the other --
    # so this check catches the rename too.
    if missing:
        raise ValueError(
            f"{Path(checkpoint_path).name} does not fit {Path(recipe_path).name}: "
            f"{len(missing)} parameter(s) would stay at random init "
            f"(e.g. {list(missing)[:5]}). Scoring this would measure a partly "
            "untrained model."
        )

    # The other direction is a checkpoint that carries more than this recipe builds
    # -- an auxiliary head the inference config leaves out. The audio path is whole,
    # so this is reported rather than fatal, and it goes into the record.
    if unexpected:
        print(
            f"[systems] {Path(checkpoint_path).name} carries {len(unexpected)} "
            f"parameter(s) {Path(recipe_path).name} does not build, so they are "
            f"unused here: {list(unexpected)[:5]}"
        )

    return CheckpointSystem(
        model=model.to(device).eval(),
        device=device,
        name=name,
        checkpoint=str(checkpoint_path),
        recipe=str(recipe_path),
        dry_blend=dry_blend,
        chunk_seconds=chunk_seconds,
        unused_checkpoint_params=tuple(unexpected),
    )


def load_system(
    recipe_path: str | Path | None,
    checkpoint_path: str | Path | None,
    **kwargs: Any,
) -> System:
    """A checkpoint when one is given, otherwise the passthrough baseline."""
    if checkpoint_path is None:
        return Passthrough()
    if recipe_path is None:
        raise ValueError("a checkpoint needs the recipe its model is built from")
    return load_checkpoint_system(recipe_path, checkpoint_path, **kwargs)
