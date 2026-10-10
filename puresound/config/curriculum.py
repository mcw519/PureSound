"""Epoch-scheduled recipe knobs -- the ``curriculum`` block.

A recipe states each knob once and trains at that value from the first step to
the last. Some knobs are better moved *during* a run: a loss term introduced only
after the model can do the base task, a row type mixed in gradually, a sampling
mixture that starts narrow and widens. Without a schedule the only way to express
that is to split the run and warm-start the second half from the first, which
leaves the schedule in the operator's notes instead of in the recipe. A
``curriculum`` block puts it back into the recipe: one track per knob, giving its
value by epoch.

    curriculum:
      used: True
      tracks:
        - path: "loss:SDRLoss"                # a registered loss's weight
          interp: linear
          points: [[0, 0.0], [30, 1.0]]
        - path: "aug:augmentation_noise.prob" # a knob a row consults
          interp: step
          points: [[0, 0.0], [40, 0.8]]
        - path: "bank:wide"                   # a union RIR bank member's weight
          points: [[0, 0.1], [40, 0.5]]

Three kinds of target, addressed by prefix, because they are applied in three
different places and a typo has to fail at config load rather than quietly do
nothing:

``aug:<block>.<field>``
    an augmentation knob, applied by the dataset -- which lives in a DataLoader
    worker, so the epoch travels to it on the item (see ``task.sampler``).
``bank:<member>``
    one member's weight in a union RIR bank. Weights are relative and the set is
    renormalised, so a track may move one member and leave the others alone.
``loss:<Type>`` or ``loss:#<index>``
    the weight of one entry of the recipe's ``loss_func``, applied on the module.

Two things are deliberately un-schedulable, both because getting them wrong is
silent rather than loud. A block's ``used`` flag: synthesis skips a disabled
block *before* its random draw, so flipping it mid-run shifts every draw that
follows and the rows either side of the flip stop being drawn from the same
process -- ramp the block's probability from zero instead, which keeps the draw
and changes only its outcome. And anything a dataset reads once when it is built
(corpus paths, bank layouts, cache sizes): scheduling those looks like it works
and changes nothing, so the allowlist below refuses them by omission.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, NamedTuple

from pydantic import BeforeValidator, Field, StrictBool, model_validator

from .base import StrictConfig


#: Augmentation knobs a schedule may move: every one is either read straight off
#: the config block while a row is being built, or held by a component the
#: dataset re-derives when a value changes
#: (``DynamicBaseDataset.rebind_augmentation_blocks``). Blocks belong to
#: different tasks and a recipe only ever names its own, so the set is one flat
#: allowlist rather than a per-task table.
#:
#: Adding an entry is a promise that the knob still has an effect after the row
#: is drawn -- check both, because a knob captured somewhere that is never
#: re-derived fails by doing nothing at all.
SCHEDULABLE_AUGMENTATION_PATHS: frozenset[str] = frozenset(
    {
        # Sources and rooms.
        "augmentation_speech.prob",
        "augmentation_speech.media_voice.prob",
        "augmentation_speech.overlap_control.no_overlap_prob",
        "augmentation_speech.overlap_control.high_overlap_prob",
        "augmentation_reverb.prob",
        "augmentation_noise.prob",
        "augmentation_noise.prob_white_noise",
        "augmentation_noise.room_coloring.prob",
        "augmentation_noise.absolute_floor.prob",
        # Capture and transmission chain.
        "augmentation_speed.prob",
        "augmentation_src.prob",
        "augmentation_ir_response.prob",
        "augmentation_hpf.prob",
        "augmentation_volume.prob",
        "augmentation_compressor.prob",
        "augmentation_codec.prob",
        "augmentation_packet_loss.prob",
        # Row types.
        "augmentation_target_absent.prob",
        "augmentation_row_initial_ambient.prob",
        "augmentation_realfar.prob",
        "augmentation_realfar.lone_far_prob",
        "augmentation_realfar.turn_taking_prob",
        "augmentation_realnear.prob",
        "augmentation_realnear.turn_taking_prob",
        "augmentation_session_rows.prob",
    }
)

#: Path prefixes, in the order the documentation introduces them.
TRACK_KINDS = ("aug", "bank", "loss")


def split_track_path(path: str) -> tuple[str, str]:
    """``"aug:augmentation_noise.prob"`` -> ``("aug", "augmentation_noise.prob")``."""
    kind, separator, reference = path.partition(":")
    kind, reference = kind.strip(), reference.strip()
    if not separator or not kind or not reference:
        raise ValueError(
            f"curriculum path must be '<kind>:<reference>', got {path!r} "
            f"(kinds: {', '.join(TRACK_KINDS)})"
        )
    if kind not in TRACK_KINDS:
        raise ValueError(
            f"unknown curriculum path kind {kind!r} in {path!r} "
            f"(kinds: {', '.join(TRACK_KINDS)})"
        )
    return kind, reference


def _as_point(value: Any) -> Any:
    """Accept ``[epoch, value]`` beside the spelled-out mapping, and take an int
    where a float is meant -- ``points: [[0, 0], [30, 1]]`` is how a ramp gets
    written in YAML, and strict validation would otherwise reject both ends."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("a curriculum point pair is [epoch, value]")
        value = {"epoch": value[0], "value": value[1]}
    if isinstance(value, Mapping) and "value" in value:
        value = dict(value)
        candidate = value["value"]
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise ValueError("curriculum point value must be a number")
        value["value"] = float(candidate)
    return value


class CurriculumPoint(StrictConfig):
    """One keyframe: at this epoch the knob has this value."""

    epoch: int = Field(ge=0)
    value: float


Point = Annotated[CurriculumPoint, BeforeValidator(_as_point)]


class CurriculumTrack(StrictConfig):
    """One knob and its value by epoch.

    ``linear`` interpolates between neighbouring points; ``step`` holds each
    point's value until the next point's epoch is reached. Outside the first and
    last points the value is held flat, so a track never extrapolates: a run that
    outlives its schedule keeps training at the value it ended on.
    """

    path: str
    interp: Literal["linear", "step"] = "linear"
    points: list[Point] = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self):
        kind, reference = split_track_path(self.path)
        if kind == "aug" and reference not in SCHEDULABLE_AUGMENTATION_PATHS:
            raise ValueError(
                f"{reference!r} is not a schedulable augmentation knob. "
                "Schedulable: " + ", ".join(sorted(SCHEDULABLE_AUGMENTATION_PATHS))
            )
        if kind == "loss" and reference.startswith("#") and not reference[1:].isdigit():
            raise ValueError(f"loss index must be '#<n>', got {reference!r}")
        epochs = [point.epoch for point in self.points]
        if any(later <= earlier for earlier, later in zip(epochs, epochs[1:])):
            raise ValueError(f"curriculum points must ascend by epoch, got {epochs}")
        return self

    @property
    def kind(self) -> str:
        return split_track_path(self.path)[0]

    @property
    def reference(self) -> str:
        return split_track_path(self.path)[1]

    def value_at(self, epoch: int) -> float:
        points = self.points
        if epoch <= points[0].epoch:
            return points[0].value
        if epoch >= points[-1].epoch:
            return points[-1].value
        for low, high in zip(points, points[1:]):
            if low.epoch <= epoch <= high.epoch:
                if self.interp == "step":
                    return high.value if epoch == high.epoch else low.value
                span = high.epoch - low.epoch
                return low.value + (epoch - low.epoch) / span * (high.value - low.value)
        return points[-1].value  # pragma: no cover -- points are sorted


class CurriculumValues(NamedTuple):
    """What every track says at one epoch, grouped by who applies it."""

    #: dotted augmentation path -> value, e.g. ``augmentation_noise.prob``
    augmentation: dict[str, float]
    #: union bank member name -> relative sampling weight
    bank_weights: dict[str, float]
    #: loss reference (``Type`` or ``#index``) -> weight
    loss_weights: dict[str, float]

    def augmentation_overrides(self) -> dict[str, dict[str, Any]]:
        """Regroup the flat paths into ``block -> nested field mapping``, the
        shape :func:`puresound.config.base.with_overrides` merges field-wise."""
        overrides: dict[str, dict[str, Any]] = {}
        for path, value in self.augmentation.items():
            block, _, remainder = path.partition(".")
            cursor = overrides.setdefault(block, {})
            fields = remainder.split(".")
            for field in fields[:-1]:
                cursor = cursor.setdefault(field, {})
            cursor[fields[-1]] = value
        return overrides

    def describe(self) -> str:
        parts = [f"{path}={value:g}" for path, value in sorted(self.augmentation.items())]
        parts += [f"bank:{name}={value:g}" for name, value in sorted(self.bank_weights.items())]
        parts += [f"loss:{name}={value:g}" for name, value in sorted(self.loss_weights.items())]
        return ", ".join(parts) if parts else "(no tracks)"


class CurriculumConfig(StrictConfig):
    """Knob schedules for one training run.

    Validating the *targets* -- that the block a track names exists and is
    enabled, that the bank member is in the union, that the loss reference
    resolves to exactly one registered loss -- needs the whole recipe and so
    lives in :mod:`puresound.config.recipe`.
    """

    used: StrictBool
    tracks: list[CurriculumTrack] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_paths(self):
        seen: set[str] = set()
        for track in self.tracks:
            if track.path in seen:
                raise ValueError(f"curriculum track {track.path!r} is scheduled twice")
            seen.add(track.path)
        return self

    def resolve(self, epoch: int) -> CurriculumValues:
        augmentation: dict[str, float] = {}
        bank_weights: dict[str, float] = {}
        loss_weights: dict[str, float] = {}
        if not self.used:
            return CurriculumValues(augmentation, bank_weights, loss_weights)
        for track in self.tracks:
            target = {
                "aug": augmentation,
                "bank": bank_weights,
                "loss": loss_weights,
            }[track.kind]
            target[track.reference] = track.value_at(int(epoch))
        return CurriculumValues(augmentation, bank_weights, loss_weights)

    def describe(self, epoch: int) -> str:
        return self.resolve(epoch).describe()
