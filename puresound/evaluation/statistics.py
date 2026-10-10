"""The statistics a benchmark record has to carry to mean anything.

Every number here exists to answer one question: **is the difference I am about to
claim bigger than the noise in how I measured it?** Three ways of getting that wrong,
and what guards against each:

- A test set whose confidence interval covers the difference being claimed has not
  measured anything. Recording that as a win promotes a version on noise, so
  :func:`verdict` can return ``"no-resolution"`` and the caller has to handle it.
- One checkpoint is not a measurement. Neighbouring epochs of the same run move some
  metrics by more than the version differences being compared, so a run is scored as
  a **block** of its last checkpoints (:func:`block_summary`).
- Unpaired comparison throws away the pairing. The same utterances go through both
  systems, so the paired difference has a fraction of the spread of either side
  (:func:`paired_bootstrap_ci`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np


Aggregate = Callable[[np.ndarray], float]

#: Lower is better for error rates, higher is better for quality scores. A verdict
#: cannot be read off an interval without knowing which.
Direction = Literal["higher_is_better", "lower_is_better"]


def _as_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got shape {array.shape}")
    return array


@dataclass(frozen=True)
class Interval:
    """A point estimate and the interval it is uncertain within."""

    point: float
    low: float
    high: float
    n: int

    @property
    def resolves(self) -> bool:
        """Does the interval exclude zero? If not, nothing was measured.

        A single item has no spread to measure, and an interval with a
        non-finite end says nothing, so neither counts as resolving.
        """
        if self.n < 2 or not (math.isfinite(self.low) and math.isfinite(self.high)):
            return False
        return not (self.low <= 0.0 <= self.high)

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}] n={self.n}"


def bootstrap_ci(
    values: Sequence[float],
    *,
    aggregate: Aggregate = np.median,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for one set of per-item scores."""
    array = _as_array(values)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")

    point = float(aggregate(array))
    if array.size == 1:
        return Interval(point, point, point, 1)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    stats = np.apply_along_axis(aggregate, 1, array[draws])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return Interval(point, float(low), float(high), int(array.size))


def paired_bootstrap_ci(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    aggregate: Aggregate = np.median,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Interval on ``treatment - baseline``, resampling the **pairs**.

    The two sequences must be the same items in the same order. Resampling them
    independently would report the spread of the metric rather than the spread of
    the difference, which is usually several times larger and hides real effects.
    """
    left, right = _as_array(treatment), _as_array(baseline)
    if left.size != right.size:
        raise ValueError(
            f"paired comparison needs equal lengths, got {left.size} and {right.size}"
        )
    return bootstrap_ci(
        left - right,
        aggregate=aggregate,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def wilcoxon_p(treatment: Sequence[float], baseline: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon p-value; ``1.0`` when every pair is identical."""
    from scipy.stats import wilcoxon

    left, right = _as_array(treatment), _as_array(baseline)
    if left.size != right.size:
        raise ValueError("paired comparison needs equal lengths")
    difference = left - right
    if left.size == 0 or np.allclose(difference, 0.0):
        return 1.0
    return float(wilcoxon(left, right, zero_method="zsplit").pvalue)


@dataclass(frozen=True)
class BlockSummary:
    """One run, scored as the block of its last checkpoints."""

    point: float
    spread: float
    per_checkpoint: tuple[float, ...]

    @property
    def n_checkpoints(self) -> int:
        return len(self.per_checkpoint)

    def __str__(self) -> str:
        return (
            f"{self.point:+.4f} (spread {self.spread:.4f} over "
            f"{self.n_checkpoints} ckpt)"
        )


def block_summary(
    per_checkpoint: Sequence[float], *, aggregate: Aggregate = np.median
) -> BlockSummary:
    """Summarise a run from several of its checkpoints.

    ``spread`` is the full range, not a standard deviation: with four or five
    checkpoints the range is what a reader needs to see, and a standard deviation
    over five points invites more confidence than five points support.
    """
    array = _as_array(per_checkpoint)
    if array.size == 0:
        raise ValueError("a block needs at least one checkpoint")
    return BlockSummary(
        point=float(aggregate(array)),
        spread=float(array.max() - array.min()),
        per_checkpoint=tuple(float(value) for value in array),
    )


def verdict(
    difference: Interval,
    *,
    direction: Direction,
    tolerance: float = 0.0,
) -> str:
    """``"pass"`` / ``"fail"`` / ``"no-resolution"`` for a treatment-minus-baseline interval.

    ``tolerance`` is the regression a stage is allowed to absorb: a do-no-harm
    monitor set to ``0.01`` passes a degradation whose whole interval stays inside
    one point of WER. It is not a way to wave through a difference the set cannot
    resolve -- that is still ``"no-resolution"``.
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(f"unknown direction: {direction!r}")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    if not difference.resolves:
        return "no-resolution"

    improved = (
        difference.low > 0.0 if direction == "higher_is_better" else difference.high < 0.0
    )
    if improved:
        return "pass"

    if tolerance:
        within = (
            difference.low > -tolerance
            if direction == "higher_is_better"
            else difference.high < tolerance
        )
        if within:
            return "pass"
    return "fail"
