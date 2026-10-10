"""The benchmark record: what was scored, how, and what the gate said about it.

A comparison that exists only in a run directory cannot be checked later and
tends to be redone under a different protocol. This is the written form. Each
field below is mandatory because leaving it out can silently invalidate a
comparison:

``recipe``
    A recipe and a checkpoint that disagree load *partially* -- keys the model does
    not have are dropped, modules the checkpoint does not carry stay at their random
    init. The run still produces numbers.
``chain_commit``
    The scoring-time source tree. A frozen synthetic set separately carries its
    build commit, recipe digest and seed in the stage's provenance; scoring-time
    state cannot identify audio that was generated earlier.
``inference``
    A number without its operating point is not reproducible. The released blend is
    part of the configuration, not an extra.
``stages[].n`` and ``stages[].ci``
    A stage whose interval covers the difference being claimed has measured nothing.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .statistics import Interval


#: A stage that can decide a release, versus one that is only watched. A set that
#: cannot separate the candidate from doing nothing is a monitor, whatever it was
#: built to be.
ROLES = ("gate", "monitor")
VERDICTS = ("pass", "fail", "no-resolution")


def chain_commit(repo_root: str | Path | None = None, paths: Sequence[str] = (".",)) -> str:
    """``<short sha>`` of the source tree, ``+dirty`` if selected paths changed."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    head = git("rev-parse", "--short", "HEAD")
    if head is None:
        return "unknown"
    dirty = git("status", "--porcelain", "--", *paths)
    return f"{head}+dirty" if dirty else head


@dataclass
class StageResult:
    """One benchmark stage, scored.

    ``value`` is the treatment's own number and ``difference`` is treatment minus
    baseline. Both are kept: the difference is what the verdict reads, and the
    absolute value is what stops a large improvement on a terrible starting point
    from reading as a good end state.
    """

    name: str
    metric: str
    role: str
    n: int
    value: float
    baseline: float | None = None
    difference: dict[str, float] | None = None
    verdict: str = "no-resolution"
    direction: str = "higher_is_better"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    @classmethod
    def from_interval(
        cls,
        name: str,
        *,
        metric: str,
        role: str,
        value: float,
        baseline: float,
        interval: Interval,
        verdict: str,
        direction: str,
        notes: str = "",
        **extra: Any,
    ) -> "StageResult":
        return cls(
            name=name,
            metric=metric,
            role=role,
            n=interval.n,
            value=float(value),
            baseline=float(baseline),
            difference={
                "point": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
            },
            verdict=verdict,
            direction=direction,
            notes=notes,
            extra=dict(extra),
        )

    def line(self) -> str:
        head = f"{self.verdict.upper():<14} {self.name:<28} {self.metric:<12}"
        body = f"{self.value:+.4f}"
        if self.difference is not None:
            body += (
                f"  vs {self.baseline:+.4f}  delta {self.difference['point']:+.4f}"
                f" [{self.difference['ci_low']:+.4f}, {self.difference['ci_high']:+.4f}]"
            )
        return f"{head} {body}  n={self.n}  ({self.role})"


@dataclass
class GateRecord:
    """Everything needed to compare this scoring run against another one."""

    tag: str
    checkpoint: str
    recipe: str
    chain_commit: str
    inference: dict[str, Any] = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Aggregate gate stages without turning uncertainty into a pass."""
        gates = [stage for stage in self.stages if stage.role == "gate"]
        if any(stage.verdict == "fail" for stage in gates):
            return "fail"
        if not gates or any(stage.verdict == "no-resolution" for stage in gates):
            return "no-resolution"
        return "pass"

    @property
    def unresolved(self) -> list[str]:
        return [
            stage.name
            for stage in self.stages
            if stage.role == "gate" and stage.verdict == "no-resolution"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "checkpoint": self.checkpoint,
            "recipe": self.recipe,
            "chain_commit": self.chain_commit,
            "inference": dict(self.inference),
            "verdict": self.verdict,
            "unresolved_gates": self.unresolved,
            "stages": [asdict(stage) for stage in self.stages],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def summary(self) -> str:
        lines = [
            f"=== {self.tag} === verdict={self.verdict}",
            f"    checkpoint : {self.checkpoint}",
            f"    recipe     : {self.recipe}",
            f"    chain      : {self.chain_commit}",
            f"    inference  : {self.inference or '{}'}",
        ]
        lines.extend("    " + stage.line() for stage in self.stages)
        if self.unresolved:
            lines.append(
                "    NOTE: gate stage(s) could not resolve a difference: "
                + ", ".join(self.unresolved)
            )
        return "\n".join(lines)


def read_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_stage_files(paths: Sequence[str | Path]) -> list[StageResult]:
    """Collect stage results each written by its own stage process."""
    stages: list[StageResult] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            stages.append(StageResult(**entry))
    return stages


def write_stage(path: str | Path, stages: StageResult | Sequence[StageResult]) -> Path:
    """One stage process writes its own result; the driver merges them at the end."""
    items = [stages] if isinstance(stages, StageResult) else list(stages)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


