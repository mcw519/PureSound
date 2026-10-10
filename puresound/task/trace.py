"""Stage-by-stage snapshots of one synthesised row, for inspection.

The synthesis path -- ``NoiseSuppressionDataset.__getitem__``, ``NoiseStage``
and ``DeviceChain`` -- hands the pair as it stands to ``SynthesisTrace.tap`` at
every stage boundary, with the values the stage drew. A trace is passive: it
copies what it is handed and never draws randomness, so a row built inside
``recording`` is bit-identical to the same seeded row built without it.
Training never installs one; the web pipeline inspector does, one row at a time.

``STAGES`` is the stage list in synthesis order. A stage that did not act on a
row records nothing, so the taps of any row are an ordered subsequence of it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class StageSpec:
    """One synthesis stage: its id, its display group, and where its knob lives.

    ``block`` is the dotted attribute path, on the dataset, of the config block
    that gates the stage (None for stages that always run); ``prob_field`` is
    that block's firing probability, None when it has none.
    """

    id: str
    group: str
    block: Optional[str] = None
    prob_field: Optional[str] = "prob"


STAGES: tuple[StageSpec, ...] = (
    StageSpec("source.load", "source", None, None),
    StageSpec("row.plan", "source", None, None),
    StageSpec("foreground.channel", "room", "augmentation_reverb_args"),
    StageSpec("interferers.sample", "scene", "augmentation_speech_args"),
    StageSpec("interferers.gate", "scene", "augmentation_speech_args.overlap_control", None),
    StageSpec("interferers.mix", "scene", "augmentation_speech_args"),
    StageSpec("row.target_absent", "scene", "augmentation_target_absent_args"),
    StageSpec("echo.playback", "scene", "augmentation_speech_args.echo_playback"),
    StageSpec("level.peak_guard", "level", None, None),
    StageSpec("speed.perturb", "level", "augmentation_speed_args"),
    StageSpec("reverb.whole_mix", "room", "augmentation_reverb_args"),
    StageSpec("ambient.lead", "level", "augmentation_row_initial_ambient_args"),
    StageSpec("noise.recorded", "noise", "augmentation_noise_args"),
    StageSpec("noise.white", "noise", "augmentation_noise_args", "prob_white_noise"),
    StageSpec("noise.floor", "noise", "augmentation_noise_args.absolute_floor"),
    StageSpec("chain.src", "device", "augmentation_src_args"),
    StageSpec("chain.iir", "device", "augmentation_ir_response_args"),
    StageSpec("chain.hpf", "device", "augmentation_hpf_args"),
    StageSpec("chain.volume", "device", "augmentation_volume_args"),
    StageSpec("chain.compressor", "device", "augmentation_compressor_args"),
    StageSpec("chain.adc", "device", None, None),
    StageSpec("chain.codec", "device", "augmentation_codec_args"),
    StageSpec("chain.packet_loss", "device", "augmentation_packet_loss_args"),
    StageSpec("row.emit", "emit", None, None),
)

STAGE_INDEX: dict[str, int] = {spec.id: index for index, spec in enumerate(STAGES)}

#: Containers longer than this are reported by type rather than listed.
_MAX_LISTED = 16


@dataclass
class Tap:
    """The pair right after one stage, and what the stage drew."""

    stage: str
    noisy: np.ndarray
    target: np.ndarray
    params: dict[str, Any]
    signals: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RirRecord:
    """One ``apply_rir`` call: which channel, for which source, in which mode.

    ``position`` is the number of taps recorded before the call; the response
    belongs to the stage of ``taps[position]``, the stage it was applied in.
    ``impulse`` is the full response as served, None for a folder RIR (the
    augmentor keeps only its path).
    """

    position: int
    role: str
    mode: str
    rir_id: str
    sample_rate: int
    metadata: dict[str, Any]
    impulse: Optional[np.ndarray]


class SynthesisTrace:
    """Taps and impulse responses of the rows built while it is installed."""

    def __init__(self) -> None:
        self.taps: list[Tap] = []
        self.rirs: list[RirRecord] = []

    def tap(
        self,
        stage: str,
        noisy,
        target,
        *,
        signals: Optional[Mapping[str, Any]] = None,
        **params: Any,
    ) -> None:
        if stage not in STAGE_INDEX:
            raise ValueError(f"unknown synthesis stage {stage!r}")
        kept: dict[str, np.ndarray] = {}
        for name, value in (signals or {}).items():
            signal = _signal(value)
            if signal is not None:
                kept[name] = signal
        self.taps.append(
            Tap(
                stage=stage,
                noisy=_first_channel(noisy),
                target=_first_channel(target),
                params={name: summarize(value) for name, value in params.items()},
                signals=kept,
            )
        )

    def record_rir(self, augmentor, applied, arguments: Mapping[str, Any]) -> None:
        rir_id = str(applied.detail.rir_id)
        cached = getattr(augmentor, "simulated_rir", {}).get(rir_id)
        self.rirs.append(
            RirRecord(
                position=len(self.taps),
                role=str(arguments.get("source_role") or "source"),
                mode=str(arguments.get("rir_mode") or ""),
                rir_id=rir_id,
                sample_rate=int(cached["sample_rate"] if cached else arguments.get("sr") or 0),
                metadata={
                    str(key): summarize(value)
                    for key, value in (applied.detail.metadata or {}).items()
                },
                impulse=_first_channel(cached["impulse"]) if cached else None,
            )
        )

    def stage_ids(self) -> list[str]:
        return [tap.stage for tap in self.taps]


def summarize(value: Any) -> Any:
    """A JSON-shaped copy of a stage parameter.

    Scalars stay as they are (NaN included -- turning it into JSON null is the
    report's job); a mapping or dataclass keeps its scalar entries and names the
    rest by type; keys starting with ``_`` are internal and dropped.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _leaf(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        summary: dict[str, Any] = {"class": type(value).__name__}
        for item in dataclasses.fields(value):
            summary[item.name] = _leaf(getattr(value, item.name))
        return summary
    return _leaf(value)


def _leaf(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else f"<tensor {list(value.shape)}>"
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= _MAX_LISTED else f"<array {list(value.shape)}>"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        if len(items) <= _MAX_LISTED and all(
            item is None or isinstance(item, (bool, int, float, str, np.generic))
            for item in items
        ):
            return [_leaf(item) for item in items]
    return f"<{type(value).__name__}>"


def _first_channel(wav) -> np.ndarray:
    """A float32 copy of the first channel of a ``[..., L]`` signal."""
    tensor = torch.as_tensor(wav).detach()
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    return tensor.reshape(-1, tensor.shape[-1])[0].to(torch.float32).cpu().numpy().copy()


def _signal(value) -> Optional[np.ndarray]:
    """A companion signal: one waveform, or the sum of a list of them."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [_first_channel(item) for item in value if item is not None]
        if not parts:
            return None
        total = np.zeros(max(part.size for part in parts), dtype=np.float32)
        for part in parts:
            total[: part.size] += part
        return total
    return _first_channel(value)


@contextlib.contextmanager
def recording(dataset) -> Iterator[SynthesisTrace]:
    """Trace the rows ``dataset`` builds inside the block.

    Installs a trace on the dataset and wraps its augmentor's ``apply_rir`` so
    every impulse response a row is convolved with is kept with the taps. Both
    are undone on exit, error or not. Not reentrant, and it only sees rows
    built in this process -- not in DataLoader workers.
    """
    if getattr(dataset, "trace", None) is not None:
        raise RuntimeError("this dataset is already being traced")
    trace = SynthesisTrace()
    augmentor = dataset.augmentor
    shadowed = vars(augmentor).get("apply_rir")
    original = augmentor.apply_rir
    signature = inspect.signature(original)

    def traced_apply_rir(*args, **kwargs):
        applied = original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        trace.record_rir(augmentor, applied, bound.arguments)
        return applied

    augmentor.apply_rir = traced_apply_rir
    dataset.trace = trace
    try:
        yield trace
    finally:
        dataset.trace = None
        if shadowed is None:
            del augmentor.apply_rir
        else:
            augmentor.apply_rir = shadowed


def bind_target(trace: Optional[SynthesisTrace], target) -> Optional[Callable[..., None]]:
    """A tap for a stage that only sees the mixture (the noise stage): the pair
    it reports is that mixture against this fixed target. None without a trace."""
    if trace is None:
        return None

    def tap(stage: str, noisy, **params: Any) -> None:
        trace.tap(stage, noisy, target, **params)

    return tap
