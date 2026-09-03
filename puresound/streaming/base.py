"""What a streaming frame model is, minus the backbone it wraps.

`dpcrn` and `dparn` each turn an offline encoder/decoder into a model that
consumes one STFT frame and carries its own state. What genuinely differs
between them is the shape of that state and what one block does with it. What
does not differ -- and was copied line for line -- is the naming of the state
ports, the causal down-convolution step, the ONNX export and its round-trip
check, and the whole ORT runtime.

The split follows what a measurement said rather than what the two files look
like. Comparing them unit by unit with the backbone names normalised away: the
export path and the loader are byte-identical, `_down_step` is byte-identical,
the port-name properties differ only by DPCRN's lookahead ports, and `forward`
differs only in whether it flattens the state through a helper. Everything else
-- the block step, the config validators, the state dataclasses -- is genuinely
different, and pulling those up would mean inventing a shared abstraction over
things that are not the same. They stay where they are.

`_up_step` is here because the difference between the two copies turned out to
be a bug rather than a design. Both overlap-add a transpose conv whose kernel
spans two time taps, and the conv adds its bias to both of them, so the sum
counts it twice where PyTorch's offline `conv_transpose` counts it once. DPCRN
subtracted the extra copy; DPARN did not, and streamed 1.15e-01 relative away
from its own offline forward until it did. Once corrected the two were the same
code, so it lives here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puresound.system.postprocess import IDENTITY, Postprocessor
from puresound.utils import load_hparam


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


class StreamingFrameModelBase(nn.Module):
    """The parts of a frame model that do not depend on its backbone.

    A subclass supplies `n_down`, `n_up`, `n_blocks`, `initial_state`,
    `_state_to_tuple`, `state_from_tensors` and `forward_frame`; it may add
    ports beyond the four standard groups through `_extra_state_names`.

    The port names are a published interface, not an implementation detail:
    they are written into the export manifest and the ORT runtime feeds the
    session by name. Renaming one silently breaks every model already exported.
    """

    def _extra_state_names(self, prefix: str = "") -> list[str]:
        """Ports beyond down/up/h/c. DPCRN's lookahead variant carries skip
        caches, a noisy cache and a frame counter; DPARN carries none."""
        return []

    @property
    def extra_output_names(self) -> list[str]:
        """Graph outputs beyond the enhanced frame.

        A model with auxiliary heads emits their per-frame logits here. They are
        SIDE INFORMATION: the graph never applies them to the audio, so a runtime
        that ignores them produces byte-identical output to one that does not
        emit them at all. Whatever consumes them decides what they mean -- the
        same division of labour as `dry_blend`, which the runtime applies and the
        manifest merely declares.
        """
        return []

    @property
    def state_input_names(self) -> list[str]:
        return (
            [f"down_cache_{i}" for i in range(self.n_down)]
            + [f"up_cache_{i}" for i in range(self.n_up)]
            + [f"h_{i}" for i in range(self.n_blocks)]
            + [f"c_{i}" for i in range(self.n_blocks)]
            + self._extra_state_names("")
        )

    @property
    def state_output_names(self) -> list[str]:
        return (
            [f"next_down_cache_{i}" for i in range(self.n_down)]
            + [f"next_up_cache_{i}" for i in range(self.n_up)]
            + [f"next_h_{i}" for i in range(self.n_blocks)]
            + [f"next_c_{i}" for i in range(self.n_blocks)]
            + self._extra_state_names("next_")
        )

    def _state_to_tuple(self, state) -> tuple[torch.Tensor, ...]:
        """Flatten the state in the order `state_input_names` announces."""
        raise NotImplementedError

    def initial_state_tensors(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, ...]:
        return self._state_to_tuple(
            self.initial_state(batch_size=batch_size, device=device)
        )

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Flat tensors in, flat tensors out -- the shape ONNX can express."""
        noisy_frame, *state_tensors = inputs
        state = self.state_from_tensors(state_tensors)
        result = self.forward_frame(noisy_frame, state)
        # A subclass with auxiliary outputs returns them in the middle; one
        # without keeps the two-tuple it always returned.
        if len(result) == 3:
            enhanced, extras, next_state = result
        else:
            enhanced, next_state = result
            extras = ()
        if len(extras) != len(self.extra_output_names):
            raise RuntimeError(
                f"{type(self).__name__} returned {len(extras)} extra outputs but "
                f"announces {len(self.extra_output_names)}: "
                f"{self.extra_output_names}. The manifest names the ports a "
                "runtime feeds by name, so a mismatch mislabels them."
            )
        return tuple([enhanced, *extras] + list(self._state_to_tuple(next_state)))

    def _down_step(self, layer: nn.Sequential, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = layer[0].padding
        conv = layer[1]
        x_ctx = torch.cat([cache, x], dim=-1) if cache.shape[-1] else x
        x_ctx = F.pad(x_ctx, (0, 0, pad[2], pad[3]))
        y = conv(x_ctx)
        y = layer[2](y)
        y = layer[3](y)
        y = layer[4](y)
        next_cache = x[..., -cache.shape[-1] :] if cache.shape[-1] else cache
        return y, next_cache

    def _up_step(
        self, layer: nn.Sequential, x: torch.Tensor, pending: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv = layer[0]
        raw = conv(x)
        # The transpose conv adds its bias to BOTH time taps of each frame's
        # splat, so overlap-adding tap[0] with the previous frame's tap[1] would
        # count the bias twice. PyTorch's offline conv_transpose adds bias once
        # per output position -> subtract one bias copy from the overlap sum.
        completed = raw[..., :1] + pending
        if conv.bias is not None:
            completed = completed - conv.bias.view(1, -1, 1, 1)
        next_pending = raw[..., 1:2]
        if len(layer) > 1:
            completed = layer[1](completed)
            completed = layer[2](completed)
        return completed, next_pending


@dataclass(frozen=True)
class StreamingVariant:
    """What the shared load and export paths need to know about one backbone.

    Three values, because that is all the two copies actually differed by: the
    name that goes into the manifest, the config validator, and the frame model
    to build.
    """

    name: str
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    frame_model: type[StreamingFrameModelBase]


def load_streaming_model(
    variant: StreamingVariant,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> StreamingFrameModelBase:
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    config_path = Path(config_path)
    config = load_hparam(str(config_path))
    variant.validate(config)
    model_dict = load_recipe(config_path).model
    system_model = init_siso_model(model_dict)
    if checkpoint_path:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        system_model.reload_checkpoint(state_dict, load_loss_func=False)
    return variant.frame_model(system_model.eval())


def _postprocess_note(postprocess: Postprocessor, delay_frames: int, geometry) -> str:
    """Prose for the manifest, matching what the shipped artefacts already say.

    Latency is part of the contract: the runtime has to take the dry copy from
    `delay_frames` back, because that is the input the graph's output carries.
    """
    if not postprocess.enabled:
        return "no post-graph relief; the runtime applies the graph output as-is."
    hop = int(geometry["hop_length"])
    sample_rate = int(geometry["sample_rate"])
    latency_ms = 1000.0 * delay_frames * hop / sample_rate
    dry = postprocess.dry_blend
    return (
        f"out = {dry:g}*enhanced + {1 - dry:.3g}*input, with the input "
        f"latency-aligned to the enhanced stream (algorithmic latency "
        f"{delay_frames} frames / {latency_ms:g} ms). Bounds attenuation at any "
        f"point to {postprocess.suppression_ceiling_db:.0f} dB, which trades a "
        "little residual interferer for far fewer deletions on capture chains "
        "the model was not trained on."
    )


def export_streaming_onnx(
    variant: StreamingVariant,
    config_path: str | Path,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 17,
    postprocess: Postprocessor = IDENTITY,
) -> dict[str, Any]:
    """Trace the frame model to ONNX and write the manifest a runtime reads.

    ``postprocess`` is recorded, not applied: the graph contains the model and
    nothing after it, so a runtime that executes the graph alone is running a
    different system than a benchmark at ``dry_blend 0.9``. Putting the setting
    in the manifest is what lets the runtime reproduce the benchmarked
    configuration instead of a shell flag having to be remembered twice.
    """
    import onnxruntime

    frame_model = load_streaming_model(variant, config_path, checkpoint_path)
    frame_model.eval()
    onnx_path = Path(onnx_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else onnx_path.with_suffix(".json")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    geometry = variant.validate(load_hparam(str(config_path)))

    noisy_frame = torch.randn(1, int(geometry["freq_bins"]), 2)
    state = frame_model.initial_state_tensors(batch_size=1)

    input_names = ["noisy_frame"] + frame_model.state_input_names
    output_names = (["enhanced_frame"] + list(frame_model.extra_output_names)
                    + frame_model.state_output_names)
    dynamic_axes = {
        "noisy_frame": {0: "batch_size"},
        "enhanced_frame": {0: "batch_size"},
    }
    for name in frame_model.state_input_names + frame_model.state_output_names:
        if name.startswith(("h_", "c_", "next_h_", "next_c_")):
            dynamic_axes[name] = {1: "batch_freq"}
        else:
            dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        frame_model,
        (noisy_frame, *state),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    providers = [provider for provider in providers if provider in available]
    session = onnxruntime.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
    ort_inputs = {"noisy_frame": noisy_frame.numpy()}
    for name, tensor in zip(frame_model.state_input_names, state):
        ort_inputs[name] = tensor.numpy()
    ort_out = session.run(None, ort_inputs)
    with torch.no_grad():
        torch_out = frame_model(noisy_frame, *state)
    n_checked = 1 + len(frame_model.extra_output_names)
    for i, name in enumerate(output_names[:n_checked]):
        if not np.allclose(torch_out[i].numpy(), ort_out[i], rtol=1e-4, atol=1e-4):
            raise AssertionError(
                f"exported ONNX output {name!r} does not match PyTorch"
            )

    manifest = {
        "model_type": f"{variant.name}_streaming_frame",
        "processor": "stft_frame_ort",
        "created_at": int(time.time()),
        "onnx_path": str(onnx_path),
        "sample_rate": int(geometry["sample_rate"]),
        "fft_length": int(geometry["fft_length"]),
        "win_length": int(geometry["win_length"]),
        "hop_length": int(geometry["hop_length"]),
        "freq_bins": int(geometry["freq_bins"]),
        "feature_bins": int(geometry["feature_bins"]),
        "streaming_delay_frames": frame_model.streaming_delay,
        "input_names": input_names,
        "output_names": output_names,
        "state_input_names": frame_model.state_input_names,
        "state_output_names": frame_model.state_output_names,
        # Side-information ports. Empty for a model with no auxiliary heads; a
        # runtime that ignores them gets byte-identical audio either way.
        "extra_output_names": list(frame_model.extra_output_names),
        "state_shapes": {
            name: tensor_shape(tensor)
            for name, tensor in zip(frame_model.state_input_names, state)
        },
        "providers": providers or ["CPUExecutionProvider"],
        # Applied by the runtime after the graph, not baked into it. Carries
        # `suppression_ceiling_db` too, because `dry_blend` bounds how deep the
        # deployed system can attenuate and that is worth reading off the
        # artefact rather than deriving it again.
        Postprocessor.MANIFEST_KEY: {
            **postprocess.as_manifest(),
            "note": _postprocess_note(postprocess, frame_model.streaming_delay, geometry),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class StreamingOrt:
    def __init__(
        self,
        onnx_path: str | Path,
        manifest_path: str | Path | None = None,
        provider: str = "auto",
        collect_extras: bool = False,
        postprocess_overrides: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ):
        self.onnx_path = Path(onnx_path)
        self.manifest_path = Path(manifest_path) if manifest_path else self.onnx_path.with_suffix(".json")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if session is None:
            import onnxruntime

            providers = self._resolve_providers(
                provider, onnxruntime.get_available_providers()
            )
            session = onnxruntime.InferenceSession(
                str(self.onnx_path), providers=providers
            )
        self.session = session
        self.providers = self.session.get_providers()
        self.sample_rate = int(self.manifest["sample_rate"])
        self.fft_length = int(self.manifest["fft_length"])
        self.win_length = int(self.manifest["win_length"])
        self.hop_length = int(self.manifest["hop_length"])
        self.freq_bins = int(self.manifest["freq_bins"])
        self.window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)
        # Side-information history. OFF by default: a long-running stream would
        # grow it without bound, and the common deployment does not read it.
        # `drain_extras()` returns and clears, so a caller that drains stays
        # bounded whatever the stream length.
        self.collect_extras = bool(collect_extras)
        self.extra_names = list(self.manifest.get("extra_output_names", []))
        # Post-graph relief, from the manifest. `export_streaming_onnx` records
        # it because the traced graph stops at the model, so a runtime that only
        # executes the graph is not the system the benchmarks measured. The SDK's
        # portable copy of this loop reads the same key -- see
        # `test_sdk_postprocess.py`, which pins the two against each other.
        # An absent section means no relief -- the documented convention, and the
        # same thing `Postprocessor()` defaults to. `dpcrn_v6` is the one shipped
        # export without it.
        postprocess = dict(self.manifest.get(Postprocessor.MANIFEST_KEY) or {})
        if postprocess_overrides:
            unknown = set(postprocess_overrides) - {"dry_blend", "spec_floor"}
            if unknown:
                raise ValueError(
                    "unsupported postprocess override(s): "
                    + ", ".join(sorted(unknown))
                )
            # Request values replace sidecar defaults.  They are applied here,
            # before the streaming loop starts, so a dry blend is never
            # post-processed a second time by a caller.
            postprocess.update(dict(postprocess_overrides))
            # The sidecar prose describes its original defaults; do not expose
            # stale text after a request replaces them.
            postprocess.pop("note", None)
        self.postprocess = dict(postprocess)
        spec_floor = float(postprocess.get("spec_floor", 0.0))
        if spec_floor > 0.0:
            raise ValueError(
                f"manifest requests spec_floor={spec_floor}, which this runtime "
                "does not implement. Re-export with spec_floor=0.0 or add it here."
            )
        self.dry_blend = float(postprocess.get("dry_blend", 1.0))
        if not 0.0 < self.dry_blend <= 1.0:
            raise ValueError(
                f"manifest dry_blend must be in (0, 1], got {self.dry_blend}"
            )
        # The overlap-add is index-aligned, so this is purely the graph's own
        # look-ahead latency: its output at a given index carries the input from
        # this many samples earlier, and the blend has to match or it mixes in a
        # slice of the mixture 30 ms away from the speech it is relieving.
        self.dry_delay = (
            int(self.manifest.get("streaming_delay_frames", 0)) * self.hop_length
        )
        self.reset()

    @staticmethod
    def _resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
        provider = provider.lower()
        if provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if provider == "auto":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        raise ValueError("provider must be one of: auto, cpu, cuda")

    def reset(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("StreamingOrt currently supports batch_size=1 for waveform streaming")
        self.state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.manifest["state_shapes"].items()
        }
        self.extras = {}
        self.extra_history = {name: [] for name in self.extra_names}
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(0, dtype=np.float32)
        self.ola_norm = np.zeros(0, dtype=np.float32)
        self.dry_history = np.zeros(0, dtype=np.float32)
        self.dry_history_start = 0
        self.emitted = 0

    def _remember_dry(self, samples: np.ndarray) -> None:
        if self.dry_blend >= 1.0:
            return
        self.dry_history = np.concatenate([self.dry_history, samples])

    def _blend_dry(self, enhanced: np.ndarray) -> np.ndarray:
        """Mix the untouched input back in, aligned to what the graph enhanced.

        Output samples with no corresponding input yet -- the first
        ``streaming_delay_frames`` worth, which is warm-up -- pass through
        unblended, because there is nothing to blend them with.
        """
        if self.dry_blend >= 1.0 or enhanced.size == 0:
            return enhanced
        start = self.emitted - self.dry_delay
        self.emitted += enhanced.size
        out = enhanced.astype(np.float32, copy=True)
        history_end = self.dry_history_start + self.dry_history.size
        lo = max(start, self.dry_history_start)
        hi = min(start + enhanced.size, history_end)
        if hi > lo:
            span = slice(lo - start, hi - start)
            reference = self.dry_history[
                lo - self.dry_history_start : hi - self.dry_history_start
            ]
            out[span] = (
                self.dry_blend * enhanced[span] + (1.0 - self.dry_blend) * reference
            )
        np.clip(out, -1.0, 1.0, out=out)
        keep_from = max(0, self.emitted - self.dry_delay - self.dry_history_start)
        if keep_from > 0:
            self.dry_history = self.dry_history[keep_from:]
            self.dry_history_start += keep_from
        return out

    def run_frame(self, noisy_frame: np.ndarray) -> np.ndarray:
        """One frame. Also stashes any side-information outputs on `self.extras`.

        The state outputs do NOT start at index 1: a graph with auxiliary heads
        puts their logits between the enhanced frame and the state, so the offset
        has to come from the manifest. Assuming 1 fed the first state port a
        rank-4 conv cache and ORT rejected it -- which is the good failure; the
        bad one is a state layout that happens to typecheck.
        """
        noisy_frame = np.asarray(noisy_frame, dtype=np.float32)
        if noisy_frame.shape != (1, self.freq_bins, 2):
            raise ValueError(f"noisy_frame must have shape (1, {self.freq_bins}, 2)")
        ort_inputs = {"noisy_frame": noisy_frame}
        ort_inputs.update(self.state)
        outputs = self.session.run(self.manifest["output_names"], ort_inputs)
        enhanced = outputs[0]
        extra_names = list(self.manifest.get("extra_output_names", []))
        first_state = 1 + len(extra_names)
        self.extras = dict(zip(extra_names, outputs[1:first_state]))
        if self.collect_extras:
            for name, value in self.extras.items():
                self.extra_history.setdefault(name, []).append(
                    float(np.asarray(value).reshape(-1)[0])
                )
        state_names = self.manifest["state_input_names"]
        state_values = outputs[first_state:]
        if len(state_values) != len(state_names):
            raise RuntimeError(
                f"graph returned {len(state_values)} state tensors for "
                f"{len(state_names)} ports; the manifest's output layout and the "
                "graph disagree"
            )
        for name, value in zip(state_names, state_values):
            self.state[name] = value
        return enhanced

    def drain_extras(self) -> dict[str, np.ndarray]:
        """The side-information collected so far, per port, then cleared.

        One value per processed frame, in order. These LEAD the emitted audio by
        `streaming_delay_frames`: they describe the bottleneck frame they were
        computed from, which the output has not reached yet. A caller aligning
        them to samples has to drop that many frames from the front.
        """
        if not self.collect_extras:
            raise RuntimeError(
                "construct StreamingOrt(collect_extras=True) to collect side "
                "information; it is off by default so a long stream cannot grow "
                "an unbounded history"
            )
        out = {name: np.asarray(vals, dtype=np.float32)
               for name, vals in self.extra_history.items()}
        self.extra_history = {name: [] for name in self.extra_names}
        return out

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window, n=self.fft_length).astype(np.complex64)
        noisy_frame = np.stack([spec.real, spec.imag], axis=-1).reshape(1, self.freq_bins, 2).astype(np.float32)
        enhanced = self.run_frame(noisy_frame)[0]
        enhanced_complex = enhanced[:, 0] + 1j * enhanced[:, 1]
        wav = np.fft.irfft(enhanced_complex, n=self.fft_length).astype(np.float32)[: self.win_length]
        return wav * self.window

    def _add_ola_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.ola.size < self.win_length:
            pad = self.win_length - self.ola.size
            self.ola = np.pad(self.ola, (0, pad))
            self.ola_norm = np.pad(self.ola_norm, (0, pad))
        self.ola[: self.win_length] += frame
        self.ola_norm[: self.win_length] += self.window * self.window
        emit = self.ola[: self.hop_length].copy()
        norm = self.ola_norm[: self.hop_length].copy()
        emit = emit / np.maximum(norm, 1e-8)
        self.ola = self.ola[self.hop_length :]
        self.ola_norm = self.ola_norm[self.hop_length :]
        return emit.astype(np.float32)

    @staticmethod
    def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("streaming inference cancelled")

    def process_samples(
        self,
        samples: np.ndarray,
        *,
        frame_callback: Callable[[], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._remember_dry(samples)
        self.input_buffer = np.concatenate([self.input_buffer, samples])
        chunks = []
        while self.input_buffer.shape[0] >= self.win_length:
            self._check_cancelled(cancel_check)
            frame = self.input_buffer[: self.win_length]
            self.input_buffer = self.input_buffer[self.hop_length :]
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
            if frame_callback is not None:
                frame_callback()
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def flush(
        self,
        *,
        frame_callback: Callable[[], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        chunks = []
        while self.input_buffer.size > 0:
            self._check_cancelled(cancel_check)
            frame = np.zeros(self.win_length, dtype=np.float32)
            n = min(self.input_buffer.size, self.win_length)
            frame[:n] = self.input_buffer[:n]
            self.input_buffer = self.input_buffer[min(self.hop_length, self.input_buffer.size) :]
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
            if frame_callback is not None:
                frame_callback()
        self._check_cancelled(cancel_check)
        if self.ola.size:
            tail = self.ola / np.maximum(self.ola_norm, 1e-8)
            chunks.append(self._blend_dry(tail.astype(np.float32)))
            self.ola = np.zeros(0, dtype=np.float32)
            self.ola_norm = np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
