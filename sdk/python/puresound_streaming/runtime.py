import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np


_PROVIDER_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    "coreml": "coreml",
    "mps": "coreml",
}
_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_COREML_PROVIDER = "CoreMLExecutionProvider"


#: Knobs of `puresound.system.onset_guard.OnsetGuard`, spelled again because the
#: SDK must not import puresound. `test_sdk_postprocess.py` pins the two copies.
_ONSET_GUARD_KNOBS = (
    "t_arm_s",
    "t_forget_s",
    "tau_up_s",
    "tau_dn_s",
    "margin_db",
    "floor_win_s",
    "floor_rise_db_per_s",
    "init_s",
    "hangover_s",
    "min_run_s",
    "snap",
)


@dataclass
class OnsetGuardState:
    """Per-stream detector state. Mutated in place by `OnsetGuard.step`."""

    hop: int
    fps: float
    n_floor: int
    n_init: int
    n_hold: int
    n_minr: int
    n_arm: int
    n_forget: Optional[int]
    rise_per_frame: float
    a_up: float
    a_dn: float
    prev_hop: Optional[np.ndarray] = None
    frame: int = 0
    init_buf: List[float] = field(default_factory=list)
    win: deque = field(default_factory=deque)
    floor_cur: float = 0.0
    last_raw: int = -(1 << 30)
    held_run: int = 0
    run: int = 0
    sil: int = 0
    confirmed: bool = False
    gain: float = 1.0
    started: bool = False


@dataclass(frozen=True)
class OnsetGuard:
    """numpy port of `puresound.system.onset_guard.OnsetGuard`'s streaming face.

    Inference-only onset protection: the output is the input, bit for bit, until
    a talker has been heard for ``t_arm_s`` of sustained speech; then the model's
    output is handed over, and ``t_forget_s`` of floor protects the next onset
    again. The exported graph does not contain it, so the runtime applies it --
    the same division of labour as ``dry_blend``.

    This is a DUPLICATE, deliberately: the SDK must not import puresound (see
    `test_portable_streaming_sdk`), so `_advance` and `step` are transcribed with
    the same names and the same order of operations as the module's, and
    `test_sdk_postprocess.py` pins the two runtimes' output bit-identical. The
    constructor coherence checks are the one thing NOT carried over -- a manifest
    is written by a guard that already passed them -- but the priming shortcut's
    exactness check is, because that one depends on the frame rate the runtime
    picks rather than on the knobs alone.
    """

    t_arm_s: float = 1.0
    t_forget_s: float = 5.0
    tau_up_s: float = 0.05
    tau_dn_s: float = 2.0
    margin_db: float = 8.0
    floor_win_s: float = 2.0
    floor_rise_db_per_s: float = 3.0
    init_s: float = 0.2
    hangover_s: float = 0.2
    min_run_s: float = 0.10
    snap: float = 1e-3

    @classmethod
    def from_manifest(cls, spec: dict) -> "OnsetGuard":
        return cls(**{k: float(v) for k, v in spec.items() if k in _ONSET_GUARD_KNOBS})

    def streaming_state(self, *, hop: int = 160, sr: float = 16000.0) -> OnsetGuardState:
        fps = float(sr) / float(hop)
        n_arm = max(1, int(round(self.t_arm_s * fps)))
        n_forget = (
            None
            if not math.isfinite(self.t_forget_s)
            else max(1, int(round(self.t_forget_s * fps)))
        )
        # int(), not round(): this is the reference implementation's arithmetic.
        n_hold = int(self.hangover_s * fps) + 1
        n_minr = max(1, int(self.min_run_s * fps))
        n_init = max(1, int(round(self.init_s * fps)))
        if n_init + 2 > n_minr + n_arm:
            raise ValueError(
                f"at {fps:g} fps the initialisation window is {n_init} frames but "
                f"the guard could arm by frame {n_minr + n_arm - 2}; the streaming "
                "form's priming shortcut is only exact when it cannot."
            )
        return OnsetGuardState(
            hop=int(hop),
            fps=fps,
            n_floor=max(1, int(round(self.floor_win_s * fps))),
            n_init=n_init,
            n_hold=n_hold,
            n_minr=n_minr,
            n_arm=n_arm,
            n_forget=n_forget,
            rise_per_frame=self.floor_rise_db_per_s / fps,
            a_up=1.0 - math.exp(-1.0 / (self.tau_up_s * fps)),
            a_dn=1.0 - math.exp(-1.0 / (self.tau_dn_s * fps)),
        )

    def _advance(self, st: OnsetGuardState, i: int, e_db: float) -> float:
        """One frame of floor -> activity -> arming -> integrator. Returns the gain."""
        # causal floor: running minimum over the last n_floor frames, allowed to
        # climb at most rise_per_frame and to fall instantly.
        win = st.win
        while win and win[-1][1] >= e_db:
            win.pop()
        win.append((i, e_db))
        if win[0][0] <= i - st.n_floor:
            win.popleft()
        c = win[0][1]
        cur = st.floor_cur
        st.floor_cur = c if c < cur else min(c, cur + st.rise_per_frame)

        # activity, streaming form: hangover = hold, min-run = confirmation delay
        if e_db > st.floor_cur + self.margin_db:
            st.last_raw = i
        held = (i - st.last_raw) <= st.n_hold - 1
        st.held_run = st.held_run + 1 if held else 0
        act = st.held_run >= st.n_minr

        # arming / re-arming
        if act:
            st.run += 1
            st.sil = 0
            if not st.confirmed and st.run >= st.n_arm:
                st.confirmed = True
        else:
            st.run = 0
            st.sil += 1
            if st.confirmed and st.n_forget is not None and st.sil >= st.n_forget:
                st.confirmed = False
        target = 0.0 if st.confirmed else 1.0

        # first-order integrator toward the target, snapping to the endpoints
        if not st.started:
            st.gain = target
            st.started = True
        else:
            g = st.gain
            if abs(g - target) <= self.snap:
                st.gain = target
            else:
                al = st.a_up if target > g else st.a_dn
                g = target + (g - target) * (1.0 - al)
                st.gain = target if abs(g - target) <= self.snap else g
        st.frame = i + 1
        return st.gain

    def step(self, state: OnsetGuardState, dry_frame: np.ndarray):
        """Consume one hop of the DRY stream, return ``(gain, state)``.

        Call ``c`` (1-based) returns the gain for frame ``c - 2``: frame energy
        spans 20 ms, so a frame is only complete once the following hop has
        arrived. The first call primes that window and returns 1.0 (dry). The
        state is mutated in place and returned for convenience, so the runtime's
        feed loop reads the same as the in-repo one.
        """
        f = np.asarray(dry_frame, dtype=np.float64).reshape(-1)
        if f.size != state.hop:
            raise ValueError(
                f"dry_frame must be exactly one hop of {state.hop} samples, got "
                f"{f.size}; pad the final frame rather than feeding a short one"
            )
        if state.prev_hop is None:
            state.prev_hop = f
            return state.gain, state
        pair = np.concatenate([state.prev_hop, f])
        e_db = float(10.0 * np.log10(float((pair * pair).mean()) + 1e-12))
        state.prev_hop = f

        i = state.frame
        if i < state.n_init - 1:
            state.init_buf.append(e_db)
            state.frame = i + 1
            return 1.0, state
        if i == state.n_init - 1:
            state.init_buf.append(e_db)
            buf = state.init_buf
            state.floor_cur = float(min(buf))
            g = 1.0
            for j, v in enumerate(buf):
                g = self._advance(state, j, v)
            state.init_buf = []
            return g, state
        return self._advance(state, i, e_db), state


class InferenceSessionLike(Protocol):
    def get_providers(self) -> list[str]:
        ...

    def run(self, output_names, inputs):
        ...


class StreamingProcessor(Protocol):
    def reset(self, batch_size: int = 1) -> None:
        ...

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        ...

    def flush(self) -> np.ndarray:
        ...

    def run_frame(self, frame: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StreamingRuntimeConfig:
    model_type: str
    processor: str
    sample_rate: int
    fft_length: int
    win_length: int
    hop_length: int
    freq_bins: int
    state_input_names: list[str]
    state_output_names: list[str]
    output_names: list[str]
    state_shapes: dict[str, list[int]]
    #: Frames the graph's output lags its input by. Non-zero for a look-ahead
    #: export (every current DPCRN export is 3), and the dry blend has to
    #: compensate for it or it mixes in the wrong slice of the input.
    streaming_delay_frames: int = 0
    #: Post-graph over-suppression relief, from the manifest. The graph does not
    #: contain it, so the runtime applies it -- otherwise the deployed system is
    #: not the one the benchmarks measured.
    dry_blend: float = 1.0
    spec_floor: float = 0.0
    #: Post-graph onset protection, from the manifest. ``None`` -- an absent
    #: section -- means no guard, the same convention the blend follows.
    onset_guard: Optional[dict] = None

    @classmethod
    def from_manifest(cls, manifest: dict) -> "StreamingRuntimeConfig":
        # `recommended_inference` is the key the exports have always carried;
        # absent means no relief, which is also what `Postprocessor()` defaults
        # to. Spelled here rather than imported: the SDK must not import
        # puresound, so the name is duplicated on purpose and
        # `test_sdk_postprocess.py` pins the two copies together.
        postprocess = manifest.get("recommended_inference") or {}
        spec_floor = float(postprocess.get("spec_floor", 0.0))
        if spec_floor > 0.0:
            # It needs the mixture spectrum at the same frame and a magnitude to
            # scale, which this processor could supply -- but no shipped export
            # sets it, so refusing beats a silent no-op.
            raise ValueError(
                f"manifest requests spec_floor={spec_floor}, which this runtime "
                "does not implement. Re-export with spec_floor=0.0 or add it here."
            )
        dry_blend = float(postprocess.get("dry_blend", 1.0))
        if not 0.0 < dry_blend <= 1.0:
            raise ValueError(f"manifest dry_blend must be in (0, 1], got {dry_blend}")
        # Onset protection travels under its own key for the same reason: the
        # graph stops at the model. Absent means no guard. Spelled as a literal
        # again -- `OnsetGuard.MANIFEST_KEY` is the other copy, and
        # `test_sdk_postprocess.py` pins them together.
        onset_guard = manifest.get("onset_guard") or None
        return cls(
            streaming_delay_frames=int(manifest.get("streaming_delay_frames", 0)),
            dry_blend=dry_blend,
            spec_floor=spec_floor,
            onset_guard=dict(onset_guard) if onset_guard else None,
            model_type=str(manifest.get("model_type", "unknown")),
            processor=str(manifest.get("processor", "stft_frame_ort")),
            sample_rate=int(manifest["sample_rate"]),
            fft_length=int(manifest["fft_length"]),
            win_length=int(manifest["win_length"]),
            hop_length=int(manifest["hop_length"]),
            freq_bins=int(manifest["freq_bins"]),
            state_input_names=list(manifest["state_input_names"]),
            state_output_names=list(manifest["state_output_names"]),
            output_names=list(manifest["output_names"]),
            state_shapes={
                name: [int(dim) for dim in shape]
                for name, shape in manifest["state_shapes"].items()
            },
        )


class StftFrameOrtProcessor:
    """STFT-frame ONNX processor used by current DPARN streaming exports."""

    def __init__(self, config: StreamingRuntimeConfig, session: InferenceSessionLike):
        self.config = config
        self.session = session
        self.sample_rate = config.sample_rate
        self.fft_length = config.fft_length
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        self.freq_bins = config.freq_bins
        self.window = np.hanning(self.win_length + 1)[:-1].astype(np.float32)
        self.dry_blend = config.dry_blend
        # How far back in the input the sample aligned with the next emitted
        # output sample sits. The overlap-add itself is index-aligned -- an
        # identity graph reconstructs the input at the same indices -- so this is
        # purely the graph's own look-ahead latency.
        self.dry_delay = config.streaming_delay_frames * self.hop_length
        self.onset_guard = (
            OnsetGuard.from_manifest(config.onset_guard)
            if config.onset_guard
            else None
        )
        # The guard's frame t is only decided once dry hop t+1 has arrived (its
        # frame energy spans two hops), and output hop m carries input frame
        # ``m - streaming_delay_frames``. When output hop m is emitted the
        # runtime has necessarily received ``m*hop + win_length`` samples, i.e.
        # ``m + win_length//hop`` whole hops, so every frame up to
        # ``m + win_length//hop - 2`` is already decided -- the gain applied is
        # the exact one, with no latency added, as long as this is >= 0. A window
        # shorter than two hops on a zero-latency graph is the only case that
        # cannot be served, and it is refused rather than served a wrong gain.
        self.onset_guard_lookahead_hops = (
            config.streaming_delay_frames + self.win_length // self.hop_length - 2
        )
        if self.onset_guard is not None and self.onset_guard_lookahead_hops < 0:
            raise ValueError(
                f"onset guard needs one hop of input look-ahead: at "
                f"streaming_delay_frames={config.streaming_delay_frames}, "
                f"win_length={self.win_length} and hop_length={self.hop_length} "
                f"the runtime is {-self.onset_guard_lookahead_hops} hop(s) short "
                "when an output hop is emitted, so the gain would be unaligned."
            )
        self.reset()

    def reset(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("waveform streaming currently supports batch_size=1")
        self.state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.config.state_shapes.items()
        }
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(0, dtype=np.float32)
        self.ola_norm = np.zeros(0, dtype=np.float32)
        # Raw input kept for the dry blend, with the absolute index its first
        # sample has. Trimmed as it is consumed so a long stream does not grow it.
        self.dry_history = np.zeros(0, dtype=np.float32)
        self.dry_history_start = 0
        self.emitted = 0
        # Onset guard: one fresh detector state per stream, so a second stream
        # starts protected again rather than inheriting the last one's anchor.
        self.guard_state = (
            self.onset_guard.streaming_state(
                hop=self.hop_length, sr=self.sample_rate
            )
            if self.onset_guard is not None
            else None
        )
        self.guard_buffer = np.zeros(0, dtype=np.float32)
        self.guard_gains: List[float] = []
        self.guard_gain_start = 0
        self.guard_hops_fed = 0
        self.guard_flushed = False

    def run_frame(self, noisy_frame: np.ndarray) -> np.ndarray:
        noisy_frame = np.asarray(noisy_frame, dtype=np.float32)
        expected = (1, self.freq_bins, 2)
        if noisy_frame.shape != expected:
            raise ValueError(f"noisy_frame must have shape {expected}")
        ort_inputs = {"noisy_frame": noisy_frame}
        ort_inputs.update(self.state)
        outputs = self.session.run(self.config.output_names, ort_inputs)
        enhanced = outputs[0]
        for name, value in zip(self.config.state_input_names, outputs[1:]):
            self.state[name] = value
        return enhanced

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._remember_dry(samples)
        self.input_buffer = np.concatenate([self.input_buffer, samples])
        chunks = []
        while self.input_buffer.shape[0] >= self.win_length:
            frame = self.input_buffer[: self.win_length]
            self.input_buffer = self.input_buffer[self.hop_length :]
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        # No more input is coming, so the guard's last frames can be decided now
        # (they need one hop past the end, which the offline framing zero-pads).
        self._flush_guard()
        chunks = []
        while self.input_buffer.size > 0:
            frame = np.zeros(self.win_length, dtype=np.float32)
            n = min(self.input_buffer.size, self.win_length)
            frame[:n] = self.input_buffer[:n]
            self.input_buffer = self.input_buffer[
                min(self.hop_length, self.input_buffer.size) :
            ]
            chunks.append(self._blend_dry(self._add_ola_frame(self._process_frame(frame))))
        if self.ola.size:
            tail = self.ola / np.maximum(self.ola_norm, 1e-8)
            chunks.append(self._blend_dry(tail.astype(np.float32)))
            self.ola = np.zeros(0, dtype=np.float32)
            self.ola_norm = np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.window, n=self.fft_length).astype(np.complex64)
        noisy_frame = np.stack([spec.real, spec.imag], axis=-1)
        noisy_frame = noisy_frame.reshape(1, self.freq_bins, 2).astype(np.float32)
        enhanced = self.run_frame(noisy_frame)[0]
        enhanced_complex = enhanced[:, 0] + 1j * enhanced[:, 1]
        wav = np.fft.irfft(enhanced_complex, n=self.fft_length).astype(np.float32)
        return wav[: self.win_length] * self.window

    def _remember_dry(self, samples: np.ndarray) -> None:
        """Keep the input the post-graph stages read back, and feed the guard.

        Kept whenever EITHER stage needs it: the onset guard consumes the dry
        stream even at ``dry_blend >= 1.0``, where the blend itself is a no-op.
        """
        if self.dry_blend >= 1.0 and self.onset_guard is None:
            return
        self.dry_history = np.concatenate([self.dry_history, samples])
        if self.onset_guard is not None:
            self._advance_guard(samples)

    def _advance_guard(self, samples: np.ndarray) -> None:
        """Feed whole hops of the DRY stream to the guard, in input order.

        `OnsetGuard.step` returns frame ``t`` on the call that feeds hop
        ``t + 1`` -- frame energy spans two hops -- so the first call's value
        belongs to no frame and is dropped. What is left is one gain per input
        frame, in frame order, which `_apply_guard` indexes by input sample.
        """
        hop = self.hop_length
        self.guard_buffer = np.concatenate([self.guard_buffer, samples])
        while self.guard_buffer.size >= hop:
            chunk = self.guard_buffer[:hop]
            self.guard_buffer = self.guard_buffer[hop:]
            gain, self.guard_state = self.onset_guard.step(self.guard_state, chunk)
            if self.guard_hops_fed:
                self.guard_gains.append(float(gain))
            self.guard_hops_fed += 1

    def _flush_guard(self) -> None:
        """Zero-pad the tail so the last frames get the gain offline computes.

        The offline face frames ``ceil(T/hop)`` frames over a signal zero-padded
        to cover the last frame's 20 ms window; the streaming form reproduces
        that exactly by feeding the final partial hop zero-padded plus one whole
        hop of zeros. Idempotent; the stream is over once it has run, so start
        another with `reset()` rather than by feeding more samples.
        """
        if self.onset_guard is None or self.guard_flushed:
            return
        self.guard_flushed = True
        hop = self.hop_length
        pad = (-self.guard_buffer.size) % hop
        tail = np.concatenate(
            [self.guard_buffer, np.zeros(pad + hop, dtype=np.float32)]
        )
        self.guard_buffer = np.zeros(0, dtype=np.float32)
        self._advance_guard(tail)

    def _apply_guard(
        self,
        out: np.ndarray,
        reference: np.ndarray,
        lo: int,
        hi: int,
        start: int,
    ) -> None:
        """``out = g*input + (1 - g)*out``, with the gain of the frame each INPUT
        sample sits in.

        ``lo``/``hi`` are absolute input sample indices and ``start`` the input
        index that ``out[0]`` carries, so this is the same alignment the blend
        uses. Applying a hop's gain to the wrong hop of audio is the exact
        mistake `_blend_dry`'s docstring warns about for the dry reference, and
        the frame indices here are integers by construction: ``dry_delay`` is a
        whole number of hops and the overlap-add emits one hop per frame.
        """
        hop = self.hop_length
        for frame in range(lo // hop, (hi - 1) // hop + 1):
            i = frame - self.guard_gain_start
            if not 0 <= i < len(self.guard_gains):
                raise RuntimeError(
                    f"onset guard has no gain for input frame {frame}; a frame is "
                    "decided one hop after it starts, which the runtime's own "
                    "look-ahead covers, so reaching here is a bug."
                )
            g = np.float32(self.guard_gains[i])
            a = max(lo, frame * hop)
            b = min(hi, (frame + 1) * hop)
            span = slice(a - start, b - start)
            out[span] = g * reference[a - lo : b - lo] + (1.0 - g) * out[span]

    def _forget_guard_gains(self) -> None:
        """Drop the gains no future emit can ask for, so a long stream stays flat."""
        if self.onset_guard is None:
            return
        oldest = (self.emitted - self.dry_delay) // self.hop_length
        drop = max(0, min(oldest - self.guard_gain_start, len(self.guard_gains)))
        if drop:
            self.guard_gains = self.guard_gains[drop:]
            self.guard_gain_start += drop

    def _blend_dry(self, enhanced: np.ndarray) -> np.ndarray:
        """Mix the untouched input back in, aligned to what the graph enhanced.

        ``dry_blend * enhanced + (1 - dry_blend) * input``, the same expression
        the offline module applies -- but the reference has to come from
        ``streaming_delay_frames`` back, because that is the input the graph's
        output at this index actually carries. Blending index-for-index would mix
        in a slice of the mixture 30 ms away from the speech it is relieving.

        The onset guard then runs on the same aligned span, in that order --
        blend first, guard last -- because the guard has to be able to restore
        the whole input rather than ``dry_blend`` of it.

        Output samples with no corresponding input yet -- the first
        ``streaming_delay_frames`` worth, which is warm-up -- pass through
        unblended: there is nothing to blend them with.
        """
        if enhanced.size == 0 or (self.dry_blend >= 1.0 and self.onset_guard is None):
            return enhanced
        start = self.emitted - self.dry_delay
        self.emitted += enhanced.size
        out = enhanced.astype(np.float32, copy=True)

        # Absolute-index overlap between what we are emitting and what we still
        # hold. Sliced rather than looped: this runs per hop on the audio thread.
        history_end = self.dry_history_start + self.dry_history.size
        lo = max(start, self.dry_history_start)
        hi = min(start + enhanced.size, history_end)
        if hi > lo:
            span = slice(lo - start, hi - start)
            reference = self.dry_history[lo - self.dry_history_start : hi - self.dry_history_start]
            if self.dry_blend < 1.0:
                # Clamped HERE and again below: the offline module clamps the
                # blend's own result and then clamps the guard's, so the guard
                # reads the first clamp's output. Idempotent when no guard
                # follows -- the final clip covers the same samples.
                out[span] = np.clip(
                    self.dry_blend * enhanced[span]
                    + (1.0 - self.dry_blend) * reference,
                    -1.0,
                    1.0,
                )
            if self.onset_guard is not None:
                self._apply_guard(out, reference, lo, hi, start)
        np.clip(out, -1.0, 1.0, out=out)
        # Everything before the next emit's reference is dead weight.
        keep_from = max(0, self.emitted - self.dry_delay - self.dry_history_start)
        if keep_from > 0:
            self.dry_history = self.dry_history[keep_from:]
            self.dry_history_start += keep_from
        self._forget_guard_gains()
        return out

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


PROCESSOR_REGISTRY = {
    "stft_frame_ort": StftFrameOrtProcessor,
}


class PureSoundStreamingRuntime:
    """Manifest-driven PureSound ONNX streaming runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        manifest_path: str | Path | None = None,
        provider: str = "auto",
        session: InferenceSessionLike | None = None,
    ):
        self.onnx_path = Path(onnx_path)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.onnx_path.with_suffix(".json")
        )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.config = StreamingRuntimeConfig.from_manifest(self.manifest)
        if session is None:
            import onnxruntime

            providers = self.resolve_providers(
                provider, onnxruntime.get_available_providers()
            )
            session = onnxruntime.InferenceSession(str(self.onnx_path), providers=providers)
        self.session = session
        self.providers = self.session.get_providers()
        self.processor = self.create_processor(self.config, self.session)
        self.sample_rate = self.processor.sample_rate
        self.fft_length = self.processor.fft_length
        self.win_length = self.processor.win_length
        self.hop_length = self.processor.hop_length
        self.freq_bins = self.processor.freq_bins

    @staticmethod
    def create_processor(
        config: StreamingRuntimeConfig,
        session: InferenceSessionLike,
    ) -> StreamingProcessor:
        processor_cls = PROCESSOR_REGISTRY.get(config.processor)
        if processor_cls is None:
            supported = ", ".join(sorted(PROCESSOR_REGISTRY))
            raise ValueError(
                f"Unsupported streaming processor: {config.processor}. "
                f"Supported processors: {supported}"
            )
        return processor_cls(config, session)

    @staticmethod
    def resolve_providers(provider: str, available: Sequence[str]) -> list[str]:
        choice = _PROVIDER_ALIASES.get(str(provider).strip().lower())
        if choice is None:
            choices = ", ".join(_PROVIDER_ALIASES)
            raise ValueError(f"provider must be one of: {choices}")
        available = list(available)
        if choice == "cpu":
            return [_CPU_PROVIDER]
        if choice == "cuda":
            return ([_CUDA_PROVIDER, _CPU_PROVIDER]
                    if _CUDA_PROVIDER in available else [_CPU_PROVIDER])
        if choice == "coreml":
            return ([_COREML_PROVIDER, _CPU_PROVIDER]
                    if _COREML_PROVIDER in available else [_CPU_PROVIDER])
        for candidate in (_CUDA_PROVIDER, _COREML_PROVIDER):
            if candidate in available:
                return [candidate, _CPU_PROVIDER]
        return [_CPU_PROVIDER]

    def reset(self, batch_size: int = 1) -> None:
        self.processor.reset(batch_size=batch_size)

    def run_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.processor.run_frame(frame)

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        return self.processor.process_samples(samples)

    def flush(self) -> np.ndarray:
        return self.processor.flush()

    def process_int16(self, samples: np.ndarray) -> np.ndarray:
        float_samples = np.asarray(samples, dtype=np.float32).reshape(-1) / 32768.0
        enhanced = self.process_samples(float_samples)
        return self.float_to_int16(enhanced)

    def flush_int16(self) -> np.ndarray:
        return self.float_to_int16(self.flush())

    @staticmethod
    def float_to_int16(samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype(np.int16)
