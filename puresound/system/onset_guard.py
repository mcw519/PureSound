"""Onset protection: do not delete anyone until a talker has been heard for a second.

`PresenceGate` and `Postprocessor` both act on *how much* the model attenuates.
This acts on *when it is allowed to attenuate at all*, and it is separate from
both because it needs no model internals whatsoever -- only the input waveform's
frame energy.

**The failure it removes.** The streaming state treats whoever it heard last as
the foreground. `reference_matrix_README` §3 measures that: ~1 s of speech builds
the anchor, 2 s saturates it, and the *next* near talker -- a different person,
same room, same 30-50 cm -- is then attenuated as a foreground change (v8 device
near clips go 0 -> 4-6 keep violations after a 2-3 s prefix). Before any anchor
exists the same thing happens to the first second of the very first talker.
`anchor_gate_README` §2 isolates the clause that fixes it: **until a talker has
been heard for >= 1 s of sustained speech, do not attenuate**. On its own, with
no distance readout, no threshold and no checkpoint dependence, it removes half
the keep violations on both v8 and v16 (26 -> 13 and 20 -> 7), and §6 turns that
into ASR: Dawn Chorus deletion 0.230 -> 0.123 (v8, `none`), 0.257 -> 0.147
(`background`), with WER 0.333 -> 0.232.

**What it costs.** Suppression depth, and the number is on the table: fit-set far
median -15.2 -> -12.0 dB, held-out 180d -19.4 -> ~-12, one or two suppress passes
lost, and the insertion rate climbs back toward the raw mix's because what the
model stopped hearing the guard lets back in. It is a keep-side safety belt whose
price is measured in far suppression; where to sit is a product choice, not a
version decision.

**Why a dry blend and not a gain.** The opposite of `PresenceGate`, deliberately.
There the job is to attenuate *past* what the model managed, so it has to be a
gain. Here the job is to hand back the input untouched, so it is exactly
``g * dry + (1 - g) * enh`` with ``g = 1`` meaning "the model is not consulted".
At ``g = 1`` the output is the input bit for bit; at ``g = 0`` it is the model's,
bit for bit. Both regimes are arithmetic, not estimates -- the integrator snaps
to exactly 0 and 1 once it is within ``snap`` of them.

**Where it sits.** Last. After `Postprocessor.blend_waveform` and after
`PresenceGate`, because it has to be able to restore the dry signal whatever the
earlier stages did to it; a gate that runs before the blend can only restore
``dry_blend`` of it.

**One characteristic of the floor tracker worth knowing.** The floor may climb
``floor_rise_db_per_s``, so a *stationary* broadband source only ``margin_db +
3*t`` dB over the floor stops counting as speech after roughly ``t`` seconds and,
``t_forget_s`` later, the guard protects again. That is the right answer for a
minimum-statistics tracker -- sustained unmodulated broadband energy IS the floor
-- and it is why a constant-noise fixture is a bad way to simulate a talker. Real
speech is modulated and stays over the floor; a fan or a stream does not, and
that is the case where the guard should get out of the model's way.

**On `t_forget_s`.** Not resolvable from the data. `onset_guard_sweep.py report`
prints how many (T_arm, tau_dn) cells the T_forget values separate on the field
set and the answer is none -- every field record has an onset inside its first
seconds, so the re-arming clause never fires there. The default 5.0 s comes from
the *model's* measured memory instead (`reference_matrix_README` §3: half the
anchor's effect is gone after 5 s of true floor, all of it by 10 s). Re-arming
sooner than the model forgets is the safe direction: it protects a talker the
model was about to treat as a foreground change.

Nothing here is learned by the network and no exported graph contains it, so a
deployment running the graph alone is running a different system -- same contract
as `Postprocessor` and `PresenceGate`, and `as_manifest()` records it the same
way.

Two faces, one arithmetic
-------------------------
`apply()` is the offline/torch face (whole utterance, used by
`SISO.forward(..., onset_guard=...)`); `streaming_state()` + `step()` is the
numpy face that consumes one hop at a time, for the ORT runtime. Both drive the
identical per-frame recursion (`_advance`), so they are bit-identical and a test
pins that. There is no lookahead in the detector: the floor is a running minimum,
the hangover is a hold and the min-run is a confirmation delay
(`onset_guard_sweep.py --strict`).

The one place a frame is not free: frame energy is measured over a 20 ms window
on the 10 ms grid (the reference implementation's `frame_energy_db`), so frame
``t`` needs hop ``t+1``. `step()` therefore emits frame ``t`` on the call that
feeds hop ``t+1`` -- see its docstring.

INTEGRATION NOTE -- ORT runtime (`puresound/streaming/base.py`)
--------------------------------------------------------------
Not wired, because that file was being edited by another session when this was
written. What it needs, in `StreamingOrt`:

* `reset()`: ``self.onset_state = self.onset_guard.streaming_state(hop=self.hop_length)``
  when `self.onset_guard` is not None (built from
  ``self.manifest.get(OnsetGuard.MANIFEST_KEY)`` in `__init__`, beside
  `self.dry_blend`), and `_remember_dry` must then keep history even at
  ``dry_blend >= 1.0``, because the guard needs the dry stream regardless of
  whether the blend does.
* feed it the dry stream hop by hop as it arrives and keep the resulting
  per-frame gains in a small ring; then in `_blend_dry`, on the span that
  already has an aligned `reference` slice, apply
  ``out[span] = g * reference + (1 - g) * out[span]`` **after** the existing
  ``dry_blend`` line and before the `np.clip`.
* alignment is the same ``dry_delay = streaming_delay_frames * hop_length`` the
  blend already uses: output sample ``n`` corresponds to dry sample
  ``n - dry_delay``, so output frame ``k`` takes the guard gain computed for dry
  frame ``k``. The guard's own 20 ms window needs dry hop ``k+1``, which has
  already arrived whenever ``streaming_delay_frames >= 1`` (DPCRN's is
  `bottleneck_delay`, ~3). At ``streaming_delay_frames == 0`` the guard would add
  one hop (10 ms) of latency of its own, and that has to be either accepted or
  paid for by holding the last frame's gain -- it must not be papered over by
  using an unaligned gain, which is exactly the mistake `_blend_dry`'s docstring
  already warns about for the dry reference.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class OnsetGuardState:
    """Streaming state for one mono stream. Mutated in place by `OnsetGuard.step`.

    Not part of the public contract beyond being opaque: build it with
    `OnsetGuard.streaming_state()` and hand it back to `step` unchanged.
    """

    hop: int
    fps: float
    # frame counts / rates, precomputed from the knobs and the frame rate
    n_floor: int
    n_init: int
    n_hold: int
    n_minr: int
    n_arm: int
    n_forget: Optional[int]
    rise_per_frame: float
    a_up: float
    a_dn: float
    # detector state
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
    """Inference-only onset protection: dry until a talker is confirmed.

    Every knob is a constructor argument and the incoherent combinations are
    rejected here rather than producing a guard that looks like it works.

    Args:
        t_arm_s: continuous activity this long confirms an anchor and hands the
            output over to the model. 1.0 s is where the model's own anchor forms
            (`reference_matrix_README` §3); 0.5 s does nothing there, so arming
            faster would hand over before the model has what it needs.
        t_forget_s: this long without activity drops the anchor, so the next
            onset is protected again. ``inf`` disables re-arming (what the
            reference sim did). See the module docstring on why this is 5.0.
        tau_up_s / tau_dn_s: first-order integrator, ``up`` toward dry (protect)
            and ``dn`` toward the model (release). Up must not be slower than
            down: protecting late costs the user a deleted word, releasing late
            costs a bystander a moment of leakage.
        margin_db: activity threshold above the tracked floor.
        floor_win_s: window of the running-minimum floor.
        floor_rise_db_per_s: how fast the floor may climb. It falls instantly --
            a floor that lags downward turns a quiet passage into "speech".
        init_s: the floor's initialisation window. This is the only part of the
            detector that is not sample-causal, and the streaming form pays for
            it by buffering (see `step`).
        hangover_s: activity is held this long after the last frame over
            threshold, bridging intra-utterance pauses.
        min_run_s: activity is only declared after it has held this long --
            a confirmation delay, which is the streaming form of "drop runs
            shorter than this".
        snap: gain residual below which the integrator snaps to exactly 0 or 1,
            which is what makes both end regimes bit-identical rather than
            approximately so.
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

    def __post_init__(self):
        for name in ("t_arm_s", "tau_up_s", "tau_dn_s", "floor_win_s",
                     "floor_rise_db_per_s", "init_s", "min_run_s", "snap"):
            v = getattr(self, name)
            if not (v > 0.0) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        if not (self.t_forget_s > 0.0):
            raise ValueError(
                f"t_forget_s must be > 0 (use math.inf to disable re-arming), "
                f"got {self.t_forget_s}"
            )
        if self.hangover_s < 0.0 or not math.isfinite(self.hangover_s):
            raise ValueError(f"hangover_s must be finite and >= 0, got {self.hangover_s}")
        if self.margin_db <= 0.0:
            raise ValueError(
                f"margin_db must be > 0, got {self.margin_db}: at or below the "
                "tracked floor every frame is 'speech' and the guard never protects"
            )
        if self.tau_up_s > self.tau_dn_s:
            raise ValueError(
                f"tau_up_s ({self.tau_up_s}) must not exceed tau_dn_s "
                f"({self.tau_dn_s}): the guard has to reach dry faster than it "
                "hands back to the model, or an onset is half-deleted while it "
                "protects and a bystander is leaked while it releases"
            )
        if self.init_s > self.floor_win_s:
            raise ValueError(
                f"init_s ({self.init_s}) must fit inside floor_win_s "
                f"({self.floor_win_s}); a longer initialisation than the tracking "
                "window makes the floor's first value unreachable afterwards"
            )
        if self.init_s >= self.t_arm_s:
            raise ValueError(
                f"init_s ({self.init_s}) must be shorter than t_arm_s "
                f"({self.t_arm_s}): the streaming form is dry while it fills the "
                "initialisation window, which is only exact if the guard cannot "
                "have armed by then"
            )

    # ------------------------------------------------------------------ #
    # frame grid

    @staticmethod
    def _frame_energy_db(x: np.ndarray, n_frames: int, hop: int) -> np.ndarray:
        """20 ms frame energy in dB on the 10 ms grid, as the reference does.

        Frame ``t`` spans samples ``[t*hop, t*hop + 2*hop)``, zero-padded at the
        end. Chunked so a long file does not materialise an index matrix of
        ``n_frames x 2*hop``; each frame's mean is over its own gathered window,
        so the chunking cannot change a value.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        need = (n_frames - 1) * hop + 2 * hop
        if x.size < need:
            x = np.pad(x, (0, need - x.size))
        out = np.empty(n_frames, dtype=np.float64)
        cols = np.arange(2 * hop)[None, :]
        chunk = max(1, (1 << 20) // (2 * hop))
        for lo in range(0, n_frames, chunk):
            hi = min(n_frames, lo + chunk)
            idx = np.arange(lo, hi)[:, None] * hop + cols
            out[lo:hi] = (x[idx] ** 2).mean(axis=1)
        return 10.0 * np.log10(out + 1e-12)

    def streaming_state(self, *, hop: int = 160, sr: float = 16000.0) -> OnsetGuardState:
        """A fresh per-stream state on the ``sr / hop`` frame grid."""
        if hop <= 0:
            raise ValueError(f"hop must be > 0, got {hop}")
        if sr <= 0.0:
            raise ValueError(f"sr must be > 0, got {sr}")
        fps = float(sr) / float(hop)
        n_arm = max(1, int(round(self.t_arm_s * fps)))
        n_forget = (None if not math.isfinite(self.t_forget_s)
                    else max(1, int(round(self.t_forget_s * fps))))
        # int(), not round(): this is the reference implementation's arithmetic.
        n_hold = int(self.hangover_s * fps) + 1
        n_minr = max(1, int(self.min_run_s * fps))
        n_init = max(1, int(round(self.init_s * fps)))
        if n_init + 2 > n_minr + n_arm:
            raise ValueError(
                f"at {fps:g} fps the initialisation window is {n_init} frames but "
                f"the guard could arm by frame {n_minr + n_arm - 2}; the streaming "
                "form's priming shortcut is only exact when it cannot. Shorten "
                "init_s or lengthen t_arm_s / min_run_s."
            )
        return OnsetGuardState(
            hop=int(hop), fps=fps,
            n_floor=max(1, int(round(self.floor_win_s * fps))),
            n_init=n_init, n_hold=n_hold, n_minr=n_minr,
            n_arm=n_arm, n_forget=n_forget,
            rise_per_frame=self.floor_rise_db_per_s / fps,
            a_up=1.0 - math.exp(-1.0 / (self.tau_up_s * fps)),
            a_dn=1.0 - math.exp(-1.0 / (self.tau_dn_s * fps)),
        )

    # ------------------------------------------------------------------ #
    # the one recursion both faces run

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

    # ------------------------------------------------------------------ #
    # offline face

    def frame_gain(self, dry: np.ndarray, *, hop: int, sr: float = 16000.0,
                   n_frames: Optional[int] = None) -> np.ndarray:
        """Per-frame gain in [0, 1] for one mono dry signal. 1.0 = dry, 0.0 = model.

        ``n_frames`` defaults to covering every sample; pass it to match another
        frame grid exactly (e.g. a cached bottleneck's).
        """
        x = np.asarray(dry, dtype=np.float64).reshape(-1)
        if n_frames is None:
            n_frames = int(math.ceil(x.size / hop)) if x.size else 0
        if n_frames <= 0:
            return np.zeros(0, dtype=np.float64)
        e_db = self._frame_energy_db(x, n_frames, hop)
        st = self.streaming_state(hop=hop, sr=sr)
        n0 = max(1, min(st.n_init, n_frames))
        st.floor_cur = float(e_db[:n0].min())
        out = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            out[i] = self._advance(st, i, float(e_db[i]))
        return out

    def apply(self, enh: torch.Tensor, dry: torch.Tensor, *, hop: int,
              sr: float = 16000.0) -> torch.Tensor:
        """Hand the dry signal back where no talker has been confirmed yet.

        ``enh`` is the finished (post-blend, post-gate) output, ``dry`` the model
        input. Only the overlapping span is touched -- the STFT round trip can
        leave the two a few samples apart -- exactly as
        `Postprocessor.blend_waveform` does. Result is clamped to [-1, 1].

        ``hop`` is the encoder hop, which is what puts the frame grid and the
        sample grid in register; the gain is held over each frame's hop.
        """
        if dry.dim() == enh.dim() + 1 and dry.shape[0] == 1:
            dry = dry.squeeze(0)
        overlap = min(enh.shape[-1], dry.shape[-1])
        if overlap <= 0:
            return enh
        d = dry[..., :overlap]
        rows = d.reshape(-1, overlap).detach().to(device="cpu", dtype=torch.float64).numpy()
        n_frames = int(math.ceil(overlap / hop))
        g = np.stack([self.frame_gain(r, hop=hop, sr=sr, n_frames=n_frames)
                      for r in rows])
        gt = torch.from_numpy(g).to(device=enh.device, dtype=enh.dtype)
        gt = gt.repeat_interleave(hop, dim=-1)[..., :overlap].reshape(d.shape)
        out = enh.clone()
        out[..., :overlap] = torch.clamp(
            gt * d.to(dtype=enh.dtype, device=enh.device)
            + (1.0 - gt) * enh[..., :overlap],
            min=-1.0, max=1.0,
        )
        return out

    # ------------------------------------------------------------------ #
    # streaming face

    def step(self, state: OnsetGuardState,
             dry_frame: np.ndarray) -> Tuple[float, OnsetGuardState]:
        """Consume one hop of the DRY stream, return ``(gain, state)``.

        Call ``c`` (1-based) returns the gain for frame ``c - 2``: frame energy
        spans 20 ms, so the frame is only complete once the following hop has
        arrived. The first call primes that window and returns 1.0 (dry), which
        is what a runtime would apply to output samples that have no input yet
        anyway. To reproduce `frame_gain`'s ``F`` gains, feed hops ``0..F``
        (the last zero-padded, as the offline framing pads) and discard the first
        call's value.

        While the floor's ``init_s`` window fills, the returned gain is 1.0
        without running the recursion; `streaming_state` refuses knobs where the
        guard could have armed by then, so that shortcut is exact rather than an
        approximation, and the replay below re-checks it.

        The state is mutated in place and returned for convenience.
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
                if j < len(buf) - 1 and g != 1.0:
                    raise RuntimeError(
                        f"priming shortcut is not exact: frame {j} of the "
                        f"initialisation window wants gain {g}, not 1.0. This "
                        "should be unreachable -- streaming_state validates it."
                    )
            state.init_buf = []
            return g, state
        return self._advance(state, i, e_db), state

    # ------------------------------------------------------------------ #

    #: Manifest key, beside `Postprocessor.MANIFEST_KEY` and
    #: `PresenceGate.MANIFEST_KEY`.
    MANIFEST_KEY = "onset_guard"

    def as_manifest(self) -> dict:
        return {
            "t_arm_s": float(self.t_arm_s),
            "t_forget_s": float(self.t_forget_s),
            "tau_up_s": float(self.tau_up_s),
            "tau_dn_s": float(self.tau_dn_s),
            "margin_db": float(self.margin_db),
            "floor_win_s": float(self.floor_win_s),
            "floor_rise_db_per_s": float(self.floor_rise_db_per_s),
            "init_s": float(self.init_s),
            "hangover_s": float(self.hangover_s),
            "min_run_s": float(self.min_run_s),
            "snap": float(self.snap),
        }

    @classmethod
    def from_manifest(cls, spec: dict) -> "OnsetGuard":
        """The inverse of `as_manifest`, so an export round-trips."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in spec.items() if k in known})
