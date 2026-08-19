"""Near-presence gating, applied after the mask and only at inference.

`Postprocessor` relieves over-suppression; this does the opposite job, and the
two are separate because they are limited by different things.

**Why a gain and not a deeper blend.** `dry_blend` bounds attenuation at
``20*log10(1 - dry_blend)`` -- 0.9 caps it at -20 dB. On the cold-start field
clips the mask attenuates the bystander by 0.39 dB, so that ceiling sits 20 dB
below where the signal actually is and is not what limits anything. Measured:
driving `dry_blend` from a presence estimate moves those clips by a median of
0.03 dB. A gain moves them by 9.6 dB. See
`egs/voice_isolate/benchmarks/probes/b_traj_README.md`.

**The quantity.** One continuous state, not a regime:

* a linear readout on the frozen bottleneck gives ``s_t`` in (0, 1), the
  probability a near user is talking in frame ``t``;
* ``b_t`` is an asymmetric leaky integrator over ``s`` -- fast up, slow down,
  started at 1.0, i.e. presuming someone is there until the evidence says
  otherwise. Measured settling from audible onset is ~1 s, and the readout opens
  faster than it closes, which is the direction that protects the user.

**The dead zone is the safety property.** The gain is exactly 1.0 for
``b >= b_hi``, so while a near user is confidently present the output is
bit-identical to running without this at all. That is not a claim about the
readout, it is arithmetic; what the readout has to earn is staying above
``b_hi``, and on the field set it does on 98.9% of frames where the user is
audibly talking.

**What it does not fix.** The gain floor is not the binding constraint: dropping
it from -20 dB to -40 dB buys 1.15 dB of measured suppression, because the limit
is how far ``b`` falls, not how far the gain is allowed to. Deeper suppression
needs a more confident readout, not a lower floor.

Nothing here is learned by the network and no exported graph contains it, so a
deployment running the graph alone is running a different system -- same contract
as `Postprocessor`, and the manifest records it the same way.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class PresenceGate:
    """Inference-only near-presence gain.

    Args:
        weight: readout coefficients over the frequency-pooled bottleneck,
            shape ``[C]``. The standardizer is expected to be folded in already,
            so scoring is one dot product.
        bias: readout intercept.
        b_hi: dead-zone edge. At or above it the gain is exactly 1.0 and the
            output is bit-identical to not gating.
        b_lo: at or below it the gain is at its floor. Must be below ``b_hi``.
        gain_floor_db: the deepest the gain goes, in dB.
        tau_up_s / tau_dn_s: integrator time constants. Up should be much faster
            than down -- opening late costs the user, closing late costs a
            bystander a moment of leakage.
        b_init: the state before any evidence. 1.0 presumes presence.
    """

    weight: torch.Tensor
    bias: float
    b_hi: float = 0.50
    b_lo: float = 0.10
    gain_floor_db: float = -26.0
    tau_up_s: float = 0.05
    tau_dn_s: float = 1.0
    b_init: float = 1.0

    def __post_init__(self):
        if self.weight.dim() != 1:
            raise ValueError(f"weight must be 1-D [C], got {tuple(self.weight.shape)}")
        if not 0.0 < self.b_lo < self.b_hi <= 1.0:
            raise ValueError(
                f"need 0 < b_lo < b_hi <= 1, got b_lo={self.b_lo}, b_hi={self.b_hi}. "
                "b_hi is the dead-zone edge; at b_hi = 1.0 nothing is ever bit-identical."
            )
        if self.gain_floor_db >= 0.0:
            raise ValueError(
                f"gain_floor_db must be negative (it attenuates), got {self.gain_floor_db}"
            )
        for name in ("tau_up_s", "tau_dn_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0.0 <= self.b_init <= 1.0:
            raise ValueError(f"b_init must be in [0, 1], got {self.b_init}")

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Union[str, Path], **overrides) -> "PresenceGate":
        """Read a readout from the ``.npz`` written by the fitting probe.

        A sibling ``.json`` may carry the same knobs, in which case explicit
        keyword overrides still win -- the file records what was fitted, the call
        site decides the operating point.
        """
        import numpy as np

        path = Path(path)
        data = np.load(path)
        kwargs = {}
        meta_path = path.with_suffix(".json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            kwargs = {k: meta[k] for k in
                      ("b_hi", "b_lo", "gain_floor_db", "tau_up_s", "tau_dn_s", "b_init")
                      if k in meta}
        kwargs.update(overrides)
        return cls(weight=torch.from_numpy(data["weight"]).float(),
                   bias=float(data["bias"]), **kwargs)

    def presence(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """``s`` per frame from a ``[N, C, F, T]`` bottleneck.

        Pooled over frequency the same way the readout was fitted. Returns
        ``[N, T]`` in (0, 1).
        """
        if bottleneck.dim() != 4:
            raise ValueError(
                f"bottleneck must be [N, C, F, T], got {tuple(bottleneck.shape)}"
            )
        pooled = bottleneck.mean(dim=2)                       # [N, C, T]
        w = self.weight.to(pooled.dtype).to(pooled.device)
        if w.shape[0] != pooled.shape[1]:
            raise ValueError(
                f"readout is {w.shape[0]}-dim but the bottleneck has "
                f"{pooled.shape[1]} channels -- wrong checkpoint for this readout"
            )
        return torch.sigmoid(torch.einsum("nct,c->nt", pooled, w) + self.bias)

    def trajectory(self, s: torch.Tensor, frame_rate: float) -> torch.Tensor:
        """``b`` per frame: asymmetric leaky integration of ``s``.

        Sequential by construction -- the whole point is that the state carries
        forward, so this is the one part that cannot be vectorised over time.
        """
        if frame_rate <= 0.0:
            raise ValueError(f"frame_rate must be > 0, got {frame_rate}")
        a_up = 1.0 - math.exp(-1.0 / (self.tau_up_s * frame_rate))
        a_dn = 1.0 - math.exp(-1.0 / (self.tau_dn_s * frame_rate))
        out = torch.empty_like(s)
        cur = torch.full_like(s[..., 0], self.b_init)
        for t in range(s.shape[-1]):
            v = s[..., t]
            cur = cur + torch.where(v > cur, a_up, a_dn) * (v - cur)
            out[..., t] = cur
        return out

    def gain(self, b: torch.Tensor) -> torch.Tensor:
        """Per-frame gain in (0, 1]: exactly 1.0 above ``b_hi``, floor below ``b_lo``.

        Linear in dB between, so the perceived reduction ramps evenly rather than
        collapsing at one end.
        """
        t = ((self.b_hi - b) / (self.b_hi - self.b_lo)).clamp(0.0, 1.0)
        return torch.pow(10.0, t * (self.gain_floor_db / 20.0))

    def apply(
        self, wav: torch.Tensor, bottleneck: torch.Tensor, *, hop: int
    ) -> torch.Tensor:
        """Gate a finished waveform using the bottleneck it came from.

        ``hop`` is the encoder hop in samples, which is what puts the frame grid
        and the sample grid in register. The gain is held over each frame's hop
        and the tail is padded with the last value; ``b`` is already smooth in
        time, so no extra smoothing is applied and none is needed.
        """
        b = self.trajectory(self.presence(bottleneck), 16000.0 / hop
                            if hop else 1.0)
        g = self.gain(b)                                       # [N, T_frames]
        g = g.repeat_interleave(hop, dim=-1)
        n = wav.shape[-1]
        if g.shape[-1] < n:
            g = torch.cat([g, g[..., -1:].expand(*g.shape[:-1], n - g.shape[-1])], -1)
        g = g[..., :n].to(wav.dtype).to(wav.device)
        while g.dim() < wav.dim():
            g = g.unsqueeze(1)
        return torch.clamp(wav * g, min=-1.0, max=1.0)

    #: Manifest key, beside `Postprocessor.MANIFEST_KEY`.
    MANIFEST_KEY = "presence_gate"

    def as_manifest(self) -> dict:
        return {
            "b_hi": float(self.b_hi),
            "b_lo": float(self.b_lo),
            "gain_floor_db": float(self.gain_floor_db),
            "tau_up_s": float(self.tau_up_s),
            "tau_dn_s": float(self.tau_dn_s),
            "b_init": float(self.b_init),
            "readout_dim": int(self.weight.shape[0]),
        }
