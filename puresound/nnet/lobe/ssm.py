"""Selective state-space (Mamba/S6) blocks for streaming enhancement.

Why this exists: the diagnosed failure mode of this project is a missing
long-range reference -- the model only ever compares, and cold start gives it
nothing to compare against. The inter(-time) LSTM is the only component that
carries context, its training exposure is 6 s, and its BPTT reach is far
shorter. An SSM keeps a per-frame recurrent form (zero extra look-ahead, CPU
step cost measured at ~2x the LSTM step and well inside the 10 ms frame
budget) while its selective decay is built for context lengths the LSTM
cannot train through.

`MambaInter` is a drop-in for the inter `SingleRNN`: [N, D, T] in, [N, D, T]
out, causal. Three execution paths, one parameter set:

* CUDA training: `mamba_ssm`'s fused `selective_scan_fn` when importable.
* Fallback (CPU, export tracing, or kernel absent): a pure-PyTorch sequential
  scan -- slower but exact, and the ONLY path a deployment needs.
* Streaming: `step()` advances one frame with an explicit (conv cache, h)
  state. The SSM state h is kept in fp32 no matter the autocast mode -- the
  same discipline the VADHead EMA bank needed, and for the same reason: tiny
  per-step increments underflow in bf16.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional CUDA kernel for fast training; inference never needs it
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # pragma: no cover - absence is a supported configuration
    selective_scan_fn = None


class MambaInter(nn.Module):
    """One S6 block matched to the inter-RNN's interface and budget.

    Width defaults (d_state 16, d_conv 4, expand 2) put the parameter count at
    ~116k for d_model 128 against the SingleRNN(LSTM 96)+proj's ~99k, so a
    swap is a like-for-like capacity trade, not a hidden size increase.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(p=dropout)

        # S4D-real initialisation: A negative-real, log-spaced across the state.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt initialisation (the part that sets the memory horizon): softplus of
        # the bias lands log-uniformly in [dt_min, dt_max].
        dt_scale = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_scale, dt_scale)
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    # ------------------------------------------------------------------ #

    def _pre(self, x: torch.Tensor):
        """Shared front: [N, D, T] -> (u [N, d_inner, T], z, delta_raw, B, C)."""
        xz = self.in_proj(x.transpose(1, 2))                    # [N, T, 2*d_inner]
        x_part, z = xz.split(self.d_inner, dim=-1)
        u = self.conv1d(F.pad(x_part.transpose(1, 2), (self.d_conv - 1, 0)))
        u = F.silu(u)                                           # [N, d_inner, T]
        x_dbl = self.x_proj(u.transpose(1, 2))                  # [N, T, R+2S]
        dt_r, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta_raw = dt_r @ self.dt_proj.weight.t()              # bias applied later
        return u, z, delta_raw, B, C

    #: sequential-scan chunk length; the training-time memory knob. Backward
    #: keeps only chunk boundaries and recomputes inside (checkpointing).
    #: Everything fp32 lives INSIDE the checkpointed region: the naive version
    #: kept full-length fp32 copies of u/dt/B/C/z outside it, which pushed the
    #: whole model past the 22 GB the LSTM baseline already runs against --
    #: that is how the second v13 launch died (OOM in the valid-loss WavLM,
    #: the last straw, not the culprit).
    SCAN_CHUNK = 64

    def _scan_chunk(self, h, u_c, draw_c, B_c, C_c, z_c):
        """One chunk, bf16 in / bf16 out, fp32 only transiently inside."""
        dev = u_c.device.type
        with torch.autocast(device_type=dev, enabled=False):
            u32 = u_c.float()
            dt = F.softplus(draw_c.float() + self.dt_proj.bias.float())     # [N,t,d_inner]
            A = -torch.exp(self.A_log.float())
            B32, C32 = B_c.float(), C_c.float()
            h = h.float()
            ys = []
            for t in range(u32.shape[-1]):
                dt_t = dt[:, t]
                h = h * torch.exp(dt_t.unsqueeze(-1) * A) \
                    + (dt_t * u32[:, :, t]).unsqueeze(-1) * B32[:, t].unsqueeze(1)
                ys.append((h * C32[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=-1) + self.D.float().unsqueeze(-1) * u32
            y = y * F.silu(z_c.float().transpose(1, 2))
        return y.to(u_c.dtype), h

    def _scan_fallback(self, u, delta_raw, B, C, z):
        """Chunked sequential scan; under grad each chunk is checkpointed and
        carries its own fp32 lifetime."""
        N, _, T = u.shape
        h = torch.zeros(N, self.d_inner, self.d_state,
                        device=u.device, dtype=torch.float32)
        use_ckpt = torch.is_grad_enabled() and self.training
        outs = []
        for a in range(0, T, self.SCAN_CHUNK):
            b = min(T, a + self.SCAN_CHUNK)
            args = (h, u[:, :, a:b], delta_raw[:, a:b], B[:, a:b], C[:, a:b],
                    z[:, a:b])
            if use_ckpt:
                y_c, h = torch.utils.checkpoint.checkpoint(
                    self._scan_chunk, *args, use_reentrant=False)
            else:
                y_c, h = self._scan_chunk(*args)
            outs.append(y_c)
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, D, T] -> [N, D, T], causal.

        Fallback training wraps the WHOLE block (projections included) in an
        outer checkpoint, nested over the per-chunk inner ones: the projection
        intermediates were the remaining +1.6 GiB against the LSTM baseline,
        which trains at the 22 GB card's edge. Recompute costs one extra
        forward of cheap matmuls."""
        use_kernel = (selective_scan_fn is not None and x.is_cuda
                      and not torch.jit.is_tracing())
        if not use_kernel and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        u, z, delta_raw, B, C = self._pre(x)
        use_kernel = (selective_scan_fn is not None and u.is_cuda
                      and not torch.jit.is_tracing())
        if use_kernel:
            y = selective_scan_fn(
                u, delta_raw.transpose(1, 2).contiguous(),
                -torch.exp(self.A_log.float()),
                B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(),
                self.D.float(), z=z.transpose(1, 2).contiguous(),
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
        else:
            y = self._scan_fallback(u, delta_raw, B, C, z)
        return self.out_proj(self.drop(y.transpose(1, 2))).transpose(1, 2)

    # ------------------------------------------------------------------ #

    def initial_stream_state(
        self, batch: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(conv cache [N, d_inner, d_conv-1] model dtype, h [N, d_inner, S] fp32)."""
        p = self.conv1d.weight
        dev = device if device is not None else p.device
        dt_ = dtype if dtype is not None else p.dtype
        return (torch.zeros(batch, self.d_inner, self.d_conv - 1, device=dev, dtype=dt_),
                torch.zeros(batch, self.d_inner, self.d_state,
                            device=dev, dtype=torch.float32))

    def step(
        self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One frame: x_t [N, D] -> y_t [N, D]. Bit-faithful to the fallback scan."""
        conv_cache, h = state
        xz = self.in_proj(x_t)                                  # [N, 2*d_inner]
        x_part, z = xz.split(self.d_inner, dim=-1)
        window = torch.cat([conv_cache, x_part.unsqueeze(-1)], dim=-1)
        u = (window * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        u = F.silu(u)                                           # [N, d_inner]
        x_dbl = self.x_proj(u)
        dt_r, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            dt = F.softplus(dt_r.float() @ self.dt_proj.weight.t().float()
                            + self.dt_proj.bias.float())        # [N, d_inner]
            A = -torch.exp(self.A_log.float())
            h = h * torch.exp(dt.unsqueeze(-1) * A) \
                + (dt * u.float()).unsqueeze(-1) * B.float().unsqueeze(1)
            y = (h * C.float().unsqueeze(1)).sum(-1) + self.D.float() * u.float()
            y = y * F.silu(z.float())
        y = self.out_proj(y.to(x_t.dtype))
        return y, (window[..., 1:], h)
