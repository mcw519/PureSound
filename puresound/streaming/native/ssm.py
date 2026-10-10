"""A traceable reference for the optional FP32 ONNX CPU operator."""

import torch

from puresound.nnet.lobe.ssm import selective_state_step

DOMAIN = "com.puresound"
OP_VERSION = 1


class _FusedSsmStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, u, b, c, h, a, skip, z):
        return selective_state_step(dt, u, b, c, h, a, skip, z)

    @staticmethod
    def symbolic(g, dt, u, b, c, h, a, skip, z):
        next_h, y = g.op(f"{DOMAIN}::FusedSsmStep", dt, u, b, c, h, a, skip, z,
                         outputs=2)
        next_h.setType(h.type())
        y.setType(u.type())
        return next_h, y


def fused_ssm_step(dt, u, b, c, h, a, skip, z):
    return _FusedSsmStep.apply(dt, u, b, c, h, a, skip, z)
