"""Run a small ONNX graph as a differentiable torch module.

A published judge (DNSMOS, a VAD, a quality predictor) usually ships as an ONNX
file. onnxruntime scores it but gives no gradient, so it can only sit in a gate.
Executing the same graph with torch ops turns it into a loss term: every weight
is a frozen buffer, the input keeps its autograd history.

Only the op subset such judges use is implemented (convolutions, pooling,
elementwise maths, reshapes, matmul). An unsupported op raises at load time
with its name, rather than at the first forward.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _attrs(node) -> dict:
    import onnx

    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _pads(pads, rank: int) -> tuple[int, ...]:
    """ONNX pads are [x1_begin, x2_begin, ..., x1_end, x2_end, ...]; torch wants
    (xN_begin, xN_end, ..., x1_begin, x1_end) for F.pad."""
    if not pads:
        return tuple(0 for _ in range(2 * rank))
    begin, end = pads[:rank], pads[rank:]
    out = []
    for b, e in zip(reversed(begin), reversed(end)):
        out += [int(b), int(e)]
    return tuple(out)


class OnnxModule(nn.Module):
    """``OnnxModule(path)(x)`` evaluates the graph on ``x`` (its single input).

    The module is inference-only (``requires_grad`` is off on every buffer) and
    keeps the graph's node order, so a Reshape whose target shape is an
    initializer, or a Slice whose bounds are, reads them as constants.
    """

    SUPPORTED = {
        "Conv", "Relu", "MaxPool", "ReduceMax", "MatMul", "Add", "Mul", "Div", "Sub",
        "Sqrt", "Pow", "Max", "Min", "Log", "Exp", "Slice", "Reshape", "Concat",
        "Transpose", "Unsqueeze", "Squeeze", "Identity", "Sigmoid", "Tanh", "Clip",
        "Flatten", "Gemm", "ReduceMean", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    }

    def __init__(self, path: str | Path):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        graph = model.graph
        self.input_names = [i.name for i in graph.input if i.name not in {t.name for t in graph.initializer}]
        self.output_names = [o.name for o in graph.output]
        self._const: dict[str, str] = {}
        for i, tensor in enumerate(graph.initializer):
            key = f"c{i}"
            self.register_buffer(key, torch.from_numpy(numpy_helper.to_array(tensor).copy()), persistent=False)
            self._const[tensor.name] = key
        unsupported = sorted({n.op_type for n in graph.node} - self.SUPPORTED)
        if unsupported:
            raise ValueError(f"{Path(path).name}: unsupported ONNX op(s) {unsupported}")
        self.nodes = [(n.op_type, list(n.input), list(n.output), _attrs(n)) for n in graph.node]
        for p in self.buffers():
            p.requires_grad_(False)

    # -- execution ---------------------------------------------------------
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        env: dict[str, torch.Tensor] = {name: getattr(self, key) for name, key in self._const.items()}
        for name, x in zip(self.input_names, inputs):
            env[name] = x
        for op, ins, outs, a in self.nodes:
            xs = [env[i] if i else None for i in ins]
            env[outs[0]] = getattr(self, f"_op_{op}")(xs, a)
        result = [env[o] for o in self.output_names]
        return result[0] if len(result) == 1 else result

    @staticmethod
    def _op_Identity(xs, a): return xs[0]
    @staticmethod
    def _op_Relu(xs, a): return F.relu(xs[0])
    @staticmethod
    def _op_Sigmoid(xs, a): return torch.sigmoid(xs[0])
    @staticmethod
    def _op_Tanh(xs, a): return torch.tanh(xs[0])
    @staticmethod
    def _op_Add(xs, a): return xs[0] + xs[1]
    @staticmethod
    def _op_Sub(xs, a): return xs[0] - xs[1]
    @staticmethod
    def _op_Mul(xs, a): return xs[0] * xs[1]
    @staticmethod
    def _op_Div(xs, a): return xs[0] / xs[1]
    # sqrt and log are floored at a tiny positive value: their derivative at 0 is
    # infinite, and one exactly-zero STFT bin (a silent frame) would otherwise
    # turn the whole batch's gradient into NaN. 1e-12 is far below any value
    # these judges see and leaves the forward unchanged.
    GRAD_FLOOR = 1e-12
    @staticmethod
    def _op_Sqrt(xs, a): return torch.sqrt(xs[0].clamp_min(OnnxModule.GRAD_FLOOR))
    @staticmethod
    def _op_Log(xs, a): return torch.log(xs[0].clamp_min(OnnxModule.GRAD_FLOOR))
    @staticmethod
    def _op_Exp(xs, a): return torch.exp(xs[0])
    @staticmethod
    def _op_Pow(xs, a): return torch.pow(xs[0], xs[1])
    @staticmethod
    def _op_Max(xs, a):
        out = xs[0]
        for x in xs[1:]:
            out = torch.maximum(out, x)
        return out
    @staticmethod
    def _op_Min(xs, a):
        out = xs[0]
        for x in xs[1:]:
            out = torch.minimum(out, x)
        return out
    @staticmethod
    def _op_Clip(xs, a):
        lo = xs[1] if len(xs) > 1 and xs[1] is not None else a.get("min")
        hi = xs[2] if len(xs) > 2 and xs[2] is not None else a.get("max")
        return torch.clamp(xs[0], min=None if lo is None else float(lo), max=None if hi is None else float(hi))
    @staticmethod
    def _op_MatMul(xs, a): return torch.matmul(xs[0], xs[1])
    @staticmethod
    def _op_Gemm(xs, a):
        x, w = xs[0], xs[1]
        x = x.t() if a.get("transA", 0) else x
        w = w.t() if a.get("transB", 0) else w
        y = a.get("alpha", 1.0) * torch.matmul(x, w)
        return y + a.get("beta", 1.0) * xs[2] if len(xs) > 2 and xs[2] is not None else y
    @staticmethod
    def _op_Flatten(xs, a):
        axis = a.get("axis", 1); x = xs[0]
        return x.reshape(int(np.prod(x.shape[:axis])) if axis else 1, -1)
    @staticmethod
    def _op_Transpose(xs, a): return xs[0].permute(*[int(p) for p in a["perm"]])
    @staticmethod
    def _op_Reshape(xs, a):
        shape = [int(s) for s in xs[1].tolist()]
        x = xs[0]
        shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return x.reshape(shape)
    @staticmethod
    def _op_Unsqueeze(xs, a):
        axes = [int(v) for v in (xs[1].tolist() if len(xs) > 1 and xs[1] is not None else a["axes"])]
        x = xs[0]
        for ax in sorted(axes):
            x = x.unsqueeze(ax if ax >= 0 else ax + x.dim() + 1)
        return x
    @staticmethod
    def _op_Squeeze(xs, a):
        axes = [int(v) for v in (xs[1].tolist() if len(xs) > 1 and xs[1] is not None else a.get("axes", []))]
        x = xs[0]
        for ax in sorted(axes, reverse=True):
            x = x.squeeze(ax)
        return x if axes else x.squeeze()
    @staticmethod
    def _op_Concat(xs, a): return torch.cat(xs, dim=int(a["axis"]))
    @staticmethod
    def _op_Slice(xs, a):
        x = xs[0]
        starts, ends = xs[1].tolist(), xs[2].tolist()
        axes = xs[3].tolist() if len(xs) > 3 and xs[3] is not None else list(range(len(starts)))
        steps = xs[4].tolist() if len(xs) > 4 and xs[4] is not None else [1] * len(starts)
        for s, e, ax, st in zip(starts, ends, axes, steps):
            n = x.shape[ax]; s, e = int(s), int(e)
            s = max(0, s + n) if s < 0 else min(s, n); e = max(0, e + n) if e < 0 else min(e, n)
            x = x.narrow(ax, s, max(0, e - s)) if int(st) == 1 else x.index_select(ax, torch.arange(s, e, int(st), device=x.device))
        return x
    @staticmethod
    def _op_ReduceMax(xs, a):
        axes = a.get("axes"); keep = bool(a.get("keepdims", 1)); x = xs[0]
        if axes is None:
            return x.amax() if not keep else x.amax().reshape([1] * x.dim())
        for ax in sorted([int(v) for v in axes], reverse=True):
            x = x.amax(dim=ax, keepdim=keep)
        return x
    @staticmethod
    def _op_ReduceMean(xs, a):
        axes = a.get("axes"); keep = bool(a.get("keepdims", 1)); x = xs[0]
        return x.mean(dim=[int(v) for v in axes], keepdim=keep) if axes is not None else x.mean()
    @staticmethod
    def _op_GlobalMaxPool(xs, a): return xs[0].amax(dim=tuple(range(2, xs[0].dim())), keepdim=True)
    @staticmethod
    def _op_GlobalAveragePool(xs, a): return xs[0].mean(dim=tuple(range(2, xs[0].dim())), keepdim=True)
    @staticmethod
    def _op_Conv(xs, a):
        x, w = xs[0], xs[1]; b = xs[2] if len(xs) > 2 else None
        rank = w.dim() - 2
        x = F.pad(x, _pads(a.get("pads"), rank))
        conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
        return conv(x, w, b, stride=[int(s) for s in a.get("strides", [1] * rank)],
                    dilation=[int(d) for d in a.get("dilations", [1] * rank)], groups=int(a.get("group", 1)))
    @staticmethod
    def _op_MaxPool(xs, a):
        k = [int(v) for v in a["kernel_shape"]]; rank = len(k)
        x = F.pad(xs[0], _pads(a.get("pads"), rank), value=float("-inf"))
        pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[rank]
        return pool(x, k, stride=[int(s) for s in a.get("strides", k)], ceil_mode=bool(a.get("ceil_mode", 0)))
    @staticmethod
    def _op_AveragePool(xs, a):
        k = [int(v) for v in a["kernel_shape"]]; rank = len(k)
        x = F.pad(xs[0], _pads(a.get("pads"), rank))
        pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[rank]
        return pool(x, k, stride=[int(s) for s in a.get("strides", k)], ceil_mode=bool(a.get("ceil_mode", 0)))
