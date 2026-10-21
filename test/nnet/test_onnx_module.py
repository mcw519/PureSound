"""What `OnnxModule` must do: evaluate a small ONNX graph exactly as onnxruntime
does, keep the input differentiable, refuse graphs it cannot run, and never
turn an exactly-zero value (a silent frame's STFT bin) into a NaN gradient.
"""

import os

import numpy as np
import pytest
import torch

from puresound.nnet.lobe.onnx_module import OnnxModule

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

DNSMOS = os.path.expanduser("~/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx")


def _graph(nodes, inputs, outputs, initializers, path):
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        nodes,
        "t",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        initializer=[numpy_helper.from_array(v, n) for n, v in initializers],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _conv_mlp(tmp_path):
    """Conv -> Relu -> MaxPool -> Flatten -> Gemm -> Sqrt(x*x + 1): the op mix a
    quality judge is made of."""
    from onnx import helper

    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 1, 3, 3)).astype(np.float32) * 0.3
    b = rng.standard_normal(4).astype(np.float32) * 0.1
    g = rng.standard_normal((4 * 4 * 4, 3)).astype(np.float32) * 0.1
    gb = rng.standard_normal(3).astype(np.float32)
    one = np.ones((), np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["p"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "g", "gb"], ["y"]),
        helper.make_node("Mul", ["y", "y"], ["yy"]),
        helper.make_node("Add", ["yy", "one"], ["yy1"]),
        helper.make_node("Sqrt", ["yy1"], ["out"]),
    ]
    return _graph(nodes, [("x", [1, 1, 8, 8])], [("out", [1, 3])], [("w", w), ("b", b), ("g", g), ("gb", gb), ("one", one)], tmp_path / "mlp.onnx")


def test_matches_onnxruntime_and_keeps_the_input_differentiable(tmp_path):
    path = _conv_mlp(tmp_path)
    module = OnnxModule(path)
    x = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    ref = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    xg = x.clone().requires_grad_(True)
    out = module(xg)
    assert np.allclose(out.detach().numpy(), ref, atol=1e-5)
    out.sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    assert all(not p.requires_grad for p in module.buffers())
    assert len(list(module.parameters())) == 0


def test_unsupported_op_is_refused_at_load(tmp_path):
    from onnx import helper

    path = _graph([helper.make_node("Softmax", ["x"], ["y"], axis=1)], [("x", [1, 4])], [("y", [1, 4])], [], tmp_path / "sm.onnx")
    with pytest.raises(ValueError, match="Softmax"):
        OnnxModule(path)


def test_sqrt_and_log_at_exactly_zero_give_finite_gradients(tmp_path):
    """A Sqrt or Log of 0 has an infinite derivative; one such element would
    poison every row of the batch. The floor keeps the forward value and
    bounds the gradient."""
    from onnx import helper

    path = _graph(
        [helper.make_node("Sqrt", ["x"], ["s"]), helper.make_node("Log", ["s"], ["y"])],
        [("x", [4])], [("y", [4])], [], tmp_path / "sl.onnx",
    )
    module = OnnxModule(path)
    x = torch.tensor([0.0, 1e-9, 1.0, 4.0], requires_grad=True)
    y = module(x)
    assert torch.isfinite(y[1:]).all()
    y[1:].sum().backward()
    assert torch.isfinite(x.grad).all()
    # the floor is below anything a judge sees: values above it are untouched
    ref = torch.log(torch.sqrt(torch.tensor([1e-9, 1.0, 4.0])))
    assert torch.allclose(y[1:].detach(), ref)


@pytest.mark.skipif(not os.path.isfile(DNSMOS), reason="DNSMOS model not downloaded")
def test_dnsmos_judge_matches_onnxruntime():
    """The graph this executor exists for: 9.01 s in, three raw scores out."""
    module = OnnxModule(DNSMOS)
    x = torch.randn(2, 144160, generator=torch.Generator().manual_seed(2)) * 0.05
    ref = ort.InferenceSession(DNSMOS, providers=["CPUExecutionProvider"]).run(None, {module.input_names[0]: x.numpy()})[0]
    out = module(x.clone().requires_grad_(True))
    assert out.shape == (2, 3)
    assert np.allclose(out.detach().numpy(), ref, atol=1e-4)
