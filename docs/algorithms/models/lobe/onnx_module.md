# puresound.nnet.lobe.onnx_module

繁體中文版本：[onnx_module.zh-TW.md](onnx_module.zh-TW.md)

Runs a small ONNX graph as a differentiable torch module. A published judge
(a quality predictor such as DNSMOS, a VAD) usually ships as an ONNX file;
onnxruntime scores it but gives no gradient. Executing the same graph with
torch ops turns it into a loss term: every weight is a frozen buffer, the
input keeps its autograd history.

## Class: `OnnxModule`

```python
OnnxModule(path: str | Path)
forward(*inputs: Tensor) -> Tensor | list[Tensor]
```

`input_names` / `output_names` are the graph's; initializers become
non-persistent buffers with `requires_grad` off, so the module has no
parameters and nothing of it is saved in a checkpoint. Nodes run in the
graph's order, so a `Reshape` whose shape is an initializer, or a `Slice`
whose bounds are, reads them as constants.

Supported ops: `Conv`, `MaxPool`, `AveragePool`, `GlobalMaxPool`,
`GlobalAveragePool`, `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Sqrt`,
`Pow`, `Log`, `Exp`, `Max`, `Min`, `Clip`, `Relu`, `Sigmoid`, `Tanh`,
`ReduceMax`, `ReduceMean`, `Slice`, `Reshape`, `Flatten`, `Concat`,
`Transpose`, `Unsqueeze`, `Squeeze`, `Identity`. A graph with any other op
raises `ValueError` naming the op at load time, not at the first forward.

`Sqrt` and `Log` floor their input at `GRAD_FLOOR = 1e-12`: their derivative
at 0 is infinite, and one exactly-zero STFT bin (a silent frame) would
otherwise turn the whole batch's gradient into NaN. The floor is far below
any value such judges see and leaves the forward unchanged.

## Use

[`DNSMOSLoss`](../../losses/dnsmos.md) wraps the DNSMOS P.835 judge with it.
The module has no notion of audio: it is the graph, nothing more.
