"""Dynamic int8 quantization of exported streaming graphs.

The LSTM and matrix-product weights are stored as int8 and their activations
are quantized per tensor at run time. The float graph is validated first; the
int8 graph can only be held to a sanity bound, since its error is the point.
Whether that error is acceptable is decided by the gate, not here.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

QUANTIZED_OPS = ("LSTM", "MatMul", "Gemm")

#: Relative RMS error of the enhanced frame over the export rollout that an int8
#: graph may show against the float model. A graph past it is broken, not
#: merely coarser: NS v3's int8 graph measures 0.038 on the export's noise frames.
INT8_MAX_RELATIVE_ERROR = 0.1


def quantize_int8(path: str | Path) -> dict[str, Any]:
    """Replace the float graph at ``path`` with its dynamic int8 version."""
    from onnx import TensorProto
    from onnxruntime.quantization import QuantType, quantize_dynamic

    path = Path(path)
    staged = path.with_name(f".{path.name}.int8.tmp")
    try:
        # A custom operator's outputs carry no inferred type; they are float.
        quantize_dynamic(path, staged, op_types_to_quantize=list(QUANTIZED_OPS),
                         weight_type=QuantType.QInt8,
                         extra_options={"DefaultTensorType": TensorProto.FLOAT})
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return {
        "type": "dynamic_int8",
        "op_types": list(QUANTIZED_OPS),
        "weight_type": "QInt8",
        "activations": "uint8, per tensor, computed at run time",
    }


def check_int8_rollout(label, actual, expected, names) -> float:
    """Every output finite and the enhanced frame within the sanity bound."""
    for index, frame in enumerate(actual):
        for name, value in zip(names, frame):
            if not np.isfinite(value).all():
                raise AssertionError(f"{label} output {name!r} is not finite at frame {index}")
    got = np.concatenate([frame[0].reshape(-1) for frame in actual])
    want = np.concatenate([frame[0].reshape(-1) for frame in expected])
    error = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12))
    if error > INT8_MAX_RELATIVE_ERROR:
        raise AssertionError(
            f"{label} output {names[0]!r} relative error {error:.3g} exceeds "
            f"{INT8_MAX_RELATIVE_ERROR} against PyTorch"
        )
    return error
