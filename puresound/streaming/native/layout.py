"""Remove copies caused solely by transposing singleton axes in fixed graphs."""

from pathlib import Path


def eliminate_singleton_transposes(path: str | Path) -> int:
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            shapes[value.name] = [dim.dim_value for dim in dims]
    names = {v.name for v in model.graph.initializer}
    names.update(name for node in model.graph.node for name in [*node.input, *node.output])
    count = 0
    for node in model.graph.node:
        if node.domain or node.op_type != "Transpose" or node.input[0] not in shapes:
            continue
        shape = shapes[node.input[0]]
        permutation = next((list(a.ints) for a in node.attribute if a.name == "perm"),
                           list(reversed(range(len(shape)))))
        if [axis for axis in permutation if shape[axis] != 1] != [
            axis for axis, size in enumerate(shape) if size != 1
        ]:
            continue
        # Only singleton axes move: element order is unchanged. Reshape is a
        # view in ORT, whereas Transpose may allocate and copy even in this case.
        name = f"puresound_singleton_shape_{count}"
        while name in names:
            name += "_"
        names.add(name)
        model.graph.initializer.append(numpy_helper.from_array(
            np.array([shape[axis] for axis in permutation], dtype=np.int64), name,
        ))
        replacement = helper.make_node("Reshape", [node.input[0], name], list(node.output), name=node.name)
        node.CopyFrom(replacement)
        count += 1
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return count
