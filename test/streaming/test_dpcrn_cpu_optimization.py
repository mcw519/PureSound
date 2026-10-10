"""Stateful parity and deployment checks for the optional CPU export path."""

import copy
import importlib.util
import json
import os
import platform
import shutil
import subprocess

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest
import torch
import yaml

from puresound.streaming import StreamingOrt, export_streaming_dpcrn_onnx, load_streaming_dpcrn_model
from puresound.streaming.dpcrn import StreamingDpcrnFrameModel
from puresound.streaming.native.build import build_library
from puresound.streaming.native.layout import eliminate_singleton_transposes
from puresound.streaming.native.runtime import create_session
from puresound.streaming.native.ssm import DOMAIN, fused_ssm_step
from test.streaming.test_dpcrn_streaming import MAMBA_CONFIG, CAUSAL_CONFIG


def _config(root, **options):
    config = yaml.safe_load(MAMBA_CONFIG.read_text())
    config["model"]["backbone"]["backbone_args"].update(
        channels=[2, 8, 12, 16], rnn_hidden=12,
        mamba_args={"d_state": 5, "d_conv": 3, "expand": 1}, **options,
    )
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("options", [
    {"delay": [0, 0, 0]},
    {"inter_type": "mamba_context", "mamba_context": {"n_bands": 8}},
    {"band_bottleneck": {"n_bands": 32}, "df_head": {"bins": 64, "order": 3, "hidden": 8}},
    {"inter_type": "lstm", "delay": [0, 0, 0]},
    {"inter_type": "lstm", "delay": [1, 1, 1]},
    {"inter_type": "lstm", "band_bottleneck": {"n_bands": 32},
     "df_head": {"bins": 64, "order": 3, "hidden": 8}},
])
def test_frequency_layout_preserves_every_state_across_warmup(tmp_path, options):
    torch.manual_seed(31)
    reference = load_streaming_dpcrn_model(_config(tmp_path, **options)).eval()
    if reference.df_head is not None:
        with torch.no_grad():
            reference.df_head.out.weight.normal_(0, 0.1)
            reference.df_head.out.bias.normal_(0, 0.1)
    optimized = copy.deepcopy(reference)
    optimized.enable_cpu_optimization(native_ssm=options.get("inter_type") != "lstm")
    state = [tensor + torch.randn_like(tensor) * 0.02
             for tensor in reference.initial_state_tensors(1)]
    # Exercise both the startup gate and warmed-up, nonzero state.
    if "counter" in reference.state_input_names:
        state[reference.state_input_names.index("counter")] *= 0
    other_state = [s.clone() for s in state]
    with torch.no_grad():
        for _ in range(40):
            frame = torch.randn(1, 257, 2)
            expected = reference(frame, *state)
            actual = optimized(frame, *other_state)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
            state = list(expected[1 + len(reference.extra_output_names):])
            other_state = list(actual[1 + len(reference.extra_output_names):])


def test_lstm_supports_portable_but_refuses_ssm_fusion():
    frame = load_streaming_dpcrn_model(CAUSAL_CONFIG)
    frame.enable_cpu_optimization()
    # The fused projections are derived from the checkpoint, never saved into it.
    assert frame.inter_lstm_weight_0.shape[1] == sum(
        frame.blocks[0].inter_rnn.rnn.weight_ih_l0.shape[1:] + frame.blocks[0].inter_rnn.rnn.weight_hh_l0.shape[1:])
    assert not any(key.startswith("inter_lstm_") for key in frame.state_dict())
    with pytest.raises(ValueError, match="Mamba inter"):
        frame.enable_cpu_optimization(native_ssm=True)


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    import sys

    if not sys.platform.startswith("linux") or not shutil.which("g++"):
        pytest.skip("optional native library requires Linux and g++")
    root = tmp_path_factory.mktemp("ssm-library")
    return build_library(root / "libpuresound_ssm.so")


@pytest.fixture(scope="module")
def exported(tmp_path_factory, native_library):
    root = tmp_path_factory.mktemp("cpu-dpcrn")
    config = _config(root, inter_type="mamba_context", mamba_context={"n_bands": 8})
    torch.manual_seed(7)
    frame = load_streaming_dpcrn_model(config).eval()
    checkpoint = root / "weights.ckpt"
    torch.save({"state_dict": frame.system_model.state_dict()}, checkpoint)
    path = root / "model.onnx"
    manifest = export_streaming_dpcrn_onnx(config, checkpoint, path,
                                          optimization="cpu", native_library=native_library)
    reference = root / "reference.onnx"
    export_streaming_dpcrn_onnx(config, checkpoint, reference)
    return path, reference, manifest


def test_native_portable_and_original_keep_all_ports_and_states(exported, native_library):
    path, reference_path, manifest = exported
    portable = StreamingOrt(path, provider="cpu", native_ssm="off")
    native = StreamingOrt(path, provider="cpu", native_ssm="required", native_library=native_library)
    reference = StreamingOrt(reference_path, provider="cpu", intra_op_num_threads=1)
    assert native.native_ssm_enabled and native.execution_path.name == "model.native.onnx"
    assert not portable.native_ssm_enabled
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert all(node.domain in {"", "ai.onnx"} for node in graph.graph.node)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
    assert manifest["streaming_delay_frames"] == 3
    names = manifest["input_names"]
    assert [port.name for port in native.session.get_inputs()] == names
    states = [{name: np.zeros(shape, np.float32) for name, shape in manifest["state_shapes"].items()}
              for _ in range(3)]
    rng = np.random.default_rng(33)
    for _ in range(40):
        frame = rng.normal(0, 0.1, (1, 257, 2)).astype(np.float32)
        results = []
        for runtime, state in zip((reference, portable, native), states):
            output = runtime.session.run(None, {"noisy_frame": frame, **state})
            results.append(output)
            state.update(zip(manifest["state_input_names"], output[1 + len(manifest["extra_output_names"]):]))
        for actual in results[1:]:
            for expected_tensor, actual_tensor in zip(results[0], actual):
                np.testing.assert_allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-4)


def test_long_stream_chunking_reset_and_independent_sessions(exported, native_library):
    path, reference_path, _ = exported
    native = StreamingOrt(path, provider="cpu", native_ssm="required", native_library=native_library)
    other = StreamingOrt(path, provider="cpu", session=native.selection)
    assert other.native_ssm_enabled and other.session is native.session
    samples = np.random.default_rng(14).normal(0, 0.05, 16000 * 6).astype(np.float32)
    original = StreamingOrt(reference_path, provider="cpu", intra_op_num_threads=1)
    expected = np.concatenate([original.process_samples(samples), original.flush()])
    parts = [native.process_samples(samples[i:i + 317]) for i in range(0, len(samples), 317)]
    actual = np.concatenate([*parts, native.flush()])
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    native.reset()
    reset = np.concatenate([native.process_samples(samples), native.flush()])
    independent = np.concatenate([other.process_samples(samples), other.flush()])
    np.testing.assert_array_equal(reset, actual)
    np.testing.assert_array_equal(independent, actual)
    assert np.isfinite(actual).all()


def test_missing_and_bad_library_fall_back_or_raise(exported, native_library, tmp_path, monkeypatch):
    path, _, _ = exported
    monkeypatch.delenv("PURESOUND_ORT_SSM_LIBRARY", raising=False)
    default = StreamingOrt(path, provider="cpu")
    assert default.native_fallback_reason == "no native library specified"
    assert not default.native_ssm_enabled
    with pytest.raises(RuntimeError, match="no native library"):
        StreamingOrt(path, provider="cpu", native_ssm="required")
    bad = tmp_path / "missing.so"
    with pytest.warns(RuntimeWarning, match="using portable ONNX"):
        fallback = StreamingOrt(path, provider="cpu", native_library=bad)
    assert fallback.execution_path == path and not fallback.native_ssm_enabled
    with pytest.raises(RuntimeError, match="could not be loaded"):
        StreamingOrt(path, provider="cpu", native_ssm="required", native_library=bad)
    monkeypatch.setenv("PURESOUND_ORT_SSM_LIBRARY", str(native_library))
    enabled = StreamingOrt(path, provider="cpu")
    assert enabled.native_ssm_enabled
    with pytest.raises(ValueError, match="positive"):
        StreamingOrt(path, provider="cpu", intra_op_num_threads=0)
    with pytest.raises(ValueError, match="injected session"):
        StreamingOrt(path, provider="cpu", session=enabled.session, native_ssm="required")


_SSM_INPUTS = ["dt", "u", "b", "c", "h", "a", "skip", "z"]


def _ssm_operator(tmp_path, native_library):
    dims = [["N", "D"], ["N", "D"], ["N", "S"], ["N", "S"],
            ["N", "D", "S"], ["D", "S"], ["D"], ["N", "D"]]
    graph = helper.make_graph(
        [helper.make_node("FusedSsmStep", _SSM_INPUTS, ["next_h", "y"], domain=DOMAIN)], "ssm",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, dim) for name, dim in zip(_SSM_INPUTS, dims)],
        [helper.make_tensor_value_info("next_h", TensorProto.FLOAT, dims[4]),
         helper.make_tensor_value_info("y", TensorProto.FLOAT, dims[0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)])
    model.ir_version = 9
    path = tmp_path / "ssm.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.register_custom_ops_library(str(native_library))
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


# 16 is Mamba's default state width and takes the vector path; the rest the general loop.
@pytest.mark.parametrize("shape", [(3, 7, 5), (24, 128, 16), (1, 1, 1), (2, 3, 17)])
def test_native_operator_handles_general_sizes_and_checks_shapes(tmp_path, native_library, shape):
    n, d, s = shape
    names = _SSM_INPUTS
    session = _ssm_operator(tmp_path, native_library)
    rng = np.random.default_rng(5)
    shapes = [(n, d), (n, d), (n, s), (n, s), (n, d, s), (d, s), (d,), (n, d)]
    values = [rng.normal(0, 0.2, size).astype(np.float32) for size in shapes]
    values[0] = np.abs(values[0])
    values[5] = -np.exp(values[5])
    before = [v.copy() for v in values]
    actual = session.run(None, dict(zip(names, values)))
    expected = fused_ssm_step(*[torch.from_numpy(v) for v in values])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b.numpy(), rtol=2e-5, atol=2e-6)
    for a, b in zip(values, before):
        np.testing.assert_array_equal(a, b)
    bad = dict(zip(names, values))
    bad["b"] = np.zeros((n, s + 1), np.float32)
    with pytest.raises(Exception, match="incompatible"):
        session.run(None, bad)


def test_native_operator_is_accurate_across_the_exp_range(tmp_path, native_library):
    # Decays from 1 down to far below float range, and gates saturating both ways.
    n, d, s = 4, 32, 16
    rng = np.random.default_rng(8)
    dt = rng.uniform(0, 30, (n, d)).astype(np.float32)
    a = -np.exp(rng.uniform(-6, 2, (d, s))).astype(np.float32)
    z = rng.uniform(-120, 120, (n, d)).astype(np.float32)
    values = [dt, rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, s)), rng.normal(0, 1, (n, s)),
              rng.normal(0, 1, (n, d, s)), a, rng.normal(0, 1, (d,)), z]
    values = [np.asarray(v, np.float32) for v in values]
    actual = _ssm_operator(tmp_path, native_library).run(None, dict(zip(_SSM_INPUTS, values)))
    expected = fused_ssm_step(*[torch.from_numpy(v).double() for v in values])
    for got, want in zip(actual, expected):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)


def test_companion_graph_cannot_escape_its_directory(tmp_path, native_library):
    with pytest.raises(ValueError, match="sibling filename"):
        create_session(tmp_path / "model.onnx", {"cpu_optimization": {"native_graph": "../bad.onnx"}},
                       ["CPUExecutionProvider"], native_library=native_library)


def test_wrong_native_digest_cannot_silently_run_different_weights(exported, native_library, tmp_path):
    path, _, manifest = exported
    altered = copy.deepcopy(manifest)
    altered["cpu_optimization"]["native_sha256"] = "0" * 64
    sidecar = tmp_path / "wrong-hash.json"
    sidecar.write_text(json.dumps(altered))
    with pytest.warns(RuntimeWarning, match="SHA256 mismatch"):
        runtime = StreamingOrt(path, manifest_path=sidecar, provider="cpu", native_library=native_library)
    assert not runtime.native_ssm_enabled
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        StreamingOrt(path, manifest_path=sidecar, provider="cpu", native_library=native_library,
                     native_ssm="required")


def test_a_companion_graph_is_only_loaded_against_a_recorded_digest(tmp_path):
    import hashlib

    graph = b"not a graph"
    (tmp_path / "model.native.onnx").write_bytes(graph)

    def load(**recorded):
        manifest = {"cpu_optimization": {"native_graph": "model.native.onnx", **recorded}}
        create_session(tmp_path / "model.onnx", manifest, ["CPUExecutionProvider"],
                       native_ssm="required", native_library=tmp_path / "missing.so")

    with pytest.raises(RuntimeError, match="missing digest"):
        load()
    # A correct digest is accepted whatever its case; loading then fails on the
    # graph itself, which is the step after the integrity check.
    with pytest.raises(RuntimeError, match="could not be loaded") as refused:
        load(native_sha256=hashlib.sha256(graph).hexdigest().upper())
    assert "SHA256" not in str(refused.value)


def test_portable_export_works_without_compiler_or_library(tmp_path):
    config = _config(tmp_path)
    frame = load_streaming_dpcrn_model(config).eval()
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": frame.system_model.state_dict()}, checkpoint)
    path = tmp_path / "portable.onnx"
    manifest = export_streaming_dpcrn_onnx(config, checkpoint, path, optimization="portable")
    assert "native_graph" not in manifest["cpu_optimization"]
    assert json.loads(path.with_suffix(".json").read_text())["cpu_optimization"]["batch_size"] == 1
    with pytest.raises(ValueError, match="native_library"):
        export_streaming_dpcrn_onnx(config, checkpoint, path, optimization="cpu")


def test_vectorized_lstm_export_preserves_nonzero_states(tmp_path):
    config = _config(tmp_path, inter_type="lstm")
    model = load_streaming_dpcrn_model(config).eval()
    checkpoint = tmp_path / "lstm.ckpt"
    torch.save({"state_dict": model.system_model.state_dict()}, checkpoint)
    path = tmp_path / "lstm.onnx"
    manifest = export_streaming_dpcrn_onnx(config, checkpoint, path, optimization="portable")
    runtime = StreamingOrt(path, provider="cpu")
    assert manifest["cpu_optimization"]["inter_lstm"] == "vectorized_gates"
    # Only the two intra (frequency) LSTMs remain sequence operators.
    assert sum(n.op_type == "LSTM" for n in onnx.load(path).graph.node) == 2
    states = model.initial_state_tensors(1)
    rng = np.random.default_rng(9)
    inputs = {name: rng.normal(0, 0.01, tensor.shape).astype(np.float32)
              for name, tensor in zip(model.state_input_names, states)}
    inputs["counter"][:] = 3  # already past the startup gate
    for _ in range(25):
        frame = rng.normal(0, 0.1, (1, 257, 2)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(frame), *[torch.from_numpy(inputs[n]) for n in model.state_input_names])
        actual = runtime.session.run(None, {"noisy_frame": frame, **inputs})
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b.numpy(), rtol=2e-4, atol=2e-5)
        inputs.update(zip(model.state_input_names, actual[1 + len(model.extra_output_names):]))


def test_singleton_rewrite_keeps_frequency_channel_transposes(tmp_path):
    path = tmp_path / "layout.onnx"
    nodes = [helper.make_node("Transpose", ["x"], ["view"], perm=[1, 0, 2]),
             helper.make_node("Transpose", ["view"], ["y"], perm=[0, 2, 1])]
    model = helper.make_model(helper.make_graph(nodes, "layout", [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5]),
    ], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5, 1])]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    assert eliminate_singleton_transposes(path) == 2
    # A transpose that exchanges two non-singleton axes must keep its copy.
    model.graph.node[1].attribute[0].ints[:] = [2, 1, 0]
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 5
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 3
    onnx.save(model, path)
    assert eliminate_singleton_transposes(path) == 1
    assert onnx.load(path).graph.node[1].op_type == "Transpose"
    x = np.arange(15, dtype=np.float32).reshape(1, 3, 5)
    session = create_session(path, {}, ["CPUExecutionProvider"], intra_op_num_threads=1).session
    np.testing.assert_array_equal(session.run(None, {"x": x})[0], x.transpose(1, 0, 2).transpose(2, 1, 0))


def test_native_library_preserves_process_denormal_handling(exported, native_library):
    path, _, _ = exported
    # Compare bits: floating-point comparisons themselves can be affected by
    # flush-to-zero. The local linker must not install crtfastmath globally.
    tiny = np.array([1], dtype=np.uint32).view(np.float32)
    assert (tiny * np.float32(1)).view(np.uint32)[0] == 1
    StreamingOrt(path, provider="cpu", native_library=native_library, native_ssm="required")
    assert (tiny * np.float32(1)).view(np.uint32)[0] == 1


def _mamba_checkpoint(root):
    config = _config(root, inter_type="mamba_context", mamba_context={"n_bands": 8})
    torch.manual_seed(7)
    frame = load_streaming_dpcrn_model(config).eval()
    checkpoint = root / "weights.ckpt"
    torch.save({"state_dict": frame.system_model.state_dict()}, checkpoint)
    return config, checkpoint


def test_export_rejects_a_kernel_with_a_wrong_state_update(tmp_path, native_library):
    # The defect only shows once the recurrence carries nonzero state, which a
    # single frame from the zero initial state (and the look-ahead gate) hides.
    source = tmp_path / "native"
    shutil.copytree(Path(build_library.__code__.co_filename).parent, source,
                    ignore=shutil.ignore_patterns("__pycache__"))
    kernel = source / "fused_ssm.cc"
    text = kernel.read_text()
    assert text.count("h[base + k] * decay") == 1
    kernel.write_text(text.replace("h[base + k] * decay", "h[base + k] * 0.5f"))
    spec = importlib.util.spec_from_file_location("broken_build", source / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    broken = module.build_library(tmp_path / "libbroken.so")
    config, checkpoint = _mamba_checkpoint(tmp_path)
    with pytest.raises(AssertionError, match="native ONNX output"):
        export_streaming_dpcrn_onnx(config, checkpoint, tmp_path / "model.onnx",
                                    optimization="cpu", native_library=broken)


def test_export_holds_the_optimized_graph_to_the_original_model(tmp_path, monkeypatch):
    config, checkpoint = _mamba_checkpoint(tmp_path)
    original = StreamingDpcrnFrameModel._dprnn_frequency_step

    def drifted(self, i, x, h, c):
        y, next_h, next_c = original(self, i, x, h, c)
        return y, next_h * 1.01, next_c

    monkeypatch.setattr(StreamingDpcrnFrameModel, "_dprnn_frequency_step", drifted)
    with pytest.raises(AssertionError, match="exported ONNX output"):
        export_streaming_dpcrn_onnx(config, checkpoint, tmp_path / "model.onnx", optimization="portable")


def test_rebuild_replaces_a_loaded_library_without_rewriting_it(tmp_path):
    if not shutil.which("g++"):
        pytest.skip("requires g++")
    path = build_library(tmp_path / "libpuresound_ssm.so")
    loaded = os.stat(path).st_ino
    assert build_library(path) == path
    # A new inode: processes that mapped the old file keep a consistent copy.
    assert os.stat(path).st_ino != loaded
    assert sorted(p.name for p in tmp_path.iterdir()) == ["libpuresound_ssm.so"]


def test_library_dispatches_its_isa_at_load_time(native_library):
    if platform.machine() != "x86_64" or not shutil.which("nm"):
        pytest.skip("checks the x86-64 ifunc dispatch")
    symbols = subprocess.run(["nm", str(native_library)], capture_output=True, text=True, check=True).stdout
    # A kernel tied to the build host's ISA would SIGILL elsewhere, past any fallback.
    assert any(" i " in line and "ssm_step" in line for line in symbols.splitlines())
    assert all(clone in symbols for clone in ("ssm_step", ".avx2", ".avx512f", ".default"))
    exported_symbols = subprocess.run(["nm", "-D", "--defined-only", str(native_library)],
                                      capture_output=True, text=True, check=True).stdout
    exported_names = [line.split()[-1] for line in exported_symbols.splitlines()]
    assert "RegisterCustomOps" in exported_names
    assert not any("ssm_step" in name for name in exported_names)


def test_int8_export_quantizes_both_graphs_and_records_it(tmp_path, native_library):
    import hashlib

    config, checkpoint = _mamba_checkpoint(tmp_path)
    path = tmp_path / "flash.onnx"
    manifest = export_streaming_dpcrn_onnx(config, checkpoint, path, optimization="cpu",
                                          native_library=native_library, quantization="int8")
    record = manifest["quantization"]
    assert record["type"] == "dynamic_int8" and set(record["relative_error"]) == {"primary", "native"}
    native_path = path.with_name(manifest["cpu_optimization"]["native_graph"])
    for graph in (path, native_path):
        ops = {node.op_type for node in onnx.load(graph).graph.node}
        assert {"DynamicQuantizeLSTM", "MatMulInteger"} <= ops
    # The digest covers the quantized companion, not the float graph it replaced.
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == manifest["cpu_optimization"]["native_sha256"]
    samples = np.random.default_rng(3).normal(0, 0.05, 16000).astype(np.float32)
    for mode, library in (("off", None), ("required", native_library)):
        runtime = StreamingOrt(path, provider="cpu", native_ssm=mode, native_library=library)
        assert np.isfinite(np.concatenate([runtime.process_samples(samples), runtime.flush()])).all()
    with pytest.raises(ValueError, match="CPU deployment"):
        export_streaming_dpcrn_onnx(config, checkpoint, tmp_path / "plain.onnx", quantization="int8")
