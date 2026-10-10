"""Re-export every catalog artifact, validate parity, then update hashes in place.

Run from the repository root with a locally built SSM library:
  uv run python model_zoo/reexport.py --native-library /path/libpuresound_ssm.so
All graphs are staged first. Primary graphs use standard ONNX operators; Mamba
models also get a hashed native companion. A variant whose sidecar records int8
quantization is re-quantized. Checkpoint weights are unchanged.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puresound.config import load_recipe  # noqa: E402
from puresound.inference import ModelZoo  # noqa: E402
from puresound.streaming import StreamingOrt, export_streaming_dpcrn_onnx  # noqa: E402
from puresound.system.onset_guard import OnsetGuard  # noqa: E402
from puresound.system.postprocess import Postprocessor  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def error(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise AssertionError("export has incorrect output shape or nonfinite values")
    difference = actual - reference
    return {"max_abs_error": float(np.max(np.abs(difference))),
            "nrms": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12))}


def stream_check(old_path, old_manifest, new_path, library, samples):
    manifest = json.loads(new_path.with_suffix(".json").read_text())
    previous = json.loads(Path(old_manifest).read_text())
    variants = {
        "original": StreamingOrt(old_path, manifest_path=old_manifest, provider="cpu", native_ssm="off",
                                 intra_op_num_threads=1),
        "portable": StreamingOrt(new_path, provider="cpu", native_ssm="off"),
    }
    # Each new graph is held to the release in the same execution mode. An int8
    # graph re-rounds its activations, so its native and portable paths differ
    # by more than a float tolerance even though each is reproducible.
    reference_of = {"portable": "original"}
    if manifest["cpu_optimization"].get("native_graph"):
        variants["native"] = StreamingOrt(new_path, provider="cpu", native_ssm="required", native_library=library)
        reference_of["native"] = "original"
        if (previous.get("cpu_optimization") or {}).get("native_graph"):
            variants["original_native"] = StreamingOrt(old_path, manifest_path=old_manifest, provider="cpu",
                                                       native_ssm="required", native_library=library)
            reference_of["native"] = "original_native"
    # Check every state port after nonzero input and across the startup gate.
    states = {name: {key: np.zeros(shape, np.float32) for key, shape in manifest["state_shapes"].items()}
              for name in variants}
    rng = np.random.default_rng(19)
    for _ in range(40):
        frame = rng.normal(0, 0.1, (1, manifest["freq_bins"], 2)).astype(np.float32)
        outputs = {}
        for name, runtime in variants.items():
            outputs[name] = runtime.session.run(None, {"noisy_frame": frame, **states[name]})
            states[name].update(zip(manifest["state_input_names"],
                                    outputs[name][1 + len(manifest["extra_output_names"]):]))
        for name, reference_name in reference_of.items():
            for actual, expected in zip(outputs[name], outputs[reference_name]):
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    results = {name: {"rtf_repeats": []} for name in variants}
    first_outputs = {}
    for repeat in range(5):  # two warmups, three interleaved measured repeats
        for name, runtime in variants.items():
            runtime.reset()
            start = time.perf_counter()
            output = np.concatenate([runtime.process_samples(samples), runtime.flush()])
            elapsed = time.perf_counter() - start
            if repeat >= 2:
                results[name]["rtf_repeats"].append(elapsed / (len(samples) / runtime.sample_rate))
            if repeat == 0:
                first_outputs[name] = output
    for name, reference_name in reference_of.items():
        results[name].update(error(first_outputs[name], first_outputs[reference_name]))
        if results[name]["nrms"] > 1e-4:
            raise AssertionError("streaming audio differs from the release reference")
    long_outputs = {}
    for name, runtime in variants.items():
        results[name]["median_rtf"] = float(np.median(results[name]["rtf_repeats"]))
        # Longer stateful check outside timing. Same sample repeated six times.
        long_samples = np.tile(samples, 6)
        runtime.reset()
        pieces = [runtime.process_samples(long_samples[i:i + 317]) for i in range(0, len(long_samples), 317)]
        long_outputs[name] = np.concatenate([*pieces, runtime.flush()])
        runtime.reset()
        reset = np.concatenate([runtime.process_samples(samples), runtime.flush()])
        np.testing.assert_array_equal(reset, first_outputs[name])
        results[name]["reset_bit_equal"] = True
    for name, reference_name in reference_of.items():
        results[name]["long_stream"] = error(long_outputs[name], long_outputs[reference_name])
        if results[name]["long_stream"]["nrms"] > 1e-4:
            raise AssertionError("long streaming audio differs from the release reference")
    return results


def embedding_check(old_path, new_path):
    old, new = session(old_path), session(new_path)
    rng = np.random.default_rng(17)
    results = []
    for batch, length in [(1, 16000), (1, 37360), (1, 80000), (2, 32000)]:
        samples = rng.normal(0, 0.05, (batch, length)).astype(np.float32)
        expected = old.run(None, {"Audio": samples})[0]
        actual = new.run(None, {"Audio": samples})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        results.append({"batch": batch, "samples": length, **error(actual, expected)})
    return {"dynamic_length_and_batch": results}


def reexport(library, report_path):
    catalog_path = ROOT / "model_zoo/catalog.yaml"
    catalog_text = catalog_path.read_text()
    raw = yaml.safe_load(catalog_text)
    sample_path = ROOT / "puresound/web/static/samples/noisy-speech.wav"
    samples, sr = sf.read(sample_path, dtype="float32")
    assert sr == 16000 and samples.ndim == 1
    report = {"ort_version": ort.__version__, "threads": 1, "provider": "CPUExecutionProvider",
              "hardware": {"machine": platform.machine(), "cpu": platform.processor()},
              "sample": str(sample_path.relative_to(ROOT)), "sample_sha256": digest(sample_path),
              "audio_seconds": len(samples) / sr, "warmup": 2, "repeats": 3,
              "timing": "interleaved; STFT/model/OLA/flush included; load and warmup excluded",
              "long_stream": {"seconds": len(samples) / sr * 6, "chunk_samples": 317,
                              "input": "six repetitions of the same noisy-speech sample"},
              "weights_changed": False, "models": {}}
    torch.set_num_threads(1)
    staged = []
    with tempfile.TemporaryDirectory(prefix="puresound-zoo-") as temp:
        for model, artifact in [(m, a) for m in raw["models"] for a in m["artifacts"]]:
            label = model["id"] if artifact["variant"] == "default" else f"{model['id']}:{artifact['variant']}"
            destination = ROOT / artifact["path"]
            output = Path(temp) / model["id"] / artifact["variant"] / destination.name
            output.parent.mkdir(parents=True)
            print(f"Exporting {label}", flush=True)
            torch.manual_seed(0)
            if artifact["processor"] == "stft_frame_ort":
                previous = json.loads((ROOT / artifact["manifest"]).read_text())
                config = yaml.safe_load((ROOT / model["source_config"]).read_text())
                inter_type = config["model"]["backbone"]["backbone_args"].get("inter_type", "lstm")
                guard_config = previous.get("onset_guard")
                guard = OnsetGuard.from_manifest(guard_config) if guard_config else None
                quantized = (previous.get("quantization") or {}).get("type") == "dynamic_int8"
                manifest = export_streaming_dpcrn_onnx(
                    ROOT / model["source_config"], ROOT / model["source_checkpoint"], output,
                    postprocess=Postprocessor(dry_blend=float(previous["recommended_inference"]["dry_blend"])),
                    onset_guard=guard, optimization="cpu" if inter_type.startswith("mamba") else "portable",
                    native_library=library, quantization="int8" if quantized else "none",
                )
                validation = stream_check(destination, ROOT / artifact["manifest"], output, library, samples)
                manifest["onnx_path"] = artifact["path"]
                output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
            else:
                from egs.speaker_embedding.main import export_onnx

                recipe = load_recipe(ROOT / model["source_config"], expected_task="speaker_embedding", expected_purpose="train")
                export_onnx(SimpleNamespace(inference_sr=16000, pretrained_ckpt_path=str(ROOT / model["source_checkpoint"]),
                                            onnx_path=output, optimization="portable"), recipe)
                validation = embedding_check(destination, output)
            report["models"][label] = {
                "original_sha256": artifact["sha256"], "sha256": digest(output),
                "checkpoint_sha256": digest(ROOT / model["source_checkpoint"]),
                "validation": validation,
            }
            print(f"Validated {label}: {json.dumps(validation)}", flush=True)
            staged.append((artifact, output, destination))
        # Promote only after every export and its state/audio checks succeed.
        for artifact, output, destination in staged:
            shutil.copy2(output, destination)
            if artifact.get("manifest"):
                shutil.copy2(output.with_suffix(".json"), ROOT / artifact["manifest"])
                manifest = json.loads(output.with_suffix(".json").read_text())
                companion = manifest["cpu_optimization"].get("native_graph")
                if companion:
                    shutil.copy2(output.with_name(companion), destination.with_name(companion))
            catalog_text = catalog_text.replace(artifact["sha256"], digest(output), 1)
        catalog_path.write_text(catalog_text)
    validation = ModelZoo.default().validate()
    report["catalog_validation"] = {"models": validation.models, "artifacts": validation.artifacts, "ok": validation.ok}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Re-exported and validated {validation.models} models; report: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "model_zoo/benchmarks/onnx_reexport_20261004.json")
    args = parser.parse_args()
    reexport(args.native_library.resolve(), args.report.resolve())
