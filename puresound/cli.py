"""Dependency-light ``puresound`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from puresound.inference import (
    InferenceError,
    ModelZoo,
    ModelZooError,
    PROVIDER_CHOICES,
    COREML_PROVIDER,
    CPU_PROVIDER,
    CUDA_PROVIDER,
    available_providers,
    load_model,
)


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, item = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("NAME must not be empty")
    return name, item


def _parameter(value: str) -> tuple[str, Any]:
    name, raw = _assignment(value)
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    else:
        try:
            parsed = float(raw) if any(char in raw for char in ".eE") else int(raw)
        except ValueError:
            parsed = raw
    return name, parsed


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _model_payload(model, zoo: ModelZoo) -> dict[str, Any]:
    payload = model.model_dump(mode="json")
    payload["artifacts"] = [
        {
            **artifact.model_dump(mode="json"),
            "resolved_path": str(zoo.path_for(artifact.path)),
            "resolved_manifest": (
                str(zoo.path_for(artifact.manifest)) if artifact.manifest else None
            ),
        }
        for artifact in model.artifacts
    ]
    if model.source_checkpoint:
        payload["resolved_source_checkpoint"] = str(zoo.path_for(model.source_checkpoint))
    if model.source_config:
        payload["resolved_source_config"] = str(zoo.path_for(model.source_config))
    return payload


def _save_outputs(outputs: dict[str, Any], destinations: dict[str, str], sample_rate: int | None) -> None:
    for name, destination in destinations.items():
        if name not in outputs:
            raise ValueError(f"runtime did not return output {name!r}")
        value = outputs[name]
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(value)
        if name == "audio" and path.suffix.lower() not in {".npy", ".npz"}:
            if sample_rate is None:
                raise ValueError("audio output has no sample rate metadata")
            import torch
            from puresound.audio.io import AudioIO

            AudioIO.save(torch.from_numpy(array.astype(np.float32)).view(1, -1), str(path), int(sample_rate))
        else:
            if path.suffix.lower() == ".npz":
                np.savez(str(path), output=array)
            else:
                np.save(str(path), array)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puresound", description="PureSound model-zoo and ONNX inference tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="inspect and validate the local model zoo")
    model_sub = models.add_subparsers(dest="models_command", required=True)
    listing = model_sub.add_parser("list", help="list runnable logical models")
    listing.add_argument("--task", help="filter by task name")
    listing.add_argument("--include-empty", action="store_true", help="include reserved models without artifacts")
    inspect = model_sub.add_parser("inspect", help="show one model and artifact contract")
    inspect.add_argument("model_id")
    validate = model_sub.add_parser("validate", help="verify paths, hashes, sidecars and ONNX I/O")
    validate.add_argument("--no-hash", action="store_true", help="skip SHA256 checks")
    validate.add_argument("--no-graph", action="store_true", help="skip ONNX Runtime graph loading")

    infer = subparsers.add_parser("infer", help="run a catalog model")
    infer.add_argument("model_id")
    infer.add_argument("--variant", help="artifact variant (defaults to the model's default variant)")
    infer.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default="auto",
        help="ONNX Runtime provider: auto, cpu, cuda, coreml, or mps (CoreML alias)",
    )
    infer.add_argument("--input", dest="inputs", action="append", type=_assignment, default=[], metavar="NAME=PATH")
    infer.add_argument("--output", dest="outputs", action="append", type=_assignment, default=[], metavar="NAME=PATH")
    infer.add_argument("--param", dest="parameters", action="append", type=_parameter, default=[], metavar="NAME=VALUE")

    subparsers.add_parser("providers", help="show ONNX Runtime providers available in this environment")

    web = subparsers.add_parser("web", help="serve the local Model Zoo web workspace")
    web.add_argument(
        "--host",
        "--ip",
        dest="host",
        default="127.0.0.1",
        help="bind IP or hostname (default: 127.0.0.1)",
    )
    web.add_argument("--port", type=_port, default=7860, help="bind port (default: 7860)")
    web.add_argument(
        "--static-dir",
        default=None,
        help="override the directory containing the browser client",
    )
    web.add_argument(
        "--allow-local-paths",
        action="store_true",
        help="allow API inputs to reference local paths (for trusted local clients)",
    )
    web.add_argument(
        "--max-upload-mb",
        type=int,
        default=64,
        help="maximum decoded audio upload size in MiB (default: 64)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        zoo = ModelZoo.default()
        if args.command == "models":
            if args.models_command == "list":
                models = zoo.list(task=args.task, runnable_only=not args.include_empty)
                print(_json_dump([_model_payload(model, zoo) for model in models]))
                return 0
            if args.models_command == "inspect":
                print(_json_dump(_model_payload(zoo.get(args.model_id), zoo)))
                return 0
            report = zoo.validate(
                verify_hash=not args.no_hash,
                check_graph=not args.no_graph,
                raise_on_error=False,
            )
            if not report.ok:
                for error in report.errors:
                    print(error, file=sys.stderr)
                return 1
            print(
                f"Model Zoo valid: {report.models} logical models, "
                f"{report.artifacts} ONNX artifacts"
            )
            return 0

        if args.command == "web":
            from puresound.web import run

            run(
                host=args.host,
                port=args.port,
                static_dir=args.static_dir,
                max_upload_bytes=max(1, args.max_upload_mb) * 1024 * 1024,
                allow_local_paths=args.allow_local_paths,
            )
            return 0

        if args.command == "providers":
            installed = available_providers()
            print(
                _json_dump(
                    {
                        "available": installed,
                        "capabilities": {
                            "cpu": CPU_PROVIDER in installed,
                            "cuda": CUDA_PROVIDER in installed,
                            "coreml": COREML_PROVIDER in installed,
                        },
                        "mps_alias": "coreml",
                        "auto_order": ["cuda", "coreml", "cpu"],
                    }
                )
            )
            return 0

        inputs = dict(args.inputs)
        parameters = dict(args.parameters)
        runtime = load_model(args.model_id, provider=args.provider, zoo=zoo, variant=args.variant)
        result = runtime.infer(inputs=inputs, parameters=parameters)
        _save_outputs(dict(result.outputs), dict(args.outputs), result.sample_rate)
        print(_json_dump(result.as_dict()))
        return 0
    except (KeyError, ValueError, InferenceError, ModelZooError, OSError) as exc:
        print(f"puresound: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
