"""The device-payload build (`sdk/web/tools/build_assets.py`): it checks its
inputs before it deletes the previous payload."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "sdk/web/tools/build_assets.py"


@pytest.fixture
def build(tmp_path):
    spec = importlib.util.spec_from_file_location("build_assets_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SDK = tmp_path / "sdk"
    module.ORT_DIST = module.SDK / "node_modules/onnxruntime-web/dist"
    module.DEST = tmp_path / "device"
    module.MODELS = {}
    return module


def _provide_inputs(module):
    for path in [
        *(module.SDK / "dist" / name for name in module.RUNTIME_FILES),
        *(module.ORT_DIST / name for name in module.ORT_FILES),
        module.SDK / "licenses/LICENSE",
        module.SDK / "licenses/ThirdPartyNotices.txt",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


def test_build_without_its_inputs_stops_and_leaves_the_previous_payload(build):
    previous = build.DEST / "assets/model.onnx"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous")

    with pytest.raises(SystemExit, match="npm run build"):
        build.main()

    assert previous.read_text() == "previous"


def test_build_catalogues_every_generated_file(build):
    _provide_inputs(build)

    build.main()

    catalog = json.loads((build.DEST / "catalog.json").read_text())
    assert {"runtime/runtime.js", "runtime/audio.js", "vendor/ort/ort-wasm-simd-threaded.wasm"} <= set(catalog["files"])
