from __future__ import annotations

import json

from puresound.cli import main
import puresound.cli as cli


def test_models_list_and_validate(capsys):
    assert main(["models", "list", "--task", "voice_isolation"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["id"] == "voice-isolate-dpcrn-v8" for item in payload)
    assert main(["models", "validate", "--no-graph"]) == 0
    assert "Model Zoo valid" in capsys.readouterr().out


def test_providers_reports_capabilities_and_mps_alias(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert main(["providers"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["capabilities"] == {"cpu": True, "cuda": True, "coreml": False}
    assert payload["mps_alias"] == "coreml"
    assert payload["auto_order"] == ["cuda", "coreml", "cpu"]
