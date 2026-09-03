from __future__ import annotations

import json

from puresound.cli import main


def test_models_list_and_validate(capsys):
    assert main(["models", "list", "--task", "voice_isolation"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["id"] == "voice-isolate-dpcrn-v8" for item in payload)
    assert main(["models", "validate", "--no-graph"]) == 0
    assert "Model Zoo valid" in capsys.readouterr().out
