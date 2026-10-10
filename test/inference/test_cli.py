"""The ``puresound`` command line: model listing and validation, provider
report, and the ``web`` subcommand's flags."""

from __future__ import annotations

import json

import pytest

from puresound.cli import build_parser, main
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


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["web", "--ip", "0.0.0.0", "--port", "9123"], {"host": "0.0.0.0", "port": 9123}),
        (["web"], {"no_history": False, "https": False}),
        (["web", "--no-history"], {"no_history": True}),
        (["web", "--history-dir", "/tmp/x"], {"history_dir": "/tmp/x"}),
        (["web", "--https"], {"https": True, "tls_cert": None}),
        (["web", "--tls-cert", "c.pem", "--tls-key", "k.pem"], {"tls_cert": "c.pem", "tls_key": "k.pem"}),
        (["web", "--pipeline-root", "/repo"], {"pipeline_root": "/repo"}),
        (["web"], {"allowed_hosts": None}),
        (["web", "--allow-host", "a.example", "--allow-host", "b.example"], {"allowed_hosts": ["a.example", "b.example"]}),
    ],
)
def test_web_subcommand_flags(argv, expected):
    args = build_parser().parse_args(argv)
    assert {name: getattr(args, name) for name in expected} == expected


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_web_subcommand_rejects_invalid_ports(port):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["web", "--port", port])


def test_web_subcommand_hands_the_extra_host_names_to_the_server(monkeypatch):
    import puresound.web

    seen = {}
    monkeypatch.setattr(puresound.web, "run", lambda **kwargs: seen.update(kwargs))
    assert main(["web", "--no-history", "--allow-host", "proxy.example"]) == 0
    assert seen["allowed_hosts"] == ["proxy.example"]
