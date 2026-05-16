import pytest
import torch

from puresound.metrics import Metrics
from puresound.system.siso import EncDecMaskBase


class EchoMetricSystem(EncDecMaskBase):
    def forward(self, noisy):
        return noisy


def test_dnsmos_metric_missing_optional_dependency_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torchmetrics.audio.dnsmos":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ModuleNotFoundError, match="DNSMOS requires"):
        Metrics.dnsmos_p835(torch.zeros(16000), torch.zeros(16000))


def test_logging_metric_dict_from_test_step():
    system = EchoMetricSystem(
        encoder=torch.nn.Identity(),
        feats=torch.nn.Identity(),
        backbone=torch.nn.Identity(),
    )
    system.register_metrics_func(
        {
            "dnsmos_p835": {
                "func": lambda clean, enhanced: {
                    "dnsmos_p808": 1.0,
                    "dnsmos_sig": 2.0,
                    "dnsmos_bak": 3.0,
                    "dnsmos_ovr": 4.0,
                },
                "sr": None,
            }
        }
    )
    batch = {
        "noisy_speech": torch.zeros(1, 16000),
        "clean_speech": torch.zeros(1, 16000),
        "sr": torch.tensor(16000),
    }

    system.test_step(batch, 0)
    scores = system.puresound_logging.average()

    assert scores["dnsmos_p808"] == 1.0
    assert scores["dnsmos_sig"] == 2.0
    assert scores["dnsmos_bak"] == 3.0
    assert scores["dnsmos_ovr"] == 4.0
