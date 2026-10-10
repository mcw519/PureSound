"""``puresound.metrics``: the wrappers' input handling and forwarding, the
scores computed in-house, and DNSMOS -- an optional dependency whose ONNX
sessions must be told their thread count by the caller."""

import builtins

import numpy as np
import pytest
import torch

import puresound.metrics as metrics_module
from puresound.metrics import Metrics, _mono_audio_tensor
from puresound.system.siso import EncDecMaskBase


@pytest.mark.parametrize("as_tensor", [False, True])
def test_check_shape_selects_first_channel_aligns_and_normalizes(as_tensor):
    clean = torch.tensor([[0.0, 2.0, -2.0, 1.0], [10.0, 10.0, 10.0, 10.0]])
    enhanced = torch.tensor([[0.0, 4.0, -4.0]])

    clean_out, enhanced_out = Metrics.check_shape(clean, enhanced, retun_as_tensor=as_tensor)

    assert isinstance(clean_out, torch.Tensor if as_tensor else np.ndarray)
    assert isinstance(enhanced_out, torch.Tensor if as_tensor else np.ndarray)
    assert clean_out.tolist() == [0.0, 1.0, -1.0]
    assert enhanced_out.tolist() == [0.0, 1.0, -1.0]


def test_pesq_and_stoi_wrappers_forward_rate_mode_and_extended_flag(monkeypatch):
    calls = []

    def fake_pesq(sr, clean, enhanced, mode):
        calls.append(("pesq", sr, mode, clean.shape, enhanced.shape))
        return 4.2 if mode == "wb" else 3.1

    def fake_stoi(clean, enhanced, sr, extended=False):
        calls.append(("stoi", sr, extended, clean.shape, enhanced.shape))
        return 0.75 if extended else 0.65

    monkeypatch.setattr(metrics_module, "pesq", fake_pesq)
    monkeypatch.setattr(metrics_module, "stoi", fake_stoi)

    clean = torch.tensor([[0.0, 1.0, -1.0]])
    enhanced = torch.tensor([[0.0, 0.5, -0.5]])

    assert Metrics.pesq_wb(clean, enhanced) == 4.2
    assert Metrics.pesq_nb(clean, enhanced) == 3.1
    assert Metrics.stoi(clean, enhanced, sr=8000) == 0.65
    assert Metrics.estoi(clean, enhanced, sr=16000) == 0.75
    assert calls == [
        ("pesq", 16000, "wb", (3,), (3,)),
        ("pesq", 8000, "nb", (3,), (3,)),
        ("stoi", 8000, False, (3,), (3,)),
        ("stoi", 16000, True, (3,), (3,)),
    ]


def test_bss_sdr_uses_mir_eval_output(monkeypatch):
    def fake_bss_eval_sources(clean, enhanced, compute_permutation):
        assert compute_permutation is False
        assert clean.shape == enhanced.shape == (3,)
        return np.array([[12.5]]), None, None, None

    monkeypatch.setattr(metrics_module, "bss_eval_sources", fake_bss_eval_sources)

    assert (
        Metrics.bss_sdr(
            torch.tensor([[0.0, 1.0, -1.0]]),
            torch.tensor([[0.0, 1.0, -1.0]]),
        )
        == 12.5
    )


def test_sisnr_is_higher_for_matching_signal_than_noisy_signal():
    clean = torch.sin(torch.linspace(0, 8, 1600)).view(1, -1)
    enhanced = clean.clone()
    noisy = clean + 0.2 * torch.randn_like(clean)

    assert Metrics.sisnr(clean, enhanced) > Metrics.sisnr(clean, noisy)
    assert Metrics.sisnr_imp(clean, enhanced, noisy) > 0


def test_f1_score_reports_binary_classification_metrics():
    result = Metrics.f1_score(
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
    )

    assert result["accuracy"] == 0.5
    assert round(result["precision"], 4) == 0.5
    assert round(result["recall"], 4) == 0.5
    assert round(result["f1_score"], 4) == 0.5


def test_noise_reduction_compares_enhanced_and_noisy_energy():
    noisy = torch.ones(1, 4)
    enhanced = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    score = Metrics.noise_reduction(noisy, enhanced)

    assert torch.allclose(score, torch.tensor([-3.0103]), atol=1e-3)


def test_mono_audio_tensor_squeezes_batches_selects_first_channel_and_clamps():
    wav = torch.tensor([[[2.0, -2.0, 0.5], [0.1, 0.2, 0.3]]])

    assert torch.allclose(_mono_audio_tensor(wav), torch.tensor([1.0, -1.0, 0.5]))


def test_dnsmos_names_its_optional_dependency_and_caches_one_session_per_thread_count(monkeypatch):
    """A scoring worker pool pins torch to one thread; onnxruntime does not
    follow that setting, so `num_threads` is forwarded to torchmetrics and is
    part of the session cache key -- the pool's 1 and a single process's None
    do not share a session."""
    import torchmetrics.audio.dnsmos as dnsmos

    built = []

    class Fake:
        def __init__(self, fs, personalized, device=None, num_threads=None, **_):
            built.append(num_threads)

        def __call__(self, wav):
            return torch.tensor([3.0, 3.1, 3.2, 3.3])

    monkeypatch.setattr(dnsmos, "DeepNoiseSuppressionMeanOpinionScore", Fake)
    monkeypatch.setattr(metrics_module, "_DNSMOS_METRICS", {})
    wav = torch.zeros(1, 16000)
    Metrics.dnsmos_p835(None, wav, sr=16000, num_threads=1)
    Metrics.dnsmos_p835(None, wav, sr=16000, num_threads=1)   # cached: no second build
    Metrics.dnsmos_p835(None, wav, sr=16000)                  # different key: builds again
    assert built == [1, None]
    assert set(metrics_module._DNSMOS_METRICS) == {(16000, False, 1), (16000, False, None)}

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torchmetrics.audio.dnsmos":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(metrics_module, "_DNSMOS_METRICS", {})
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ModuleNotFoundError, match="DNSMOS requires"):
        Metrics.dnsmos_p835(torch.zeros(16000), torch.zeros(16000))


class EchoMetricSystem(EncDecMaskBase):
    def forward(self, noisy):
        return noisy


def test_a_metric_returning_a_dict_logs_every_key_from_test_step():
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
