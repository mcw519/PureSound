import numpy as np
import torch

import puresound.metrics as metrics_module
from puresound.metrics import Metrics, _mono_audio_tensor


def test_check_shape_selects_first_channel_aligns_and_normalizes():
    clean = torch.tensor([[0.0, 2.0, -2.0, 1.0], [10.0, 10.0, 10.0, 10.0]])
    enhanced = torch.tensor([[0.0, 4.0, -4.0]])

    clean_out, enhanced_out = Metrics.check_shape(clean, enhanced)

    assert isinstance(clean_out, np.ndarray)
    assert clean_out.tolist() == [0.0, 1.0, -1.0]
    assert enhanced_out.tolist() == [0.0, 1.0, -1.0]


def test_check_shape_can_return_tensors():
    clean, enhanced = Metrics.check_shape(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[0.0, 4.0]]),
        retun_as_tensor=True,
    )

    assert isinstance(clean, torch.Tensor)
    assert isinstance(enhanced, torch.Tensor)
    assert torch.allclose(clean, torch.tensor([0.0, 1.0]))
    assert torch.allclose(enhanced, torch.tensor([0.0, 1.0]))


def test_pesq_wrappers_call_expected_sample_rates_and_modes(monkeypatch):
    calls = []

    def fake_pesq(sr, clean, enhanced, mode):
        calls.append((sr, mode, clean.shape, enhanced.shape))
        return 4.2 if mode == "wb" else 3.1

    monkeypatch.setattr(metrics_module, "pesq", fake_pesq)

    clean = torch.tensor([[0.0, 1.0, -1.0]])
    enhanced = torch.tensor([[0.0, 0.5, -0.5]])

    assert Metrics.pesq_wb(clean, enhanced) == 4.2
    assert Metrics.pesq_nb(clean, enhanced) == 3.1
    assert calls == [(16000, "wb", (3,), (3,)), (8000, "nb", (3,), (3,))]


def test_stoi_and_estoi_wrappers_forward_sr_and_extended_flag(monkeypatch):
    calls = []

    def fake_stoi(clean, enhanced, sr, extended=False):
        calls.append((sr, extended, clean.shape, enhanced.shape))
        return 0.75 if extended else 0.65

    monkeypatch.setattr(metrics_module, "stoi", fake_stoi)

    clean = torch.tensor([[0.0, 1.0, -1.0]])
    enhanced = torch.tensor([[0.0, 0.5, -0.5]])

    assert Metrics.stoi(clean, enhanced, sr=8000) == 0.65
    assert Metrics.estoi(clean, enhanced, sr=16000) == 0.75
    assert calls == [(8000, False, (3,), (3,)), (16000, True, (3,), (3,))]


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
