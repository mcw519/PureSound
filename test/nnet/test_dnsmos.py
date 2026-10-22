"""What `DNSMOSLoss` must do: score like the DNSMOS metric (same calibration),
reward a cleaner signal with a lower loss, give a finite gradient on every
batch it can see (silent rows, rows with zero stretches, batches with nothing
to score), and keep a silent row away from the judge instead of masking it.
"""

import os

import pytest
import torch

from puresound.nnet.loss import DNSMOSLoss
from puresound.nnet.loss.dnsmos import INPUT_SAMPLES

SR = 16000
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.expanduser("~/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx")),
    reason="DNSMOS model not downloaded",
)


def _speechlike(seed: int, seconds: float = 3.0, level: float = 0.1) -> torch.Tensor:
    """Harmonic comb with a slow syllable-rate envelope: enough structure for the
    judge to score it well above its noise floor."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * SR)) / SR
    f0 = 120.0 + 20.0 * torch.sin(2 * torch.pi * 0.7 * t)
    phase = 2 * torch.pi * torch.cumsum(f0, 0) / SR
    tone = sum(torch.sin(k * phase + float(torch.rand(1, generator=g)) * 6.28) / k for k in range(1, 12))
    env = 0.5 * (1 + torch.sin(2 * torch.pi * 4.0 * t)) ** 2
    x = tone * env
    return (level * x / x.abs().max()).unsqueeze(0)


def _noise(seed: int, seconds: float = 3.0, level: float = 0.05) -> torch.Tensor:
    return torch.randn(1, int(seconds * SR), generator=torch.Generator().manual_seed(seed)) * level


@pytest.fixture(scope="module")
def loss():
    return DNSMOSLoss(sig_weight=1.0, bak_weight=1.0)


def test_scores_match_the_metric_calibration(loss):
    pytest.importorskip("torchmetrics")
    try:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos
    except ImportError:  # pragma: no cover - optional metric dependencies
        pytest.skip("torchmetrics DNSMOS dependencies missing")
    x = _speechlike(1, seconds=INPUT_SAMPLES / SR) + _noise(2, seconds=INPUT_SAMPLES / SR, level=0.01)
    x = x * (loss.reference_rms / x.pow(2).mean().sqrt())  # the metric sees the row at the loss's reference level
    try:
        ref = dnsmos(x, SR, personalized=False)  # [1, 4] = p808, sig, bak, ovr
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torchmetrics DNSMOS unavailable: {e}")
    s, keep = loss.scores(x)
    assert bool(keep.all())
    for i, k in enumerate(("sig", "bak", "ovr")):
        assert float(s[k]) == pytest.approx(float(ref[0, i + 1]), abs=2e-3), k


def test_noisier_rows_cost_more_and_level_cannot_buy_a_better_score(loss):
    clean = _speechlike(3)
    noisy = clean + _noise(4, level=0.05)
    l_clean, l_noisy = float(loss(clean)), float(loss(noisy))
    assert l_noisy > l_clean
    # the judge itself is level-sensitive ...
    natural = DNSMOSLoss(sig_weight=1.0, bak_weight=1.0, reference_rms=None)
    s_a, _ = natural.scores(noisy)
    s_b, _ = natural.scores(noisy * 0.05)  # 26 dB quieter, still above the silence floor
    assert max(abs(float(s_a[k]) - float(s_b[k])) for k in ("sig", "bak", "ovr")) > 0.1
    # ... so the term fixes the level before scoring, and a level change buys nothing
    s_a, _ = loss.scores(noisy)
    s_b, _ = loss.scores(noisy * 0.05)
    for k in ("sig", "bak", "ovr"):
        assert float(s_a[k]) == pytest.approx(float(s_b[k]), abs=1e-4), k


def test_silent_rows_are_not_scored_and_do_not_poison_the_batch(loss):
    x = torch.cat([_speechlike(5) + _noise(6, level=0.02), torch.zeros(1, 3 * SR)]).requires_grad_(True)
    s, keep = loss.scores(x)
    assert keep.tolist() == [True, False]
    assert float(s["sig"][1]) == 0.0  # excluded row carries no score
    value = loss(x)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0
    assert x.grad[1].abs().sum() == 0

    silent = torch.zeros(2, 2 * SR, requires_grad=True)
    value = loss(silent)
    assert float(value) == 0.0
    value.backward()  # stays attached to the graph: a loss with nothing to score must not break the step
    assert torch.isfinite(silent.grad).all()


def test_zero_stretches_inside_a_row_keep_the_gradient_finite(loss):
    """A dropped packet leaves exactly-zero STFT bins; the judge's Sqrt/Log of 0
    once turned the whole batch's gradient into NaN."""
    x = _speechlike(7) + _noise(8, level=0.02)
    x[..., SR : SR + 4096] = 0.0
    x[..., : 2048] = 0.0
    x = x.requires_grad_(True)
    loss(x).backward()
    assert torch.isfinite(x.grad).all()


def test_short_rows_are_tiled_and_long_rows_truncated(loss):
    short = _speechlike(9, seconds=1.0)
    long = _speechlike(9, seconds=12.0)
    for x in (short, long):
        s, keep = loss.scores(x)
        assert bool(keep.all()) and torch.isfinite(torch.stack(list(s.values()))).all()


def test_max_rows_scores_a_subset_and_the_floor_hinge_engages():
    x = torch.cat([_speechlike(s) + _noise(s + 10, level=0.05) for s in (11, 12, 13)]).requires_grad_(True)
    sub = DNSMOSLoss(bak_weight=1.0, max_rows=1)
    sub(x).backward()
    assert torch.isfinite(x.grad).all()
    assert sum(int(x.grad[i].abs().sum() > 0) for i in range(3)) == 1

    s, _ = DNSMOSLoss(bak_weight=1.0).scores(x[:1].detach())
    sig = float(s["sig"][0])
    base = DNSMOSLoss(bak_weight=1.0)
    with_floor = DNSMOSLoss(bak_weight=1.0, sig_floor=sig + 1.0, floor_weight=2.0)
    assert float(with_floor(x[:1].detach())) == pytest.approx(float(base(x[:1].detach())) + 2.0, abs=1e-4)
    idle_floor = DNSMOSLoss(bak_weight=1.0, sig_floor=sig - 1.0)
    assert float(idle_floor(x[:1].detach())) == pytest.approx(float(base(x[:1].detach())), abs=1e-6)


def test_configuration_errors_are_raised_at_construction():
    with pytest.raises(ValueError, match="16 kHz"):
        DNSMOSLoss(sample_rate=48000)
    with pytest.raises(ValueError, match="no active term"):
        DNSMOSLoss(sig_weight=0.0, bak_weight=0.0, ovr_weight=0.0)
    with pytest.raises(FileNotFoundError):
        DNSMOSLoss(model_path="/nonexistent/sig_bak_ovr.onnx")


def test_judge_stays_in_eval_mode_inside_a_training_module():
    term = DNSMOSLoss(bak_weight=1.0)
    term.train()
    assert not term.judge.training
    assert all(not b.requires_grad for b in term.judge.buffers())
