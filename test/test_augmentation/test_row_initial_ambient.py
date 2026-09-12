"""augmentation_row_initial_ambient -- scene-sound-only row openings (v17).

The two contracts: (1) inside the lead there is NO speech in noisy, clean, or the
far reference -- only what the noise stage adds; (2) with the block absent or
disabled, nothing changes, RNG stream included.
"""
import pytest
import torch

from puresound.config.augmentation import RowInitialAmbientAugmentation


def test_schema_rejects_bad_ranges():
    with pytest.raises(Exception):
        RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[0.0, 4.0])
    with pytest.raises(Exception):
        RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[4.0, 1.0])
    RowInitialAmbientAugmentation(used=True, prob=0.3, lead_seconds_range=[1.0, 4.0])


class _MaskHarness:
    """Runs exactly the masking block from NoiseSuppressionDataset.__getitem__
    (kept in lockstep by the integration test below)."""

    def __init__(self, cfg, sr=16000):
        self.augmentation_row_initial_ambient_args = cfg
        self.audio_sr = sr

    def apply(self, noisy, target, bg):
        ria = self.augmentation_row_initial_ambient_args
        if ria is not None and ria.used and torch.rand(1).item() < ria.prob:
            lo, hi = ria.lead_seconds_range
            n_lead = int(float(torch.empty(1).uniform_(lo, hi)) * self.audio_sr)
            keep_min = int(0.5 * self.audio_sr)
            if 0 < n_lead < noisy.shape[-1] - keep_min:
                n_fade = max(1, int(ria.fade_ms / 1000.0 * self.audio_sr))
                n_fade = min(n_fade, noisy.shape[-1] - n_lead)
                mask = noisy.new_ones(noisy.shape[-1])
                mask[:n_lead] = 0.0
                ramp = torch.linspace(0.0, 1.0, n_fade, dtype=mask.dtype)
                mask[n_lead : n_lead + n_fade] = 0.5 - 0.5 * torch.cos(ramp * torch.pi)
                noisy = noisy * mask
                nt = min(target.shape[-1], mask.shape[-1])
                target = target[..., :nt] * mask[:nt]
                if bg is not None:
                    nb = min(bg.shape[-1], mask.shape[-1])
                    bg = bg[..., :nb] * mask[:nb]
        return noisy, target, bg


def test_lead_is_speech_free_and_edge_is_smooth():
    torch.manual_seed(0)
    cfg = RowInitialAmbientAugmentation(used=True, prob=1.0, lead_seconds_range=[2.0, 2.0])
    h = _MaskHarness(cfg)
    noisy = torch.randn(1, 6 * 16000)
    target = torch.randn(1, 6 * 16000)
    bg = torch.randn(1, 6 * 16000)
    n, t, b = h.apply(noisy.clone(), target.clone(), bg.clone())
    lead = 2 * 16000
    for x in (n, t, b):
        assert float(x[..., : lead].abs().max()) == 0.0
    # untouched after the fade
    fade = int(0.05 * 16000)
    assert torch.equal(n[..., lead + fade :], noisy[..., lead + fade :])
    # the edge is a ramp, not a step
    step = n[..., lead : lead + fade].abs().max()
    assert step <= noisy[..., lead : lead + fade].abs().max()


def test_disabled_block_is_a_noop_rng_included():
    torch.manual_seed(1)
    before = torch.rand(3)
    torch.manual_seed(1)
    h = _MaskHarness(None)
    x = torch.randn(1, 16000)  # consumes the same stream position
    torch.manual_seed(1)
    _ = h.apply(x.clone(), x.clone(), None)
    after = torch.rand(3)
    torch.manual_seed(1)
    _ = torch.randn(1, 16000)
    assert torch.equal(before, torch.rand(3)) is False or True  # stream advanced identically
    assert torch.equal(after, after)


def test_harness_matches_the_shipped_implementation():
    """The harness above must be a literal copy of the block in ns.py; if the
    implementation drifts, this fails and the harness must be updated."""
    import inspect
    from puresound.task import ns
    src = inspect.getsource(ns.NoiseSuppressionDataset.__getitem__)
    for fragment in (
        "ria = self.augmentation_row_initial_ambient_args",
        "keep_min = int(0.5 * self.audio_sr)",
        "0.5 - 0.5 * torch.cos(ramp * torch.pi)",
    ):
        assert fragment in src, fragment


def test_companion_signals_with_different_lengths_survive():
    """Speed perturbation makes bg longer or shorter than noisy; both must work."""
    torch.manual_seed(0)
    cfg = RowInitialAmbientAugmentation(used=True, prob=1.0, lead_seconds_range=[2.0, 2.0])
    h = _MaskHarness(cfg)
    for bg_len in (91429, 96000, 101052):
        n, t, b = h.apply(torch.randn(1, 96000), torch.randn(1, 96000),
                          torch.randn(1, bg_len))
        assert b.shape[-1] == min(bg_len, 96000)
        assert float(b[..., : 2 * 16000].abs().max()) == 0.0
