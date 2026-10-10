"""Over-suppression relief, tested without building a model: what `dry_blend`
and `spec_floor` do, what they cannot do, and -- the one worth having in
writing -- the ceiling `dry_blend` puts under suppression.
"""

import math

import pytest
import torch

from puresound.system.postprocess import IDENTITY, Postprocessor, resolve


def test_the_defaults_are_a_no_op():
    post = Postprocessor()
    assert not post.enabled
    enh, mix = torch.randn(2, 400), torch.randn(2, 400)
    tf_enh, tf_mix = torch.randn(2, 2, 9, 5), torch.randn(2, 2, 9, 5)
    assert post.blend_waveform(enh, mix) is enh
    assert post.floor_spectrum(tf_enh, tf_mix) is tf_enh
    assert resolve(None, 1.0, 0.0) is IDENTITY
    assert post.suppression_ceiling_db == -math.inf, "no blend means no ceiling"


@pytest.mark.parametrize(
    "dry_blend,spec_floor",
    [(0.0, 0.0), (1.5, 0.0), (-0.1, 0.0), (1.0, 1.0), (1.0, 1.2), (1.0, -0.1)],
)
def test_a_value_outside_the_usable_range_is_rejected(dry_blend, spec_floor):
    """`dry_blend=0` discards the model and `spec_floor=1` masks nothing; both
    are almost certainly a typo for the disabled value at the other end."""
    with pytest.raises(ValueError):
        Postprocessor(dry_blend=dry_blend, spec_floor=spec_floor)


@pytest.mark.parametrize(
    "dry_blend,spec_floor", [(1.5, 0.0), (1.0, -0.1), (0.0, 0.0), (1.0, 1.0)]
)
def test_the_keyword_form_rejects_a_value_outside_the_range_too(dry_blend, spec_floor):
    """`forward(dry_blend=1.5)` must not read as the disabled value."""
    with pytest.raises(ValueError):
        resolve(None, dry_blend, spec_floor)


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dry_blend,expected_db", [(0.8, -13.979), (0.9, -20.0), (0.95, -26.021)]
)
def test_the_blend_ceiling_is_arithmetic_and_reported(dry_blend, expected_db):
    """The output keeps `1 - dry_blend` of the input whatever the model did, so
    suppression cannot go deeper than that -- 0.9 caps it at exactly -20 dB.

    Worth a test because it is invisible at the call site: a far-field residual
    near the ceiling in a benchmark run with a blend is measuring the blend.
    """
    post = Postprocessor(dry_blend=dry_blend)
    assert post.suppression_ceiling_db == pytest.approx(expected_db, abs=1e-3)

    # And it really is the floor: a perfect suppressor still leaves that much.
    torch.manual_seed(0)
    interferer = torch.randn(1, 4000) * 0.1
    perfect = torch.zeros_like(interferer)
    out = post.blend_waveform(perfect, interferer)
    achieved = 20 * math.log10(
        float(out.pow(2).mean().sqrt() / interferer.pow(2).mean().sqrt())
    )
    assert achieved == pytest.approx(expected_db, abs=1e-3)


# --------------------------------------------------------------------------- #
# What each knob does
# --------------------------------------------------------------------------- #


def test_the_spectral_floor_lifts_only_partially_suppressed_bins():
    """Per-bin, so a bin the mask got right is left exactly as it was.

    A bin the mask zeroed outright is NOT lifted: the floor scales the enhanced
    value up to the target magnitude, and scaling 0+0j reaches nothing -- there
    is no phase to keep. So the knob relieves partial over-suppression and not
    total; pinned because it is the opposite of what the name suggests, and
    because it is why `dry_blend` is still the knob that covers those bins.
    """
    mix = torch.zeros(1, 2, 3, 1)
    mix[0, 0, :, 0] = torch.tensor([1.0, 1.0, 1.0])  # |mix| = 1 in every bin
    enh = torch.zeros(1, 2, 3, 1)
    enh[0, 0, :, 0] = torch.tensor([0.9, 0.05, 0.0])  # above / below / zeroed

    out = Postprocessor(spec_floor=0.2).floor_spectrum(enh, mix)
    mag = out[0, 0, :, 0]
    assert mag[0] == pytest.approx(0.9)  # above the floor: untouched
    assert mag[1] == pytest.approx(0.2, abs=1e-5)  # below it: lifted
    assert float(out[0, :, 2].abs().max()) == 0.0  # zeroed: stays zero

    # And the lift keeps the enhanced phase.
    torch.manual_seed(0)
    enh, mix = torch.randn(2, 2, 9, 5), torch.randn(2, 2, 9, 5) * 5.0
    out = Postprocessor(spec_floor=0.5).floor_spectrum(enh, mix)
    before = torch.atan2(*reversed(torch.chunk(enh, 2, dim=1)))
    after = torch.atan2(*reversed(torch.chunk(out, 2, dim=1)))
    assert torch.allclose(before, after, atol=1e-5)


def test_the_blend_keeps_the_non_overlapping_tail_and_full_scale():
    """The STFT round trip can leave enhanced and input a few samples apart:
    blending the overlap and keeping the rest beats truncating the output. The
    result stays in the range the module clamps its own output to."""
    post = Postprocessor(dry_blend=0.5)
    out = post.blend_waveform(torch.ones(1, 100), torch.zeros(1, 90))
    assert torch.allclose(out[..., :90], torch.full((1, 90), 0.5))
    assert torch.allclose(out[..., 90:], torch.ones(1, 10))

    out = post.blend_waveform(torch.full((1, 8), 1.0), torch.full((1, 8), 3.0))
    assert float(out.max()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_the_two_calling_conventions_cannot_be_mixed_and_the_manifest_records_them():
    built = Postprocessor(dry_blend=0.9)
    assert resolve(built, 1.0, 0.0) is built
    assert resolve(None, 0.9, 0.0) == built
    with pytest.raises(ValueError, match="not both"):
        resolve(built, 0.8, 0.0)

    # The manifest records what a deployment would have to reproduce: an
    # exported graph contains none of this, so a runtime that only runs the
    # graph is running a different system than one with the blend.
    manifest = Postprocessor(dry_blend=0.9, spec_floor=0.1).as_manifest()
    assert manifest == {
        "dry_blend": 0.9,
        "spec_floor": 0.1,
        "suppression_ceiling_db": pytest.approx(-20.0),
    }


# --------------------------------------------------------------------------- #
# ...and that the module actually asks
# --------------------------------------------------------------------------- #


class _Stub(torch.nn.Module):
    """Stands in for encoder / feats / backbone.

    `forward` reaches the mask-type branch after three calls and nothing else,
    so the refusal is checkable without building a real model -- which matters,
    because these four mask types have no shipped recipe to test through.
    """

    def __init__(self, returns):
        super().__init__()
        self.returns = returns

    def forward(self, *args, **kwargs):
        return self.returns


@pytest.mark.parametrize("mask_type", ["deepfilter", "wiener", "mvdr", "mapping"])
def test_a_spectral_floor_is_refused_where_it_cannot_be_applied(mask_type):
    """Only the complex mask can apply the floor; on the other four a recipe
    asking for it would get relief that silently does nothing.

    Checked both on `Postprocessor` and through `forward`, which must *ask*:
    removing the ask from any one branch restores the silent no-op, and no other
    test notices.
    """
    from puresound.system.siso import EncDecMaskBase

    with pytest.raises(ValueError, match=mask_type):
        Postprocessor(spec_floor=0.2).reject_spec_floor(mask_type)
    Postprocessor().reject_spec_floor(mask_type)  # disabled: nothing to refuse

    spectrum = torch.randn(1, 2, 9, 5)
    module = object.__new__(EncDecMaskBase)
    torch.nn.Module.__init__(module)
    module.mask_type = mask_type
    module.encoder = _Stub(spectrum)
    module.feats = _Stub((spectrum, spectrum))
    # wiener / mvdr unpack the backbone's output into (mask, ifc, cov) before
    # the branch is reached.
    module.backbone = _Stub(
        (spectrum, torch.randn(1, 2, 9, 4), torch.randn(1, 2, 9, 4))
        if mask_type in ("wiener", "mvdr")
        else spectrum
    )

    with pytest.raises(ValueError, match=mask_type):
        module(torch.randn(1, 800), spec_floor=0.2)
