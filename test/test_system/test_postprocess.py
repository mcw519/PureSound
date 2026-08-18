"""Over-suppression relief, tested without building a model.

`dry_blend` and `spec_floor` were two arguments threaded through `forward()` and
applied a hundred lines apart. Pulling them into one object is mostly about being
able to write these: what the knobs do, what they cannot do, and -- the one worth
having in writing -- the ceiling `dry_blend` puts under suppression.
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


@pytest.mark.parametrize(
    "dry_blend,spec_floor",
    [(0.0, 0.0), (1.5, 0.0), (-0.1, 0.0), (1.0, 1.0), (1.0, 1.2), (1.0, -0.1)],
)
def test_a_value_outside_the_usable_range_is_rejected(dry_blend, spec_floor):
    """`dry_blend=0` discards the model and `spec_floor=1` masks nothing; both
    are almost certainly a typo for the disabled value at the other end."""
    with pytest.raises(ValueError):
        Postprocessor(dry_blend=dry_blend, spec_floor=spec_floor)


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dry_blend,expected_db", [(0.8, -13.979), (0.9, -20.0), (0.95, -26.021)]
)
def test_the_blend_ceiling_is_arithmetic_and_reported(dry_blend, expected_db):
    """The output keeps `1 - dry_blend` of the input whatever the model did, so
    suppression cannot go deeper than that -- 0.9 caps it at exactly -20 dB.

    Worth a test because it is invisible at the call site and the field benchmark
    runs at 0.9: a far-field residual near -20 dB there is measuring the blend.
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


def test_no_blend_means_no_ceiling():
    assert Postprocessor().suppression_ceiling_db == -math.inf


# --------------------------------------------------------------------------- #
# What each knob does
# --------------------------------------------------------------------------- #


def test_the_spectral_floor_lifts_only_the_bins_below_it():
    """Per-bin, so a bin the mask got right is left exactly as it was."""
    mix = torch.zeros(1, 2, 3, 1)
    mix[0, 0, :, 0] = torch.tensor([1.0, 1.0, 1.0])  # |mix| = 1 in every bin
    enh = torch.zeros(1, 2, 3, 1)
    enh[0, 0, :, 0] = torch.tensor([0.9, 0.05, 0.0])  # above / below / zeroed

    out = Postprocessor(spec_floor=0.2).floor_spectrum(enh, mix)
    mag = out[0, 0, :, 0]
    assert mag[0] == pytest.approx(0.9)  # above the floor: untouched
    assert mag[1] == pytest.approx(0.2, abs=1e-5)  # below it: lifted


def test_a_completely_zeroed_bin_cannot_be_lifted():
    """The floor scales the enhanced value up to the target magnitude, and
    scaling 0+0j reaches nothing -- there is no phase to keep.

    So the knob relieves partial over-suppression and not total, which is the
    harder case. Pinned because it is the opposite of what the name suggests,
    and because it is why `dry_blend` is still the knob that covers those bins.
    """
    mix = torch.zeros(1, 2, 1, 1)
    mix[0, 0, 0, 0] = 1.0
    enh = torch.zeros(1, 2, 1, 1)  # the mask killed this bin outright
    out = Postprocessor(spec_floor=0.2).floor_spectrum(enh, mix)
    assert float(out.abs().max()) == 0.0


def test_the_spectral_floor_keeps_the_enhanced_phase():
    torch.manual_seed(0)
    enh, mix = torch.randn(2, 2, 9, 5), torch.randn(2, 2, 9, 5) * 5.0
    out = Postprocessor(spec_floor=0.5).floor_spectrum(enh, mix)
    before = torch.atan2(*reversed(torch.chunk(enh, 2, dim=1)))
    after = torch.atan2(*reversed(torch.chunk(out, 2, dim=1)))
    assert torch.allclose(before, after, atol=1e-5)


def test_the_blend_leaves_the_non_overlapping_tail_alone():
    """The STFT round trip can leave enhanced and input a few samples apart.
    Blending the overlap and keeping the rest beats truncating the output."""
    enh = torch.ones(1, 100)
    mix = torch.zeros(1, 90)
    out = Postprocessor(dry_blend=0.5).blend_waveform(enh, mix)
    assert torch.allclose(out[..., :90], torch.full((1, 90), 0.5))
    assert torch.allclose(out[..., 90:], torch.ones(1, 10))


def test_the_blend_output_stays_inside_full_scale():
    """Same range the module clamps its own output to."""
    out = Postprocessor(dry_blend=0.5).blend_waveform(
        torch.full((1, 8), 1.0), torch.full((1, 8), 3.0)
    )
    assert float(out.max()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mask_type", ["deepfilter", "wiener", "mvdr", "mapping"])
def test_a_spectral_floor_is_refused_where_it_cannot_be_applied(mask_type):
    """It was silently ignored on four of the five mask types -- those branches
    never mentioned it -- so a recipe asking for relief it could not get looked
    like relief that did not work."""
    with pytest.raises(ValueError, match=mask_type):
        Postprocessor(spec_floor=0.2).reject_spec_floor(mask_type)
    Postprocessor().reject_spec_floor(mask_type)  # disabled: nothing to refuse


def test_the_two_calling_conventions_cannot_be_mixed():
    built = Postprocessor(dry_blend=0.9)
    assert resolve(built, 1.0, 0.0) is built
    assert resolve(None, 0.9, 0.0) == built
    with pytest.raises(ValueError, match="not both"):
        resolve(built, 0.8, 0.0)


def test_the_manifest_records_what_a_deployment_would_have_to_reproduce():
    """An exported graph contains none of this, so a runtime that only runs the
    graph is running a different system than a benchmark at dry_blend 0.9."""
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
def test_the_module_refuses_a_spectral_floor_it_cannot_apply(mask_type):
    """Pins that `forward` *asks*, not just that `Postprocessor` would refuse.

    Removing the ask from any one branch restores the silent no-op, and no other
    test in the suite notices.
    """
    from puresound.system.siso import EncDecMaskBase

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
