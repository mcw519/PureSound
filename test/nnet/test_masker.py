"""Masker: operations that act on a batch must give each row what it would get alone."""

import pytest
import torch

from puresound.nnet.masker import Masker


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_envelope_postfilter_acts_on_each_row_independently(batch):
    """The post-filter gain is per (row, bin, frame); it must not broadcast
    against the real/imaginary axis, which would crash for most batch sizes and
    mix rows when the batch happens to be 2."""
    torch.manual_seed(0)
    spec = torch.randn(batch, 2, 8, 5)
    mask = 0.3 * torch.randn(batch, 2, 8, 5)

    together = Masker.apply_complex_mask_on_reim(spec, mask, postfilter=True)
    alone = torch.cat(
        [
            Masker.apply_complex_mask_on_reim(spec[i : i + 1], mask[i : i + 1], postfilter=True)
            for i in range(batch)
        ]
    )
    assert together.shape == spec.shape
    assert torch.allclose(together, alone)


def test_deep_filter_accepts_a_contiguous_spectrum():
    """The spectrum arrives as a permuted view of the encoder output in the
    pipeline, but a contiguous [N, 2, C, T] tensor is the same input."""
    torch.manual_seed(1)
    n, c, t, order, bins = 2, 16, 10, 3, 8
    encoder_out = torch.randn(n, c, t, 2)
    masks = torch.randn(n, 2 * order, bins, t)

    from_view = Masker.apply_df_on_reim(encoder_out.permute(0, 3, 1, 2), masks, bins, order)
    from_copy = Masker.apply_df_on_reim(
        encoder_out.permute(0, 3, 1, 2).contiguous(), masks, bins, order
    )
    assert from_copy.shape == (n, 2, c, t)
    assert torch.equal(from_view, from_copy)
