"""Unet: the frequency bookkeeping agrees with the convolutions, and the
skip-by-convolution variant can be trained."""

import pytest
import torch

from puresound.nnet.unet import Unet


def _unet(input_dim, dilation_f=(1, 1, 1), **kwargs):
    return Unet(
        input_dim=input_dim, channels=(2, 4, 4, 4), kernel_t=(2, 2, 2), stride_t=(1, 1, 1),
        dilation_t=(1, 1, 1), kernel_f=(3, 3, 3), stride_f=(2, 2, 1), dilation_f=dilation_f,
        delay=(0, 0, 0), **kwargs,
    )


@pytest.mark.parametrize("input_dim", [64, 66, 130, 257])
def test_shape_info_lists_the_frequency_size_every_down_stage_produces(input_dim):
    net = _unet(input_dim).eval()
    x = torch.randn(1, 2, input_dim, 6)
    produced = [input_dim]
    with torch.no_grad():
        x = net.input_norm(x)
        for layer in net.cnn_down:
            x = layer(x)
            produced.append(x.shape[2])
    assert net.shape_info()[0] == produced


@pytest.mark.parametrize("dilation_f", [(1, 1, 1), (1, 2, 2)])
def test_shape_info_lists_the_frequency_size_every_up_stage_produces(dilation_f):
    net = _unet(64, dilation_f=dilation_f).eval()
    skips, produced = [], []
    with torch.no_grad():
        x = net.input_norm(torch.randn(1, 2, 64, 6))
        skips.append(x)
        for layer in net.cnn_down:
            x = layer(x)
            skips.append(x)
        for i, layer in enumerate(net.cnn_up):
            x = layer(torch.cat([x, skips[-i - 1]], dim=1))[..., :-1]
            produced.append(x.shape[2])
    assert net.shape_info()[1][1:] == produced


def test_skip_conv_variant_backpropagates():
    """The skip branch reads the encoder output the decoder is also updating;
    adding into it in place breaks autograd."""
    net = _unet(64, skip_conv=True)
    net(torch.randn(2, 2, 64, 12)).sum().backward()
    assert all(p.grad is not None for p in net.skip_cnn.parameters())
