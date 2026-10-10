"""DPCRN: optional blocks cost nothing when off, training-only heads never write
the mask, and a block configuration that cannot work is refused at build time."""

import io

import pytest
import torch
from pydantic import ValidationError

from puresound.nnet.dpcrn import DPCRN, DPRNNblock2D


def tiny_dpcrn(**kw):
    return DPCRN(
        input_dim=64, channels=(1, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=16, **kw,
    )


def test_optional_heads_off_is_the_plain_network_bit_for_bit():
    """Disabled blocks build nothing, change no output and add no checkpoint key,
    so a checkpoint from before a head existed still loads strictly."""
    torch.manual_seed(0)
    plain = tiny_dpcrn().eval()
    torch.manual_seed(0)
    with_blocks = tiny_dpcrn(
        vad_head={"enabled": False},
        identity_head={"enabled": False},
        proximity_head={"enabled": False},
        expose_bottleneck=False,
    ).eval()

    x = torch.randn(2, 1, 64, 30)
    with torch.no_grad():
        assert torch.equal(plain(x), with_blocks(x))

    for name in ("vad_head", "identity_head", "proximity_head"):
        assert getattr(with_blocks, name) is None
    for name in ("last_vad_logits", "last_identity_emb", "last_proximity"):
        assert getattr(with_blocks, name) is None

    buf = io.BytesIO()
    torch.save(plain.state_dict(), buf)
    buf.seek(0)
    report = with_blocks.load_state_dict(torch.load(buf), strict=True)
    assert list(report.missing_keys) == [] and list(report.unexpected_keys) == []


def test_training_only_heads_read_the_bottleneck_and_never_write_the_mask():
    """Same separator weights with and without the heads give the same mask. And
    the pooled bottleneck the identity loss's teacher is handed gives the same
    readout as the live head on the raw features -- otherwise teacher and
    student would see different features and nothing would say so."""
    torch.manual_seed(0)
    plain = tiny_dpcrn().eval()
    headed = tiny_dpcrn(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    ).eval()
    headed.load_state_dict(plain.state_dict(), strict=False)

    x = torch.randn(2, 1, 64, 24)
    with torch.no_grad():
        assert torch.equal(plain(x), headed(x))
        pooled = headed.last_bottleneck_graph
        assert torch.allclose(headed.identity_head(pooled), headed.last_identity_emb, atol=1e-6)
        assert torch.allclose(headed.proximity_head(pooled), headed.last_proximity, atol=1e-6)


_DEEP = dict(
    input_dim=64, channels=(2, 8, 8, 8), kernel_t=(2, 2, 2), stride_t=(1, 1, 1),
    dilation_t=(1, 1, 1), kernel_f=(5, 3, 3), stride_f=(2, 2, 1), dilation_f=(1, 1, 1),
    delay=(0, 0, 0), rnn_hidden=8,
)
_MAMBA_CONTEXT = dict(
    input_dim=256, channels=[2, 8], rnn_hidden=12, kernel_t=[2], stride_t=[1],
    dilation_t=[1], kernel_f=[5], stride_f=[2], dilation_f=[1], delay=[0],
    inter_type="mamba_context", mamba_args={"d_state": 4, "expand": 1},
    mamba_context={"n_bands": 8},
)


@pytest.mark.parametrize(
    "build, error, match",
    [
        # `extra="forbid"`: a misspelled knob must not train a whole run at the default.
        pytest.param(lambda: tiny_dpcrn(identity_head={"enabled": True, "dimm": 32}),
                     ValidationError, "dimm", id="identity-head-typo"),
        pytest.param(lambda: tiny_dpcrn(proximity_head={"enabled": True, "kernel_t": 7}),
                     ValidationError, "kernel_t", id="proximity-head-extra-key"),
        pytest.param(lambda: DPCRN(**_DEEP, df_head={"bins": 128}),
                     ValueError, "exceeds", id="df-head-wider-than-mask"),
        pytest.param(lambda: DPCRN(**_MAMBA_CONTEXT, band_bottleneck={"n_bands": 16}),
                     ValueError, "mutually exclusive", id="two-bandings"),
        pytest.param(lambda: DPRNNblock2D(64, 64, intra_type="conv", fused_type="FiLM"),
                     ValueError, "intra_type", id="unknown-intra-type"),
        pytest.param(lambda: DPRNNblock2D(64, 64, embedding_size=8),
                     ValueError, "fused_type", id="embedding-without-fusion"),
    ],
)
def test_an_unusable_block_configuration_is_refused(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_intra_attention_is_a_smaller_drop_in_for_the_intra_lstm():
    lstm = DPRNNblock2D(64, 64, intra_type="lstm", fused_type="FiLM")
    attn = DPRNNblock2D(64, 64, intra_type="attention", fused_type="FiLM")
    x = torch.randn(2, 64, 32, 10)
    assert lstm(x).shape == attn(x).shape == x.shape

    def intra_params(block):
        return sum(p.numel() for n, p in block.named_parameters() if n.startswith("intra"))

    assert intra_params(attn) < intra_params(lstm)


def test_mamba_context_bands_only_the_temporal_stream_on_the_real_cnn_grid():
    """Same-padded 257 / 2 / 2 gives 65 positions, not floor(257 / 4) = 64, and
    the inter (temporal) model sees one sequence per band, not per unit."""
    net = DPCRN(
        input_dim=257, channels=[2, 8, 16], rnn_hidden=12,
        kernel_t=[2, 2], stride_t=[1, 1], dilation_t=[1, 1],
        kernel_f=[5, 3], stride_f=[2, 2], dilation_f=[1, 1], delay=[0, 0],
        inter_type="mamba_context", mamba_args={"d_state": 4, "expand": 1},
        mamba_context={"n_bands": 8, "sample_rate": 16000},
    ).eval()
    block = net.dprnn_block1
    assert block.context_bottleneck.n_units == 65

    seen = []
    handle = block.inter_rnn.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape))
    )
    x = torch.randn(2, 16, 65, 7)
    with torch.no_grad():
        y = block(x)
    handle.remove()
    assert y.shape == x.shape
    assert seen == [(2 * 8, 16, 7)]


def test_get_args_rebuilds_the_same_network():
    torch.manual_seed(0)
    net = DPCRN(
        **_DEEP, dvec_dim=8, vad_head={"enabled": True},
        proximity_head={"enabled": True, "hidden": 8}, spectral_compress=True,
    ).eval()
    rebuilt = DPCRN(**net.get_args).eval()
    rebuilt.load_state_dict(net.state_dict(), strict=True)
    x, dvec = torch.randn(2, 2, 64, 12), torch.randn(2, 8)
    with torch.no_grad():
        assert torch.equal(net(x, dvec), rebuilt(x, dvec))
    assert DPRNNblock2D(16, 16).fused_type is None


def test_skip_conv_leaves_the_stashed_bottleneck_intact():
    """The decoder adds the skip branch into its running tensor; the tensor the
    inference-time readout stashed must not be that same storage."""
    torch.manual_seed(0)
    net = tiny_dpcrn(skip_conv=True).eval()
    net.stash_bottleneck = True
    seen = []
    net.dprnn_block2.register_forward_hook(lambda _m, _i, out: seen.append(out.detach().clone()))
    with torch.no_grad():
        net(torch.randn(1, 1, 64, 12))
    assert torch.equal(net.last_bottleneck, seen[0])
