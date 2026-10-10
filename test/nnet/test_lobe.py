"""Building blocks under ``puresound/nnet/lobe``: shape contracts, the STFT
encoders' round trip, and the causality the streaming heads rely on."""

from pathlib import Path

import numpy as np
import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.nnet.features import WeightedSum
from puresound.nnet.lobe.attention import MhaSelfAttenLayer
from puresound.nnet.lobe.cnn import FFC, DepthwiseSeparableConv1d, SpectralTransform
from puresound.nnet.lobe.dsp import FrequencyEQLayer
from puresound.nnet.lobe.encoder import ConvEncDec, UnifiedConvEncDec
from puresound.nnet.lobe.group_op import GroupedGRU
from puresound.nnet.lobe.heads import IdentityHead, ProximityHead
from puresound.nnet.lobe.rnn import FSMN, ConditionFSMN
from puresound.nnet.lobe.trivial import SplitMerge

TEST_AUDIO_PATH = str(Path(__file__).resolve().parents[1] / "test_case" / "1272-141231-0008.flac")


@pytest.mark.nnet
@pytest.mark.parametrize(
    "block, conditioned, causal",
    [
        pytest.param(lambda: FSMN(256, 256, 192, 3, 3), False, False, id="fsmn"),
        pytest.param(lambda: FSMN(256, 256, 192, 3, 0), False, True, id="fsmn-causal"),
        pytest.param(lambda: ConditionFSMN(256, 256, 192, 192, 3, 3), True, False,
                     id="cond-fsmn"),
        pytest.param(lambda: ConditionFSMN(256, 256, 192, 192, 3, 3, use_film=True), True,
                     False, id="cond-fsmn-film"),
    ],
)
def test_fsmn_blocks_keep_the_time_axis_and_a_causal_one_ignores_the_future(
    block, conditioned, causal
):
    block = block().eval()
    x, memory, dvec = torch.rand(3, 256, 100), torch.rand(3, 192, 100), torch.rand(3, 192)
    with torch.no_grad():
        out, memory_out = block(x, dvec, memory) if conditioned else block(x, memory)
    assert x.shape[-1] == out.shape[-1] == memory_out.shape[-1]

    if causal:  # no right context: the first poisoned frame is the first NaN out
        poisoned = x.clone()
        poisoned[..., 50:] = np.inf
        with torch.no_grad():
            out, _ = block(poisoned, memory)
        assert np.where(np.isnan(out))[-1][0] == 50


@pytest.mark.nnet
def test_split_and_merge_is_lossless():
    x = torch.rand(3, 256, 1000)
    assert torch.allclose(SplitMerge.merge(*SplitMerge.split(x, 40)), x)


@pytest.mark.nnet
def test_frequency_eq_layer_is_differentiable():
    FrequencyEQLayer()(torch.rand(1, 2, 257, 100)).sum().backward()


def _round_trip_error(wav, reconstructed, edge):
    n = min(wav.shape[-1], reconstructed.shape[-1])
    return float((wav[..., edge : n - edge] - reconstructed[..., edge : n - edge]).abs().max())


def _conv_encdec(n_fft, win_type, trainable):
    encoder = ConvEncDec(fft_length=n_fft, win_type=win_type, win_length=n_fft, sr=16000,
                         fmin=0, fmax=8000, freq_scale="no", trainable=trainable)
    return 16000, n_fft, encoder, lambda wav: encoder.inverse(encoder(wav))


def _unified(sr):
    """An int rate picks one encoder; a per-row tensor of rates picks per row."""
    rate = sr if isinstance(sr, int) else int(sr.item())
    encoder = UnifiedConvEncDec(win_type="hann", trainable=False)
    return rate, rate // 10, encoder, lambda wav: encoder.inverse(encoder(wav, sr), sr)


@pytest.mark.nnet
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _conv_encdec(512, "hann", False), id="conv-512-hann"),
        pytest.param(lambda: _conv_encdec(1024, "hamming", True), id="conv-1024-hamming-trainable"),
        pytest.param(lambda: _unified(8000), id="unified-8k"),
        pytest.param(lambda: _unified(torch.tensor([24000])), id="unified-24k-per-row"),
        pytest.param(lambda: _unified(48000), id="unified-48k"),
    ],
)
def test_stft_encoders_invert_their_own_transform(build):
    rate, edge, _, round_trip = build()
    wav, _ = AudioIO.open(f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=None,
                          resample_to=rate)
    assert _round_trip_error(wav, round_trip(wav), edge) < 1e-5


@pytest.mark.parametrize("improved", [False, True])
def test_self_attention_layer_keeps_its_shape_with_default_options(improved):
    """``improved`` replaces positional encoding with an LSTM, so the default
    ``position_encoding=True`` is ignored there rather than failing."""
    layer = MhaSelfAttenLayer(16, 32, nhead=4, improved=improved)
    x = torch.randn(2, 16, 10)
    assert layer(x).shape == x.shape


def test_unified_encoder_registers_every_rate_as_a_submodule():
    encoder = UnifiedConvEncDec(trainable=True).to(torch.float64)
    assert {key.split(".")[1] for key in encoder.state_dict()} == {
        "8000", "16000", "22050", "24000", "32000", "44100", "48000"
    }
    assert all(p.dtype == torch.float64 for p in encoder.parameters())
    assert len(list(encoder.parameters())) == 2 * 7  # wsin, wcos per rate


@pytest.mark.parametrize(
    "head",
    [
        pytest.param(lambda: IdentityHead(enc_channels=8, dim=8, kernel_t=5), id="identity"),
        pytest.param(lambda: ProximityHead(enc_channels=8, hidden=8, kernel_t=5), id="proximity"),
    ],
)
def test_bottleneck_heads_are_causal_in_time(head):
    """Changing the future must not change the past: a turn mean that can see a
    later talker's frames is a leak a turn-level loss would exploit, and a
    non-causal readout could not be streamed."""
    torch.manual_seed(0)
    head = head().eval()
    x = torch.randn(1, 8, 4, 60)
    future = x.clone()
    future[..., 40:] += 10.0
    with torch.no_grad():
        before, after = head(x), head(future)
    assert torch.allclose(before[:, :40], after[:, :40], atol=1e-6)
    assert not torch.allclose(before[:, 40:], after[:, 40:], atol=1e-3)


@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_fsmn_memory_taps_can_be_dilated_and_a_causal_one_stays_causal(dilation):
    torch.manual_seed(0)
    block = FSMN(8, 8, 8, 2, 0, dilation=dilation).eval()
    x = torch.randn(2, 8, 40)
    future = x.clone()
    future[..., 25:] += torch.randn_like(future[..., 25:])
    with torch.no_grad():
        before, _ = block(x)
        after, _ = block(future)
    assert before.shape == x.shape
    assert torch.equal(before[..., :25], after[..., :25])
    assert not torch.equal(before[..., 25:], after[..., 25:])


@pytest.mark.parametrize("kernel", [1, 3])
def test_causal_depthwise_separable_conv_keeps_every_frame_for_any_kernel(kernel):
    """A one-tap kernel needs no padding, so there is nothing to trim."""
    torch.manual_seed(0)
    conv = DepthwiseSeparableConv1d(4, 4, kernel=kernel, norm_cls="cLN", causal=True).eval()
    x = torch.randn(2, 4, 30)
    future = x.clone()
    future[..., 20:] += torch.randn_like(future[..., 20:])
    with torch.no_grad():
        before, after = conv(x), conv(future)
    assert before.shape == x.shape
    assert torch.allclose(before[..., :20], after[..., :20], atol=1e-6)


@pytest.mark.parametrize("freq", [16, 17])
@pytest.mark.parametrize(
    "build, channels",
    [
        pytest.param(lambda: SpectralTransform(4, 4), 4, id="spectral-transform"),
        pytest.param(lambda: FFC(8, 8), 8, id="ffc"),
    ],
)
def test_fourier_convolution_blocks_accept_an_odd_frequency_size(build, channels, freq):
    x = torch.randn(2, channels, freq, 7)
    assert build().eval()(x).shape == x.shape


def test_bidirectional_grouped_gru_feeds_every_channel_to_the_next_layer():
    """A bidirectional layer emits twice its hidden width; a layer stacked on it
    has to read all of it, not the first half."""
    torch.manual_seed(0)
    gru = GroupedGRU(16, 16, num_layers=2, groups=4, bidirectional=True).eval()
    x = torch.randn(2, 16, 10)
    widened = []

    def perturb_second_half(_module, args):
        widened.append(args[0].shape[1])
        features = args[0].clone()
        features[:, features.shape[1] // 2 :] += 1.0
        return (features, *args[1:])

    with torch.no_grad():
        plain = gru(x)
        gru.grus[1].register_forward_pre_hook(perturb_second_half)
        changed = gru(x)
    assert widened == [32] and plain.shape == (2, 32, 10)
    assert not torch.allclose(plain, changed)


@pytest.mark.parametrize("trainable", [True, False])
def test_weighted_sum_weights_follow_the_trainable_flag(trainable):
    layer = WeightedSum(4, trainable=trainable)
    assert [name for name, _ in layer.named_parameters()] == (["w"] if trainable else [])
    assert layer.w.requires_grad is trainable
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), x.mean(-1))
