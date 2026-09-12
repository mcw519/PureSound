"""The frame grid turn pooling happens on (v20 R1a).

Both new losses index per-frame model outputs with per-frame labels, and the two
grids are *nominally* the same 100 fps (hop 160 at 16 kHz) but are not the same
length: the labeler's analysis window is 400 samples
(`puresound.audio.vad.frame_count`) and the STFT encoder's is 512, so the label
grid runs up to one frame longer than the bottleneck's. A silent off-by-one here
is a loss trained on labels shifted against the audio, which no scorecard would
attribute to the pooling. So: pin the two grids against the real pipeline, and
pin that pooling takes the common prefix with frame 0 aligned to frame 0.
"""

import pytest
import torch

from puresound.audio.vad import EnergyVADLabeler, frame_count
from puresound.nnet import ConvEncDec, FeatureEncoder
from puresound.nnet.dpcrn import DPCRN
from puresound.nnet.loss import identity as identity_module
from puresound.nnet.loss import proximity as proximity_module
from puresound.nnet.loss.identity import align_frames, align_turn_frames, pool_turn_means

HOP = 160
LABEL_WINDOW = 400
STFT_WINDOW = 512


def pipeline():
    """The shipped DPCRN time geometry (fft 512 / hop 160, kernel_t 2, delay 1),
    narrowed in channels -- the frame count is set by the encoder and the
    transpose-conv trimming, not by the widths."""
    encoder = ConvEncDec(
        fft_length=512, win_type="hann", win_length=512, hop_length=HOP,
        fmin=0, fmax=8000, sr=16000, trainable=False,
    )
    feats = FeatureEncoder(
        feats_type="complex", drop_stft_first_bin=True, trainable=False,
        include_specaug=False,
    )
    backbone = DPCRN(
        input_dim=256, channels=(2, 8, 16, 32), kernel_t=(2, 2, 2),
        stride_t=(1, 1, 1), dilation_t=(1, 1, 1), kernel_f=(5, 3, 3),
        stride_f=(2, 2, 1), dilation_f=(1, 1, 1), delay=(1, 1, 1), rnn_hidden=16,
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    ).eval()
    return encoder, feats, backbone


@pytest.mark.parametrize("samples", [16000, 32000, 12345, 192000])
def test_the_head_grid_is_the_stft_grid_and_the_label_grid_is_within_one_frame(samples):
    """Measured, not assumed. ``ceil(samples/160)`` -- the round figure for
    "100 fps" -- is neither of them: at 16 kHz it is 100 against the labeler's
    98 and the head's 97, because both grids drop the frames a full analysis
    window does not fit in.
    """
    encoder, feats, backbone = pipeline()
    with torch.no_grad():
        spectrum, _ = feats(encoder(torch.zeros(1, samples)))
        backbone(spectrum)

    head_frames = backbone.last_identity_emb.shape[1]
    assert head_frames == backbone.last_proximity.shape[1]
    assert head_frames == backbone.last_bottleneck_graph.shape[-1]
    assert head_frames == spectrum.shape[-1]
    assert head_frames == (samples - STFT_WINDOW) // HOP + 1

    label_frames = frame_count(samples, LABEL_WINDOW, HOP)
    assert label_frames == (samples - LABEL_WINDOW) // HOP + 1
    assert 0 <= label_frames - head_frames <= 1
    # And the labeler really does produce that grid.
    labels = EnergyVADLabeler(frame_length=LABEL_WINDOW, hop_length=HOP)(
        torch.randn(1, samples)
    )
    assert labels.shape[-1] == label_frames


def test_pooling_takes_the_common_prefix_rather_than_padding_either_side():
    head_frames, label_frames = 97, 98
    values = torch.zeros(1, head_frames, 1)
    turn_id = torch.ones(1, label_frames, dtype=torch.long)
    (aligned,), ids, _ = align_turn_frames(
        [values], {"turn_id": turn_id}, torch.device("cpu")
    )
    assert aligned.shape[1] == ids.shape[1] == head_frames
    assert align_frames(values, turn_id)[1].shape[1] == head_frames


def test_a_turn_pools_exactly_the_frames_its_label_names():
    """The direct test for a shift: the per-frame value IS the frame index, so
    the pooled mean is arithmetic and a one-frame offset changes it."""
    n_frames = 40
    values = torch.arange(n_frames, dtype=torch.float32).view(1, n_frames, 1)
    turn_id = torch.zeros(1, n_frames, dtype=torch.long)
    turn_id[:, 5:11] = 1     # frames 5..10
    turn_id[:, 20:24] = 2    # frames 20..23

    means, counts = pool_turn_means(values, turn_id, n_turns=2)
    assert counts.tolist() == [[6, 4]]
    assert torch.allclose(means[0, 0], torch.tensor([7.5]))    # mean(5..10)
    assert torch.allclose(means[0, 1], torch.tensor([21.5]))   # mean(20..23)


def test_a_turn_running_past_the_head_s_last_frame_pools_only_what_exists():
    """The label grid is the longer one, so the final turn of every row is the
    one that gets truncated. Its mean must be over the frames the model
    actually produced -- silently zero-padding the model output would pull every
    row's last turn toward zero."""
    head_frames, label_frames = 20, 22
    values = torch.arange(head_frames, dtype=torch.float32).view(1, head_frames, 1)
    turn_id = torch.zeros(1, label_frames, dtype=torch.long)
    turn_id[:, 16:22] = 1  # 6 label frames, only 4 of them exist in the output

    (aligned,), ids, _ = align_turn_frames(
        [values], {"turn_id": turn_id}, torch.device("cpu")
    )
    means, counts = pool_turn_means(aligned, ids, n_turns=1)
    assert counts.tolist() == [[4]]
    assert torch.allclose(means[0, 0], torch.tensor([17.5]))  # mean(16..19)


def test_a_turn_id_beyond_k_max_is_named_rather_than_mis_binned():
    values = torch.zeros(1, 10, 1)
    turn_id = torch.zeros(1, 10, dtype=torch.long)
    turn_id[:, 3:6] = 4
    with pytest.raises(ValueError, match="turn_id reaches 4"):
        pool_turn_means(values, turn_id, n_turns=2)


def test_the_pooling_arithmetic_stays_float32_under_bf16_autocast():
    """The shipped recipe trains at ``precision: bf16-mixed``.

    bf16 has 8 mantissa bits, so a 12 s turn's frame count -- 1197 -- is not
    representable and accumulates to 1200, and the sum of its frames is wrong by
    the same 0.25 %. The turn mean would be quietly mis-scaled, worst for the
    longest turns, which are exactly the ones the session objective is built
    around. ``autocast(enabled=False)``, not a bare cast, for the reason
    `VADHead._ema_bank` records.
    """
    n = 1197
    values = torch.full((1, n, 1), 0.5).bfloat16()  # 0.5 is exact in bf16
    turn_id = torch.ones(1, n, dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        means, counts = pool_turn_means(values, turn_id, n_turns=1)

    assert means.dtype is torch.float32
    assert int(counts[0, 0]) == n            # a bf16 accumulator reports 1200
    assert abs(float(means[0, 0]) - 0.5) < 1e-6  # ...and would divide by it


def test_both_losses_pool_through_the_same_implementation():
    """One definition of "which grid do turns pool on". Two copies of this
    arithmetic is how the identity loss and the proximity loss end up
    disagreeing about frame 0 without either being obviously wrong."""
    assert proximity_module.align_turn_frames is identity_module.align_turn_frames
    assert proximity_module.pool_turn_means is identity_module.pool_turn_means
    assert proximity_module.eligible_turns is identity_module.eligible_turns
