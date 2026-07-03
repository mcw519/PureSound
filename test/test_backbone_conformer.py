import sys

import pytest
import torch

from puresound.nnet import TSConformer
from puresound.nnet.loss import VADHeadBCELoss
from puresound.streaming.conformer import (
    StreamingTSConformer,
    verify_streaming_consistency,
)

sys.path.insert(0, "./")


def _model(**overrides):
    args = dict(
        input_dim=256,
        enc_channels=48,
        n_blocks=4,
        chunk_size=4,
        left_context_chunks=3,
        right_lookahead_frames=0,
        distance_embedding_dim=64,
        vad_head={"enabled": True, "hidden": 32, "kernel_t": 4},
    )
    args.update(overrides)
    return TSConformer(**args).eval()


@pytest.mark.backbone
@pytest.mark.parametrize("n_frames", [16, 37, 100])
def test_tsconformer_shapes(n_frames):
    model = _model()
    x = torch.rand(2, 2, 256, n_frames)
    y = model(x, query_distance=torch.tensor([1.0, 0.6]))
    assert y.shape == x.shape
    assert model.last_vad_logits.shape == (2, n_frames)


@pytest.mark.backbone
def test_tsconformer_distance_zero_init_is_noop():
    # zero-initialised distance projection => conditioned == unconditioned at init
    model = _model()
    x = torch.rand(2, 2, 256, 60)
    with torch.no_grad():
        y0 = model(x)
        y1 = model(x, query_distance=torch.tensor([1.0, 0.6]))
    assert torch.allclose(y0, y1, atol=1e-6)


@pytest.mark.backbone
def test_tsconformer_baseline_no_subtasks():
    model = TSConformer(
        input_dim=256, dual_decoder=False, distance_embedding_dim=0, vad_head=None
    ).eval()
    y = model(torch.rand(1, 2, 256, 50))
    assert y.shape == (1, 2, 256, 50)
    assert model.last_vad_logits is None


@pytest.mark.backbone
@pytest.mark.parametrize("n_frames", [16, 37, 100])
def test_tsconformer_streaming_matches_full(n_frames):
    model = _model()
    x = torch.rand(2, 2, 256, n_frames)
    d = torch.tensor([1.0, 0.6])
    diff = verify_streaming_consistency(model, x, query_distance=d)
    assert diff < 1e-3, f"streaming diverged from full forward: {diff}"


@pytest.mark.backbone
def test_tsconformer_streaming_vad_matches_full():
    model = _model()
    x = torch.rand(1, 2, 256, 40)
    d = torch.tensor([0.8])
    with torch.no_grad():
        model(x, query_distance=d)
        full_vad = model.last_vad_logits.clone()
        runner = StreamingTSConformer(model)
        runner.reset(batch_size=1)
        runner.process_full(x, query_distance=d)
        stream_vad = runner.last_vad_logits
    assert full_vad.shape == stream_vad.shape
    assert torch.allclose(full_vad, stream_vad, atol=1e-3)


@pytest.mark.nnet
def test_vad_head_bce_loss():
    loss = VADHeadBCELoss(false_positive_weight=2.0)
    assert loss.uses_vad_logits is True
    # tolerates off-by-one frame counts
    out = loss(torch.randn(2, 100), (torch.rand(2, 101) > 0.5).float())
    assert torch.isfinite(out)
    with pytest.raises(ValueError):
        loss(None, torch.rand(2, 100))
    with pytest.raises(ValueError):
        loss(torch.randn(2, 100), None)
