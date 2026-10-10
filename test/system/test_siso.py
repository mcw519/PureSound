"""``EncDecMaskBase``: what it hands each loss, the optional training terms it
adds, and the optimizer groups it builds.

Most tests bypass Lightning's ``__init__`` (``object.__new__`` plus
``nn.Module.__init__``): ``compute_loss`` and ``_loss_providers`` touch only the
loss list and the backbone, and building the encoder / feature / optimizer
machinery would test nothing more.
"""

import pytest
import torch
import torch.nn as nn

from puresound.config.recipe import OptimizerConfig, SchedulerConfig
from puresound.nnet import DPCRN, ConvEncDec, FeatureEncoder
from puresound.nnet.lobe.heads import VADHead
from puresound.nnet.loss import (
    BackgroundVADHeadBCELoss,
    ResidualReferenceLoss,
    VADHeadBCELoss,
)
from puresound.streaming import create_streaming_dpcrn_model
from puresound.system import siso
from puresound.system.base import invoke_loss
from puresound.system.optim import create_optimizer_and_scheduler
from puresound.system.siso import EncDecMaskBase

HEAD_PROVIDERS = ("bottleneck", "identity_emb", "identity_head", "proximity")
#: Listed rather than derived: the point is that adding to the table did not
#: rename or drop anything a registered loss already asks for.
CORE_PROVIDERS = (
    "enhanced", "target", "batch", "inactive_labels", "vad_target", "vad_logits",
    "background_vad_logits", "background_vad_target", "dist_preds",
)


class _BackboneStub:
    """Exposes only the side outputs ``compute_loss`` reads."""

    def __init__(self, vad_logits=None, background_vad_logits=None):
        self.last_vad_logits = vad_logits
        self.last_background_vad_logits = background_vad_logits
        self.last_aux_outputs = {}


def _bare(backbone, losses=(), weights=None):
    module = object.__new__(EncDecMaskBase)
    torch.nn.Module.__init__(module)
    module.backbone = backbone
    module.loss_func_list = torch.nn.ModuleList(losses)
    module.loss_func_list_w = list(weights or [1.0] * len(losses))
    return module


def _providers(backbone, batch=None):
    return _bare(backbone)._loss_providers(
        enhanced=None, target=None, vad_target=None, batch=batch, inactive_labels=None
    )


def tiny_dpcrn(**kw):
    return DPCRN(
        input_dim=64, channels=(1, 8, 16), kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=16, **kw,
    )


def dpcrn_system(fft=64, hop=32, sr=16000, input_dim=32, channels=(2, 4, 8), rnn_hidden=4,
                 backbone_args=None, **module_args):
    """A real ``EncDecMaskBase`` over a narrow DPCRN."""
    encoder = ConvEncDec(fft_length=fft, win_type="hann", win_length=fft, hop_length=hop,
                         fmin=0, fmax=sr // 2, sr=sr, trainable=False)
    feats = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False,
                           include_specaug=False)
    backbone = DPCRN(
        input_dim=input_dim, channels=channels, kernel_t=(2, 2), stride_t=(1, 1),
        dilation_t=(1, 1), kernel_f=(5, 3), stride_f=(2, 2), dilation_f=(1, 1),
        delay=(0, 0), rnn_hidden=rnn_hidden, **(backbone_args or {}),
    )
    return EncDecMaskBase(encoder, feats, backbone, mask_type="complex", **module_args)


def streaming_system(**backbone_args):
    """The shipped DPCRN time geometry (fft 512 / hop 160), narrow."""
    return dpcrn_system(fft=512, hop=160, input_dim=256, channels=(2, 8, 16), rnn_hidden=16,
                        backbone_args={"vad_head": {"enabled": True}, **backbone_args})


# --------------------------------------------------------------------------- #
# what a loss is handed
# --------------------------------------------------------------------------- #


def test_the_provider_table_keeps_every_name_and_off_heads_answer_none():
    """Off by default means the provider answers None, so a misconfigured recipe
    hits the loss's own "enable the head" error rather than a shape mismatch."""
    backbone = tiny_dpcrn()
    backbone(torch.randn(2, 1, 64, 20))
    table = _providers(backbone)
    assert set(CORE_PROVIDERS) | set(HEAD_PROVIDERS) <= set(table)
    for name in HEAD_PROVIDERS:
        assert table[name]() is None, name


def test_the_head_providers_hand_over_the_forward_s_side_outputs():
    backbone = tiny_dpcrn(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
    )
    backbone(torch.randn(3, 1, 64, 24))
    table = _providers(backbone)

    emb = table["identity_emb"]()
    assert emb.shape == (3, 24, 8)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(3, 24), atol=1e-5)
    assert table["proximity"]().shape == (3, 24)
    # The module itself, not a tensor: the identity loss keeps an EMA copy of it.
    assert table["identity_head"]() is backbone.identity_head


def test_the_background_head_loss_gets_background_inputs_and_zeros_when_absent():
    """``BackgroundVADHeadBCELoss`` subclasses ``VADHeadBCELoss``; it must be
    handed the background head's logits and target, never the foreground ones.
    A batch where no row has background speech carries no target at all -- a
    valid state meaning "no background activity" -- so the target is
    synthesized as zeros and the result equals an explicit zero target."""
    assert BackgroundVADHeadBCELoss.required_inputs == (
        "background_vad_logits", "background_vad_target",
    )
    assert VADHeadBCELoss.required_inputs == ("vad_logits", "vad_target")

    foreground, background = torch.full((2, 8), 3.0), torch.full((2, 8), -3.0)
    loss = _Recorder(BackgroundVADHeadBCELoss.required_inputs)
    invoke_loss(loss, _providers(_BackboneStub(foreground, background), batch={}))
    assert torch.equal(loss.seen[0], background)
    assert torch.equal(loss.seen[1], torch.zeros_like(background))

    logits = torch.randn(2, 19)
    enhanced, target = torch.randn(2, 1600), torch.randn(2, 1600)
    synthesized, losses = _bare(
        _BackboneStub(background_vad_logits=logits), [BackgroundVADHeadBCELoss()]
    ).compute_loss(enhanced, target, vad_target=None, batch={})
    explicit, _ = _bare(
        _BackboneStub(background_vad_logits=logits), [BackgroundVADHeadBCELoss()]
    ).compute_loss(enhanced, target, vad_target=None,
                   batch={"background_vad_target": torch.zeros_like(logits)})
    assert torch.isfinite(synthesized) and len(losses) == 1
    assert torch.allclose(synthesized, explicit)


def test_a_missing_foreground_vad_target_is_a_configuration_error():
    """The zero-fill is background-only: missing foreground labels are not a
    valid silent state."""
    module = _bare(_BackboneStub(vad_logits=torch.randn(2, 19)), [VADHeadBCELoss()])
    with pytest.raises(ValueError, match="vad_target"):
        module.compute_loss(torch.randn(2, 1600), torch.randn(2, 1600), vad_target=None, batch={})


def test_the_residual_reference_loss_reads_its_context_from_the_batch():
    enhanced = torch.tensor([[0.8, 0.1, -0.2, 0.0]])
    noisy = torch.tensor([[1.0, 0.0, 0.0, 0.5]])
    overall, losses = _bare(_BackboneStub(), [ResidualReferenceLoss(loss="l1")], [0.2]).compute_loss(
        enhanced, enhanced.clone(),
        batch={"noisy_speech": noisy, "consistency_noise": noisy - enhanced,
               "target_present": torch.ones(1)},
    )
    assert torch.allclose(overall, torch.tensor(0.0))
    assert losses == [0.0]


class _Recorder(torch.nn.Module):
    """A loss that reports what it was handed instead of computing anything."""

    def __init__(self, required_inputs):
        super().__init__()
        self.required_inputs = required_inputs
        self.seen = None

    def forward(self, *args):
        self.seen = args
        return torch.zeros(())


# --------------------------------------------------------------------------- #
# training-only heads
# --------------------------------------------------------------------------- #


def test_turn_losses_on_a_batch_without_session_keys_are_a_graph_carrying_zero():
    """A batch without the session keys must not change a recipe's number, and
    must still carry a graph edge so DDP never reports an unused head."""
    from puresound.nnet.loss import IdentityContrastiveLoss, RelativeProximityLoss

    torch.manual_seed(0)
    model = streaming_system(
        identity_head={"enabled": True, "dim": 8, "kernel_t": 3},
        proximity_head={"enabled": True, "hidden": 8},
        expose_bottleneck=True,
    )
    model.register_loss_func(
        torch.nn.ModuleList([IdentityContrastiveLoss(), RelativeProximityLoss()]), [1.0, 1.0]
    )
    model.train()
    noisy = torch.randn(2, 8000) * 0.1
    total, values = model.compute_loss(
        enhanced=model(noisy), target=torch.randn(2, 8000) * 0.1, batch={}
    )
    assert values == [0.0, 0.0] and float(total) == 0.0
    total.backward()


def test_training_only_heads_leave_inference_and_the_streaming_export_alone():
    """``forward`` flips ``stash_bottleneck`` for a presence gate, but
    ``expose_bottleneck`` is a build-time flag: an inference call that started
    retaining the graph would leak memory per frame.

    The frame model enumerates exportable heads by name, so identity and
    proximity heads add no state port and no output. Streaming them later means
    updating this test on purpose, not discovering a manifest mismatch."""
    model = streaming_system()
    with torch.no_grad():
        model(torch.randn(1, 16000))
    assert model.backbone.expose_bottleneck is False
    assert model.backbone.last_bottleneck_graph is None

    off = create_streaming_dpcrn_model(model)
    on = create_streaming_dpcrn_model(
        streaming_system(identity_head={"enabled": True}, proximity_head={"enabled": True},
                         expose_bottleneck=True)
    )
    assert on.state_input_names == off.state_input_names
    assert on.state_output_names == off.state_output_names
    assert on.extra_output_names == off.extra_output_names == ["vad_logit"]
    assert [name for name, _ in on.heads] == ["vad_head"]


def test_gate_only_training_emits_frame_logits_and_keeps_the_waveform_shape():
    model = dpcrn_system(
        backbone_args={"vad_head": {"enabled": True, "hidden": 8, "kernel_t": 5}},
        train_vad_head_only=True, gate_head_lr_factor=1.0,
    ).eval()
    wav = torch.randn(2, 1024)
    assert model(wav).shape == wav.shape
    logits = model.backbone.last_vad_logits
    assert logits is not None and logits.shape[0] == 2 and logits.shape[-1] > 0


# --------------------------------------------------------------------------- #
# channel-consistency regularizer
# --------------------------------------------------------------------------- #


class MaskEchoSystem(EncDecMaskBase):
    """Forward stub whose 'mask' depends on the input, so it is channel-sensitive."""

    def __init__(self, **kw):
        super().__init__(encoder=torch.nn.Identity(), feats=torch.nn.Identity(),
                         backbone=torch.nn.Identity(), **kw)
        self.p = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, noisy):
        self.last_mask = noisy * self.p
        return noisy


def _wire(system):
    system.logged = []
    system.log = lambda *args, **kwargs: system.logged.append((args, kwargs))
    system.register_loss_func(torch.nn.ModuleList([torch.nn.L1Loss()]), [1.0])
    return system


def _consistency(**config):
    return _wire(MaskEchoSystem(channel_consistency=config or None))


def _fired(system):
    return any(args[0] == "train_step_cons_loss" for args, _ in system.logged)


def _batch():
    torch.manual_seed(0)
    return {"noisy_speech": torch.randn(2, 256).clamp(-1, 1), "clean_speech": torch.zeros(2, 256)}


def _fail_first_irfft(monkeypatch):
    real_irfft, calls = torch.fft.irfft, {"n": 0}

    def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("cuFFT error: CUFFT_INTERNAL_ERROR")
        return real_irfft(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "irfft", fail_once)


@pytest.mark.parametrize(
    "shape, fft_fails", [((3, 512), False), ((2, 1, 4096), True)], ids=["plain", "fft-fails"]
)
def test_the_channel_perturbation_changes_the_signal_but_not_its_shape(
    monkeypatch, shape, fft_fails
):
    """A device FFT plan that fails once costs that step's perturbation nothing:
    it is retried, and the result is still a perturbation, not a passthrough."""
    if fft_fails:
        _fail_first_irfft(monkeypatch)
    wav = torch.randn(*shape).clamp(-1, 1)
    out = _consistency(enabled=True, prob=1.0)._random_channel_perturb(wav)
    assert out.shape == wav.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, wav, atol=1e-4)
    assert out.abs().max() <= 1.0 and not out.requires_grad


def test_a_failed_fft_plan_falls_back_to_the_same_filter(monkeypatch):
    """A plan can fail to allocate on a busy device, which says nothing about the
    row; losing a multi-day run to it is the outcome worth engineering away."""
    system = _consistency(enabled=True, prob=1.0)
    wav = torch.randn(3, 4096).clamp(-1, 1)
    gain = torch.ones(3, wav.shape[-1] // 2 + 1)
    expected = torch.fft.irfft(torch.fft.rfft(wav, dim=-1) * gain, n=wav.shape[-1], dim=-1)

    _fail_first_irfft(monkeypatch)
    before = siso.EncDecMaskBase._fft_fallbacks
    out = system._filter_via_fft(wav, gain, wav.shape[-1])
    assert torch.allclose(out, expected, atol=1e-5)
    assert out.dtype == wav.dtype and out.device == wav.device
    assert siso.EncDecMaskBase._fft_fallbacks == before + 1, "the fallback is counted"


def test_the_consistency_term_is_logged_differentiable_and_capped_by_max_rows():
    system = _consistency(enabled=True, prob=1.0, weight=1.0, max_rows=1)
    seen = []
    forward = system.forward
    system.forward = lambda noisy: seen.append(noisy.shape[0]) or forward(noisy)

    out = system.training_step(_batch(), 0)
    assert _fired(system)
    # Main forward on the full batch, then the perturbed leading sub-batch.
    assert seen == [2, 1]
    out["loss"].backward()
    assert system.p.grad is not None and system.p.grad.abs() > 0


def test_the_consistency_schedule_is_a_function_of_the_batch_index_and_off_is_a_no_op():
    """prob 0.3 with period 10 fires on batch_idx % 10 in {0, 1, 2}. A per-rank
    random draw would desync SyncBN collectives and deadlock DDP."""
    system = _consistency(enabled=True, prob=0.3, weight=1.0)
    fired = []
    for idx in range(10):
        system.logged = []
        system.training_step(_batch(), idx)
        fired.append(_fired(system))
    assert fired == [True, True, True] + [False] * 7

    absent, disabled = _consistency(), _consistency(enabled=False, prob=1.0)
    loss_a = absent.training_step(_batch(), 0)["loss"]
    loss_b = disabled.training_step(_batch(), 0)["loss"]
    assert torch.isclose(loss_a, loss_b)
    assert not _fired(absent) and not _fired(disabled)


def test_the_consistency_term_runs_through_a_real_dpcrn():
    system = _wire(dpcrn_system(channel_consistency={"enabled": True, "prob": 1.0, "weight": 0.5}))
    torch.manual_seed(1)
    batch = {"noisy_speech": torch.randn(2, 1024).clamp(-1, 1),
             "clean_speech": torch.randn(2, 1024).clamp(-1, 1)}
    assert torch.isfinite(system.training_step(batch, 0)["loss"])
    assert _fired(system)


# --------------------------------------------------------------------------- #
# optimizer groups
# --------------------------------------------------------------------------- #


def test_param_groups_carry_lr_factors_into_the_optimizer():
    encoder = ConvEncDec(fft_length=64, win_type="hann", win_length=64, hop_length=32,
                         fmin=0, fmax=4000, sr=8000, trainable=False)
    feats = FeatureEncoder(feats_type="complex", drop_stft_first_bin=True, trainable=False,
                           include_specaug=False)

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(2, 2, 1)
            self.vad_head = VADHead(enc_channels=2, hidden=4, kernel_t=3)

        def forward(self, x):
            return self.proj(x)

    model = EncDecMaskBase(encoder=encoder, feats=feats, backbone=TinyBackbone(),
                           mask_type="complex", encoder_lr_factor=0.1, feats_lr_factor=0.5,
                           backbone_lr_factor=1.0)
    groups = model.get_total_param_groups()
    assert set(groups) == {"encoder", "feats", "backbone"}

    optimizer, scheduler = create_optimizer_and_scheduler(
        groups,
        OptimizerConfig(type="AdamW", learning_rate=1e-3, args={"weight_decay": 0.0}),
        SchedulerConfig(type="CosineAnnealingWarmRestarts", warmup_step=0, args={"T_0": 20}),
    )
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 5e-4, 1e-3]
    assert scheduler.optimizer is optimizer
