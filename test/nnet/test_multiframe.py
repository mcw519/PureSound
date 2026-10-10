"""The deep-filter residual head: what the filter computes, and that adding it
to a trained mask-only network changes nothing until it has learned something.

The zero-init contract is what lets a mask-only checkpoint warm-start the new
model with no re-initialisation debt; if it breaks, a "warm start" silently
begins from a different model than the one that was trained.
"""

import copy
from pathlib import Path

import torch
import yaml

from puresound.nnet.lobe.multiframe import DeepFilterResidualHead, deep_filter_residual
from puresound.nnet.masker import Masker

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn_heads.yaml"


def _complex(x):
    return torch.complex(x[:, 0], x[:, 1])


def test_the_filter_is_a_complex_fir_over_past_frames():
    """Tap k multiplies frame t - k: a unit coefficient on one tap is a pure
    delay, and arbitrary coefficients are a complex multiply-accumulate."""
    torch.manual_seed(0)
    n, f, t, order = 2, 6, 10, 4
    spec = torch.randn(n, 2, f, t)
    history = torch.nn.functional.pad(spec, (order - 1, 0))
    for k in range(order):
        coefs = torch.zeros(n, order, 2, f, t)
        coefs[:, k, 0] = 1.0
        expected = torch.nn.functional.pad(spec, (k, 0))[..., :t]
        assert torch.allclose(deep_filter_residual(history, coefs), expected), f"tap {k}"

    coefs = torch.randn(n, order, 2, f, t)
    out = _complex(deep_filter_residual(history, coefs))
    x = torch.nn.functional.pad(_complex(spec), (order - 1, 0))
    ref = sum(torch.complex(coefs[:, k, 0], coefs[:, k, 1]) * x[..., order - 1 - k : order - 1 - k + t]
              for k in range(order))
    assert torch.allclose(out, ref, atol=1e-6)


def test_mask_plus_residual_is_causal_and_is_the_plain_mask_without_coefficients():
    torch.manual_seed(2)
    n, c, t, bins, order = 1, 16, 12, 8, 5
    spec = torch.randn(n, 2, c, t)
    mask = torch.randn(n, 2, c, t)
    coefs = torch.randn(n, order, 2, bins, t)
    before = Masker.apply_complex_mask_with_df(spec, mask, coefs)
    changed = spec.clone()
    changed[..., 7:] += torch.randn_like(changed[..., 7:])
    after = Masker.apply_complex_mask_with_df(changed, mask, coefs)
    assert torch.equal(before[..., :7], after[..., :7])
    assert not torch.equal(before[..., 7:], after[..., 7:])

    assert torch.equal(Masker.apply_complex_mask_with_df(spec, mask, None),
                       Masker.apply_complex_mask_on_reim(spec, mask))


def test_the_head_starts_at_zero_can_still_learn_and_interleaves_its_bins():
    """Zero last layer: output 0 (so warm starts are exact), but its own weight
    gets a gradient on the first step -- a head that could not move would turn
    training it into a no-op that looks like a negative result. Fine bin f
    comes from coarse bin f // upsample."""
    torch.manual_seed(4)
    head = DeepFilterResidualHead(in_channels=8, upsample=2, bins=12, order=5, hidden=16)
    coefs = head(torch.randn(2, 8, 10, 7, requires_grad=True))
    assert coefs.shape == (2, 5, 2, 12, 7)
    assert torch.count_nonzero(coefs) == 0
    ((coefs - torch.randn_like(coefs)) ** 2).mean().backward()
    assert head.out.weight.grad is not None and head.out.weight.grad.abs().sum() > 0

    head = DeepFilterResidualHead(in_channels=1, upsample=2, bins=6, order=1, hidden=1)
    with torch.no_grad():
        head.proj[0].weight.fill_(1.0); head.proj[0].bias.zero_()
        head.proj[1].weight.fill_(1.0)                    # PReLU slope: identity for x>0
        head.out.weight.fill_(1.0); head.out.bias.zero_()
    feat = torch.tensor([0.1, 0.2, 0.3, 9.0]).reshape(1, 1, 4, 1)   # 4th coarse bin unused
    coefs = head(feat)[0, 0, 0, :, 0]
    assert torch.allclose(coefs, torch.tanh(torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])))


def _system(tmp_path, df_head):
    from puresound.streaming import load_streaming_dpcrn_model

    config = copy.deepcopy(yaml.safe_load(open(_CONFIG)))
    args = config["model"]["backbone"]["backbone_args"]
    args["inter_type"] = "lstm"
    if df_head:
        args["df_head"] = df_head
    path = tmp_path / ("df.yaml" if df_head else "plain.yaml")
    yaml.safe_dump(config, open(path, "w"))
    return load_streaming_dpcrn_model(str(path)).system_model.eval()


def test_a_mask_only_checkpoint_loads_into_the_df_model_and_computes_the_same(tmp_path):
    """The warm-start contract: every missing key belongs to the new head, and
    the output is bit-identical until the head learns."""
    torch.manual_seed(5)
    plain = _system(tmp_path, None)
    with_df = _system(tmp_path, {"bins": 128, "order": 5, "hidden": 32})
    missing, unexpected = with_df.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(k.startswith("backbone.df_head.") for k in missing), missing

    wav = 0.1 * torch.randn(1, 16000)
    with torch.no_grad():
        a, b = plain(wav), with_df(wav)
    assert with_df.backbone.last_df_coefs is not None
    assert torch.equal(a, b)
