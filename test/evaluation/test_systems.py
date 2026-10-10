"""Loading a system, and the mismatch that must never load silently."""

import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.evaluation.systems import (
    CheckpointSystem,
    Passthrough,
    PrecomputedSystem,
    load_system,
    run_system,
)


class _Gain(torch.nn.Module):
    def __init__(self, gain=0.5):
        super().__init__()
        self.gain = gain

    def forward(self, wav, dry_blend=1.0, **kwargs):
        return wav * self.gain


class _TupleModel(torch.nn.Module):
    def forward(self, wav, dry_blend=1.0, **kwargs):
        return wav * 0.5, {"aux": 1}


def _write(path, wav, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioIO.save(wav.reshape(1, -1), str(path), sr)


def test_no_checkpoint_means_the_untouched_baseline_and_a_checkpoint_needs_its_recipe():
    """The baseline is what lets the gate run before a model exists."""
    for recipe in (None, "recipe.yaml"):
        system = load_system(recipe, None)
        assert isinstance(system, Passthrough)
    wav = torch.randn(1, 16000)
    assert torch.equal(system.process(wav, 16000), wav)
    assert system.describe() == {"system": "passthrough"}

    with pytest.raises(ValueError, match="recipe"):
        load_system(None, "some.ckpt")


@pytest.mark.parametrize(
    "model, chunk_seconds",
    [(_Gain(), None), (_Gain(), 0.3), (_TupleModel(), None)],
    ids=["whole", "chunked-with-short-tail", "tuple-output"],
)
def test_a_checkpoint_system_returns_the_whole_waveform(model, chunk_seconds):
    """Chunking is off by default because context changes the answer; when on,
    a 1 s signal in 0.3 s chunks still comes back whole, remainder included."""
    assert CheckpointSystem(model=_Gain(), device=torch.device("cpu")).chunk_seconds is None
    system = CheckpointSystem(model=model, device=torch.device("cpu"), chunk_seconds=chunk_seconds)
    wav = torch.ones(1, 16000)
    out = run_system(system, "ignored", wav, 16000)
    assert out.shape == wav.shape
    assert torch.allclose(out, wav * 0.5)


def test_describe_carries_what_a_record_needs(tmp_path):
    described = CheckpointSystem(
        model=_Gain(), device=torch.device("cpu"),
        checkpoint="a.ckpt", recipe="b.yaml", dry_blend=0.9,
    ).describe()
    assert (described["dry_blend"], described["checkpoint"], described["recipe"]) == (
        0.9, "a.ckpt", "b.yaml"
    )

    out = tmp_path / "dfn3"
    out.mkdir()
    assert PrecomputedSystem(out, "dfn3").describe() == {
        "system": "precomputed", "directory": str(out.resolve()), "name": "dfn3"
    }


@pytest.mark.parametrize(
    "filename, file_rate",
    [("p232_001_mix.wav", 16000), ("p232_001_mix.wav", 48000),
     ("p232_001_mix_DeepFilterNet3.wav", 16000)],
    ids=["exact-stem", "resampled-to-stage-rate", "unique-tool-suffix"],
)
def test_a_precomputed_system_returns_the_file_named_by_the_item_stem(tmp_path, filename, file_rate):
    """DeepFilterNet writes <stem>_DeepFilterNet3.wav; one such match is the file."""
    out = tmp_path / "theirs"
    tone = 0.1 * torch.sin(torch.arange(file_rate) / file_rate * 2 * torch.pi * 440)
    _write(out / filename, tone, sr=file_rate)
    system = PrecomputedSystem(out, "theirs")
    assert system.resolve("p232_001_mix").name == filename
    got = run_system(system, "p232_001_mix", torch.zeros(1, 16000), 16000)
    assert abs(got.shape[-1] - 16000) <= 2
    if file_rate == 16000:
        assert torch.allclose(got.reshape(-1), tone, atol=2e-4)  # 16-bit round trip


def test_a_stem_is_matched_literally_not_as_a_glob_pattern(tmp_path):
    """`take[1]` as a pattern means `take1`, which is another item's output."""
    out = tmp_path / "theirs"
    _write(out / "take[1]_DeepFilterNet3.wav", torch.zeros(8000))
    _write(out / "take1_DeepFilterNet3.wav", torch.zeros(8000))
    assert PrecomputedSystem(out, "theirs").resolve("take[1]").name == "take[1]_DeepFilterNet3.wav"


@pytest.mark.parametrize(
    "files, error, match",
    [
        (["ns00007_mix_a.wav", "ns00007_mix_b.wav"], ValueError, "2 outputs match"),
        ([], FileNotFoundError, "no output for"),
        (None, FileNotFoundError, "directory not found"),
        ("waveform-only", TypeError, "run_system"),
    ],
    ids=["two-candidates", "missing-output", "missing-directory", "waveform-only-call"],
)
def test_a_precomputed_system_refuses_rather_than_guesses(tmp_path, files, error, match):
    """Two candidates, a missing file or a call with no item key would all score
    the wrong audio -- or silence -- if they did not raise."""
    out = tmp_path / "dfn"
    with pytest.raises(error, match=match):
        if files is None:
            PrecomputedSystem(tmp_path / "nope", "dfn")
        elif files == "waveform-only":
            out.mkdir()
            PrecomputedSystem(out, "dfn").process(torch.zeros(1, 10), 16000)
        else:
            out.mkdir()
            for name in files:
                _write(out / name, torch.zeros(8000))
            PrecomputedSystem(out, "dfn").resolve("ns00007_mix")
