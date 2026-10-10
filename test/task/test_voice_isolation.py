"""`VoiceIsolationDataset`: the real-recording pools, the distance-level SIR and
the scalar metadata each row carries.

The real far-field branch replaces the clean-speech-convolved-with-RIR far
channel with a genuine loudspeaker->air->mic recording drawn from a pool
manifest, no RIR applied; the real near-field branch makes a keep row whose
foreground is such a recording.
"""

import json
import math

import pytest
import torch

from puresound.config.augmentation import MixModeEntry
from puresound.task.voice_isolation import MIX_MODE_CODES, VoiceIsolationDataset

NAN = float("nan")

SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 1.0,
    "add_n_cases": 1,
    "snr_range": [-5, 5],
}


def _dataset_args(metafile_path, **overrides):
    args = {
        "metafile_path": str(metafile_path),
        "min_utt_length_in_seconds": 0.05,
        "min_utts_in_each_speaker": 2,
        "target_sr": 16000,
        "training_sample_length_in_seconds": 0.1,
        "audio_gain_normalized_to": None,
    }
    args.update(overrides)
    return args


def _write_pool(path, write_tone_wav, entries, duration=0.3, base_freq=300.0):
    """A pool manifest of tone wavs; ``entries`` is one (room, distance) per file."""
    lines = []
    for i, (room, distance) in enumerate(entries):
        wav_path = path.parent / path.stem / f"utt{i}.wav"
        write_tone_wav(wav_path, sample_rate=16000, duration=duration,
                       freq=base_freq + 40 * i)
        lines.append(json.dumps({"wav_path": str(wav_path), "room": room,
                                 "distance_m": distance, "speaker": f"{path.stem}{i}"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _far_pool(tmp_path, write_tone_wav, **kw):
    entries = [("rm1", 1.5 + i) for i in range(4)]
    return _write_pool(tmp_path / "realfar.jsonl", write_tone_wav, entries, **kw)


def _near_pool(tmp_path, write_tone_wav, room="rm1", **kw):
    entries = [(room, 0.74)] * 3
    return _write_pool(tmp_path / "realnear.jsonl", write_tone_wav, entries,
                       base_freq=500.0, **kw)


# --------------------------------------------------------------------------- #
# Real far-field interferers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lone_far_prob", [0.0, 1.0], ids=["target-present", "lone-far"])
def test_a_real_far_interferer_is_mixed_in_and_lone_far_strips_the_target(
    tmp_path, write_puresound_metafile, write_tone_wav, lone_far_prob
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    pool = _far_pool(tmp_path, write_tone_wav)
    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 1.0,
                                   "lone_far_prob": lone_far_prob,
                                   "pool_manifest": str(pool)},
        vad_label_args={"used": False},
    )
    assert len(dataset._realfar_pool) == 4

    sample = dataset[("corpus_spk0", 16000, 7)]

    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)
    assert sample["noisy_speech"].abs().sum() > 0
    if lone_far_prob:
        assert sample["clean_speech"].abs().sum() == 0
    else:
        assert sample["clean_speech"].abs().sum() > 0
        assert not torch.allclose(sample["noisy_speech"], sample["clean_speech"])


@pytest.mark.parametrize("block", ["realfar", "realnear"])
def test_a_disabled_real_pool_consumes_no_randomness(
    tmp_path, write_puresound_metafile, write_tone_wav, block
):
    """A seeded item is identical whether the block is absent or disabled, so
    recipes without it regenerate what they always did."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    pool = (_far_pool if block == "realfar" else _near_pool)(tmp_path, write_tone_wav)

    def _sample(args):
        ds = VoiceIsolationDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=SPEECH_ARGS,
            vad_label_args={"used": False},
            **{f"augmentation_{block}_args": args},
        )
        return ds[("corpus_spk0", 16000, 123)]

    baseline = _sample(None)
    disabled = _sample({"used": False, "prob": 1.0, "pool_manifest": str(pool)})

    assert torch.allclose(baseline["noisy_speech"], disabled["noisy_speech"])
    assert torch.allclose(baseline["clean_speech"], disabled["clean_speech"])


# --------------------------------------------------------------------------- #
# Real near-field keep rows
# --------------------------------------------------------------------------- #


def test_a_real_near_row_keeps_its_foreground_and_carries_the_pool_distances(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 0.0, "lone_far_prob": 1.0,
                                   "pool_manifest": str(_far_pool(tmp_path, write_tone_wav))},
        augmentation_realnear_args={"used": True, "prob": 1.0,
                                    "pool_manifest": str(_near_pool(tmp_path, write_tone_wav))},
        vad_label_args={"used": False},
    )
    assert len(dataset._realnear_pool) == 3

    sample = dataset[("corpus_spk0", 16000, 11)]

    # lone-far must never hit a keep row, even at lone_far_prob 1.0
    assert float(sample["target_absent"]) == 0.0
    assert sample["clean_speech"].abs().sum() > 0
    assert float(sample["foreground_distance"]) == pytest.approx(0.74)
    assert float(sample["nearest_interferer_distance"]) >= 1.0
    assert not torch.allclose(sample["noisy_speech"], sample["clean_speech"])


def test_a_real_near_row_prefers_an_interferer_from_the_same_room(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    # rmA entries at 2.0 m, rmB at 9.0 m: the pick shows in the distance label
    far_pool = _write_pool(
        tmp_path / "far_rooms.jsonl", write_tone_wav,
        [("rmA", 2.0), ("rmA", 2.0), ("rmB", 9.0), ("rmB", 9.0)],
    )
    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 0.0, "lone_far_prob": 0.0,
                                   "pool_manifest": str(far_pool)},
        augmentation_realnear_args={
            "used": True, "prob": 1.0,
            "pool_manifest": str(_near_pool(tmp_path, write_tone_wav, room="rmA")),
        },
        vad_label_args={"used": False},
    )

    for seed in range(5):
        sample = dataset[("corpus_spk0", 16000, seed)]
        assert float(sample["nearest_interferer_distance"]) == pytest.approx(2.0)


def test_real_near_turn_taking_silences_the_target_for_far_solo_stretches(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    """The target is silenced while the mixture still carries the far
    interferer, so the row supervises "lone far voice = suppress"."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    # 6 s rows so the turn script fits conversational turns; far_first_prob 1.0
    # makes the row-initial far-solo turn deterministic.
    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile, training_sample_length_in_seconds=6.0),
        augmentation_speech_args={**SPEECH_ARGS, "overlap_control": {
            "used": True, "no_overlap_prob": 0.0, "high_overlap_prob": 1.0,
            "high_overlap_range": [0.9, 1.0], "fade_samples": 8,
            "far_first_prob": 1.0}},
        augmentation_realfar_args={
            "used": True, "prob": 0.0, "lone_far_prob": 0.0,
            "pool_manifest": str(_far_pool(tmp_path, write_tone_wav, duration=7.0)),
        },
        augmentation_realnear_args={
            "used": True, "prob": 1.0, "turn_taking_prob": 1.0,
            "pool_manifest": str(_near_pool(tmp_path, write_tone_wav, duration=7.0)),
        },
        vad_label_args={"used": True, "backend": "energy",
                        "args": {"frame_length": 80, "hop_length": 40}},
    )

    gated = 0
    for seed in range(6):
        s = dataset[("corpus_spk0", 16000, seed)]
        clean, noisy = s["clean_speech"].view(-1), s["noisy_speech"].view(-1)
        # 10 ms frames where the target is silent but the mixture is not; the
        # row-initial far turn is at least 2 s long
        f = 160
        n_fr = clean.shape[0] // f
        c = clean[: n_fr * f].view(n_fr, f).abs().amax(dim=1)
        m = noisy[: n_fr * f].view(n_fr, f).abs().amax(dim=1)
        if ((c < 1e-4) & (m > 1e-3)).sum().item() >= 100:
            gated += 1
    assert gated >= 5, f"turn-taking produced far-solo stretches in only {gated}/6 rows"


# --------------------------------------------------------------------------- #
# Distance-level SIR
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fg,interferers,expected",
    [
        # the nearest interferer is the loudest: 20 log10(2.0 / 0.5)
        ({"source_receiver_distance": 0.5},
         [{"source_receiver_distance": 3.0}, {"source_receiver_distance": 2.0}],
         20.0 * math.log10(2.0 / 0.5)),
        # unknown geometry falls back to the mode's other draw
        (None, [{"source_receiver_distance": 2.0}], None),
        ({"source_receiver_distance": 0.5}, [], None),
        ({"source_receiver_distance": None}, [{"source_receiver_distance": 2.0}], None),
    ],
    ids=["inverse-distance", "no-foreground", "no-interferer", "unknown-distance"],
)
def test_distance_level_sir_follows_the_inverse_distance_law(fg, interferers, expected):
    vi = object.__new__(VoiceIsolationDataset)
    mode = MixModeEntry(name="distance_level", distance_level=True, jitter_db=[0.0, 0.0])
    sir = vi._distance_level_sir(mode, fg, interferers)
    if expected is None:
        assert sir is None
    else:
        assert sir == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Scalar metadata
# --------------------------------------------------------------------------- #


def _metadata(**kwargs):
    defaults = dict(
        foreground_metadata=None,
        interferer_metadata=[],
        target_absent=False,
        background_speech_reference=None,
    )
    defaults.update(kwargs)
    return VoiceIsolationDataset._build_voice_isolation_metadata(
        object.__new__(VoiceIsolationDataset), **defaults
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            dict(foreground_distance=NAN, foreground_drr=NAN,
                 nearest_interferer_distance=NAN, strongest_interferer_drr=NAN,
                 drr_gap=NAN, rt60=NAN, n_interferers=0.0, target_present=1.0,
                 has_background_speech=0.0),
        ),
        (
            dict(
                foreground_metadata={"source_receiver_distance": 0.5, "drr_db": 6.0,
                                     "rt60": 0.4},
                interferer_metadata=[
                    {"source_receiver_distance": 3.0, "drr_db": -4.0},
                    {"source_receiver_distance": 2.0, "drr_db": -9.0},
                ],
                target_absent=True,
                background_speech_reference=torch.ones(1, 10),
                mix_mode="physical",
                realized_speech_sir=4.5,
                turn_taking=1.0,
            ),
            dict(nearest_interferer_distance=2.0,   # min distance
                 strongest_interferer_drr=-4.0,     # max DRR
                 drr_gap=10.0,                      # foreground - strongest
                 n_interferers=2.0, target_absent=1.0, target_present=0.0,
                 has_background_speech=1.0, mix_mode=MIX_MODE_CODES["physical"],
                 realized_speech_sir=4.5, turn_taking=1.0),
        ),
        (
            # real rows carry a distance but no DRR/rt60 -- exactly the NaN
            # masking DistHeadRegressionLoss depends on
            dict(
                foreground_metadata={"source_receiver_distance": 0.74, "origin": "real"},
                interferer_metadata=[{"source_receiver_distance": 3.02, "origin": "real"}],
            ),
            dict(foreground_distance=0.74, foreground_drr=NAN,
                 nearest_interferer_distance=3.02, strongest_interferer_drr=NAN,
                 drr_gap=NAN),
        ),
    ],
    ids=["nothing-known", "synthetic-row", "real-row"],
)
def test_metadata_is_derived_from_the_right_sources_and_unknown_is_nan(kwargs, expected):
    """Auxiliary supervision and eval bucketing read these, so 'unknown' must be
    NaN, never a fabricated number."""
    md = _metadata(**kwargs)
    for key, value in expected.items():
        got = float(md[key])
        if math.isnan(value):
            assert math.isnan(got), key
        else:
            assert got == pytest.approx(value), key
