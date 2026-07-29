"""Tests for the real far-field interferer branch in ns.py.

The branch replaces the clean-speech-convolved-with-RIR far channel with a
genuine loudspeaker->air->mic recording drawn from a pool manifest, no RIR
applied. These tests exercise it and guard the critical invariant: when the
augmentation_realfar block is absent/disabled, the synthesis path is
bit-identical to before (the branch never touches the RNG stream).
"""

import json

import torch

from puresound.task.ns import NoiseSuppressionDataset
from puresound.task.voice_isolation import VoiceIsolationDataset


def _dataset_args(metafile_path):
    return {
        "metafile_path": str(metafile_path),
        "min_utt_length_in_seconds": 0.05,
        "min_utts_in_each_speaker": 2,
        "target_sr": 16000,
        "training_sample_length_in_seconds": 0.1,
        "audio_gain_nomalized_to": None,
    }


def _write_realfar_pool(tmp_path, write_tone_wav, n=4, duration=0.3):
    """A tiny real-far pool manifest of tone wavs standing in for VOiCES distant."""
    pool_path = tmp_path / "realfar_pool.jsonl"
    lines = []
    for i in range(n):
        wav_path = tmp_path / "realfar" / f"far{i}.wav"
        write_tone_wav(wav_path, sample_rate=16000, duration=duration, freq=300.0 + 40 * i)
        lines.append(
            json.dumps(
                {"wav_path": str(wav_path), "room": "rm1", "distance_m": 1.5 + i, "speaker": f"sp{i}"}
            )
        )
    pool_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pool_path


_SPEECH_ARGS = {
    "used": True,
    "is_target": False,
    "prob": 1.0,
    "add_n_cases": 1,
    "snr_range": [-5, 5],
}


def test_realfar_interferer_produces_valid_sample(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    pool = _write_realfar_pool(tmp_path, write_tone_wav)
    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=_SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 1.0, "lone_far_prob": 0.0,
                                   "pool_manifest": str(pool)},
        vad_label_args={"used": False},
    )
    assert len(dataset._realfar_pool) == 4

    sample = dataset[("corpus_spk0", 16000, 7)]

    # target present (lone_far_prob=0) -> clean speech kept, interferer mixed in.
    assert sample["noisy_speech"].shape == (1, 1600)
    assert sample["clean_speech"].shape == (1, 1600)
    assert sample["clean_speech"].abs().sum() > 0
    assert not torch.allclose(sample["noisy_speech"], sample["clean_speech"])


def test_realfar_lone_far_zeros_the_target(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    pool = _write_realfar_pool(tmp_path, write_tone_wav)
    dataset = NoiseSuppressionDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=_SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 1.0, "lone_far_prob": 1.0,
                                   "pool_manifest": str(pool)},
        vad_label_args={"used": False},
    )

    sample = dataset[("corpus_spk0", 16000, 7)]

    # lone-far (target-absent): near foreground stripped, only real far remains.
    assert sample["clean_speech"].abs().sum() == 0
    assert sample["noisy_speech"].abs().sum() > 0


def test_realfar_disabled_is_bit_identical(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    """The guard must not consume RNG when the block is absent/disabled: a seeded
    item is identical whether augmentation_realfar is None or {'used': False}."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    pool = _write_realfar_pool(tmp_path, write_tone_wav)

    def _sample(realfar_args):
        ds = NoiseSuppressionDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=_SPEECH_ARGS,
            augmentation_realfar_args=realfar_args,
            vad_label_args={"used": False},
        )
        return ds[("corpus_spk0", 16000, 123)]

    baseline = _sample(None)
    disabled = _sample({"used": False, "prob": 1.0, "pool_manifest": str(pool)})

    assert torch.allclose(baseline["noisy_speech"], disabled["noisy_speech"])
    assert torch.allclose(baseline["clean_speech"], disabled["clean_speech"])


def _write_realnear_pool(tmp_path, write_tone_wav, room="rm1", n=3, duration=0.3):
    """A tiny real-near keep pool (VOiCES <1m mics stand-in)."""
    pool_path = tmp_path / "realnear_pool.jsonl"
    lines = []
    for i in range(n):
        wav_path = tmp_path / "realnear" / f"near{i}.wav"
        write_tone_wav(wav_path, sample_rate=16000, duration=duration, freq=500.0 + 30 * i)
        lines.append(
            json.dumps(
                {"wav_path": str(wav_path), "room": room, "distance_m": 0.74,
                 "speaker": f"nearsp{i}"}
            )
        )
    pool_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pool_path


def test_realnear_keep_row_pool_foreground_and_metadata(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    far_pool = _write_realfar_pool(tmp_path, write_tone_wav)
    near_pool = _write_realnear_pool(tmp_path, write_tone_wav)
    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile),
        augmentation_speech_args=_SPEECH_ARGS,
        augmentation_realfar_args={"used": True, "prob": 0.0, "lone_far_prob": 1.0,
                                   "pool_manifest": str(far_pool)},
        augmentation_realnear_args={"used": True, "prob": 1.0,
                                    "pool_manifest": str(near_pool)},
        vad_label_args={"used": False},
    )
    assert len(dataset._realnear_pool) == 3

    sample = dataset[("corpus_spk0", 16000, 11)]

    # Keep row: target present (lone-far must NEVER hit realnear rows even with
    # lone_far_prob 1.0), foreground carries the pool's real distance, and the
    # interferer is a real-far pool draw.
    assert float(sample["target_absent"]) == 0.0
    assert sample["clean_speech"].abs().sum() > 0
    assert abs(float(sample["foreground_distance"]) - 0.74) < 1e-6
    assert float(sample["nearest_interferer_distance"]) >= 1.0
    assert not torch.allclose(sample["noisy_speech"], sample["clean_speech"])


def test_realnear_prefers_same_room_interferer(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    near_pool = _write_realnear_pool(tmp_path, write_tone_wav, room="rmA")
    # Far pool: rmA entries at 2.0 m, rmB entries at 9.0 m -> a same-room pick is
    # observable through nearest_interferer_distance.
    far_path = tmp_path / "far_rooms.jsonl"
    lines = []
    for i, (room, dist) in enumerate([("rmA", 2.0), ("rmA", 2.0), ("rmB", 9.0), ("rmB", 9.0)]):
        wav_path = tmp_path / "far_rooms" / f"far{i}.wav"
        write_tone_wav(wav_path, sample_rate=16000, duration=0.3, freq=300.0 + 40 * i)
        lines.append(json.dumps({"wav_path": str(wav_path), "room": room,
                                 "distance_m": dist, "speaker": f"farsp{i}"}))
    far_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = VoiceIsolationDataset(
        **_dataset_args(metafile),
        augmentation_speech_args={**_SPEECH_ARGS, "add_n_cases": 1},
        augmentation_realfar_args={"used": True, "prob": 0.0, "lone_far_prob": 0.0,
                                   "pool_manifest": str(far_path)},
        augmentation_realnear_args={"used": True, "prob": 1.0,
                                    "pool_manifest": str(near_pool)},
        vad_label_args={"used": False},
    )

    for seed in range(5):
        sample = dataset[("corpus_spk0", 16000, seed)]
        assert abs(float(sample["nearest_interferer_distance"]) - 2.0) < 1e-6


def test_realnear_turn_taking_gates_target(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    """turn_taking_prob on realnear rows creates far-solo stretches: the target
    (near) is silenced there while the mix still carries the far interferer, so
    the row supervises the absolute "lone far voice = suppress" decision."""
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    # 6 s clips so the turn script fits conversational turns (1.5-4.5 s);
    # far_first=1.0 makes the row-initial far-solo turn deterministic.
    far_pool = _write_realfar_pool(tmp_path, write_tone_wav, duration=7.0)
    near_pool = _write_realnear_pool(tmp_path, write_tone_wav, duration=7.0)
    ds_args = {**_dataset_args(metafile), "training_sample_length_in_seconds": 6.0}
    dataset = VoiceIsolationDataset(
        **ds_args,
        augmentation_speech_args={**_SPEECH_ARGS, "overlap_control": {
            "used": True, "no_overlap_prob": 0.0, "high_overlap_prob": 1.0,
            "high_overlap_range": [0.9, 1.0], "fade_samples": 8,
            "far_first_prob": 1.0}},
        augmentation_realfar_args={"used": True, "prob": 0.0, "lone_far_prob": 0.0,
                                   "pool_manifest": str(far_pool)},
        augmentation_realnear_args={"used": True, "prob": 1.0,
                                    "turn_taking_prob": 1.0,
                                    "pool_manifest": str(near_pool)},
        vad_label_args={"used": True, "backend": "energy",
                        "args": {"frame_length": 80, "hop_length": 40}},
    )

    gated = 0
    for seed in range(6):
        s = dataset[("corpus_spk0", 16000, seed)]
        clean, noisy = s["clean_speech"].view(-1), s["noisy_speech"].view(-1)
        # frames where target ~silent but mix active = far-solo turn; the
        # row-initial far turn is >=2 s => >=200 of the 10 ms frames.
        f = 160
        n_fr = clean.shape[0] // f
        c = clean[: n_fr * f].view(n_fr, f).abs().amax(dim=1)
        m = noisy[: n_fr * f].view(n_fr, f).abs().amax(dim=1)
        far_solo = ((c < 1e-4) & (m > 1e-3)).sum().item()
        if far_solo >= 100:
            gated += 1
    assert gated >= 5, f"turn-taking produced far-solo stretches in only {gated}/6 rows"


def test_realnear_disabled_is_bit_identical(
    tmp_path, write_puresound_metafile, write_tone_wav
):
    metafile = write_puresound_metafile(tmp_path / "meta.csv", speakers=3)
    near_pool = _write_realnear_pool(tmp_path, write_tone_wav)

    def _sample(realnear_args):
        ds = NoiseSuppressionDataset(
            **_dataset_args(metafile),
            augmentation_speech_args=_SPEECH_ARGS,
            augmentation_realnear_args=realnear_args,
            vad_label_args={"used": False},
        )
        return ds[("corpus_spk0", 16000, 123)]

    baseline = _sample(None)
    disabled = _sample({"used": False, "prob": 1.0, "pool_manifest": str(near_pool)})

    assert torch.allclose(baseline["noisy_speech"], disabled["noisy_speech"])
    assert torch.allclose(baseline["clean_speech"], disabled["clean_speech"])
