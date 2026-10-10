"""What the pipeline inspector reports about a pair, an impulse response and a
room, on hand-made signals."""

import json

import numpy as np
import pytest

from puresound.evaluation import pipeline_report as report

SR = 16000


def _tone(seconds=1.0, amplitude=0.1, freq=220.0):
    time = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def test_sanitize_turns_every_non_finite_number_into_null():
    value = {"a": float("nan"), "b": [1.0, float("inf"), np.float32(2.0)], "c": np.array([np.nan, 3.0]), "d": "x"}
    clean = report.sanitize(value)
    assert clean == {"a": None, "b": [1.0, None, 2.0], "c": [None, 3.0], "d": "x"}
    json.dumps(clean, allow_nan=False)


def test_pair_metrics_measure_the_effective_snr_of_the_pair():
    target = _tone()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(target.size).astype(np.float32)
    noise *= np.sqrt(np.sum(target**2) / np.sum(noise**2) / 10.0)
    metrics = report.pair_metrics(target + noise, target, SR)
    assert metrics["esnr_state"] == "finite"
    assert metrics["esnr_db"] == pytest.approx(10.0, abs=0.01)
    assert metrics["input_si_sdr_db"] == pytest.approx(10.0, abs=0.2)
    assert metrics["samples"] == target.size and metrics["seconds"] == pytest.approx(1.0)
    assert len(metrics["frame_esnr_db"]) == target.size // int(SR * report.FRAME_SECONDS)


def test_an_identical_pair_and_a_silent_target_have_no_ratio():
    target = _tone()
    same = report.pair_metrics(target, target, SR)
    assert (same["esnr_state"], same["esnr_db"]) == ("identical", None)
    absent = report.pair_metrics(target, np.zeros_like(target), SR)
    assert (absent["esnr_state"], absent["esnr_db"], absent["input_si_sdr_db"]) == ("no_target", None, None)
    assert absent["target_rms_dbfs"] is None
    json.dumps(report.sanitize(absent), allow_nan=False)


def test_frame_esnr_skips_frames_where_the_target_is_silent():
    target = np.concatenate([np.zeros(SR // 2, dtype=np.float32), _tone(0.5)])
    noisy = target + 0.001
    frames = report.frame_esnr_db(noisy, target, SR)
    assert frames[0] is None and frames[-1] is not None


def test_gain_staging_divides_the_pair_by_its_shared_peak_only_above_full_scale():
    noisy, target = np.array([0.5, -2.0], np.float32), np.array([0.25, 0.5], np.float32)
    staged_noisy, staged_target, gain = report.gain_staged(noisy, target)
    assert gain == 0.5
    np.testing.assert_allclose(staged_noisy, [0.25, -1.0])
    np.testing.assert_allclose(staged_target, [0.125, 0.25])
    quiet_noisy, _, quiet_gain = report.gain_staged(noisy * 0.1, target)
    assert quiet_gain == 1.0 and quiet_noisy is not None


def test_score_pair_is_scale_invariant_and_undefined_without_a_target():
    target = _tone()
    estimate = 0.5 * target + 0.0005 * np.random.default_rng(1).standard_normal(target.size).astype(np.float32)
    scores = report.score_pair(target, estimate, SR)
    assert scores["si_sdr_db"] > 30
    assert report.score_pair(np.zeros_like(target), estimate, SR)["si_sdr_db"] is None
    assert report.si_sdr_db(target, target) == report.IDENTICAL_DB


def test_rir_summary_reads_the_decay_of_an_exponential_tail():
    rt60 = 0.5
    time = np.arange(int(0.8 * SR)) / SR
    tail = np.random.default_rng(2).standard_normal(time.size) * np.exp(-6.9 * time / rt60)
    impulse = np.concatenate([np.zeros(80), [1.0], 0.3 * tail]).astype(np.float32)
    summary = report.rir_summary(impulse, SR)
    assert summary["peak_ms"] == pytest.approx(5.0)
    assert summary["early_end_ms"] == pytest.approx(55.0)
    assert summary["t20_s"] == pytest.approx(rt60, rel=0.15)
    assert len(summary["envelope_db"]) == pytest.approx(summary["length_ms"], abs=1)
    assert max(summary["envelope_db"]) == 0.0 and summary["edc_db"][0] == 0.0


def test_room_geometry_marks_the_sources_a_row_used(tmp_path):
    wav = tmp_path / "room_1.wav"
    wav.write_bytes(b"")
    (tmp_path / "room_1.json").write_text(json.dumps({
        "scene": {
            "room_dim": [5.0, 4.0, 3.0],
            "rt60": 0.4,
            "mic_pos": [2.0, 2.0, 1.0],
            "channel_map": [
                {"channel": 0, "label": "near_0", "source_pos": [2.5, 2.0, 1.2], "distance_m": 0.54},
                {"channel": 1, "label": "far_0", "source_pos": [4.0, 3.0, 1.5], "distance_m": 2.29},
            ],
            "obstacles": [{"footprint": [[1, 1], [1.5, 1], [1.5, 1.5]], "z_min": 0.0, "z_max": 1.2, "material": "wood"}],
        }
    }))
    scene = {"room_id": "room_1", "wav_path": str(wav)}
    rirs = [{"role": "foreground", "metadata": {"label": "near_0"}}, {"role": "interferer", "metadata": {"label": "far_0"}}]
    room = report.room_geometry(scene, rirs)
    assert room["kind"] == "box" and room["room_dim"] == [5.0, 4.0, 3.0] and room["receiver"] == [2.0, 2.0, 1.0]
    assert [source["roles"] for source in room["sources"]] == [["foreground"], ["interferer"]]
    assert room["obstacles"][0]["z_max"] == 1.2
    assert report.room_geometry(None, rirs) is None
    assert report.room_geometry({"wav_path": str(tmp_path / "missing.wav")}, rirs) is None
