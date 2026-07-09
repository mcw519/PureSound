import importlib.util

import numpy as np
import pytest
import torch

from puresound.audio.hybrid_rir import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    HybridRIRConfig,
    HybridRIRScene,
    PolygonObstacle,
    apply_obstacle_high_frequency_effects,
    generate_hybrid_rir,
    hybrid_crossover,
    obstacle_effects_metadata,
    sample_hybrid_rir_scene,
    write_hybrid_rir_dataset_item,
    _obstacle_floor_coverage,
    _polygons_overlap,
)


class DummyHighFrequencyBackend:
    def simulate(self, scene, config):
        rir = np.zeros((config.num_sources, config.num_samples), dtype=np.float32)
        for idx, src in enumerate(scene.source_pos):
            distance = np.linalg.norm(np.asarray(src) - np.asarray(scene.mic_pos))
            direct = int(round(distance / config.sound_speed * config.sample_rate))
            if direct < config.num_samples:
                rir[idx, direct] = 1.0 / max(distance, 0.1)
        return apply_obstacle_high_frequency_effects(rir, scene, config)


def test_scene_sampler_enforces_two_near_and_three_far_sources():
    config = HybridRIRConfig(
        room_dim_range=((6.0, 6.0), (6.0, 6.0), (2.8, 2.8)),
        num_obstacles_range=(1, 1),
    )
    scene = sample_hybrid_rir_scene(config, seed=7)

    assert scene.source_labels == ["near_0", "near_1", "far_0", "far_1", "far_2"]
    distances = scene.source_distances()
    horizontal_distances = scene.source_horizontal_distances()
    assert len(distances) == 5
    assert all(distance < 1.0 for distance in horizontal_distances[:2])
    assert all(distance > 2.0 for distance in horizontal_distances[2:])
    assert config.mic_height_range[0] <= scene.mic_pos[2] <= config.mic_height_range[1]
    assert all(
        config.speech_source_height_range[0] <= source[2] <= config.speech_source_height_range[1]
        for source in scene.source_pos
    )
    assert len(scene.obstacles) == 1


def test_area_aware_obstacles_respect_coverage_and_clearance():
    config = HybridRIRConfig(
        room_dim_range=((8.0, 8.0), (7.0, 7.0), (3.0, 3.0)),
        num_obstacles_range=(4, 8),
        max_obstacle_floor_coverage=0.18,
        obstacle_obstacle_clearance=0.08,
    )
    scene = sample_hybrid_rir_scene(config, seed=17)

    assert 1 <= len(scene.obstacles) <= 8
    assert _obstacle_floor_coverage(
        scene.obstacles, np.asarray(scene.room_dim, dtype=np.float64)
    ) <= config.max_obstacle_floor_coverage + 1e-6
    for idx, obstacle in enumerate(scene.obstacles):
        footprint = np.asarray(obstacle.footprint, dtype=np.float64)
        for other in scene.obstacles[idx + 1 :]:
            assert not _polygons_overlap(
                footprint, np.asarray(other.footprint, dtype=np.float64)
            )


def test_obstacle_count_tracks_room_area():
    small = HybridRIRConfig(
        room_dim_range=((3.5, 3.5), (3.5, 3.5), (2.8, 2.8)),
        num_obstacles_range=(1, 8),
    )
    large = HybridRIRConfig(
        room_dim_range=((8.0, 8.0), (7.0, 7.0), (2.8, 2.8)),
        num_obstacles_range=(1, 8),
    )

    small_counts = [len(sample_hybrid_rir_scene(small, seed=seed).obstacles) for seed in range(8)]
    large_counts = [len(sample_hybrid_rir_scene(large, seed=seed).obstacles) for seed in range(8)]

    assert np.mean(large_counts) > np.mean(small_counts)


def test_hybrid_generation_returns_five_channel_rir():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        crossover_hz=1000.0,
        room_dim_range=((6.0, 6.0), (6.0, 6.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    rir, metadata = generate_hybrid_rir(
        config=config,
        low_backend=AnalyticModalLowFrequencyBackend(num_modes_per_axis=2),
        high_backend=DummyHighFrequencyBackend(),
        seed=11,
    )

    assert tuple(rir.shape) == (5, 800)
    assert metadata["scene"]["channel_map"][0]["label"] == "near_0"
    assert "horizontal_distance_m" in metadata["scene"]["channel_map"][0]
    assert metadata["obstacle_effects"]["obstacle_model"] == "high_frequency_post_occlusion_scatter"
    assert metadata["bands"]["low"]["frequency_hz"] == [20.0, 1000.0]
    assert torch.max(torch.abs(rir)) <= 0.981


def _occlusion_scene(z_max: float) -> HybridRIRScene:
    return HybridRIRScene(
        room_dim=[4.0, 4.0, 2.8],
        rt60=0.4,
        mic_pos=[3.0, 2.0, 1.4],
        source_pos=[
            [1.0, 2.0, 1.4],
            [3.0, 3.0, 1.4],
            [3.0, 3.2, 1.4],
            [3.0, 3.4, 1.4],
            [3.0, 3.6, 1.4],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
        obstacles=[
            PolygonObstacle(
                footprint=[[1.8, 1.7], [2.2, 1.7], [2.2, 2.3], [1.8, 2.3]],
                z_min=0.0,
                z_max=z_max,
                material="wood",
                absorption=0.25,
                scattering=0.5,
            )
        ],
    )


def test_obstacle_effect_attenuates_occluded_source():
    config = HybridRIRConfig(sample_rate=8000, duration=0.05)
    # A tall obstacle spans the head-height direct path and must occlude it.
    scene = _occlusion_scene(z_max=1.6)
    rir = np.ones((5, config.num_samples), dtype=np.float32)
    out = apply_obstacle_high_frequency_effects(rir, scene, config)

    assert out[0, 0] < rir[0, 0]
    assert out[1, 0] == rir[1, 0]
    metadata = obstacle_effects_metadata(scene, config)
    assert metadata["events"][0]["source_index"] == 0
    assert metadata["events"][0]["attenuation"] < 1.0


def test_obstacle_below_path_height_does_not_occlude():
    config = HybridRIRConfig(sample_rate=8000, duration=0.05)
    # A low table (top at 1.2 m) is below the 1.4 m direct path; the 3D-aware
    # occlusion test must leave the source untouched.
    scene = _occlusion_scene(z_max=1.2)
    rir = np.ones((5, config.num_samples), dtype=np.float32)
    out = apply_obstacle_high_frequency_effects(rir, scene, config)

    assert out[0, 0] == rir[0, 0]


def test_write_dataset_item_creates_five_channel_wav_and_metadata(tmp_path):
    config = HybridRIRConfig(sample_rate=8000, duration=0.02)
    rir = torch.zeros(5, config.num_samples)
    metadata = {"scene": {"channel_map": []}}

    wav_path, json_path = write_hybrid_rir_dataset_item(
        tmp_path, "room_000000", rir, metadata, config.sample_rate
    )

    assert wav_path.exists()
    assert json_path.exists()


def test_crossover_preserves_shape():
    config = HybridRIRConfig(sample_rate=8000, duration=0.05)
    low = np.random.default_rng(0).normal(size=(5, config.num_samples))
    high = np.random.default_rng(1).normal(size=(5, config.num_samples))

    out = hybrid_crossover(low, high, config)

    assert out.shape == (5, config.num_samples)
    assert out.dtype == np.float32


def test_vendored_pytard_backend_smoke():
    config = HybridRIRConfig(
        sample_rate=16000,
        duration=0.005,
        room_dim_range=((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        num_obstacles_range=(0, 0),
    )
    scene = HybridRIRScene(
        room_dim=[1.0, 1.0, 1.0],
        rt60=0.2,
        mic_pos=[0.7, 0.5, 0.5],
        source_pos=[
            [0.3, 0.5, 0.5],
            [0.4, 0.5, 0.5],
            [0.5, 0.3, 0.5],
            [0.5, 0.4, 0.5],
            [0.5, 0.5, 0.3],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    backend = GpuARDPytARDBackend(low_sample_rate=16000)

    rir = backend.simulate(scene, config)

    assert rir.shape == (5, config.num_samples)
    assert rir.dtype == np.float64
    assert np.max(np.abs(rir)) > 0.01


def test_cupy_pytard_backend_requires_cupy_or_runs_smoke():
    config = HybridRIRConfig(sample_rate=16000, duration=0.005)
    scene = HybridRIRScene(
        room_dim=[1.0, 1.0, 1.0],
        rt60=0.2,
        mic_pos=[0.7, 0.5, 0.5],
        source_pos=[
            [0.3, 0.5, 0.5],
            [0.4, 0.5, 0.5],
            [0.5, 0.3, 0.5],
            [0.5, 0.4, 0.5],
            [0.5, 0.5, 0.3],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    backend = GpuARDPytARDCuPyBackend(low_sample_rate=16000)

    if importlib.util.find_spec("cupy") is None:
        with pytest.raises(ImportError, match="requires CuPy"):
            backend.simulate(scene, config)
        return

    rir = backend.simulate(scene, config)
    assert rir.shape == (5, config.num_samples)


def test_crossover_matches_tiny_low_band_to_high_band_energy():
    config = HybridRIRConfig(sample_rate=8000, duration=0.1)
    rng = np.random.default_rng(123)
    low = rng.normal(scale=1e-10, size=(5, config.num_samples))
    high = rng.normal(scale=1.0, size=(5, config.num_samples))

    out = hybrid_crossover(low, high, config)

    assert out.shape == (5, config.num_samples)
    assert np.max(np.abs(out)) > 0.1


def test_pytard_calibrated_signal_has_rt60_tail_decay():
    from puresound.audio.hybrid_rir import _calibrate_pytard_signal

    signal = np.ones(8000, dtype=np.float64)
    out = _calibrate_pytard_signal(
        signal,
        distance_m=1.0,
        target_peak=0.05,
        sample_rate=8000,
        rt60=0.2,
        apply_decay=True,
    )

    early = np.sqrt(np.mean(out[:400] ** 2))
    late = np.sqrt(np.mean(out[-400:] ** 2))
    assert late < early * 1e-3


def test_decay_envelope_starts_at_direct_path_not_signal_peak():
    from puresound.audio.hybrid_rir import _apply_rt60_decay_envelope

    # Energy concentrated late (as in a lossless modal field). The envelope must
    # be keyed to the supplied direct-path origin, not the signal peak: samples
    # before the origin stay untouched and samples after it decay.
    signal = np.zeros(8000, dtype=np.float64)
    signal[6000] = 1.0
    origin = 800
    out = _apply_rt60_decay_envelope(signal, sample_rate=8000, rt60=0.2, origin_idx=origin)

    assert out[origin - 1] == signal[origin - 1]
    # 10 ** (-3 * elapsed / rt60); elapsed at sample 6000 = (6000-800)/8000 s.
    expected = 10.0 ** (-3.0 * ((6000 - origin) / 8000.0) / 0.2)
    assert out[6000] == pytest.approx(expected, rel=1e-6)


def test_high_band_reverberation_follows_requested_rt60():
    if importlib.util.find_spec("pyroomacoustics") is None:
        pytest.skip("pyroomacoustics not installed")
    from puresound.audio.hybrid_rir import PyroomacousticsHighFrequencyBackend

    config = HybridRIRConfig(sample_rate=16000, duration=0.8, num_obstacles_range=(0, 0))

    def tail_decay_db(rt60: float) -> float:
        scene = HybridRIRScene(
            room_dim=[6.0, 5.0, 3.0],
            rt60=rt60,
            mic_pos=[3.0, 2.5, 1.5],
            source_pos=[
                [2.0, 2.5, 1.5],
                [2.2, 2.5, 1.5],
                [5.0, 4.0, 1.5],
                [1.0, 1.0, 1.5],
                [5.0, 1.0, 1.5],
            ],
            source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
        )
        rir = PyroomacousticsHighFrequencyBackend().simulate(scene, config)
        channel = rir[2]
        peak = int(np.argmax(np.abs(channel)))
        early = np.sqrt(np.mean(channel[peak : peak + 800] ** 2))
        late = np.sqrt(np.mean(channel[peak + 6000 : peak + 8000] ** 2))
        return 20.0 * np.log10((late + 1e-12) / (early + 1e-12))

    short = tail_decay_db(0.25)
    long = tail_decay_db(0.8)
    # A longer RT60 must leave more energy in the tail (less negative dB).
    assert long > short + 6.0
