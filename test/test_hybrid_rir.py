import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.audio.hybrid_rir import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    HybridRIRConfig,
    HybridRIRScene,
    ImpedanceModalLowFrequencyBackend,
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    PolygonObstacle,
    apply_obstacle_high_frequency_effects,
    generate_hybrid_rir,
    hybrid_crossover,
    obstacle_effects_metadata,
    sample_hybrid_rir_scene,
    upgrade_hybrid_scene_to_v2,
    write_hybrid_rir_dataset_item,
    _obstacle_floor_coverage,
    _polygons_overlap,
    _hybrid_crossover_with_metadata,
    _clip_rir_before_physical_arrival,
    _pytard_green_delta_excitation,
    _solve_modal_ard,
)
from puresound.audio.acoustic_impedance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.impedance_modes import (
    RECTANGULAR_BOUNDARIES,
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.impedance_residues import (
    ImpedanceModalResidueCalibration,
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


class ZeroHighFrequencyBackend:
    def simulate(self, scene, config):
        return np.zeros(
            (config.num_sources, config.num_samples),
            dtype=np.float32,
        )


def test_modal_solver_can_capture_a_pressure_slice_for_visualization():
    sim_param = SimpleNamespace(
        c=343,
        delta_t=1.0 / 8000.0,
        number_of_samples=32,
        spatial_samples_per_wave_length=2,
        max_simulation_frequency=1000,
    )
    capture = {
        "source_index": 0,
        "z_m": 1.0,
        "stride": 2,
        "max_frames": 8,
    }
    signals = _solve_modal_ard(
        sim_param=sim_param,
        room_dim=np.asarray([2.0, 2.0, 2.0]),
        mic_pos=np.asarray([1.0, 1.0, 1.0]),
        source_positions=np.asarray([[0.5, 0.5, 0.5]]),
        impulse=np.r_[1.0, np.zeros(31)],
        xp=np,
        cp=None,
        field_capture=capture,
    )

    assert len(signals) == 1
    assert signals[0].shape == (32,)
    assert capture["frames"].shape == (8, 11, 11)
    assert capture["times_s"].shape == (8,)
    assert np.all(np.diff(capture["times_s"]) > 0.0)
    assert capture["grid_shape_zyx"] == [11, 11, 11]


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
    assert metadata["obstacle_effects"]["obstacle_model"] == (
        "direct_early_occlusion_with_diffuse_recovery"
    )
    assert metadata["bands"]["low"]["frequency_hz"] == [20.0, 1000.0]
    assert metadata["bands"]["low"]["mode_excitation_model"] == (
        "rectangular_eigenfunction_source_receiver_coupling"
    )
    assert metadata["bands"]["low"]["analytic_max_modes"] == 256
    assert torch.max(torch.abs(rir)) <= 0.981


def test_analytic_modal_backend_uses_physical_nodes_and_is_reciprocal():
    config = HybridRIRConfig(
        sample_rate=4000,
        duration=0.3,
        low_fmin_hz=55.0,
        low_fmax_hz=60.0,
        crossover_hz=200.0,
        room_dim_range=((3.0, 3.0), (2.0, 2.0), (2.0, 2.0)),
        num_obstacles_range=(0, 0),
    )
    mic = [0.4, 0.8, 0.9]
    node = [1.5, 0.8, 0.9]
    off_node = [0.75, 0.8, 0.9]
    backend = AnalyticModalLowFrequencyBackend(
        num_modes_per_axis=2,
        physical_mode_coupling=True,
    )

    def scene(source):
        return HybridRIRScene(
            room_dim=[3.0, 2.0, 2.0],
            rt60=0.5,
            mic_pos=mic,
            source_pos=[source] * config.num_sources,
            source_labels=[
                f"source_{index}" for index in range(config.num_sources)
            ],
        )

    node_rir = backend.simulate(scene(node), config)[0]
    off_node_rir = backend.simulate(scene(off_node), config)[0]
    node_direct = int(
        round(
            np.linalg.norm(np.asarray(node) - np.asarray(mic))
            / config.sound_speed
            * config.sample_rate
        )
    )
    off_node_direct = int(
        round(
            np.linalg.norm(np.asarray(off_node) - np.asarray(mic))
            / config.sound_speed
            * config.sample_rate
        )
    )

    assert np.all(node_rir[:node_direct] == 0.0)
    assert np.linalg.norm(node_rir[node_direct + 1 :]) < 1e-10
    assert np.linalg.norm(off_node_rir[off_node_direct + 1 :]) > 1e-3

    reciprocal = HybridRIRScene(
        room_dim=[3.0, 2.0, 2.0],
        rt60=0.5,
        mic_pos=off_node,
        source_pos=[mic] * config.num_sources,
        source_labels=[
            f"source_{index}" for index in range(config.num_sources)
        ],
    )
    reciprocal_rir = backend.simulate(reciprocal, config)[0]
    assert np.allclose(off_node_rir, reciprocal_rir)


def test_3d_impedance_modal_backend_integrates_with_hybrid_metadata():
    config = HybridRIRConfig(
        sample_rate=4000,
        duration=0.2,
        crossover_hz=200.0,
        low_fmin_hz=40.0,
        low_fmax_hz=180.0,
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(0, 0),
    )
    legacy = HybridRIRScene(
        room_dim=[3.0, 3.0, 3.0],
        rt60=0.5,
        mic_pos=[1.8, 1.4, 1.2],
        source_pos=[
            [1.0, 1.4, 1.2],
            [1.2, 1.8, 1.2],
            [2.6, 2.4, 1.4],
            [0.5, 0.6, 1.5],
            [2.5, 0.5, 1.6],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    scene = upgrade_hybrid_scene_to_v2(
        legacy,
        seed=17,
        room_type="office",
    )
    boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.01,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    boundary_config = RectangularImpedanceBoundaryConfig(
        reference_id="uniform_static_test",
        boundaries={name: boundary for name in RECTANGULAR_BOUNDARIES},
        source={"evidence_tier": "synthetic_validation"},
        applicability={
            "scope": "solver_validation_only",
            "automatic_scene_catalog_mapping": False,
            "valid_frequency_range_hz": [40.0, 180.0],
        },
    )
    backend = ImpedanceModalLowFrequencyBackend(
        boundary_config=boundary_config,
        num_modes_per_axis=1,
        max_modes=7,
    )

    rir, metadata = generate_hybrid_rir(
        config,
        scene=scene,
        low_backend=backend,
        high_backend=ZeroHighFrequencyBackend(),
    )

    assert tuple(rir.shape) == (5, config.num_samples)
    assert torch.all(torch.isfinite(rir))
    assert torch.count_nonzero(rir) > 0
    low = metadata["bands"]["low"]
    assert low["boundary_model"] == (
        "separable_rational_impedance_eigenproblem"
    )
    assert low["global_rt60_envelope_applied"] is False
    assert low["mode_excitation_model"] == (
        "separable_complex_eigenfunction_source_receiver_coupling"
    )
    assert low["impedance_modes"]["mode_count"] > 0
    assert low["impedance_modes"][
        "production_material_mapping_enabled"
    ] is False
    assert low["impedance_modes"][
        "modal_residue_production_validated"
    ] is False
    assert low["impedance_modes"]["modal_residue_fdtd_validated"] is False

    residue_calibration = ImpedanceModalResidueCalibration(
        reference_id="uniform_static_fdtd_residue",
        boundary_reference_id=boundary_config.reference_id,
        complex_scale=0.45 + 0.1j,
        frequency_exponent=0.3,
        reference_frequency_hz=100.0,
        valid_frequency_range_hz=(40.0, 180.0),
        source={"evidence_tier": "independent_numerical_reference"},
        applicability={
            "scope": "controlled_rectangular_room_validation",
            "production_material_mapping_enabled": False,
        },
        fit_summary={"accepted": True},
    )
    calibrated_backend = ImpedanceModalLowFrequencyBackend(
        boundary_config=boundary_config,
        residue_calibration=residue_calibration,
        num_modes_per_axis=1,
        max_modes=7,
    )

    calibrated = calibrated_backend.simulate(scene, config)

    assert np.all(np.isfinite(calibrated))
    assert np.count_nonzero(calibrated) > 0
    assert calibrated_backend.last_modal_metadata[
        "modal_residue_fdtd_validated"
    ] is True
    assert calibrated_backend.last_modal_metadata["modal_residue_model"] == (
        "fdtd_calibrated_complex_scale_power_law"
    )
    assert calibrated_backend.last_modal_metadata[
        "modal_residue_production_validated"
    ] is False
    assert calibrated_backend.last_modal_metadata[
        "direct_path_amplitude_calibrated_by_residue_fit"
    ] is False
    assert calibrated_backend.last_modal_metadata[
        "direct_path_source_convention_matched"
    ] is True
    assert calibrated_backend.last_modal_metadata[
        "modal_residue_source_transform"
    ] == (
        "puresound.pressure_state_residue_to_free_field_1_over_r.v1"
    )

    preserving_config = replace(
        config,
        match_crossover_energy=True,
        preserve_source_convention_at_crossover=True,
    )
    _calibrated_hybrid, calibrated_metadata = generate_hybrid_rir(
        preserving_config,
        scene=scene,
        low_backend=calibrated_backend,
        high_backend=ZeroHighFrequencyBackend(),
    )
    crossover = calibrated_metadata["crossover"]
    assert crossover["energy_matching_requested"] is True
    assert crossover["energy_matching_applied"] is False
    assert crossover["source_convention_preserved"] is True
    assert crossover["policy"] == "preserve_validated_source_convention"
    assert crossover["low_band_gain_by_channel"] == [1.0] * 5


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
    config = HybridRIRConfig(sample_rate=8000, duration=0.2)
    # A tall obstacle spans the head-height direct path and must occlude it.
    scene = _occlusion_scene(z_max=1.6)
    rir = np.ones((5, config.num_samples), dtype=np.float32)
    out = apply_obstacle_high_frequency_effects(rir, scene, config)

    direct = int(
        np.floor(
            np.linalg.norm(
                np.asarray(scene.source_pos[0]) - np.asarray(scene.mic_pos)
            )
            / config.sound_speed
            * config.sample_rate
        )
    )
    recovery = direct + int(
        round(config.obstacle_occlusion_recovery_ms * config.sample_rate / 1000.0)
    )
    assert out[0, direct] < rir[0, direct]
    assert out[0, min(recovery + 1, config.num_samples - 1)] == pytest.approx(1.0)
    assert out[0, 0] == rir[0, 0]
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


def test_crossover_tail_fade_reaches_exact_zero():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        output_mode="calibrated",
        match_crossover_energy=False,
        tail_fade_ms=20.0,
    )
    low = np.ones((5, config.num_samples), dtype=np.float64)
    high = np.zeros_like(low)

    output, metadata = _hybrid_crossover_with_metadata(low, high, config)

    assert np.all(output[:, -1] == 0.0)
    assert metadata["tail_fade"]["sample_count"] == 160


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
    assert backend.last_excitation_metadata["legacy_bipolar_fir_used"] is False
    assert backend.last_excitation_metadata["policy"].endswith("green_delta.v1")


def test_pytard_green_excitation_has_no_legacy_390_hz_comb():
    excitation = _pytard_green_delta_excitation(16000)
    magnitude = np.abs(np.fft.rfft(excitation))
    frequencies = np.fft.rfftfreq(excitation.size, d=1.0 / 16000.0)
    analysis = (frequencies >= 20.0) & (frequencies <= 1000.0)
    relative_db = 20.0 * np.log10(
        np.maximum(magnitude[analysis], 1e-20) / np.max(magnitude[analysis])
    )

    assert np.max(np.abs(relative_db)) < 1e-12
    legacy_notch = int(np.argmin(np.abs(frequencies - 16000.0 / 41.0)))
    assert magnitude[legacy_notch] == pytest.approx(1.0)


def test_pytard_render_ensemble_has_no_shared_390_hz_notch():
    config = HybridRIRConfig(
        sample_rate=16000,
        duration=0.1,
        crossover_hz=1000.0,
        room_dim_range=((3.5, 4.5), (3.5, 4.5), (2.7, 3.0)),
        num_obstacles_range=(0, 0),
    )
    dips_db = []
    for seed in range(3):
        scene = sample_hybrid_rir_scene(config, seed=seed)
        rendered = GpuARDPytARDBackend(low_sample_rate=16000).simulate(
            scene,
            config,
        )
        for channel in rendered:
            spectrum = np.abs(np.fft.rfft(channel, n=8192)) + 1e-20
            frequencies = np.fft.rfftfreq(8192, d=1.0 / config.sample_rate)
            notch = int(np.argmin(np.abs(frequencies - 16000.0 / 41.0)))
            neighbours = (
                (np.abs(frequencies - frequencies[notch]) >= 30.0)
                & (np.abs(frequencies - frequencies[notch]) <= 80.0)
            )
            dips_db.append(
                20.0
                * np.log10(
                    np.median(spectrum[neighbours]) / spectrum[notch]
                )
            )

    assert abs(float(np.median(dips_db))) < 6.0
    assert float(np.percentile(dips_db, 90.0)) < 10.0


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

    try:
        rir = backend.simulate(scene, config)
    except RuntimeError as exc:
        if "CUDA device initialization failed" in str(exc):
            pytest.skip("CuPy is installed but no usable CUDA device is available")
        raise
    assert rir.shape == (5, config.num_samples)


def test_crossover_matches_tiny_low_band_to_high_band_energy():
    config = HybridRIRConfig(sample_rate=8000, duration=0.1)
    rng = np.random.default_rng(123)
    low = rng.normal(scale=1e-10, size=(5, config.num_samples))
    high = rng.normal(scale=1.0, size=(5, config.num_samples))

    out = hybrid_crossover(low, high, config)

    assert out.shape == (5, config.num_samples)
    assert np.max(np.abs(out)) > 0.1


def test_crossover_match_band_tracks_nondefault_crossover():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        crossover_hz=240.0,
        output_mode="calibrated",
    )
    direct = np.zeros((5, config.num_samples), dtype=np.float64)
    direct[:, 80] = 1.0

    _output, metadata = _hybrid_crossover_with_metadata(
        direct,
        direct,
        config,
    )

    assert metadata["effective_match_band_hz"] == [168.0, 312.0]
    assert metadata["low_band_gain_by_channel"] != [2.0] * 5


def test_crossover_has_flat_magnitude_for_identical_complex_inputs():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.2,
        crossover_hz=240.0,
        output_mode="calibrated",
        match_crossover_energy=False,
    )
    direct = np.zeros((5, config.num_samples), dtype=np.float64)
    direct[:, 100] = 1.0

    output = hybrid_crossover(direct, direct, config)
    response = np.fft.rfft(output[0])
    frequencies_hz = np.fft.rfftfreq(
        output.shape[-1],
        d=1.0 / config.sample_rate,
    )
    analysis = (
        (frequencies_hz >= 20.0)
        & (frequencies_hz <= 0.45 * config.sample_rate)
    )
    magnitude_error_db = 20.0 * np.log10(
        np.maximum(np.abs(response[analysis]), 1e-12)
    )

    assert float(np.max(np.abs(magnitude_error_db))) < 2e-5


def test_crossover_rejects_audit_band_that_excludes_crossover():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.1,
        crossover_hz=240.0,
        crossover_match_band_hz=(700.0, 1300.0),
    )
    direct = np.zeros((5, config.num_samples), dtype=np.float64)

    with pytest.raises(ValueError, match="must contain crossover"):
        hybrid_crossover(direct, direct, config)


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


def test_pyroom_ray_tracing_is_repeatable_when_task_seed_is_reset():
    if importlib.util.find_spec("pyroomacoustics") is None:
        pytest.skip("pyroomacoustics not installed")
    from puresound.audio.hybrid_rir import PyroomacousticsHighFrequencyBackend

    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.2,
        num_near_sources=1,
        num_far_sources=0,
        num_obstacles_range=(0, 0),
    )
    scene = HybridRIRScene(
        room_dim=[4.0, 3.5, 2.8],
        rt60=0.35,
        mic_pos=[2.0, 1.7, 1.2],
        source_pos=[[1.0, 1.7, 1.2]],
        source_labels=["near_0"],
    )
    backend = PyroomacousticsHighFrequencyBackend(
        max_order=1,
        ray_tracing=True,
        n_rays=2000,
    )

    backend.set_rng_seed(7331)
    first = backend.simulate(scene, config)
    backend.set_rng_seed(7331)
    second = backend.simulate(scene, config)

    assert np.array_equal(first, second)


def test_pyroom_v2_backend_renders_scene_source_directivity():
    if importlib.util.find_spec("pyroomacoustics") is None:
        pytest.skip("pyroomacoustics not installed")
    from puresound.audio.hybrid_rir import PyroomacousticsHighFrequencyBackend

    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.08,
        num_near_sources=1,
        num_far_sources=0,
        num_obstacles_range=(0, 0),
    )
    legacy = HybridRIRScene(
        room_dim=[4.0, 4.0, 2.8],
        rt60=0.35,
        mic_pos=[3.0, 2.0, 1.4],
        source_pos=[[1.0, 2.0, 1.4]],
        source_labels=["near_0"],
    )
    scene = upgrade_hybrid_scene_to_v2(legacy, seed=91, room_type="office")
    receiver = replace(
        scene.receivers[0],
        pose=replace(scene.receivers[0].pose, position_m=[3.0, 2.0, 1.4]),
    )
    front_source = replace(
        scene.sources[0],
        pose=replace(
            scene.sources[0].pose,
            position_m=[1.0, 2.0, 1.4],
            orientation_ypr_deg=[0.0, 0.0, 0.0],
        ),
    )
    back_source = replace(
        front_source,
        pose=replace(front_source.pose, orientation_ypr_deg=[180.0, 0.0, 0.0]),
    )
    front = replace(scene, sources=[front_source], receivers=[receiver], objects=[])
    back = replace(scene, sources=[back_source], receivers=[receiver], objects=[])
    backend = PyroomacousticsHighFrequencyBackend(
        max_order=0,
        ray_tracing=False,
        air_absorption=False,
    )

    front_rir = backend.simulate(front, config)[0]
    back_rir = backend.simulate(back, config)[0]
    direct = int(round(2.0 / front.environment.sound_speed_m_s * config.sample_rate))
    front_peak = float(np.max(np.abs(front_rir[direct : direct + 16])))
    back_peak = float(np.max(np.abs(back_rir[direct : direct + 16])))

    assert front_peak > 0.0
    assert back_peak < front_peak * 0.05


def test_opt_in_path_event_high_backend_runs_material_scene():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.08,
        num_obstacles_range=(0, 0),
    )
    legacy = HybridRIRScene(
        room_dim=[4.0, 3.5, 2.8],
        rt60=0.4,
        mic_pos=[2.0, 1.7, 1.2],
        source_pos=[
            [1.0, 1.7, 1.2],
            [1.2, 2.0, 1.3],
            [3.2, 2.7, 1.4],
            [0.7, 0.8, 1.5],
            [3.3, 0.8, 1.6],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    scene = upgrade_hybrid_scene_to_v2(
        legacy,
        seed=23,
        room_type="office",
    )

    rir = PathEventHighFrequencyBackend(max_order=1).simulate(
        scene,
        config,
    )

    assert rir.shape == (config.num_sources, config.num_samples)
    assert np.all(np.isfinite(rir))
    assert np.all(np.max(np.abs(rir), axis=1) > 0.0)


def test_opt_in_m4_backend_preserves_early_paths_and_serializes_coupling():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.35,
        crossover_hz=1000.0,
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(0, 0),
    )
    legacy = HybridRIRScene(
        room_dim=[4.0, 3.5, 2.8],
        rt60=0.4,
        mic_pos=[2.0, 1.7, 1.2],
        source_pos=[
            [1.0, 1.7, 1.2],
            [1.2, 2.0, 1.3],
            [3.2, 2.7, 1.4],
            [0.7, 0.8, 1.5],
            [3.3, 0.8, 1.6],
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    scene = upgrade_hybrid_scene_to_v2(
        legacy,
        seed=24,
        scene_id="m4-backend-test",
        room_type="office",
    )
    m3_backend = PathEventHighFrequencyBackend(max_order=2)
    m4_backend = PathEventFDNHighFrequencyBackend(
        max_order=2,
        mixing_time_s=0.024,
        transition_duration_s=0.016,
        fdn_seed=25,
    )
    coherent = m3_backend.simulate(scene, config)
    coupled = m4_backend.simulate(scene, config)

    assert coupled.shape == coherent.shape
    assert np.all(np.isfinite(coupled))
    assert m4_backend.last_late_field_metadata["production_default_changed"] is False
    assert m4_backend.last_late_field_metadata["target_rt60_origin"] == (
        "scene_material_predicted_octave_rt60_s"
    )
    for channel_metadata in m4_backend.last_late_field_metadata["channels"]:
        channel = channel_metadata["channel"]
        start = channel_metadata["transition_start_sample"]
        assert np.array_equal(coupled[channel, : start + 1], coherent[channel, : start + 1])
        assert channel_metadata["pre_transition_max_abs_error"] == 0.0
        assert channel_metadata["energy"]["relative_energy_error"] < 1e-10

    rir, metadata = generate_hybrid_rir(
        config,
        scene=scene,
        low_backend=ZeroHighFrequencyBackend(),
        high_backend=m4_backend,
    )
    late_field = metadata["bands"]["high"]["late_field"]

    assert tuple(rir.shape) == (5, config.num_samples)
    assert late_field["policy"] == "puresound.path_event_fdn_coupling.v1"
    assert late_field["renderer"] == "path_event_early_multiband_fdn_late"
    assert metadata["bands"]["high"]["backend"] == (
        "PathEventFDNHighFrequencyBackend"
    )


def test_high_band_alignment_removes_energy_before_physical_arrival():
    from puresound.audio.hybrid_rir import _align_high_band_direct

    config = HybridRIRConfig(
        sample_rate=1000,
        duration=0.5,
        sound_speed=100.0,
        num_obstacles_range=(0, 0),
    )
    distances = [1.0, 1.2, 1.4, 1.6, 1.8]
    scene = HybridRIRScene(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.4,
        mic_pos=[2.0, 2.0, 1.5],
        source_pos=[
            [2.0 + distance, 2.0, 1.5] for distance in distances
        ],
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    rir = np.zeros((5, config.num_samples), dtype=np.float64)
    for channel, distance in enumerate(distances):
        expected = int(round(distance / config.sound_speed * config.sample_rate))
        rir[channel, 0] = 0.01
        rir[channel, expected + 40] = 1.0

    aligned = _align_high_band_direct(rir, scene, config)

    for channel, distance in enumerate(distances):
        first_physical = int(
            np.floor(distance / config.sound_speed * config.sample_rate)
        )
        assert np.count_nonzero(aligned[channel, :first_physical]) == 0
        direct_sample = int(np.argmax(np.abs(aligned[channel])))
        assert first_physical <= direct_sample <= first_physical + 1


def test_low_band_causality_clip_preserves_arrival_sample():
    config = HybridRIRConfig(
        sample_rate=1000,
        duration=0.5,
        sound_speed=100.0,
        num_obstacles_range=(0, 0),
    )
    scene = HybridRIRScene(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.4,
        mic_pos=[2.0, 2.0, 1.5],
        source_pos=[[3.0, 2.0, 1.5]] * 5,
        source_labels=["near_0", "near_1", "far_0", "far_1", "far_2"],
    )
    rir = np.zeros((5, config.num_samples), dtype=np.float64)
    rir[:, :10] = 1.0
    rir[:, 10] = 2.0

    clipped = _clip_rir_before_physical_arrival(rir, scene, config)

    assert np.count_nonzero(clipped[:, :10]) == 0
    assert np.all(clipped[:, 10] == 2.0)


def test_scene_sampling_constrains_3d_distance_and_feasible_rt60():
    """near/far membership is decided on the true 3D source-receiver distance
    (the value written to channel_map distance_m), and the sampled rt60 is
    always Sabine-feasible so the metadata matches the realized reverberation."""
    import numpy as np

    from puresound.audio.hybrid_rir import (
        HybridRIRConfig,
        _min_feasible_rt60,
        sample_hybrid_rir_scene,
    )

    cfg = HybridRIRConfig()
    for seed in range(60):
        scene = sample_hybrid_rir_scene(cfg, seed=seed)
        assert scene.rt60 >= _min_feasible_rt60(np.asarray(scene.room_dim)) - 1e-9
        for d3, label in zip(scene.source_distances(), scene.source_labels):
            if label.startswith("near"):
                lo, hi = cfg.near_distance_range
                assert lo - 1e-6 <= d3 <= hi + 1e-6, (label, d3)
            else:
                assert d3 <= cfg.far_distance_range[1] + 1e-6, (label, d3)
