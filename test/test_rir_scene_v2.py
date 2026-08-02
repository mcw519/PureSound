from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from puresound.audio.hybrid_rir import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    HybridRIRConfig,
    HybridRIRScene,
    PyroomacousticsHighFrequencyBackend,
    generate_hybrid_rir,
    material_modal_damping_metadata,
    sample_material_first_rir_scene,
    upgrade_hybrid_scene_to_v2,
)
from puresound.audio.rir_metrics import octave_band_rir
from puresound.audio.rir_scene import (
    MaterialSpectrum,
    RoomSceneV2,
    SurfaceMaterial,
)


class _ImpulseBackend:
    def simulate(self, scene, config):
        output = np.zeros(
            (config.num_sources, config.num_samples), dtype=np.float64
        )
        output[:, 20] = 1.0
        return output


def _small_v2_scene(seed=1):
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.25,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    return config, sample_material_first_rir_scene(
        config, seed=seed, room_type="office"
    )


def test_material_spectrum_interpolates_in_log_frequency():
    spectrum = MaterialSpectrum([125.0, 500.0], [0.1, 0.5], [0.01, 0.02])

    assert spectrum.at(250.0) == pytest.approx(0.3)
    assert spectrum.at(20.0) == pytest.approx(0.1)
    assert spectrum.at(8000.0) == pytest.approx(0.5)


def test_v2_scene_metadata_round_trips_without_loss():
    _, scene = _small_v2_scene()

    restored = RoomSceneV2.from_json(scene.to_json())

    assert restored.to_dict() == scene.to_dict()
    assert sorted(surface.boundary for surface in restored.surfaces) == [
        "ceiling",
        "east",
        "floor",
        "north",
        "south",
        "west",
    ]
    assert {patch.role for surface in restored.surfaces for patch in surface.patches} == {
        "door",
        "window",
    }
    assert restored.to_metadata()["rt60_origin"] == (
        "surface_material_sabine_prediction"
    )


def test_v2_scene_round_trips_synchronized_receiver_array():
    _, scene = _small_v2_scene(seed=47)
    original = scene.receivers[0]
    second = replace(
        original,
        transducer_id="receiver-1",
        pose=replace(
            original.pose,
            position_m=(np.asarray(original.pose.position_m) + [0.0, 0.08, 0.0]).tolist(),
        ),
        array_id="array-0",
        channel_index=1,
    )
    first = replace(original, array_id="array-0", channel_index=0)
    array_scene = replace(scene, receivers=[first, second])

    restored = RoomSceneV2.from_json(array_scene.to_json())

    assert len(restored.receivers) == 2
    assert restored.receivers[1].transducer_id == "receiver-1"
    assert restored.receivers[1].channel_index == 1
    assert restored.to_dict() == array_scene.to_dict()


def test_legacy_five_source_generator_rejects_multi_receiver_scene():
    config, scene = _small_v2_scene(seed=48)
    original = scene.receivers[0]
    second = replace(
        original,
        transducer_id="receiver-1",
        pose=replace(
            original.pose,
            position_m=(np.asarray(original.pose.position_m) + [0.0, 0.05, 0.0]).tolist(),
        ),
        channel_index=1,
    )
    array_scene = replace(scene, receivers=[original, second])

    with pytest.raises(ValueError, match="M4 spatial renderer"):
        generate_hybrid_rir(
            config,
            array_scene,
            low_backend=_ImpulseBackend(),
            high_backend=_ImpulseBackend(),
        )


def test_complex_impedance_round_trips_and_interpolates_in_si_units():
    _, scene = _small_v2_scene()
    surface = scene.surfaces[0]
    material = scene.materials[surface.material_id]
    centers = [125.0, 500.0]
    impedance_material = replace(
        material,
        impedance_real=MaterialSpectrum(centers, [400.0, 800.0]),
        impedance_imag=MaterialSpectrum(centers, [-200.0, 200.0]),
    )
    impedance_scene = replace(
        scene,
        materials={
            **scene.materials,
            surface.material_id: impedance_material,
        },
    )

    restored = RoomSceneV2.from_json(impedance_scene.to_json())
    restored_material = restored.materials[surface.material_id]

    assert restored_material.impedance_at(250.0) == pytest.approx(
        complex(600.0, 0.0)
    )
    assert restored.to_dict() == impedance_scene.to_dict()


def test_surface_material_rejects_incomplete_or_active_impedance():
    centers = [125.0, 250.0]
    unit_interval = MaterialSpectrum(centers, [0.1, 0.2])
    kwargs = {
        "material_id": "test",
        "name": "test",
        "family": "test",
        "absorption": unit_interval,
        "scattering": unit_interval,
        "transmission": unit_interval,
        "provenance": "test",
    }

    with pytest.raises(ValueError, match="requires both"):
        SurfaceMaterial(
            **kwargs,
            impedance_real=MaterialSpectrum(centers, [400.0, 400.0]),
        )
    with pytest.raises(ValueError, match="negative real"):
        SurfaceMaterial(
            **kwargs,
            impedance_real=MaterialSpectrum(centers, [-1.0, 400.0]),
            impedance_imag=MaterialSpectrum(centers, [0.0, 0.0]),
        )


def test_effective_boundary_mixes_declared_patch_impedance_as_admittance():
    _, scene = _small_v2_scene()
    surface = next(surface for surface in scene.surfaces if surface.patches)
    components = [
        (
            1.0 - sum(patch.area_fraction for patch in surface.patches),
            surface.material_id,
            400.0,
        ),
        *[
            (patch.area_fraction, patch.material_id, 800.0 + 100.0 * index)
            for index, patch in enumerate(surface.patches)
        ],
    ]
    materials = dict(scene.materials)
    for _weight, material_id, impedance in components:
        material = materials[material_id]
        centers = material.absorption.center_frequencies_hz
        materials[material_id] = replace(
            material,
            impedance_real=MaterialSpectrum.constant(impedance, centers),
            impedance_imag=MaterialSpectrum.constant(0.0, centers),
        )
    impedance_scene = replace(scene, materials=materials)

    effective = impedance_scene.effective_boundary_materials()[surface.boundary]
    expected = 1.0 / sum(
        weight / impedance for weight, _material_id, impedance in components
    )

    assert effective.impedance_at(125.0) == pytest.approx(complex(expected, 0.0))
    assert scene.effective_boundary_materials()[surface.boundary].impedance_at(
        125.0
    ) is None


def test_material_sampler_is_deterministic_and_decay_is_not_scalar():
    _, first = _small_v2_scene(seed=19)
    _, second = _small_v2_scene(seed=19)

    assert first.to_dict() == second.to_dict()
    predicted = first.predicted_octave_rt60_s()
    assert predicted["125"] > predicted["4000"]
    assert first.rt60 == pytest.approx(
        np.median([predicted["500"], predicted["1000"]])
    )


def test_calibrated_mode_preserves_source_power_scale_without_peak_normalizing():
    config, scene = _small_v2_scene()
    sources = [
        replace(source, power_db_spl_at_1m=74.0)
        for source in scene.sources
    ]
    quiet_scene = replace(scene, sources=sources)
    loud_scene = replace(
        scene,
        sources=[replace(source, power_db_spl_at_1m=94.0) for source in sources],
    )
    calibrated = replace(
        config,
        output_mode="calibrated",
        match_crossover_energy=False,
        record_realized_metrics=False,
    )
    backend = _ImpulseBackend()

    quiet, _ = generate_hybrid_rir(
        calibrated, quiet_scene, low_backend=backend, high_backend=backend
    )
    loud, metadata = generate_hybrid_rir(
        calibrated, loud_scene, low_backend=backend, high_backend=backend
    )

    assert float(loud.abs().max() / quiet.abs().max()) == pytest.approx(10.0, rel=1e-5)
    assert metadata["output_calibration"]["per_item_peak_normalized"] is False
    assert float(loud.abs().max()) != pytest.approx(config.normalize_peak)


def test_v2_generation_records_realized_octave_metrics():
    config, scene = _small_v2_scene()
    config = replace(
        config,
        output_mode="calibrated",
        record_realized_metrics=True,
        match_crossover_energy=False,
    )

    rir, metadata = generate_hybrid_rir(
        config,
        scene,
        low_backend=AnalyticModalLowFrequencyBackend(num_modes_per_axis=2),
        high_backend=_ImpulseBackend(),
    )

    assert rir.shape == (5, config.num_samples)
    assert metadata["scene"]["schema_version"] == "rir_scene.v2"
    assert metadata["bands"]["high"]["boundary_model"] == (
        "frequency_dependent_surface_materials"
    )
    assert len(metadata["realized_acoustics"]["channels"]) == 5
    assert "octave_bands" in metadata["realized_acoustics"]["channels"][0]["metrics"]


def test_material_modal_damping_is_surface_and_mode_dependent():
    config, scene = _small_v2_scene(seed=7)
    baseline = material_modal_damping_metadata(
        scene, config, max_mode_index=4, max_modes=128
    )
    west_surface = next(
        surface for surface in scene.surfaces if surface.boundary == "west"
    )
    west_material = scene.materials[west_surface.material_id]
    centers = west_material.absorption.center_frequencies_hz
    absorbing_id = "test:absorbing_west"
    absorbing_west = replace(
        west_material,
        material_id=absorbing_id,
        absorption=MaterialSpectrum.constant(0.9, centers),
    )
    changed_surfaces = [
        replace(surface, material_id=absorbing_id)
        if surface.boundary == "west"
        else surface
        for surface in scene.surfaces
    ]
    changed_scene = replace(
        scene,
        surfaces=changed_surfaces,
        materials={**scene.materials, absorbing_id: absorbing_west},
    )
    changed = material_modal_damping_metadata(
        changed_scene, config, max_mode_index=4, max_modes=128
    )

    def rates(metadata):
        return {
            tuple(mode["indices"]): mode["amplitude_decay_rate_per_s"]
            for mode in metadata["modes"]
        }

    baseline_rates = rates(baseline)
    changed_rates = rates(changed)
    delta_x = changed_rates[(1, 0, 0)] - baseline_rates[(1, 0, 0)]
    delta_y = changed_rates[(0, 1, 0)] - baseline_rates[(0, 1, 0)]

    assert delta_x > 0.0
    assert delta_y > 0.0
    # The x-axial mode has twice the west-wall participation because its
    # cosine-mode volume norm along x is Lx/2 rather than Lx.
    assert delta_x / delta_y == pytest.approx(2.0, rel=1e-6)
    assert len({round(value, 6) for value in changed_rates.values()}) > 1


def test_material_modal_loss_scale_changes_decay_rate_and_q_reciprocally():
    config, scene = _small_v2_scene(seed=8)
    baseline = material_modal_damping_metadata(
        scene,
        config,
        max_mode_index=4,
        max_modes=128,
    )
    scaled = material_modal_damping_metadata(
        scene,
        config,
        max_mode_index=4,
        max_modes=128,
        loss_scale=0.5,
    )

    assert scaled["material_modal_loss_scale"] == 0.5
    for baseline_mode, scaled_mode in zip(
        baseline["modes"],
        scaled["modes"],
    ):
        assert scaled_mode["indices"] == baseline_mode["indices"]
        assert scaled_mode["amplitude_decay_rate_per_s"] == pytest.approx(
            0.5 * baseline_mode["amplitude_decay_rate_per_s"]
        )
        assert scaled_mode["quality_factor"] == pytest.approx(
            2.0 * baseline_mode["quality_factor"]
        )


def test_material_damped_low_backend_removes_global_rt60_envelope_metadata():
    config, scene = _small_v2_scene(seed=9)
    config = replace(
        config,
        output_mode="calibrated",
        record_realized_metrics=False,
        match_crossover_energy=False,
    )
    low_backend = AnalyticModalLowFrequencyBackend(
        num_modes_per_axis=3,
        material_modal_damping=True,
        material_modal_loss_scale=0.5,
    )

    rir, metadata = generate_hybrid_rir(
        config,
        scene,
        low_backend=low_backend,
        high_backend=_ImpulseBackend(),
    )

    low_metadata = metadata["bands"]["low"]
    assert rir.shape == (5, config.num_samples)
    assert low_metadata["boundary_model"] == "per_mode_surface_material_damping"
    assert low_metadata["global_rt60_envelope_applied"] is False
    assert low_metadata["modal_damping"]["material_modal_loss_scale"] == 0.5
    assert low_metadata["modal_damping"]["mode_count"] > 1
    assert low_metadata["modal_damping"]["rt60_s"]["minimum"] < (
        low_metadata["modal_damping"]["rt60_s"]["maximum"]
    )


def test_exact_pytard_modal_recurrence_decays_faster_for_absorbing_boundaries():
    config = HybridRIRConfig(
        sample_rate=16000,
        duration=0.12,
        crossover_hz=1000.0,
        room_dim_range=((1.0, 1.0),) * 3,
        num_obstacles_range=(0, 0),
    )
    legacy = HybridRIRScene(
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
    scene = upgrade_hybrid_scene_to_v2(
        legacy, seed=3, room_type="office"
    )

    def with_uniform_absorption(value):
        return replace(
            scene,
            materials={
                key: replace(
                    material,
                    absorption=MaterialSpectrum.constant(
                        value, material.absorption.center_frequencies_hz
                    ),
                )
                for key, material in scene.materials.items()
            },
        )

    backend = GpuARDPytARDBackend(
        low_sample_rate=16000,
        material_modal_damping=True,
    )
    zero_scene = with_uniform_absorption(0.0)
    zero_damped = backend.simulate(zero_scene, config)
    legacy_undamped = GpuARDPytARDBackend(
        low_sample_rate=16000,
        material_modal_damping=False,
        apply_rt60_decay=False,
    ).simulate(zero_scene, config)
    reflective = backend.simulate(with_uniform_absorption(0.03), config)[0]
    absorbing = backend.simulate(with_uniform_absorption(0.6), config)[0]

    assert np.allclose(zero_damped, legacy_undamped, atol=1e-10, rtol=1e-9)
    late_start = int(0.08 * config.sample_rate)
    reflective_late = float(np.square(reflective[late_start:]).sum())
    absorbing_late = float(np.square(absorbing[late_start:]).sum())
    assert np.isfinite(absorbing).all()
    assert absorbing_late < reflective_late * 1e-4


@pytest.mark.skipif(
    importlib.util.find_spec("pyroomacoustics") is None,
    reason="pyroomacoustics not installed",
)
def test_pyroom_backend_applies_frequency_dependent_surface_absorption():
    config = HybridRIRConfig(
        sample_rate=16000,
        duration=0.6,
        room_dim_range=((6.0, 6.0), (5.0, 5.0), (3.0, 3.0)),
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(
        config, seed=2, room_type="classroom"
    )
    centers = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]

    def with_absorption(values):
        materials = {
            key: replace(
                material,
                absorption=MaterialSpectrum(centers, values, [0.01] * len(centers)),
            )
            for key, material in scene.materials.items()
        }
        return replace(scene, materials=materials)

    flat = with_absorption([0.08] * len(centers))
    high_absorbing = with_absorption([0.08, 0.08, 0.10, 0.20, 0.50, 0.85, 0.90])
    backend = PyroomacousticsHighFrequencyBackend(
        max_order=12, ray_tracing=False
    )
    flat_rir = backend.simulate(flat, config)[2]
    absorbing_rir = backend.simulate(high_absorbing, config)[2]
    direct = int(
        round(
            scene.source_distances()[2]
            / float(scene.environment.sound_speed_m_s)
            * config.sample_rate
        )
    )

    def late_to_early_db(rir, center_hz):
        band = octave_band_rir(rir, config.sample_rate, center_hz)
        boundary = direct + int(0.08 * config.sample_rate)
        early = float(np.square(band[direct:boundary]).sum())
        late = float(np.square(band[boundary:]).sum())
        return 10.0 * np.log10((late + 1e-20) / (early + 1e-20))

    low_difference = late_to_early_db(absorbing_rir, 125.0) - late_to_early_db(
        flat_rir, 125.0
    )
    high_difference = late_to_early_db(absorbing_rir, 4000.0) - late_to_early_db(
        flat_rir, 4000.0
    )
    assert abs(low_difference) < 1.0
    assert high_difference < -20.0
