import json
from dataclasses import replace

import numpy as np
import pytest

from puresound.audio.multiband_fdn import design_multiband_fdn
from puresound.audio.binaural_renderer import analytic_first_order_binaural_decoder
from puresound.audio.hybrid_rir import HybridRIRConfig, sample_material_first_rir_scene
from puresound.audio.spatial_rir import render_room_scene_spatial_rir
from puresound.audio.spatial_late_field import (
    couple_receiver_array_early_late,
    fibonacci_sphere_directions,
    render_spatial_fdn_late_field,
)


def _render(seed=51):
    sample_rate = 8000
    sample_count = 2400
    design = design_multiband_fdn(
        sample_rate,
        {500.0: 0.35, 1000.0: 0.30, 2000.0: 0.25},
        delay_line_count=8,
        seed=50,
    )
    excitation = np.zeros(sample_count)
    excitation[12] = 1.0
    positions = [[-0.085, 0.0, 0.0], [0.085, 0.0, 0.0]]
    return render_spatial_fdn_late_field(
        design,
        excitation,
        positions,
        plane_wave_count=24,
        seed=seed,
    )


def test_fibonacci_directions_are_unit_seeded_and_nearly_balanced():
    first = fibonacci_sphere_directions(64, seed=48)
    repeated = fibonacci_sphere_directions(64, seed=48)
    changed = fibonacci_sphere_directions(64, seed=49)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert np.linalg.norm(first, axis=1) == pytest.approx(np.ones(64), abs=1e-14)
    assert np.linalg.norm(np.mean(first, axis=0)) < 0.01


def test_spatial_fdn_render_is_synchronized_deterministic_and_json_safe():
    first = _render()
    repeated = _render()

    assert first.receiver_rirs.shape == (2, 2400)
    assert first.ambisonic_acn_sn3d.shape == (4, 2400)
    assert np.array_equal(first.receiver_rirs, repeated.receiver_rirs)
    assert np.array_equal(first.ambisonic_acn_sn3d, repeated.ambisonic_acn_sn3d)
    assert not np.array_equal(first.receiver_rirs[0], first.receiver_rirs[1])
    assert np.all(np.isfinite(first.receiver_rirs))
    assert first.metadata["ambisonic"]["channel_labels"] == ["W", "Y", "Z", "X"]
    json.dumps(first.metadata, allow_nan=False)


def test_receiver_directivity_is_a_shared_field_projection():
    omni = _render(seed=52)
    sample_rate = omni.metadata["sample_rate"]
    design = design_multiband_fdn(
        sample_rate,
        {500.0: 0.35, 1000.0: 0.30, 2000.0: 0.25},
        delay_line_count=8,
        seed=50,
    )
    excitation = np.zeros(2400)
    excitation[12] = 1.0
    directional = render_spatial_fdn_late_field(
        design,
        excitation,
        [[-0.085, 0.0, 0.0], [0.085, 0.0, 0.0]],
        receiver_directivity_ids=["cardioid", "figure_eight"],
        receiver_orientations_ypr_deg=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        plane_wave_count=24,
        seed=52,
    )

    assert not np.array_equal(omni.receiver_rirs, directional.receiver_rirs)
    assert directional.metadata["receiver_directivity_ids"] == [
        "cardioid",
        "figure_eight",
    ]


def test_array_coupling_preserves_early_samples_and_post_start_energy():
    sample_rate = 8000
    coherent = np.zeros((2, 1600))
    coherent[0, 20] = 1.0
    coherent[1, 21] = 0.9
    coherent[:, 120] = [0.3, 0.25]
    rng = np.random.default_rng(53)
    late = rng.standard_normal((2, 1600)) * np.exp(-np.arange(1600) / 400.0)

    result = couple_receiver_array_early_late(
        coherent,
        late,
        sample_rate,
        [20, 21],
    )

    target_energy = 0.0
    output_energy = 0.0
    gains = set()
    for receiver_index, channel in enumerate(result.metadata["channels"]):
        start = channel["transition_start_sample"]
        target_energy += float(np.dot(coherent[receiver_index, start:], coherent[receiver_index, start:]))
        output_energy += float(np.dot(result.rir[receiver_index, start:], result.rir[receiver_index, start:]))
        gains.add(channel["shared_diffuse_gain"])
        assert np.array_equal(
            result.rir[receiver_index, : start + 1],
            coherent[receiver_index, : start + 1],
        )
    assert output_energy == pytest.approx(target_energy, rel=1e-12, abs=1e-15)
    assert len(gains) == 1
    assert result.metadata["energy_policy"].startswith("one_shared_array_gain")


def test_high_level_scene_renderer_returns_complete_array_foa_and_brir():
    config = HybridRIRConfig(
        sample_rate=8000,
        duration=0.35,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(config, seed=54, room_type="office")
    original = scene.receivers[0]
    first = replace(
        original,
        pose=replace(
            original.pose,
            position_m=(np.asarray(original.pose.position_m) + [-0.04, 0.0, 0.0]).tolist(),
        ),
        array_id="binaural",
        channel_index=0,
    )
    second = replace(
        original,
        transducer_id="receiver-right",
        pose=replace(
            original.pose,
            position_m=(np.asarray(original.pose.position_m) + [0.04, 0.0, 0.0]).tolist(),
        ),
        array_id="binaural",
        channel_index=1,
    )
    array_scene = replace(scene, receivers=[first, second])

    rendered = render_room_scene_spatial_rir(
        array_scene,
        sample_rate=8000,
        duration_s=0.35,
        max_order=1,
        delay_line_count=8,
        plane_wave_count=16,
        seed=55,
        decoder=analytic_first_order_binaural_decoder(8000),
    )

    assert rendered.receiver_rirs.shape == (2, 2800)
    assert rendered.ambisonic_acn_sn3d.shape == (4, 2800)
    assert rendered.binaural is not None
    assert rendered.binaural.brir.shape == (2, 2800)
    assert rendered.metadata["production_default_changed"] is False
    assert rendered.metadata["finite_output"] is True
    json.dumps(rendered.metadata, allow_nan=False)
