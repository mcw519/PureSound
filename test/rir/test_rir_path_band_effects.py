"""Band-limited fractional delay, band gains, speech directivity, occlusion."""

from dataclasses import replace

import numpy as np
import pytest

from puresound.audio.rir.path_events import (
    PathBandGain,
    PathEvent,
    apply_fresnel_kirchhoff_occlusion,
    directivity_band_gain,
    directivity_pressure_gain,
    fractional_read,
    generate_scene_shoebox_path_events,
    material_absorption_relaxation_models,
    minimum_phase_band_filter,
    render_path_events,
    screen_insertion_gain,
    windowed_sinc_kernel,
)
from puresound.audio.rir.path_events.directivity import SPEECH_DIRECTIVITY_RELATIVE_DB
from puresound.audio.rir.scene.schema import Pose, SceneObject
from puresound.audio.rir.scene.world_presets import world_preset

SR = 16000


def magnitude_db(taps, frequency, offsets=None):
    offsets = np.arange(len(taps)) if offsets is None else offsets
    w = 2 * np.pi * frequency / SR
    return 20 * np.log10(abs(np.sum(taps * np.exp(-1j * w * offsets))))


@pytest.mark.parametrize("delay", [0.0, 7.25, 20.5, 100.875])
def test_windowed_sinc_kernel_and_read_agree(delay):
    signal = np.random.default_rng(0).standard_normal(2000)
    start, kernel = windowed_sinc_kernel(delay)
    assert start == int(np.floor(delay)) - 15 and kernel.size == 32
    convolved = np.convolve(signal, kernel)
    expected = np.zeros(2100)
    lo = max(start, 0)
    expected[lo : start + convolved.size] = convolved[lo - start :][: 2100 - lo]
    read = fractional_read(signal, np.arange(2100) - delay)
    np.testing.assert_allclose(read, expected, atol=1e-5)


def test_windowed_sinc_is_flat_for_every_fraction():
    for fraction in np.linspace(0, 1, 41)[:-1]:
        start, kernel = windowed_sinc_kernel(50 + fraction)
        offsets = np.arange(kernel.size) + start - (50 + fraction)
        for frequency in (500, 4000, 7000):
            assert abs(magnitude_db(kernel, frequency, offsets)) < 0.02


def test_fractional_read_is_zero_away_from_the_signal():
    out = fractional_read(np.ones(10), np.array([-100.0, -40.0, 5.0, 60.0]))
    assert out[0] == 0 and out[1] == 0 and out[3] == 0
    assert out[2] == pytest.approx(1.0, abs=0.02)


def test_static_renderer_offers_windowed_sinc_without_changing_its_default():
    spec = world_preset("turn")
    bounds, _ = material_absorption_relaxation_models(spec.room)
    events = generate_scene_shoebox_path_events(
        spec.room, max_order=0, boundary_admittance_models=bounds
    )
    kwargs = dict(sample_rate_hz=SR, num_samples=600)
    lagrange = render_path_events(events, **kwargs)
    sinc = render_path_events(events, fractional_delay="windowed_sinc", **kwargs)
    first = int(np.floor(events.events[0].delay_s * SR))
    assert not np.any(lagrange[:first])  # the causal contract of the default
    assert np.any(sinc[first - 15 : first])  # band-limited pre-ringing
    assert np.sum(sinc) == pytest.approx(np.sum(lagrange), rel=1e-3)
    with pytest.raises(ValueError, match="fractional_delay"):
        render_path_events(events, fractional_delay="cubic", **kwargs)


def test_band_gain_interpolates_in_log_frequency_and_combines():
    gain = PathBandGain([125.0, 1000.0], [1.0, 0.25], "test")
    assert gain.at([0.0, 125.0, np.sqrt(125.0 * 1000.0), 8000.0]) == pytest.approx(
        [1.0, 1.0, 0.625, 0.25]
    )
    product = gain.times(PathBandGain([500.0], [2.0], "flat"))
    assert product.frequencies_hz == [125.0, 500.0, 1000.0]
    assert product.at([1000.0])[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        PathBandGain([1000.0, 500.0], [1.0, 1.0], "unordered")
    with pytest.raises(ValueError):
        PathBandGain([1000.0], [-1.0], "negative")


def test_band_gain_round_trips_and_is_absent_from_plain_events():
    spec = world_preset("turn")
    event = generate_scene_shoebox_path_events(spec.room, max_order=0).events[0]
    plain = replace(event, source_directivity_gain=1.0, band_gain=None)
    assert "band_gain" not in plain.to_dict()
    banded = replace(plain, band_gain=PathBandGain([250.0, 4000.0], [0.8, 0.1], "t"))
    assert PathEvent.from_dict(banded.to_dict()) == banded


def test_minimum_phase_band_filter_matches_a_directivity_curve():
    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    target_db = SPEECH_DIRECTIVITY_RELATIVE_DB[-1]
    gain = PathBandGain(bands, list(10 ** (target_db / 20)), "behind")
    taps = minimum_phase_band_filter(gain, SR)
    for band, expected in zip(bands, target_db):
        assert magnitude_db(taps, band) == pytest.approx(expected, abs=0.15)
    energy = np.cumsum(taps**2) / np.sum(taps**2)
    assert energy[7] > 0.9  # minimum phase: the arrival stays at the start


def test_speech_directivity_follows_the_measured_table():
    forward = [0.0, 0.0, 0.0]
    for angle, row in ((0, 0), (90, 6), (180, 12)):
        direction = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0]
        gain = directivity_band_gain("speech_human", forward, direction)
        assert 20 * np.log10(gain.magnitude) == pytest.approx(
            SPEECH_DIRECTIVITY_RELATIVE_DB[row], abs=1e-9
        )
    with pytest.raises(ValueError, match="frequency"):
        directivity_pressure_gain("speech_human", forward, [1.0, 0.0, 0.0])


def test_generator_attaches_speech_directivity_as_band_gain():
    spec = world_preset("turn")
    bounds, _ = material_absorption_relaxation_models(spec.room)

    def events_with(directivity):
        tr = replace(
            spec.room.sources[0], directivity_id=directivity, pose=Pose([3.0, 2.5, 1.5], [0, 0, 0])
        )
        return generate_scene_shoebox_path_events(
            replace(spec.room, sources=[tr]), max_order=1, boundary_admittance_models=bounds
        )

    human = events_with("speech_human")
    assert all(e.band_gain is not None and e.source_directivity_gain == 1.0 for e in human.events)
    assert human.metadata["source_directivity_model"] == "monson2012_octave_band_table"
    cardioid = events_with("speech_cardioid")
    assert all(e.band_gain is None for e in cardioid.events)
    # Facing away from the microphone: the direct path is 19 dB down at 4 kHz.
    direct = human.events[0]
    assert 20 * np.log10(direct.band_gain.at([4000.0])[0]) == pytest.approx(-18.9, abs=0.5)


def screen(footprint, z_max=3.0):
    return SceneObject("s", "partition", footprint, 0.0, z_max, "m", 0.3, 0.1)


def test_screen_gain_limits():
    start, stop = np.array([1.0, 0.0, 1.5]), np.array([5.0, 0.0, 1.5])
    half_plane = screen([[2.9, -50], [3.0, -50], [3.0, 0], [2.9, 0]])
    gain = screen_insertion_gain(start, stop, half_plane, room_height_m=3.0, sound_speed_m_s=343.0)
    np.testing.assert_allclose(20 * np.log10(np.abs(gain)), -6.02, atol=0.1)
    wall = screen([[2.9, -50], [3.0, -50], [3.0, 50], [2.9, 50]])
    gain = screen_insertion_gain(start, stop, wall, room_height_m=3.0, sound_speed_m_s=343.0)
    assert np.all(20 * np.log10(np.abs(gain)) < -35)
    behind = screen([[6.0, -0.5], [6.5, -0.5], [6.5, 0.5], [6.0, 0.5]])
    gain = screen_insertion_gain(start, stop, behind, room_height_m=3.0, sound_speed_m_s=343.0)
    np.testing.assert_allclose(gain, 1.0)


def test_occlusion_preset_fades_continuously_and_darkens_with_frequency():
    spec = world_preset("occlusion")
    bounds, _ = material_absorption_relaxation_models(spec.room)
    source = spec.sources[0]
    levels = []
    for t in np.arange(0.0, 10.0, 0.02):
        state = source.pose_at(np.array([t]))[0]
        tr = replace(spec.room.sources[0], pose=Pose(state[:3].tolist(), [state[3], 0, 0]))
        events = generate_scene_shoebox_path_events(
            replace(spec.room, sources=[tr, *spec.room.sources[1:]]),
            max_order=0,
            occlusion_model="fresnel_kirchhoff",
            boundary_admittance_models=bounds,
        )
        direct = events.events[0]
        assert direct.visible
        levels.append(20 * np.log10(direct.band_gain.at([250.0, 1000.0, 4000.0])))
    levels = np.array(levels)
    assert np.max(np.abs(np.diff(levels, axis=0))) < 1.0  # no switching
    deepest = levels[len(levels) // 2]
    assert deepest[0] > -4 and deepest[2] < -10  # low frequencies bend around


def test_occlusion_without_objects_leaves_events_untouched():
    spec = world_preset("turn")
    events = generate_scene_shoebox_path_events(spec.room, max_order=1)
    assert apply_fresnel_kirchhoff_occlusion(events, spec.room) is events


def test_fresnel_occlusion_is_exclusive_with_discrete_interactions():
    spec = world_preset("occlusion")
    with pytest.raises(ValueError, match="include_scene_interactions"):
        generate_scene_shoebox_path_events(
            spec.room, occlusion_model="fresnel_kirchhoff", include_scene_interactions=True
        )
    with pytest.raises(ValueError, match="occlusion_model"):
        generate_scene_shoebox_path_events(spec.room, occlusion_model="ray")


def test_interaction_paths_carry_the_speech_directivity_of_their_own_departure():
    spec = world_preset("occlusion")
    behind_screen = Pose([4.0, 2.5, 1.5], [0.0, 0.0, 0.0])  # facing away from the microphone
    talker = replace(spec.room.sources[0], pose=behind_screen)
    blocker = replace(spec.room.objects[0], transmission=0.2)
    room = replace(spec.room, sources=[talker], objects=[blocker])
    bounds, _ = material_absorption_relaxation_models(room)
    events = generate_scene_shoebox_path_events(
        room, max_order=1, boundary_admittance_models=bounds, include_scene_interactions=True
    )
    derived = [e for e in events.events if e.path_type in {"scattering", "transmission", "diffraction"}]
    assert {e.path_type for e in derived} == {"scattering", "transmission", "diffraction"}
    for event in derived:
        expected = directivity_band_gain(
            "speech_human", talker.pose.orientation_ypr_deg, event.departure_direction_unit
        )
        assert event.band_gain is not None, event.event_id
        np.testing.assert_allclose(event.band_gain.magnitude, expected.magnitude)
