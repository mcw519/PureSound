from dataclasses import replace
import json
import numpy as np
import pytest

from puresound.audio.rir.scene.dynamic import (
    DynamicSceneSpec,
    DynamicSource,
    MotionKeyframe,
)
from puresound.audio.rir.scene.schema import Pose
from puresound.audio.rir.scene.world_presets import world_preset, PRESETS
from puresound.audio.rir.render.dynamic import render_dynamic_scene, near_weights


def direct_scene(position=(2.0, 2.5, 1.4), duration=0.2, end=None):
    s = world_preset("turn", duration_s=duration)
    tr = replace(s.room.sources[0], directivity_id="omnidirectional")
    source = DynamicSource(
        "speaker-a",
        "a",
        (MotionKeyframe(0, position), MotionKeyframe(duration, end or position)),
    )
    return replace(
        s,
        room=replace(s.room, sources=[tr]),
        sources=(source,),
        max_order=0,
        late_reverb=False,
    )


def test_roundtrip_and_presets():
    for name in PRESETS:
        s = world_preset(name)
        assert (
            DynamicSceneSpec.from_dict(json.loads(json.dumps(s.to_dict()))).to_dict()
            == s.to_dict()
        )
    assert near_weights([0.9, 1, 1.1]).tolist() == pytest.approx([1, 0.5, 0])


def test_speech_repeats_only_complete_clips_and_preserves_start_and_gain():
    from puresound.audio.rir.render.dynamic import _emission

    spec = direct_scene(duration=0.25)
    source = replace(spec.sources[0], repeat=True, start_s=0.02, gain_db=-6)
    raw = np.linspace(-0.1, 0.1, 1600)
    original = raw.copy()
    dry = _emission(spec, source, raw, 4000)
    # Two whole 100 ms clips starting at 20 ms, then 30 ms of silence.
    assert not np.any(dry[:320])
    np.testing.assert_array_equal(dry[320:1920], dry[1920:3520])
    assert dry[1120] == pytest.approx(raw[800] * 10 ** (-6 / 20))
    assert not np.any(dry[3520:])
    np.testing.assert_array_equal(raw, original)
    # Noise can still fill the remaining partial cycle.
    noise = _emission(spec, replace(source, role="noise"), raw, 4000)
    assert np.any(noise[3520:])


@pytest.mark.parametrize("repeat", [False, True])
def test_speech_clip_that_cannot_fit_is_rejected_instead_of_cut(repeat):
    from puresound.audio.rir.render.dynamic import _emission

    spec = direct_scene(duration=0.2)
    source = replace(spec.sources[0], repeat=repeat, start_s=0.11)
    with pytest.raises(ValueError, match="complete speech clip needs 0.100 s"):
        _emission(spec, source, np.ones(1600), 3200)


def test_whole_clip_repetition_keeps_reverberation_after_speech_stops():
    spec = replace(direct_scene(duration=0.25), late_reverb=True)
    raw = np.zeros(1600)
    raw[1200] = 0.1
    repeated = render_dynamic_scene(
        replace(spec, sources=(replace(spec.sources[0], repeat=True),)), {"a": raw}
    )
    expected = render_dynamic_scene(spec, {"a": np.r_[raw, raw, np.zeros(800)]})
    np.testing.assert_array_equal(repeated.mixture, expected.mixture)
    assert np.linalg.norm(repeated.mixture[4000:]) > 0


def test_free_field_arrival_distance_and_additivity():
    audio = np.zeros(3200)
    audio[100] = 0.1
    a = render_dynamic_scene(direct_scene(), {"a": audio})
    b = render_dynamic_scene(direct_scene((3.0, 2.5, 1.4)), {"a": audio})
    peak = np.argmax(np.abs(a.mixture))
    sound_speed = a.metadata["scene"]["room"]["environment"]["sound_speed_m_s"]
    assert abs(peak - (100 + 16000 / sound_speed)) <= 1
    # Integrate the fractional-arrival pulse instead of comparing phase-dependent peaks.
    assert 20 * np.log10(abs(b.mixture.sum() / a.mixture.sum())) == pytest.approx(
        -6.0206, abs=0.2
    )
    np.testing.assert_array_equal(sum(a.stems.values()), a.mixture)
    np.testing.assert_array_equal(a.references["target"], a.reference_stems["speaker-a"])


def test_motion_doppler_and_block_invariance():
    s = direct_scene((2.0, 2.5, 1.4), duration=1, end=(3.0, 2.5, 1.4))
    audio = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    a = render_dynamic_scene(s, {"a": audio}, block_size=128)
    b = render_dynamic_scene(s, {"a": audio}, block_size=1024)
    np.testing.assert_array_equal(a.mixture, b.mixture)
    section = a.mixture[4000:14000]
    crossings = np.flatnonzero((section[:-1] < 0) & (section[1:] >= 0))
    frequency = 16000 / np.mean(np.diff(crossings))
    c = s.room.environment.sound_speed_m_s
    assert frequency == pytest.approx(1000 * c / (c + 1), abs=0.25)


@pytest.mark.parametrize(
    "end,match",
    [((0.9, 2.5, 1.4), "0.2 m"), ((7, 2.5, 1.4), "walls"), ((4, 2.5, 1.4), "speed")],
)
def test_trajectory_validation(end, match):
    with pytest.raises(ValueError, match=match):
        direct_scene(end=end)


def test_reverb_survives_silence_and_is_deterministic():
    s = replace(direct_scene(), late_reverb=True)
    audio = np.zeros(3200)
    audio[10] = 0.1
    a = render_dynamic_scene(s, {"a": audio})
    b = render_dynamic_scene(s, {"a": audio})
    np.testing.assert_array_equal(a.mixture, b.mixture)
    # The diffuse field exists even when no reflection is rendered coherently.
    late = a.mixture[int(0.06 * 16000) :]
    assert np.sum(late**2) > 1e-3 * np.sum(a.mixture**2)
    assert np.linalg.norm(a.mixture[-1000:]) < np.linalg.norm(a.mixture[1000:2000])


@pytest.mark.parametrize("late_reverb,error_db", [(False, -120), (True, -40)])
def test_stationary_trajectory_matches_static_path_renderer(late_reverb, error_db):
    """A source that does not move renders exactly as the static pipeline:
    windowed-sinc PathEvent RIR, coupled to the FDN when a late field is on."""
    from scipy.signal import fftconvolve
    from puresound.audio.rir.scene.schema import Pose
    from puresound.audio.rir.path_events import (
        generate_scene_shoebox_path_events,
        material_absorption_relaxation_models,
        render_path_events,
    )
    from puresound.audio.rir.render.coupling import couple_path_event_rir_with_fdn

    spec = replace(direct_scene(duration=0.1), max_order=2, late_reverb=late_reverb)
    source = spec.sources[0]
    room = replace(
        spec.room,
        sources=[
            replace(
                spec.room.sources[0], pose=Pose(list(source.keyframes[0].position_m))
            )
        ],
    )
    bounds, _ = material_absorption_relaxation_models(room)
    events = generate_scene_shoebox_path_events(
        room,
        max_order=2,
        include_scene_interactions=False,
        boundary_admittance_models=bounds,
    )
    audio = np.zeros(1600)
    audio[100:110] = np.linspace(0.01, 0.02, 10)
    rendered = render_dynamic_scene(spec, {"a": audio})
    rir = render_path_events(
        events,
        sample_rate_hz=16000,
        num_samples=len(rendered.mixture),
        fractional_delay="windowed_sinc",
        surface_admittance_models={
            s.surface_id: bounds[s.boundary] for s in room.surfaces
        },
        maximum_boundary_filter_tail_samples=384,
    )
    if late_reverb:
        distance = np.linalg.norm(
            np.array(source.keyframes[0].position_m) - np.array(room.mic_pos)
        )
        rir = couple_path_event_rir_with_fdn(
            rir,
            16000,
            int(round(distance / room.environment.sound_speed_m_s * 16000)),
            {float(k): v for k, v in rendered.metadata["rt60_s"].items()},
            seed=rendered.metadata["fdn_seeds"]["speaker-a"],
        ).rir
    expected = (
        fftconvolve(audio, rir)[: len(rendered.mixture)]
        * rendered.metadata["input_gain"]
    )
    error = rendered.mixture - expected
    assert 10 * np.log10(np.sum(error**2) / np.sum(expected**2)) < error_db


def test_near_weights_follow_emission_before_propagation():
    spec = direct_scene((1.95, 2.5, 1.4), duration=1, end=(2.05, 2.5, 1.4))
    audio = np.zeros(16000)
    emission = 4800
    audio[emission] = 0.1
    result = render_dynamic_scene(spec, {"a": audio})
    distance = np.linalg.norm(
        spec.sources[0].pose_at(emission / 16000)[:3] - np.array(spec.room.mic_pos)
    )
    weight = near_weights(distance, spec.near_radius_m)
    np.testing.assert_allclose(
        result.references["near"], result.references["target"] * weight, atol=1e-8
    )
    assert np.flatnonzero(np.abs(result.references["near"]) > 1e-9)[0] > emission


def test_two_talkers_in_near_region_and_empty_region():
    spec = direct_scene((1.8, 2.5, 1.4), duration=0.2)
    b = DynamicSource("speaker-b", "b", (MotionKeyframe(0, (1.0, 3.3, 1.4)),))
    tr = replace(spec.room.sources[0], transducer_id="speaker-b")
    spec = replace(
        spec,
        room=replace(spec.room, sources=[*spec.room.sources, tr]),
        sources=(*spec.sources, b),
    )
    audio = np.zeros(3200)
    audio[100] = 0.03
    result = render_dynamic_scene(spec, {"a": audio, "b": audio * 0.5})
    np.testing.assert_array_equal(
        result.references["near"],
        result.reference_stems["speaker-a"] + result.reference_stems["speaker-b"],
    )
    empty = render_dynamic_scene(
        replace(spec, near_radius_m=0.2), {"a": audio, "b": audio * 0.5}
    )
    assert not np.any(empty.references["near"])


def test_explicit_repetition_keeps_speech_across_motion_without_loop_spikes():
    spec = direct_scene(duration=0.2)
    source = replace(spec.sources[0], repeat=True)
    audio = np.ones(640) * 0.01
    repeated = render_dynamic_scene(replace(spec, sources=(source,)), {"a": audio})
    once = render_dynamic_scene(spec, {"a": audio})
    assert np.linalg.norm(repeated.mixture[2000:3000]) > 0
    assert np.max(np.abs(once.mixture[2000:3000])) < 1e-12
    assert np.max(np.abs(np.diff(repeated.mixture[500:3000]))) < 0.0002


def test_canonical_float32_asset_can_replay_float64_input():
    spec = direct_scene()
    audio = np.sin(np.arange(3200) * 0.05) * 0.03
    rendered = render_dynamic_scene(spec, {"a": audio})
    replay = render_dynamic_scene(spec, rendered.source_audio)
    np.testing.assert_array_equal(rendered.mixture, replay.mixture)


def test_moving_source_keeps_a_flat_high_frequency_envelope():
    """The fractional part of a drifting delay must not modulate the level."""
    from scipy.signal import hilbert

    s = direct_scene((2.0, 2.5, 1.4), duration=1, end=(3.0, 2.5, 1.4))
    audio = 0.1 * np.sin(2 * np.pi * 6000 * np.arange(16000) / 16000)
    mixture = render_dynamic_scene(s, {"a": audio}).mixture
    envelope = np.abs(hilbert(mixture[3000:13000]))
    distance = np.linspace(1.0, 2.0, 16000)[3000:13000]
    level = 20 * np.log10(envelope * distance)  # undo spherical spreading
    assert np.ptp(level[200:-200]) < 0.2


def test_each_talker_gets_its_own_late_tail():
    spec = replace(direct_scene((2.5, 2.5, 1.4), duration=0.2), late_reverb=True)
    b = DynamicSource("speaker-b", "b", (MotionKeyframe(0, (2.5, 3.0, 1.4)),))
    tr = replace(spec.room.sources[0], transducer_id="speaker-b")
    spec = replace(
        spec,
        room=replace(spec.room, sources=[*spec.room.sources, tr]),
        sources=(*spec.sources, b),
    )
    audio = np.zeros(3200)
    audio[0] = 0.1
    result = render_dynamic_scene(spec, {"a": audio, "b": audio})
    late = slice(int(0.08 * 16000), int(0.3 * 16000))
    a_tail, b_tail = (result.stems[k][late] for k in ("speaker-a", "speaker-b"))
    correlation = np.dot(a_tail, b_tail) / np.linalg.norm(a_tail) / np.linalg.norm(b_tail)
    assert abs(correlation) < 0.3
    seeds = result.metadata["fdn_seeds"]
    assert seeds["speaker-a"] != seeds["speaker-b"]


def test_presets_use_measured_speech_directivity():
    for name in PRESETS:
        room = world_preset(name).room
        talkers = {s.source_id for s in world_preset(name).sources if s.role != "noise"}
        assert {
            tr.directivity_id for tr in room.sources if tr.transducer_id in talkers
        } == {"speech_human"}


def test_late_field_level_follows_a_moving_source():
    """Half-way along a walk the diffuse level matches a source standing there."""

    def late_energy(keyframes, window):
        spec = direct_scene(keyframes[0].position_m, duration=6.0)
        spec = replace(spec, sources=(replace(spec.sources[0], keyframes=keyframes),), late_reverb=True)
        audio = 0.05 * np.random.default_rng(0).standard_normal(6 * 16000)
        on = render_dynamic_scene(spec, {"a": audio})
        off = render_dynamic_scene(replace(spec, late_reverb=False), {"a": audio})
        late = on.mixture[: len(off.mixture)] / on.metadata["input_gain"] - off.mixture / off.metadata["input_gain"]
        return np.mean(late[window] ** 2)

    start, end = (4.6, 1.0, 1.4), (2.4, 4.0, 1.4)
    middle = tuple((a + b) / 2 for a, b in zip(start, end))
    window = slice(int(2.5 * 16000), int(3.5 * 16000))
    moving = late_energy((MotionKeyframe(0, start), MotionKeyframe(6.0, end)), window)
    standing = late_energy((MotionKeyframe(0, middle),), window)
    assert abs(10 * np.log10(moving / standing)) < 1.0


def scene_with(sources, **fields):
    """A scene in the "turn" room holding ``sources`` = [(id, role, keyframes)]."""
    base = world_preset("turn")
    template = base.room.sources[0]
    transducers = [
        replace(
            template,
            transducer_id=sid,
            directivity_id="omnidirectional" if role == "noise" else "speech_human",
            pose=Pose(list(keys[0].position_m), [keys[0].yaw_deg, 0, 0]),
        )
        for sid, role, keys in sources
    ]
    dynamic = tuple(DynamicSource(sid, sid, tuple(keys), role) for sid, role, keys in sources)
    return replace(base, room=replace(base.room, sources=transducers), sources=dynamic, **fields)


def still(x, y, z=1.5):
    return [MotionKeyframe(0, (x, y, z), 180.0)]


def test_v1_scene_converts_to_roles():
    v1 = world_preset("exchange").to_dict()
    v1.pop("reference_rir")
    v1["schema_version"] = "puresound.dynamic_scene.v1"
    v1["fixed_source_id"] = "speaker-b"
    for source in v1["sources"]:
        source["role"] = "noise" if source["role"] == "noise" else "speech"
    spec = DynamicSceneSpec.from_dict(v1)
    assert {s.source_id: s.role for s in spec.sources} == {
        "speaker-a": "interferer",
        "speaker-b": "target",
        "noise": "noise",
    }
    assert spec.reference_rir == "early"
    assert spec.to_dict()["schema_version"] == "puresound.dynamic_scene.v2"


def test_many_talkers_and_moving_noise_are_valid():
    moving = [MotionKeyframe(0, (5.0, 1.0, 0.8)), MotionKeyframe(1.0, (5.0, 2.0, 0.8))]
    spec = scene_with(
        [
            ("ta", "target", still(2.0, 2.0)),
            ("tb", "target", still(2.0, 3.2)),
            ("ia", "interferer", still(4.5, 1.5)),
            ("ib", "interferer", still(4.5, 3.8)),
            ("na", "noise", still(3.5, 1.0, 0.8)),
            ("nb", "noise", moving),
        ]
    )
    assert [s.role for s in spec.sources] == ["target", "target", "interferer", "interferer", "noise", "noise"]


def test_more_than_eight_sources_are_rejected():
    sources = [(f"s{i}", "interferer", still(1.6 + 0.45 * i, 4.0)) for i in range(9)]
    with pytest.raises(ValueError, match="1 to 8 sources"):
        scene_with(sources)


def test_unknown_role_and_reference_rir_are_rejected():
    with pytest.raises(ValueError, match="role"):
        scene_with([("x", "speech", still(2.0, 2.0))])
    with pytest.raises(ValueError, match="reference_rir"):
        scene_with([("x", "target", still(2.0, 2.0))], reference_rir="wet")


def test_busy_preset_has_moving_noise_and_several_talkers():
    spec = world_preset("busy")
    roles = [s.role for s in spec.sources]
    assert roles.count("target") == 1 and roles.count("interferer") == 2 and roles.count("noise") == 2
    noise = [s for s in spec.sources if s.role == "noise"]
    assert any(len(s.keyframes) > 1 for s in noise)


def impulse_render(kind):
    spec = replace(direct_scene((2.5, 2.5, 1.4), duration=0.5), max_order=2, late_reverb=True, reference_rir=kind)
    audio = np.zeros(8000)
    audio[400] = 0.2
    rendered = render_dynamic_scene(spec, {"a": audio})
    sound_speed = spec.room.environment.sound_speed_m_s
    arrival = 400 + 1.5 / sound_speed * 16000
    return rendered, arrival


@pytest.mark.parametrize("kind,window_s", [("early", 0.05), ("direct", 0.006)])
def test_windowed_references_stop_after_their_window(kind, window_s):
    rendered, arrival = impulse_render(kind)
    reference = rendered.reference_stems["speaker-a"].astype(float)
    stem = rendered.stems["speaker-a"].astype(float)
    after = int(arrival + window_s * 16000) + 20  # past the band-limited ringing
    assert np.sum(reference[after:] ** 2) < 1e-6 * np.sum(reference**2)
    assert np.sum(stem[after:] ** 2) > 0.01 * np.sum(stem**2)
    peak = int(round(arrival))
    np.testing.assert_allclose(reference[peak - 2 : peak + 3], stem[peak - 2 : peak + 3], atol=1e-6)
    assert rendered.metadata["reference_rir"] == kind


def test_full_reference_is_the_stem():
    rendered, _ = impulse_render("full")
    np.testing.assert_array_equal(rendered.reference_stems["speaker-a"], rendered.stems["speaker-a"])


def test_anechoic_reference_is_the_dry_signal_at_the_direct_arrival():
    rendered, arrival = impulse_render("anechoic")
    reference = rendered.reference_stems["speaker-a"].astype(float)
    peak = int(np.argmax(np.abs(reference)))
    assert abs(peak - arrival) <= 1
    total = 0.2 * rendered.metadata["input_gain"]
    assert np.sum(reference[peak - 16 : peak + 17]) == pytest.approx(total, rel=1e-3)
    assert np.sum(reference[peak + 40 :] ** 2) < 1e-9


def test_references_sum_their_roles():
    spec = replace(
        scene_with(
            [
                ("ta", "target", still(1.6, 2.5, 1.4)),
                ("tb", "target", still(4.0, 1.5)),
                ("ib", "interferer", still(4.5, 3.8)),
                ("nb", "noise", still(3.5, 1.0, 0.8)),
            ],
            duration_s=0.3,
        ),
        max_order=0,
        late_reverb=False,
    )
    rng = np.random.default_rng(3)
    assets = {s.asset_id: 0.05 * rng.standard_normal(4800) for s in spec.sources}
    rendered = render_dynamic_scene(spec, assets)
    refs = rendered.reference_stems
    np.testing.assert_allclose(rendered.references["target"], refs["ta"] + refs["tb"], atol=1e-7)
    np.testing.assert_allclose(rendered.references["speech"], refs["ta"] + refs["tb"] + refs["ib"], atol=1e-7)
    # Only "ta" (0.6 m) is inside the 1 m near region.
    np.testing.assert_allclose(rendered.references["near"], refs["ta"], atol=1e-7)
    assert set(rendered.audio()) >= {"reference-target", "reference-speech", "reference-near", "reference-source-ta"}
