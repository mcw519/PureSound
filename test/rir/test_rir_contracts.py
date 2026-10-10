"""Contracts in ``puresound.audio.rir.contracts``.

``RIRArray``, ``RenderContext``, ``BackendCapabilities`` and the metadata
validator are what every renderer and bank reader builds on, including the
sound-speed precedence that decides where the causality boundary falls.
"""

from __future__ import annotations

import numpy as np
import pytest

from puresound.audio.rir.contracts import (
    RIR_AXIS_ORDER,
    RIR_COMPUTE_DTYPE,
    RIR_DELIVERY_DTYPE,
    BackendBand,
    BackendCapabilities,
    HybridRIRConfig,
    RenderContext,
    RIRArray,
    resolve_sound_speed,
    validate_rir_metadata,
)


def _metadata(**overrides):
    payload = {
        "bands": {},
        "config": {"sound_speed": 343.0},
        "crossover": {},
        "output_calibration": {},
        "rir_filename": "room.wav",
        "metadata_filename": "room.json",
        "rir_index": 0,
        "room_id": "room_000000",
        "room_index": 0,
        "sample_id": "room_000000_000000",
        "scene": {
            "environment": {"sound_speed_m_s": 344.36},
            "channel_map": [
                {
                    "channel": 0,
                    "label": "near_0",
                    "source_pos": [1.0, 1.0, 1.4],
                    "distance_m": 0.63,
                    "horizontal_distance_m": 0.5,
                }
            ],
        },
    }
    payload.update(overrides)
    return payload


def _render_context(**overrides):
    kwargs = dict(
        sample_rate=16000,
        num_samples=25600,
        sound_speed=343.0,
        crossover_hz=1000.0,
        low_fmin_hz=20.0,
        low_fmax_hz=1000.0,
    )
    kwargs.update(overrides)
    return RenderContext(**kwargs)


def test_rir_array_layout_dtypes_and_inspection():
    assert RIR_AXIS_ORDER == ("channel", "sample")
    rir = RIRArray(np.zeros((5, 320), dtype=np.float32), 16000)
    assert (rir.num_channels, rir.num_samples) == (5, 320)
    assert rir.duration_s == pytest.approx(0.02)
    assert rir.as_compute().samples.dtype == RIR_COMPUTE_DTYPE
    assert rir.as_delivery().samples.dtype == RIR_DELIVERY_DTYPE
    assert rir.first_nonzero_sample(0) is None

    payload = np.zeros((2, 16))
    payload[0, 7] = 0.5
    assert RIRArray(payload, 16000).is_finite()
    assert RIRArray(payload, 16000).first_nonzero_sample(0) == 7
    payload[1, 3] = np.nan
    assert not RIRArray(payload, 16000).is_finite()


@pytest.mark.parametrize(
    "samples,sample_rate",
    [
        (np.zeros(320), 16000),
        (np.zeros((2, 3, 4)), 16000),
        (np.zeros((0, 320)), 16000),
        (np.zeros((2, 8), dtype=complex), 16000),
        (np.zeros((2, 8)), 0),
    ],
)
def test_rir_array_rejects_bad_shape_dtype_or_rate(samples, sample_rate):
    with pytest.raises(ValueError):
        RIRArray(samples, sample_rate)


def test_causality_check_allows_the_arrival_sample_and_needs_one_boundary_per_channel():
    """Energy at exactly floor(d/c*fs) is legal; one sample earlier is not."""

    payload = np.zeros((2, 64))
    payload[0, 20] = 1.0
    payload[1, 19] = 1.0
    assert RIRArray(payload, 16000).violates_causality([20, 20]) == (1,)
    with pytest.raises(ValueError):
        RIRArray(np.zeros((3, 16)), 16000).violates_causality([1, 2])


def test_render_context_derived_quantities_and_from_config():
    ctx = _render_context()
    assert ctx.duration_s == pytest.approx(1.6)
    assert ctx.nyquist_hz == 8000.0

    config = HybridRIRConfig(sample_rate=16000, duration=1.6)
    from_config = RenderContext.from_config(config)
    assert from_config.sample_rate == 16000
    assert from_config.num_samples == config.num_samples
    assert from_config.crossover_hz == config.crossover_hz


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"num_samples": 0},
        {"sound_speed": 0.0},
        {"crossover_hz": 9000.0},  # above Nyquist
        {"low_fmin_hz": 2000.0},  # not below low_fmax_hz
    ],
)
def test_render_context_rejects_invalid_configurations(kwargs):
    with pytest.raises(ValueError):
        _render_context(**kwargs)


def test_first_physical_sample_floors_and_depends_on_sound_speed():
    """A 1.4 m/s difference is enough to shift the boundary by a sample."""

    scene_c = _render_context(sound_speed=344.3618948236363)
    # 2.3172 m / 344.3619 * 16000 = 107.66 -> floor 107
    assert scene_c.first_physical_sample(2.3172) == 107
    assert _render_context().first_physical_sample(2.3172) == 108
    assert scene_c.first_physical_sample(0.0) == 0
    with pytest.raises(ValueError):
        scene_c.first_physical_sample(-1.0)


def test_backend_capabilities_declare_band_determinism_and_dependencies():
    cap = BackendCapabilities(
        backend_id="pyroomacoustics",
        band=BackendBand.HIGH,
        deterministic_for_fixed_seed=False,
        source_convention="1/(4*pi*d)",
        optional_dependencies=("numpy", "definitely_not_installed_xyz"),
    )
    assert cap.deterministic_for_fixed_seed is False
    assert cap.band is BackendBand.HIGH
    assert cap.missing_dependencies() == ("definitely_not_installed_xyz",)
    assert not cap.is_available

    available = BackendCapabilities(
        "x", "low", True, "1/r", optional_dependencies=("numpy",)
    )
    assert available.band is BackendBand.LOW
    assert available.is_available

    for backend_id, convention in (("", "1/r"), ("x", "  ")):
        with pytest.raises(ValueError):
            BackendCapabilities(backend_id, BackendBand.LOW, True, convention)


def _drop_crossover(payload):
    del payload["crossover"]
    return "crossover"


def _drop_channel_distance(payload):
    del payload["scene"]["channel_map"][0]["distance_m"]
    return "distance_m"


def _duplicate_channel(payload):
    payload["scene"]["channel_map"].append(dict(payload["scene"]["channel_map"][0]))
    return "duplicate channel"


def _nan_distance(payload):
    payload["scene"]["channel_map"][0]["distance_m"] = float("nan")
    return "finite"


def _non_mapping_scene(payload):
    payload["scene"] = "not-a-mapping"
    return ""


@pytest.mark.parametrize(
    "corrupt",
    [
        _drop_crossover,
        _drop_channel_distance,
        _duplicate_channel,
        _nan_distance,
        _non_mapping_scene,
    ],
)
def test_metadata_validator_accepts_a_well_formed_item_and_reports_each_defect(corrupt):
    assert validate_rir_metadata(_metadata()) == ()
    payload = _metadata()
    expected = corrupt(payload)
    problems = validate_rir_metadata(payload)
    assert problems != ()
    assert any(expected in text for text in problems)


@pytest.mark.parametrize(
    "environment_speed,expected",
    [
        (344.36, 344.36),  # the scene environment wins
        (None, 343.0),  # no environment: fall back to the config
        (0.0, 343.0),
        (float("inf"), 343.0),
        (True, 343.0),  # a boolean is not a number
    ],
)
def test_resolve_sound_speed_prefers_a_valid_scene_value(environment_speed, expected):
    payload = _metadata()
    if environment_speed is None:
        del payload["scene"]["environment"]
    else:
        payload["scene"]["environment"]["sound_speed_m_s"] = environment_speed
    assert resolve_sound_speed(payload) == pytest.approx(expected)


def test_resolve_sound_speed_raises_when_none_is_available():
    payload = _metadata(config={})
    del payload["scene"]["environment"]
    with pytest.raises(ValueError):
        resolve_sound_speed(payload)
