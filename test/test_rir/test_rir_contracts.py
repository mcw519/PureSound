"""Unit tests for the frozen R0 contracts.

These tests cover ``RIRArray``,
``RenderContext``, ``BackendCapabilities`` and a metadata contract.  These
tests cover the behaviour the later stages will rely on, including the
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


class TestRIRArray:
    def test_axis_order_is_channels_then_samples(self):
        assert RIR_AXIS_ORDER == ("channel", "sample")
        rir = RIRArray(np.zeros((5, 320)), 16000)
        assert rir.num_channels == 5
        assert rir.num_samples == 320
        assert rir.duration_s == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "samples", [np.zeros(320), np.zeros((2, 3, 4)), np.zeros((0, 320))]
    )
    def test_rejects_bad_shapes(self, samples):
        with pytest.raises(ValueError):
            RIRArray(samples, 16000)

    def test_rejects_complex_and_bad_rate(self):
        with pytest.raises(ValueError):
            RIRArray(np.zeros((2, 8), dtype=complex), 16000)
        with pytest.raises(ValueError):
            RIRArray(np.zeros((2, 8)), 0)

    def test_dtype_conversions(self):
        rir = RIRArray(np.zeros((2, 8), dtype=np.float32), 16000)
        assert rir.as_compute().samples.dtype == RIR_COMPUTE_DTYPE
        assert rir.as_delivery().samples.dtype == RIR_DELIVERY_DTYPE

    def test_is_finite_detects_nan(self):
        payload = np.zeros((2, 8))
        assert RIRArray(payload, 16000).is_finite()
        payload[1, 3] = np.nan
        assert not RIRArray(payload, 16000).is_finite()

    def test_first_nonzero_sample(self):
        payload = np.zeros((1, 16))
        payload[0, 7] = 0.5
        rir = RIRArray(payload, 16000)
        assert rir.first_nonzero_sample(0) == 7
        assert RIRArray(np.zeros((1, 16)), 16000).first_nonzero_sample(0) is None

    def test_causality_preserves_the_arrival_sample(self):
        """Energy at exactly floor(d/c*fs) is legal; one sample earlier is not."""

        payload = np.zeros((2, 64))
        payload[0, 20] = 1.0  # exactly at the boundary -> allowed
        payload[1, 19] = 1.0  # one sample early -> violation
        rir = RIRArray(payload, 16000)
        assert rir.violates_causality([20, 20]) == (1,)

    def test_causality_requires_one_boundary_per_channel(self):
        with pytest.raises(ValueError):
            RIRArray(np.zeros((3, 16)), 16000).violates_causality([1, 2])


class TestRenderContext:
    def test_derived_quantities(self):
        ctx = RenderContext(16000, 25600, 343.0, 1000.0, 20.0, 1000.0)
        assert ctx.duration_s == pytest.approx(1.6)
        assert ctx.nyquist_hz == 8000.0

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
    def test_rejects_invalid_configurations(self, kwargs):
        base = dict(
            sample_rate=16000,
            num_samples=25600,
            sound_speed=343.0,
            crossover_hz=1000.0,
            low_fmin_hz=20.0,
            low_fmax_hz=1000.0,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            RenderContext(**base)

    def test_from_config_reads_the_hybrid_config_fields(self):
        from puresound.audio.rir.contracts import HybridRIRConfig

        config = HybridRIRConfig(sample_rate=16000, duration=1.6)
        ctx = RenderContext.from_config(config)
        assert ctx.sample_rate == 16000
        assert ctx.num_samples == config.num_samples
        assert ctx.crossover_hz == config.crossover_hz

    def test_first_physical_sample_uses_floor(self):
        ctx = RenderContext(16000, 25600, 344.3618948236363, 1000.0, 20.0, 1000.0)
        # 2.3172 m / 344.3619 * 16000 = 107.66 -> floor 107
        assert ctx.first_physical_sample(2.3172) == 107
        assert ctx.first_physical_sample(0.0) == 0
        with pytest.raises(ValueError):
            ctx.first_physical_sample(-1.0)

    def test_sound_speed_choice_moves_the_boundary(self):
        """A 1.4 m/s difference is enough to shift the boundary by a sample."""

        default_c = RenderContext(16000, 25600, 343.0, 1000.0, 20.0, 1000.0)
        scene_c = RenderContext(
            16000, 25600, 344.3618948236363, 1000.0, 20.0, 1000.0
        )
        assert default_c.first_physical_sample(2.3172) == 108
        assert scene_c.first_physical_sample(2.3172) == 107


class TestBackendCapabilities:
    def test_declares_determinism_explicitly(self):
        cap = BackendCapabilities(
            backend_id="pyroomacoustics",
            band=BackendBand.HIGH,
            deterministic_for_fixed_seed=False,
            source_convention="1/(4*pi*d)",
            optional_dependencies=("pyroomacoustics",),
        )
        assert cap.deterministic_for_fixed_seed is False
        assert cap.band is BackendBand.HIGH

    def test_band_accepts_a_plain_string(self):
        cap = BackendCapabilities("x", "low", True, "1/r")
        assert cap.band is BackendBand.LOW

    def test_rejects_empty_identity(self):
        with pytest.raises(ValueError):
            BackendCapabilities("", BackendBand.LOW, True, "1/r")
        with pytest.raises(ValueError):
            BackendCapabilities("x", BackendBand.LOW, True, "  ")

    def test_missing_dependencies_reports_absent_modules(self):
        cap = BackendCapabilities(
            "x",
            BackendBand.HIGH,
            True,
            "1/r",
            optional_dependencies=("numpy", "definitely_not_installed_xyz"),
        )
        assert cap.missing_dependencies() == ("definitely_not_installed_xyz",)
        assert not cap.is_available

    def test_available_when_all_dependencies_import(self):
        cap = BackendCapabilities(
            "x", BackendBand.LOW, True, "1/r", optional_dependencies=("numpy",)
        )
        assert cap.is_available


class TestMetadataContract:
    def test_accepts_a_well_formed_item(self):
        assert validate_rir_metadata(_metadata()) == ()

    def test_reports_missing_top_level_keys(self):
        payload = _metadata()
        del payload["crossover"]
        problems = validate_rir_metadata(payload)
        assert any("crossover" in text for text in problems)

    def test_reports_missing_channel_map_keys(self):
        payload = _metadata()
        del payload["scene"]["channel_map"][0]["distance_m"]
        problems = validate_rir_metadata(payload)
        assert any("distance_m" in text for text in problems)

    def test_reports_duplicate_channel_indices(self):
        payload = _metadata()
        entry = dict(payload["scene"]["channel_map"][0])
        payload["scene"]["channel_map"].append(entry)
        problems = validate_rir_metadata(payload)
        assert any("duplicate channel" in text for text in problems)

    def test_reports_non_finite_distance(self):
        payload = _metadata()
        payload["scene"]["channel_map"][0]["distance_m"] = float("nan")
        problems = validate_rir_metadata(payload)
        assert any("finite" in text for text in problems)

    def test_reports_a_non_mapping_scene(self):
        payload = _metadata(scene="not-a-mapping")
        assert validate_rir_metadata(payload) != ()


class TestResolveSoundSpeed:
    def test_prefers_the_scene_environment(self):
        assert resolve_sound_speed(_metadata()) == pytest.approx(344.36)

    def test_falls_back_to_the_config(self):
        payload = _metadata()
        del payload["scene"]["environment"]
        assert resolve_sound_speed(payload) == pytest.approx(343.0)

    def test_ignores_non_positive_or_non_finite_values(self):
        payload = _metadata()
        payload["scene"]["environment"]["sound_speed_m_s"] = 0.0
        assert resolve_sound_speed(payload) == pytest.approx(343.0)
        payload["scene"]["environment"]["sound_speed_m_s"] = float("inf")
        assert resolve_sound_speed(payload) == pytest.approx(343.0)

    def test_raises_when_no_sound_speed_is_available(self):
        payload = _metadata(config={})
        del payload["scene"]["environment"]
        with pytest.raises(ValueError):
            resolve_sound_speed(payload)

    def test_rejects_boolean_masquerading_as_a_number(self):
        payload = _metadata()
        payload["scene"]["environment"]["sound_speed_m_s"] = True
        assert resolve_sound_speed(payload) == pytest.approx(343.0)


