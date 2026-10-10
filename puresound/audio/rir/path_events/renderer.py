"""Fractional-delay rendering of a PathEventSet.

With the default forward Lagrange kernel, samples before the discrete arrival
bin ``floor(delay * fs)`` are exactly zero, which is the causality contract the
rest of the pipeline and bank QC rely on.  ``fractional_delay="windowed_sinc"``
trades that strict support (about 1 ms of band-limited pre-ringing) for a
response that is flat to 7 kHz at every fractional delay; see
``path_events.fractional_delay``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from puresound.audio.rir.physics.impedance.admittance import (
    digital_locally_reacting_reflection_filter,
)
from puresound.audio.rir.path_events.band_filter import minimum_phase_band_filter
from puresound.audio.rir.path_events.fractional_delay import windowed_sinc_kernel
from puresound.audio.rir.path_events.schema import (
    NormalizedAdmittanceModel,
    PathEvent,
    PathEventSet,
    locally_reacting_reflection_coefficient,
)


FRACTIONAL_DELAY_POLICY = "puresound.causal_forward_lagrange.v1"
FRACTIONAL_DELAY_METHODS = ("lagrange", "windowed_sinc")


def causal_fractional_delay_kernel(
    delay_samples: float,
    *,
    order: int = 3,
) -> tuple[int, np.ndarray]:
    """Return a forward Lagrange kernel and its integer start sample.

    For ``delay_samples = N + mu``, taps are placed at ``N .. N+order`` and
    interpolate the continuous impulse at ``mu``.  Consequently all samples
    before ``floor(delay_samples)`` are exactly zero.  The one-sided filter is
    causal but trades high-frequency accuracy for that strict support rule.
    """
    delay = float(delay_samples)
    interpolation_order = int(order)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("delay_samples must be finite and non-negative")
    if interpolation_order != order or not 1 <= interpolation_order <= 8:
        raise ValueError("fractional-delay order must be an integer in [1, 8]")
    start = int(math.floor(delay))
    fraction = delay - float(start)
    if fraction <= 1e-12:
        fraction = 0.0
    elif 1.0 - fraction <= 1e-12:
        start += 1
        fraction = 0.0
    nodes = np.arange(interpolation_order + 1, dtype=np.float64)
    coefficients = np.ones(interpolation_order + 1, dtype=np.float64)
    for index in range(interpolation_order + 1):
        for other in range(interpolation_order + 1):
            if other != index:
                coefficients[index] *= (
                    (fraction - nodes[other]) / (nodes[index] - nodes[other])
                )
    return start, coefficients


def render_path_events(
    event_set_or_events: PathEventSet | Iterable[PathEvent],
    *,
    sample_rate_hz: float,
    num_samples: int,
    fractional_delay_order: int = 3,
    fractional_delay: str = "lagrange",
    include_propagation_delay: bool = True,
    surface_admittance_models: (
        Mapping[str, NormalizedAdmittanceModel] | None
    ) = None,
    boundary_gain_tolerance: float = 1e-9,
    maximum_boundary_filter_tail_samples: int | None = None,
) -> np.ndarray:
    """Render events into one causal real RIR channel.

    Scalar real gains are rendered directly. A complex specular path is
    renderable only when every interacting surface has a supported passive
    rational admittance model. The stored analog gain samples are checked
    against those models before their causal digital reflection filters are
    applied. A path's ``band_gain`` is applied as a minimum-phase FIR.
    ``include_propagation_delay=False`` returns the causal path filter at zero
    time, for renderers that apply continuous propagation separately.
    """
    sample_rate = float(sample_rate_hz)
    length = int(num_samples)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if length != num_samples or length <= 0:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(event_set_or_events, PathEventSet):
        events = event_set_or_events.events
    else:
        events = list(event_set_or_events)
    if not events:
        raise ValueError("at least one path event is required")
    if fractional_delay not in FRACTIONAL_DELAY_METHODS:
        raise ValueError(
            f"fractional_delay must be one of {FRACTIONAL_DELAY_METHODS}"
        )
    tolerance = float(boundary_gain_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary gain tolerance must be finite and non-negative")
    admittance_models = dict(surface_admittance_models or {})
    if maximum_boundary_filter_tail_samples is None:
        boundary_tail_limit = None
    else:
        boundary_tail_limit = int(maximum_boundary_filter_tail_samples)
        if (
            boundary_tail_limit != maximum_boundary_filter_tail_samples
            or boundary_tail_limit < 1
        ):
            raise ValueError(
                "maximum boundary-filter tail samples must be a positive integer"
            )
    output = np.zeros(length, dtype=np.float64)
    # Discretizing one boundary reflection costs a root solve and an
    # eigenvalue check, and the same (surface model, incidence angle) pair
    # recurs across image paths that strike a wall at the same angle, which
    # is common on a shoebox.  The models are frozen value-equal
    # dataclasses, so caching on the value is safe.  Scoped to this call so
    # nothing accumulates across scenes.
    boundary_filters: dict[tuple[Any, float, float], Any] = {}

    def boundary_filter_for(model, incidence_cosine: float):
        key = (model, float(incidence_cosine), sample_rate)
        cached = boundary_filters.get(key)
        if cached is None:
            cached = digital_locally_reacting_reflection_filter(
                model,
                incidence_cosine,
                sample_rate,
            )
            boundary_filters[key] = cached
        return cached

    def with_band_gain(event: PathEvent, response: np.ndarray) -> np.ndarray:
        if event.band_gain is None:
            return response
        return np.convolve(
            response, minimum_phase_band_filter(event.band_gain, sample_rate)
        )

    for event in events:
        if not event.visible:
            continue
        if include_propagation_delay and fractional_delay == "windowed_sinc":
            start, kernel = windowed_sinc_kernel(event.delay_s * sample_rate)
            if start < 0:
                kernel, start = kernel[-start:], 0
        elif include_propagation_delay:
            start, kernel = causal_fractional_delay_kernel(
                event.delay_s * sample_rate, order=fractional_delay_order,
            )
        else:
            # Moving-source transport owns travel time; retain the physical
            # event contract and render only its causal boundary response.
            start, kernel = 0, np.ones(1, dtype=np.float64)
        if start >= length:
            continue
        available = length - start
        direction_gain = (
            float(event.source_directivity_gain)
            * float(event.receiver_directivity_gain)
        )
        interaction_models = [
            admittance_models.get(surface_id)
            for surface_id in event.surface_ids
        ]
        filterable_boundary_path = bool(
            event.path_type in {
                "specular_reflection",
                "scattering",
            }
            and event.interaction_types
            and all(
                interaction in {"reflection", "scattering"}
                for interaction in event.interaction_types
            )
        )
        if filterable_boundary_path and any(
            model is not None for model in interaction_models
        ):
            if any(model is None for model in interaction_models):
                raise ValueError(
                    "every reflection surface in a filtered path requires "
                    "an admittance model"
                )
            expected_gain = []
            for frequency in event.gain_spectrum.frequencies_hz:
                reflection_product = complex(1.0, 0.0)
                for model, incidence_cosine in zip(
                    interaction_models,
                    event.incidence_cosines,
                ):
                    reflection_product *= (
                        locally_reacting_reflection_coefficient(
                            model.normalized_admittance(frequency),
                            incidence_cosine,
                        )
                    )
                expected_gain.append(
                    math.sqrt(event.energy_partition_fraction)
                    * reflection_product
                    / float(event.distance_m)
                )
            maximum_gain_error = float(
                np.max(
                    np.abs(
                        event.gain_spectrum.values
                        - np.asarray(expected_gain, dtype=np.complex128)
                    )
                )
            )
            if maximum_gain_error > tolerance:
                raise ValueError(
                    "stored path gain spectrum does not match the supplied "
                    f"boundary model (maximum error {maximum_gain_error:.3g})"
                )
            local_length = available
            if boundary_tail_limit is not None:
                local_length = min(
                    available,
                    int(kernel.size) + boundary_tail_limit,
                )
            local_signal = np.zeros(local_length, dtype=np.float64)
            local_signal[: min(local_length, kernel.size)] = kernel[:local_length]
            for model, incidence_cosine in zip(
                interaction_models,
                event.incidence_cosines,
            ):
                boundary_filter = boundary_filter_for(model, incidence_cosine)
                local_signal = boundary_filter.filter_signal(local_signal)
            local_signal = with_band_gain(event, local_signal)[:available]
            output[start : start + local_signal.size] += (
                direction_gain
                * math.sqrt(event.energy_partition_fraction)
                / float(event.distance_m)
                * local_signal
            )
        else:
            gain = (
                event.gain_spectrum.constant_real_value()
                * direction_gain
            )
            segment = with_band_gain(event, kernel)
            stop = min(length, start + segment.size)
            output[start:stop] += gain * segment[: stop - start]
    return output


def partition_path_events_by_arrival(
    event_set_or_events: PathEventSet | Iterable[PathEvent],
    *,
    early_window_s: float = 0.050,
) -> dict[str, list[PathEvent]]:
    """Partition paths into direct, early-reflection, and later buckets.

    The early/later threshold is relative to the direct arrival. Boundary
    filter tails remain owned by the path that generated them; this is a
    geometric arrival partition, not a time-domain sample window.
    """
    if isinstance(event_set_or_events, PathEventSet):
        events = list(event_set_or_events.events)
    else:
        events = list(event_set_or_events)
    window = float(early_window_s)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("early_window_s must be finite and positive")
    direct = [event for event in events if event.path_type == "direct"]
    if len(direct) != 1:
        raise ValueError("arrival partition requires exactly one direct path")
    direct_delay = float(direct[0].delay_s)
    reflected = [event for event in events if event.path_type != "direct"]
    early = [
        event
        for event in reflected
        if event.delay_s <= direct_delay + window
    ]
    later = [
        event
        for event in reflected
        if event.delay_s > direct_delay + window
    ]
    if len(direct) + len(early) + len(later) != len(events):
        raise RuntimeError("path arrival partition did not preserve all events")
    return {
        "direct": direct,
        "early_reflections": early,
        "later_reflections": later,
    }


__all__ = [
    "FRACTIONAL_DELAY_POLICY",
    "causal_fractional_delay_kernel",
    "partition_path_events_by_arrival",
    "render_path_events",
]
