"""Optional first-order Ambisonic to binaural BRIR rendering for M4.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


AMBISONIC_BINAURAL_POLICY = "puresound.ambisonic_binaural_decoder.v1"


@dataclass(frozen=True)
class AmbisonicBinauralDecoder:
    """Causal two-ear FIR decoder for ACN/SN3D first-order Ambisonics.

    ``firs`` has shape ``[ear, ambisonic_channel, tap]``.  An HRTF-derived
    decoder can be supplied without changing the spatial late-field renderer;
    callers must retain its dataset identity and processing provenance.
    """

    sample_rate: int
    firs: np.ndarray
    reference_id: str
    provenance: Mapping[str, Any]
    decoder_kind: str = "hrtf_derived"

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.firs, dtype=np.float64)
        if self.sample_rate <= 0:
            raise ValueError("decoder sample_rate must be positive")
        if coefficients.ndim != 3 or coefficients.shape[:2] != (2, 4):
            raise ValueError("decoder FIRs must have shape [2 ears, 4 channels, taps]")
        if coefficients.shape[2] < 1 or not np.all(np.isfinite(coefficients)):
            raise ValueError("decoder FIRs must contain finite taps")
        if not self.reference_id:
            raise ValueError("decoder reference_id is required")
        if not self.decoder_kind:
            raise ValueError("decoder_kind is required")
        object.__setattr__(self, "firs", coefficients.copy())
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self, *, include_coefficients: bool = False) -> dict[str, Any]:
        """Return strict-JSON metadata, optionally including FIR coefficients."""

        result: dict[str, Any] = {
            "policy": AMBISONIC_BINAURAL_POLICY,
            "sample_rate": int(self.sample_rate),
            "decoder_kind": str(self.decoder_kind),
            "reference_id": str(self.reference_id),
            "provenance": dict(self.provenance),
            "input": {
                "order": 1,
                "channel_order": "ACN",
                "normalization": "SN3D",
                "channel_labels": ["W", "Y", "Z", "X"],
            },
            "ear_order": ["left", "right"],
            "fir_shape": list(self.firs.shape),
            "causal_fir": True,
            "finite_coefficients": bool(np.all(np.isfinite(self.firs))),
        }
        if include_coefficients:
            result["firs"] = self.firs.tolist()
        return result


@dataclass(frozen=True)
class BinauralBRIRRender:
    """Rendered left/right BRIR and the decoder contract used to produce it."""

    brir: np.ndarray
    metadata: Mapping[str, Any]


def analytic_first_order_binaural_decoder(
    sample_rate: int,
) -> AmbisonicBinauralDecoder:
    """Return a one-tap lateral first-order demonstration decoder.

    This decoder is intentionally labelled analytic and headless.  It proves
    the optional BRIR pipeline without pretending to be a measured HRTF.  A
    production binaural render should inject measured, licensed FIRs through
    :class:`AmbisonicBinauralDecoder`.
    """

    coefficients = np.zeros((2, 4, 1), dtype=np.float64)
    coefficients[0, 0, 0] = 0.5
    coefficients[0, 1, 0] = 0.5
    coefficients[1, 0, 0] = 0.5
    coefficients[1, 1, 0] = -0.5
    return AmbisonicBinauralDecoder(
        sample_rate=int(sample_rate),
        firs=coefficients,
        reference_id="puresound-analytic-headless-foa-demo-v1",
        decoder_kind="analytic_demonstration_not_hrtf",
        provenance={
            "measurement": False,
            "license": "project_code_only",
            "description": "lateral first-order pressure decoder",
            "production_hrtf_replacement_required": True,
        },
    )


def render_ambisonic_brir(
    ambisonic_acn_sn3d: Any,
    sample_rate: int,
    decoder: AmbisonicBinauralDecoder,
) -> BinauralBRIRRender:
    """Convolve a four-channel FOA RIR with a causal two-ear FIR decoder."""

    ambisonic = np.asarray(ambisonic_acn_sn3d, dtype=np.float64)
    if ambisonic.ndim != 2 or ambisonic.shape[0] != 4:
        raise ValueError("Ambisonic input must have shape [4 channels, samples]")
    if ambisonic.shape[1] < 1 or not np.all(np.isfinite(ambisonic)):
        raise ValueError("Ambisonic input must contain finite samples")
    if sample_rate <= 0 or int(sample_rate) != decoder.sample_rate:
        raise ValueError("Ambisonic and decoder sample rates must match")

    output_length = ambisonic.shape[1] + decoder.firs.shape[2] - 1
    brir = np.zeros((2, output_length), dtype=np.float64)
    for ear_index in range(2):
        for channel_index in range(4):
            brir[ear_index] += np.convolve(
                ambisonic[channel_index],
                decoder.firs[ear_index, channel_index],
                mode="full",
            )
    metadata = {
        "policy": AMBISONIC_BINAURAL_POLICY,
        "operation": "causal_fir_decode",
        "sample_rate": int(sample_rate),
        "input_sample_count": int(ambisonic.shape[1]),
        "output_sample_count": int(output_length),
        "decoder": decoder.to_dict(include_coefficients=False),
        "finite_output": bool(np.all(np.isfinite(brir))),
    }
    return BinauralBRIRRender(
        brir=np.asarray(brir, dtype=np.float64),
        metadata=metadata,
    )


__all__ = [
    "AMBISONIC_BINAURAL_POLICY",
    "AmbisonicBinauralDecoder",
    "BinauralBRIRRender",
    "analytic_first_order_binaural_decoder",
    "render_ambisonic_brir",
]
