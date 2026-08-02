import json

import numpy as np
import pytest

from puresound.audio.rir.render.binaural import (
    AmbisonicBinauralDecoder,
    analytic_first_order_binaural_decoder,
    render_ambisonic_brir,
)


def test_hrtf_decoder_contract_validates_shape_and_provenance():
    decoder = AmbisonicBinauralDecoder(
        sample_rate=16000,
        firs=np.zeros((2, 4, 8)),
        reference_id="licensed-dataset-subject-001",
        provenance={"dataset": "unit-test", "license": "unit-test"},
    )

    assert decoder.to_dict()["input"]["channel_labels"] == ["W", "Y", "Z", "X"]
    json.dumps(decoder.to_dict(include_coefficients=True), allow_nan=False)
    with pytest.raises(ValueError, match="shape"):
        AmbisonicBinauralDecoder(
            sample_rate=16000,
            firs=np.zeros((2, 2, 8)),
            reference_id="bad",
            provenance={},
        )


def test_analytic_decoder_renders_causal_distinct_ears_deterministically():
    decoder = analytic_first_order_binaural_decoder(8000)
    ambisonic = np.zeros((4, 128))
    ambisonic[0, 10] = 1.0
    ambisonic[1, 10] = 0.5

    first = render_ambisonic_brir(ambisonic, 8000, decoder)
    repeated = render_ambisonic_brir(ambisonic, 8000, decoder)

    assert first.brir.shape == (2, 128)
    assert np.array_equal(first.brir, repeated.brir)
    assert np.all(first.brir[:, :10] == 0.0)
    assert first.brir[0, 10] == pytest.approx(0.75)
    assert first.brir[1, 10] == pytest.approx(0.25)
    assert first.metadata["decoder"]["decoder_kind"] == (
        "analytic_demonstration_not_hrtf"
    )


def test_binaural_render_rejects_sample_rate_mismatch():
    decoder = analytic_first_order_binaural_decoder(16000)
    with pytest.raises(ValueError, match="sample rates"):
        render_ambisonic_brir(np.zeros((4, 32)), 8000, decoder)
