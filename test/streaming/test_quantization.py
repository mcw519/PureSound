"""The catalog's int8 (flash) graphs on the machine running the tests.

int8 kernels differ by instruction set, and on some an int8 product can
saturate: x86 without VNNI multiplies uint8 activations by int8 weights through
16-bit pairs. Each flash artifact is therefore held to its float graph here, on
whatever CPU runs the suite, over real speech. Run with ``-s`` to see each
measured distance.
"""

from __future__ import annotations

import pytest

from puresound.inference import ModelZoo
from puresound.inference.benchmark import flash_deviation
from puresound.streaming.quantization import INT8_MAX_RELATIVE_ERROR


def _models_with_flash():
    zoo = ModelZoo.default()
    return [pytest.param(zoo, model, id=model.id) for model in zoo.list()
            if {"default", "flash"} <= {artifact.variant for artifact in model.artifacts}]


@pytest.mark.parametrize("zoo, model", _models_with_flash())
def test_every_flash_graph_stays_near_its_float_graph_on_this_cpu(zoo, model):
    error = flash_deviation(zoo, model)
    print(f"{model.id}: flash relative error {error:.4f} on this CPU")
    assert error < INT8_MAX_RELATIVE_ERROR, (
        f"{model.id}: flash relative error {error:.3f} against its float graph on this CPU"
    )
