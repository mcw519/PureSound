import json

import numpy as np

from puresound.audio.rir_constrained_residual import (
    evaluate_residual_ablation,
    fit_causal_decay_residual,
)


def test_constrained_residual_is_causal_bounded_and_improves_holdout():
    sample_rate = 8000
    physical = []
    targets = []
    direct = (20, 30, 40)
    template = np.zeros(1000)
    template[16:] = (
        0.01
        * np.sin(2.0 * np.pi * 700.0 * np.arange(984) / sample_rate)
        * np.exp(-np.arange(984) / 250.0)
    )
    for index, direct_sample in enumerate(direct):
        value = np.zeros(1200)
        value[direct_sample] = 1.0 - 0.1 * index
        value[direct_sample + 80] = 0.2
        residual = np.zeros_like(value)
        residual[direct_sample : direct_sample + template.size] = (
            np.linalg.norm(value[direct_sample:]) * template
        )
        physical.append(value)
        targets.append(value + residual)

    fit = fit_causal_decay_residual(
        targets[:2],
        physical[:2],
        direct[:2],
        sample_rate,
        maximum_residual_to_physical_energy_ratio=0.2,
    )
    ablation = evaluate_residual_ablation(
        targets[2:],
        physical[2:],
        direct[2:],
        sample_rate,
        fit.model,
        physical_first_samples=direct[2:],
    )

    residual = ablation["residual_rirs"][0]
    assert np.count_nonzero(residual[: direct[2]]) == 0
    assert (
        fit.constrained_normalized_template_energy
        <= fit.model.maximum_residual_to_physical_energy_ratio
    )
    assert ablation["combined"]["total"] < ablation["physical_only"]["total"]
    assert ablation["combined"]["total"] < ablation["residual_only"]["total"]
    json.dumps(fit.to_dict(), allow_nan=False)
