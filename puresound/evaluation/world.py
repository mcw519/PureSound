"""Target-policy comparisons for reproducible dynamic acoustic scenes."""

import numpy as np

from puresound.evaluation.reference import reference_metrics


def si_sdr(target, candidate):
    target = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    target = target - target.mean()
    candidate = candidate - candidate.mean()
    energy = float(target @ target)
    if energy < 1e-10:
        return None
    projected = target * float(candidate @ target) / energy
    residual = candidate - projected
    return float(
        10
        * np.log10(
            max(float(projected @ projected), 1e-12)
            / max(float(residual @ residual), 1e-12)
        )
    )


PURPOSES = {
    "target": "target talkers",
    "speech": "all speech",
    "near": "near-region speech",
}


def world_metrics(rendered, enhanced):
    """Scores of ``enhanced`` against each reference, whole-file and per 1 s
    window (0.5 s hop).  A silent reference reports the output's level."""
    sr = rendered.metadata["sample_rate"]
    result = {}
    for policy, reference in rendered.references.items():
        windows = []
        for start in range(0, len(reference), sr // 2):
            end = min(start + sr, len(reference))
            if end - start < sr // 4:
                break
            target, raw, out = (
                reference[start:end],
                rendered.mixture[start:end],
                enhanced[start:end],
            )
            base, score = si_sdr(target, raw), si_sdr(target, out)
            windows.append(
                {
                    "start_s": start / sr,
                    "end_s": end / sr,
                    "input_si_sdr_db": base,
                    "output_si_sdr_db": score,
                    "improvement_db": None if score is None else score - base,
                    "residual_dbfs": float(
                        10
                        * np.log10(max(float(np.mean(out.astype(float) ** 2)), 1e-12))
                    )
                    if score is None
                    else None,
                }
            )
        active = np.mean(reference.astype(float) ** 2) > 1e-12
        result[policy] = {
            "windows": windows,
            "input": reference_metrics(reference, rendered.mixture, sr)
            if active
            else None,
            "output": reference_metrics(reference, enhanced, sr) if active else None,
            "purpose": PURPOSES[policy],
        }
    return result
