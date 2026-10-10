#!/usr/bin/env python
"""Build a level-consistent dry/M3/M4/spatial-BRIR listening comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DRY = (
    REPO_ROOT
    / "egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1/dry/dry_00_1272-141231-0008.wav"
)
DEFAULT_M3_RIR = (
    REPO_ROOT
    / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_spatial_coherent_receiver_2ch.wav"
)
DEFAULT_M4_RIR = (
    REPO_ROOT
    / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_spatial_receiver_rir_2ch.wav"
)
DEFAULT_BRIR = (
    REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav"
)
DEFAULT_SPATIAL_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/rir_m4_listening_comparison"


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"audio must contain finite samples: {path}")
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1, dtype=np.float64)


def _convolve_mono_with_rir(dry: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [fftconvolve(dry, rir[:, channel], mode="full") for channel in range(2)]
    )


def _pad_stereo(signal: np.ndarray, sample_count: int) -> np.ndarray:
    stereo = np.column_stack((signal, signal))
    if stereo.shape[0] >= sample_count:
        return stereo[:sample_count]
    return np.pad(stereo, ((0, sample_count - stereo.shape[0]), (0, 0)))


def _signal_summary(audio: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(audio.shape[0]),
        "channel_count": int(audio.shape[1]),
        "peak_abs": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "energy": float(np.sum(np.square(audio))),
    }


def build_comparison(
    dry: np.ndarray,
    m3_rir: np.ndarray,
    m4_rir: np.ndarray,
    analytic_brir: np.ndarray,
    sample_rate: int,
    spatial_report: dict[str, Any],
    *,
    target_peak_dbfs: float = -1.0,
    silence_s: float = 0.75,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convolve one mono source and apply one common non-clipping master gain."""

    if dry.ndim != 1 or dry.size < 1:
        raise ValueError("dry source must be a non-empty mono signal")
    rirs = {
        "m3_coherent_array": np.asarray(m3_rir, dtype=np.float64),
        "m4_spatial_array": np.asarray(m4_rir, dtype=np.float64),
        "m4_analytic_brir": np.asarray(analytic_brir, dtype=np.float64),
    }
    if any(rir.ndim != 2 or rir.shape[1] != 2 for rir in rirs.values()):
        raise ValueError("every comparison RIR must have shape [sample, 2]")
    if any(not np.all(np.isfinite(rir)) for rir in rirs.values()):
        raise ValueError("comparison RIRs must be finite")

    wet = {
        name: _convolve_mono_with_rir(dry, rir)
        for name, rir in rirs.items()
    }
    output_length = max(audio.shape[0] for audio in wet.values())
    dry_stereo = _pad_stereo(dry, output_length)
    unmastered = {"dry_reference": dry_stereo, **wet}
    maximum_peak = max(
        float(np.max(np.abs(audio))) for audio in unmastered.values()
    )
    target_peak = float(10.0 ** (float(target_peak_dbfs) / 20.0))
    master_gain = (
        min(1.0, target_peak / maximum_peak)
        if maximum_peak > np.finfo(np.float64).tiny
        else 1.0
    )
    mastered = {
        name: np.asarray(master_gain * audio, dtype=np.float64)
        for name, audio in unmastered.items()
    }
    mastered["m4_minus_m3_difference"] = (
        mastered["m4_spatial_array"] - mastered["m3_coherent_array"]
    )
    silence = np.zeros(
        (max(1, int(round(float(silence_s) * sample_rate))), 2),
        dtype=np.float64,
    )
    mastered["ab_m3_then_m4"] = np.vstack(
        (
            mastered["m3_coherent_array"],
            silence,
            mastered["m4_spatial_array"],
        )
    )

    transition_starts = [
        int(channel["transition_start_sample"])
        for channel in spatial_report["renderer"]["receiver_coupling"][
            "channels"
        ]
    ]
    early_errors = [
        float(
            np.max(
                np.abs(
                    rirs["m4_spatial_array"][: start + 1, channel]
                    - rirs["m3_coherent_array"][: start + 1, channel]
                )
            )
        )
        for channel, start in enumerate(transition_starts)
    ]
    manifest = {
        "schema_version": "puresound.m4_listening_comparison.v1",
        "sample_rate": int(sample_rate),
        "comparison": {
            "A": "M3 coherent PathEvent receiver array",
            "B": "M4 shared spatial-FDN receiver array",
            "C": "M4 FOA decoded by analytic headless non-HRTF decoder",
            "controlled_variables": (
                "same dry source, scene, source, receiver array, direct/early paths, "
                "sample rate, and one shared master gain"
            ),
        },
        "mastering": {
            "policy": "one_common_linear_gain_for_dry_and_all_wet_renders",
            "target_peak_dbfs": float(target_peak_dbfs),
            "unmastered_maximum_peak": float(maximum_peak),
            "common_master_gain": float(master_gain),
            "per_file_loudness_normalization": False,
        },
        "rir_early_identity": {
            "transition_start_samples": transition_starts,
            "maximum_absolute_error_by_channel": early_errors,
            "exact": bool(max(early_errors) == 0.0),
        },
        "outputs": {
            name: _signal_summary(audio) for name, audio in mastered.items()
        },
        "listening_notes": [
            "Compare m3_coherent_array with m4_spatial_array for late-tail density and width.",
            "Use m4_minus_m3_difference only to identify what changed; it is not a natural render.",
            "The analytic BRIR is a decoder demonstration, not a measured HRTF reference.",
        ],
    }
    return mastered, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", type=Path, default=DEFAULT_DRY)
    parser.add_argument("--m3-rir", type=Path, default=DEFAULT_M3_RIR)
    parser.add_argument("--m4-rir", type=Path, default=DEFAULT_M4_RIR)
    parser.add_argument("--analytic-brir", type=Path, default=DEFAULT_BRIR)
    parser.add_argument(
        "--spatial-report",
        type=Path,
        default=DEFAULT_SPATIAL_REPORT,
    )
    parser.add_argument("--target-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--silence-s", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        dry_audio, dry_rate = _read_audio(args.dry)
        m3_rir, m3_rate = _read_audio(args.m3_rir)
        m4_rir, m4_rate = _read_audio(args.m4_rir)
        analytic_brir, brir_rate = _read_audio(args.analytic_brir)
        if len({dry_rate, m3_rate, m4_rate, brir_rate}) != 1:
            raise ValueError("dry source and every RIR must share one sample rate")
        spatial_report = json.loads(
            args.spatial_report.read_text(encoding="utf-8")
        )
        outputs, manifest = build_comparison(
            _mono(dry_audio),
            m3_rir,
            m4_rir,
            analytic_brir,
            dry_rate,
            spatial_report,
            target_peak_dbfs=args.target_peak_dbfs,
            silence_s=args.silence_s,
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered_names = (
        "dry_reference",
        "m3_coherent_array",
        "m4_spatial_array",
        "m4_analytic_brir",
        "m4_minus_m3_difference",
        "ab_m3_then_m4",
    )
    for index, name in enumerate(ordered_names):
        path = args.output_dir / f"{index:02d}_{name}.wav"
        sf.write(path, outputs[name], dry_rate, subtype="FLOAT")
        manifest["outputs"][name]["path"] = str(path)
        print(f"# wrote {path}")
    manifest["inputs"] = {
        "dry": str(args.dry),
        "m3_rir": str(args.m3_rir),
        "m4_rir": str(args.m4_rir),
        "analytic_brir": str(args.analytic_brir),
        "spatial_report": str(args.spatial_report),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"# wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
