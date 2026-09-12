#!/usr/bin/env python
"""Validate that the M4.5 FOA late field stays isotropic through early/late coupling.

``couple_receiver_array_early_late`` renormalizes the diffuse tail so its energy
matches the coherent path tail.  For the FOA output that renormalization runs
over W/Y/Z/X, and the question this validator answers is whether it disturbs the
directional balance of the shared plane-wave field.

It must not.  The four channels are projections of one plane-wave set, so an
isotropic field satisfies ``E[Y²] = E[Z²] = E[X²] = E[W²]/3`` (SN3D first
order, −4.77 dB).  A gain solved independently per channel would replace that
with the *coherent* tail's directional energy, and would drive the diffuse
component of a channel whose coherent tail vanishes towards zero.  The
implementation avoids this by solving a single shared gain over the
concatenated channels — ``energy_policy`` names it
``one_shared_array_gain_preserves_spatial_ratios``.

Written to settle review finding A11, which
claimed the per-channel behaviour.  It is kept as a validator rather than a
one-off because the property is worth re-checking whenever the coupling policy
changes: the counterfactual it reports shows a per-channel solve would attenuate
Z by 12–31 dB.

Measurements are taken *after* ``transition_end_sample``, where the early weight
is exactly zero and the output is the diffuse field alone.

Exit code 0 if every scene stays within the isotropy tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.metrics import octave_band_rir
from puresound.audio.rir.render.coupling import energy_preserving_diffuse_gain
from puresound.audio.rir.render.spatial import render_room_scene_spatial_rir
from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene
from puresound.audio.rir.scene.schema import RoomSceneV2


DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT
    / "egs/rir_generation/phases/m4_spatial_late_field/reports"
    / "m4_foa_diffuse_isotropy_report.json"
)

#: ACN channel order with the SN3D basis ``[1, y, z, x]``.
CHANNEL_LABELS = ("W", "Y", "Z", "X")

#: ``E[Y²] = E[W²]/3`` for an isotropic field under SN3D first-order weighting.
ISOTROPIC_REL_W_DB = 10.0 * math.log10(1.0 / 3.0)

#: Broadband tolerance.  The residual is finite-quadrature and finite-tail
#: statistics, not a systematic tilt; 128 directions leave under a dB.
BROADBAND_TOLERANCE_DB = 1.5

#: Per-octave tolerance.  Narrower bands hold fewer independent samples, so the
#: same statistics spread further.
OCTAVE_TOLERANCE_DB = 3.0

#: The array and the FOA are coupled by two independent solves.  They describe
#: one field, so their gains must agree closely.
GAIN_AGREEMENT_TOLERANCE_DB = 1.0

OCTAVE_CENTERS_HZ = (500.0, 1000.0, 2000.0, 4000.0)


def _rel_db(value: float, reference: float) -> float:
    if reference <= 0.0 or value <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(value / reference)


def _channel_energies(block: np.ndarray) -> list[float]:
    return [float(np.dot(row, row)) for row in np.asarray(block, dtype=np.float64)]


def _sample_scene(
    *,
    sample_rate: int,
    duration_s: float,
    seed: int,
    room_type: str,
) -> RoomSceneV2:
    config = HybridRIRConfig(
        sample_rate=int(sample_rate),
        duration=float(duration_s),
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(0, 0),
    )
    return sample_material_first_rir_scene(
        config,
        seed=int(seed),
        room_type=room_type,
        scene_id=f"m4-foa-isotropy-{room_type}-{seed}",
    )


def _with_receiver_pair(scene: RoomSceneV2, spacing_m: float) -> RoomSceneV2:
    """Expand the sampled single receiver into an x-axis pair."""

    original = scene.receivers[0]
    center = np.asarray(original.pose.position_m, dtype=np.float64)
    half_spacing = 0.5 * float(spacing_m)
    receivers = [
        replace(
            original,
            transducer_id=f"{original.transducer_id}:{name}",
            array_id=original.array_id or "foa-isotropy-pair",
            channel_index=index,
            pose=replace(
                original.pose,
                position_m=(center + offset).tolist(),
            ),
        )
        for index, (name, offset) in enumerate(
            (
                ("left", np.array([-half_spacing, 0.0, 0.0])),
                ("right", np.array([half_spacing, 0.0, 0.0])),
            )
        )
    ]
    return replace(scene, receivers=receivers)


def _vertically_symmetric(scene: RoomSceneV2) -> RoomSceneV2:
    """Place transducers at mid-height with floor and ceiling sharing a material.

    Image sources then pair up symmetrically about the horizontal plane and the
    coherent FOA Z channel very nearly cancels — the condition under which a
    per-channel energy solve would zero the Z diffuse component.  This is the
    adversarial case, not a realistic scene.
    """

    mid_height = 0.5 * float(scene.dimensions_m[2])
    floor_material = next(
        surface.material_id
        for surface in scene.surfaces
        if surface.boundary == "floor"
    )
    surfaces = [
        replace(surface, material_id=floor_material)
        if surface.boundary in {"floor", "ceiling"}
        else surface
        for surface in scene.surfaces
    ]
    at_mid_height = lambda transducer: replace(  # noqa: E731
        transducer,
        pose=replace(
            transducer.pose,
            position_m=[*transducer.pose.position_m[:2], mid_height],
        ),
    )
    return replace(
        scene,
        surfaces=surfaces,
        sources=[at_mid_height(source) for source in scene.sources],
        receivers=[at_mid_height(receiver) for receiver in scene.receivers],
    )


def _counterfactual_per_channel_gains(coupling, coherent, late) -> list[float]:
    """Gains a per-channel solve would have produced, for comparison only."""

    gains: list[float] = []
    for channel, coherent_row, late_row in zip(
        coupling.metadata["channels"],
        np.asarray(coherent, dtype=np.float64),
        np.asarray(late, dtype=np.float64),
    ):
        index = int(channel["receiver_index"])
        try:
            gain, _aggregate = energy_preserving_diffuse_gain(
                coherent_row,
                coherent_row * coupling.early_weight[index],
                late_row * coupling.late_weight[index],
                int(channel["transition_start_sample"]),
                target_energy=float(channel["target_post_transition_energy"]),
            )
        except ValueError:
            gain = float("nan")
        gains.append(float(gain))
    return gains


def measure_scene(
    name: str,
    scene: RoomSceneV2,
    *,
    sample_rate: int,
    duration_s: float,
    seed: int,
    plane_wave_count: int,
    max_order: int,
) -> dict[str, Any]:
    rendered = render_room_scene_spatial_rir(
        scene,
        sample_rate=int(sample_rate),
        duration_s=float(duration_s),
        max_order=int(max_order),
        plane_wave_count=int(plane_wave_count),
        seed=int(seed),
    )
    ambisonic = rendered.ambisonic_coupling
    receivers = rendered.receiver_coupling
    transition_end = int(ambisonic.metadata["channels"][0]["transition_end_sample"])
    early_after_end = float(
        np.max(np.abs(np.asarray(ambisonic.early_weight)[:, transition_end:]))
    )
    pure_diffuse = np.asarray(ambisonic.rir, dtype=np.float64)[:, transition_end:]

    broadband = _channel_energies(pure_diffuse)
    broadband_rel_w = [_rel_db(value, broadband[0]) for value in broadband]
    broadband_deviation = max(
        abs(value - ISOTROPIC_REL_W_DB) for value in broadband_rel_w[1:]
    )

    octaves: dict[str, list[float]] = {}
    octave_deviation = 0.0
    for center_hz in OCTAVE_CENTERS_HZ:
        filtered = np.vstack(
            [octave_band_rir(row, int(sample_rate), center_hz) for row in pure_diffuse]
        )
        energies = _channel_energies(filtered)
        relative = [_rel_db(value, energies[0]) for value in energies]
        octaves[f"{center_hz:g}"] = relative
        octave_deviation = max(
            octave_deviation,
            max(abs(value - ISOTROPIC_REL_W_DB) for value in relative[1:]),
        )

    # Pseudo-intensity diffuseness: 1 for an ideal isotropic field, 0 for a
    # single plane wave.
    w, y, z, x = pure_diffuse
    intensity = np.array(
        [float(np.dot(w, x)), float(np.dot(w, y)), float(np.dot(w, z))]
    )
    total_energy = 0.5 * sum(float(np.dot(row, row)) for row in pure_diffuse)
    diffuseness = 1.0 - math.sqrt(2.0) * float(np.linalg.norm(intensity)) / total_energy

    ambisonic_gain = float(ambisonic.metadata["shared_diffuse_gain"])
    receiver_gain = float(receivers.metadata["shared_diffuse_gain"])
    gain_agreement_db = _rel_db(ambisonic_gain**2, receiver_gain**2)

    coherent = np.asarray(rendered.coherent_ambisonic_acn_sn3d, dtype=np.float64)
    transition_start = int(ambisonic.metadata["channels"][0]["transition_start_sample"])
    coherent_tail = [
        float(np.dot(row[transition_start:], row[transition_start:]))
        for row in coherent
    ]
    diffuse_energy = _channel_energies(
        np.asarray(ambisonic.diffuse_component)[:, transition_start:]
    )

    checks = {
        "single_shared_gain": (
            ambisonic.metadata["energy_policy"]
            == "one_shared_array_gain_preserves_spatial_ratios"
        ),
        "early_weight_zero_after_transition": early_after_end < 1e-12,
        "broadband_isotropy": broadband_deviation <= BROADBAND_TOLERANCE_DB,
        "octave_isotropy": octave_deviation <= OCTAVE_TOLERANCE_DB,
        "no_channel_zeroed": all(value > 0.0 for value in diffuse_energy),
        "array_and_foa_gains_agree": (
            abs(gain_agreement_db) <= GAIN_AGREEMENT_TOLERANCE_DB
        ),
    }

    return {
        "scene": name,
        "scene_id": scene.scene_id,
        "dimensions_m": [round(float(value), 3) for value in scene.dimensions_m],
        "transition_end_sample": transition_end,
        "pure_diffuse_samples": int(pure_diffuse.shape[1]),
        "max_early_weight_after_transition": early_after_end,
        "energy_policy": ambisonic.metadata["energy_policy"],
        "shared_diffuse_gain_ambisonic": ambisonic_gain,
        "shared_diffuse_gain_receivers": receiver_gain,
        "gain_agreement_db": gain_agreement_db,
        "coherent_tail_rel_w_db": [
            _rel_db(value, coherent_tail[0]) for value in coherent_tail
        ],
        "pure_diffuse_rel_w_db": broadband_rel_w,
        "broadband_max_deviation_db": broadband_deviation,
        "octave_rel_w_db": octaves,
        "octave_max_deviation_db": octave_deviation,
        "diffuseness": diffuseness,
        "counterfactual_per_channel_gains": _counterfactual_per_channel_gains(
            ambisonic, coherent, rendered.spatial_late_field.ambisonic_acn_sn3d
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _print_scene(result: dict[str, Any]) -> None:
    print(f"\n=== {result['scene']} — room {result['dimensions_m']} ===")
    print(
        f"  policy {result['energy_policy']}\n"
        f"  FOA gain {result['shared_diffuse_gain_ambisonic']:.6f}   "
        f"array gain {result['shared_diffuse_gain_receivers']:.6f}   "
        f"Δ {result['gain_agreement_db']:+.3f} dB"
    )
    print(
        f"  pure diffuse region: {result['pure_diffuse_samples']} samples, "
        f"early weight ≤ {result['max_early_weight_after_transition']:.2g}"
    )
    print(f"\n   band      |{''.join(f'{label:>9}' for label in CHANNEL_LABELS[1:])}"
          f" | max dev")
    print(
        "   broadband |"
        + "".join(f"{value:>+9.2f}" for value in result["pure_diffuse_rel_w_db"][1:])
        + f" | {result['broadband_max_deviation_db']:.2f} dB"
    )
    for center, relative in result["octave_rel_w_db"].items():
        deviation = max(abs(v - ISOTROPIC_REL_W_DB) for v in relative[1:])
        print(
            f"   {center:>6} Hz |"
            + "".join(f"{value:>+9.2f}" for value in relative[1:])
            + f" | {deviation:.2f} dB"
        )
    print(
        f"   (isotropic reference {ISOTROPIC_REL_W_DB:+.2f} dB, "
        f"diffuseness {result['diffuseness']:.4f})"
    )
    print(
        "\n  coherent tail rel W: "
        + "  ".join(
            f"{label} {value:+.2f}"
            for label, value in zip(
                CHANNEL_LABELS, result["coherent_tail_rel_w_db"]
            )
        )
    )
    shared = result["shared_diffuse_gain_ambisonic"]
    print(
        "  counterfactual per-channel gains: "
        + "  ".join(
            f"{label} {gain:.4g} ({_rel_db(gain**2, shared**2):+.2f} dB)"
            for label, gain in zip(
                CHANNEL_LABELS, result["counterfactual_per_channel_gains"]
            )
        )
    )
    for check, ok in result["checks"].items():
        if not ok:
            print(f"  FAIL {check}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--plane-waves", type=int, default=128)
    parser.add_argument("--receiver-spacing-m", type=float, default=0.17)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args()

    cases = [
        ("office", 20260731, False),
        ("office", 20260731, True),
        ("meeting_room", 4242, False),
        ("living_room", 77, False),
    ]
    results = []
    for room_type, seed, symmetric in cases:
        scene = _sample_scene(
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            seed=seed,
            room_type=room_type,
        )
        if symmetric:
            scene = _vertically_symmetric(scene)
        scene = _with_receiver_pair(scene, args.receiver_spacing_m)
        name = (
            f"{room_type} seed {seed}"
            f"{' — vertically symmetric (adversarial)' if symmetric else ''}"
        )
        result = measure_scene(
            name,
            scene,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            seed=seed,
            plane_wave_count=args.plane_waves,
            max_order=args.max_order,
        )
        _print_scene(result)
        results.append(result)

    passed = all(result["passed"] for result in results)
    report = {
        "validator": "m4_foa_diffuse_isotropy",
        "question": (
            "does early/late coupling disturb the directional balance of the "
            "shared plane-wave FOA late field?"
        ),
        "isotropic_reference_rel_w_db": ISOTROPIC_REL_W_DB,
        "tolerances_db": {
            "broadband": BROADBAND_TOLERANCE_DB,
            "octave": OCTAVE_TOLERANCE_DB,
            "array_foa_gain_agreement": GAIN_AGREEMENT_TOLERANCE_DB,
        },
        "settings": {
            "sample_rate": int(args.sample_rate),
            "duration_s": float(args.duration),
            "max_order": int(args.max_order),
            "plane_wave_count": int(args.plane_waves),
            "receiver_spacing_m": float(args.receiver_spacing_m),
        },
        "scenes": results,
        "all_passed": passed,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n{'PASS' if passed else 'FAIL'} — report written to {args.output_report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
