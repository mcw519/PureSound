#!/usr/bin/env python
"""Render an opt-in M4 receiver-array, Ambisonic, and optional binaural RIR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.render.binaural import analytic_first_order_binaural_decoder
from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene
from puresound.audio.rir.scene.schema import RoomSceneV2
from puresound.audio.rir.render.spatial import render_room_scene_spatial_rir


DEFAULT_OUTPUT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/spatial_demo"


def _sample_scene(sample_rate: int, duration_s: float, seed: int) -> RoomSceneV2:
    config = HybridRIRConfig(
        sample_rate=sample_rate,
        duration=duration_s,
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(0, 0),
    )
    return sample_material_first_rir_scene(
        config,
        seed=seed,
        room_type="office",
        scene_id=f"m4-spatial-demo-{seed}",
    )


def _expand_single_receiver(
    scene: RoomSceneV2,
    spacing_m: float,
) -> RoomSceneV2:
    if len(scene.receivers) != 1 or spacing_m <= 0.0:
        return scene
    original = scene.receivers[0]
    center = np.asarray(original.pose.position_m, dtype=np.float64)
    half_spacing = 0.5 * spacing_m
    left = replace(
        original,
        transducer_id=f"{original.transducer_id}:left",
        array_id=original.array_id or "m4-binaural",
        channel_index=0,
        pose=replace(
            original.pose,
            position_m=(center + [-half_spacing, 0.0, 0.0]).tolist(),
        ),
    )
    right = replace(
        original,
        transducer_id=f"{original.transducer_id}:right",
        array_id=original.array_id or "m4-binaural",
        channel_index=1,
        pose=replace(
            original.pose,
            position_m=(center + [half_spacing, 0.0, 0.0]).tolist(),
        ),
    )
    return replace(scene, receivers=[left, right])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-json",
        type=Path,
        help="RoomSceneV2 JSON; omit to sample a deterministic office",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--delay-lines", type=int, default=16)
    parser.add_argument("--plane-waves", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--binaural-spacing-m",
        type=float,
        default=0.17,
        help="expand a one-receiver scene into an x-axis pair; <=0 disables",
    )
    parser.add_argument(
        "--binaural-decoder",
        choices=("none", "analytic"),
        default="analytic",
        help="analytic is a pipeline demo, not a measured HRTF",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        if args.scene_json is None:
            scene = _sample_scene(
                args.sample_rate,
                args.duration,
                args.seed,
            )
        else:
            scene = RoomSceneV2.from_json(
                args.scene_json.read_text(encoding="utf-8")
            )
        scene = _expand_single_receiver(scene, args.binaural_spacing_m)
        decoder = (
            analytic_first_order_binaural_decoder(args.sample_rate)
            if args.binaural_decoder == "analytic"
            else None
        )
        rendered = render_room_scene_spatial_rir(
            scene,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            source_index=args.source_index,
            max_order=args.max_order,
            delay_line_count=args.delay_lines,
            plane_wave_count=args.plane_waves,
            seed=args.seed,
            decoder=decoder,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / "scene.json"
    metadata_path = args.output_dir / "metadata.json"
    receiver_path = args.output_dir / "receiver_rir.wav"
    ambisonic_path = args.output_dir / "ambisonic_acn_sn3d_rir.wav"
    scene_path.write_text(scene.to_json(indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(rendered.metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    sf.write(
        receiver_path,
        rendered.receiver_rirs.T,
        args.sample_rate,
        subtype="FLOAT",
    )
    sf.write(
        ambisonic_path,
        rendered.ambisonic_acn_sn3d.T,
        args.sample_rate,
        subtype="FLOAT",
    )
    print(f"# wrote {scene_path}")
    print(f"# wrote {metadata_path}")
    print(f"# wrote {receiver_path}")
    print(f"# wrote {ambisonic_path}")
    if rendered.binaural is not None:
        binaural_path = args.output_dir / "binaural_brir.wav"
        sf.write(
            binaural_path,
            rendered.binaural.brir.T,
            args.sample_rate,
            subtype="FLOAT",
        )
        print(f"# wrote {binaural_path}")
        print(
            "# warning: analytic binaural decoder is not a measured HRTF; "
            "inject licensed HRTF-derived FIRs for production"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
