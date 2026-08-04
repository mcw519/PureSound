import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

# Cap native math-library threads BEFORE numpy/scipy/torch are imported so each
# generation worker stays single-threaded. Otherwise every worker process spawns
# a BLAS/OMP thread pool sized to the full core count, and running many workers
# oversubscribes the CPU and slows the whole batch down. These read their values
# at import time, so setting them later (e.g. in _init_worker) would be too late.
# Users can still override by exporting the variables before launching.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.high_frequency import (
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
)
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
from puresound.audio.rir.render.low_frequency import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    ImpedanceModalLowFrequencyBackend,
)
from puresound.audio.rir.scene.geometry import (
    distance_point_to_polygon as _distance_point_to_polygon,
    max_room_horizontal_distance_from_point as _max_room_horizontal_distance_from_point,
)
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    min_feasible_rt60 as _min_feasible_rt60,
    sample_point as _sample_point,
    sample_polygon_obstacles,
    sample_source_in_horizontal_shell as _sample_source_in_horizontal_shell,
    upgrade_hybrid_scene_to_v2,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir.physics.impedance.residues import (
    ImpedanceModalResidueCalibration,
)
from puresound.audio.rir.scene.materials import ROOM_TYPE_RECIPES
from puresound.audio.rir.bank.schema import (
    BankGeneratorProvenance,
    BankRendererProfile,
    BankSplitPolicy,
    RIRBankItem,
    RIRBankManifest,
    audit_rir_bank_manifest,
    canonical_json_sha256,
    canonicalize_float_wav_header,
    sha256_file,
    task_plan_rows,
    write_split_indexes,
)
from puresound.audio.rir.scene.schema import SCENE_SCHEMA_VERSION


_WORKER_CONTEXT = {}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate 5-channel hybrid RIR WAVs with near/far sources."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-rooms", "--num-rooms", dest="n_rooms", type=int, default=10)
    parser.add_argument("--rir-per-room", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel generation workers. Defaults to one worker per --gpu-devices entry.",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated CUDA devices assigned to workers, for example '0,1'.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip RIRs already present in --output-dir and generate only the "
        "rest. Scene sampling is deterministic, so resuming with the same "
        "--seed/--n-rooms/--rir-per-room/room settings reproduces the exact same "
        "room/mic/source geometry an uninterrupted run would have used (the "
        "high-band acoustic realization is task-seeded when M6 emission is on).",
    )
    parser.add_argument(
        "--emit-m6-manifest",
        action="store_true",
        help="Emit and strictly audit a draft puresound.rir_bank.v2 manifest, "
        "split indexes, and generation audit after all items complete.",
    )
    parser.add_argument(
        "--m6-bank-id",
        help="Stable bank identity. Defaults to a path-independent config hash.",
    )
    parser.add_argument(
        "--m6-code-revision",
        help="Pinned code revision. Defaults to git HEAD with a -dirty suffix "
        "when the working tree has changes.",
    )
    parser.add_argument("--m6-split-seed", type=int, default=20260803)
    parser.add_argument("--m6-train-fraction", type=float, default=0.8)
    parser.add_argument("--m6-validation-fraction", type=float, default=0.1)
    parser.add_argument("--m6-test-fraction", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--crossover-hz", type=float, default=1000.0)
    parser.add_argument("--low-fmin-hz", type=float, default=20.0)
    parser.add_argument("--low-fmax-hz", type=float, default=1000.0)
    parser.add_argument(
        "--scene-version",
        choices=["v0", "v1"],
        default="v0",
        help="v0 uses requested broadband RT60; v1 samples frequency-dependent "
        "surface materials and derives decay from them.",
    )
    parser.add_argument(
        "--room-type",
        choices=["mixed", *sorted(ROOM_TYPE_RECIPES)],
        default="mixed",
        help="Correlated v1 material recipe. Ignored by --scene-version v0.",
    )
    parser.add_argument(
        "--material-variation-scale",
        type=float,
        default=1.0,
        help="Scale of correlated v1 absorption/scattering uncertainty.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["peak_normalized", "calibrated"],
        default=None,
        help="Defaults to peak_normalized for v0 and calibrated for v1.",
    )
    parser.add_argument(
        "--record-realized-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store generated broadband/octave acoustic metrics. Defaults on for v1.",
    )
    parser.add_argument(
        "--low-backend",
        choices=[
            "pytard",
            "pytard-material",
            "pytard-cupy",
            "pytard-cupy-material",
            "analytic",
            "analytic-material",
            "analytic-impedance",
        ],
        default="pytard",
        help="The '-material' variants enable M2 per-mode boundary damping and "
        "require --scene-version v1. 'analytic-impedance' requires an explicit "
        "--impedance-boundary-config and never infers phase from scene absorption.",
    )
    parser.add_argument(
        "--impedance-boundary-config",
        type=Path,
        help="Explicit six-wall passive rational boundary JSON for the "
        "experimental analytic-impedance backend.",
    )
    parser.add_argument(
        "--impedance-residue-calibration",
        type=Path,
        help="Optional fixed-pole FDTD residue calibration JSON for the "
        "experimental analytic-impedance backend.",
    )
    parser.add_argument(
        "--impedance-mode-index-limit",
        type=int,
        default=5,
        help="Maximum rigid reference index on each axis for "
        "analytic-impedance mode enumeration.",
    )
    parser.add_argument(
        "--impedance-max-modes",
        type=int,
        default=128,
        help="Maximum candidate modes solved by analytic-impedance.",
    )
    parser.add_argument(
        "--impedance-continuation-steps",
        type=int,
        default=8,
        help="Boundary-strength continuation steps used by each nonlinear "
        "3D impedance eigenvalue solve.",
    )
    parser.add_argument(
        "--material-modal-loss-scale",
        type=float,
        default=1.0,
        help="Experimental multiplier on per-mode material loss. Only used by "
        "'*-material' low backends; 1.0 is the uncalibrated physical prior.",
    )
    parser.add_argument("--room-x", nargs=2, type=float, default=[3.5, 8.0])
    parser.add_argument("--room-y", nargs=2, type=float, default=[3.5, 7.0])
    parser.add_argument("--room-z", nargs=2, type=float, default=[2.4, 3.6])
    parser.add_argument("--rt60", nargs=2, type=float, default=[0.25, 0.8])
    parser.add_argument("--obstacles", nargs=2, type=int, default=[1, 6])
    parser.add_argument(
        "--near-dist",
        nargs=2,
        type=float,
        default=[0.35, 0.95],
        help="Mic-to-source distance range (m) for the near_* channels.",
    )
    parser.add_argument(
        "--far-dist",
        nargs=2,
        type=float,
        default=[2.05, 5.5],
        help="Mic-to-source distance range (m) for the far_* channels. The "
        "default leaves 0.95-2.05 m unoccupied; pass e.g. '--far-dist 1.10 3.5' "
        "to build a boundary-coverage bank.",
    )
    parser.add_argument("--pra-max-order", type=int, default=12)
    parser.add_argument("--pra-n-rays", type=int, default=20000)
    parser.add_argument(
        "--high-backend",
        choices=("pyroomacoustics", "path-events-m3", "path-events-m4"),
        default="pyroomacoustics",
        help=(
            "Use the existing Pyroomacoustics backend, the opt-in M3 coherent "
            "PathEvent backend, or the opt-in M4 PathEvent-early/FDN-late backend."
        ),
    )
    parser.add_argument(
        "--fdn-mixing-time-ms",
        type=float,
        default=24.0,
        help="Direct-relative M4 late-field transition center in milliseconds.",
    )
    parser.add_argument(
        "--fdn-transition-ms",
        type=float,
        default=16.0,
        help="Equal-power PathEvent/FDN crossfade duration in milliseconds.",
    )
    parser.add_argument(
        "--fdn-delay-lines",
        type=int,
        default=16,
        help="Power-of-two delay-line count for --high-backend=path-events-m4.",
    )
    parser.add_argument(
        "--fdn-seed",
        type=int,
        default=20260731,
        help="Base deterministic M4 FDN seed; room/source IDs derive channel seeds.",
    )
    parser.add_argument(
        "--pytard-low-sample-rate",
        type=int,
        default=16000,
        help="Internal pytARD simulation rate. Output is resampled to --sample-rate.",
    )
    parser.add_argument(
        "--pytard-spatial-samples-per-wavelength",
        type=int,
        default=2,
        help="pytARD spatial resolution. Higher is more accurate but much slower.",
    )
    parser.add_argument(
        "--pytard-calibration-peak",
        type=float,
        default=0.05,
        help="Peak used when normalizing raw pytARD low-band output.",
    )
    parser.add_argument(
        "--pytard-rt60-decay-scale",
        type=float,
        default=1.0,
        help="Scale applied to scene RT60 when damping pytARD low-band tail.",
    )
    parser.add_argument(
        "--crossover-target-db",
        type=float,
        default=0.0,
        help="Target low-band RMS relative to high-band RMS near crossover.",
    )
    parser.add_argument(
        "--crossover-max-gain",
        type=float,
        default=8.0,
        help=(
            "Maximum gain allowed during low-band crossover matching. The "
            "measured requirement reaches 5.3x because the pytARD low band is "
            "peak-normalized, so this match sets its level; a ceiling below "
            "that silently ships an under-level low band."
        ),
    )
    parser.add_argument(
        "--preserve-crossover-source-convention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Do not apply per-channel low-band energy rescaling when the "
            "selected impedance calibration already shares the high band's "
            "1/r source convention. Disable only for legacy diagnostics."
        ),
    )
    return parser.parse_args(argv)


def _parse_gpu_devices(raw_devices):
    if raw_devices is None or raw_devices.strip() == "":
        return []
    return [device.strip() for device in raw_devices.split(",") if device.strip()]


def _warn(message):
    print(f"[generate_hybrid_rir] warning: {message}", file=sys.stderr, flush=True)


def _warn_gpu_configuration(args, gpu_devices):
    """Surface mismatches between the requested backend and GPU placement.

    Only the ``pytard-cupy*`` low backends run on the GPU; ``pytard*``/
    ``analytic*`` and the Pyroomacoustics high band are always CPU. Multiple
    cupy processes per device each hold a separate CUDA context and memory pool,
    so sharing a device risks out-of-memory failures.
    """
    uses_gpu = args.low_backend.startswith("pytard-cupy")
    if gpu_devices and not uses_gpu:
        _warn(
            f"--gpu-devices={args.gpu_devices} is set but --low-backend="
            f"{args.low_backend} runs entirely on CPU (only 'pytard-cupy*' uses "
            "the GPU; the high-frequency band is always CPU). No GPU work will "
            "happen; the device list only restricts each worker's CUDA visibility."
        )
    if not uses_gpu:
        return
    device_count = len(gpu_devices)
    if device_count == 0:
        _warn(
            f"--low-backend={args.low_backend} without --gpu-devices: all "
            f"{args.num_workers} worker(s) share the currently visible CUDA "
            "device(s). Each process holds its own CUDA context and memory pool, "
            "which can exhaust GPU memory. Pass --gpu-devices to pin one worker "
            "per device."
        )
    elif args.num_workers > device_count:
        extra = args.num_workers - device_count
        _warn(
            f"--num-workers={args.num_workers} exceeds the --gpu-devices count "
            f"({device_count}); {extra} extra worker(s) will share a GPU with "
            "another process. Each cupy process holds its own CUDA context and "
            "memory pool, which can exhaust GPU memory. Prefer one worker per "
            "device."
        )


def _optional_asset_sha256(path):
    return sha256_file(path) if path is not None else None


def _package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _renderer_runtime_versions(args):
    """Return the runtime identity that can change rendered WAV bytes."""

    versions = {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "soundfile": _package_version("soundfile"),
        "torch": _package_version("torch"),
        "torchaudio": _package_version("torchaudio"),
    }
    if args.high_backend == "pyroomacoustics":
        versions["pyroomacoustics"] = _package_version("pyroomacoustics")
    if args.low_backend.startswith("pytard-cupy"):
        versions["cupy"] = _package_version("cupy-cuda12x")
    return versions


def _detect_code_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _m6_renderer_config(args):
    return {
        "low_backend": args.low_backend,
        "high_backend": args.high_backend,
        "pra_max_order": int(args.pra_max_order),
        "pra_n_rays": int(args.pra_n_rays),
        "fdn_mixing_time_ms": float(args.fdn_mixing_time_ms),
        "fdn_transition_ms": float(args.fdn_transition_ms),
        "fdn_delay_lines": int(args.fdn_delay_lines),
        "fdn_seed": int(args.fdn_seed),
        "pytard_low_sample_rate": int(args.pytard_low_sample_rate),
        "pytard_spatial_samples_per_wavelength": int(
            args.pytard_spatial_samples_per_wavelength
        ),
        "pytard_calibration_peak": float(args.pytard_calibration_peak),
        "pytard_rt60_decay_scale": float(args.pytard_rt60_decay_scale),
        "material_modal_loss_scale": float(args.material_modal_loss_scale),
        "impedance_mode_index_limit": int(args.impedance_mode_index_limit),
        "impedance_max_modes": int(args.impedance_max_modes),
        "impedance_continuation_steps": int(args.impedance_continuation_steps),
        "impedance_boundary_sha256": _optional_asset_sha256(
            args.impedance_boundary_config
        ),
        "impedance_residue_sha256": _optional_asset_sha256(
            args.impedance_residue_calibration
        ),
        "runtime_versions": _renderer_runtime_versions(args),
    }


def _m6_scene_plan_config(config, args):
    values = asdict(config)
    scene_fields = (
        "room_dim_range",
        "rt60_range",
        "sound_speed",
        "mic_margin",
        "source_margin",
        "mic_height_range",
        "speech_source_height_range",
        "near_distance_range",
        "far_distance_range",
        "num_near_sources",
        "num_far_sources",
        "num_obstacles_range",
        "obstacle_density_per_m2",
        "max_obstacle_floor_coverage",
        "obstacle_clearance",
        "obstacle_obstacle_clearance",
        "obstacle_margin",
        "obstacle_height_range",
        "obstacle_radius_range",
    )
    return {
        "seed": int(args.seed),
        "rir_per_room": int(args.rir_per_room),
        "scene_version": args.scene_version,
        "room_type": args.room_type,
        "material_variation_scale": float(args.material_variation_scale),
        "scene_sampling": {field: values[field] for field in scene_fields},
    }


def _m6_generation_context(config, args):
    renderer_config = _m6_renderer_config(args)
    scene_plan_config = _m6_scene_plan_config(config, args)
    content_config = {
        "hybrid_rir_config": asdict(config),
        "scene_plan": scene_plan_config,
        "renderer": renderer_config,
        "n_rooms": int(args.n_rooms),
        "rir_per_room": int(args.rir_per_room),
    }
    config_sha256 = canonical_json_sha256(content_config)
    renderer_sha256 = canonical_json_sha256(renderer_config)
    scene_plan_sha256 = canonical_json_sha256(scene_plan_config)
    policy = BankSplitPolicy(
        seed=int(args.m6_split_seed),
        train_fraction=float(args.m6_train_fraction),
        validation_fraction=float(args.m6_validation_fraction),
        test_fraction=float(args.m6_test_fraction),
    )
    bank_id = args.m6_bank_id or f"puresound-rir-{config_sha256[:16]}"
    code_revision = args.m6_code_revision or _detect_code_revision()
    return {
        "bank_id": bank_id,
        "code_revision": code_revision,
        "config_sha256": config_sha256,
        "renderer_config_sha256": renderer_sha256,
        "renderer_profile_id": f"renderer-{renderer_sha256[:16]}",
        "scene_plan_sha256": scene_plan_sha256,
        "split_policy": policy,
        "scene_schema_version": (
            SCENE_SCHEMA_VERSION
            if args.scene_version == "v1"
            else "puresound.hybrid_rir_scene.v0"
        ),
    }


def _m6_task_seed(base_seed, sample_id):
    payload = f"puresound.m6.task.v1\0{int(base_seed)}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _m6_acoustic_space_id(scene):
    metadata = scene.to_metadata()
    if metadata.get("schema_version") == SCENE_SCHEMA_VERSION:
        room_payload = {
            key: metadata[key]
            for key in (
                "schema_version",
                "room_type",
                "dimensions_m",
                "surfaces",
                "materials",
                "environment",
                "objects",
                "catalog_version",
            )
        }
    else:
        room_payload = {
            key: metadata[key] for key in ("room_dim", "rt60", "obstacles")
        }
    return f"synthetic:{canonical_json_sha256(room_payload)}"


def _attach_m6_task_contract(tasks, config, args):
    context = _m6_generation_context(config, args)
    policy = context["split_policy"]
    acoustic_space_by_room = {}
    for task in tasks:
        acoustic_space_id = _m6_acoustic_space_id(task["scene"])
        previous = acoustic_space_by_room.setdefault(
            task["room_id"], acoustic_space_id
        )
        if previous != acoustic_space_id:
            raise RuntimeError("one room_id produced multiple acoustic-space identities")
        task_seed = _m6_task_seed(args.seed, task["sample_id"])
        task["m6"] = {
            "bank_id": context["bank_id"],
            "code_revision": context["code_revision"],
            "acoustic_space_id": acoustic_space_id,
            "split": policy.assign(acoustic_space_id),
            "scene_sha256": canonical_json_sha256(task["scene"].to_metadata()),
            "generation_config_sha256": context["config_sha256"],
            "renderer_profile_id": context["renderer_profile_id"],
            "generation_seed": task_seed,
            "sample_rate": int(config.sample_rate),
            "channel_count": int(config.num_sources),
            "frame_count": int(config.num_samples),
        }
    return context


def _point_hits_obstacle(point, obstacles, clearance):
    point_xy = np.asarray(point[:2], dtype=np.float64)
    for obstacle in obstacles:
        footprint = np.asarray(obstacle.footprint, dtype=np.float64)
        if obstacle.contains_xy(point) or (
            _distance_point_to_polygon(point_xy, footprint) < float(clearance)
        ):
            return True
    return False


def _sample_clear_point(room_dim, margin, obstacles, clearance, rng, height_range=None):
    for _ in range(512):
        point = _sample_point(room_dim, margin, rng, height_range=height_range)
        if not _point_hits_obstacle(point, obstacles, clearance):
            return point
    raise RuntimeError("Failed to sample a point clear of room obstacles.")


def _sample_clear_source(room_dim, mic_pos, min_dist, max_dist, config, obstacles, rng):
    for _ in range(512):
        source = _sample_source_in_horizontal_shell(
            room_dim,
            mic_pos,
            min_dist,
            max_dist,
            config.source_margin,
            config.speech_source_height_range,
            rng,
        )
        if not _point_hits_obstacle(source, obstacles, config.obstacle_clearance):
            return source
    raise RuntimeError("Failed to sample a source position clear of room obstacles.")


def _rounded_position(point, precision=3):
    return tuple(round(float(coord), precision) for coord in point)


def _scene_positions(scene):
    points = [scene.mic_pos, *scene.source_pos]
    return tuple(_rounded_position(point) for point in points)


def _scene_signature(scene):
    return _scene_positions(scene)


def _sample_room_scene(
    room_dim,
    rt60,
    obstacles,
    config,
    rng,
    used_signatures,
    used_positions,
):
    for _ in range(512):
        mic_pos = _sample_clear_point(
            room_dim,
            config.mic_margin,
            obstacles,
            config.obstacle_clearance,
            rng,
            height_range=config.mic_height_range,
        )
        source_pos = []
        labels = []
        for idx in range(config.num_near_sources):
            source_pos.append(
                _sample_clear_source(
                    room_dim,
                    mic_pos,
                    config.near_distance_range[0],
                    config.near_distance_range[1],
                    config,
                    obstacles,
                    rng,
                )
            )
            labels.append(f"near_{idx}")
        for idx in range(config.num_far_sources):
            max_far = min(
                float(config.far_distance_range[1]),
                _max_room_horizontal_distance_from_point(
                    room_dim, mic_pos, config.source_margin
                ),
            )
            min_far = min(float(config.far_distance_range[0]), max_far)
            source_pos.append(
                _sample_clear_source(
                    room_dim,
                    mic_pos,
                    min_far,
                    max_far,
                    config,
                    obstacles,
                    rng,
                )
            )
            labels.append(f"far_{idx}")
        scene = HybridRIRScene(
            room_dim=room_dim.astype(float).tolist(),
            rt60=float(rt60),
            mic_pos=mic_pos.astype(float).tolist(),
            source_pos=[source.astype(float).tolist() for source in source_pos],
            source_labels=labels,
            obstacles=obstacles,
        )
        signature = _scene_signature(scene)
        positions = _scene_positions(scene)
        if signature in used_signatures or any(
            point in used_positions for point in positions
        ):
            continue
        used_signatures.add(signature)
        used_positions.update(positions)
        return scene
    raise RuntimeError("Failed to sample a unique RIR position set for this room.")


def _sample_room_geometry(config, rng):
    room_dim = np.asarray(
        [rng.uniform(low, high) for low, high in config.room_dim_range],
        dtype=np.float64,
    )
    rt60 = max(
        float(rng.uniform(*config.rt60_range)),
        _min_feasible_rt60(room_dim, config.sound_speed),
    )
    obstacles = sample_polygon_obstacles(
        room_dim=room_dim,
        protected_points=[],
        config=config,
        rng=rng,
    )
    return room_dim, rt60, obstacles


def _make_low_backend(args):
    material_damping = args["low_backend"].endswith("-material")
    if args["low_backend"] == "analytic-impedance":
        boundary_config = RectangularImpedanceBoundaryConfig.from_json(
            args["impedance_boundary_config"]
        )
        residue_calibration = (
            ImpedanceModalResidueCalibration.from_json(
                args["impedance_residue_calibration"]
            )
            if args["impedance_residue_calibration"] is not None
            else None
        )
        return ImpedanceModalLowFrequencyBackend(
            boundary_config=boundary_config,
            num_modes_per_axis=args["impedance_mode_index_limit"],
            max_modes=args["impedance_max_modes"],
            residue_calibration=residue_calibration,
            continuation_steps=args["impedance_continuation_steps"],
        )
    if args["low_backend"] in {"pytard", "pytard-material"}:
        return GpuARDPytARDBackend(
            low_sample_rate=args["pytard_low_sample_rate"],
            spatial_samples_per_wave_length=args[
                "pytard_spatial_samples_per_wavelength"
            ],
            calibration_peak=args["pytard_calibration_peak"],
            rt60_decay_scale=args["pytard_rt60_decay_scale"],
            material_modal_damping=material_damping,
            material_modal_loss_scale=args["material_modal_loss_scale"],
        )
    if args["low_backend"] in {"pytard-cupy", "pytard-cupy-material"}:
        return GpuARDPytARDCuPyBackend(
            low_sample_rate=args["pytard_low_sample_rate"],
            spatial_samples_per_wave_length=args[
                "pytard_spatial_samples_per_wavelength"
            ],
            calibration_peak=args["pytard_calibration_peak"],
            rt60_decay_scale=args["pytard_rt60_decay_scale"],
            material_modal_damping=material_damping,
            material_modal_loss_scale=args["material_modal_loss_scale"],
        )
    return AnalyticModalLowFrequencyBackend(
        material_modal_damping=material_damping,
        material_modal_loss_scale=args["material_modal_loss_scale"],
    )


def _make_high_backend(args):
    if args["high_backend"] == "path-events-m4":
        return PathEventFDNHighFrequencyBackend(
            max_order=args["pra_max_order"],
            include_scene_interactions=True,
            mixing_time_s=1e-3 * args["fdn_mixing_time_ms"],
            transition_duration_s=1e-3 * args["fdn_transition_ms"],
            delay_line_count=args["fdn_delay_lines"],
            fdn_seed=args["fdn_seed"],
        )
    if args["high_backend"] == "path-events-m3":
        return PathEventHighFrequencyBackend(
            max_order=args["pra_max_order"],
            include_scene_interactions=True,
        )
    return PyroomacousticsHighFrequencyBackend(
        max_order=args["pra_max_order"],
        n_rays=args["pra_n_rays"],
    )


def _init_worker(config, args, output_dir, gpu_device):
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    # torch reads OMP_NUM_THREADS at import (already set at module load), but its
    # intra-op pool still defaults to the core count, so pin it explicitly too.
    torch.set_num_threads(1)
    _WORKER_CONTEXT["config"] = config
    _WORKER_CONTEXT["args"] = args
    _WORKER_CONTEXT["output_dir"] = Path(output_dir)
    _WORKER_CONTEXT["gpu_device"] = gpu_device
    _WORKER_CONTEXT["low_backend"] = _make_low_backend(args)
    _WORKER_CONTEXT["high_backend"] = _make_high_backend(args)
    gpu_label = gpu_device if gpu_device is not None else "default"
    print(f"worker ready gpu={gpu_label}", flush=True)


def _task_output_paths(output_dir, task):
    room_dir = Path(output_dir) / task["room_id"]
    wav_path = room_dir / f"{task['sample_id']}.wav"
    json_path = room_dir / f"{task['sample_id']}.json"
    return wav_path, json_path


def _task_is_complete(output_dir, task):
    """Return True when a task's RIR was already fully written.

    ``_write_room_rir_item`` writes the WAV first and the JSON last, so a present
    and parseable JSON sidecar implies a complete WAV. We still require both files
    and that the JSON parses, so a run interrupted mid-write is regenerated.
    """
    wav_path, json_path = _task_output_paths(output_dir, task)
    if not (wav_path.exists() and json_path.exists()):
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = task.get("m6")
    if expected is None:
        return True
    recorded = metadata.get("m6")
    if not isinstance(recorded, dict) or any(
        recorded.get(key) != value for key, value in expected.items()
    ):
        return False
    try:
        if recorded.get("rir_sha256") != sha256_file(wav_path):
            return False
        if canonical_json_sha256(metadata.get("scene")) != expected["scene_sha256"]:
            return False
        info = sf.info(wav_path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if (
        info.samplerate != expected["sample_rate"]
        or info.channels != expected["channel_count"]
        or info.frames != expected["frame_count"]
    ):
        return False
    return True


def _write_room_rir_item(output_dir, task, rir, metadata, sample_rate):
    room_dir = Path(output_dir) / task["room_id"]
    room_dir.mkdir(parents=True, exist_ok=True)
    wav_path = room_dir / f"{task['sample_id']}.wav"
    json_path = room_dir / f"{task['sample_id']}.json"

    wav = rir.detach().cpu()
    if wav.ndim != 2 or wav.shape[0] != 5:
        raise ValueError(f"Expected RIR tensor [5, samples], got {tuple(wav.shape)}")
    metadata["rir_filename"] = wav_path.name
    metadata["metadata_filename"] = json_path.name
    torchaudio.save(str(wav_path), wav, int(sample_rate), encoding="PCM_F")
    canonicalize_float_wav_header(wav_path)
    if task.get("m6") is not None:
        metadata["m6"] = {
            **task["m6"],
            "rir_sha256": sha256_file(wav_path),
        }
    payload = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if task.get("m6") is None:
        json_path.write_text(payload, encoding="utf-8")
    else:
        temporary = json_path.with_suffix(json_path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(json_path)
    return wav_path, json_path


def _generate_task(task):
    config = _WORKER_CONTEXT["config"]
    if task.get("m6") is not None:
        task_seed = int(task["m6"]["generation_seed"])
        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        high_backend = _WORKER_CONTEXT["high_backend"]
        if isinstance(high_backend, PyroomacousticsHighFrequencyBackend):
            high_backend.set_rng_seed(task_seed)
    rir, metadata = generate_hybrid_rir(
        config=config,
        scene=task["scene"],
        low_backend=_WORKER_CONTEXT["low_backend"],
        high_backend=_WORKER_CONTEXT["high_backend"],
    )
    metadata["sample_id"] = task["sample_id"]
    metadata["room_id"] = task["room_id"]
    metadata["room_index"] = task["room_index"]
    metadata["rir_index"] = task["rir_index"]
    metadata["gpu_device"] = _WORKER_CONTEXT["gpu_device"]
    _write_room_rir_item(
        output_dir=_WORKER_CONTEXT["output_dir"],
        task=task,
        rir=rir,
        metadata=metadata,
        sample_rate=config.sample_rate,
    )
    return task["sample_id"]


def _iter_tasks(config, args):
    rng = np.random.default_rng(args.seed)
    for room_idx in range(args.n_rooms):
        room_id = f"room_{room_idx:06d}"
        room_dim, rt60, obstacles = _sample_room_geometry(config, rng)
        material_seed = (
            int(rng.integers(0, np.iinfo(np.uint32).max))
            if args.scene_version == "v1"
            else None
        )
        used_signatures = set()
        used_positions = set()
        for rir_idx in range(args.rir_per_room):
            scene = _sample_room_scene(
                room_dim=room_dim,
                rt60=rt60,
                obstacles=obstacles,
                config=config,
                rng=rng,
                used_signatures=used_signatures,
                used_positions=used_positions,
            )
            sample_id = f"{room_id}_{rir_idx:06d}"
            if args.scene_version == "v1":
                scene = upgrade_hybrid_scene_to_v2(
                    scene,
                    seed=material_seed,
                    room_type=(
                        None if args.room_type == "mixed" else args.room_type
                    ),
                    scene_id=sample_id,
                    material_variation_scale=args.material_variation_scale,
                )
            yield {
                "sample_id": sample_id,
                "room_id": room_id,
                "room_index": room_idx,
                "rir_index": rir_idx,
                "scene": scene,
            }


def _build_m6_items(output_dir, tasks, config):
    items = []
    for task in sorted(tasks, key=lambda value: value["sample_id"]):
        if not _task_is_complete(output_dir, task):
            raise RuntimeError(
                f"cannot emit M6 manifest from incomplete item {task['sample_id']}"
            )
        wav_path, json_path = _task_output_paths(output_dir, task)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        info = sf.info(wav_path)
        m6 = task["m6"]
        items.append(
            RIRBankItem(
                item_id=task["sample_id"],
                room_id=task["room_id"],
                acoustic_space_id=m6["acoustic_space_id"],
                scene_id=task["sample_id"],
                split=m6["split"],
                generation_seed=m6["generation_seed"],
                origin="synthetic",
                renderer_profile_id=m6["renderer_profile_id"],
                signal_variant="physical",
                level_policy=str(config.output_mode),
                rir_path=wav_path.relative_to(output_dir).as_posix(),
                metadata_path=json_path.relative_to(output_dir).as_posix(),
                rir_sha256=sha256_file(wav_path),
                metadata_sha256=sha256_file(json_path),
                scene_sha256=canonical_json_sha256(metadata["scene"]),
                sample_rate=info.samplerate,
                channel_count=info.channels,
                frame_count=info.frames,
                qc_status="pending",
            )
        )
    return tuple(items)


def _emit_m6_manifest(
    output_dir, tasks, config, args, context, *, skipped, elapsed_seconds=None
):
    output_root = Path(output_dir)
    items = _build_m6_items(output_root, tasks, config)
    split_indexes = write_split_indexes(output_root, items)
    profile = BankRendererProfile(
        profile_id=context["renderer_profile_id"],
        renderer_id="puresound.hybrid_rir",
        renderer_version="M6.2",
        low_backend=args.low_backend,
        high_backend=args.high_backend,
        scene_schema_version=context["scene_schema_version"],
        renderer_config_sha256=context["renderer_config_sha256"],
        evidence_tier="development",
    )
    generator = BankGeneratorProvenance(
        generator_id="egs.rir_generation.generate_hybrid_rir",
        generator_version="M6.2",
        code_revision=context["code_revision"],
        config_sha256=context["config_sha256"],
        task_plan_sha256=canonical_json_sha256(task_plan_rows(items)),
        seed=int(args.seed),
    )
    manifest = RIRBankManifest(
        bank_id=context["bank_id"],
        release_status="draft",
        split_policy=context["split_policy"],
        generator=generator,
        renderer_profiles=(profile,),
        items=items,
        split_indexes=split_indexes,
    ).with_content_sha256()
    manifest_path = output_root / "rir_bank_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    audit = audit_rir_bank_manifest(manifest, output_root)
    audit["generation_run"] = {
        "task_count": len(tasks),
        "resume_requested": bool(args.resume),
        "items_skipped_as_complete": int(skipped),
        "items_generated": len(tasks) - int(skipped),
        "num_workers": int(args.num_workers),
        "elapsed_seconds": (
            None if elapsed_seconds is None else float(elapsed_seconds)
        ),
        # A worker exception propagates through ``future.result()`` and aborts the
        # run before this manifest is written, so a run that gets here generated
        # every task it attempted.  Recorded explicitly because M6.5's throughput
        # contract requires the failure count to be stated rather than inferred.
        "items_failed": 0,
        "items_failed_policy": "any_item_failure_aborts_the_run_before_manifest_emit",
    }
    audit_path = output_root / "rir_bank_generation_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not audit["ready_for_m6_bank_generation"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"M6 bank audit failed: {', '.join(failed)}")
    return manifest, audit


def _worker_plan(num_workers, gpu_devices):
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not gpu_devices:
        return [(None, num_workers)]
    plan = []
    for idx, device in enumerate(gpu_devices):
        count = num_workers // len(gpu_devices)
        if idx < num_workers % len(gpu_devices):
            count += 1
        if count > 0:
            plan.append((device, count))
    return plan


def _generate_tasks_serial(tasks, total, config, args, output_dir, gpu_device, initial=0):
    _init_worker(config, args, output_dir, gpu_device)
    with tqdm(total=total, initial=initial, desc="hybrid-rir") as progress:
        for task in tasks:
            _generate_task(task)
            progress.update(1)


def _generate_tasks_parallel(tasks, total, config, args, output_dir, worker_plan, initial=0):
    mp_context = get_context("spawn")
    executors = []
    futures = set()
    max_pending = max(1, sum(max_workers for _, max_workers in worker_plan) * 2)
    try:
        for gpu_device, max_workers in worker_plan:
            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(config, args, str(output_dir), gpu_device),
            )
            executors.append(executor)

        with tqdm(total=total, initial=initial, desc="hybrid-rir") as progress:
            for index, task in enumerate(tasks):
                while len(futures) >= max_pending:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    progress.update(len(done))
                executor = executors[index % len(executors)]
                futures.add(executor.submit(_generate_task, task))

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                progress.update(len(done))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def main():
    args = _parse_args()
    if args.n_rooms < 1:
        raise ValueError("--n-rooms must be >= 1")
    if args.rir_per_room < 1:
        raise ValueError("--rir-per-room must be >= 1")
    gpu_devices = _parse_gpu_devices(args.gpu_devices)
    if args.num_workers is None:
        args.num_workers = len(gpu_devices) if gpu_devices else 1
    _warn_gpu_configuration(args, gpu_devices)
    if args.material_variation_scale < 0.0:
        raise ValueError("--material-variation-scale must be >= 0")
    if not 0.0 < args.low_fmin_hz < args.low_fmax_hz:
        raise ValueError("--low-fmin-hz must be below --low-fmax-hz")
    if args.low_fmax_hz >= 0.5 * args.sample_rate:
        raise ValueError("--low-fmax-hz must be below output Nyquist")
    if not 0.0 < args.crossover_hz < 0.5 * args.sample_rate:
        raise ValueError("--crossover-hz must lie below output Nyquist")
    if args.material_modal_loss_scale < 0.0:
        raise ValueError("--material-modal-loss-scale must be >= 0")
    if args.high_backend in {"path-events-m3", "path-events-m4"}:
        if args.scene_version != "v1":
            raise ValueError(
                f"--high-backend={args.high_backend} requires --scene-version v1"
            )
        if not 0 <= args.pra_max_order <= 20:
            raise ValueError(
                "PathEvent --pra-max-order must be in [0, 20]"
            )
    if args.high_backend == "path-events-m4":
        if args.fdn_mixing_time_ms <= 0.0:
            raise ValueError("--fdn-mixing-time-ms must be positive")
        if args.fdn_transition_ms <= 0.0:
            raise ValueError("--fdn-transition-ms must be positive")
        if (
            args.fdn_delay_lines < 2
            or args.fdn_delay_lines & (args.fdn_delay_lines - 1)
        ):
            raise ValueError("--fdn-delay-lines must be a power of two >= 2")
    if args.low_backend.endswith("-material") and args.scene_version != "v1":
        raise ValueError(
            f"--low-backend={args.low_backend} requires --scene-version v1"
        )
    if args.low_backend == "analytic-impedance":
        if args.scene_version != "v1":
            raise ValueError(
                "--low-backend=analytic-impedance requires --scene-version v1"
            )
        if args.impedance_boundary_config is None:
            raise ValueError(
                "--low-backend=analytic-impedance requires "
                "--impedance-boundary-config"
            )
        if args.impedance_mode_index_limit < 1:
            raise ValueError("--impedance-mode-index-limit must be positive")
        if args.impedance_max_modes < 1:
            raise ValueError("--impedance-max-modes must be positive")
        if args.impedance_continuation_steps < 1:
            raise ValueError("--impedance-continuation-steps must be positive")
    elif args.impedance_boundary_config is not None:
        raise ValueError(
            "--impedance-boundary-config only applies to "
            "--low-backend=analytic-impedance"
        )
    if (
        args.low_backend != "analytic-impedance"
        and args.impedance_residue_calibration is not None
    ):
        raise ValueError(
            "--impedance-residue-calibration only applies to "
            "--low-backend=analytic-impedance"
        )
    if (
        not args.low_backend.endswith("-material")
        and args.material_modal_loss_scale != 1.0
    ):
        raise ValueError(
            "--material-modal-loss-scale only applies to a '*-material' "
            "low backend"
        )
    if args.output_mode is None:
        args.output_mode = (
            "calibrated" if args.scene_version == "v1" else "peak_normalized"
        )
    if args.record_realized_metrics is None:
        args.record_realized_metrics = args.scene_version == "v1"

    config = HybridRIRConfig(
        sample_rate=args.sample_rate,
        duration=args.duration,
        crossover_hz=args.crossover_hz,
        low_fmin_hz=args.low_fmin_hz,
        low_fmax_hz=args.low_fmax_hz,
        room_dim_range=(tuple(args.room_x), tuple(args.room_y), tuple(args.room_z)),
        rt60_range=tuple(args.rt60),
        num_obstacles_range=tuple(args.obstacles),
        near_distance_range=tuple(args.near_dist),
        far_distance_range=tuple(args.far_dist),
        output_mode=args.output_mode,
        record_realized_metrics=args.record_realized_metrics,
        crossover_match_target_db=args.crossover_target_db,
        crossover_match_gain_range=(1e-4, args.crossover_max_gain),
        preserve_source_convention_at_crossover=(
            args.preserve_crossover_source_convention
        ),
    )

    # Materialize the deterministic task plan up front. Scene sampling only draws
    # positions (cheap relative to the RIR solve) and consumes the RNG in a fixed
    # order, so building the full list yields the same room/mic/source geometry a
    # fresh run would, while letting us count how many items are already done.
    all_tasks = list(_iter_tasks(config, args))
    m6_context = None
    if args.emit_m6_manifest:
        m6_context = _attach_m6_task_contract(all_tasks, config, args)
        represented_splits = {task["m6"]["split"] for task in all_tasks}
        required_splits = {"train", "validation", "test"}
        if represented_splits != required_splits:
            missing = sorted(required_splits - represented_splits)
            raise ValueError(
                "M6 task plan must contain train, validation, and test; "
                f"missing {missing}. Increase --n-rooms or choose another "
                "--m6-split-seed."
            )
    tasks = all_tasks
    total = len(all_tasks)
    skipped = 0
    if args.resume:
        pending = [
            task for task in tasks if not _task_is_complete(args.output_dir, task)
        ]
        skipped = total - len(pending)
        print(
            f"[generate_hybrid_rir] resume: {skipped} already generated, "
            f"{len(pending)} remaining (of {total}).",
            flush=True,
        )
        if not pending:
            print(
                "[generate_hybrid_rir] resume: nothing to do; all RIRs present.",
                flush=True,
            )
        tasks = pending

    generation_started = time.perf_counter()
    if tasks:
        worker_args = vars(args).copy()
        plan = _worker_plan(args.num_workers, gpu_devices)
        if args.num_workers == 1:
            gpu_device = plan[0][0]
            _generate_tasks_serial(
                tasks,
                total,
                config,
                worker_args,
                args.output_dir,
                gpu_device,
                initial=skipped,
            )
        else:
            _generate_tasks_parallel(
                tasks,
                total,
                config,
                worker_args,
                args.output_dir,
                plan,
                initial=skipped,
            )
    generation_elapsed_s = time.perf_counter() - generation_started
    if args.emit_m6_manifest:
        manifest, audit = _emit_m6_manifest(
            args.output_dir,
            all_tasks,
            config,
            args,
            m6_context,
            skipped=skipped,
            elapsed_seconds=generation_elapsed_s,
        )
        print(
            f"[generate_hybrid_rir] M6 manifest {manifest.manifest_sha256}; "
            f"{len(manifest.items)} items; audit "
            f"{'PASS' if audit['ready_for_m6_bank_generation'] else 'FAIL'}.",
            flush=True,
        )


if __name__ == "__main__":
    main()
