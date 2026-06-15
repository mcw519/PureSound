import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
import torch
import torchaudio
from tqdm import tqdm

from puresound.audio.hybrid_rir import (
    AnalyticModalLowFrequencyBackend,
    GpuARDPytARDBackend,
    GpuARDPytARDCuPyBackend,
    HybridRIRConfig,
    HybridRIRScene,
    PyroomacousticsHighFrequencyBackend,
    _distance_point_to_polygon,
    _max_room_distance_from_point,
    _sample_point,
    _sample_source_in_shell,
    generate_hybrid_rir,
    sample_polygon_obstacles,
)


_WORKER_CONTEXT = {}


def _parse_args():
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
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--crossover-hz", type=float, default=1000.0)
    parser.add_argument(
        "--low-backend",
        choices=["pytard", "pytard-cupy", "analytic"],
        default="pytard",
        help="Use 'analytic' for smoke tests, 'pytard-cupy' for GPU low-band.",
    )
    parser.add_argument("--room-x", nargs=2, type=float, default=[3.5, 8.0])
    parser.add_argument("--room-y", nargs=2, type=float, default=[3.5, 7.0])
    parser.add_argument("--room-z", nargs=2, type=float, default=[2.4, 3.6])
    parser.add_argument("--rt60", nargs=2, type=float, default=[0.25, 0.8])
    parser.add_argument("--obstacles", nargs=2, type=int, default=[2, 6])
    parser.add_argument("--pra-max-order", type=int, default=12)
    parser.add_argument("--pra-n-rays", type=int, default=20000)
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
        default=2.0,
        help="Maximum gain allowed during low-band crossover matching.",
    )
    return parser.parse_args()


def _parse_gpu_devices(raw_devices):
    if raw_devices is None or raw_devices.strip() == "":
        return []
    return [device.strip() for device in raw_devices.split(",") if device.strip()]


def _warn(message):
    print(f"[generate_hybrid_rir] warning: {message}", file=sys.stderr, flush=True)


def _warn_gpu_configuration(args, gpu_devices):
    """Surface mismatches between the requested backend and GPU placement.

    Only the ``pytard-cupy`` low backend runs on the GPU; ``pytard``/``analytic``
    and the Pyroomacoustics high band are always CPU. Multiple cupy processes per
    device each hold a separate CUDA context and memory pool, so sharing a device
    risks out-of-memory failures.
    """
    uses_gpu = args.low_backend == "pytard-cupy"
    if gpu_devices and not uses_gpu:
        _warn(
            f"--gpu-devices={args.gpu_devices} is set but --low-backend="
            f"{args.low_backend} runs entirely on CPU (only 'pytard-cupy' uses "
            "the GPU; the high-frequency band is always CPU). No GPU work will "
            "happen; the device list only restricts each worker's CUDA visibility."
        )
    if not uses_gpu:
        return
    device_count = len(gpu_devices)
    if device_count == 0:
        _warn(
            "--low-backend=pytard-cupy without --gpu-devices: all "
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


def _point_hits_obstacle(point, obstacles, clearance):
    point_xy = np.asarray(point[:2], dtype=np.float64)
    for obstacle in obstacles:
        footprint = np.asarray(obstacle.footprint, dtype=np.float64)
        if obstacle.contains_xy(point) or (
            _distance_point_to_polygon(point_xy, footprint) < float(clearance)
        ):
            return True
    return False


def _sample_clear_point(room_dim, margin, obstacles, clearance, rng):
    for _ in range(512):
        point = _sample_point(room_dim, margin, rng)
        if not _point_hits_obstacle(point, obstacles, clearance):
            return point
    raise RuntimeError("Failed to sample a point clear of room obstacles.")


def _sample_clear_source(room_dim, mic_pos, min_dist, max_dist, config, obstacles, rng):
    for _ in range(512):
        source = _sample_source_in_shell(
            room_dim,
            mic_pos,
            min_dist,
            max_dist,
            config.source_margin,
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
                _max_room_distance_from_point(room_dim, mic_pos, config.source_margin),
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
    rt60 = float(rng.uniform(*config.rt60_range))
    obstacles = sample_polygon_obstacles(
        room_dim=room_dim,
        protected_points=[],
        config=config,
        rng=rng,
    )
    return room_dim, rt60, obstacles


def _make_low_backend(args):
    if args["low_backend"] == "pytard":
        return GpuARDPytARDBackend(
            low_sample_rate=args["pytard_low_sample_rate"],
            spatial_samples_per_wave_length=args[
                "pytard_spatial_samples_per_wavelength"
            ],
            calibration_peak=args["pytard_calibration_peak"],
            rt60_decay_scale=args["pytard_rt60_decay_scale"],
        )
    if args["low_backend"] == "pytard-cupy":
        return GpuARDPytARDCuPyBackend(
            low_sample_rate=args["pytard_low_sample_rate"],
            spatial_samples_per_wave_length=args[
                "pytard_spatial_samples_per_wavelength"
            ],
            calibration_peak=args["pytard_calibration_peak"],
            rt60_decay_scale=args["pytard_rt60_decay_scale"],
        )
    return AnalyticModalLowFrequencyBackend()


def _make_high_backend(args):
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
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return wav_path, json_path


def _generate_task(task):
    config = _WORKER_CONTEXT["config"]
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
            yield {
                "sample_id": f"{room_id}_{rir_idx:06d}",
                "room_id": room_id,
                "room_index": room_idx,
                "rir_index": rir_idx,
                "scene": scene,
            }


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


def _generate_tasks_serial(tasks, total, config, args, output_dir, gpu_device):
    _init_worker(config, args, output_dir, gpu_device)
    with tqdm(total=total, desc="hybrid-rir") as progress:
        for task in tasks:
            _generate_task(task)
            progress.update(1)


def _generate_tasks_parallel(tasks, total, config, args, output_dir, worker_plan):
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

        with tqdm(total=total, desc="hybrid-rir") as progress:
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

    config = HybridRIRConfig(
        sample_rate=args.sample_rate,
        duration=args.duration,
        crossover_hz=args.crossover_hz,
        room_dim_range=(tuple(args.room_x), tuple(args.room_y), tuple(args.room_z)),
        rt60_range=tuple(args.rt60),
        num_obstacles_range=tuple(args.obstacles),
        crossover_match_target_db=args.crossover_target_db,
        crossover_match_gain_range=(1e-4, args.crossover_max_gain),
    )

    total = args.n_rooms * args.rir_per_room
    tasks = _iter_tasks(config, args)
    worker_args = vars(args).copy()
    plan = _worker_plan(args.num_workers, gpu_devices)
    if args.num_workers == 1:
        gpu_device = plan[0][0]
        _generate_tasks_serial(
            tasks, total, config, worker_args, args.output_dir, gpu_device
        )
    else:
        _generate_tasks_parallel(tasks, total, config, worker_args, args.output_dir, plan)


if __name__ == "__main__":
    main()
