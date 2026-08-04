#!/usr/bin/env python3
"""Generate, QC, and package one deterministic M6 synthetic RIR bank.

This is the public one-command M6 entry point.  It keeps the phase-specific
generator, QC, and release tools behind a small stable interface so a user can
run a complete training-bank candidate without remembering three commands.
The resulting release is an M6 candidate; production promotion still requires
the empirical evidence and approvals described in the M6 contract.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "egs/rir_generation/generate_hybrid_rir.py"
RUN_QC = REPO_ROOT / "egs/rir_generation/phases/m6_bank/scripts/run_m6_item_qc.py"
BUILD_RELEASE = (
    REPO_ROOT / "egs/rir_generation/phases/m6_bank/scripts/build_m6_variant_release.py"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output root. The bank and release are created below it as "
            "<backend>_bank and <backend>_release."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("pyroomacoustics", "path-events-m4"),
        default="path-events-m4",
        help=(
            "High-frequency renderer used for this M6 candidate. Defaults to "
            "path-events-m4: against 1465 measured RIRs its octave decay "
            "shape sits 8x closer than pyroomacoustics, whose high-frequency "
            "reverberation runs ~2.2x long (RIR_EXP_LOG.md 6.6.6). "
            "pyroomacoustics remains available for speed and as the A/B arm."
        ),
    )
    parser.add_argument("--n-rooms", type=int, default=1000)
    parser.add_argument("--rir-per-room", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--gpu-devices",
        help=(
            "Comma-separated CUDA devices for pytard-cupy workers, for example "
            "'0' or '0,1'. Requires --low-backend=pytard-cupy-material "
            "for the M6 material-damping default."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.6)
    parser.add_argument(
        "--low-backend",
        default="pytard-material",
        help=(
            "Low-frequency renderer. M6 defaults to material-derived per-mode "
            "damping; use pytard-cupy-material for GPU execution."
        ),
    )
    parser.add_argument("--scene-version", choices=("v0", "v1"), default="v1")
    parser.add_argument(
        "--room-type",
        choices=("mixed", "classroom", "living_room", "meeting_room", "office"),
        default="mixed",
    )
    parser.add_argument(
        "--output-mode",
        choices=("peak_normalized", "calibrated"),
        default="calibrated",
    )
    parser.add_argument(
        "--record-realized-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record per-item acoustic metrics (default: enabled).",
    )
    parser.add_argument(
        "--m6-bank-id",
        help="Stable bank identifier (default is derived from backend and seed).",
    )
    parser.add_argument(
        "--release-id",
        help="Release identifier (default is derived from backend and seed).",
    )
    parser.add_argument(
        "--normalized-peak",
        type=float,
        default=0.98,
        help="Peak used for the synthetic_peak_normalized variant.",
    )
    parser.add_argument(
        "--qc-workers",
        type=int,
        default=1,
        help="Parallel item evaluators for M6 QC and normalized release.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an interrupted bank generation (default: enabled).",
    )
    return parser


def _run(command: list[str], env: dict[str, str]) -> None:
    print(f"[generate_m6_bank] $ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.n_rooms < 1 or args.rir_per_room < 1:
        raise SystemExit("--n-rooms and --rir-per-room must both be positive")
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be positive")
    if args.duration <= 0 or args.sample_rate <= 0:
        raise SystemExit("--duration and --sample-rate must be positive")
    if not 0 < args.normalized_peak <= 1:
        raise SystemExit("--normalized-peak must be in (0, 1]")
    if args.qc_workers < 1:
        raise SystemExit("--qc-workers must be positive")
    if args.gpu_devices and not args.low_backend.startswith("pytard-cupy"):
        raise SystemExit(
            "--gpu-devices requires --low-backend=pytard-cupy or "
            "--low-backend=pytard-cupy-material"
        )

    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root = output_root.resolve()
    bank_dir = output_root / f"{args.backend}_bank"
    release_dir = output_root / f"{args.backend}_release"
    if release_dir.exists():
        raise SystemExit(
            f"release already exists: {release_dir}\n"
            "choose a new --output-dir; existing releases are never overwritten"
        )

    bank_id = args.m6_bank_id or f"puresound-m6-{args.backend}-{args.seed}"
    release_id = args.release_id or bank_id
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python_path = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path
    python = sys.executable

    generate_command = [
        python,
        str(GENERATOR),
        "--output-dir",
        str(bank_dir),
        "--n-rooms",
        str(args.n_rooms),
        "--rir-per-room",
        str(args.rir_per_room),
        "--sample-rate",
        str(args.sample_rate),
        "--duration",
        str(args.duration),
        "--scene-version",
        args.scene_version,
        "--room-type",
        args.room_type,
        "--output-mode",
        args.output_mode,
        "--low-backend",
        args.low_backend,
        "--high-backend",
        args.backend,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--emit-m6-manifest",
        "--m6-bank-id",
        bank_id,
    ]
    if args.gpu_devices:
        generate_command.extend(["--gpu-devices", args.gpu_devices])
    generate_command.append(
        "--record-realized-metrics"
        if args.record_realized_metrics
        else "--no-record-realized-metrics"
    )
    if args.resume:
        generate_command.append("--resume")
    _run(generate_command, env)

    _run(
        [
            python,
            str(RUN_QC),
            "--bank",
            str(bank_dir),
            "--workers",
            str(args.qc_workers),
        ],
        env,
    )
    _run(
        [
            python,
            str(BUILD_RELEASE),
            "--source-bank",
            str(bank_dir),
            "--output-dir",
            str(release_dir),
            "--release-id",
            release_id,
            "--normalized-peak",
            str(args.normalized_peak),
            "--qc-workers",
            str(args.qc_workers),
        ],
        env,
    )

    print(f"[generate_m6_bank] bank ready: {bank_dir}")
    print(f"[generate_m6_bank] release ready: {release_dir}")
    print("[generate_m6_bank] recipe=synthetic_calibrated split=train")
    print("[generate_m6_bank] status=candidate (production promotion remains gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
