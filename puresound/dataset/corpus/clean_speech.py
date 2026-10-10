"""Put a speech corpus through an enhancement model before it becomes training targets.

"Clean" speech corpora are not clean: read audiobooks carry room tone, hum and the
recorder's own noise floor, and a model trained to reproduce its target reproduces
that too. DPDFNet ran all of its training speech through DPCRN for this reason.
This module does the same with any checkpoint `puresound.evaluation.systems` can
load, writing a mirrored tree and a metafile that points at it.

It also measures what the cleaner did, per file, because "run everything through
a model" has a failure mode of its own: a cleaner that colours clean speech writes
its colouring into every target, and the student then learns it as the answer.
Each file gets

``removed_db``
    energy of ``input - output`` relative to the input. Large means the cleaner
    changed a lot -- noise, or speech.
``floor_in_db`` / ``floor_out_db``
    level of the quietest 20 % of 32 ms frames before and after. Their difference
    is the noise floor the cleaner took away, the part it exists to remove.

so a policy ("replace only where the floor dropped by more than X dB") can be
chosen from the numbers after the run instead of before it.

Batches are padded at the end only, and the models are causal, so an item's output
does not depend on what it was batched with; ``--verify`` checks that on real files
instead of assuming it.

Cleaned audio goes to its own tree, named after the cleaner, never beside the
source: a target that has been through a model must not be mistaken for a
recording, and two cleaners' outputs must not mix.

Run::

    python -m puresound.dataset.corpus.clean_speech \\
        --metafile egs/noise_suppression/data/dns5_all/dnsde_train.csv \\
        --source-root /path/to/audio/dns-5/datasets_fullband_16k \\
        --dest-root /path/to/training_set/ns_speech_cleaned/dpcrn_mamba_v2/dns5 \\
        --recipe egs/noise_suppression/config/infer_dpcrn.yaml \\
        --ckpt egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \\
        --out-metafile egs/noise_suppression/data/dns5_all/dnsde_train.v2clean.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from puresound.audio.io import AudioIO

from .records import AudioRecord, read_metafile, write_metafile
from .resample import ensure_distinct_destinations, mirrored_path


TAIL_PAD_SAMPLES = 2048
FLOOR_FRAME_SECONDS = 0.032
FLOOR_QUANTILE = 0.2
_EPS = 1e-10


def floor_db(wav: np.ndarray, sample_rate: int) -> float:
    """Mean level of the quietest ``FLOOR_QUANTILE`` of frames, in dBFS."""
    frame = max(1, int(FLOOR_FRAME_SECONDS * sample_rate))
    usable = (len(wav) // frame) * frame
    if usable == 0:
        return float(10 * np.log10(np.mean(wav**2) + _EPS))
    energy = np.mean(wav[:usable].reshape(-1, frame) ** 2, axis=1)
    count = max(1, int(math.ceil(len(energy) * FLOOR_QUANTILE)))
    return float(10 * np.log10(np.mean(np.sort(energy)[:count]) + _EPS))


def cleaning_stats(before: np.ndarray, after: np.ndarray, sample_rate: int) -> dict[str, float]:
    """What the cleaner did to one file; see the module docstring."""
    power_in = float(np.mean(before**2)) + _EPS
    return {
        "in_db": round(10 * math.log10(power_in), 2),
        "out_db": round(10 * math.log10(float(np.mean(after**2)) + _EPS), 2),
        "removed_db": round(10 * math.log10(float(np.mean((before - after) ** 2)) / power_in + _EPS), 2),
        "floor_in_db": round(floor_db(before, sample_rate), 2),
        "floor_out_db": round(floor_db(after, sample_rate), 2),
        "peak_out": round(float(np.max(np.abs(after))) if len(after) else 0.0, 4),
    }


def plan_batches(
    records: Sequence[AudioRecord], max_batch_samples: int, max_batch_items: int
) -> list[list[int]]:
    """Indices grouped by similar length, so padding stays small.

    ``max_batch_samples`` bounds the padded batch (items x longest item), which
    is what GPU memory follows.
    """
    order = sorted(range(len(records)), key=lambda i: records[i].length)
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        longest = records[index].length  # sorted: the newcomer is the longest
        if current and (
            len(current) + 1 > max_batch_items or (len(current) + 1) * longest > max_batch_samples
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


class _BatchLoader(torch.utils.data.Dataset):
    """One padded batch per item, read in DataLoader workers."""

    def __init__(self, records: Sequence[AudioRecord], batches: list[list[int]], sample_rate: int):
        self.records = records
        self.batches = batches
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int):
        torch.set_num_threads(1)
        waves = []
        for i in self.batches[index]:
            wav, rate = AudioIO.open(str(self.records[i].path), resample_to=self.sample_rate)
            waves.append(wav.reshape(-1))
        # A frame-based model returns up to one window short of its input, so
        # the batch runs a little past its longest item and every item is cut
        # back to its own length afterwards.
        longest = max(w.shape[-1] for w in waves) + TAIL_PAD_SAMPLES
        padded = torch.zeros(len(waves), longest)
        for row, wav in enumerate(waves):
            padded[row, : wav.shape[-1]] = wav
        return self.batches[index], padded, [w.shape[-1] for w in waves]


def _forward(system, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = system.model(batch.to(system.device), dry_blend=system.dry_blend)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(batch.shape[0], -1).float().cpu()


def _write(dest: Path, wav: np.ndarray, sample_rate: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(wav, -1.0, 1.0)
    AudioIO.save(torch.from_numpy(clipped).view(1, -1), str(dest), sample_rate, subtype="PCM_16")


def read_done(stats_path: Path) -> dict[str, dict[str, Any]]:
    """Stats already written by an earlier (interrupted) run, keyed by uttid."""
    done: dict[str, dict[str, Any]] = {}
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["uttid"]] = row
    return done


def verify_batching(system, records: Sequence[AudioRecord], sample_rate: int, count: int) -> float:
    """Largest |batched - alone| over ``count`` files, in dB below full scale."""
    chosen = list(records)[:count]
    loader = _BatchLoader(chosen, [list(range(len(chosen)))], sample_rate)
    _, padded, lengths = loader[0]
    batched = _forward(system, padded)
    worst = 0.0
    for row, length in enumerate(lengths):
        alone = _forward(system, padded[row : row + 1, : length + TAIL_PAD_SAMPLES])
        worst = max(worst, float((batched[row, :length] - alone[0, :length]).abs().max()))
    return 20 * math.log10(worst + _EPS)


def clean_records(
    records: Sequence[AudioRecord],
    system,
    *,
    source_root: Path,
    dest_root: Path,
    stats_path: Path,
    sample_rate: int = 16000,
    max_batch_samples: int = 16000 * 40,
    max_batch_items: int = 32,
    workers: int = 8,
    suffix: str = ".flac",
    progress_every: int = 200,
) -> Iterator[AudioRecord]:
    """Clean every record, yielding the repointed record as each file lands.

    Resumable: a uttid already in ``stats_path`` whose output exists is skipped.
    """
    ensure_distinct_destinations(records, source_root, dest_root, suffix)
    done = read_done(stats_path)
    pending: list[AudioRecord] = []
    for record in records:
        dest = mirrored_path(record.path, source_root, dest_root, suffix)
        if record.uttid in done and dest.is_file():
            yield _repoint(record, dest, sample_rate)
        else:
            pending.append(record)
    if not pending:
        return

    batches = plan_batches(pending, max_batch_samples, max_batch_items)
    loader = torch.utils.data.DataLoader(
        _BatchLoader(pending, batches, sample_rate),
        batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "a", encoding="utf-8") as stats, ThreadPoolExecutor(max(1, workers // 2)) as writers:
        # A stats row is what marks a file done on resume, so it is written only
        # once that file is on disk.
        in_flight: list[tuple[Any, dict[str, Any]]] = []

        def settle() -> None:
            for future, row_stats in in_flight:
                future.result()
                stats.write(json.dumps(row_stats) + "\n")
            in_flight.clear()
            stats.flush()

        for step, (indices, padded, lengths) in enumerate(loader, start=1):
            cleaned = _forward(system, padded)
            for row, (index, length) in enumerate(zip(indices, lengths)):
                record = pending[index]
                before = padded[row, :length].numpy()
                after = cleaned[row, :length].numpy()
                dest = mirrored_path(record.path, source_root, dest_root, suffix)
                row_stats = {"uttid": record.uttid, "spkid": record.spkid, **cleaning_stats(before, after, sample_rate)}
                in_flight.append((writers.submit(_write, dest, after, sample_rate), row_stats))
                yield _repoint(record, dest, sample_rate, length)
            if len(in_flight) > 2048:
                settle()
            if progress_every and step % progress_every == 0:
                print(f"  batch {step}/{len(batches)}", flush=True)
        settle()


def _repoint(record: AudioRecord, dest: Path, sample_rate: int, length: int | None = None) -> AudioRecord:
    return AudioRecord(
        uttid=record.uttid,
        spkid=record.spkid,
        gender=record.gender,
        path=dest,
        length=int(length if length is not None else record.length),
        sample_rate=int(sample_rate),
        channels=1,
        tags=record.tags,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.clean_speech",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--metafile", required=True, action="append", help="Repeatable.")
    parser.add_argument("--source-root", required=True, help="Root the metafile paths are mirrored from.")
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-blend", type=float, default=1.0,
        help="1.0 = the model's own output. The export default of 0.9 is not a cleaner.",
    )
    parser.add_argument("--out-metafile", default=None, help="Only with a single --metafile.")
    parser.add_argument("--stats", default=None, help="Defaults to <out-metafile>.stats.jsonl.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--max-batch-seconds", type=float, default=40.0,
        help="Padded audio per batch. DPCRN-Mamba v2 peaks near 1 GB per 10 s on GPU.",
    )
    parser.add_argument("--max-batch-items", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N rows of each metafile.")
    parser.add_argument("--verify", type=int, default=4, help="Check batched == alone on N files first; 0 skips.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from puresound.evaluation.systems import load_checkpoint_system

    args = build_parser().parse_args(argv)
    if args.out_metafile and len(args.metafile) > 1:
        print("--out-metafile names one output; pass one --metafile with it.", file=sys.stderr)
        return 2
    if not 0.0 < args.dry_blend <= 1.0:
        print("--dry-blend must be in (0, 1].", file=sys.stderr)
        return 2

    # cuDNN runs its convolutions and RNNs in TF32 by default, which is enough to
    # make a file's output depend on what it was batched with; fp32 costs no
    # speed here. A target should be a function of the cleaner and the input only.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    system = load_checkpoint_system(
        args.recipe, args.ckpt, task=args.task, device=args.device, dry_blend=args.dry_blend
    )
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()

    for metafile in args.metafile:
        metafile = Path(metafile).expanduser().resolve()
        records = read_metafile(metafile)[: args.limit] if args.limit else read_metafile(metafile)
        out = Path(args.out_metafile) if args.out_metafile else metafile.with_suffix(".clean.csv")
        stats = Path(args.stats) if args.stats else out.with_suffix(".stats.jsonl")
        print(f"{metafile.name}: {len(records)} file(s) -> {dest_root}")
        if args.verify:
            worst = verify_batching(system, records, args.sample_rate, args.verify)
            print(f"  batched vs alone, worst sample difference: {worst:.1f} dBFS")
            if worst > -60.0:
                print("  batching changes the output; refusing to run batched.", file=sys.stderr)
                return 1
        cleaned = list(
            clean_records(
                records, system,
                source_root=source_root, dest_root=dest_root, stats_path=stats,
                sample_rate=args.sample_rate,
                max_batch_samples=int(args.max_batch_seconds * args.sample_rate),
                max_batch_items=args.max_batch_items, workers=args.workers,
            )
        )
        write_metafile(out, sorted(cleaned, key=lambda record: record.uttid))
        print(f"  wrote {len(cleaned)} row(s) -> {out}; stats -> {stats}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
