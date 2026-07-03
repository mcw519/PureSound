"""Speaker-disjoint train/dev split for voice_isolate metafiles (plan section 9.4).

The previous dns5-read.dev.list was simply the first ~1000 rows of
dns5-read.list, so every dev speaker also appeared in the train split and
validation could not measure speaker generalization. This script regroups a
metafile by spkid, holds out a fraction of speakers for dev, and writes both
splits back with the original rows untouched (so any downstream parser sees
the exact same format).

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/split_by_speaker.py \
        egs/voice_isolate/data/dns5-read.list \
        --train-out egs/voice_isolate/data/dns5-read.train.list \
        --dev-out egs/voice_isolate/data/dns5-read.dev.list \
        --dev-speaker-frac 0.05 --seed 0
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

HEADER = "uttid, spkid, gender, path, length, sample rate, channels"


def read_rows(f_path: Path) -> list[str]:
    rows = []
    for line in f_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("uttid"):
            continue
        rows.append(stripped)
    return rows


def spkid_of(row: str) -> str:
    return row.split(",")[1].strip()


def write_split(f_path: Path, rows: list[str]) -> None:
    f_path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_list")
    p.add_argument("--train-out", required=True)
    p.add_argument("--dev-out", required=True)
    p.add_argument("--dev-speaker-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = read_rows(Path(args.input_list))
    by_spk: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_spk[spkid_of(row)].append(row)

    speakers = sorted(by_spk)
    n_dev_spk = max(1, math.ceil(len(speakers) * args.dev_speaker_frac))
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    dev_spk = set(speakers[:n_dev_spk])
    train_spk = set(speakers[n_dev_spk:])
    assert not (dev_spk & train_spk), "train/dev speaker sets must be disjoint"

    train_rows = [r for r in rows if spkid_of(r) in train_spk]
    dev_rows = [r for r in rows if spkid_of(r) in dev_spk]
    assert len(train_rows) + len(dev_rows) == len(rows)

    write_split(Path(args.train_out), train_rows)
    write_split(Path(args.dev_out), dev_rows)

    print(f"input    : {args.input_list} ({len(rows)} utts, {len(by_spk)} speakers)")
    print(f"train    : {args.train_out} ({len(train_rows)} utts, {len(train_spk)} speakers)")
    print(f"dev      : {args.dev_out} ({len(dev_rows)} utts, {len(dev_spk)} speakers)")
    print("overlap  : 0 speakers (asserted)")


if __name__ == "__main__":
    main()
