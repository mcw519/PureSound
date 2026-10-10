"""Audio inventory records, and the metafile format the dataset classes read.

The metafile is a seven-column CSV with a fixed header. It is the only thing a
training recipe consumes, and its shape is not ours to extend -- anything a corpus
knows beyond those seven columns (a noise category, a capture device, a room) rides
in :attr:`AudioRecord.tags` and is written to a separate inventory file. Keeping the
two apart is what lets a corpus carry arbitrary metadata without every dataset
parser having to grow a column for it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


METAFILE_HEADER = "uttid, spkid, gender, path, length, sample rate, channels"
METAFILE_COLUMNS = 7


@dataclass(frozen=True)
class AudioRecord:
    """One audio file, as both recipes and inventories see it.

    ``length`` is in samples at ``sample_rate``; the two always describe the file
    at ``path``, never a resampled copy of it. :func:`replace_audio` is the only
    supported way to point a record at a converted file, because it re-reads the
    converted file's own header instead of scaling the old numbers.
    """

    uttid: str
    spkid: str
    gender: str
    path: Path
    length: int
    sample_rate: int
    channels: int
    tags: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Seconds, or 0.0 for a record whose sample rate never got filled in."""
        return self.length / self.sample_rate if self.sample_rate else 0.0

    def metafile_row(self) -> str:
        # The metafile is comma-separated with no quoting, and every reader
        # (these tools and the training dataset) splits on the comma. A comma in
        # a field shifts every column after it -- LibriLight has chapters named
        # "..._alger_jr.,_64kb.flac" -- so it is refused here, where it can be
        # named, rather than surfacing later as an unparseable length.
        for name, value in (("uttid", self.uttid), ("spkid", self.spkid), ("path", str(self.path))):
            if "," in value:
                raise ValueError(f"metafile {name} contains a comma: {value!r}")
        return (
            f"{self.uttid}, {self.spkid}, {self.gender}, {self.path}, "
            f"{self.length}, {self.sample_rate}, {self.channels}"
        )

    def inventory_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "uttid": self.uttid,
            "spkid": self.spkid,
            "gender": self.gender,
            "path": str(self.path),
            "length": self.length,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        if self.tags:
            entry["tags"] = dict(self.tags)
        return entry


def _is_header(row: Sequence[str]) -> bool:
    return (
        len(row) >= 5
        and row[3].strip().lower() == "path"
        and row[4].strip().lower() == "length"
    )


def write_metafile(path: str | Path, records: Iterable[AudioRecord]) -> int:
    """Write the seven-column metafile. Returns how many rows were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(METAFILE_HEADER + "\n")
        for record in records:
            handle.write(record.metafile_row() + "\n")
            count += 1
    return count


def read_metafile(path: str | Path) -> list[AudioRecord]:
    """Read a metafile back into records. Tags are not recoverable from it."""
    records: list[AudioRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            row = [cell.strip() for cell in row]
            if not row or not any(row) or _is_header(row):
                continue
            if len(row) < METAFILE_COLUMNS:
                raise ValueError(f"{path}: expected {METAFILE_COLUMNS} columns, got {row!r}")
            records.append(
                AudioRecord(
                    uttid=row[0],
                    spkid=row[1],
                    gender=row[2],
                    path=Path(row[3]),
                    length=int(row[4]),
                    sample_rate=int(row[5]),
                    channels=int(row[6]),
                )
            )
    return records


def write_inventory(path: str | Path, records: Iterable[AudioRecord]) -> int:
    """Write one JSON object per line, tags included. Returns the line count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.inventory_entry(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_inventory(path: str | Path) -> Iterator[AudioRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            yield AudioRecord(
                uttid=entry["uttid"],
                spkid=entry["spkid"],
                gender=entry["gender"],
                path=Path(entry["path"]),
                length=int(entry["length"]),
                sample_rate=int(entry["sample_rate"]),
                channels=int(entry["channels"]),
                tags=entry.get("tags", {}),
            )


def tag_histogram(records: Iterable[AudioRecord], key: str) -> dict[str, int]:
    """Count records per value of one tag -- the per-category view of a set.

    Records without the tag are counted under ``"unknown"`` rather than dropped,
    so the total always matches the set size and a mostly-unlabelled inventory is
    obvious instead of looking like a small clean one.
    """
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.tags.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
