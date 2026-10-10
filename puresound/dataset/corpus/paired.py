"""Write a paired (noisy, clean) set and the manifest the scoring tools read.

Some benchmarks do not need synthesising -- they ship both sides already paired.
What they need is to be brought to one sample rate and described the same way a
synthesised set is, so an imported set and a generated one go through identical
scoring code rather than two code paths that drift.

Transcripts ride in the manifest when the corpus has them. That is what lets one
set serve both the reference metrics and the WER guardrail: a WER set assembled
separately would answer a slightly different question on slightly different cuts.

SNR is measured from ``noisy - clean``, the same definition the synthesised sets
use, rather than taken from a corpus README -- so the bands mean the same thing
in both.

``provenance.json`` is written beside the manifest, as it is for a synthesised
set. What it pins is different: a synthesised set records the commit and seed that
produced its audio, because that audio changes when the synthesis chain does; an
imported set records where the audio came from and what was done to it, because
the audio itself is fixed and the resampling is the only thing we did to it.
Either way a scoring run can say what it scored.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from puresound.audio.io import AudioIO


#: The SNR bands every manifest row is filed under; the scoring tools break
#: their results down by these.
SNR_BANDS = ((-math.inf, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, math.inf))


def band_of(value: float, bands: Sequence[tuple[float, float]] = SNR_BANDS) -> str:
    for low, high in bands:
        if low <= value < high:
            low_text = "-inf" if low == -math.inf else f"{low:g}"
            high_text = "inf" if high == math.inf else f"{high:g}"
            return f"{low_text}..{high_text}"
    return "unknown"


def rms_dbfs(wav: torch.Tensor) -> float:
    power = float(wav.reshape(-1).square().mean())
    return 10.0 * math.log10(power + 1e-12)


def measured_snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """SNR of what actually ended up in the mixture."""
    noise = noisy.reshape(-1) - clean.reshape(-1)
    signal_power = float(clean.reshape(-1).square().sum())
    noise_power = float(noise.square().sum())
    return 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))


@dataclass
class PairedSetReport:
    written: int = 0
    skipped_unpaired: list[str] = field(default_factory=list)
    missing_transcript: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.written} pair(s)"]
        if self.skipped_unpaired:
            parts.append(f"{len(self.skipped_unpaired)} unpaired skipped")
        if self.missing_transcript:
            parts.append(f"{len(self.missing_transcript)} without a transcript")
        return ", ".join(parts)


def read_transcript(path: Path) -> str:
    """One utterance of text, whitespace-normalised."""
    return " ".join(path.read_text(encoding="utf-8", errors="replace").split())


def describe_pair(
    item_id: str,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    sample_rate: int,
    source: str,
) -> dict[str, object]:
    """The manifest row every scoring tool reads, from the audio alone.

    One function, because three writers produce these rows -- an imported set, a
    synthesised one, and a mixed one -- and a field that one of them spells
    differently is a set the tools cannot band or compare.
    """
    length = int(min(noisy.shape[-1], clean.shape[-1]))
    snr = measured_snr_db(clean, noisy)
    return {
        "id": item_id,
        "mix": f"{item_id}_mix.wav",
        "clean": f"{item_id}_clean.wav",
        "sample_rate": int(sample_rate),
        "samples": length,
        "duration_s": round(length / sample_rate, 3),
        "snr_db": round(snr, 3),
        "snr_band": band_of(snr),
        "mix_rms_dbfs": round(rms_dbfs(noisy), 2),
        "clean_rms_dbfs": round(rms_dbfs(clean), 2),
        "source": source,
    }


def write_pair(
    out_dir: Path,
    item_id: str,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    sample_rate: int,
    source: str,
) -> dict[str, object]:
    """Save ``<id>_mix.wav`` / ``<id>_clean.wav`` and return their manifest row."""
    AudioIO.save(noisy, str(out_dir / f"{item_id}_mix.wav"), sample_rate)
    AudioIO.save(clean, str(out_dir / f"{item_id}_clean.wav"), sample_rate)
    return describe_pair(item_id, noisy, clean, sample_rate, source)


def write_paired_set(
    noisy_dir: str | Path,
    clean_dir: str | Path,
    out_dir: str | Path,
    *,
    sample_rate: int = 16000,
    transcript_dir: str | Path | None = None,
    transcript_suffix: str = ".txt",
    source: str | None = None,
    limit: int | None = None,
    extra_tags: Callable[[str], Mapping[str, object]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PairedSetReport:
    """Pair by filename, resample, and write ``<id>_mix.wav`` / ``<id>_clean.wav``.

    A file with no counterpart on the other side is skipped and named in the
    report rather than silently dropped -- a set that is quietly smaller than the
    corpus is a set whose numbers do not match anyone else's.
    """
    noisy_dir, clean_dir = Path(noisy_dir).expanduser().resolve(), Path(clean_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = Path(transcript_dir).expanduser().resolve() if transcript_dir else None

    noisy_files = {p.name: p for p in noisy_dir.glob("*.wav")}
    clean_files = {p.name: p for p in clean_dir.glob("*.wav")}
    paired = sorted(set(noisy_files) & set(clean_files))
    report = PairedSetReport(
        skipped_unpaired=sorted((set(noisy_files) | set(clean_files)) - set(paired))
    )
    if not paired:
        raise ValueError(f"No filename matches between {noisy_dir} and {clean_dir}.")
    if limit:
        paired = paired[:limit]

    # Rewritten from scratch every run, so a stale one cannot describe new audio.
    (out_dir / "provenance.json").unlink(missing_ok=True)

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, name in enumerate(paired, start=1):
            noisy, _ = AudioIO.open(str(noisy_files[name]), resample_to=sample_rate)
            clean, _ = AudioIO.open(str(clean_files[name]), resample_to=sample_rate)
            noisy, clean = noisy.reshape(1, -1), clean.reshape(1, -1)
            length = min(noisy.shape[-1], clean.shape[-1])
            noisy, clean = noisy[..., :length], clean[..., :length]

            item_id = Path(name).stem
            entry = write_pair(out_dir, item_id, noisy, clean, sample_rate, source or out_dir.name)
            if transcripts is not None:
                text_path = transcripts / f"{item_id}{transcript_suffix}"
                if text_path.is_file():
                    entry["transcript"] = read_transcript(text_path)
                else:
                    report.missing_transcript.append(item_id)
            if extra_tags is not None:
                entry.update(dict(extra_tags(item_id)))

            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            report.written += 1
            if progress is not None and index % 200 == 0:
                progress(index, len(paired))

    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "imported",
                "source": source or out_dir.name,
                "noisy_dir": str(noisy_dir),
                "clean_dir": str(clean_dir),
                "transcript_dir": str(transcripts) if transcripts else None,
                "sample_rate": sample_rate,
                "items": report.written,
                "skipped_unpaired": len(report.skipped_unpaired),
                "missing_transcript": len(report.missing_transcript),
                "manifest_sha256": _digest(out_dir / "manifest.jsonl"),
                "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
