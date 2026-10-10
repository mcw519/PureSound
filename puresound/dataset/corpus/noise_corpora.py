"""Public noise corpora -> 16 kHz noise folders, filtered for licence and for voices.

Each corpus is listed into :class:`AudioRecord` s with the tags that decide whether
a clip may be used (licence, labels, scene), filtered, then converted with the same
resampler as everything else. What stays is written as a folder the synthesis
pipeline draws from (`augmentation_noise.noise_sources`), plus an inventory with
the tags and, for attribution licences, an attribution list.

What is filtered here, and why:

* **Licence.** FSD50K licenses clip by clip; about 15 % are CC BY-NC or Sampling+,
  which a commercial model cannot train on. Only CC0 and CC BY are kept.
* **Labelled voices.** Noise suppression keeps every voice in its input, so a noise
  clip that is somebody talking teaches deletion. Clips labelled anywhere in
  AudioSet's *Human voice* subtree (speech, singing, shouting, laughter, crying)
  are dropped by label. Crowd and chatter are kept on purpose -- unintelligible
  babble is the restaurant/station noise we are short of -- and go through
  `speech_screen`, like every scene recording, to catch the talkers labels miss.
* **Vocals in music.** MUSAN marks which music tracks have vocals; those go.
  MUSAN's speech part is LibriVox, the source of our LibriTTS test speakers, and
  is never used.
* **Very short clips** are tiled to the row length by the pipeline; a 0.5 s knock
  tiled over 6 s is a metronome, not a noise. ``--min-seconds`` drops them.

Run::

    python -m puresound.dataset.corpus.noise_corpora fsd50k /path/to/audio/FSD50K \\
        --dest-root /path/to/training_set/ns_noise/fsd50k
    python -m puresound.dataset.corpus.noise_corpora cochlscene /path/to/audio/CochlScene \\
        --dest-root /path/to/training_set/ns_noise/cochlscene
    python -m puresound.dataset.corpus.noise_corpora musan /path/to/audio/musan \\
        --dest-root /path/to/training_set/ns_noise/musan
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from .records import AudioRecord, write_inventory
from .resample import build_resampled_tree
from .scan import audio_info, sanitize_id


#: FSD50K labels (with their ancestors, as the ground truth ships them) that mark a
#: clip as a voice. ``Human_group_actions`` (Crowd, Chatter, Applause) is NOT here.
FSD50K_VOICE_LABELS = frozenset({
    "Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Shout", "Yell",
    "Screaming", "Whispering", "Laughter", "Giggle", "Chuckle_and_chortle", "Singing",
    "Male_singing", "Female_singing", "Crying_and_sobbing", "Baby_cry_and_infant_cry",
})

#: Licences a commercial model can train on, by the URL FSD50K stores.
COMMERCIAL_LICENSES = ("publicdomain/zero/", "licenses/by/")


def commercial_ok(license_url: str) -> bool:
    url = license_url.lower()
    return any(marker in url for marker in COMMERCIAL_LICENSES) and "-nc" not in url and "sampling" not in url


def fsd50k_records(root: Path) -> tuple[list[AudioRecord], Counter]:
    """Kept clips and a tally of why the others went."""
    meta_dir = root / "FSD50K.metadata"
    truth_dir = root / "FSD50K.ground_truth"
    audio_dirs = {"dev": root / "FSD50K.dev_audio", "eval": root / "FSD50K.eval_audio"}
    info: dict[str, dict[str, Any]] = {}
    for split in ("dev", "eval"):
        info.update(json.loads((meta_dir / f"{split}_clips_info_FSD50K.json").read_text()))

    kept: list[AudioRecord] = []
    why: Counter = Counter()
    for split in ("dev", "eval"):
        with open(truth_dir / f"{split}.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                fname = row["fname"]
                labels = row["labels"].split(",")
                clip = info.get(fname, {})
                if not commercial_ok(clip.get("license", "")):
                    why["licence"] += 1
                    continue
                if FSD50K_VOICE_LABELS.intersection(labels):
                    why["voice label"] += 1
                    continue
                path = audio_dirs[split] / f"{fname}.wav"
                if not path.is_file():
                    why["missing audio"] += 1
                    continue
                kept.append(_record(path, f"fsd50k_{fname}", {
                    "labels": labels, "license": clip.get("license"),
                    "uploader": clip.get("uploader"), "title": clip.get("title"),
                    "freesound_id": fname, "split": split,
                }))
    return kept, why


def cochlscene_records(root: Path) -> tuple[list[AudioRecord], Counter]:
    """Every clip, tagged with its scene (``<Split>/<Scene>/<file>.{wav,flac}``).

    The Zenodo release is WAV; the Hugging Face mirror (``yotarokubo/CochlScene``)
    ships the same clips as FLAC, which is how it is laid out here.
    """
    kept: list[AudioRecord] = []
    why: Counter = Counter()
    paths = [
        p for split in ("Train", "Val", "Test") if (root / split).is_dir()
        for p in sorted((root / split).rglob("*")) if p.suffix.lower() in {".wav", ".flac"}
    ]
    for path in paths:
        scene = path.parent.name
        kept.append(_record(path, f"cochl_{sanitize_id(path.stem)}", {
            "scene": scene, "split": path.parent.parent.name, "license": "CC BY-SA 3.0",
        }))
        why[f"scene:{scene}"] += 1
    return kept, why


def musan_records(root: Path) -> tuple[list[AudioRecord], Counter]:
    """MUSAN noise, and the music tracks annotated as having no vocals."""
    base = root / "musan" if (root / "musan").is_dir() else root
    kept: list[AudioRecord] = []
    why: Counter = Counter()
    for path in sorted((base / "noise").rglob("*.wav")):
        kept.append(_record(path, f"musan_{sanitize_id(path.stem)}", {"part": "noise"}))
    vocals: dict[str, str] = {}
    for listing in (base / "music").rglob("ANNOTATIONS"):
        for line in listing.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 3:
                vocals[fields[0]] = fields[2]  # <file> <genre> <vocals Y/N> <artist>
    for path in sorted((base / "music").rglob("*.wav")):
        flag = vocals.get(path.stem)
        if flag is None:
            why["music without annotation"] += 1
            continue
        if flag.upper() != "N":
            why["music with vocals"] += 1
            continue
        kept.append(_record(path, f"musan_{sanitize_id(path.stem)}", {"part": "music"}))
    why["speech part never used"] += len(list((base / "speech").rglob("*.wav")))
    return kept, why


def _record(path: Path, uttid: str, tags: dict[str, Any]) -> AudioRecord:
    sample_rate, samples, _, channels = audio_info(path)
    return AudioRecord(
        uttid=uttid, spkid=uttid, gender="None", path=path, length=int(samples),
        sample_rate=int(sample_rate), channels=int(channels), tags=tags,
    )


CORPORA: dict[str, Callable[[Path], tuple[list[AudioRecord], Counter]]] = {
    "fsd50k": fsd50k_records,
    "cochlscene": cochlscene_records,
    "musan": musan_records,
}


def needs_attribution(tags: dict[str, Any]) -> bool:
    license_text = str(tags.get("license", ""))
    return "licenses/by" in license_text or "CC BY" in license_text


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    lister = CORPORA[args.corpus]
    records, why = lister(root)
    short = [r for r in records if r.length / r.sample_rate < args.min_seconds]
    records = [r for r in records if r.length / r.sample_rate >= args.min_seconds]
    why[f"shorter than {args.min_seconds}s"] += len(short)
    hours = sum(r.length / r.sample_rate for r in records) / 3600
    print(f"{args.corpus}: keeping {len(records)} clip(s), {hours:.1f} h")
    for reason, count in why.most_common():
        print(f"  {count:7d}  {reason}")
    if args.dry_run or not records:
        return 0 if records else 1

    dest_root = Path(args.dest_root).expanduser().resolve()
    converted, report = build_resampled_tree(
        records, source_root=root, dest_root=dest_root, target_sample_rate=16000,
        jobs=args.jobs, progress=lambda done, total: print(f"  {done}/{total}", flush=True) if done % 5000 == 0 else None,
    )
    print(f"  {report.summary()}")
    inventory = dest_root.parent / f"{dest_root.name}.inventory.jsonl"
    write_inventory(inventory, converted)
    print(f"inventory -> {inventory}")
    attributions = [r.tags for r in converted if needs_attribution(r.tags)]
    if attributions:
        credits = dest_root.parent / f"{dest_root.name}.attribution.jsonl"
        with open(credits, "w", encoding="utf-8") as handle:
            for tags in attributions:
                handle.write(json.dumps(tags, ensure_ascii=False) + "\n")
        print(f"attribution list ({len(attributions)} clip(s)) -> {credits}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.noise_corpora",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("corpus", choices=sorted(CORPORA))
    parser.add_argument("root")
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only list and tally.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
