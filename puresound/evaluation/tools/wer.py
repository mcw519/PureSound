"""The deletion guardrail: does enhancement cost words?

This is the stage the quality metrics are worst at covering. A model that removes
the noise and part of the speech with it can hold its PESQ and lose the sentence,
and a weak recogniser can hide exactly that where a strong one reveals it. That is
why the recogniser here defaults to a large model: the point is to expose
over-suppression, not to be fast.

Three rules the numbers follow.

**Read the delta, not the rate.** A WER of 0.17 means nothing until you know the
unprocessed mixture scores 0.19 on the same cuts. Both are transcribed here and
the paired difference is what the verdict reads.

**Deletions are reported separately.** They are the failure mode with a direction:
substitutions and insertions can come from the recogniser having a bad day, but a
rising deletion rate against the same reference is the model removing speech.

**The reference is the corpus transcript, never the recogniser on clean audio.**
A recogniser's own output as ground truth measures agreement with itself, and
flatters whichever system sounds most like what it was trained on.

Run::

    python -m puresound.evaluation.tools.wer --set-dir data_report/vctk_demand_test \\
        --recipe config/infer.yaml --ckpt exp/.../epoch=59.ckpt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from puresound.audio.io import AudioIO
from puresound.evaluation.records import StageResult, write_stage
from puresound.evaluation.statistics import paired_bootstrap_ci, verdict
from puresound.evaluation.systems import (
    Passthrough,
    PrecomputedSystem,
    System,
    load_system,
    run_system,
)


def normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace.

    Scoring raw text would count a comma as an error, which says nothing about
    whether the words survived.
    """
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    # An apostrophe is kept only between two letters or digits ("don't"); a
    # quotation mark around a word is punctuation, not part of the word.
    text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    return " ".join(text.split())


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Levenshtein alignment counts over words: S, D, I and the reference length."""
    ref, hyp = reference.split(), hypothesis.split()
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = np.zeros((rows, cols), dtype=np.int32)
    cost[:, 0] = np.arange(rows)
    cost[0, :] = np.arange(cols)
    # 0 = match/sub, 1 = deletion, 2 = insertion
    back = np.zeros((rows, cols), dtype=np.int8)
    back[:, 0] = 1
    back[0, :] = 2

    for i in range(1, rows):
        for j in range(1, cols):
            sub = cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            dele = cost[i - 1, j] + 1
            ins = cost[i, j - 1] + 1
            best = min(sub, dele, ins)
            cost[i, j] = best
            back[i, j] = 0 if best == sub else (1 if best == dele else 2)

    i, j = len(ref), len(hyp)
    counts = {"sub": 0, "del": 0, "ins": 0, "hit": 0, "ref_words": len(ref)}
    while i > 0 or j > 0:
        move = back[i, j]
        if move == 0:
            counts["hit" if ref[i - 1] == hyp[j - 1] else "sub"] += 1
            i, j = i - 1, j - 1
        elif move == 1:
            counts["del"] += 1
            i -= 1
        else:
            counts["ins"] += 1
            j -= 1
    return counts


def rates(counts: dict[str, int]) -> dict[str, float]:
    n = max(counts["ref_words"], 1)
    return {
        "wer": (counts["sub"] + counts["del"] + counts["ins"]) / n,
        "del": counts["del"] / n,
        "ins": counts["ins"] / n,
        "sub": counts["sub"] / n,
    }


def hypothesis_row(
    item_id: str, system: str, reference: str, hypothesis: str
) -> dict[str, Any]:
    """One transcription, with its edit counts, for ``--hypotheses``.

    The stage reports rates; a rate cannot say *what* the recogniser wrote. When
    insertions double on an enhanced signal, the question is whether the words
    are hallucinated over suppressed speech or mis-segmented real ones, and only
    the hypothesis text answers it.
    """
    counts = edit_counts(reference, hypothesis)
    return {
        "id": item_id,
        "system": system,
        "reference": reference,
        "hypothesis": hypothesis,
        **counts,
        **rates(counts),
    }


#: A hypothesis this many times longer than its reference is a recogniser loop,
#: not a transcription. Whisper answers a segment it cannot parse by repeating a
#: phrase, and the looped item's insertion rate is then orders of magnitude above
#: the corpus mean -- so a handful of loops can decide a corpus delta and most of
#: the width of its confidence interval.
LOOP_RATIO = 1.5


def loop_count(rows: Sequence[dict[str, Any]], ratio: float = LOOP_RATIO) -> int:
    """How many hypotheses are long enough past their reference to be loops."""
    return sum(
        1
        for row in rows
        if len(str(row["hypothesis"]).split()) > ratio * max(int(row["ref_words"]), 1)
    )


def read_manifest(set_dir: Path) -> list[dict[str, Any]]:
    path = set_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing.")
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with_text = [item for item in items if item.get("transcript")]
    if not with_text:
        raise ValueError(
            f"{path} has no `transcript` field. Import the set with "
            "`python -m puresound.dataset.corpus.vctk_demand testset ...` (it carries the "
            "transcripts), or point at a set that has them."
        )
    return with_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.wer",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--set-dir", required=True)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--ckpt", default=None, help="Omit to transcribe the baseline alone.")
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-blend", type=float, default=1.0)
    parser.add_argument("--precomputed", default=None, metavar="DIR",
                        help="Score audio another system already produced, one file per utterance named by the input stem.")
    parser.add_argument("--precomputed-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--asr-model", default="large-v3",
        help="faster-whisper model. A small one hides over-suppression; that is the "
        "whole thing this stage exists to see.",
    )
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument(
        "--asr-temperature", type=float, default=0.0,
        help="Decoding temperature. faster-whisper's default is a fallback ladder "
        "(0, 0.2, ..., 1.0) that SAMPLES when a segment fails its quality checks, "
        "so identical audio transcribes differently run to run. A paired gate needs "
        "the recogniser to be a function of the audio; "
        "0.0 makes it one.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.0,
        help="WER regression the stage may absorb, e.g. 0.01 for a do-no-harm monitor.",
    )
    parser.add_argument(
        "--band", action="append", default=["snr_band"],
        help="Manifest field to break the per-utterance rates down by; repeatable. "
        "Noise suppression behaves differently at 0 dB and at 15 dB, and one corpus "
        "WER says nothing about where a model earns or loses its words.",
    )
    parser.add_argument(
        "--hypotheses",
        default=None,
        metavar="PATH",
        help="Also write every transcription as JSONL (id, system, reference, "
        "hypothesis, edit counts), so a rate can be traced to the words behind it.",
    )
    parser.add_argument("--stage-name", default="wer")
    parser.add_argument("--role", default="gate", choices=["gate", "monitor"])
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_dir = Path(args.set_dir).expanduser().resolve()
    items = read_manifest(set_dir)
    if args.limit:
        items = items[: args.limit]

    from faster_whisper import WhisperModel

    asr = WhisperModel(
        args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type
    )

    def transcribe(wav: torch.Tensor, sample_rate: int) -> str:
        audio = wav.reshape(-1).float().numpy()
        segments, _ = asr.transcribe(
            audio, language="en", beam_size=5, temperature=args.asr_temperature
        )
        return normalise(" ".join(segment.text for segment in segments))

    baseline = Passthrough()
    systems: list[System] = [baseline]
    if args.ckpt and args.precomputed:
        print("--ckpt and --precomputed are two systems; score them in two runs.", file=sys.stderr)
        return 2
    if args.ckpt:
        systems.append(
            load_system(args.recipe, args.ckpt, task=args.task, device=args.device,
                        dry_blend=args.dry_blend, name="model")
        )
    if args.precomputed:
        systems.append(PrecomputedSystem(
            Path(args.precomputed), args.precomputed_name or Path(args.precomputed).name
        ))

    print(f"{len(items)} utterance(s), {args.asr_model} on {args.asr_device}")
    per_system: dict[str, list[dict[str, float]]] = {s.name: [] for s in systems}
    per_system_rows: dict[str, list[dict[str, Any]]] = {s.name: [] for s in systems}
    hypotheses = None
    if args.hypotheses:
        hypotheses_path = Path(args.hypotheses).expanduser()
        hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
        hypotheses = hypotheses_path.open("w", encoding="utf-8")
    for index, item in enumerate(items, start=1):
        sample_rate = int(item["sample_rate"])
        mix, _ = AudioIO.open(str(set_dir / item["mix"]), resample_to=sample_rate)
        mix = mix.reshape(1, -1)
        reference = normalise(item["transcript"])
        for system in systems:
            hypothesis = transcribe(run_system(system, Path(item["mix"]).stem, mix, sample_rate), sample_rate)
            row = hypothesis_row(str(item["id"]), system.name, reference, hypothesis)
            per_system[system.name].append({k: row[k] for k in ("wer", "del", "ins", "sub")})
            per_system_rows[system.name].append(row)
            if hypotheses is not None:
                hypotheses.write(json.dumps(row, ensure_ascii=False) + "\n")
        if index % 50 == 0:
            print(f"  {index}/{len(items)}", flush=True)
    if hypotheses is not None:
        hypotheses.close()
        print(f"wrote hypotheses -> {args.hypotheses}")

    # Printed always, because a corpus mean silently decided by a few looped
    # transcriptions is a number about the recogniser, not about the system.
    loops = {name: loop_count(rows) for name, rows in per_system_rows.items()}
    if any(loops.values()):
        print(
            "\n-- recogniser loops (hypothesis > "
            f"{LOOP_RATIO}x its reference; each one moves the corpus mean) --"
        )
        for name, count in loops.items():
            print(f"  {name:<16s} {count:3d} / {len(items)}")
        print(
            "  These are recogniser failures, not per-word evidence. Re-read the "
            "delta with them dropped before ranking; `--hypotheses` writes the text."
        )

    def corpus(scores: Sequence[dict[str, float]], key: str) -> float:
        return float(np.mean([s[key] for s in scores]))

    print(f"\n-- {baseline.name} --")
    for key in ("wer", "del", "ins", "sub"):
        print(f"  {key:<4} {corpus(per_system[baseline.name], key):.4f}")

    from collections import defaultdict

    for field in args.band:
        buckets: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
            lambda: {name: [] for name in per_system}
        )
        for index, item in enumerate(items):
            band = str(item.get(field, "unknown"))
            for name in per_system:
                buckets[band][name].append(per_system[name][index])
        print(f"\n-- wer / sub by {field} (absolute per system) --")
        header = "".join(f"{name:>18s}" for name in per_system)
        print(f"  {'band':<12s}{'n':>5s}{header}")
        for band in sorted(buckets):
            n = len(buckets[band][baseline.name])
            cells = "".join(
                f"{corpus(buckets[band][name], 'wer'):9.4f}/{corpus(buckets[band][name], 'sub'):.4f}"
                for name in per_system
            )
            print(f"  {band:<12s}{n:>5d}{cells}")

    if len(systems) == 1:
        print("\nNo checkpoint given: this is the reference every result is read against.")
        return 0

    treatment_name = next(s.name for s in systems if s.name != baseline.name)
    treatment, reference_scores = per_system[treatment_name], per_system[baseline.name]
    print("\n-- model, paired against the unprocessed mixture --")
    stages: list[StageResult] = []
    # Deletions are the one-sided guardrail; WER and substitutions only report.
    # A WER difference between checkpoints can be statistically resolvable and
    # still only a few words per thousand -- resolvable is not important -- so WER
    # must not veto or rank a release.
    for key, role, tolerance in (
        ("wer", "monitor", 0.0),
        ("del", args.role, args.tolerance),
        ("ins", "monitor", 0.0),
        ("sub", "monitor", 0.0),
    ):
        after = [s[key] for s in treatment]
        before = [s[key] for s in reference_scores]
        interval = paired_bootstrap_ci(after, before, aggregate=np.mean)
        decision = verdict(interval, direction="lower_is_better", tolerance=tolerance)
        stage = StageResult.from_interval(
            f"{args.stage_name}.{key}", metric=key, role=role,
            value=corpus(treatment, key), baseline=corpus(reference_scores, key),
            interval=interval, verdict=decision, direction="lower_is_better",
            notes=f"{set_dir.name}, {args.asr_model}, reference = corpus transcript",
        )
        stages.append(stage)
        print("  " + stage.line())

    if args.out:
        write_stage(args.out, stages)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
