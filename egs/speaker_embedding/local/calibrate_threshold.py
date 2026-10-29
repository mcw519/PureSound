"""Calibrate the speaker models' cosine threshold on LibriSpeech-style trials.

    uv run python egs/speaker_embedding/local/calibrate_threshold.py \\
        LibriSpeech/test-clean --speakers LibriSpeech/SPEAKERS.TXT \\
        --out model_zoo/benchmarks/speaker_threshold_librispeech_test_clean.json

Every utterance is embedded once through the zoo runtime, as the web app and
the CLI run it (16 kHz, levelled to the model's target dBFS), and every pair of
utterances is a trial: the same reader is a target trial, two readers a
non-target one. Trials are scored on raw cosine similarity, the scale the
runtime compares its threshold with, and a trial is accepted when its score is
above the threshold.

For each model the report gives the equal error rate (EER) and the threshold
at it, the thresholds at 1% and 0.1% false acceptance, and what the model's
current default threshold does. Harder subsets come with it: target trials
across chapters (recorded apart), non-target trials within one gender, and
every utterance cut to its first ``--crop`` seconds, as a short phone
recording would be.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from puresound.inference import ModelZoo, load_model

MODELS = ("speaker-verification-ps-spk-v1-1", "speaker-verification-ps-spk-v1")


def readers(speakers_txt: Path) -> dict[str, str]:
    """Reader id -> sex, from LibriSpeech's SPEAKERS.TXT."""
    sexes = {}
    for line in speakers_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        fields = [field.strip() for field in line.split("|")]
        sexes[fields[0]] = fields[1]
    return sexes


def error_rates(target: np.ndarray, nontarget: np.ndarray, threshold: float) -> tuple[float, float]:
    """(false acceptance, false rejection) at a threshold."""
    return float(np.mean(nontarget > threshold)), float(np.mean(target <= threshold))


def equal_error(target: np.ndarray, nontarget: np.ndarray) -> dict[str, float]:
    """The threshold where false acceptance meets false rejection, by bisection."""
    target, nontarget = np.sort(target), np.sort(nontarget)
    low, high = -1.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        far = 1 - np.searchsorted(nontarget, middle, side="right") / nontarget.size
        frr = np.searchsorted(target, middle, side="right") / target.size
        low, high = (middle, high) if far > frr else (low, middle)
    return {"eer": round(float((far + frr) / 2), 5), "threshold": round(middle, 4)}


def at_false_acceptance(target: np.ndarray, nontarget: np.ndarray, rate: float) -> dict[str, float]:
    threshold = float(np.quantile(nontarget, 1 - rate))
    return {"threshold": round(threshold, 4), "frr": round(float(np.mean(target <= threshold)), 5)}


def summary(target: np.ndarray, nontarget: np.ndarray, current: float) -> dict:
    far, frr = error_rates(target, nontarget, current)
    return {
        "trials": {"target": int(target.size), "nontarget": int(nontarget.size)},
        **equal_error(target, nontarget),
        "far_1pct": at_false_acceptance(target, nontarget, 0.01),
        "far_0.1pct": at_false_acceptance(target, nontarget, 0.001),
        "at_current_threshold": {"threshold": current, "far": round(far, 5), "frr": round(frr, 5)},
        "target_scores": np.percentile(target, [1, 10, 50]).round(4).tolist(),
        "nontarget_scores": np.percentile(nontarget, [50, 90, 99, 99.9]).round(4).tolist(),
    }


def trials(vectors: np.ndarray, speaker: np.ndarray, chapter: np.ndarray, sex: np.ndarray, current: float) -> dict:
    scores = vectors @ vectors.T
    upper = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    same_reader = speaker[:, None] == speaker[None, :]
    same_chapter = chapter[:, None] == chapter[None, :]
    same_sex = sex[:, None] == sex[None, :]
    target, nontarget = scores[upper & same_reader], scores[upper & ~same_reader]
    return {
        "all": summary(target, nontarget, current),
        "across_chapters": summary(scores[upper & same_reader & ~same_chapter], nontarget, current),
        "same_sex_impostors": summary(target, scores[upper & ~same_reader & same_sex], current),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="LibriSpeech split folder: <reader>/<chapter>/*.flac")
    parser.add_argument("--speakers", type=Path, required=True, help="LibriSpeech SPEAKERS.TXT")
    parser.add_argument("--out", type=Path, required=True, help="report JSON")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--crop", type=float, default=3.0, help="seconds for the short-recording subset")
    args = parser.parse_args()

    files = sorted(args.data.glob("*/*/*.flac"))
    sexes = readers(args.speakers)
    speaker = np.array([path.parts[-3] for path in files])
    chapter = np.array([f"{path.parts[-3]}-{path.parts[-2]}" for path in files])
    sex = np.array([sexes[reader] for reader in speaker])
    audio = [sf.read(path, dtype="float32")[0] for path in files]
    rate = sf.info(files[0]).samplerate
    long_enough = np.array([clip.size >= args.crop * rate for clip in audio])

    zoo = ModelZoo.default()
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "name": args.data.name, "utterances": len(files), "readers": int(len(set(speaker))),
            "female": int(len({s for s, x in zip(speaker, sex) if x == "F"})),
            "male": int(len({s for s, x in zip(speaker, sex) if x == "M"})),
            "hours": round(sum(clip.size for clip in audio) / rate / 3600, 2),
        },
        "scale": "raw cosine similarity; a trial is accepted when it is above the threshold",
        "models": {},
    }
    for model_id in args.models:
        runtime = load_model(model_id, provider="cpu")
        current = float(zoo.get(model_id).parameters["threshold"].default)
        started = time.perf_counter()

        def embed(clips):
            vectors = np.stack([runtime.embed(clip) for clip in clips])
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        full = embed(audio)
        cropped = embed([clip[: int(args.crop * rate)] for clip, keep in zip(audio, long_enough) if keep])
        report["models"][model_id] = {
            "current_threshold": current,
            "full_utterances": trials(full, speaker, chapter, sex, current),
            f"first_{args.crop:g}s": {
                "utterances": int(long_enough.sum()),
                **trials(cropped, speaker[long_enough], chapter[long_enough], sex[long_enough], current),
            },
            "seconds_to_embed": round(time.perf_counter() - started, 1),
        }
        print(model_id, json.dumps({k: v["all"] for k, v in report["models"][model_id].items() if isinstance(v, dict) and "all" in v}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
