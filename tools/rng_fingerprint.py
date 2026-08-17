#!/usr/bin/env python
"""Prove a refactor did not change what the synthesis path produces.

`test/test_utils/test_synthesis_fingerprint.py` guards the invariants that hold
at any point in time -- a default lives in one place, a seeded item is a pure
function of its seed. It cannot catch a change that moves *everything*
consistently: delete a stage, reorder two RNG draws, and both of its recipes
shift together and it still passes.

This is the tool for that case. It is a before/after comparison, so it needs a
baseline captured on the unmodified tree:

    git stash                                     # or check out the base commit
    uv run python tools/rng_fingerprint.py CONFIG before.json
    git stash pop
    ...refactor...
    uv run python tools/rng_fingerprint.py CONFIG after.json
    uv run python tools/rng_fingerprint.py --compare before.json after.json

Point it at a real recipe. Anything the recipe leaves off is not covered --
a fingerprint over disabled blocks proves nothing about them -- so use the
config whose behaviour you are claiming to preserve, and say in the commit
message which one it was.

Note the ``PYTHONHASHSEED`` caveat is gone: the utterance pool is drawn in
metafile order (see `choose_an_utterance_by_speaker_name`), so runs compare
across processes without pinning it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from puresound.config import load_recipe  # noqa: E402

#: Only the noise-suppression lineage accepts a per-item seed in its key (the
#: 3-tuple the seeded sampler emits). Speaker-embedding and TSE take a plain
#: (speaker, sr) pair, so an item there is only reproducible if every RNG is
#: seeded around it -- do that rather than silently fingerprint noise.
PER_ITEM_SEED_TASKS = frozenset({"voice_isolation", "noise_suppression"})

TASK_DATASETS = {
    "voice_isolation": ("puresound.task.voice_isolation", "VoiceIsolationDataset"),
    "noise_suppression": ("puresound.task.ns", "NoiseSuppressionDataset"),
    "speaker_embedding": ("puresound.task.sv", "SpeakerEmbeddingDataset"),
    "target_speaker_extraction": (
        "puresound.task.tse",
        "TargetSpeakerExtractDataset",
    ),
}


def _digest(value) -> str:
    if torch.is_tensor(value):
        payload = value.detach().to(torch.float64).contiguous().numpy().tobytes()
    else:
        payload = repr(value).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def capture(config_path: str, n_items: int) -> dict:
    import importlib

    recipe = load_recipe(config_path)
    module_name, class_name = TASK_DATASETS[recipe.task]
    dataset_cls = getattr(importlib.import_module(module_name), class_name)

    corpus = recipe.dataset
    dataset = dataset_cls(
        metafile_path=corpus.train_metafile,
        min_utt_length_in_seconds=corpus.filter_min_utterance_length,
        min_utts_in_each_speaker=corpus.filter_min_utterance_per_speaker,
        target_sr=corpus.target_sample_rate,
        training_sample_length_in_seconds=corpus.training_length_seconds,
        audio_gain_normalized_to=corpus.gain_normalized_to,
        dataset_role="train",
        pipeline_role=corpus.train_pipeline_role,
        **recipe.augmentation_kwargs(),
        # The one argument that is not an augmentation block and not shared.
        **(
            {"enroll_speech_args": recipe.enroll_speech}
            if recipe.task == "target_speaker_extraction"
            else {}
        ),
    )

    speakers = sorted(dataset.total_spks)
    per_item_seed = recipe.task in PER_ITEM_SEED_TASKS
    out = {}
    for index in range(n_items):
        seed = 1000 + index
        speaker = speakers[index % len(speakers)]
        if per_item_seed:
            key = (speaker, corpus.target_sample_rate, seed)
        else:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            key = (speaker, corpus.target_sample_rate)
        sample = dataset[key]
        out[f"item{index:03d}"] = {k: _digest(v) for k, v in sorted(sample.items())}
    return out


def compare(before_path: str, after_path: str) -> int:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    total = sum(len(v) for v in before.values())
    drift = [
        (group, key)
        for group in before
        for key in before[group]
        if key not in after.get(group, {}) or before[group][key] != after[group][key]
    ]
    if not drift:
        print(f"identical: {total} hashes over {len(before)} items")
        return 0
    print(f"CHANGED: {len(drift)} / {total} hashes")
    for group, key in drift[:20]:
        print(f"  {group}.{key}")
    if len(drift) > 20:
        print(f"  ... and {len(drift) - 20} more")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", nargs="?", help="recipe to synthesise from")
    parser.add_argument("out_path", nargs="?", help="where to write the fingerprint")
    parser.add_argument("-n", "--n-items", type=int, default=32)
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two fingerprints"
    )
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not (args.config_path and args.out_path):
        parser.error("need CONFIG and OUT, or --compare BEFORE AFTER")

    fingerprint = capture(args.config_path, args.n_items)
    Path(args.out_path).write_text(json.dumps(fingerprint, indent=1, sort_keys=True))
    total = sum(len(v) for v in fingerprint.values())
    print(f"wrote {args.out_path}: {total} hashes over {len(fingerprint)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
