"""Freeze a synthetic (noisy, clean) test set from the recipe's own synthesis.

Why freeze rather than score the dynamic validation loader directly: a set that is
re-synthesised on every run measures the synthesis as much as the model, so two
checkpoints scored at different times are not comparable. Freezing pins the audio;
``provenance.json`` pins the commit, recipe content and seed that produced it.

Why drive the recipe's pipeline rather than a separate mixer: a test set built by
its own code path tests that code path, not the one that trains. There is one
synthesis here, and this runs it with a fixed seed.

Every item's SNR is **measured**, not taken from the config that asked for it --
``noisy - clean`` is exactly the noise that ended up in the mixture. That is what
makes the SNR bands in the diagnostic layer mean something.

Run::

    python -m puresound.evaluation.tools.build_eval_set config/eval/testset.yaml \\
        --out-dir data_report/ns_testset --n 500 --seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.dataset.corpus.paired import (  # noqa: F401 -- re-exported for callers
    SNR_BANDS,
    band_of,
    measured_snr_db,
    rms_dbfs,
)
from puresound.evaluation.records import chain_commit


def recipe_sha256(path: str | Path) -> str:
    """Content identity of the recipe that synthesised a frozen set."""
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.build_eval_set",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("config_path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--task", default="noise_suppression")
    parser.add_argument(
        "--min-active",
        type=float,
        default=1e-4,
        help="Drop an item whose clean target is essentially silent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import lightning as L

    L.seed_everything(seed=args.seed, workers=True)

    recipe = load_recipe(args.config_path, expected_task=args.task)

    if args.task == "voice_isolation":
        from puresound.task.voice_isolation import (
            VoiceIsolationCollateFunc as Collate,
            VoiceIsolationDataset as Dataset,
        )
    else:
        from puresound.task.ns import (
            NoiseSuppressionCollateFunc as Collate,
            NoiseSuppressionDataset as Dataset,
        )

    from puresound.system import runner

    _, valid_loader = runner.build_dataloaders(
        dataset_cls=Dataset, collate_fn=Collate(), recipe=recipe
    )

    sample_rate = int(recipe.dataset.target_sample_rate or 16000)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance.json"
    # A failed rebuild must not leave an old identity next to a new partial manifest.
    provenance_path.unlink(missing_ok=True)

    saved = 0
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for batch in valid_loader:
            for row in range(batch["clean_speech"].shape[0]):
                clean = batch["clean_speech"][row].reshape(1, -1).cpu()
                noisy = batch["noisy_speech"][row].reshape(1, -1).cpu()
                if float(clean.abs().max()) < args.min_active:
                    continue

                item_id = f"ns{saved:05d}"
                AudioIO.save(noisy, str(out_dir / f"{item_id}_mix.wav"), sample_rate)
                AudioIO.save(clean, str(out_dir / f"{item_id}_clean.wav"), sample_rate)

                snr = measured_snr_db(clean, noisy)
                manifest.write(
                    json.dumps(
                        {
                            "id": item_id,
                            "mix": f"{item_id}_mix.wav",
                            "clean": f"{item_id}_clean.wav",
                            "sample_rate": sample_rate,
                            "samples": int(clean.numel()),
                            "duration_s": round(clean.numel() / sample_rate, 3),
                            "snr_db": round(snr, 3),
                            "snr_band": band_of(snr),
                            "mix_rms_dbfs": round(rms_dbfs(noisy), 2),
                            "clean_rms_dbfs": round(rms_dbfs(clean), 2),
                        }
                    )
                    + "\n"
                )
                saved += 1
                if saved >= args.n:
                    break
            if saved >= args.n:
                break

    provenance = {
        "chain_commit": chain_commit(),
        "recipe": str(Path(args.config_path).expanduser().resolve()),
        "recipe_sha256": recipe_sha256(args.config_path),
        "seed": args.seed,
        "sample_rate": sample_rate,
        "requested_items": args.n,
        "saved_items": saved,
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {saved} (mix, clean) pair(s) + manifest/provenance -> {out_dir}")
    print(f"seed={args.seed}  sample_rate={sample_rate}  recipe={args.config_path}")
    if saved < args.n:
        print(f"NOTE: asked for {args.n}; the loader ran out first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
