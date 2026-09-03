"""Re-issue the benchmark's eval recipes for a non-default inter path.

Every stage of run_full_benchmark.sh builds its model from a recipe of its own,
and all of them declare the shipped LSTM backbone. Point one at a checkpoint
whose backbone differs and the loader keeps going: the extra checkpoint keys are
warned about and dropped, the missing module stays at its random init, and the
benchmark silently scores a DIFFERENT MODEL than the one that was trained.

This writes copies of those recipes with the architecture knobs of a training
recipe injected into `backbone_args`, so a round that changes the backbone can
be benchmarked without hand-editing six files. Only the knobs move; everything
else in each eval recipe (its dataset, bank, augmentation) is untouched.

    uv run python scripts/make_arch_eval_configs.py \
        --from config/exp/train_dpcrn_v14_mambaparallel.yaml --out-dir /tmp/cfg_v14

Pair it with preflight_ckpt_recipe.py, which proves the result actually loads
the checkpoint whole.
"""
import argparse
import pathlib
import re
import sys

# The recipes run_full_benchmark.sh passes to its stages.
BENCH_CONFIGS = (
    "config/infer_dpcrn.yaml",                          # 1 field, 6 dawn, 9 turn-taking
    "config/exp/eval_indomain_phase1.yaml",             # 2
    "config/exp/eval_targetabsent_probe.yaml",          # 3
    "config/exp/eval_targetabsent_probe_high.yaml",     # 4
    "config/exp/eval_targetabsent_probe_boundary.yaml", # 5
    "config/exp/eval_but_real.yaml",                    # 7a, 7b, 8
)

# backbone_args entries that describe the ARCHITECTURE rather than the data.
KNOBS = ("inter_type", "mamba_args")


def extract_knobs(text: str) -> str:
    """The knob lines of a training recipe's backbone_args, indentation kept."""
    out, keep, indent = [], False, None
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped.split(":")[0]
        if not stripped or stripped.startswith("#"):
            if keep:
                out.append(line)
            continue
        current = len(line) - len(line.lstrip())
        if keep and current <= indent and key not in KNOBS:
            keep = False
        if key in KNOBS:
            keep, indent = True, current
        if keep:
            out.append(line)
    return "\n".join(out)


def inject(text: str, knobs: str) -> str:
    if re.search(r"^\s*inter_type:", text, re.M):
        raise SystemExit("target recipe already sets inter_type -- refusing to guess")
    anchor = re.search(r"^(\s*)rnn_hidden:.*$", text, re.M)
    if anchor is None:
        raise SystemExit("no rnn_hidden line to anchor the injection to")
    return text[: anchor.end()] + "\n" + knobs + text[anchor.end():]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", required=True,
                    help="training recipe carrying the architecture knobs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", nargs="*", default=list(BENCH_CONFIGS))
    args = ap.parse_args()

    knobs = extract_knobs(pathlib.Path(args.src).read_text())
    if not knobs.strip():
        raise SystemExit(f"{args.src} declares none of {KNOBS} -- nothing to inject")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in args.configs:
        dst = out_dir / pathlib.Path(rel).name
        dst.write_text(inject(pathlib.Path(rel).read_text(), knobs))
        print(dst)


if __name__ == "__main__":
    sys.exit(main())
