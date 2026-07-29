"""Dump tensorboard scalar curves to a compact ASCII summary."""

import argparse
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print(f"need tensorboard: {e}", file=sys.stderr)
    raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("event_dir")
    p.add_argument("--keep", nargs="*", default=None, help="filter scalar tag substring")
    p.add_argument("--n_samples", type=int, default=20)
    args = p.parse_args()

    ea = EventAccumulator(args.event_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"Tags: {tags}")
    for tag in tags:
        if args.keep and not any(k in tag for k in args.keep):
            continue
        events = ea.Scalars(tag)
        n = len(events)
        if n == 0:
            continue
        idx_show = [int(i * (n - 1) / max(args.n_samples - 1, 1)) for i in range(min(args.n_samples, n))]
        print(f"\n--- {tag} (N={n}) ---")
        for i in idx_show:
            e = events[i]
            print(f"  step={e.step:>6} value={e.value:+.4f}")


if __name__ == "__main__":
    main()
