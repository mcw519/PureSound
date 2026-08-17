"""How much does a checkpoint suppress REAL far-field speech, by distance?

Synthetic far-only probes over-estimate this by more than 20 dB, because clean speech
convolved with an RIR is missing everything else a capture chain does (transducer
non-linearity, directivity, noise floor, level and spectral tilt). Corpora that replay
speech through a loudspeaker in a real room and re-record it do contain all of it, so
this measures suppression on those recordings instead.

Target behaviour: a voice beyond ~1 m is a suppress target (reduction should be very
negative), a voice within 1 m is keep (reduction ~0 dB).

Two corpora, same measurement:
  * ``--corpus voices``  VOiCES retransmitted recordings, bucketed by (room, mic).
    Speech-active spans come from an energy gate on the recording itself, so no
    alignment with the source is needed.
  * ``--corpus realman`` RealMAN val static recordings, bucketed by annotated
    speaker->array distance. Spans come from the sample-aligned direct-path
    reference (dp_speech), so scene noise cannot inflate them.

Reduction is energy on the model output over the same spans, relative to the input.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_far_suppression.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
        --corpus voices --voices-root /path/to/VOiCES --device cuda --per-bucket 40

    uv run python egs/voice_isolate/scripts/eval_far_suppression.py \
        egs/voice_isolate/config/infer_dpcrn.yaml --ckpt <ckpt> \
        --corpus realman --realman-val /path/to/realman/val --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

# Foreground-loudspeaker distance per (room, mic id) in the VOiCES release, metres.
# rm1/rm2 from the corpus README mic table; rm3/rm4 from recording_data distances.csv
# ("foreground" column), which is also the per-file source when it is available.
VOICES_MIC_DISTANCE_M = {
    ("rm1", 1): 0.97, ("rm1", 2): 0.97, ("rm1", 3): 1.83, ("rm1", 4): 1.83,
    ("rm1", 5): 3.02, ("rm1", 6): 3.02, ("rm1", 7): 0.74, ("rm1", 8): 0.74,
    ("rm1", 9): 1.47, ("rm1", 10): 1.91, ("rm1", 11): 1.91, ("rm1", 12): 3.30,
    ("rm2", 1): 2.03, ("rm2", 2): 2.03, ("rm2", 3): 3.33, ("rm2", 4): 3.33,
    ("rm2", 5): 5.79, ("rm2", 6): 5.79, ("rm2", 7): 0.74, ("rm2", 8): 0.74,
    ("rm2", 9): 2.77, ("rm2", 10): 3.25, ("rm2", 11): 3.25, ("rm2", 12): 2.95,
    ("rm3", 1): 1.70, ("rm3", 2): 1.70, ("rm3", 3): 3.71, ("rm3", 4): 3.71,
    ("rm3", 5): 7.14, ("rm3", 6): 7.14, ("rm3", 7): 1.47, ("rm3", 8): 1.47,
    ("rm3", 9): 2.24, ("rm3", 10): 4.47, ("rm3", 11): 4.47, ("rm3", 12): 6.65,
    ("rm4", 1): 1.83, ("rm4", 2): 1.83, ("rm4", 3): 4.24, ("rm4", 4): 4.24,
    ("rm4", 5): 9.83, ("rm4", 6): 9.83, ("rm4", 7): 1.80, ("rm4", 8): 1.80,
    ("rm4", 9): 3.25, ("rm4", 10): 4.39, ("rm4", 11): 4.45, ("rm4", 12): 9.65,
}

VOICES_FNAME_RE = re.compile(
    r"(?P<room>rm\d+)-(?P<distractor>[a-z]+)-sp(?P<spk>\d+)-ch(?P<ch>\d+)"
    r"-s(?:e)?g(?P<seg>\d+)-mc(?P<mic>\d+)-(?P<mtype>[a-z]+)-(?P<loc>[a-z]+)"
)

DISTANCE_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99.0)]


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = init_siso_model(
        load_recipe(config_path, expected_task="voice_isolation").model
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict, load_loss_func=False)
    else:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def active_spans(
    wav: torch.Tensor,
    sr: int,
    frame: int = 400,
    hop: int = 160,
    floor_offset_db: float = 12.0,
    min_span_sec: float = 0.2,
) -> list[tuple[int, int]]:
    """Speech-active sample spans from frame energy above the clip's own noise floor."""
    x = wav.view(-1)
    n_frames = max((x.shape[0] - frame) // hop + 1, 0)
    if n_frames < 10:
        return []
    frames = x[: (n_frames - 1) * hop + frame].unfold(0, frame, hop)
    rms_db = 10.0 * torch.log10(frames.square().mean(dim=1) + 1e-10)
    floor = torch.quantile(rms_db, 0.10)
    active = rms_db > (floor + floor_offset_db)
    spans: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start * hop, i * hop + frame))
            start = None
    if start is not None:
        spans.append((start * hop, x.shape[0]))
    return [(a, b) for a, b in spans if (b - a) >= int(min_span_sec * sr)]


def reduction_db(enh: torch.Tensor, raw: torch.Tensor, spans) -> float:
    n = min(enh.shape[-1], raw.shape[-1])
    e = r = 0.0
    for a, b in spans:
        b = min(b, n)
        if b <= a:
            continue
        e += float(enh[..., a:b].square().sum())
        r += float(raw[..., a:b].square().sum())
    if r <= 0.0:
        return float("nan")
    return 10.0 * torch.log10(torch.tensor(e / r + 1e-12)).item()


def quantiles(vals: list[float]) -> tuple[float, float, float]:
    t = torch.tensor(sorted(vals))
    return (
        torch.quantile(t, 0.25).item(),
        torch.quantile(t, 0.50).item(),
        torch.quantile(t, 0.75).item(),
    )


def bucket_label(dist: float) -> str:
    for lo, hi in DISTANCE_BUCKETS:
        if lo <= dist < hi:
            return f"{lo:.0f}-{hi:.0f}m" if hi < 99 else f"{lo:.0f}m+"
    return "unknown"


def collect_voices(args) -> list[dict]:
    """One item per retransmitted recording, bucketed by (room, mic)."""
    speech_root = Path(args.voices_root) / "distant-16k" / "speech"
    if not speech_root.is_dir():
        sys.exit(f"not found: {speech_root}")

    by_mic: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for wav_path in speech_root.rglob(f"*-{args.distractor}-*.wav"):
        m = VOICES_FNAME_RE.search(wav_path.name)
        if m is None:
            continue
        by_mic[(m.group("room"), int(m.group("mic")))].append(wav_path)
    if not by_mic:
        sys.exit(f"no files matching distractor={args.distractor} under {speech_root}")

    # A mic with no documented distance cannot enter any distance statistic, so it is
    # dropped before scoring rather than carried as an unusable bucket.
    scorable = {k: v for k, v in by_mic.items() if k in VOICES_MIC_DISTANCE_M}
    undocumented = sorted(k for k in by_mic if k not in VOICES_MIC_DISTANCE_M)

    print(f"# VOiCES buckets (distractor={args.distractor}):")
    for key in sorted(scorable):
        print(f"#   {key[0]} mc{key[1]:02d} d={VOICES_MIC_DISTANCE_M[key]:.2f}m  n={len(scorable[key])}")
    if undocumented:
        dropped = sum(len(by_mic[k]) for k in undocumented)
        print(f"# dropped {len(undocumented)} bucket(s) / {dropped} recording(s) with no documented "
              f"mic distance: " + ", ".join(f"{r} mc{m:02d}" for r, m in undocumented))
    if not scorable:
        sys.exit("no bucket has a documented mic distance")

    rng = random.Random(args.seed)
    items = []
    for key in sorted(scorable):
        files = scorable[key]
        picks = files if len(files) <= args.per_bucket else rng.sample(files, args.per_bucket)
        for wav_path in sorted(picks):
            m = VOICES_FNAME_RE.search(wav_path.name)
            items.append({
                "wav": wav_path,
                "span_ref": None,                       # spans from the recording itself
                "distance_m": VOICES_MIC_DISTANCE_M[key],
                "bucket": f"{key[0]} mc{key[1]:02d}",
                "info": m.group("loc"),
                "min_active_sec": 1.0,
            })
    return items


def collect_realman(args) -> list[dict]:
    """One item per val static recording, bucketed by annotated distance."""
    val_root = Path(args.realman_val)
    csv_path = val_root / "val_static_source_location.csv"
    if not csv_path.is_file():
        sys.exit(f"not found: {csv_path}")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            dist = float(row["distance"])
            if not 0.0 < dist < 50.0:      # -10000 sentinel / bad rows
                continue
            by_bucket[bucket_label(dist)].append({"filename": row["filename"], "distance": dist})

    print("# RealMAN static val rows with a valid distance, per bucket:")
    for lo, hi in DISTANCE_BUCKETS:
        b = f"{lo:.0f}-{hi:.0f}m" if hi < 99 else f"{lo:.0f}m+"
        print(f"#   {b}: {len(by_bucket.get(b, []))}")

    rng = random.Random(args.seed)
    items = []
    for lo, hi in DISTANCE_BUCKETS:
        b = f"{lo:.0f}-{hi:.0f}m" if hi < 99 else f"{lo:.0f}m+"
        pool = by_bucket.get(b, [])
        picks = pool if len(pool) <= args.per_bucket else rng.sample(pool, args.per_bucket)
        for row in picks:
            rel = row["filename"]
            rel = rel[4:] if rel.startswith("val/") else rel   # paths in the csv start with val/
            wav = val_root / rel.replace(".flac", f"_{args.channel}.flac")
            dp = val_root / rel.replace("ma_noisy_speech", "dp_speech")
            if not (wav.is_file() and dp.is_file()):
                continue
            items.append({
                "wav": wav,
                "span_ref": dp,                          # clean direct-path reference
                "distance_m": row["distance"],
                "bucket": b,
                "info": rel.split("/")[1],               # scene name
                "min_active_sec": 0.5,
            })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--corpus", required=True, choices=["voices", "realman"])
    parser.add_argument("--voices-root", help="VOiCES root (contains distant-16k/)")
    parser.add_argument("--realman-val", help="RealMAN val root (contains val_static_source_location.csv)")
    parser.add_argument("--distractor", default="none", choices=["none", "babb", "musi", "tele"],
                        help="VOiCES competing-sound condition; none = a lone far voice")
    parser.add_argument("--channel", default="CH0", help="RealMAN array channel")
    parser.add_argument("--per-bucket", type=int, default=40, help="recordings scored per bucket")
    parser.add_argument("--dry-blend", type=float, default=1.0,
                        help="inference over-suppression relief: enh*b + mix*(1-b)")
    parser.add_argument("--spec-floor", type=float, default=0.0,
                        help="clamp enhanced |bin| to >= floor * |mix bin|")
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None, help="write per-file rows as JSONL")
    parser.add_argument("--list-only", action="store_true", help="only print the bucket inventory")
    args = parser.parse_args()

    if args.corpus == "voices" and not args.voices_root:
        parser.error("--corpus voices needs --voices-root")
    if args.corpus == "realman" and not args.realman_val:
        parser.error("--corpus realman needs --realman-val")

    items = collect_voices(args) if args.corpus == "voices" else collect_realman(args)
    if args.list_only:
        return
    if not items:
        sys.exit("no scorable recordings found")

    device = torch.device(args.device)
    model = load_model(args.config_path, args.ckpt, device)

    rows: list[dict] = []
    seen_buckets: set[str] = set()
    # Rows are streamed as they are scored: a run interrupted after an hour of
    # inference still leaves everything it measured on disk.
    sink = open(args.out_json, "w") if args.out_json else None
    for item in items:
        raw, sr = AudioIO.open(f_path=str(item["wav"]), target_lvl=None, resample_to=16000)
        raw = raw.view(1, -1)
        if item["span_ref"] is None:
            span_src = raw
        else:
            ref, _ = AudioIO.open(f_path=str(item["span_ref"]), target_lvl=None, resample_to=16000)
            span_src = ref.view(1, -1)
        spans = active_spans(span_src, sr)
        if sum(b - a for a, b in spans) < item["min_active_sec"] * sr:
            continue
        with torch.no_grad():
            enh = model(raw.to(device), dry_blend=args.dry_blend,
                        spec_floor=args.spec_floor).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
        row = {
            "file": item["wav"].name,
            "bucket": item["bucket"],
            "info": item["info"],
            "distance_m": item["distance_m"],
            "active_reduction_db": reduction_db(enh, raw, spans),
            "active_sec": sum(b - a for a, b in spans) / sr,
        }
        rows.append(row)
        if sink is not None:
            print(json.dumps(row), file=sink, flush=True)
        if item["bucket"] not in seen_buckets:
            seen_buckets.add(item["bucket"])
            print(f"# scoring {item['bucket']} ...", flush=True)

    if sink is not None:
        sink.close()

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    print()
    print("bucket\t\tdist_m\tn\tp25_dB\tmedian_dB\tp75_dB\tinfo")
    for bucket in sorted(by_bucket, key=lambda b: by_bucket[b][0]["distance_m"]):
        rs = by_bucket[bucket]
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in rs])
        dist = rs[0]["distance_m"]
        info = rs[0]["info"] if args.corpus == "voices" else f"{len({r['info'] for r in rs})} scenes"
        print(f"{bucket:<15}\t{dist:.2f}\t{len(rs)}\t{p25:7.2f}\t{med:7.2f}\t{p75:7.2f}\t{info}")

    far = [r for r in rows if r["distance_m"] > 1.0]
    near = [r for r in rows if r["distance_m"] <= 1.0]
    print()
    if far:
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in far])
        print(f"# ALL >1m  (suppress targets): n={len(far)} p25={p25:.2f} median={med:.2f} p75={p75:.2f} dB")
    if near:
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in near])
        print(f"# ALL <=1m (keep targets)    : n={len(near)} median={med:.2f} dB (should be ~0)")


if __name__ == "__main__":
    main()
