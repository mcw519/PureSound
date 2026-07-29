"""Far-field suppression probe on retransmitted corpora (VOiCES).

Retransmitted corpora replay clean speech through a loudspeaker in a real room and
re-record it, so each file carries a whole capture chain (loudspeaker -> air -> mic
-> ADC) that an RIR convolution cannot reproduce. This probe measures how much a
checkpoint suppresses that recorded far-field speech, bucketed by room and mic
distance, which is the quantity synthetic far-only probes systematically
over-estimate.

Target behaviour: any voice >1m is a suppress target, <1m is keep. Every VOiCES mic
except rm1-clo (0.97 m) and beh (0.74 m, behind the loudspeaker) is >1m, so for
those buckets "more negative = better".

Reduction is measured on speech-active spans detected from the RAW distant input
(energy gate above its own noise floor), so no alignment with the source is needed.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/probe_retransmitted_farfield.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v6.ckpt \
        --voices-root /work/any_exp_link/puresound_exp/real_e2e_corpora/voices/VOiCES_devkit \
        --device cuda --per-bucket 40
"""

from __future__ import annotations

import argparse
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
from puresound.recipes import init_siso_model, load_siso_recipe_config

# Foreground-loudspeaker distance per (room, mic id), inches -> meters.
# Source: Lab41-SRI-VOiCES_README.md mic table.
MIC_DISTANCE_M = {
    ("rm1", 1): 0.97, ("rm1", 2): 0.97, ("rm1", 3): 1.83, ("rm1", 4): 1.83,
    ("rm1", 5): 3.02, ("rm1", 6): 3.02, ("rm1", 7): 0.74, ("rm1", 8): 0.74,
    ("rm1", 9): 1.47, ("rm1", 10): 1.91, ("rm1", 11): 1.91, ("rm1", 12): 3.30,
    ("rm2", 1): 2.03, ("rm2", 2): 2.03, ("rm2", 3): 3.33, ("rm2", 4): 3.33,
    ("rm2", 5): 5.79, ("rm2", 6): 5.79, ("rm2", 7): 0.74, ("rm2", 8): 0.74,
    ("rm2", 9): 2.77, ("rm2", 10): 3.25, ("rm2", 11): 3.25, ("rm2", 12): 2.95,
    # rm3/rm4 shipped in the final release (not in the 2-room README); distances
    # from recording_data distances.csv ("foreground" column, inches -> m).
    ("rm3", 1): 1.70, ("rm3", 2): 1.70, ("rm3", 3): 3.71, ("rm3", 4): 3.71,
    ("rm3", 5): 7.14, ("rm3", 6): 7.14, ("rm3", 7): 1.47, ("rm3", 8): 1.47,
    ("rm3", 9): 2.24, ("rm3", 10): 4.47, ("rm3", 11): 4.47, ("rm3", 12): 6.65,
    ("rm4", 1): 1.83, ("rm4", 2): 1.83, ("rm4", 3): 4.24, ("rm4", 4): 4.24,
    ("rm4", 5): 9.83, ("rm4", 6): 9.83, ("rm4", 7): 1.80, ("rm4", 8): 1.80,
    ("rm4", 9): 3.25, ("rm4", 10): 4.39, ("rm4", 11): 4.45, ("rm4", 12): 9.65,
}

FNAME_RE = re.compile(
    r"(?P<room>rm\d+)-(?P<distractor>[a-z]+)-sp(?P<spk>\d+)-ch(?P<ch>\d+)"
    r"-s(?:e)?g(?P<seg>\d+)-mc(?P<mic>\d+)-(?P<mtype>[a-z]+)-(?P<loc>[a-z]+)"
)


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model_dict = load_siso_recipe_config(config_path)[5]
    model = init_siso_model(model_dict)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--voices-root", required=True, help="extracted devkit root (contains distant-16k/)")
    parser.add_argument("--distractor", default="none", choices=["none", "babb", "musi", "tele"])
    parser.add_argument("--per-bucket", type=int, default=40, help="segments per (room, mic) bucket")
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None, help="write per-file rows as JSONL")
    parser.add_argument("--list-only", action="store_true", help="only print bucket inventory")
    args = parser.parse_args()

    speech_root = Path(args.voices_root) / "distant-16k" / "speech"
    if not speech_root.is_dir():
        sys.exit(f"not found: {speech_root}")

    buckets: dict[tuple[str, int], list[Path]] = defaultdict(list)
    pattern = f"*-{args.distractor}-*.wav"
    for wav_path in speech_root.rglob(pattern):
        m = FNAME_RE.search(wav_path.name)
        if m is None:
            continue
        buckets[(m.group("room"), int(m.group("mic")))].append(wav_path)

    if not buckets:
        sys.exit(f"no files matching distractor={args.distractor} under {speech_root}")

    rng = random.Random(args.seed)
    print(f"# buckets (distractor={args.distractor}):")
    for key in sorted(buckets):
        room, mic = key
        print(f"#   {room} mc{mic:02d} d={MIC_DISTANCE_M.get(key, float('nan')):.2f}m  n={len(buckets[key])}")
    if args.list_only:
        return

    device = torch.device(args.device)
    model = load_model(args.config_path, args.ckpt, device)

    rows = []
    for key in sorted(buckets):
        files = buckets[key]
        picks = files if len(files) <= args.per_bucket else rng.sample(files, args.per_bucket)
        for wav_path in sorted(picks):
            raw, sr = AudioIO.open(f_path=str(wav_path), target_lvl=None, resample_to=16000)
            raw = raw.view(1, -1)
            spans = active_spans(raw, sr)
            if sum(b - a for a, b in spans) < sr:  # <1 s of speech -> unusable
                continue
            with torch.no_grad():
                enh = model(raw.to(device)).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
            m = FNAME_RE.search(wav_path.name)
            rows.append(
                {
                    "file": wav_path.name,
                    "room": key[0],
                    "mic": key[1],
                    "loc": m.group("loc"),
                    "distance_m": MIC_DISTANCE_M.get(key),
                    "active_reduction_db": reduction_db(enh, raw, spans),
                    "full_reduction_db": reduction_db(enh, raw, [(0, raw.shape[-1])]),
                    "active_sec": sum(b - a for a, b in spans) / sr,
                }
            )
        done = [r for r in rows if (r["room"], r["mic"]) == key]
        if done:
            _, med, _ = quantiles([r["active_reduction_db"] for r in done])
            print(f"# {key[0]} mc{key[1]:02d} done: n={len(done)} median={med:.2f} dB", flush=True)

    if args.out_json:
        Path(args.out_json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print()
    print("room\tmic\tloc\tdist_m\tn\tp25_dB\tmedian_dB\tp75_dB")
    by_bucket: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[(r["room"], r["mic"])].append(r)
    for key in sorted(by_bucket, key=lambda k: (k[0], MIC_DISTANCE_M.get(k, 0.0))):
        rs = by_bucket[key]
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in rs])
        print(
            f"{key[0]}\tmc{key[1]:02d}\t{rs[0]['loc']}\t{rs[0]['distance_m']:.2f}\t{len(rs)}"
            f"\t{p25:7.2f}\t{med:7.2f}\t{p75:7.2f}"
        )
    far_rows = [r for r in rows if (r["distance_m"] or 0) > 1.0]
    if far_rows:
        p25, med, p75 = quantiles([r["active_reduction_db"] for r in far_rows])
        print(f"\n# ALL >1m (suppress targets): n={len(far_rows)} p25={p25:.2f} median={med:.2f} p75={p75:.2f} dB")


if __name__ == "__main__":
    main()
