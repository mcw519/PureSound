"""Build the voice_isolate field benchmark from audio_annotator.html span exports.

Reads ``egs/voice_isolate/test_vector/*.spans.json`` plus the matching wav and writes
16 kHz mono ``<clip>_raw.wav`` files and a ``windows.json`` describing two scoring modes:

  * ``<tag>_session``            STREAM: the whole recording, spans in place
  * ``<tag>_{near,far,dt}N``     COLD-START: each span cut out and scored alone

Tag routing follows the annotator convention: a tag containing keep/near/double is a
keep span, one containing far/suppress is a suppress span.

Each recording also gets a reference block so the scorer can report absolute levels,
not only relative attenuation:

  * ``floor_dbfs``     5th-percentile 100 ms frame energy of the raw recording -- the
                       capture noise floor. A span is only scorable if it sits clearly
                       above this; otherwise there is no energy to remove and any
                       ``reduction_db`` is an artefact of the floor, not of the model.
  * ``near_ref_dbfs``  energy of the recording's keep spans -- the user's voice as
                       captured. Residual leakage is meaningful relative to this.

Where things live. The labels beside this script are version-controlled; the recordings
they describe are not -- they are internal field material that must not be
redistributed, so they and every clip cut from them stay under the ignored
``data_report/`` tree. This script is the bridge: it reads the private audio, writes the
private clips, and writes ``windows.json`` twice -- once beside the clips for the scorer
to read, once beside these labels as the reviewable definition. Both come from one run,
so they cannot drift.

Run from the repo root:

    uv run python egs/voice_isolate/benchmarks/field_test_vector/build_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RECIPE = REPO / "egs/voice_isolate"
AUDIO_SRC = RECIPE / "test_vector"                                   # private, gitignored
SPANS = HERE / "spans"                                               # tracked labels
OUT = RECIPE / "data_report/field_cases/test_vector_cases"           # private, gitignored
SR = 16000
MIN_CLIP_S = 1.0
FLOOR_PERCENTILE = 0.05
FRAME = 1600   # 100 ms
HOP = 800

SOURCES = [("90d", "real_record_test_90D"), ("270d", "real_record_test_270D")]


def load_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def db(x: torch.Tensor) -> float:
    return float(10.0 * torch.log10(x.square().mean() + 1e-12))


def noise_floor_dbfs(wav: torch.Tensor) -> float:
    frames = wav.view(-1).unfold(-1, FRAME, HOP).square().mean(-1)
    return float(10.0 * torch.log10(torch.quantile(frames, FLOOR_PERCENTILE) + 1e-12))


def span_energy_dbfs(wav: torch.Tensor, spans) -> float:
    total = n = 0.0
    for a, b in spans:
        i, j = int(a * SR), min(int(b * SR), wav.shape[-1])
        if j <= i:
            continue
        total += float(wav[..., i:j].square().sum())
        n += j - i
    return float(10.0 * torch.log10(torch.tensor(total / max(n, 1.0) + 1e-12)))


def route(tag: str) -> str:
    t = tag.lower()
    if "double" in t:
        return "double"
    if "far" in t or "suppress" in t:
        return "suppress"
    if "keep" in t or "near" in t:
        return "keep"
    return "exclude"


def main() -> None:
    windows: dict[str, object] = {
        "_comment": (
            "voice_isolate field benchmark v2 (2026-08-16), built by build_cases.py from the "
            "hand labelling in egs/voice_isolate/test_vector/*.spans.json (tools/audio_annotator.html). "
            "90D = real_record_test_90D.wav (bit-identical to the older field_cases/raw/"
            "real_distance_test_vector_90D.wav -- only the labels changed), 270D = real_record_test_270D.wav. "
            "Near talker 30 cm on 90D, 50 cm on 270D; far talkers at 200/300 cm per span label. "
            "TWO MODES, never averaged: '*_session' is STREAM (full recording, near anchor present = "
            "mid-conversation); the cut-out near*/far*/dt* clips are COLD-START (span fed from t=0, no context). "
            "double-talk is scored keep-only: the near voice must survive; the far side is not "
            "separable without a reference. Each clip carries floor_dbfs / near_ref_dbfs so the scorer "
            "can report absolute residual level, not only relative attenuation -- a span that starts "
            "close to the noise floor has little removable energy and its reduction_db is not comparable "
            "with a loud one."
        )
    }

    for tag, stem in SOURCES:
        spans_doc = json.loads((SPANS / f"{stem}.spans.json").read_text())
        wav = load_mono_16k(AUDIO_SRC / f"{stem}.wav")
        n = wav.shape[-1]

        keep_spans, supp_spans = [], []
        counters = {"keep": 0, "suppress": 0, "double": 0}
        cuts = []
        for s in spans_doc["spans"]:
            kind = route(s["tag"])
            if kind == "exclude":
                continue
            a, b = float(s["start_s"]), float(s["end_s"])
            (supp_spans if kind == "suppress" else keep_spans).append([a, b])
            counters[kind] += 1
            cuts.append((kind, counters[kind], a, b, s.get("label", "")))

        floor = round(noise_floor_dbfs(wav), 2)
        near_ref = round(span_energy_dbfs(wav, keep_spans), 2)

        session = f"{tag}_session"
        torchaudio.save(str(OUT / f"{session}_raw.wav"), wav, SR, encoding="PCM_S", bits_per_sample=16)
        windows[session] = {
            "role": f"STREAM: full {tag.upper()} recording (near anchor present)",
            "group": tag,
            "keep": keep_spans,
            "suppress": supp_spans,
            "floor_dbfs": floor,
            "near_ref_dbfs": near_ref,
            "source": f"{stem}.wav",
            "duration_s": round(n / SR, 3),
        }

        prefix = {"keep": "near", "suppress": "far", "double": "dt"}
        for kind, idx, a, b, label in cuts:
            if b - a < MIN_CLIP_S:
                continue
            clip = f"{tag}_{prefix[kind]}{idx}"
            seg = wav[..., int(a * SR): min(int(b * SR), n)]
            torchaudio.save(str(OUT / f"{clip}_raw.wav"), seg, SR, encoding="PCM_S", bits_per_sample=16)
            dur = round(seg.shape[-1] / SR, 3)
            spec: dict[str, object] = {
                "role": {
                    "keep": "COLD-START lone near voice (keep)",
                    "suppress": "COLD-START lone far voice, no near anchor (suppress)",
                    "double": "COLD-START double-talk, keep-only (near must survive)",
                }[kind],
                "group": tag,
                "label": label,
                "floor_dbfs": floor,
                "near_ref_dbfs": near_ref,
                "in_dbfs": round(db(seg), 2),
                "source_span_s": [a, b],
            }
            spec["suppress" if kind == "suppress" else "keep"] = [[0.0, dur]]
            windows[clip] = spec

    serialized = json.dumps(windows, indent=1, ensure_ascii=False) + "\n"
    (OUT / "windows.json").write_text(serialized)   # beside the clips, for the scorer
    (HERE / "windows.json").write_text(serialized)  # beside the labels, for review
    clips = [k for k in windows if not k.startswith("_")]
    print(f"wrote {OUT} : {len(clips)} clips")
    for tag, _ in SOURCES:
        s = windows[f"{tag}_session"]
        print(f"  {tag}: floor {s['floor_dbfs']} dBFS, near reference {s['near_ref_dbfs']} dBFS")


if __name__ == "__main__":
    main()
