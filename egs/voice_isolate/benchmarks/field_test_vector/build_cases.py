"""Build the voice_isolate field benchmark from audio_annotator.html span exports.

Reads the tracked labels in ``spans/*.spans.json`` plus the matching wav from
``egs/voice_isolate/test_vec/`` and writes 16 kHz mono ``<clip>_raw.wav`` files
and a ``windows.json`` describing two scoring modes:

  * ``<tag>_session``            STREAM: the whole recording, spans in place
  * ``<tag>_{near,far,dt}N``     COLD-START: each span cut out and scored alone

A recording only gets a session row when it carries ``SESSION_MIN_SPANS`` or more
spans. A one-span clip IS its own span, so emitting both would score the same
audio twice under two names and quietly double its weight in any aggregate.

Tag routing follows the annotator convention: a tag containing keep/near/double
is a keep span, one containing far/suppress is a suppress span.

Each recording also gets a reference block so the scorer can report absolute
levels, not only relative attenuation:

  * ``floor_dbfs``     5th-percentile 100 ms frame energy of the raw recording --
                       the capture noise floor. A span is only scorable if it sits
                       clearly above this; otherwise there is no energy to remove
                       and any ``reduction_db`` is an artefact of the floor.
  * ``near_ref_dbfs``  energy of the recording's keep spans -- the user's voice as
                       captured. Residual leakage is meaningful relative to this.

``held_out: true`` marks material no readout, threshold or probe may be FITTED on.
It is the only defence against the failure this benchmark exists to catch: a
presence readout fitted on one room scored 0.958 held-out within that room and
deleted the user everywhere else (benchmarks/full_gate/v8_gate050_VERDICT.md).

Where things live. The labels beside this script are version-controlled; the
recordings they describe are not -- they are internal field material that must not
be redistributed, so they and every clip cut from them stay under ignored trees
(``test_vec/`` and ``data_report/``). This script is the bridge: it reads the
private audio, writes the private clips, and writes ``windows.json`` twice -- once
beside the clips for the scorer, once beside these labels as the reviewable
definition. Both come from one run, so they cannot drift.

Run from the repo root:

    uv run python egs/voice_isolate/benchmarks/field_test_vector/build_cases.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
import torchaudio

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RECIPE = REPO / "egs/voice_isolate"
AUDIO_SRC = RECIPE / "test_vec"                                      # private, gitignored
SPANS = HERE / "spans"                                               # tracked labels
OUT = RECIPE / "data_report/field_cases/test_vector_cases"           # private, gitignored
#: QVF2.2's own output for the three clips it published, carried over from the
#: retired qvf22_real_cases set. The scorer picks these up as `reference`.
QVF_REF_SRC = RECIPE / "data_report/field_cases/_superseded/qvf22_real_cases_retired"
SR = 16000
MIN_CLIP_S = 1.0
SESSION_MIN_SPANS = 3
FLOOR_PERCENTILE = 0.05
FRAME = 1600   # 100 ms
HOP = 800

#: Recordings no fitting may touch. 180D is a third device orientation that no
#: readout has seen, which is what makes cross-orientation transfer measurable
#: at all rather than asserted.
HELD_OUT = {"180d"}

#: Single-span keep clips that exist to catch a regression, not to grade one.
SENTINEL = {"0d"}


def tag_for(stem: str) -> str:
    """Short stable tag from a recording's filename.

    ``90D_capmfx_2026_3_17-...`` -> ``90d``; ``QVF_keep_in_touch`` -> ``qvf_keep_in_touch``.
    The device stems carry a capture timestamp and serial that would make every
    windows.json key change if the same room were re-recorded, so the tag keeps
    only the orientation.
    """
    m = re.match(r"^(\d+)D_", stem)
    if m:
        return f"{m.group(1)}d"
    return stem.lower()


def qvf_reference_for(stem: str) -> Path | None:
    """The published QVF2.2 output matching a clip, if there is one."""
    m = re.match(r"^QVF_(scenario\d)$", stem)
    if not m:
        return None
    candidate = QVF_REF_SRC / f"{m.group(1)}_qvf22.wav"
    return candidate if candidate.is_file() else None


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
        if j > i:
            total += float(wav[..., i:j].square().sum())
            n += j - i
    return 10.0 * torch.log10(torch.tensor(total / n + 1e-12)).item() if n else float("nan")


def route(tag: str) -> str:
    t = tag.lower()
    if "double" in t:
        return "double"
    if "far" in t or "suppress" in t:
        return "suppress"
    if "keep" in t or "near" in t:
        return "keep"
    return "exclude"


def dedup(spans: list, stem: str) -> list:
    """Drop byte-identical repeats of the same span.

    The annotator can emit a span twice when a region is confirmed twice; the
    duplicate would double that stretch's weight in every aggregate. Reported
    rather than silently fixed, and the source label file is left alone so
    re-exporting from the tool does not fight this.
    """
    seen, out = set(), []
    for s in spans:
        key = (float(s["start_s"]), float(s["end_s"]), s["tag"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    if len(out) != len(spans):
        print(f"  ! {stem}: dropped {len(spans) - len(out)} duplicate span(s)")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        (tag_for(p.name[: -len(".spans.json")]), p.name[: -len(".spans.json")])
        for p in SPANS.glob("*.spans.json")
    )
    if not sources:
        raise SystemExit(f"no labels in {SPANS}")

    windows: dict[str, object] = {
        "_comment": (
            "voice_isolate field benchmark v3 (2026-08-20), built by build_cases.py from the "
            "hand labelling in egs/voice_isolate/test_vec/*.spans.json (tools/audio_annotator.html). "
            "Supersedes v2, which covered only 90D/270D and kept the QVF clips in a separate set -- "
            "v2 numbers are NOT comparable, the clip inventory and the labels both changed. "
            "Sources: four device orientations (0D/90D/180D/270D, 32 kHz stereo, downmixed to 16 kHz "
            "mono here) plus seven QVF2.2 publication clips; scenario1-3 keep QVF's own output as "
            "'reference'. Near talker 30 cm on 90D, 50 cm on 270D; far talkers at 200/300 cm per span "
            "label. TWO MODES, never averaged: '*_session' is STREAM (full recording, near anchor "
            "present = mid-conversation); the cut-out near*/far*/dt* clips are COLD-START (span fed "
            "from t=0, no context). A recording with fewer than "
            f"{SESSION_MIN_SPANS} spans gets no session row -- it would be the same audio twice. "
            "double-talk is scored keep-only: the near voice must survive; the far side is not "
            "separable without a reference. 'held_out: true' means NOTHING may be fitted on that "
            "recording -- 180D is reserved for cross-orientation transfer. 'sentinel: true' marks a "
            "single-span keep clip that exists to catch a regression, not to grade one. Each clip "
            "carries floor_dbfs / near_ref_dbfs so the scorer can report absolute residual level, not "
            "only relative attenuation -- a span that starts close to the noise floor has little "
            "removable energy and its reduction_db is not comparable with a loud one."
        )
    }

    for tag, stem in sources:
        spans_doc = json.loads((SPANS / f"{stem}.spans.json").read_text())
        source_wav = AUDIO_SRC / f"{stem}.wav"
        if not source_wav.is_file():
            raise SystemExit(f"label {stem}.spans.json has no audio at {source_wav}")
        wav = load_mono_16k(source_wav)
        n = wav.shape[-1]

        keep_spans, supp_spans = [], []
        counters = {"keep": 0, "suppress": 0, "double": 0}
        cuts = []
        for s in dedup(spans_doc["spans"], stem):
            kind = route(s["tag"])
            if kind == "exclude":
                continue
            a, b = float(s["start_s"]), float(s["end_s"])
            (supp_spans if kind == "suppress" else keep_spans).append([a, b])
            counters[kind] += 1
            cuts.append((kind, counters[kind], a, b, s.get("label") or ""))

        floor = round(noise_floor_dbfs(wav), 2)
        near_ref = round(span_energy_dbfs(wav, keep_spans), 2)
        common = {
            "group": tag,
            "floor_dbfs": floor,
            "near_ref_dbfs": near_ref,
        }
        if tag in HELD_OUT:
            common["held_out"] = True

        if len(cuts) >= SESSION_MIN_SPANS:
            session = f"{tag}_session"
            torchaudio.save(str(OUT / f"{session}_raw.wav"), wav, SR,
                            encoding="PCM_S", bits_per_sample=16)
            windows[session] = {
                "role": f"STREAM: full {tag.upper()} recording (near anchor present)",
                **common,
                "keep": keep_spans,
                "suppress": supp_spans,
                "source": f"{stem}.wav",
                "duration_s": round(n / SR, 3),
            }

        prefix = {"keep": "near", "suppress": "far", "double": "dt"}
        for kind, idx, a, b, label in cuts:
            if b - a < MIN_CLIP_S:
                print(f"  . {tag}: skip {prefix[kind]}{idx} ({b - a:.2f}s < {MIN_CLIP_S}s)")
                continue
            clip = f"{tag}_{prefix[kind]}{idx}"
            seg = wav[..., int(a * SR): min(int(b * SR), n)]
            torchaudio.save(str(OUT / f"{clip}_raw.wav"), seg, SR,
                            encoding="PCM_S", bits_per_sample=16)
            dur = round(seg.shape[-1] / SR, 3)
            role = {
                "keep": "COLD-START lone near voice (keep)",
                "suppress": "COLD-START lone far voice, no near anchor (suppress)",
                "double": "COLD-START double-talk, keep-only (near must survive)",
            }[kind]
            spec: dict[str, object] = {
                "role": ("SENTINEL " + role) if tag in SENTINEL else role,
                **common,
                "label": label,
                "in_dbfs": round(db(seg), 2),
                "source_span_s": [a, b],
            }
            if tag in SENTINEL:
                spec["sentinel"] = True
            spec["suppress" if kind == "suppress" else "keep"] = [[0.0, dur]]
            windows[clip] = spec

            reference = qvf_reference_for(stem)
            if reference is not None and len(cuts) < SESSION_MIN_SPANS:
                # Whole-clip case: QVF's output lines up with the clip itself.
                shutil.copyfile(reference, OUT / f"{clip}_qvf22.wav")

        # Multi-span QVF clips: the reference matches the full recording, so it
        # belongs to the session row rather than to any single cut-out span.
        reference = qvf_reference_for(stem)
        if reference is not None and len(cuts) >= SESSION_MIN_SPANS:
            ref_wav = load_mono_16k(reference)
            torchaudio.save(str(OUT / f"{tag}_session_qvf22.wav"), ref_wav, SR,
                            encoding="PCM_S", bits_per_sample=16)

    serialized = json.dumps(windows, indent=1, ensure_ascii=False) + "\n"
    (OUT / "windows.json").write_text(serialized)   # beside the clips, for the scorer
    (HERE / "windows.json").write_text(serialized)  # beside the labels, for review
    clips = [k for k in windows if not k.startswith("_")]
    sessions = [k for k in clips if k.endswith("_session")]
    print(f"\nwrote {len(clips)} entries ({len(sessions)} session, "
          f"{len(clips) - len(sessions)} cold-start) to {OUT}")
    print(f"held out: {sorted(HELD_OUT)}   sentinel: {sorted(SENTINEL)}")


if __name__ == "__main__":
    main()
