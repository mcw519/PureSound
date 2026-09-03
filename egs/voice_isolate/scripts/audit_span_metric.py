"""Attribute a span score to what actually changed, instead of one energy ratio.

The shipped field score is a broadband energy ratio over a whole span, so target
speech, interferer speech, room tone and the reverberation tail all land in one
number and cannot be told apart. The 2026-08-24 listening test produced three
counter-examples; this is the instrument that settles them.

Two ideas do the work, and neither needs a clean reference:

1. HARMONIC PEAKS vs INTER-HARMONIC VALLEYS, at syllable nuclei, 150 Hz-4 kHz,
   with a window long enough to resolve harmonics (64 ms). Voiced speech puts its
   energy on the harmonics; noise and the late reverberation tail fill the space
   between them.
     peaks flat, valleys down, contrast up  -> noise/tail came out, talker intact
     peaks down as far as valleys           -> plain attenuation of everything
     peaks down further than valleys        -> the talker is being damaged
   This is immune to the trap that breaks a noise-budget argument: a capture with
   a gate or AGC ducks in the pauses, so pause level does not estimate the noise
   sitting under the speech. Peaks and valleys are measured in the same frames.

2. TWO LAYERS for suppress spans: the change on nucleus frames (the interferer's
   own speech) reported separately from the change on quiet frames (room tone).
   When the quiet layer moves at least as much as the speech layer, the headline
   reduction is partly denoising and must not be read as bystander removal.

Reads the wavs render_examples.py wrote -- no model, no inference:

    uv run python egs/voice_isolate/scripts/audit_span_metric.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] / "data_report/listening/v8_vs_v11"
SR = 16000
# long analysis window: 64 ms resolves an adult F0 into separate harmonic bins
H_FRAME, H_HOP, H_NFFT = 1024, 160, 2048
# short window for the level/activity layers
L_FRAME, L_HOP, L_NFFT = 400, 160, 512
HARM_LO, HARM_HI = 150.0, 4000.0
SPEECH_LO, SPEECH_HI = 150.0, 8000.0
ACTIVE_OVER_FLOOR, NUCLEI_FRACTION = 8.0, 0.25
PEAK_FRACTION, VALLEY_FRACTION = 0.15, 0.40
GATE_SPREAD_WARN = 15.0


def stft_pow(x: np.ndarray, frame: int, hop: int, nfft: int) -> np.ndarray:
    n = 1 + max(0, len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    f = x[idx] * np.hanning(frame)[None, :]
    pad = np.zeros((f.shape[0], nfft), dtype=np.float64)
    pad[:, :frame] = f
    return np.abs(np.fft.rfft(pad, axis=-1)) ** 2


def band(nfft: int, lo: float, hi: float) -> slice:
    bins = np.fft.rfftfreq(nfft, 1 / SR)
    return slice(int(np.searchsorted(bins, lo)), int(np.searchsorted(bins, hi)))


def db(a: float, b: float) -> float:
    return float("nan") if b <= 0 else 10.0 * np.log10(max(a, 1e-30) / b)


def audit_span(short: dict, long_: dict, t0: float, t1: float) -> dict | None:
    tags = [t for t in short if t != "mix"]
    nfs = min(p.shape[0] for p in short.values())
    nfl = min(p.shape[0] for p in long_.values())
    s0, s1 = int(t0 * SR / L_HOP), min(int(t1 * SR / L_HOP), nfs)
    l0, l1 = int(t0 * SR / H_HOP), min(int(t1 * SR / H_HOP), nfl)
    if s1 - s0 < 8 or l1 - l0 < 8:
        return None

    SPEECH = band(L_NFFT, SPEECH_LO, SPEECH_HI)
    mix_s = short["mix"][s0:s1]
    lvl = 10.0 * np.log10(mix_s[:, SPEECH].sum(1) + 1e-30)
    floor = float(np.percentile(lvl, 10))
    active = lvl > floor + ACTIVE_OVER_FLOOR
    if active.sum() < 2:
        active = lvl > np.percentile(lvl, 75)
    act = np.flatnonzero(active)
    quiet = np.flatnonzero(~active)
    k = max(1, int(round(NUCLEI_FRACTION * act.size)))
    nuc_s = act[np.argsort(lvl[act])[-k:]]

    qspread = float(np.ptp(lvl[quiet])) if quiet.size > 2 else float("nan")

    HARM = band(H_NFFT, HARM_LO, HARM_HI)
    mix_l = long_["mix"][l0:l1]
    lvl_l = 10.0 * np.log10(mix_l[:, HARM].sum(1) + 1e-30)
    fl_l = float(np.percentile(lvl_l, 10))
    act_l = np.flatnonzero(lvl_l > fl_l + ACTIVE_OVER_FLOOR)
    if act_l.size < 2:
        act_l = np.flatnonzero(lvl_l > np.percentile(lvl_l, 75))
    kl = max(1, int(round(NUCLEI_FRACTION * act_l.size)))
    nuc_l = act_l[np.argsort(lvl_l[act_l])[-kl:]]

    M = mix_l[nuc_l][:, HARM]
    order = np.argsort(M, axis=1)
    nb = M.shape[1]
    pk = order[:, -max(1, int(PEAK_FRACTION * nb)):]
    vl = order[:, : max(1, int(VALLEY_FRACTION * nb))]
    rows = np.arange(M.shape[0])[:, None]
    pk_in, vl_in = M[rows, pk].sum(), M[rows, vl].sum()

    out = {
        "t0": round(t0, 3), "t1": round(t1, 3),
        "active_pct": round(float(active.mean()) * 100, 1),
        "nuclei_over_pause_db": round(float(np.median(lvl[nuc_s]) - floor), 2),
        "pause_spread_db": None if qspread != qspread else round(qspread, 1),
        "gated_capture_suspect": bool(qspread == qspread and qspread > GATE_SPREAD_WARN),
        "mix_harm_contrast_db": round(db(pk_in, vl_in), 1),
        "systems": {},
    }
    for t in tags:
        q_s = short[t][s0:s1]
        q_l = long_[t][l0:l1]
        Q = q_l[nuc_l][:, HARM]
        peak = db(Q[rows, pk].sum(), pk_in)
        valley = db(Q[rows, vl].sum(), vl_in)
        out["systems"][t] = {
            "harm_peak": round(peak, 2),
            "harm_valley": round(valley, 2),
            "contrast_delta": round(peak - valley, 2),
            "speech_layer": round(db(q_s[nuc_s[:, None], SPEECH].sum(),
                                     mix_s[nuc_s[:, None], SPEECH].sum()), 2),
            "quiet_layer": (round(db(q_s[quiet[:, None], :].sum(),
                                     mix_s[quiet[:, None], :].sum()), 2)
                            if quiet.size else None),
            "broadband": round(db(q_s.sum(), mix_s.sum()), 2),
        }
    return out


def verdict_keep(m: dict) -> str:
    if m["harm_peak"] > -2.5 and m["contrast_delta"] > 2.0:
        return "INTACT (noise/tail removed)"
    if m["harm_peak"] <= -6.0 and m["contrast_delta"] < -1.5:
        return "DAMAGED (harmonics hit hardest)"
    if m["harm_peak"] <= -6.0:
        return "ATTENUATED (uniform, no spectral damage)"
    if m["harm_peak"] > -2.5:
        return "INTACT"
    return "TRIMMED (2.5-6 dB, no spectral damage)"


def verdict_supp(m: dict) -> str:
    if m["quiet_layer"] is None:
        return "no in-span quiet frames -> cannot separate"
    if m["speech_layer"] > -6.0:
        return "NOT SUPPRESSED"
    if m["quiet_layer"] <= m["speech_layer"]:
        return "CONTAMINATED (room tone drops >= the speech)"
    return "SUPPRESSED"


def main() -> None:
    report = json.loads((ROOT / "report.json").read_text())
    result: dict = {}
    for clip in (sys.argv[1:] or list(report)):
        info = report[clip]
        tags = [t for t in info["metrics"] if t != "qvf22"]
        short, long_ = {}, {}
        for t in tags:
            x, sr = sf.read(ROOT / clip / f"{t}.wav", dtype="float64")
            assert sr == SR, sr
            x = x if x.ndim == 1 else x.mean(1)
            short[t] = stft_pow(x, L_FRAME, L_HOP, L_NFFT)
            long_[t] = stft_pow(x, H_FRAME, H_HOP, H_NFFT)

        print(f"\n{'='*112}\n{clip}")
        rows = []
        for kind in ("keep", "suppress"):
            for i, (a, b) in enumerate(info["spans_excerpt"].get(kind, []), start=1):
                d = audit_span(short, long_, a, b)
                if d is None:
                    continue
                d["kind"], d["i"] = kind, i
                rows.append(d)
                flag = "  [gated/AGC capture: pause level is not the noise under speech]" if d["gated_capture_suspect"] else ""
                print(f"  {kind.upper()} #{i}  {a:.2f}-{b:.2f} s   active {d['active_pct']:.0f}%   "
                      f"nuclei-over-pause {d['nuclei_over_pause_db']:.1f} dB   "
                      f"mix harmonic contrast {d['mix_harm_contrast_db']:.1f} dB{flag}")
                print(f"      {'sys':>4}{'harm pk':>9}{'harm vl':>9}{'d contr':>9}"
                      f"{'speech':>9}{'quiet':>9}{'BROAD':>9}   attribution")
                for t, m in d["systems"].items():
                    v = verdict_keep(m) if kind == "keep" else verdict_supp(m)
                    q = "      —" if m["quiet_layer"] is None else f"{m['quiet_layer']:+9.2f}"
                    print(f"      {t:>4}{m['harm_peak']:+9.2f}{m['harm_valley']:+9.2f}"
                          f"{m['contrast_delta']:+9.2f}{m['speech_layer']:+9.2f}{q}"
                          f"{m['broadband']:+9.2f}   {v}")
        result[clip] = rows
    (ROOT / "span_attribution.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\nwrote {ROOT / 'span_attribution.json'}")


if __name__ == "__main__":
    main()
