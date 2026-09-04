"""Part (c): put the real and synthetic onset profiles side by side and take the ratio."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent
WINDOWS = (0.5, 1.0, 2.0)


def load():
    real = [json.loads(l) for l in (OUT / "real_onset_rows.jsonl").read_text().splitlines() if l.strip()]
    synth = []
    for tag in ("v8", "v16"):
        p = OUT / f"synth_onset_rows_{tag}.jsonl"
        if p.is_file():
            synth += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return real, synth


def stat(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    return v


def line(label, rows, on_key, rest_key, exc_key):
    on, rest, exc = stat(rows, on_key), stat(rows, rest_key), stat(rows, exc_key)
    if exc is None:
        return f"{label:52s}      n/a"
    n = len(exc)
    # onset-only deletion: onset window below -3 dB while the steady state is not
    o = np.array([r[on_key] for r in rows if r.get(exc_key) is not None], float)
    s = np.array([r[rest_key] for r in rows if r.get(exc_key) is not None], float)
    only = int(((o < -3.0) & (s >= -3.0)).sum())
    p = "n/a"
    if n >= 8 and np.any(o != s):
        try:
            p = f"{wilcoxon(o, s).pvalue:.2g}"
        except Exception:
            p = "n/a"
    return (f"{label:52s} {np.median(on):7.2f} {np.median(rest):7.2f} "
            f"{np.median(exc):7.2f} {np.percentile(exc, 25):7.2f} {np.percentile(exc, 10):7.2f} "
            f"{np.percentile(exc, 5):7.2f} {only:4d}/{n:<4d} {p:>9s}")


HDR = (f"{'group':52s} {'d_on':>7s} {'d_rest':>7s} {'excess':>7s} {'p25':>7s} {'p10':>7s} "
       f"{'p5':>7s} {'onlyDel':>9s} {'wilcox_p':>9s}")


def main():
    real, synth = load()
    txt = []

    def emit(s=""):
        txt.append(s)
        print(s)

    for X in WINDOWS:
        emit(f"\n================ window = first {X} s ================")
        emit("### A. keep-level delta (span_dbfs enh - mix), REAL")
        emit(HDR)
        for tag in ("v8", "v16"):
            for sub in ("none", "stream", "near_speech"):
                for side in ("near", "dt"):
                    rr = [r for r in real if r["tag"] == tag and r["world"] == "field"
                          and r["subset"] == sub and r["side"] == side and r["window"] == X]
                    if rr:
                        emit(line(f"{tag}  field {sub:11s} {side}", rr, "d_onset", "d_rest", "excess"))
            rr = [r for r in real if r["tag"] == tag and r["world"] == "dawn" and r["window"] == X]
            if rr:
                emit(line(f"{tag}  dawn  none (real rec, real bg lead-in)", rr, "d_onset", "d_rest", "excess"))

        emit("\n### B. keep-level delta (span_dbfs enh - mix), SYNTHETIC")
        emit(HDR)
        for tag in ("v8", "v16"):
            for sub in ("wer_set_moderate_test", "indomain_wer_set"):
                for var in ("t0", "bg1", "sil1"):
                    rr = [r for r in synth if r["tag"] == tag and r["subset"] == sub
                          and r["variant"] == var and r["window"] == X]
                    if rr:
                        emit(line(f"{tag}  synth {sub[:22]:22s} {var}", rr, "d_on", "d_rest", "d_excess"))

        emit("\n### B2. bg1 split by where the 1 s prefix came from (level delta)")
        emit(HDR)
        for tag in ("v8", "v16"):
            for src in ("inactive_loop", "head_fallback"):
                rr = [r for r in synth if r["tag"] == tag and r["variant"] == "bg1"
                      and r["window"] == X and r.get("bg_src") == src
                      and r.get("d_excess") is not None]
                if rr:
                    emit(line(f"{tag}  synth bg1 prefix={src}", rr, "d_on", "d_rest", "d_excess"))

        emit("\n### C. foreground projection gain (immune to interferer content); ref available")
        emit(HDR.replace("d_on", "g_on").replace("d_rest", "g_rest"))
        for tag in ("v8", "v16"):
            rr = [r for r in real if r["tag"] == tag and r["world"] == "dawn" and r["window"] == X
                  and r.get("g_excess") is not None]
            if rr:
                emit(line(f"{tag}  dawn  none  (REAL)", rr, "g_on", "g_rest", "g_excess"))
            for sub in ("wer_set_moderate_test", "indomain_wer_set"):
                for var in ("t0", "bg1", "sil1"):
                    rr = [r for r in synth if r["tag"] == tag and r["subset"] == sub
                          and r["variant"] == var and r["window"] == X
                          and r.get("g_excess") is not None]
                    if rr:
                        emit(line(f"{tag}  synth {sub[:22]:22s} {var} (SYNTH)", rr, "g_on", "g_rest", "g_excess"))

    # ---- ratios ----
    emit("\n\n================ (c) real / synthetic ratio ================")
    emit("onset deletion = -(median excess), dB.  Matched pair: Dawn none (real recordings, real")
    emit("background lead-in) vs synthetic bg1 (own background prepended).  Also field vs synth t0.")
    emit(f"{'ckpt':5s} {'window':7s} {'measure':10s} {'real_src':16s} {'real':>7s} "
         f"{'synth_src':10s} {'synth':>7s} {'ratio':>7s}")
    rat = []
    for tag in ("v8", "v16"):
        for X in WINDOWS:
            pairs = [
                ("level", "dawn none", [r["excess"] for r in real if r["tag"] == tag and r["world"] == "dawn"
                                        and r["window"] == X and r["excess"] is not None],
                 "bg1 (both)", [r["d_excess"] for r in synth if r["tag"] == tag and r["variant"] == "bg1"
                                and r["window"] == X and r.get("d_excess") is not None]),
                ("fgproj", "dawn none", [r["g_excess"] for r in real if r["tag"] == tag and r["world"] == "dawn"
                                         and r["window"] == X and r.get("g_excess") is not None],
                 "bg1 (both)", [r["g_excess"] for r in synth if r["tag"] == tag and r["variant"] == "bg1"
                                and r["window"] == X and r.get("g_excess") is not None]),
                ("level", "field near+dt", [r["excess"] for r in real if r["tag"] == tag and r["world"] == "field"
                                            and r["subset"] in ("stream", "near_speech") and r["window"] == X
                                            and r["excess"] is not None],
                 "t0 (both)", [r["d_excess"] for r in synth if r["tag"] == tag and r["variant"] == "t0"
                               and r["window"] == X and r.get("d_excess") is not None]),
            ]
            for meas, rs, rv, ss, sv in pairs:
                if not rv or not sv:
                    continue
                a, b = -np.median(rv), -np.median(sv)
                r = a / b if abs(b) > 1e-6 else float("inf")
                emit(f"{tag:5s} {X:<7} {meas:10s} {rs:16s} {a:7.2f} {ss:10s} {b:7.2f} {r:7.2f}")
                rat.append((tag, X, meas, a, b, r))
            # tail (p5) version -- the deletions live in the tail
            for meas, rs, rv, ss, sv in pairs:
                if not rv or not sv:
                    continue
                a, b = -np.percentile(rv, 5), -np.percentile(sv, 5)
                r = a / b if abs(b) > 1e-6 else float("inf")
                emit(f"{tag:5s} {X:<7} {meas+'-p5':10s} {rs:16s} {a:7.2f} {ss:10s} {b:7.2f} {r:7.2f}")

    (OUT / "onset_profile_tables.txt").write_text("\n".join(txt) + "\n")


if __name__ == "__main__":
    main()
