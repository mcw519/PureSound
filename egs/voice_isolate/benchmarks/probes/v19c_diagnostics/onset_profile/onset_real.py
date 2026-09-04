"""Part (a): onset profile on REAL data, from the anchor_gate cache (no model calls).

For every field near/dt clip (conditions none / stream / near_speech) and every Dawn utterance
(condition none), compare the keep-level delta (enh vs mix, span_dbfs definition of
scripts/eval_realcase.py) over the FIRST X s of the keep span / foreground activity against the
REMAINDER of that same span.

onset_excess(X) = delta(first X s) - delta(remainder)      [dB]
   negative  => the onset is attenuated MORE than the steady state  => onset deletion
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

CACHE = Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/agcache")
OUT = Path(__file__).resolve().parent
SR = 16000
WINDOWS = (0.5, 1.0, 2.0)
MIN_REST = 0.5   # need at least this much steady state left to compare against


def level_dbfs(wav: np.ndarray, mask: np.ndarray) -> float:
    """10 log10 mean-square over the masked samples (span_dbfs, restricted to a mask)."""
    n = int(mask.sum())
    if n <= 0:
        return float("nan")
    return 10.0 * math.log10(float((wav[mask].astype(np.float64) ** 2).sum()) / n + 1e-12)


def span_mask(n: int, spans, t0: float, t1: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in spans:
        i, j = int(max(a, t0) * SR), int(min(b, t1) * SR)
        if j > i:
            m[i:min(j, n)] = True
    return m


def frame_active(ref: np.ndarray, thr_db: float = 40.0) -> np.ndarray:
    """Per-sample activity mask from 25 ms / 10 ms frame energy > max - thr_db."""
    frame, hop = 400, 160
    n = (len(ref) - frame) // hop + 1
    if n <= 0:
        return np.ones(len(ref), dtype=bool)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    e = (ref[idx].astype(np.float64) ** 2).mean(axis=1)
    db = 10.0 * np.log10(e + 1e-12)
    act = db > (db.max() - thr_db)
    m = np.zeros(len(ref), dtype=bool)
    for i in np.nonzero(act)[0]:
        m[i * hop: i * hop + frame] = True
    return m


def fg_gain_db(enh, ref, mask):
    r = ref[mask].astype(np.float64)
    e = enh[mask].astype(np.float64)
    den = float((r * r).sum())
    if den <= 1e-12:
        return float("nan")
    g = float((e * r).sum()) / den
    return 10.0 * math.log10(g * g + 1e-12)


def profile(mix, enh, base_mask, t_start, t_end, ref=None):
    """base_mask: samples belonging to the keep region. t_start/t_end: region bounds in s."""
    n = min(len(mix), len(enh))
    mix, enh, base_mask = mix[:n], enh[:n], base_mask[:n]
    if ref is not None:
        ref = ref[:n]
    tgrid = np.arange(n) / SR
    out = {}
    for X in WINDOWS:
        if t_end - t_start < X + MIN_REST:
            out[X] = None
            continue
        on = base_mask & (tgrid >= t_start) & (tgrid < t_start + X)
        rest = base_mask & (tgrid >= t_start + X) & (tgrid < t_end)
        if on.sum() < 0.05 * SR or rest.sum() < 0.05 * SR:
            out[X] = None
            continue
        d_on = level_dbfs(enh, on) - level_dbfs(mix, on)
        d_rest = level_dbfs(enh, rest) - level_dbfs(mix, rest)
        rec = [d_on, d_rest, d_on - d_rest]
        if ref is not None:
            g_on, g_rest = fg_gain_db(enh, ref, on), fg_gain_db(enh, ref, rest)
            rec += [g_on, g_rest, g_on - g_rest]
        out[X] = tuple(rec)
    return out


def agg(vals):
    v = np.array([x for x in vals if x is not None and x == x])
    if len(v) == 0:
        return "     n/a", "     n/a", 0
    return f"{np.median(v):8.2f}", f"{np.percentile(v, 10):8.2f}", len(v)


def run_field(tag, rows_out):
    idx = [json.loads(l) for l in (CACHE / tag / "field_index.jsonl").read_text().splitlines() if l.strip()]
    per = {}
    for r in idx:
        if r["side"] not in ("near", "dt"):
            continue
        if r["condition"] not in ("none", "stream", "near_speech"):
            continue
        keep = r["spans"]["keep"]
        if not keep:
            continue
        off = r["offset_s"]
        keep = [[a + off, b + off] for a, b in keep]
        d = np.load(r["file"])
        mix = d["mix"].astype(np.float32)
        enh = d["enh"].astype(np.float32)
        n = min(len(mix), len(enh))
        base = span_mask(n, keep, -1e9, 1e9)
        # first keep span only (the cold onset)
        a, b = keep[0]
        p = profile(mix, enh, base, a, b)
        key = (r["condition"], r["side"])
        per.setdefault(key, []).append((r["clip"], p))
        for X in WINDOWS:
            v = p[X]
            rows_out.append(dict(tag=tag, world="field", subset=r["condition"], side=r["side"],
                                 item=r["clip"], window=X,
                                 d_onset=None if v is None else round(v[0], 3),
                                 d_rest=None if v is None else round(v[1], 3),
                                 excess=None if v is None else round(v[2], 3),
                                 span_len=round(b - a, 2)))
    return per


def run_dawn(tag, rows_out, limit=None):
    idx = [json.loads(l) for l in (CACHE / tag / "dawn_index.jsonl").read_text().splitlines() if l.strip()]
    idx = [r for r in idx if r["condition"] == "none"]
    if limit:
        idx = idx[:limit]
    per = {}
    for k, r in enumerate(idx):
        d = np.load(r["file"])
        mix = d["mix"].astype(np.float32)
        enh = d["enh"].astype(np.float32)
        ref = d["ref"].astype(np.float32)
        n = min(len(mix), len(enh), len(ref))
        mix, enh, ref = mix[:n], enh[:n], ref[:n]
        act = frame_active(ref)
        nz = np.nonzero(act)[0]
        if len(nz) < 0.2 * SR:
            continue
        t0, t1 = nz[0] / SR, (nz[-1] + 1) / SR
        p = profile(mix, enh, act, t0, t1, ref=ref)
        per.setdefault(("none", "dawn"), []).append((r["id"], p))
        for X in WINDOWS:
            v = p[X]
            extra = {} if v is None else dict(g_on=round(v[3], 3), g_rest=round(v[4], 3),
                                              g_excess=round(v[5], 3))
            rows_out.append(dict(tag=tag, world="dawn", subset="none", side="fg", item=r["id"],
                                 window=X, d_onset=None if v is None else round(v[0], 3),
                                 d_rest=None if v is None else round(v[1], 3),
                                 excess=None if v is None else round(v[2], 3),
                                 span_len=round(t1 - t0, 2), onset_s=round(t0, 2), **extra))
        if (k + 1) % 100 == 0:
            print(f"  dawn {k+1}/{len(idx)}", flush=True)
    return per


def main():
    rows = []
    tables = []
    for tag in ("v8", "v16"):
        pf = run_field(tag, rows)
        pd_ = run_dawn(tag, rows)
        for key in sorted(pf) + sorted(pd_):
            items = pf.get(key, pd_.get(key))
            line = [tag, key[0], key[1]]
            for X in WINDOWS:
                med, p10, n = agg([p[X][2] if p[X] else None for _, p in items])
                mo, _, _ = agg([p[X][0] if p[X] else None for _, p in items])
                mr, _, _ = agg([p[X][1] if p[X] else None for _, p in items])
                line += [mo.strip(), mr.strip(), med.strip(), p10.strip(), str(n)]
            tables.append(line)
    hdr = ["ckpt", "subset", "side"]
    for X in WINDOWS:
        hdr += [f"d_on@{X}", f"d_rest@{X}", f"excess_med@{X}", f"excess_p10@{X}", f"n@{X}"]
    with open(OUT / "real_onset_summary.tsv", "w") as f:
        f.write("\t".join(hdr) + "\n")
        for l in tables:
            f.write("\t".join(l) + "\n")
    with open(OUT / "real_onset_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\t".join(hdr))
    for l in tables:
        print("\t".join(l))


if __name__ == "__main__":
    main()
