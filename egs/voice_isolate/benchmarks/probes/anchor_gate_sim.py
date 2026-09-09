"""Step 2 of the anchor-qualification gate: does an inference-only supervisor help?

Reads ONLY the frame cache written by `anchor_gate_cache.py` (no model, no GPU) and asks
three questions in order:

  `readability` -- can the DistHead, read over a sliding window of speech-active frames,
                   tell a 30-50 cm keep talker from a 2-3 m suppress talker at all, and does
                   it read the CONTEXT PREFIX (the anchor) as near vs far correctly? If not,
                   the rest of this file is moot and says so.
  `sweep`       -- simulate the AnchorGate on the cached frames: a slow anchor estimate A
                   over the most recent >= 1 s of sustained speech (held through silence, no
                   decay), a current estimate C over the last W_c s of active frames, and
                   triggers (a) relative C < r*A, (b) absolute A > d_far / A undefined,
                   (c) either. Gain integrator (fast attack, slow release) blends the output
                   back to the dry mix:  out = g*mix + (1-g)*enh. `--feature` selects which
                   of the head's readouts drives A and C: `dist` (fg distance, m) or `drr`
                   (DRR, dB -- the readable one on the QVF chain, see `readability`).
  `dawn`        -- the same gate on the Dawn Chorus cache: SI-SDR vs the clean foreground and
                   an energy-deletion proxy over reference-active frames. No ASR.

Operating-point protocol (`presence_selfcal_README` Finding 4): choose on the FIT set only
(groups 90d / 270d / 0d and every qvf_* record); `180d` is held out and read ONCE.

Levels use `eval_realcase.span_dbfs`'s definition (mean-square over the concatenated spans, dB);
the gated variants accumulate the same quantity on the 10 ms frame grid the gain lives on, which
is exact up to rounding each span edge to a 10 ms block (validated against the sample-exact
function on the ungated arm -- see the `max |exact - block|` line the sweep prints).

Usage (from egs/voice_isolate):
  uv run python benchmarks/probes/anchor_gate_sim.py readability --cache <dir> --tags v8 v16 \
      --out <dir>/readability.json
  uv run python benchmarks/probes/anchor_gate_sim.py sweep --cache <dir> --tag v8 --out <dir>/sweep_v8.json
  uv run python benchmarks/probes/anchor_gate_sim.py report --sweep <dir>/sweep_v8.json
  uv run python benchmarks/probes/anchor_gate_sim.py dawn --cache <dir> --tag v8 --rule rel --r 0.7 \
      --wc 0.5 --tau-dn 2.0 --protect 1 --out <dir>/dawn_v8.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 100.0
HOP = 160
SR = 16000
ACTIVE_MARGIN_DB = 8.0          # over the record's floor
HANGOVER_S = 0.2                # bridge intra-utterance pauses
MIN_RUN_S = 0.10                # drop activity blips
ANCHOR_MIN_RUN_S = 1.0          # sustained speech before the anchor may update
ANCHOR_WIN_S = 1.0              # window the anchor is estimated over
ANCHOR_TAU_S = 2.0              # slow anchor integrator (log10 m), fixed
C_MIN_FRAMES = 20               # 0.2 s of active frames before C exists
SNAP = 1e-3                     # gain residual below which it snaps to exactly 0 or 1
KEEP_VIOLATION_DB = -3.0
SUPPRESS_PASS_DB = -6.0

DEVICE_GROUPS = {"0d", "90d", "180d", "270d"}
FIT_GROUPS = {"0d", "90d", "270d"}          # + every qvf_* group
HELD_OUT_GROUPS = {"180d"}
ANCHOR_CONDITIONS = ("near_speech", "far_speech", "stream", "floor")


# ------------------------------------------------------------------ features


def load_head(cache: Path, tag: str):
    import torch

    sd = torch.load(cache / tag / "dist_head_net.pt", map_location="cpu")
    w0 = sd["0.weight"].numpy().astype(np.float64)
    b0 = sd["0.bias"].numpy().astype(np.float64)
    w2 = sd["2.weight"].numpy().astype(np.float64)
    b2 = sd["2.bias"].numpy().astype(np.float64)
    return w0, b0, w2, b2


def head_apply(head, x: np.ndarray) -> np.ndarray:
    """x: [N, 128] -> [N, 3]. Linear -> SiLU -> Linear, exactly as DistHead.net."""
    w0, b0, w2, b2 = head
    h = x @ w0.T + b0
    h = h / (1.0 + np.exp(-h))
    return h @ w2.T + b2


def frame_energy_db(mix: np.ndarray, n_frames: int) -> np.ndarray:
    """20 ms frames on the 10 ms grid the bottleneck lives on."""
    x = mix.astype(np.float64)
    need = (n_frames - 1) * HOP + 2 * HOP
    if x.size < need:
        x = np.pad(x, (0, need - x.size))
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(2 * HOP)[None, :]
    e = (x[idx] ** 2).mean(axis=1)
    return 10.0 * np.log10(e + 1e-12)


def activity(e_db: np.ndarray, thr_db: float) -> np.ndarray:
    a = e_db > thr_db
    # bridge gaps <= HANGOVER_S
    gap = int(HANGOVER_S * FPS)
    out = a.copy()
    idx = np.flatnonzero(a)
    if idx.size:
        for i, j in zip(idx[:-1], idx[1:]):
            if 1 < j - i <= gap + 1:
                out[i:j] = True
    # drop runs shorter than MIN_RUN_S
    minr = int(MIN_RUN_S * FPS)
    if minr > 1:
        d = np.diff(np.concatenate(([0], out.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s < minr:
                out[s:e] = False
    return out


def sliding_estimates(feat: np.ndarray, act: np.ndarray, head, windows) -> dict:
    """For each W in `windows`: per-frame fg distance (m) and DRR (dB) from
    MLP(mean of feat over the last W s of ACTIVE frames), NaN on inactive frames or
    before C_MIN_FRAMES active frames exist. feat is [C, F]."""
    F = feat.shape[1]
    ai = np.flatnonzero(act)
    out = {}
    if ai.size == 0:
        for W in windows:
            out[W] = (np.full(F, np.nan), np.full(F, np.nan))
        return out
    X = feat[:, ai].T.astype(np.float64)                    # [n_act, C]
    cs = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    for W in windows:
        n = max(1, int(round(W * FPS)))
        k = np.arange(1, ai.size + 1)
        lo = np.maximum(0, k - n)
        cnt = (k - lo)[:, None]
        means = (cs[k] - cs[lo]) / cnt
        y = head_apply(head, means)                          # [n_act, 3]
        fg = np.clip(10.0 ** y[:, 1], 0.1, 30.0)
        drr = y[:, 0] * 10.0
        valid = (k - lo) >= min(C_MIN_FRAMES, n)
        f_fg = np.full(F, np.nan)
        f_drr = np.full(F, np.nan)
        f_fg[ai] = np.where(valid, fg, np.nan)
        f_drr[ai] = np.where(valid, drr, np.nan)
        out[W] = (f_fg, f_drr)
    return out


def anchor_track(feat: np.ndarray, act: np.ndarray, head) -> tuple:
    """A (fg distance, m) per frame: a tau=2 s integrator in log10 m over the ANCHOR_WIN_S
    window, updated only while the contiguous active run is >= ANCHOR_MIN_RUN_S, held
    (frozen) otherwise. Returns (A_m [F], A_drr_db [F]); NaN before the first update."""
    F = feat.shape[1]
    A = np.full(F, np.nan)
    Ad = np.full(F, np.nan)
    if not act.any():
        return A, Ad
    # contiguous-run length in frames
    run = np.zeros(F, dtype=np.int64)
    c = 0
    for t in range(F):
        c = c + 1 if act[t] else 0
        run[t] = c
    upd = np.flatnonzero(run >= int(ANCHOR_MIN_RUN_S * FPS))
    if upd.size == 0:
        return A, Ad
    n = int(ANCHOR_WIN_S * FPS)
    X = feat.T.astype(np.float64)
    cs = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)], axis=0)
    lo = upd + 1 - n
    means = (cs[upd + 1] - cs[lo]) / n
    y = head_apply(head, means)
    obs_log = np.log10(np.clip(10.0 ** y[:, 1], 0.1, 30.0))
    obs_drr = y[:, 0] * 10.0
    a = 1.0 - math.exp(-1.0 / (ANCHOR_TAU_S * FPS))
    cur, curd = obs_log[0], obs_drr[0]
    prev = int(upd[0])
    A[prev], Ad[prev] = 10.0 ** cur, curd
    for k in range(1, upd.size):
        t = int(upd[k])
        A[prev + 1:t] = A[prev]          # held through silence, no decay
        Ad[prev + 1:t] = Ad[prev]
        cur = cur + a * (obs_log[k] - cur)
        curd = curd + a * (obs_drr[k] - curd)
        A[t], Ad[t] = 10.0 ** cur, curd
        prev = t
    A[prev + 1:] = A[prev]
    Ad[prev + 1:] = Ad[prev]
    return A, Ad


# ------------------------------------------------------------------ record


@dataclass
class Record:
    meta: dict
    F: int
    L: int
    act: np.ndarray
    A: np.ndarray      # anchor, fg distance (m)
    Adrr: np.ndarray   # anchor, DRR (dB)
    C: dict            # W -> fg distance per frame
    Cdrr: dict         # W -> DRR (dB) per frame
    Smm: np.ndarray    # per 10 ms block sums, length F
    Sme: np.ndarray
    See: np.ndarray
    keep_blocks: np.ndarray
    supp_blocks: np.ndarray
    exact: dict        # sample-exact span dBFS for mix / enh


def block_sums(mix: np.ndarray, enh: np.ndarray, F: int):
    L = min(mix.size, enh.size, F * HOP)
    nb = L // HOP
    m = mix[: nb * HOP].astype(np.float64).reshape(nb, HOP)
    e = enh[: nb * HOP].astype(np.float64).reshape(nb, HOP)
    return (m * m).sum(1), (m * e).sum(1), (e * e).sum(1), nb, L


def span_blocks(spans, offset_s: float, nb: int) -> np.ndarray:
    idx = np.zeros(nb, dtype=bool)
    for a, b in spans:
        i = int(round((a + offset_s) * FPS))
        j = int(round((b + offset_s) * FPS))
        i, j = max(0, i), min(nb, j)
        if j > i:
            idx[i:j] = True
    return idx


def span_dbfs_exact(x: np.ndarray, spans, offset_s: float, limit: int) -> float:
    tot = n = 0.0
    for a, b in spans:
        i, j = int((a + offset_s) * SR), min(int((b + offset_s) * SR), limit)
        if j <= i:
            continue
        seg = x[i:j].astype(np.float64)
        tot += float((seg * seg).sum())
        n += j - i
    if n <= 0:
        return float("nan")
    return 10.0 * math.log10(tot / n + 1e-12)


def load_record(entry: dict, head, windows) -> Record:
    d = np.load(entry["file"], allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    feat = d["feat"].astype(np.float32)
    F = feat.shape[1]
    mix = d["mix"]
    enh = d["enh"]
    e_db = frame_energy_db(mix, F)
    floor = meta.get("floor_dbfs")
    thr = (floor if floor is not None else np.percentile(e_db, 5)) + ACTIVE_MARGIN_DB
    act = activity(e_db, thr)
    A, Adrr = anchor_track(feat, act, head)
    est = sliding_estimates(feat, act, head, windows)
    C = {W: est[W][0] for W in windows}
    Cd = {W: est[W][1] for W in windows}
    Smm, Sme, See, nb, L = block_sums(mix, enh, F)
    keep = span_blocks(meta["spans"].get("keep") or [], meta["offset_s"], nb)
    supp = span_blocks(meta["spans"].get("suppress") or [], meta["offset_s"], nb)
    exact = {}
    for kind, spans in (("keep", meta["spans"].get("keep") or []),
                        ("suppress", meta["spans"].get("suppress") or [])):
        if spans:
            exact[f"mix_{kind}"] = span_dbfs_exact(mix, spans, meta["offset_s"], L)
            exact[f"enh_{kind}"] = span_dbfs_exact(enh, spans, meta["offset_s"], L)
    return Record(meta=meta, F=F, L=L, act=act[:nb], A=A[:nb], Adrr=Adrr[:nb],
                  C={W: C[W][:nb] for W in windows},
                  Cdrr={W: Cd[W][:nb] for W in windows},
                  Smm=Smm, Sme=Sme, See=See,
                  keep_blocks=keep, supp_blocks=supp, exact=exact)


# ------------------------------------------------------------------ the gate


def gate_gain(rec: Record, *, rule: str, r: float, Wc: float, d_far: float,
              tau_dn: float, protect_no_anchor: bool, feature: str = "dist",
              tau_up: float = 0.05) -> np.ndarray:
    """Per-frame blend gain g in [0, 1]; 1 = dry mix, 0 = enhanced.

    ``feature="dist"``: A and C are the head's fg distance in m; the current talker reads
    nearer when ``C < r * A``, the anchor is far when ``A > d_far`` (m).
    ``feature="drr"``: A and C are the head's DRR in dB; nearer means ``C > A + r`` (r is a
    margin in dB), and the anchor is far when ``A < d_far`` (dB).
    """
    nb = rec.Smm.size
    if feature == "drr":
        C = rec.Cdrr[Wc]
        A = rec.Adrr
    else:
        C = rec.C[Wc]
        A = rec.A
    act = rec.act
    init = 1.0 if protect_no_anchor else 0.0
    has_a = np.isfinite(A)
    if feature == "drr":
        Af = np.where(has_a, A, -np.inf)
        rel = np.isfinite(C) & (C > Af + r)
        ab = Af < d_far
    else:
        Af = np.where(has_a, A, np.inf)
        rel = np.isfinite(C) & (C < r * Af)
        ab = Af > d_far
    if rule == "rel":
        trig = rel
    elif rule == "abs":
        trig = ab
    else:
        trig = rel | ab
    raw = np.where(has_a, trig.astype(np.float64), init)
    # the decision is taken on active frames only and held through silence
    last = np.where(act, np.arange(nb), -1)
    np.maximum.accumulate(last, out=last)
    target = np.where(last >= 0, raw[np.maximum(last, 0)], init)
    a_up = 1.0 - math.exp(-1.0 / (tau_up * FPS))
    a_dn = 1.0 - math.exp(-1.0 / (tau_dn * FPS))
    g = np.empty(nb)
    cur = target[0]
    # segment-wise closed form: target is piecewise constant
    edges = np.flatnonzero(np.diff(target) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [nb]))
    for s, e in zip(starts, ends):
        v = target[s]
        n = e - s
        if abs(cur - v) <= SNAP:
            g[s:e] = v
            cur = v
            continue
        al = a_up if v > cur else a_dn
        k = np.arange(1, n + 1)
        traj = v + (cur - v) * (1.0 - al) ** k
        traj = np.where(np.abs(traj - v) <= SNAP, v, traj)
        g[s:e] = traj
        cur = g[e - 1]
    return g


def span_level(rec: Record, blocks: np.ndarray, g: np.ndarray | None) -> float:
    n = int(blocks.sum()) * HOP
    if n == 0:
        return float("nan")
    if g is None:
        tot = rec.See[blocks].sum()
    else:
        gg = g[blocks]
        tot = (gg * gg * rec.Smm[blocks] + 2.0 * gg * (1.0 - gg) * rec.Sme[blocks]
               + (1.0 - gg) ** 2 * rec.See[blocks]).sum()
    return 10.0 * math.log10(tot / n + 1e-12)


def mix_level(rec: Record, blocks: np.ndarray) -> float:
    n = int(blocks.sum()) * HOP
    if n == 0:
        return float("nan")
    return 10.0 * math.log10(rec.Smm[blocks].sum() / n + 1e-12)


# ------------------------------------------------------------------ commands


def read_index(cache: Path, tag: str, which: str) -> list:
    p = cache / tag / f"{which}_index.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def group_of(clip: str) -> str:
    for tail in ("_far", "_near", "_dt", "_session"):
        if tail in clip:
            return clip[: clip.rindex(tail)]
    return clip


def chain_of(group: str) -> str:
    return "device" if group in DEVICE_GROUPS else "qvf"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: pos.size].sum()
    return (rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def sliding_proximity(values, act, windows):
    """Average direct ProximityHead readouts over trailing active-frame windows.

    No DistHead MLP, metres conversion, or absolute near threshold is involved.
    """
    n = min(len(values), len(act)); values, act = values[:n], act[:n]
    ai = np.flatnonzero(act & np.isfinite(values)); out = {}
    cs = np.r_[0., np.cumsum(values[ai], dtype=float)]
    k = np.arange(1, len(ai)+1)
    for window in windows:
        width = max(1, round(window*FPS)); lo = np.maximum(0, k-width)
        result = np.full(n, np.nan)
        valid = k-lo >= min(C_MIN_FRAMES, width)
        result[ai[valid]] = ((cs[k]-cs[lo]) / (k-lo))[valid]
        out[window] = result
    return out


def cmd_proximity_readability(args):
    cache, windows = Path(args.cache), [0.5, 1., 2.]
    out = {"head": "proximity", "units": "raw scalar", "near_direction": "higher",
           "windows": windows, "tags": {}}
    for tag in args.tags:
        pools, prefixes = {}, {}
        for entry in read_index(cache, tag, "field"):
            with np.load(entry['file']) as data:
                if 'proximity' not in data:
                    raise ValueError(f"cache lacks proximity; recache with --head proximity: {entry['file']}")
                values, mix = data['proximity'], data['mix']
                meta = json.loads(str(data['meta']))
            floor = meta.get('floor_dbfs')
            energy = frame_energy_db(mix, len(values))
            if floor is None:
                floor = float(np.percentile(energy, 10))
            estimates = sliding_proximity(values, activity(energy, floor+ACTIVE_MARGIN_DB), windows)
            chain = chain_of(group_of(entry['clip']))
            scope = 'session' if entry['side'] == 'session' else 'clip'
            offset = float(meta.get('offset_s', 0.))
            for window, estimate in estimates.items():
                if entry['condition'] == 'none' and entry['side'] != 'dt':
                    for kind in ('keep', 'suppress'):
                        mask = np.zeros(len(estimate), dtype=bool)
                        for start, end in meta.get('spans', {}).get(kind, []):
                            a, b = max(0, round((start+offset)*FPS)), min(len(mask), round((end+offset)*FPS))
                            mask[a:b] = True
                        valid = estimate[mask & np.isfinite(estimate)]
                        pools.setdefault((chain, scope, window, kind), []).extend(valid.tolist())
                elif offset > 0:
                    valid = estimate[:round(offset*FPS)]; valid = valid[np.isfinite(valid)]
                    if valid.size:
                        prefixes.setdefault((chain, entry['condition'], window), []).append(float(np.median(valid)))
        rows = []
        for chain, scope, window in sorted({key[:3] for key in pools}):
            keep = np.asarray(pools.get((chain, scope, window, 'keep'), []))
            suppress = np.asarray(pools.get((chain, scope, window, 'suppress'), []))
            rows.append(dict(chain=chain, scope=scope, W=window,
                auc_proximity=auc(keep, suppress) if keep.size and suppress.size else None,
                n_keep_frames=int(keep.size), n_supp_frames=int(suppress.size),
                median_keep=float(np.median(keep)) if keep.size else None,
                median_suppress=float(np.median(suppress)) if suppress.size else None))
        out['tags'][tag] = {'auc': rows, 'prefix': [dict(chain=ch, condition=cond, W=w,
            n=len(vals), median=float(np.median(vals)), p10=float(np.percentile(vals, 10)),
            p90=float(np.percentile(vals, 90))) for (ch,cond,w),vals in sorted(prefixes.items())]}
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, indent=2, allow_nan=False))


def cmd_readability(args):
    if getattr(args, "head", "dist") == "proximity":
        return cmd_proximity_readability(args)
    cache = Path(args.cache)
    windows = [0.5, 1.0, 2.0]
    out = {"windows": windows, "tags": {}}
    for tag in args.tags:
        head = load_head(cache, tag)
        index = [e for e in read_index(cache, tag, "field")]
        pools = {}      # (chain, kind, W) -> list of arrays
        dpools = {}
        prefix = {}     # (chain, condition, W) -> list of per-record medians
        anom = []
        for e in index:
            if e["condition"] != "none" and e["condition"] not in ANCHOR_CONDITIONS \
                    and e["condition"] != "event":
                continue
            rec = load_record(e, head, windows)
            g = group_of(e["clip"])
            ch = chain_of(g)
            side = e["side"]
            if not np.isfinite(rec.A).any():
                anom.append(f"{tag}:{e['clip']}:{e['variant']}: anchor never armed")
            if e["condition"] == "none":
                for W in windows:
                    C = rec.C[W]
                    for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
                        if not blocks.any():
                            continue
                        if side == "dt":
                            continue
                        key = (ch, "session" if side == "session" else "clip", kind, W)
                        v = C[: blocks.size][blocks]
                        pools.setdefault(key, []).append(v[np.isfinite(v)])
                        vd = rec.Cdrr[W][: blocks.size][blocks]
                        dpools.setdefault(key, []).append(vd[np.isfinite(vd)])
            if e["condition"] in ("near_speech", "far_speech", "stream", "floor", "event"):
                off = int(round(e["offset_s"] * FPS))
                if off > 0:
                    for W in windows:
                        v = rec.C[W][:off]
                        v = v[np.isfinite(v)]
                        if v.size:
                            vd = rec.Cdrr[W][:off]
                            vd = vd[np.isfinite(vd)]
                            prefix.setdefault((ch, e["condition"], W), []).append(
                                (e["clip"], float(np.median(v)),
                                 float(np.median(vd)) if vd.size else float("nan")))
        rows = []
        for ch in ("device", "qvf"):
            for scope in ("clip", "session"):
                for W in windows:
                    pk = pools.get((ch, scope, "keep", W))
                    ps = pools.get((ch, scope, "suppress", W))
                    if not pk or not ps:
                        continue
                    K = np.concatenate(pk)
                    S = np.concatenate(ps)
                    Kd = np.concatenate(dpools[(ch, scope, "keep", W)])
                    Sd = np.concatenate(dpools[(ch, scope, "suppress", W)])
                    rows.append({"chain": ch, "scope": scope, "W": W,
                                 "auc_dist": auc(-K, -S), "auc_drr": auc(Kd, Sd),
                                 "median_keep_drr": float(np.median(Kd)),
                                 "median_supp_drr": float(np.median(Sd)),
                                 "n_keep_frames": int(K.size), "n_supp_frames": int(S.size),
                                 "median_keep_m": float(np.median(K)),
                                 "median_supp_m": float(np.median(S))})
        pref = []
        for (ch, cond, W), lst in sorted(prefix.items()):
            vals = np.array([v for _, v, _ in lst])
            dvals = np.array([d for _, _, d in lst])
            pref.append({"chain": ch, "condition": cond, "W": W, "n": int(vals.size),
                         "median_m": float(np.median(vals)),
                         "median_drr_db": float(np.nanmedian(dvals)),
                         "p10_m": float(np.percentile(vals, 10)),
                         "p90_m": float(np.percentile(vals, 90))})
        out["tags"][tag] = {"auc": rows, "prefix": pref, "anomalies": anom}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _fit(e) -> bool:
    g = group_of(e["clip"])
    return g in FIT_GROUPS or g.startswith("qvf")


def cmd_sweep(args):
    cache = Path(args.cache)
    tag = args.tag
    head = load_head(cache, tag)
    windows = [0.5, 1.0, 2.0]
    index = read_index(cache, tag, "field")
    if args.conditions:
        index = [e for e in index if e["condition"] in args.conditions]
    if args.split == "fit":
        index = [e for e in index if _fit(e)]
    elif args.split == "held":
        index = [e for e in index if group_of(e["clip"]) in HELD_OUT_GROUPS]
    configs = []
    feat_name = args.feature
    for tau_dn in args.tau_dn:
        for pna in args.protect:
            for d in args.d_far:
                configs.append(dict(rule="abs", r=0.0, Wc=1.0, d_far=d, tau_dn=tau_dn,
                                    protect_no_anchor=bool(pna), feature=feat_name))
            for r in args.r:
                for Wc in args.wc:
                    configs.append(dict(rule="rel", r=r, Wc=Wc,
                                        d_far=(-1e9 if feat_name == "drr" else 1e9),
                                        tau_dn=tau_dn, protect_no_anchor=bool(pna),
                                        feature=feat_name))
                    for d in args.d_far:
                        configs.append(dict(rule="either", r=r, Wc=Wc, d_far=d,
                                            tau_dn=tau_dn, protect_no_anchor=bool(pna),
                                            feature=feat_name))
    print(f"{tag}: {len(index)} records x {len(configs)} configs", flush=True)
    rows = []
    max_err = 0.0
    anomalies = []
    for i, e in enumerate(index):
        rec = load_record(e, head, windows)
        base = {"clip": e["clip"], "group": group_of(e["clip"]), "side": e["side"],
                "condition": e["condition"], "variant": e["variant"],
                "held_out": bool(e.get("held_out")), "sentinel": bool(e.get("sentinel"))}
        for kind, blocks in (("keep", rec.keep_blocks), ("suppress", rec.supp_blocks)):
            if not blocks.any():
                continue
            mx = mix_level(rec, blocks)
            en = span_level(rec, blocks, None)
            ex_m = rec.exact.get(f"mix_{kind}")
            ex_e = rec.exact.get(f"enh_{kind}")
            if ex_m is not None and np.isfinite(ex_m):
                max_err = max(max_err, abs(ex_m - mx), abs(ex_e - en))
            r0 = {**base, "kind": kind, "mix_dbfs": mx, "ungated": en - mx}
            for ci, cfg in enumerate(configs):
                g = gate_gain(rec, **cfg)
                r0[f"c{ci}"] = span_level(rec, blocks, g) - mx
            rows.append(r0)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(index)}", flush=True)
    out = {"tag": tag, "split": args.split, "configs": configs, "rows": rows,
           "max_block_vs_exact_db": max_err, "anomalies": anomalies}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}; max |exact - block| = {max_err:.4f} dB")


KEEP_OBJ_CONDS = ("near_speech", "far_speech", "stream", "floor")
SUPP_OBJ_CONDS = ("near_speech", "event", "stream")


def objective(rows, key):
    """(keep violations over the anchor conditions + sessions,
        suppression over the conditions where the anchor unlocks it + sessions)."""
    kv = kn = 0
    kvals = []
    sp = sn = 0
    svals = []
    for r in rows:
        v = r[key]
        if not np.isfinite(v):
            continue
        is_sess = r["side"] == "session"
        if r["kind"] == "keep" and (is_sess or r["condition"] in KEEP_OBJ_CONDS):
            kn += 1
            kvals.append(v)
            kv += int(v < KEEP_VIOLATION_DB)
        if r["kind"] == "suppress" and (is_sess or r["condition"] in SUPP_OBJ_CONDS):
            sn += 1
            svals.append(v)
            sp += int(v <= SUPPRESS_PASS_DB)
    return {"kv": kv, "kn": kn,
            "kmed": float(np.median(kvals)) if kvals else float("nan"),
            "sp": sp, "sn": sn,
            "smed": float(np.median(svals)) if svals else float("nan"),
            "smean": float(np.mean(svals)) if svals else float("nan")}


def cmd_report(args):
    data = json.loads(Path(args.sweep).read_text())
    rows, configs = data["rows"], data["configs"]
    arms = [("ungated", None, objective(rows, "ungated"))]
    for ci, cfg in enumerate(configs):
        arms.append((f"c{ci}", cfg, objective(rows, f"c{ci}")))
    ung = arms[0][2]
    arms_sorted = sorted(arms[1:], key=lambda x: (x[2]["kv"], -x[2]["sp"], -x[2]["smed"]))
    print(f"# {data['tag']} split={data['split']}  keep-obj conds={KEEP_OBJ_CONDS}+sessions  "
          f"supp-obj conds={SUPP_OBJ_CONDS}+sessions")
    hdr = f"{'arm':>6} {'kviol':>7} {'kmed':>7} {'spass':>7} {'smed':>7} {'smean':>7}  config"
    print(hdr)
    print(f"{'ungated':>6} {str(ung['kv'])+'/'+str(ung['kn']):>7} {ung['kmed']:>7.2f} "
          f"{str(ung['sp'])+'/'+str(ung['sn']):>7} {ung['smed']:>7.2f} {ung['smean']:>7.2f}")
    for name, cfg, s in arms_sorted[: args.top]:
        c = (f"feat={cfg.get('feature', 'dist')} rule={cfg['rule']} r={cfg['r']} "
             f"Wc={cfg['Wc']} d_far={cfg['d_far']:g} tau_dn={cfg['tau_dn']} "
             f"pna={int(cfg['protect_no_anchor'])}")
        print(f"{name:>6} {str(s['kv'])+'/'+str(s['kn']):>7} {s['kmed']:>7.2f} "
              f"{str(s['sp'])+'/'+str(s['sn']):>7} {s['smed']:>7.2f} {s['smean']:>7.2f}  {c}")
    if args.detail:
        for name in args.detail:
            key = name
            print(f"\n## per condition/side, arm={key}")
            print(f"{'condition':>12} {'side':>8} {'kind':>9} {'n':>4} {'ungated':>8} {'gated':>8} "
                  f"{'kv_u':>5} {'kv_g':>5} {'sp_u':>5} {'sp_g':>5}")
            buckets = {}
            for r in rows:
                buckets.setdefault((r["condition"], r["side"], r["kind"]), []).append(r)
            for k in sorted(buckets):
                rs = buckets[k]
                u = np.array([r["ungated"] for r in rs], float)
                g = np.array([r[key] for r in rs], float)
                if k[2] == "keep":
                    a, b = int((u < KEEP_VIOLATION_DB).sum()), int((g < KEEP_VIOLATION_DB).sum())
                    c = d = 0
                else:
                    a = b = 0
                    c, d = int((u <= SUPPRESS_PASS_DB).sum()), int((g <= SUPPRESS_PASS_DB).sum())
                print(f"{k[0]:>12} {k[1]:>8} {k[2]:>9} {len(rs):>4} {np.median(u):>8.2f} "
                      f"{np.median(g):>8.2f} {a:>5} {b:>5} {c:>5} {d:>5}")


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est.astype(np.float64)
    ref = ref.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float((ref * ref).sum())
    if denom <= 0:
        return float("nan")
    a = float((est * ref).sum()) / denom
    t = a * ref
    e = est - t
    num = float((t * t).sum())
    den = float((e * e).sum())
    if num <= 0 or den <= 0:
        return float("nan")
    return 10.0 * math.log10(num / den)


def cmd_dawn(args):
    cache = Path(args.cache)
    head = load_head(cache, args.tag)
    cfg = dict(rule=args.rule, r=args.r, Wc=args.wc, d_far=args.d_far,
               tau_dn=args.tau_dn, protect_no_anchor=bool(args.protect), feature=args.feature)
    index = read_index(cache, args.tag, "dawn")
    if args.limit:
        index = index[: args.limit]
    rows = []
    anomalies = []
    for i, e in enumerate(index):
        d = np.load(e["file"], allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        feat = d["feat"].astype(np.float32)
        F = feat.shape[1]
        mix, enh, ref = d["mix"], d["enh"], d["ref"]
        e_db = frame_energy_db(mix, F)
        act = activity(e_db, float(np.percentile(e_db, 5)) + ACTIVE_MARGIN_DB)
        A, Adrr = anchor_track(feat, act, head)
        _est = sliding_estimates(feat, act, head, [args.wc])[args.wc]
        C, Cdrr = _est
        Smm, Sme, See, nb, L = block_sums(mix, enh, F)
        rec = Record(meta=meta, F=F, L=L, act=act[:nb], A=A[:nb], Adrr=Adrr[:nb],
                     C={args.wc: C[:nb]},
                     Smm=Smm, Sme=Sme, See=See,
                     Cdrr={args.wc: Cdrr[:nb]},
                     keep_blocks=np.zeros(nb, bool), supp_blocks=np.zeros(nb, bool), exact={})
        g = gate_gain(rec, **cfg)
        L = min(mix.size, enh.size)          # the STFT tail is short of `mix` by ~1 frame
        gs = np.repeat(g, HOP)
        if gs.size < L:
            gs = np.concatenate([gs, np.full(L - gs.size, gs[-1])])
        gs = gs[:L]
        m = mix[:L].astype(np.float64)
        h = enh[:L].astype(np.float64)
        out = gs * m + (1.0 - gs) * h
        off = int(round(meta["offset_s"] * SR))
        n = min(ref.size, L - off)
        if n <= 0:
            anomalies.append(f"{e['id']}:{e['condition']}: no clip region (off={off}, L={L})")
            continue
        if n < ref.size:
            anomalies.append(f"{e['id']}:{e['condition']}: clip {ref.size - n} samples short")
        R = ref[:n].astype(np.float64)
        # deletion proxy over reference-active frames
        rf = R[: (n // HOP) * HOP].reshape(-1, HOP)
        re_db = 10.0 * np.log10((rf * rf).mean(1) + 1e-12)
        ra = re_db > (re_db.max() - 40.0)
        row = {"id": e["id"], "condition": e["condition"],
               "sisdr_mix": si_sdr(m[off:off + n], R),
               "sisdr_enh": si_sdr(h[off:off + n], R),
               "sisdr_out": si_sdr(out[off:off + n], R)}
        for name, sig in (("enh", h), ("out", out)):
            num = ((sig[off:off + (n // HOP) * HOP].reshape(-1, HOP) ** 2).sum(1)[ra]).sum()
            den = ((m[off:off + (n // HOP) * HOP].reshape(-1, HOP) ** 2).sum(1)[ra]).sum()
            row[f"del_{name}"] = 10.0 * math.log10(num / den + 1e-12) if den > 0 else float("nan")
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(index)}", flush=True)
    out = {"tag": args.tag, "config": cfg, "rows": rows, "anomalies": anomalies}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}  ({len(rows)} rows, {len(anomalies)} anomalies)")
    for cond in ("none", "background"):
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        f = lambda k: np.array([r[k] for r in rs], float)
        print(f"{cond:>11}  n={len(rs)}  SI-SDR mix {np.median(f('sisdr_mix')):6.2f} | "
              f"enh {np.median(f('sisdr_enh')):6.2f} -> gated {np.median(f('sisdr_out')):6.2f}   "
              f"del-proxy enh {np.median(f('del_enh')):6.2f} -> gated {np.median(f('del_out')):6.2f} dB "
              f"(mean {np.mean(f('del_enh')):6.2f} -> {np.mean(f('del_out')):6.2f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("readability")
    s.add_argument("--cache", required=True)
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--head", choices=["dist", "proximity"], default="dist")
    s.set_defaults(fn=cmd_readability)

    s = sub.add_parser("sweep")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--split", default="fit", choices=["fit", "held", "all"])
    s.add_argument("--conditions", nargs="*", default=None)
    s.add_argument("--r", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    s.add_argument("--wc", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    s.add_argument("--d-far", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    s.add_argument("--tau-dn", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    s.add_argument("--protect", nargs="+", type=int, default=[0, 1])
    s.add_argument("--feature", default="dist", choices=["dist", "drr"])
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("report")
    s.add_argument("--sweep", required=True)
    s.add_argument("--conditions", nargs="*", default=None)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--detail", nargs="*", default=None)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("dawn")
    s.add_argument("--cache", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rule", default="rel", choices=["rel", "abs", "either"])
    s.add_argument("--r", type=float, default=0.7)
    s.add_argument("--wc", type=float, default=0.5)
    s.add_argument("--d-far", type=float, default=1e9)
    s.add_argument("--tau-dn", type=float, default=2.0)
    s.add_argument("--protect", type=int, default=1)
    s.add_argument("--feature", default="dist", choices=["dist", "drr"])
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_dawn)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
