"""Reference-provenance matrix: WHOSE room tone unlocks a cold-start clip, and how much is enough?

Why (2026-09-03). The "+11..13 dB from 3 s of room ambience" result (v17_round_design.md §2b,
COLDSTART_V2.md) rests on ONE recording: the only >=3 s un-annotated gap in set v3 is 90D's
98.4-107.0 s, so every `ambient` far row is a 90D clip primed with 90D tone. Before the
decision problem is reframed around "give the model a reference", three things must be
measured, all eval-only, on checkpoint BLOCKS (FIELD_BLOCK.md):

  Q1 provenance  -- own recording vs same room/other recording vs same chain/other room vs
                    other chain vs digital silence. If any real tone works, the mechanism is a
                    generic "acoustic prior" and not a room reference; if only own/same-room
                    tone works, it is a reference and the QVF clips need THEIR room's tone.
  Q2 keep side   -- the same matrix on lone-near and double-talk clips: does priming also
                    heal the keep violations (0d sentinel, qvf_price_near1, keep_in_touch)?
  Q3 how long    -- pad-length sweep 0.25..8 s of own tone (90D), the calibration-speed curve
                    that decides whether an explicit reference architecture is worth building.

Plus `stream`: the audio that actually preceded the span in its source recording (up to 3 s,
whatever it contains -- the deployment "state was never reset" truth), with the content of
that prefix logged (fractions of keep / suppress / gap).

Rows carry the per-checkpoint value; summaries are block means per (clip, condition).
dry_blend 1.0 throughout (same as the v2 standings).

Usage (from egs/voice_isolate):
  uv run python benchmarks/probes/reference_matrix_probe.py score config/infer_dpcrn.yaml \
      --ckpt <5 ckpts> --tag v8_block --out <json> --device cuda:0 [--sweep]
  uv run python benchmarks/probes/reference_matrix_probe.py report A.json [B.json ...]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_DIR = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RECIPE_DIR / "scripts"))

from puresound.audio.io import AudioIO           # noqa: E402
from puresound.config import load_recipe         # noqa: E402
from puresound.recipes import init_siso_model    # noqa: E402
from eval_realcase import span_dbfs              # noqa: E402

CASES = RECIPE_DIR / "data_report/field_cases/test_vector_cases"
TEST_VEC = RECIPE_DIR / "test_vec"
SR = 16000
PAD = 3.0
DRAWS = 3
STEADY_FROM = 3.0
SWEEP_LENS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

# What the un-annotated gaps of set v3 actually contain (checked 2026-09-03 by level profile +
# whisper): the 90D gap 98.4-107.0 holds an UNLABELLED UTTERANCE at 101.4-103.9 s (-47 dBFS,
# Mandarin, "對隨便隨便隨便唸"); its quiet parts sit at -62..-69. The 180D gap 62.9-64.9 is true floor
# (-79). qvf_gym's tail 29.7-34.1 holds speech too. So "room tone" pads drawn from the gaps were a
# mixture of floor and speech -- the conditions below separate the two explicitly.
TONE = {   # true quiet floor, no speech: (file, t0, t1)
    "dev_floor_180d": ("90d/180d", "180d_session_raw.wav", 62.99, 64.83),     # -79 dBFS, 1.84 s
    "dev_floor_90d":  ("90d/180d", "90d_session_raw.wav", 104.9, 107.0),      # -61 dBFS, 2.1 s
    "qvf_floor_gym":  ("qvf", "qvf_gym_session_raw.wav", 0.1, 1.82),          # -47 dBFS, 1.7 s
    "qvf_floor_kit":  ("qvf", "test_vec:QVF_keep_in_touch.wav", 0.0, 2.2),   # -55 dBFS, 2.2 s
}
EVENT = ("90d_session_raw.wav", 101.4, 103.9)   # the unlabelled utterance in the 90D gap
SPEECH = {  # first 3 s of a labelled span from ANOTHER recording of the same room / same chain
    "dev": {"near": {"90d": "180d_near1", "270d": "180d_near1", "0d": "180d_near1", "180d": "90d_near1"},
            "far":  {"90d": "180d_far1",  "270d": "180d_far1",  "0d": "180d_far1",  "180d": "90d_far1"}},
    "qvf": {"near": {"_default": "qvf_gym_near1", "qvf_gym": "qvf_price_near2"},
            "far":  {"_default": "qvf_gym_far1",  "qvf_gym": "qvf_price_far2"}},
}
# Where each cold-start clip's source recording lives (for the `stream` prefix).
SOURCE_FILES = {
    "0d": "0D_capmfx_2025_5_21-10_56_42_0000021545EFC860.wav",
    "90d": "90D_capmfx_2026_3_17-14_20_33_0000014340D33DE0.wav",
    "180d": "180D_capmfx_2026_3_17-14_24_20_0000014340D33DE0.wav",
    "270d": "270D_capmfx_2026_3_17-14_29_0_0000014340D33DE0.wav",
    "qvf_gym": "QVF_gym.wav", "qvf_price": "QVF_price.wav", "qvf_scenario1": "QVF_scenario1.wav",
    "qvf_scenario2": "QVF_scenario2.wav", "qvf_scenario3": "QVF_scenario3.wav",
    "qvf_keep_in_touch": "QVF_keep_in_touch.wav", "qvf_plumbing": "QVF_plumbing.wav",
}
DEVICE_GROUPS = {"0d", "90d", "180d", "270d"}
SWEEP_LENS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
DECAY_GAPS = [2.0, 5.0, 10.0, 20.0]

_wav_cache: dict = {}


def load_mono(path: Path) -> torch.Tensor:
    key = str(path)
    if key not in _wav_cache:
        wav, _ = AudioIO.open(f_path=key, target_lvl=None, resample_to=SR)
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        _wav_cache[key] = wav.view(-1)
    return _wav_cache[key]


def cut(fname: str, t0: float, t1: float) -> torch.Tensor:
    path = TEST_VEC / fname[9:] if fname.startswith("test_vec:") else CASES / fname
    return load_mono(path)[int(t0 * SR): int(t1 * SR)].clone()


def clip_head(clip: str, seconds: float) -> torch.Tensor:
    return load_mono(CASES / f"{clip}_raw.wav")[: int(seconds * SR)].clone()


def group_of(clip: str) -> str:
    for tail in ("_far", "_near", "_dt", "_session"):
        if tail in clip:
            return clip[: clip.rindex(tail)]
    return clip


def stream_prefix(clip: str, spec: dict, windows: dict) -> tuple:
    """Up to 3 s of the source recording immediately before the clip's span, plus what it
    contained (fractions of keep / suppress spans of the session labelling / unlabelled)."""
    g = group_of(clip)
    src = spec.get("source_span_s")
    if src is None or g not in SOURCE_FILES:
        return None, None
    t1 = float(src[0])
    t0 = max(0.0, t1 - PAD)
    if t1 - t0 < 0.5:
        return None, None
    wav = load_mono(TEST_VEC / SOURCE_FILES[g])
    seg = wav[int(t0 * SR): int(t1 * SR)].clone()
    sess = windows.get(f"{g}_session")
    content = {"len_s": round(t1 - t0, 2)}
    if sess is not None:
        for kind in ("keep", "suppress"):
            cov = 0.0
            for a, b in (sess.get(kind) or []):
                cov += max(0.0, min(b, t1) - max(a, t0))
            content[kind] = round(cov / (t1 - t0), 2)
    else:
        cov_k = cov_s = 0.0
        for c, s in windows.items():
            if group_of(c) != g or c == clip or s.get("source_span_s") is None:
                continue
            a, b = s["source_span_s"]
            ov = max(0.0, min(b, t1) - max(a, t0))
            if "lone far" in s.get("role", ""):
                cov_s += ov
            else:
                cov_k += ov
        content["keep"] = round(cov_k / (t1 - t0), 2)
        content["suppress"] = round(cov_s / (t1 - t0), 2)
    content["gap"] = round(max(0.0, 1.0 - content["keep"] - content["suppress"]), 2)
    return seg, content


def tile(seg: torch.Tensor, seconds: float) -> torch.Tensor:
    n = int(seconds * SR)
    reps = n // seg.numel() + 1
    return seg.repeat(reps)[:n]


def conditions_for(clip: str, spec: dict, windows: dict, sweep: bool) -> list:
    """[(condition, variant, pad_tensor_or_None, meta)]"""
    g = group_of(clip)
    dev = g in DEVICE_GROUPS
    chain = "dev" if dev else "qvf"
    out = [("none", "none", None, {}), ("silence", "silence", torch.zeros(int(PAD * SR)), {})]
    # true quiet floor of the same chain (no speech)
    for name, (which, f, a, b) in TONE.items():
        if (which == "qvf") != dev:
            out.append(("floor", name, cut(f, a, b), {"source": name}))
    # speech from another recording: same room (device) / same chain (QVF)
    tab = SPEECH[chain]
    near_src = tab["near"].get(g, tab["near"].get("_default"))
    far_src = tab["far"].get(g, tab["far"].get("_default"))
    out.append(("near_speech", "near_speech", clip_head(near_src, PAD), {"source": near_src}))
    out.append(("far_speech", "far_speech", clip_head(far_src, PAD), {"source": far_src}))
    if dev:
        out.append(("event", "event", cut(*EVENT), {"source": "90d gap utterance"}))
    # speech from the OTHER chain (device <-> QVF)
    x_src = "qvf_gym_near1" if dev else "180d_near1"
    out.append(("near_xchain", "near_xchain", clip_head(x_src, PAD), {"source": x_src}))
    seg, content = stream_prefix(clip, spec, windows)
    if seg is not None:
        out.append(("stream", "stream", seg, {"content": content}))
    if sweep and dev and g != "0d":
        anchor_full = load_mono(CASES / f"{near_src}_raw.wav")
        for L in SWEEP_LENS:
            out.append((f"anchor_len{L:g}", f"anchor_len{L:g}", anchor_full[: int(L * SR)].clone(),
                        {"source": near_src, "len_s": L}))
        floor = cut(*TONE["dev_floor_180d"][1:])
        anchor3 = anchor_full[: int(PAD * SR)]
        for G in DECAY_GAPS:
            out.append((f"decay{G:g}", f"decay{G:g}", torch.cat([anchor3, tile(floor, G)]),
                        {"source": near_src, "gap_s": G}))
    return out


def score_ckpt(model, windows: dict, device, sweep: bool) -> list:
    rows = []
    for clip, spec in windows.items():
        if clip.startswith("_") or clip.endswith("_session"):
            continue
        role = spec.get("role", "")
        if "lone far" in role:
            side, spans = "far", spec["suppress"]
        elif "lone near" in role:
            side, spans = "near", spec["keep"]
        elif "double-talk" in role:
            side, spans = "dt", spec["keep"]
        else:
            continue
        wav = load_mono(CASES / f"{clip}_raw.wav").view(1, -1)
        floor = spec.get("floor_dbfs")
        for cond, variant, pad, meta in conditions_for(clip, spec, windows, sweep):
            if pad is not None:
                x = torch.cat([pad.view(1, -1), wav], dim=-1)
                off = pad.numel() / SR
            else:
                x, off = wav, 0.0
            with torch.no_grad():
                out = model(x.to(device)).detach().cpu().view(1, -1)
            n = min(out.shape[-1], x.shape[-1])
            sh = [[a + off, b + off] for a, b in spans]
            in_db = span_dbfs(x, sh, SR, n)
            out_db = span_dbfs(out, sh, SR, n)
            row = {"clip": clip, "group": group_of(clip), "side": side, "condition": cond,
                   "variant": variant, "pad_len_s": round(off, 2), "delta_db": out_db - in_db,
                   "residual_db": (out_db - floor) if (side == "far" and floor is not None) else None,
                   **meta}
            steady = [[max(a, off + STEADY_FROM), b] for a, b in sh]
            steady = [s for s in steady if s[1] - s[0] > 0.2]
            if steady:
                row["delta_steady_db"] = span_dbfs(out, steady, SR, n) - span_dbfs(x, steady, SR, n)
            rows.append(row)
    return rows


def cmd_score(args) -> None:
    windows = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items()
               if not k.startswith("_")}
    device = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config_path, expected_task="voice_isolation").model)
    all_rows = []
    for ck in args.ckpt:
        state = torch.load(ck, map_location="cpu")
        model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
        model = model.to(device).eval()
        rows = score_ckpt(model, windows, device, args.sweep)
        for r in rows:
            r["ckpt"] = Path(ck).name
        all_rows.extend(rows)
        print(f"  scored {Path(ck).name}: {len(rows)} rows", flush=True)
    rec = {"tag": args.tag, "ckpts": [str(c) for c in args.ckpt], "pad_seconds": PAD, "rows": all_rows}
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {args.out}")
    report([rec])


def block_means(rec: dict) -> dict:
    acc = {}
    for r in rec["rows"]:
        acc.setdefault((r["clip"], r["side"], r["condition"]), []).append(r["delta_db"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def report(recs: list) -> None:
    from scipy.stats import wilcoxon
    for rec in recs:
        bm = block_means(rec)
        conds = []
        for k in bm:
            if k[2] not in conds:
                conds.append(k[2])
        print(f"\n== {rec['tag']} (block of {len(rec['ckpts'])} ckpts) ==")
        for side, bar in (("far", -6.0), ("near", -3.0), ("dt", -3.0)):
            print(f"-- {side} --  {'condition':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s}"
                  f"  {'vs none (paired)':>28s}")
            base = {k[0]: v for k, v in bm.items() if k[1] == side and k[2] == "none"}
            for cond in conds:
                cur = {k[0]: v for k, v in bm.items() if k[1] == side and k[2] == cond}
                if not cur:
                    continue
                v = np.array(list(cur.values()))
                common = sorted(set(cur) & set(base))
                extra = ""
                if cond != "none" and len(common) >= 5:
                    a = np.array([cur[c] for c in common]); b = np.array([base[c] for c in common])
                    try:
                        p = wilcoxon(a, b).pvalue
                    except ValueError:
                        p = float("nan")
                    extra = f"  delta med {np.median(a - b):+6.2f}  p={p:.3f} (n={len(common)})"
                elif cond != "none":
                    ds = [cur[c] - base[c] for c in common]
                    extra = f"  delta per clip {['%+.2f' % d for d in ds]}"
                hits = int((v <= bar).sum())
                tag = f"<={bar:.0f}: {hits}/{len(v)}" if side == "far" else f"viol(<{bar:.0f}): {hits}/{len(v)}"
                print(f"   {cond:12s} {len(v):3d} {np.median(v):+8.2f} {np.percentile(v, 10):+8.2f} "
                      f"{np.percentile(v, 90):+8.2f}  {tag:12s}{extra}")
        # per-clip table for the keep-violation clips and every far clip
        print("-- per clip (block mean delta dB) --")
        clips = sorted({k[0] for k in bm}, key=lambda c: (bm.get((c, [k[1] for k in bm if k[0] == c][0], 'none'), 0)))
        main_conds = [c for c in ("none", "silence", "floor", "near_speech", "far_speech", "event", "near_xchain", "stream") if c in conds]
        print(f"   {'clip':26s} {'side':4s} " + " ".join(f"{c:>10s}" for c in main_conds) + "   stream-content")
        content = {}
        for r in rec["rows"]:
            if r["condition"] == "stream" and "content" in r:
                content[r["clip"]] = r["content"]
        for c in sorted({k[0] for k in bm}):
            side = [k[1] for k in bm if k[0] == c][0]
            vals = [bm.get((c, side, cond)) for cond in main_conds]
            line = " ".join(f"{v:+10.2f}" if v is not None else f"{'-':>10s}" for v in vals)
            print(f"   {c:26s} {side:4s} {line}   {content.get(c, '')}")
        for prefix, title in (("anchor_len", "anchor-length sweep (3 s near speech, other recording, same room)"),
                              ("decay", "decay: 3 s near anchor + N s true floor before the clip")):
            names = sorted({k[2] for k in bm if k[2].startswith(prefix)}, key=lambda s: float(s[len(prefix):]))
            if not names:
                continue
            print(f"-- {title}, block-median delta dB (device clips) --")
            for side in ("far", "near"):
                none_v = [bm[k] for k in bm if k[1] == side and k[2] == "none" and k[0].split("_")[0] in ("90d", "180d", "270d")]
                cells = []
                for nm in names:
                    v = [bm[k] for k in bm if k[1] == side and k[2] == nm]
                    if v:
                        cells.append(f"{nm[len(prefix):]}s {np.median(v):+7.2f} (n={len(v)})")
                if cells:
                    print(f"   {side:4s} none {np.median(none_v):+7.2f} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("config_path"); s.add_argument("--ckpt", action="append", required=True)
    s.add_argument("--tag", required=True); s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda"); s.add_argument("--sweep", action="store_true")
    s.set_defaults(fn=cmd_score)
    r = sub.add_parser("report"); r.add_argument("results", nargs="+")
    r.set_defaults(fn=lambda a: report([json.loads(Path(p).read_text()) for p in a.results]))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
