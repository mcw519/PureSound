"""Part (b): onset profile on SYNTHETIC in-domain data.

Runs a checkpoint over every utterance of data_report/wer_set_moderate_test and
data_report/indomain_wer_set (dry_blend 1.0) in three prefix variants and measures the same
onset profile as onset_real.py:

  t0    : the mix as it is (both sets start speaking at t ~ 0.04 s -- no leading silence)
  bg1   : 1.0 s of the utterance's OWN background (mix - ref, drawn from the longest
          ref-inactive stretch and loop-tiled with 20 ms crossfades) prepended
  sil1  : 1.0 s of digital silence prepended (known to be a non-neutral prefix on the device
          chain -- reference_matrix_README section 2 -- reported for contrast only)

Two measures per window X in {0.5, 1.0, 2.0} s:
  keep delta  : span_dbfs(enh) - span_dbfs(mix) over ref-active samples in the window
                (same definition as scripts/eval_realcase.py, restricted to a mask)
  fg gain     : 10 log10 ( (<enh,ref>/<ref,ref>)^2 ) over the window -- how much of the
                foreground itself survived, immune to how much interferer sits in the window

  excess(X) = value(first X s of foreground) - value(remainder of the foreground)

Usage:
  uv run python onset_synth.py --tag v8  --config config/train_dpcrn.yaml \
      --ckpt pretrained_ckpt/dpcrn_v8.ckpt --device cuda:0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path("/home/milowu/A4Audio/PureSound")
RECIPE = REPO_ROOT / "egs/voice_isolate"
sys.path.insert(0, str(REPO_ROOT))

from puresound.config import load_recipe          # noqa: E402
from puresound.recipes import init_siso_model     # noqa: E402

SR = 16000
WINDOWS = (0.5, 1.0, 2.0)
MIN_REST = 0.5
OUT = Path(__file__).resolve().parent


def load_model(config_path, ckpt, device):
    model = init_siso_model(load_recipe(config_path, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    state = torch.load(ckpt, map_location="cpu")
    model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
    return model.to(device).eval()


def frame_active(ref, thr_db=40.0):
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


def level_dbfs(wav, mask):
    n = int(mask.sum())
    if n <= 0:
        return float("nan")
    return 10.0 * math.log10(float((wav[mask].astype(np.float64) ** 2).sum()) / n + 1e-12)


def fg_gain_db(enh, ref, mask):
    r = ref[mask].astype(np.float64)
    e = enh[mask].astype(np.float64)
    den = float((r * r).sum())
    if den <= 1e-12:
        return float("nan")
    g = float((e * r).sum()) / den
    return 10.0 * math.log10(g * g + 1e-12)


def build_bg_prefix(mix, ref, act, want=1.0):
    """1 s of the utterance's own background = mix - ref over the longest ref-inactive stretch,
    loop-tiled with 20 ms crossfades. Returns (prefix, source) where source in
    {'inactive_loop', 'head_fallback'}."""
    bg = (mix - ref).astype(np.float32)
    inact = ~act
    # longest contiguous inactive run
    best = (0, 0)
    i = 0
    while i < len(inact):
        if inact[i]:
            j = i
            while j < len(inact) and inact[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    a, b = best
    seg = bg[a:b]
    src = "inactive_loop"
    if len(seg) < int(0.12 * SR):
        seg = bg[: int(min(want, len(bg) / SR) * SR)]
        src = "head_fallback"
    n_want = int(want * SR)
    xf = min(int(0.02 * SR), len(seg) // 4)
    out = np.zeros(0, dtype=np.float32)
    while len(out) < n_want:
        if len(out) == 0 or xf <= 0:
            out = np.concatenate([out, seg])
        else:
            tail = out[-xf:].copy()
            w = np.linspace(0, 1, xf, dtype=np.float32)
            out = np.concatenate([out[:-xf], tail * (1 - w) + seg[:xf] * w, seg[xf:]])
    return out[:n_want], src


def profile(mix, enh, ref, base_mask, t_start, t_end):
    n = min(len(mix), len(enh), len(ref), len(base_mask))
    mix, enh, ref, base_mask = mix[:n], enh[:n], ref[:n], base_mask[:n]
    tg = np.arange(n) / SR
    out = {}
    for X in WINDOWS:
        if t_end - t_start < X + MIN_REST:
            out[X] = None
            continue
        on = base_mask & (tg >= t_start) & (tg < t_start + X)
        rest = base_mask & (tg >= t_start + X) & (tg < t_end)
        if on.sum() < 0.05 * SR or rest.sum() < 0.05 * SR:
            out[X] = None
            continue
        d_on = level_dbfs(enh, on) - level_dbfs(mix, on)
        d_rest = level_dbfs(enh, rest) - level_dbfs(mix, rest)
        g_on = fg_gain_db(enh, ref, on)
        g_rest = fg_gain_db(enh, ref, rest)
        out[X] = dict(d_on=d_on, d_rest=d_rest, d_excess=d_on - d_rest,
                      g_on=g_on, g_rest=g_rest, g_excess=g_on - g_rest)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(str(RECIPE / args.config) if not args.config.startswith("/") else args.config,
                       str(RECIPE / args.ckpt) if not args.ckpt.startswith("/") else args.ckpt, device)

    rows = []
    src_counts = {}
    for setname in ("wer_set_moderate_test", "indomain_wer_set"):
        d = RECIPE / "data_report" / setname
        ids = sorted(p.name[:-8] for p in d.glob("*_mix.wav"))
        if args.limit:
            ids = ids[: args.limit]
        for k, uid in enumerate(ids):
            mix, _ = sf.read(d / f"{uid}_mix.wav", dtype="float32")
            ref, _ = sf.read(d / f"{uid}_ref.wav", dtype="float32")
            L = min(len(mix), len(ref))
            mix, ref = mix[:L], ref[:L]
            act = frame_active(ref)
            bg, src = build_bg_prefix(mix, ref, act)
            src_counts[src] = src_counts.get(src, 0) + 1
            variants = {
                "t0": (mix, ref, act, 0),
                "bg1": (np.concatenate([bg, mix]), np.concatenate([np.zeros_like(bg), ref]),
                        np.concatenate([np.zeros(len(bg), bool), act]), len(bg)),
                "sil1": (np.concatenate([np.zeros(SR, np.float32), mix]),
                         np.concatenate([np.zeros(SR, np.float32), ref]),
                         np.concatenate([np.zeros(SR, bool), act]), SR),
            }
            for vname, (m, r, a, pad) in variants.items():
                x = torch.from_numpy(m.astype(np.float32)).view(1, -1).to(device)
                enh = model(x).detach().cpu().view(-1).numpy().astype(np.float32)
                nz = np.nonzero(a)[0]
                t0, t1 = nz[0] / SR, (nz[-1] + 1) / SR
                p = profile(m, enh, r, a, t0, t1)
                for X in WINDOWS:
                    v = p[X]
                    rows.append(dict(tag=args.tag, world="synth", subset=setname, variant=vname,
                                     item=uid, window=X, onset_s=round(t0, 3),
                                     span_len=round(t1 - t0, 2), bg_src=src,
                                     **({} if v is None else {kk: round(vv, 3) for kk, vv in v.items()})))
            if (k + 1) % 50 == 0:
                print(f"  {setname} {k+1}/{len(ids)}", flush=True)
    fn = OUT / f"synth_onset_rows_{args.tag}.jsonl"
    with open(fn, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("bg prefix sources:", src_counts)
    print("wrote", fn)


if __name__ == "__main__":
    main()
