"""Keep-side speech-to-floor sweep -- the guard that can actually move.

A level-swept keep probe is vacuous for this architecture (InstantLN makes the
network scale-equivariant). What CAN tilt is the keep decision as the room gets
noisier: add noise to every field KEEP clip (lone-near and double-talk) so its
speech-to-floor drops to 20/15/12/8 dB and measure keep preservation. Baseline on
v16 ep19 (2026-08-31 review): median -0.25 -> -0.31 -> -0.38 -> -0.55 -> -1.02 dB,
monotone, with a cliff on qvf_keep_in_touch_near1 (-51.8 dB at s2f 8). A candidate is
judged against THAT curve, not against "flat".

    uv run python scripts/eval_s2f_keep_probe.py config/infer_dpcrn.yaml --ckpt CKPT [--ckpt ...] --out out.json
    uv run python scripts/eval_s2f_keep_probe.py compare A.json B.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "egs/voice_isolate/scripts"))
from puresound.config import load_recipe           # noqa: E402
from puresound.recipes import init_siso_model      # noqa: E402
from eval_realcase import span_dbfs                # noqa: E402
from eval_coldstart_v2 import CASES, SR, load_wav  # noqa: E402

LEVELS = (None, 20, 15, 12, 8)


def noisy_version(wav, keep_spans, s2f, clip):
    if s2f is None:
        return wav
    seed = int(hashlib.sha1(f"{clip}:{s2f}".encode()).hexdigest()[:8], 16)   # NOT hash(): salted per process
    g = torch.Generator().manual_seed(seed)
    v_db = span_dbfs(wav, keep_spans, SR, wav.shape[-1])
    nz = torch.randn(wav.shape, generator=g)
    nz = nz * (10 ** ((v_db - s2f) / 20) / nz.pow(2).mean().sqrt())
    return wav + nz


def score(args):
    windows = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items() if not k.startswith("_")}
    device = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config_path, expected_task="voice_isolation").model)
    rows = []
    for ck in args.ckpt:
        st = torch.load(ck, map_location="cpu")
        model.reload_checkpoint(st.get("state_dict", st), load_loss_func=False)
        model = model.to(device).eval()
        for clip, spec in windows.items():
            keep = spec.get("keep") or []
            if not keep or clip.endswith("_session") or "lone far" in spec.get("role", ""):
                continue
            wav = load_wav(clip)
            for s2f in LEVELS:
                x = noisy_version(wav, keep, s2f, clip)
                with torch.no_grad():
                    out = model(x.to(device)).detach().cpu().view(1, -1)
                n = min(out.shape[-1], x.shape[-1])
                rows.append({"ckpt": Path(ck).name, "clip": clip, "s2f": "native" if s2f is None else s2f,
                             "keep_db": span_dbfs(out, keep, SR, n) - span_dbfs(x, keep, SR, n)})
        print(f"  scored {Path(ck).name}", flush=True)
    rec = {"tag": args.tag, "ckpts": args.ckpt, "rows": rows}
    Path(args.out).write_text(json.dumps(rec, indent=1))
    report(rec)


def curve(rec):
    acc = {}
    for r in rec["rows"]:
        acc.setdefault((r["clip"], str(r["s2f"])), []).append(r["keep_db"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def report(rec):
    c = curve(rec)
    print(f"\n== {rec['tag']}  keep preservation vs added-noise s2f (block of {len(rec['ckpts'])})")
    for s2f in ("native", "20", "15", "12", "8"):
        v = np.array([x for (clip, s), x in c.items() if s == s2f])
        worst = min(((clip, x) for (clip, s), x in c.items() if s == s2f), key=lambda t: t[1])
        print(f"  s2f {s2f:6s} n={len(v):2d} median {np.median(v):+6.2f}  violations(<-3) {int((v<-3).sum())}  worst {worst[1]:+7.2f} ({worst[0]})")


def compare(args):
    from scipy.stats import wilcoxon
    a, b = (json.loads(Path(p).read_text()) for p in args.results)
    ca, cb = curve(a), curve(b)
    print(f"compare {a['tag']} vs {b['tag']}  (delta = A - B, paired per clip)")
    for s2f in ("native", "20", "15", "12", "8"):
        keys = sorted(k for k in ca if k in cb and k[1] == s2f)
        xa = np.array([ca[k] for k in keys]); xb = np.array([cb[k] for k in keys])
        p = wilcoxon(xa, xb).pvalue if len(keys) >= 6 else float("nan")
        print(f"  s2f {s2f:6s} n={len(keys):2d} {np.median(xa):+6.2f} vs {np.median(xb):+6.2f}  delta med {np.median(xa-xb):+6.2f}  p={p:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("config_path"); s.add_argument("--ckpt", action="append", required=True)
    s.add_argument("--tag", required=True); s.add_argument("--out", required=True); s.add_argument("--device", default="cuda")
    s.set_defaults(func=score)
    c = sub.add_parser("compare"); c.add_argument("results", nargs=2); c.set_defaults(func=compare)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
