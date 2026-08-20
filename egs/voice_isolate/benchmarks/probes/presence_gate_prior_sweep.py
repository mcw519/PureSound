"""b_init x tau_dn: what the gate's PRIOR costs and what it buys.

The four-actuator run found PresenceGate safe (35 of 38 keep clips bit-identical)
and nearly inert (median far reduction -2.33 vs the mask's -2.30). The cause is
not the ceiling this time -- it is that `b` starts at 1.0 ("presume a user is
there") and falls on a 1 s time constant, so on a cold-start clip of a few seconds
the gate is still opening when the clip ends.

`b_init` IS the prior. On a session it barely matters -- real evidence moves `b`
within the first second either way. On a cold-start clip it is the whole story:
nothing has been observed, and 1.0 asserts a user anyway.

Forward pass is cached once per clip, so the grid is free. Scored with
eval_realcase's own constants and span arithmetic.
"""
import argparse, json, math, os, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
RECIPE = REPO / "egs/voice_isolate"
sys.path.insert(0, str(REPO))

from puresound.audio.io import AudioIO
from puresound.config import load_recipe
from puresound.recipes import init_siso_model
from puresound.system.presence_gate import PresenceGate

CASES = pathlib.Path("data_report/field_cases/test_vector_cases")
KEEP_VIOLATION_DB = -3.0
SUPPRESS_FAIL_DB = -6.0
SUPPRESS_PARTIAL_DB = 12.0
MIN_HEADROOM = 6.0


def span_dbfs(wav, spans, sr, limit):
    tot = n = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), limit)
        if j > i:
            tot += float(wav[..., i:j].square().sum()); n += j - i
    return 10.0 * math.log10(tot / n + 1e-12) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-blend", type=float, default=0.9)
    ap.add_argument("--b-hi", type=float, default=0.50)
    ap.add_argument("--b-lo", type=float, default=0.10)
    ap.add_argument("--floor-db", type=float, default=-26.0)
    ap.add_argument("--tau-up", type=float, default=0.05)
    a = ap.parse_args()
    os.chdir(RECIPE)

    model = init_siso_model(load_recipe(a.config_path, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["state_dict"],
                          strict=False)
    model = model.eval().to(a.device)
    hop = model.encoder.hop_length
    windows = json.loads((CASES / "windows.json").read_text())

    cache = []
    with torch.no_grad():
        for clip, spec in windows.items():
            if clip.startswith("_"):
                continue
            wav, sr = AudioIO.open(f_path=str(CASES / f"{clip}_raw.wav"),
                                   target_lvl=None, resample_to=16000)
            wav = wav.view(1, -1)
            enh = model(wav.to(a.device), dry_blend=a.dry_blend).detach().cpu()
            lg = model.backbone.last_vad_logits.detach().cpu()
            n = min(enh.shape[-1], wav.shape[-1])
            floor = spec.get("floor_dbfs", float("nan"))
            cache.append(dict(
                clip=clip, spec=spec, sr=sr, n=n, enh=enh[..., :n], mix=wav[..., :n],
                logits=lg, floor=floor,
                keep_in=span_dbfs(wav, spec.get("keep", []), sr, n),
                supp_in=span_dbfs(wav, spec.get("suppress", []), sr, n)))
    print(f"cached {len(cache)} clips\n")

    def score(gate):
        kv = 0; worst = 0.0; ident = 0; nkeep = 0
        sf = 0; reds = []; sess = {}
        for c in cache:
            out = (c["enh"] if gate is None
                   else gate.apply(c["enh"], hop=hop, logits=c["logits"]))
            spec, sr, n = c["spec"], c["sr"], c["n"]
            if spec.get("keep"):
                k = span_dbfs(out, spec["keep"], sr, n) - c["keep_in"]
                if not c["clip"].endswith("_session"):
                    nkeep += 1
                    if k < KEEP_VIOLATION_DB: kv += 1
                    worst = min(worst, k)
                    if torch.equal(out, c["enh"]): ident += 1
            if spec.get("suppress"):
                s = span_dbfs(out, spec["suppress"], sr, n) - c["supp_in"]
                hr = c["supp_in"] - c["floor"]
                if not c["clip"].endswith("_session"):
                    reds.append(s)
                    if hr == hr and hr >= MIN_HEADROOM and s > SUPPRESS_FAIL_DB:
                        sf += 1
                if c["clip"].endswith("_session"):
                    sess[c["clip"]] = s
        return dict(kv=kv, worst=worst, ident=ident, nkeep=nkeep, sf=sf,
                    red=float(np.median(reds)), s180=sess.get("180d_session", float("nan")),
                    s90=sess.get("90d_session", float("nan")))

    base = score(None)
    print(f"{'b_init':>7s} {'tau_dn':>7s} | {'KEEPviol':>8s} {'worst':>7s} "
          f"{'identical':>10s} | {'SUPPfail':>8s} {'med red':>8s} {'180d':>7s} {'90d':>7s}")
    print(f"{'mask only (no gate)':>15s} | {base['kv']:8d} {base['worst']:7.2f} "
          f"{base['ident']:>4d}/{base['nkeep']:<5d} | {base['sf']:8d} {base['red']:8.2f} "
          f"{base['s180']:7.2f} {base['s90']:7.2f}")
    print("-" * 88)
    for b_init in (1.0, 0.80, 0.60, 0.50, 0.35):
        for tau_dn in (1.0, 0.50, 0.25, 0.10):
            g = PresenceGate(b_hi=a.b_hi, b_lo=a.b_lo, gain_floor_db=a.floor_db,
                             tau_up_s=a.tau_up, tau_dn_s=tau_dn, b_init=b_init)
            r = score(g)
            flag = ""
            if r["kv"] <= base["kv"] and r["red"] < base["red"] - 1.0:
                flag = "  <== safe AND acts"
            print(f"{b_init:7.2f} {tau_dn:7.2f} | {r['kv']:8d} {r['worst']:7.2f} "
                  f"{r['ident']:>4d}/{r['nkeep']:<5d} | {r['sf']:8d} {r['red']:8.2f} "
                  f"{r['s180']:7.2f} {r['s90']:7.2f}{flag}")
    print(f"\n  tau_up fixed at {a.tau_up}s (the earlier sweep found faster is strictly better)")
    print(f"  b_hi {a.b_hi} / b_lo {a.b_lo} / floor {a.floor_db} dB")
    print("  'identical' counts cold-start keep clips byte-for-byte unchanged -- the dead zone holding.")


if __name__ == "__main__":
    main()
