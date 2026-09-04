"""FIXED chain-stage probe on the DistHead readout.

Two fixes over `comp_readability.py` (and over `benchmarks/probes/eq_probe.py`, which has
the same defect):

  1. `compressor_gain` RETURNS the gain curve, it does not apply it (see its docstring and
     `puresound/task/device_chain.py::_compressor`, which multiplies). eq_probe.py fed the
     GAIN CURVE to the model as if it were audio. On these field clips (peak -34 dBFS, below
     the -28 dB threshold) the curve is all-ones, so after RMS restore the model was shown a
     CONSTANT DC signal -- which is why every compressor operating point returned an identical
     readout. Here the gain is multiplied in.
  2. Level. The recipe normalises sources to `gain_normalized_to: -28` dBFS RMS, which is the
     level the compressor sees in training; the field clips sit 20+ dB below that, where the
     -34..-22 dBFS thresholds are a no-op. Each clip is therefore normalised to -28 dBFS RMS
     BEFORE the compressor and restored to its original RMS afterwards, so level is never the
     variable and the compressor operates in its training range. The measured mean/max gain
     reduction is reported per condition to prove the stage actually acted.

Metric identical to `anchor_gate_sim.py readability`: AUC of the sliding W-second readout over
keep spans (near, 30-50 cm) vs suppress spans (far, 2-3 m); activity mask taken once from the
unmodified clip so the label set is fixed.
"""
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

REPO = Path("/home/milowu/A4Audio/PureSound")
RECIPE = REPO / "egs/voice_isolate"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(RECIPE / "scripts"))
sys.path.insert(0, str(RECIPE / "benchmarks/probes"))

from puresound.audio.dsp import compressor_gain          # noqa: E402
from puresound.config import load_recipe                 # noqa: E402
from puresound.recipes import init_siso_model            # noqa: E402
import anchor_gate_sim as ag                             # noqa: E402

SR = 16000
CASES = RECIPE / "data_report/field_cases/test_vector_cases"
WINDOWS = [0.5, 1.0, 2.0]
OP_DBFS = -28.0        # recipe's gain_normalized_to


def rms(x):
    return float(x.pow(2).mean().sqrt().clamp_min(1e-12))


def rms_restore(y, ref):
    return y * (rms(ref) / max(rms(y), 1e-12))


def eq_broadcast(x):
    y = AF.highpass_biquad(x, SR, 120.0)
    y = AF.bass_biquad(y, SR, -4.0, central_freq=200.0)
    return AF.equalizer_biquad(y, SR, 3000.0, 6.0, Q=1.0)


def apply_comp(x, thr, ratio, attack=5.0, release=120.0):
    """Normalise to the training operating level, apply the gain curve, restore level.
    Returns (y, mean_gr_db, max_gr_db)."""
    scale = (10.0 ** (OP_DBFS / 20.0)) / max(rms(x), 1e-12)
    xn = x * scale
    g = compressor_gain(xn.view(1, -1), SR, threshold_db=thr, ratio=ratio,
                        attack_ms=attack, release_ms=release).view(-1)
    y = xn * g[: xn.numel()]
    gr = -20.0 * torch.log10(g.clamp_min(1e-8))
    return y / scale, float(gr.mean()), float(gr.max())


CONDS = [
    ("none",             lambda x: (x, 0.0, 0.0)),
    ("comp_thr-34_r2",   lambda x: apply_comp(x, -34.0, 2.0)),
    ("comp_thr-28_r4",   lambda x: apply_comp(x, -28.0, 4.0)),
    ("comp_thr-22_r6",   lambda x: apply_comp(x, -22.0, 6.0)),
    ("broadcast",        lambda x: (eq_broadcast(x), 0.0, 0.0)),
    ("comp+broadcast",   lambda x: (lambda t: (eq_broadcast(t[0]), t[1], t[2]))(apply_comp(x, -28.0, 4.0))),
    ("dark_tilt(ctrl)",  lambda x: (AF.treble_biquad(x, SR, -4.0, central_freq=2500.0), 0.0, 0.0)),
    ("wshape_p0.8",      lambda x: (waveshape(x, 0.8), 0.0, 0.0)),
    ("wshape_p0.6",      lambda x: (waveshape(x, 0.6), 0.0, 0.0)),
    ("wshape_p0.4",      lambda x: (waveshape(x, 0.4), 0.0, 0.0)),
    ("comp_r6_hardknee", lambda x: apply_comp(x, -40.0, 6.0)),
]


def waveshape(t, p):
    """`Augmentor.apply_media_coloring`'s |x|^p waveshaper -- the stage
    `compression_probe.py` (2026-08-21) used, the one whose causality was actually
    established. Not the same operator as `compressor_gain`."""
    peak = t.abs().amax().clamp_min(1e-8)
    return torch.sign(t / peak) * (t / peak).abs().pow(p) * peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True); ap.add_argument("--chain", default="both")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    st = torch.load(args.ckpt, map_location="cpu")
    model.reload_checkpoint(st.get("state_dict", st), load_loss_func=False)
    model = model.to(dev).eval()
    if getattr(model.backbone, "dist_head", None) is None:
        raise SystemExit("no dist_head")
    model.backbone.stash_bottleneck = True
    sd = {k: v.cpu() for k, v in model.backbone.dist_head.net.state_dict().items()}
    head = (sd["0.weight"].numpy().astype(np.float64), sd["0.bias"].numpy().astype(np.float64),
            sd["2.weight"].numpy().astype(np.float64), sd["2.bias"].numpy().astype(np.float64))

    w = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items() if not k.startswith("_")}
    clips = []
    for k, spec in w.items():
        if k.endswith("_session"):
            continue
        role = spec.get("role", "")
        if "lone near" in role:
            kind, spans = "keep", spec.get("keep") or []
        elif "lone far" in role:
            kind, spans = "suppress", spec.get("suppress") or []
        else:
            continue
        ch = ag.chain_of(ag.group_of(k))
        if args.chain != "both" and ch != args.chain:
            continue
        clips.append((k, kind, spans, spec, ch))

    pools, dpool, gr_log, per_clip = {}, {}, {}, []
    for k, kind, spans, spec, ch in sorted(clips):
        x0, _ = sf.read(CASES / f"{k}_raw.wav")
        x0 = torch.from_numpy(np.ascontiguousarray(x0)).float().view(-1)
        act_cache = blocks = None
        for cname, fn in CONDS:
            y, gr_mean, gr_max = fn(x0.clone())
            y = rms_restore(y, x0)
            with torch.no_grad():
                model(y.view(1, -1).to(dev))
                feat = model.backbone.last_bottleneck.mean(dim=2)[0].detach().cpu().numpy().astype(np.float32)
            F = feat.shape[1]
            if act_cache is None:
                e_db = ag.frame_energy_db(x0.numpy(), F)
                thr = (spec.get("floor_dbfs") if spec.get("floor_dbfs") is not None
                       else np.percentile(e_db, 5)) + ag.ACTIVE_MARGIN_DB
                act_cache = ag.activity(e_db, thr)
                blocks = ag.span_blocks(spans, 0.0, F)
            gr_log.setdefault(cname, []).append((gr_mean, gr_max))
            est = ag.sliding_estimates(feat, act_cache[:F], head, WINDOWS)
            for W in WINDOWS:
                fg, drr = est[W]
                m = blocks[:F] & np.isfinite(fg)
                if not m.any():
                    continue
                pools.setdefault((ch, cname, kind, W), []).append(fg[m])
                dpool.setdefault((ch, cname, kind, W), []).append(drr[m])
                if W == 1.0:
                    per_clip.append({"clip": k, "chain": ch, "kind": kind, "cond": cname,
                                     "n_frames": int(m.sum()), "median_m": float(np.median(fg[m])),
                                     "median_drr_db": float(np.median(drr[m])),
                                     "gr_mean_db": gr_mean, "gr_max_db": gr_max})
        print(f"  {k} done", flush=True)

    rows = []
    for ch in sorted({c for *_, c in clips}):
        for cname, _ in CONDS:
            for W in WINDOWS:
                pk, ps = pools.get((ch, cname, "keep", W)), pools.get((ch, cname, "suppress", W))
                if not pk or not ps:
                    continue
                K, S = np.concatenate(pk), np.concatenate(ps)
                Kd, Sd = np.concatenate(dpool[(ch, cname, "keep", W)]), np.concatenate(dpool[(ch, cname, "suppress", W)])
                g = np.array(gr_log[cname])
                rows.append({"tag": args.tag, "chain": ch, "cond": cname, "W": W,
                             "auc_dist": ag.auc(-K, -S), "auc_drr": ag.auc(Kd, Sd),
                             "median_keep_m": float(np.median(K)), "median_supp_m": float(np.median(S)),
                             "median_keep_drr": float(np.median(Kd)), "median_supp_drr": float(np.median(Sd)),
                             "gr_mean_db": float(g[:, 0].mean()), "gr_max_db": float(g[:, 1].max()),
                             "n_keep_frames": int(K.size), "n_supp_frames": int(S.size)})
    Path(args.out).write_text(json.dumps({"rows": rows, "per_clip": per_clip}, indent=1))
    print("\n" + args.tag)
    print(f"{'chain':7s} {'cond':17s} {'AUCdist':8s} {'AUCdrr':8s} {'keep_m':7s} {'supp_m':7s} {'kDRR':7s} {'sDRR':7s} {'GRmean':7s} {'GRmax':6s}")
    for r in rows:
        if r["W"] != 1.0:
            continue
        print(f"{r['chain']:7s} {r['cond']:17s} {r['auc_dist']:<8.3f} {r['auc_drr']:<8.3f} {r['median_keep_m']:<7.3f} "
              f"{r['median_supp_m']:<7.3f} {r['median_keep_drr']:<7.2f} {r['median_supp_drr']:<7.2f} "
              f"{r['gr_mean_db']:<7.2f} {r['gr_max_db']:<6.2f}")


if __name__ == "__main__":
    main()
