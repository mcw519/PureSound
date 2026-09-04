"""Offline chain-stage probe on the DistHead readout, measured the SAME way as
anchor_gate_sim.py `readability`: AUC of the sliding (W s) readout over keep spans
(30-50 cm near talkers) vs suppress spans (2-3 m far talkers).

Instead of comparing chains (device vs QVF) we APPLY the suspected chain stages
offline to the device-chain clips and watch the readout move. Conditions follow
eq_probe.py exactly (same compressor operating point thr -28 / ratio 4, same
broadcast EQ, same RMS restore, `dark_tilt` as the directional control).

The speech-activity mask is computed ONCE from the unmodified clip and reused for
every condition, so the label set is identical and only the readout moves.

  uv run python comp_readability.py --tag v8 --config config/train_dpcrn.yaml \
      --ckpt pretrained_ckpt/dpcrn_v8.ckpt --device cuda:1 --out <dir>/comp_v8.json
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPE / "scripts"))
sys.path.insert(0, str(RECIPE / "benchmarks/probes"))

from puresound.audio.dsp import compressor_gain          # noqa: E402
from puresound.config import load_recipe                 # noqa: E402
from puresound.recipes import init_siso_model            # noqa: E402
import anchor_gate_sim as ag                             # noqa: E402

SR = 16000
CASES = RECIPE / "data_report/field_cases/test_vector_cases"
WINDOWS = [0.5, 1.0, 2.0]


def rms_restore(y, ref):
    return y * (ref.pow(2).mean().sqrt().clamp_min(1e-8) / y.pow(2).mean().sqrt().clamp_min(1e-8))


def eq_broadcast(x):
    y = AF.highpass_biquad(x, SR, 120.0)
    y = AF.bass_biquad(y, SR, -4.0, central_freq=200.0)
    return AF.equalizer_biquad(y, SR, 3000.0, 6.0, Q=1.0)


def comp(x, thr=-28.0, ratio=4.0):
    return compressor_gain(x.view(1, -1), SR, threshold_db=thr, ratio=ratio).view(-1)


CONDS = [
    ("none",            lambda x: x),
    ("comp_thr-34_r2",  lambda x: comp(x, -34.0, 2.0)),
    ("comp_thr-28_r4",  lambda x: comp(x, -28.0, 4.0)),
    ("comp_thr-22_r6",  lambda x: comp(x, -22.0, 6.0)),
    ("broadcast",       eq_broadcast),
    ("comp+broadcast",  lambda x: eq_broadcast(comp(x, -28.0, 4.0))),
    ("dark_tilt(ctrl)", lambda x: AF.treble_biquad(x, SR, -4.0, central_freq=2500.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain", default="device", choices=["device", "qvf", "both"])
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = init_siso_model(load_recipe(args.config, expected_task="voice_isolation",
                                        expected_purpose="train").model)
    state = torch.load(args.ckpt, map_location="cpu")
    model.reload_checkpoint(state.get("state_dict", state), load_loss_func=False)
    model = model.to(dev).eval()
    if getattr(model.backbone, "dist_head", None) is None:
        raise SystemExit("no dist_head -- pass the TRAINING config")
    model.backbone.stash_bottleneck = True
    head_sd = {k: v.cpu() for k, v in model.backbone.dist_head.net.state_dict().items()}
    head = (head_sd["0.weight"].numpy().astype(np.float64), head_sd["0.bias"].numpy().astype(np.float64),
            head_sd["2.weight"].numpy().astype(np.float64), head_sd["2.bias"].numpy().astype(np.float64))

    w = {k: v for k, v in json.loads((CASES / "windows.json").read_text()).items()
         if not k.startswith("_")}
    clips = []
    for k, spec in w.items():
        if k.endswith("_session"):
            continue
        role = spec.get("role", "")
        if "lone near" in role:
            kind = "keep"; spans = spec.get("keep") or []
        elif "lone far" in role:
            kind = "suppress"; spans = spec.get("suppress") or []
        else:
            continue
        ch = ag.chain_of(ag.group_of(k))
        if args.chain != "both" and ch != args.chain:
            continue
        clips.append((k, kind, spans, spec, ch))

    pools = {}   # (chain, cond, kind, W) -> list arrays  (dist)
    dpool = {}
    per_clip = []
    for k, kind, spans, spec, ch in sorted(clips):
        x0, _ = sf.read(CASES / f"{k}_raw.wav")
        x0 = torch.from_numpy(np.ascontiguousarray(x0)).float().view(-1)
        act_cache = None
        for cname, fn in CONDS:
            y = rms_restore(fn(x0.clone()), x0)
            with torch.no_grad():
                model(y.view(1, -1).to(dev))
                bott = model.backbone.last_bottleneck
                feat = bott.mean(dim=2)[0].detach().cpu().numpy().astype(np.float32)   # [C, F]
            F = feat.shape[1]
            if act_cache is None:
                e_db = ag.frame_energy_db(x0.numpy(), F)
                thr = (spec.get("floor_dbfs") if spec.get("floor_dbfs") is not None
                       else np.percentile(e_db, 5)) + ag.ACTIVE_MARGIN_DB
                act_cache = ag.activity(e_db, thr)
                blocks = ag.span_blocks(spans, 0.0, F)
            act = act_cache[:F]
            est = ag.sliding_estimates(feat, act, head, WINDOWS)
            for W in WINDOWS:
                fg, drr = est[W]
                m = blocks[:F] & np.isfinite(fg)
                v, vd = fg[m], drr[m]
                if v.size == 0:
                    continue
                pools.setdefault((ch, cname, kind, W), []).append(v)
                dpool.setdefault((ch, cname, kind, W), []).append(vd)
                if W == 1.0:
                    per_clip.append({"clip": k, "chain": ch, "kind": kind, "cond": cname,
                                     "n_frames": int(v.size), "median_m": float(np.median(v)),
                                     "median_drr_db": float(np.median(vd))})
        print(f"  {k} done", flush=True)

    rows = []
    for ch in sorted({c for _, _, _, _, c in clips}):
        for cname, _ in CONDS:
            for W in WINDOWS:
                pk = pools.get((ch, cname, "keep", W)); ps = pools.get((ch, cname, "suppress", W))
                if not pk or not ps:
                    continue
                K, S = np.concatenate(pk), np.concatenate(ps)
                Kd, Sd = np.concatenate(dpool[(ch, cname, "keep", W)]), np.concatenate(dpool[(ch, cname, "suppress", W)])
                rows.append({"tag": args.tag, "chain": ch, "cond": cname, "W": W,
                             "auc_dist": ag.auc(-K, -S), "auc_drr": ag.auc(Kd, Sd),
                             "median_keep_m": float(np.median(K)), "median_supp_m": float(np.median(S)),
                             "median_keep_drr": float(np.median(Kd)), "median_supp_drr": float(np.median(Sd)),
                             "n_keep_frames": int(K.size), "n_supp_frames": int(S.size)})
    Path(args.out).write_text(json.dumps({"rows": rows, "per_clip": per_clip}, indent=1))
    hdr = f"{'chain':7s} {'cond':17s} {'W':4s} {'AUCdist':8s} {'AUCdrr':8s} {'keep_m':7s} {'supp_m':7s} {'keep_drr':9s} {'supp_drr':9s}"
    print("\n" + args.tag); print(hdr)
    for r in rows:
        if r["W"] != 1.0:
            continue
        print(f"{r['chain']:7s} {r['cond']:17s} {r['W']:<4.1f} {r['auc_dist']:<8.3f} {r['auc_drr']:<8.3f} "
              f"{r['median_keep_m']:<7.3f} {r['median_supp_m']:<7.3f} {r['median_keep_drr']:<9.2f} {r['median_supp_drr']:<9.2f}")


if __name__ == "__main__":
    main()
