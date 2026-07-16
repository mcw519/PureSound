"""Turn-taking keep/suppress scorecard WITH the near-field gate applied to the output.

Like eval_turntaking_set.py, but applies the trained DPCRN gate head as a per-frame
multiplicative gain on the enhanced output, so a gate checkpoint's effect is visible
(the frozen separator's mask path is identical across gate epochs). Reports, per system,
median/mean KEEP preservation and SUPPRESS reduction on a frozen (mix,target) turn-taking
set:

  * ours_raw   -- frozen separator, no gate (== eval_turntaking_set.py baseline)
  * ours_soft  -- separator * sigmoid(gate)
  * ours_hard  -- separator * (sigmoid(gate) >= threshold)

KEEP spans (target active) want preservation ~0 dB; < -3 = KEEP-VIOLATION (near user
killed). SUPPRESS spans (target silent, mix active = far competitor solo) want reduction
<< 0 dB; > -6 = SUPPRESS-FAIL (far leak). This is the on-domain (real-RIR) turn-taking
test of whether the gate suppresses far-only stretches.

Usage (from egs/voice_isolate):
    uv run python scripts/eval_turntaking_gated.py config/train_dpcrn_gate.yaml \
        --ckpt exp/dpcrn_gate_synth/lightning_logs/version_0/checkpoints/epoch=5-step=1500.ckpt \
        --set-dir /data/audio/eval_noisy_data/turntaking_set_realrir --device cpu
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402

KEEP_VIOLATION_DB = -3.0
SUPPRESS_FAIL_DB = -6.0


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    (_c, _t, _o, _s, _l, model_dict, *_r) = load_siso_recipe_config(config_path)
    model = init_siso_model(model_dict)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("loss_func_list.")}
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"# [ckpt] missing {len(missing)}: {missing[:3]}")
    return model.to(device).eval()


def gate_gain(logits: torch.Tensor, n_samples: int, hop: int, threshold):
    prob = torch.sigmoid(logits.reshape(-1).float())
    if threshold is not None:
        prob = (prob >= threshold).float()
    gain = prob.repeat_interleave(hop)
    if gain.shape[-1] < n_samples:
        gain = torch.cat([gain, gain[-1:].expand(n_samples - gain.shape[-1])])
    return gain[:n_samples].view(1, -1)


def frame_spans(target, mix, sr, win_ms=25.0, floor_db=-40.0):
    win = max(1, int(sr * win_ms / 1000.0))
    x, m = target.reshape(-1), mix.reshape(-1)
    n = (min(x.numel(), m.numel()) // win) * win
    if n == 0:
        return [], []
    tf = x[:n].reshape(-1, win).square().mean(1).sqrt()
    mf = m[:n].reshape(-1, win).square().mean(1).sqrt()
    t_db = 20 * torch.log10((tf / tf.max().clamp_min(1e-12)).clamp_min(1e-12))
    m_db = 20 * torch.log10((mf / mf.max().clamp_min(1e-12)).clamp_min(1e-12))
    t_active = t_db > floor_db
    m_active = m_db > floor_db
    keep_flag = t_active
    supp_flag = (~t_active) & m_active

    def to_spans(flag):
        spans, run = [], None
        for i, f in enumerate(flag.tolist()):
            if f and run is None:
                run = i
            elif not f and run is not None:
                spans.append((run * win / sr, i * win / sr))
                run = None
        if run is not None:
            spans.append((run * win / sr, len(flag) * win / sr))
        return spans

    return to_spans(keep_flag), to_spans(supp_flag)


def span_power_db(num, den, spans, sr):
    n = min(num.shape[-1], den.shape[-1])
    ne = de = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), n)
        if j <= i:
            continue
        ne += float(num[..., i:j].square().sum())
        de += float(den[..., i:j].square().sum())
    if de <= 0.0:
        return float("nan")
    return 10.0 * torch.log10(torch.tensor(ne / de + 1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    set_dir = Path(args.set_dir)
    mixes = sorted(set_dir.glob("*_mix.wav"))
    if args.limit:
        mixes = mixes[: args.limit]
    if not mixes:
        sys.exit(f"no *_mix.wav under {set_dir}")

    model = load_model(args.config_path, args.ckpt, device)
    systems = ["ours_raw", "ours_soft", "ours_hard"]
    keep = {s: [] for s in systems}
    supp = {s: [] for s in systems}
    kviol = {s: 0 for s in systems}
    sfail = {s: 0 for s in systems}
    n = 0

    for mp in mixes:
        item = mp.name[: -len("_mix.wav")]
        tp = set_dir / f"{item}_target.wav"
        if not tp.is_file():
            continue
        mix, sr = AudioIO.open(f_path=str(mp), target_lvl=None, resample_to=16000)
        tgt, _ = AudioIO.open(f_path=str(tp), target_lvl=None, resample_to=16000)
        mix, tgt = mix.view(1, -1), tgt.view(1, -1)
        with torch.no_grad():
            enh = model(mix.to(device)).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
            logits = model.backbone.last_vad_logits
            if logits is not None:
                logits = logits.detach().cpu()
        if logits is None:
            raise RuntimeError("backbone.last_vad_logits is None; gate head disabled?")

        L = min(enh.shape[-1], mix.shape[-1], tgt.shape[-1])
        mix, tgt, enh = mix[..., :L], tgt[..., :L], enh[..., :L]
        outs = {
            "ours_raw": enh,
            "ours_soft": (enh * gate_gain(logits, L, args.hop, None)).clamp(-1.0, 1.0),
            "ours_hard": (enh * gate_gain(logits, L, args.hop, args.threshold)).clamp(-1.0, 1.0),
        }
        keep_spans, supp_spans = frame_spans(tgt, mix, sr)
        n += 1
        for s in systems:
            kp = span_power_db(outs[s], tgt, keep_spans, sr) if keep_spans else float("nan")
            sd_ = span_power_db(outs[s], mix, supp_spans, sr) if supp_spans else float("nan")
            if kp == kp:
                keep[s].append(kp)
                if kp < KEEP_VIOLATION_DB:
                    kviol[s] += 1
            if sd_ == sd_:
                supp[s].append(sd_)
                if sd_ > SUPPRESS_FAIL_DB:
                    sfail[s] += 1

    def med(v):
        return st.median(v) if v else float("nan")

    def mean(v):
        return st.mean(v) if v else float("nan")

    print("=" * 78)
    print(f"Gated turn-taking scorecard  (n={n})   set={set_dir.name}")
    print(f"ckpt={args.ckpt}")
    print("=" * 78)
    print(f"{'system':<10} {'KEEP med/mean':>18} {'ok/viol':>10}   {'SUPPRESS med/mean':>20} {'ok/fail':>10}")
    for s in systems:
        nk = len(keep[s])
        ns = len(supp[s])
        print(f"{s:<10} {med(keep[s]):+7.2f}/{mean(keep[s]):+6.2f} dB "
              f"{nk - kviol[s]:>3}/{kviol[s]:<3}   "
              f"{med(supp[s]):+8.2f}/{mean(supp[s]):+7.2f} dB "
              f"{ns - sfail[s]:>3}/{sfail[s]:<3}")
    print()
    print(f"# KEEP want ~0 (<{KEEP_VIOLATION_DB}=VIOLATION); SUPPRESS want <<0 (>{SUPPRESS_FAIL_DB}=FAIL)")


if __name__ == "__main__":
    main()
