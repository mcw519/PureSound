"""Turn-taking keep/suppress scorecard on a FROZEN (mix, target) turn-taking set.

Unlike eval_realcase_faronly.py (manual windows.json on 3 real clips), this derives
keep / suppress spans automatically from the TARGET energy of every item in a dumped
turn-taking set (scripts/dump_turntaking_samples.py):

  * KEEP spans   = target is ACTIVE  -> the near user's turns. Output should preserve
                   them: keep_preservation_db = 10log10( E(enh)/E(target) ) ~ 0 dB.
                   << 0 = KEEP-VIOLATION (the model killed the user -> bot goes deaf).
  * SUPPRESS spans = target is SILENT while the MIX is active -> the far competitor's
                   solo. Output should silence them: suppress_reduction_db =
                   10log10( E(enh)/E(mix) ), very negative = good.
                   > threshold = SUPPRESS-FAIL (a >1 m voice leaked into ASR).

Plus whole-clip SI-SDR(enh, target). Runs on any set of <id>_mix.wav + <id>_target.wav.
Use --rir-folder when dumping the set to score on REAL-RIR turn-taking (e.g. the BUT
office bank) rather than synthetic reverb.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_turntaking_set.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
        --set-dir /data/audio/eval_noisy_data/turntaking_set_realrir --device cuda
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

KEEP_VIOLATION_DB = -3.0   # keep-span preservation below this = user/foreground killed
SUPPRESS_FAIL_DB = -6.0    # suppress-span reduction shallower than this = far leak


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    (_corpus, _trainer, _optim, _sched, _loss, model_dict, *_rest) = \
        load_siso_recipe_config(config_path)
    model = init_siso_model(model_dict)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(sd)
    else:
        model.load_state_dict(sd)
    return model.to(device).eval()


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    est = est.reshape(-1) - est.reshape(-1).mean()
    ref = ref.reshape(-1) - ref.reshape(-1).mean()
    a = (est @ ref) / ((ref @ ref) + eps)
    s = a * ref
    e = est - s
    return float(10.0 * torch.log10(((s @ s) + eps) / ((e @ e) + eps)))


def frame_spans(target: torch.Tensor, mix: torch.Tensor, sr: int,
                win_ms: float = 25.0, floor_db: float = -40.0):
    """Split the clip into KEEP (target active) and SUPPRESS (target silent, mix
    active) span lists [(a_s, b_s), ...] from framed energy, floor relative to the
    per-clip peak frame of each signal."""
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
    keep_flag = t_active                      # target present -> keep
    supp_flag = (~t_active) & m_active        # target silent but mix has energy -> suppress

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


def span_power_db(num: torch.Tensor, den: torch.Tensor, spans, sr: int) -> float:
    """10*log10( sum(num^2) / sum(den^2) ) over the concatenated spans."""
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
    out_rows = []
    keep_pres, supp_red, sisdr = [], [], []
    keep_viol = supp_fail = 0

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

        # DPCRN look-ahead can shorten the output by a few samples -- align lengths.
        L = min(enh.shape[-1], mix.shape[-1], tgt.shape[-1])
        mix, tgt, enh = mix[..., :L], tgt[..., :L], enh[..., :L]

        keep_spans, supp_spans = frame_spans(tgt, mix, sr)
        kp = span_power_db(enh, tgt, keep_spans, sr) if keep_spans else float("nan")
        sr_db = span_power_db(enh, mix, supp_spans, sr) if supp_spans else float("nan")
        ss = si_sdr(enh, tgt)

        if kp == kp:
            keep_pres.append(kp)
            if kp < KEEP_VIOLATION_DB:
                keep_viol += 1
        if sr_db == sr_db:
            supp_red.append(sr_db)
            if sr_db > SUPPRESS_FAIL_DB:
                supp_fail += 1
        sisdr.append(ss)
        out_rows.append({"id": item, "keep_preservation_db": round(kp, 2) if kp == kp else None,
                         "suppress_reduction_db": round(sr_db, 2) if sr_db == sr_db else None,
                         "si_sdr_db": round(ss, 2),
                         "keep_span_s": round(sum(b - a for a, b in keep_spans), 2),
                         "suppress_span_s": round(sum(b - a for a, b in supp_spans), 2)})

    (set_dir / "turntaking_scores.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n")

    def med(v):
        return st.median(v) if v else float("nan")

    n = len(out_rows)
    print("=" * 64)
    print(f"Turn-taking keep/suppress scorecard  (n={n})")
    print(f"  set-dir : {set_dir}")
    print(f"  ckpt    : {args.ckpt}")
    print("=" * 64)
    print(f"KEEP preservation  (enh vs target, want ~0 dB) : "
          f"median {med(keep_pres):+.2f} / mean {st.mean(keep_pres):+.2f} dB   "
          f"[{n - keep_viol}/{n} ok, {keep_viol} KEEP-VIOLATION (<{KEEP_VIOLATION_DB})]")
    print(f"SUPPRESS reduction (enh vs mix, want <<0 dB)   : "
          f"median {med(supp_red):+.2f} / mean {st.mean(supp_red):+.2f} dB   "
          f"[{len(supp_red) - supp_fail}/{len(supp_red)} ok, {supp_fail} SUPPRESS-FAIL (>{SUPPRESS_FAIL_DB})]")
    print(f"SI-SDR (enh vs target, whole clip)             : "
          f"median {med(sisdr):+.2f} / mean {st.mean(sisdr):+.2f} dB")
    print(f"\nper-item scores -> {set_dir / 'turntaking_scores.jsonl'}")


if __name__ == "__main__":
    main()
