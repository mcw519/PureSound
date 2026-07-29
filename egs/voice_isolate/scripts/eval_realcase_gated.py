"""Real-case voicebot scorecard WITH the near-field gate applied to the output.

eval_realcase_faronly.py scores `model(raw)`, i.e. the mask path only. Because gate
training freezes the separator, that scorecard is identical for every gate epoch (the
gate head never touches the audio). This script instead APPLIES the trained gate as a
per-frame multiplicative gain on the enhanced output, so the scorecard reflects
whether the learned near-field gate closes the far-only passthrough gap:

  gated[t] = enhanced[t] * sigmoid(gate_logit[frame(t)])          (soft gate)
  gated[t] = enhanced[t] * (sigmoid(gate_logit[frame(t)]) >= thr) (hard gate)

Reported side by side per clip/span:
  * ours_raw   -- frozen separator, no gate (baseline; == eval_realcase_faronly.py)
  * ours_soft  -- separator * soft gate probability
  * ours_hard  -- separator * hard gate (threshold)
  * qvf22      -- reference ceiling

keep_preserv_dB ~ 0 is good (a large negative on a keep span = the gate muted a real
user). suppress_reduc_dB very negative is good (far-only voice silenced). This is the
direct test of the gate's product value; still a handful of real clips, not a corpus.

Usage (from egs/voice_isolate):
    uv run python scripts/eval_realcase_gated.py config/infer_dpcrn.yaml \
        --ckpt exp/dpcrn_gate_synth/lightning_logs/version_0/checkpoints/epoch=5-step=1500.ckpt \
        --hop 160 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.recipes import init_siso_model, load_siso_recipe_config  # noqa: E402

SUPPRESS_FAIL_DB = -6.0
KEEP_VIOLATION_DB = -3.0


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    (_c, _t, _o, _s, _l, model_dict, *_rest) = load_siso_recipe_config(config_path)
    model = init_siso_model(model_dict)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_func_list.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"# [ckpt] missing {len(missing)}: {missing[:3]}")
    return model.to(device).eval()


def gate_gain(logits: torch.Tensor, n_samples: int, hop: int, threshold: float | None) -> torch.Tensor:
    """Upsample per-frame gate logits to a sample-rate multiplicative gain in [0,1]."""
    prob = torch.sigmoid(logits.reshape(-1).float())
    if threshold is not None:
        prob = (prob >= threshold).float()
    gain = prob.repeat_interleave(hop)
    if gain.shape[-1] < n_samples:
        gain = torch.cat([gain, gain[-1:].expand(n_samples - gain.shape[-1])])
    return gain[:n_samples].view(1, -1)


def window_power_db(processed: torch.Tensor, raw: torch.Tensor, spans, sr: int) -> float:
    n = min(processed.shape[-1], raw.shape[-1])
    p_e = r_e = 0.0
    for a, b in spans:
        i, j = int(a * sr), min(int(b * sr), n)
        if j <= i:
            continue
        p_e += float(processed[..., i:j].square().sum())
        r_e += float(raw[..., i:j].square().sum())
    if r_e <= 0.0:
        return float("nan")
    return 10.0 * torch.log10(torch.tensor(p_e / r_e + 1e-12)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cases-dir", default="data_report/qvf22_real_cases")
    parser.add_argument("--windows", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hop", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    cases_dir = Path(args.cases_dir)
    windows_path = Path(args.windows) if args.windows else cases_dir / "windows.json"
    windows = {k: v for k, v in json.loads(windows_path.read_text()).items() if not k.startswith("_")}

    model = load_model(args.config_path, args.ckpt, device)

    print("\t".join(["clip", "role", "system", "keep_preserv_dB", "suppress_reduc_dB", "verdict"]))
    fails = {"ours_raw": [], "ours_soft": [], "ours_hard": [], "qvf22": []}
    for clip, spec in windows.items():
        raw_path = cases_dir / f"{clip}_raw.wav"
        if not raw_path.is_file():
            print(f"# skip {clip}: missing {raw_path.name}")
            continue
        raw_wav, sr = AudioIO.open(f_path=str(raw_path), target_lvl=None, resample_to=16000)
        raw_wav = raw_wav.view(1, -1)
        with torch.no_grad():
            enh = model(raw_wav.to(device)).detach().cpu().view(1, -1).clamp(-1.0, 1.0)
            logits = model.backbone.last_vad_logits
        if logits is None:
            raise RuntimeError("backbone.last_vad_logits is None; gate head disabled?")
        logits = logits.detach().cpu()
        n = enh.shape[-1]
        soft = (enh * gate_gain(logits, n, args.hop, None)).clamp(-1.0, 1.0)
        hard = (enh * gate_gain(logits, n, args.hop, args.threshold)).clamp(-1.0, 1.0)

        systems = {"ours_raw": enh, "ours_soft": soft, "ours_hard": hard}
        qvf22_path = cases_dir / f"{clip}_qvf22.wav"
        if qvf22_path.is_file():
            qvf22_wav, _ = AudioIO.open(f_path=str(qvf22_path), target_lvl=None, resample_to=16000)
            systems["qvf22"] = qvf22_wav.view(1, -1)

        keep_spans = spec.get("keep", [])
        supp_spans = spec.get("suppress", [])
        role = spec.get("role", "")
        for sysname, processed in systems.items():
            keep = window_power_db(processed, raw_wav, keep_spans, sr) if keep_spans else float("nan")
            supp = window_power_db(processed, raw_wav, supp_spans, sr) if supp_spans else float("nan")
            flags = []
            if keep == keep and keep < KEEP_VIOLATION_DB:
                flags.append("KEEP-VIOLATION")
                fails.setdefault(sysname, []).append(f"{clip}:keep")
            if supp == supp and supp > SUPPRESS_FAIL_DB:
                flags.append("SUPPRESS-FAIL")
                fails.setdefault(sysname, []).append(f"{clip}:suppress")
            verdict = ",".join(flags) if flags else "ok"
            ks = f"{keep:8.2f}" if keep == keep else "     n/a"
            ss = f"{supp:8.2f}" if supp == supp else "     n/a"
            print(f"{clip}\t{role}\t{sysname}\t{ks}\t{ss}\t{verdict}")

    print()
    print(f"# keep ~0 dB; < {KEEP_VIOLATION_DB} = KEEP-VIOLATION.  suppress very negative; > {SUPPRESS_FAIL_DB} = SUPPRESS-FAIL")
    for sysname in ("ours_raw", "ours_soft", "ours_hard", "qvf22"):
        f = fails.get(sysname, [])
        print(f"# {sysname}: {'PASS' if not f else 'FAIL -> ' + ', '.join(f)}")


if __name__ == "__main__":
    main()
