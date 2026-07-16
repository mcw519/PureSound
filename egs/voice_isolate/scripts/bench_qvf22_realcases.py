"""Real-case benchmark against ai-coustics Voice Focus 2.2 sample clips.

3 real-world clips scraped from the ai-coustics Voice Focus 2.2 announcement
(https://ai-coustics.com/blog/voice-focus-2.2-speaker-isolation): "raw audio"
input and their own "Quail Voice Focus 2.2" enhanced output, per scenario
(single near-field speaker / single far-field speaker / near+far compete --
exact scenario-to-file mapping is inferred from page order, not labelled in
the source). No clean reference or transcript is available, so this is a
no-reference comparison only (DNSMOS, RMS/energy delta, clipping) -- not a
SI-SDR/WER benchmark like eval_but_wer.py / eval_dawn_chorus.py.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/bench_qvf22_realcases.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_wide_antisup_ep19.ckpt \
        --cases-dir egs/voice_isolate/data_report/qvf22_real_cases \
        --out-dir egs/voice_isolate/data_report/qvf22_real_cases_bench \
        --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.metrics import Metrics
from puresound.recipes import init_siso_model, load_siso_recipe_config


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    (
        _corpus_dict,
        _trainer_dict,
        _optim_dict,
        _scheduler_dict,
        _loss_dict,
        model_dict,
        *_rest,
    ) = load_siso_recipe_config(config_path)
    model = init_siso_model(model_dict)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def no_ref_row(name: str, wav: torch.Tensor, sr: int) -> dict:
    row = {"name": name}
    row["rms_dbfs"] = float(20.0 * torch.log10(wav.square().mean().sqrt().clamp_min(1e-12)))
    row["peak"] = float(wav.abs().max())
    row["clipping_ratio"] = float((wav.abs() >= 0.999).float().mean())
    try:
        dnsmos = Metrics.dnsmos_p835(wav, wav, sr=sr)
        row.update(dnsmos)
    except Exception as exc:  # pragma: no cover -- optional dependency
        row["dnsmos_error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cases-dir", default="data_report/qvf22_real_cases")
    parser.add_argument("--out-dir", default="data_report/qvf22_real_cases_bench")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.config_path, args.ckpt, device)

    raw_files = sorted(cases_dir.glob("scenario*_raw.wav"))
    if not raw_files:
        raise FileNotFoundError(f"no scenario*_raw.wav found under {cases_dir}")

    rows: list[dict] = []
    for raw_path in raw_files:
        scenario = raw_path.stem.split("_raw")[0]
        qvf22_path = cases_dir / f"{scenario}_qvf22.wav"
        if not qvf22_path.is_file():
            print(f"skip {scenario}: missing {qvf22_path.name}")
            continue

        raw_wav, sr = AudioIO.open(f_path=str(raw_path), target_lvl=None, resample_to=16000)
        raw_wav = raw_wav.view(1, -1)
        with torch.no_grad():
            ours = model(raw_wav.to(device)).detach().cpu().clamp(min=-1.0, max=1.0)
        qvf22_wav, _ = AudioIO.open(f_path=str(qvf22_path), target_lvl=None, resample_to=16000)
        qvf22_wav = qvf22_wav.view(1, -1)

        AudioIO.save(raw_wav, str(out_dir / f"{scenario}_1raw.wav"), sr)
        AudioIO.save(ours, str(out_dir / f"{scenario}_2ours.wav"), sr)
        AudioIO.save(qvf22_wav, str(out_dir / f"{scenario}_3qvf22.wav"), sr)

        rows.append(no_ref_row(f"{scenario}_raw", raw_wav, sr))
        rows.append(no_ref_row(f"{scenario}_ours", ours, sr))
        rows.append(no_ref_row(f"{scenario}_qvf22", qvf22_wav, sr))

        n = min(ours.shape[-1], raw_wav.shape[-1])
        ours_nr = float(Metrics.noise_reduction(raw_wav[..., :n], ours[..., :n]))
        n2 = min(qvf22_wav.shape[-1], raw_wav.shape[-1])
        qvf22_nr = float(Metrics.noise_reduction(raw_wav[..., :n2], qvf22_wav[..., :n2]))
        print(f"{scenario}: energy_delta_db ours={ours_nr:.2f} qvf22={qvf22_nr:.2f} (raw->enhanced, dB)")

    header = ["name", "rms_dbfs", "peak", "clipping_ratio", "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr"]
    print()
    print("\t".join(header))
    for row in rows:
        print("\t".join(f"{row.get(k, ''):.4f}" if isinstance(row.get(k), float) else str(row.get(k, "")) for k in header))

    print(f"\nwrote {len(rows) // 3} scenario(s) x 3 wavs to {out_dir}")


if __name__ == "__main__":
    main()
