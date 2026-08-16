"""Decompose the synthetic-to-real gap on matched VOiCES triples.

For a (room, mic) that has BOTH real retransmitted speech and a measured
impulse response, build three versions of the same utterance:

    R = the real recording               (distant-16k/speech)
    M = source (*) measured impulse      (distant-16k/room-response, same room+mic)
    S = source (*) synthetic bank RIR    (matched distance, similar RT60)

Everything except the construction is held constant, so the differences
decompose the gap:

    R vs M  ->  everything an LTI convolution cannot represent for this exact
                room+mic: transducer non-linearity, time variance, the
                recording's noise floor, level structure.
    M vs S  ->  the RIR model's fidelity (measured vs simulated impulse).

Reported per (room, mic):
  * fit_db(X->R): energy of R explained by the gain-matched, aligned X
    (a system-identification fit; higher = closer to the recording)
  * noise_floor_db(R): silent-gap level relative to speech-active level
  * behavioral (with --ckpt): the model's active-span suppression on R / M / S
    -- the training-relevant view of the same gap.

Loudspeaker orientation (dg token) varies per recording while the impulse was
measured at one orientation; the dg is recorded per row so orientation spread
can be separated if needed.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_domain_gap.py \\
        --voices-root /work/any_exp_link/puresound_exp/real_e2e_corpora/voices/VOiCES_rebuilt \\
        --bank /work/any_exp_link/puresound_exp/hybrid_rir_16k_levels/wide \\
        --config egs/voice_isolate/config/infer_dpcrn.yaml \\
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \\
        --per-mic 6 --out-json data_report/domain_gap.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_far_suppression import (  # noqa: E402
    VOICES_FNAME_RE,
    VOICES_MIC_DISTANCE_M,
    active_spans,
    load_model,
    quantiles,
    reduction_db,
)

from puresound.audio.io import AudioIO  # noqa: E402
from puresound.audio.rir.bank.loader import PreGeneratedRoomBank
from puresound.utils import fftconvolve  # noqa: E402


def _align_and_scale(x: torch.Tensor, ref: torch.Tensor, spans) -> torch.Tensor:
    """Shift x to max cross-correlation with ref, then LS-scale on active spans."""
    n = min(x.shape[-1], ref.shape[-1])
    a, b = ref.reshape(-1)[:n], x.reshape(-1)[:n]
    # coarse integer lag via FFT cross-correlation, search +-2000 samples
    corr = torch.fft.irfft(
        torch.fft.rfft(a, 2 * n) * torch.conj(torch.fft.rfft(b, 2 * n)), 2 * n
    )
    lags = torch.cat([corr[-2000:], corr[:2000]])
    lag = int(torch.argmax(lags)) - 2000
    if lag > 0:
        b = torch.cat([torch.zeros(lag), b[: n - lag]])
    elif lag < 0:
        b = torch.cat([b[-lag:], torch.zeros(-lag)])
    mask = torch.zeros(n, dtype=torch.bool)
    for s, e in spans:
        mask[s:min(e, n)] = True
    denom = float((b[mask] * b[mask]).sum())
    scale = float((a[mask] * b[mask]).sum()) / denom if denom > 0 else 0.0
    return (b * scale).view(1, -1)


def fit_db(x: torch.Tensor, ref: torch.Tensor, spans) -> float:
    """10log10( E[ref] / E[ref - x] ) over active spans (x already aligned+scaled)."""
    n = min(x.shape[-1], ref.shape[-1])
    mask = torch.zeros(n, dtype=torch.bool)
    for s, e in spans:
        mask[s:min(e, n)] = True
    r = ref.reshape(-1)[:n][mask]
    e = r - x.reshape(-1)[:n][mask]
    return float(10.0 * torch.log10(r.square().sum() / e.square().sum().clamp_min(1e-12)))


def noise_floor_db(ref: torch.Tensor, spans, sr: int) -> float:
    """Silent-gap RMS relative to active-span RMS (negative; higher = noisier)."""
    n = ref.shape[-1]
    act = torch.zeros(n, dtype=torch.bool)
    for s, e in spans:
        act[s:min(e, n)] = True
    sil = ~act
    # keep a 100 ms guard around active edges out of the "silence"
    guard = int(0.1 * sr)
    sil_idx = torch.where(sil)[0]
    if sil_idx.numel() < sr // 2:
        return float("nan")
    x = ref.reshape(-1)
    a_rms = x[act].square().mean().sqrt().clamp_min(1e-12)
    s_rms = x[sil][guard:-guard].square().mean().sqrt().clamp_min(1e-12) if sil_idx.numel() > 2 * guard else x[sil].square().mean().sqrt().clamp_min(1e-12)
    return float(20.0 * torch.log10(s_rms / a_rms))


def pick_bank_rirs(bank: PreGeneratedRoomBank, distance_m: float, k: int, tol: float, rng) -> list[torch.Tensor]:
    """Draw up to k far-channel RIRs whose distance is within +-tol of target."""
    out = []
    role = "interferer" if distance_m >= 1.0 else "foreground"
    for _ in range(400):
        if len(out) >= k:
            break
        scene = bank.sample_scene()
        rir, meta, _sr = bank.select_channel(
            scene, source_role=role,
            distance_range_override=[distance_m - tol, distance_m + tol],
        )
        if abs(float(meta["source_receiver_distance"]) - distance_m) <= tol:
            out.append(rir.reshape(-1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voices-root", required=True)
    ap.add_argument("--bank", required=True, help="synthetic bank view for the S leg")
    ap.add_argument("--config", default=str(REPO_ROOT / "egs/voice_isolate/config/infer_dpcrn.yaml"))
    ap.add_argument("--ckpt", default=None, help="checkpoint for the behavioral leg (optional)")
    ap.add_argument("--dry-blend", type=float, default=1.0)
    ap.add_argument("--rooms", nargs="*", default=["rm1", "rm2", "rm3", "rm4"])
    ap.add_argument("--mics", nargs="*", type=int, default=[1, 5],
                    help="mic ids (default: 1=closest, 5=farthest studio mics)")
    ap.add_argument("--per-mic", type=int, default=6, help="utterances per (room, mic)")
    ap.add_argument("--distance-tol", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1618)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    root = Path(args.voices_root)
    rng = random.Random(args.seed)
    bank = PreGeneratedRoomBank(folder=args.bank)
    model = load_model(args.config, args.ckpt, torch.device(args.device)) if args.ckpt else None

    rows = []
    for room in args.rooms:
        for mic in args.mics:
            dist = VOICES_MIC_DISTANCE_M.get((room, mic))
            imp = list((root / "distant-16k/room-response" / room / "impulse").glob(f"*-mc{mic:02d}-*.wav"))
            if dist is None or not imp:
                print(f"# skip {room} mc{mic:02d}: no impulse/distance")
                continue
            h, sr = AudioIO.open(f_path=str(imp[0]), target_lvl=None, resample_to=16000)
            h = h[0].reshape(-1)
            # trim the measured impulse: from 5 ms before its peak to +1.5 s
            pk = int(h.abs().argmax())
            h = h[max(0, pk - 80): pk + int(1.5 * sr)]

            speech_dir = root / "distant-16k/speech/train" / room / "none"
            files = sorted(speech_dir.rglob(f"*-mc{mic:02d}-*.wav"))
            picks = rng.sample(files, min(args.per_mic, len(files)))
            srirs = pick_bank_rirs(bank, dist, k=len(picks), tol=args.distance_tol, rng=rng)
            if not srirs:
                print(f"# skip {room} mc{mic:02d}: no bank RIR within {args.distance_tol} m of {dist} m")
                continue

            for i, f in enumerate(picks):
                m = VOICES_FNAME_RE.search(f.name)
                if m is None:
                    continue
                src = (root / "source-16k/train" / f"sp{m.group('spk')}" /
                       f"Lab41-SRI-VOiCES-src-sp{m.group('spk')}-ch{m.group('ch')}-sg{m.group('seg')}.wav")
                if not src.is_file():
                    continue
                R, _ = AudioIO.open(f_path=str(f), target_lvl=None, resample_to=16000)
                s, _ = AudioIO.open(f_path=str(src), target_lvl=None, resample_to=16000)
                R, s = R[0].reshape(1, -1), s[0].reshape(1, -1)
                spans = active_spans(R, sr)
                if sum(b - a for a, b in spans) < sr:
                    continue

                M = fftconvolve(s, h.view(1, -1), mode="full")[..., : R.shape[-1]]
                S = fftconvolve(s, srirs[i % len(srirs)].view(1, -1), mode="full")[..., : R.shape[-1]]
                M = _align_and_scale(M, R, spans)
                S = _align_and_scale(S, R, spans)

                row = {
                    "room": room, "mic": mic, "distance_m": dist,
                    "dg": f.name.rsplit("-", 1)[-1].replace(".wav", ""),
                    "file": f.name,
                    "fit_measured_db": round(fit_db(M, R, spans), 2),
                    "fit_synthetic_db": round(fit_db(S, R, spans), 2),
                    "noise_floor_db": round(noise_floor_db(R, spans, sr), 2),
                }
                if model is not None:
                    with torch.no_grad():
                        for tag, x in (("R", R), ("M", M), ("S", S)):
                            y = model(x.to(args.device), dry_blend=args.dry_blend)
                            y = y.detach().cpu().view(1, -1).clamp(-1, 1)
                            row[f"suppress_{tag}_db"] = round(reduction_db(y, x, spans), 2)
                rows.append(row)
            done = [r for r in rows if r["room"] == room and r["mic"] == mic]
            if done:
                _, fm, _ = quantiles([r["fit_measured_db"] for r in done])
                _, fs, _ = quantiles([r["fit_synthetic_db"] for r in done])
                print(f"# {room} mc{mic:02d} d={dist:.2f}m n={len(done)}  "
                      f"fit(M)={fm:.1f}dB fit(S)={fs:.1f}dB", flush=True)

    if args.out_json:
        Path(args.out_json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print()
    print("room\tmic\td_m\tn\tfit_M_dB\tfit_S_dB\tfloor_dB" +
          ("\tsup_R\tsup_M\tsup_S" if model is not None else ""))
    keys = sorted({(r["room"], r["mic"]) for r in rows})
    for room, mic in keys:
        rs = [r for r in rows if r["room"] == room and r["mic"] == mic]
        med = lambda k: quantiles([r[k] for r in rs])[1]  # noqa: E731
        line = (f"{room}\tmc{mic:02d}\t{rs[0]['distance_m']:.2f}\t{len(rs)}"
                f"\t{med('fit_measured_db'):7.2f}\t{med('fit_synthetic_db'):7.2f}"
                f"\t{med('noise_floor_db'):7.2f}")
        if model is not None:
            line += (f"\t{med('suppress_R_db'):5.2f}\t{med('suppress_M_db'):5.2f}"
                     f"\t{med('suppress_S_db'):5.2f}")
        print(line)
    print("\n# fit_M = ceiling of ANY LTI convolution for this room+mic (what R-M cannot explain")
    print("#         is non-linearity + time variance + noise floor); fit_S = the synthetic RIR.")
    print("# fit_M - fit_S = RIR-model fidelity gap; small = better RIRs would not help.")


if __name__ == "__main__":
    main()
