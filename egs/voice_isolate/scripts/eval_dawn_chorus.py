"""Evaluate foreground isolation on ai-coustics Dawn Chorus (real recordings).

Dawn Chorus (HuggingFace ``ai-coustics/dawn_chorus_en``) is the open eval set
released with Voice Focus 2.0: 450 real-world recordings of a foreground
speaker against competing speech / noise, each with the clean foreground audio
and a transcript. This script measures what the QVF2 deepdive measures:

1. SI-SDR / SDR of raw mix vs enhanced output against the clean foreground.
2. (optional, when an ASR backend is installed) WER broken into
   substitutions / insertions / deletions, raw vs enhanced -- insertion
   reduction is the headline number for "the agent stops hearing too much".

ASR backends are auto-detected: ``faster-whisper`` preferred, then
``openai-whisper``; without either, the script still reports signal metrics.

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/eval_dawn_chorus.py \
        egs/voice_isolate/config/infer_dpcrn.yaml \
        --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
        --dry-blend 0.9 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.dsp import wav_resampling
from puresound.config import load_recipe
from puresound.recipes import init_siso_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gate_flags import (add_presence_gate_arg, build_presence_gate,
                         add_onset_guard_arg, build_onset_guard)

DATASET_REPO = "ai-coustics/dawn_chorus_en"


def _zero_mean(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = _zero_mean(est)
    ref = _zero_mean(ref)
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    s_target = alpha * ref
    e_noise = est - s_target
    return 10.0 * np.log10(
        (float(np.dot(s_target, s_target)) + eps)
        / (float(np.dot(e_noise, e_noise)) + eps)
    )


def load_wav_bytes(blob: bytes, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    if sr != target_sr:
        t = torch.from_numpy(wav).view(1, -1)
        t, _ = wav_resampling(wav=t, origin_sr=sr, target_sr=target_sr, backend="sox")
        wav = t.view(-1).numpy()
    return wav


def load_model(config_path: str, ckpt_path: str, device: str):
    model = init_siso_model(
        load_recipe(config_path, expected_task="voice_isolation").model
    )
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.reload_checkpoint(state, load_loss_func=False)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: str,
    dry_blend: float = 1.0,
    spec_floor: float = 0.0,
    presence_gate=None,
    onset_guard=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    x = torch.from_numpy(noisy.astype(np.float32)).view(1, -1).to(device)
    y = model(x, dry_blend=dry_blend, spec_floor=spec_floor,
              presence_gate=presence_gate, onset_guard=onset_guard)
    if isinstance(y, (list, tuple)):
        y = y[0]
    # The VAD head writes its frame logits onto the backbone as a side output of
    # the forward() above (see puresound/system/siso.py:150). Pull them here and
    # convert to per-frame speech probability; None if the backbone has no head.
    vad_logits = getattr(getattr(model, "backbone", None), "last_vad_logits", None)
    vad_prob = (
        torch.sigmoid(vad_logits).detach().cpu().numpy().reshape(-1)
        if vad_logits is not None
        else None
    )
    return y.detach().cpu().numpy().reshape(-1), vad_prob


def vad_to_waveform(
    vad_prob: np.ndarray,
    length: int,
    threshold: float = 0.5,
    high: float = 0.9,
) -> np.ndarray:
    """Render the per-frame VAD decision as an audio-length step waveform:
    ``high`` (0.9) where the head predicts speech, 0 otherwise. Each frame is
    held for hop samples and the result is clipped/padded to ``length`` so it
    lines up sample-for-sample with the mix/enh/ref wavs in an audio editor."""
    frames = (vad_prob > threshold).astype(np.float32) * high
    hop = max(1, int(np.ceil(length / len(frames))))
    wave = np.repeat(frames, hop)[:length]
    if len(wave) < length:
        wave = np.pad(wave, (0, length - len(wave)))
    return wave


CONTEXT_CHOICES = ("none", "self", "background")
BG_PAD_SEC = 3.0
BG_MIN_SEC = 0.3


def add_context_arg(parser):
    """`--context` -- what streaming state the model gets before the utterance.

    Today every WER gate feeds each utterance from zero state; deployment never
    does. The prefix only touches the model input: the scored output is sliced
    back to the utterance, so WER stays comparable across the choices.
    """
    parser.add_argument("--context", default="none", choices=list(CONTEXT_CHOICES),
                        help="model warm-up prefix: none=cold start (default, "
                             "bit-identical to before), self=the mix prepended to "
                             "itself, background=~3 s of the mix's "
                             "foreground-inactive parts (room + bystanders only)")
    return parser


def background_pad(
    mix: np.ndarray,
    ref: np.ndarray,
    sr: int,
    pad_sec: float = BG_PAD_SEC,
    frame_ms: float = 20.0,
    rel_db: float = 40.0,
    abs_dbfs: float = -60.0,
) -> np.ndarray | None:
    """The foreground-inactive stretches of ``mix``, in time order, looped to
    ``pad_sec``. A 20 ms frame is inactive when the clean reference's frame
    energy is under (max frame energy - 40 dB) or under -60 dBFS. Returns None
    when under ``BG_MIN_SEC`` of them exist (caller falls back to no context)."""
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    n_frames = min(len(mix), len(ref)) // n
    if n_frames == 0:
        return None
    e = (ref[: n_frames * n].reshape(n_frames, n).astype(np.float64) ** 2).mean(axis=1)
    inactive = (e < e.max() * 10.0 ** (-rel_db / 10.0)) | (e < 10.0 ** (abs_dbfs / 10.0))
    pad = mix[: n_frames * n].reshape(n_frames, n)[inactive].reshape(-1)
    if len(pad) < int(BG_MIN_SEC * sr):
        return None
    need = int(pad_sec * sr)
    if len(pad) < need:
        pad = np.tile(pad, int(np.ceil(need / len(pad))))
    return pad[:need]


def context_input(
    mix: np.ndarray, ref: np.ndarray, sr: int, context: str
) -> tuple[np.ndarray, int, bool]:
    """(model input, samples to drop off the front of the output, fell back)."""
    if context == "self":
        return np.concatenate([mix, mix]), len(mix), False
    if context == "background":
        pad = background_pad(mix, ref, sr)
        if pad is None:
            return mix, 0, True
        return np.concatenate([pad, mix]), len(pad), False
    return mix, 0, False


def init_asr(backend: str, model_size: str, device: str):
    """Returns (name, transcribe_fn) or (None, None)."""
    if backend in ("auto", "faster-whisper"):
        try:
            from faster_whisper import WhisperModel

            m = WhisperModel(
                model_size,
                device="cuda" if device.startswith("cuda") else "cpu",
                compute_type="float16" if device.startswith("cuda") else "int8",
            )

            def transcribe(wav: np.ndarray, sr: int) -> str:
                segments, _ = m.transcribe(wav, language="en", beam_size=5)
                return " ".join(s.text for s in segments)

            return f"faster-whisper/{model_size}", transcribe
        except ImportError:
            if backend == "faster-whisper":
                raise
    if backend in ("auto", "openai-whisper"):
        try:
            import whisper

            m = whisper.load_model(model_size, device=device)

            def transcribe(wav: np.ndarray, sr: int) -> str:
                return m.transcribe(wav.astype(np.float32), language="en")["text"]

            return f"openai-whisper/{model_size}", transcribe
        except ImportError:
            if backend == "openai-whisper":
                raise
    if backend in ("azure", "azure-once"):
        import os
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as e:
            raise ImportError(
                "azure backend needs the Speech SDK: "
                "`uv pip install azure-cognitiveservices-speech`"
            ) from e

        key = os.environ.get("SPEECH_KEY") or os.environ.get("AZURE_SPEECH_KEY")
        region = os.environ.get("SPEECH_REGION") or os.environ.get("AZURE_SPEECH_REGION")
        if not key or not region:
            raise RuntimeError(
                "azure backend needs SPEECH_KEY + SPEECH_REGION env vars "
                "(or AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)"
            )
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_recognition_language = os.environ.get("SPEECH_LANG", "en-US")
        # model_size is ignored for Azure (cloud model); device is irrelevant.

        # "azure-once" = recognize_once(): ONE result, and the service stops at the
        # first end-of-speech it detects. Measured on the moderate WER set: 65% of the
        # deletions on the UNPROCESSED mix sit in the last third of the utterance
        # (uniform would be 33%), so a pause or a suppressed stretch truncates the
        # tail and inflates deletion for mix and enhanced alike. "azure" (default)
        # runs continuous recognition over the whole pushed buffer and concatenates
        # every Recognized segment -- the number an ASR downstream would actually see.
        once = backend == "azure-once"
        if not once:
            # do not end the utterance on a mid-sentence pause; the clips are <= 10 s
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2000")
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")

        def transcribe(wav: np.ndarray, sr: int) -> str:
            # float32 [-1,1] -> 16-bit little-endian PCM, fed via a push stream
            pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            fmt = speechsdk.audio.AudioStreamFormat(
                samples_per_second=int(sr), bits_per_sample=16, channels=1
            )
            stream = speechsdk.audio.PushAudioInputStream(fmt)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=speechsdk.audio.AudioConfig(stream=stream),
            )
            stream.write(pcm)
            stream.close()
            if once:
                result = recognizer.recognize_once()
                # display form (ITN/punct/caps) -> downstream EnglishTextNormalizer canonicalises it
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
                return ""  # NoMatch / silence -> empty hyp
            import threading
            parts, done = [], threading.Event()
            recognizer.recognized.connect(
                lambda evt: parts.append(evt.result.text)
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech else None)
            recognizer.session_stopped.connect(lambda evt: done.set())
            recognizer.canceled.connect(lambda evt: done.set())
            recognizer.start_continuous_recognition()
            done.wait(timeout=60.0)
            recognizer.stop_continuous_recognition()
            return " ".join(t for t in parts if t)

        return (f"azure-once/{region}" if once else f"azure/{region}"), transcribe
    return None, None


def wer_breakdown(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    tf = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    out = jiwer.process_words(
        refs, hyps, reference_transform=tf, hypothesis_transform=tf
    )
    n_ref = sum(len(r) for r in out.references)
    return {
        "wer": out.wer,
        "substitution_rate": out.substitutions / n_ref,
        "insertion_rate": out.insertions / n_ref,
        "deletion_rate": out.deletions / n_ref,
        "n_ref_words": n_ref,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config_path")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--limit", type=int, default=None, help="evaluate first N samples only")
    p.add_argument("--asr", default="auto",
                   choices=["auto", "none", "faster-whisper", "openai-whisper", "azure"])
    p.add_argument("--asr-model", default="small")
    add_presence_gate_arg(p)
    add_onset_guard_arg(p)
    add_context_arg(p)
    p.add_argument("--dry-blend", type=float, default=1.0,
                   help="inference over-suppression relief: enh*b + mix*(1-b)")
    p.add_argument("--spec-floor", type=float, default=0.0,
                   help="clamp enhanced |bin| to >= floor * |mix bin| (deletion cap)")
    p.add_argument("--save-audio", type=int, default=0,
                   help="save mix/enhanced/ref/vad wavs for the first N samples")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="prob above which the saved VAD wav reads 0.9 (else 0)")
    p.add_argument("--out", default=str(REPO_ROOT / "egs/voice_isolate/data_report/dawn_chorus"))
    args = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(DATASET_REPO, "eval.parquet", repo_type="dataset")
    table = pq.read_table(parquet_path)
    n_total = table.num_rows
    n_eval = min(args.limit, n_total) if args.limit else n_total
    print(f"Dawn Chorus: {n_total} samples, evaluating {n_eval}")

    model = load_model(args.config_path, args.ckpt, args.device)
    gate = build_presence_gate(args)
    guard = build_onset_guard(args)
    print(f"model loaded from {args.ckpt}")

    asr_name, transcribe = (None, None)
    if args.asr != "none":
        asr_name, transcribe = init_asr(args.asr, args.asr_model, args.device)
    print(f"ASR backend: {asr_name or 'none (signal metrics only)'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "wavs"
    if args.save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ids, refs, hyps_raw, hyps_enh = [], [], [], []
    n_ctx_fallback = 0
    for i in range(n_eval):
        item = table.slice(i, 1).to_pylist()[0]
        mix = load_wav_bytes(item["mix"]["bytes"], args.sr)
        ref = load_wav_bytes(item["speech"]["bytes"], args.sr)
        length = min(len(mix), len(ref))
        mix, ref = mix[:length], ref[:length]

        model_in, offset, fell_back = context_input(mix, ref, args.sr, args.context)
        n_ctx_fallback += int(fell_back)
        enh, vad_prob = run_inference(model, model_in, args.device,
                                      dry_blend=args.dry_blend,
                                      spec_floor=args.spec_floor,
                                      presence_gate=gate,
                                      onset_guard=guard)
        enh = enh[offset:][:length]
        if len(enh) < length:
            enh = np.pad(enh, (0, length - len(enh)))

        row = {
            "id": item["id"],
            "conversation_type": item["conversation_type"],
            "speech_source": item["speech_source"],
            "si_sdr_mix": si_sdr(mix, ref),
            "si_sdr_enh": si_sdr(enh, ref),
        }
        row["si_sdr_i"] = row["si_sdr_enh"] - row["si_sdr_mix"]
        if vad_prob is not None:
            row["vad_active_frac"] = float((vad_prob > 0.5).mean())
            row["vad_prob_mean"] = float(vad_prob.mean())

        if transcribe is not None:
            ids.append(item["id"])
            refs.append(item["transcript"])
            hyps_raw.append(transcribe(mix, args.sr))
            hyps_enh.append(transcribe(enh, args.sr))

        if args.save_audio and i < args.save_audio:
            sf.write(audio_dir / f"{item['id']}_mix.wav", mix, args.sr)
            sf.write(audio_dir / f"{item['id']}_enh.wav", enh, args.sr)
            sf.write(audio_dir / f"{item['id']}_ref.wav", ref, args.sr)
            if vad_prob is not None:
                vad_wave = vad_to_waveform(vad_prob, length, threshold=args.vad_threshold)
                sf.write(audio_dir / f"{item['id']}_vad.wav", vad_wave, args.sr)

        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == n_eval:
            print(f"  {i + 1}/{n_eval}  "
                  f"running SI-SDRi: {np.mean([r['si_sdr_i'] for r in rows]):+.2f} dB")

    report = {
        "dataset": DATASET_REPO,
        "n_eval": n_eval,
        "ckpt": args.ckpt,
        "asr": asr_name,
        "context": args.context,
        "n_context_fallback": n_ctx_fallback,
        "si_sdr_mix_mean": float(np.mean([r["si_sdr_mix"] for r in rows])),
        "si_sdr_enh_mean": float(np.mean([r["si_sdr_enh"] for r in rows])),
        "si_sdr_i_mean": float(np.mean([r["si_sdr_i"] for r in rows])),
        "si_sdr_i_median": float(np.median([r["si_sdr_i"] for r in rows])),
    }
    if "vad_active_frac" in rows[0]:
        report["vad_active_frac_mean"] = float(
            np.mean([r["vad_active_frac"] for r in rows])
        )
        report["vad_prob_mean"] = float(np.mean([r["vad_prob_mean"] for r in rows]))
    if transcribe is not None:
        report["wer_raw"] = wer_breakdown(refs, hyps_raw)
        report["wer_enhanced"] = wer_breakdown(refs, hyps_enh)

    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # per-utterance transcripts, so paired deletion can be recomputed offline
    tr_path = out_dir / "transcripts.jsonl"
    if transcribe is not None:
        with tr_path.open("w", encoding="utf-8") as fh:
            for uid, rf, hr, he in zip(ids, refs, hyps_raw, hyps_enh):
                fh.write(json.dumps({"id": uid, "ref": rf, "hyp_raw": hr,
                                     "hyp_enh": he}, ensure_ascii=False) + "\n")

    print("\n" + "=" * 64)
    print("DAWN CHORUS SUMMARY")
    print("=" * 64)
    print(f"context: {args.context}"
          + (f"  ({n_ctx_fallback} utterances fell back to none)"
             if n_ctx_fallback else ""))
    print(f"SI-SDR  mix -> enhanced : {report['si_sdr_mix_mean']:+.2f} -> "
          f"{report['si_sdr_enh_mean']:+.2f} dB  "
          f"(SI-SDRi mean {report['si_sdr_i_mean']:+.2f}, "
          f"median {report['si_sdr_i_median']:+.2f})")
    if "wer_raw" in report:
        for tag in ("wer_raw", "wer_enhanced"):
            w = report[tag]
            print(f"{tag:13s}: WER {w['wer']:.3f}  "
                  f"(sub {w['substitution_rate']:.3f} / "
                  f"ins {w['insertion_rate']:.3f} / "
                  f"del {w['deletion_rate']:.3f})")
    else:
        print("WER skipped -- install `faster-whisper` (or `openai-whisper`) "
              "and rerun for the QVF2-style WER breakdown.")
    print(f"\nper-sample -> {csv_path}\nreport     -> {report_path}")
    if transcribe is not None:
        print(f"transcripts -> {tr_path}")


if __name__ == "__main__":
    main()
