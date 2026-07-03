import argparse
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.io import AudioIO
from puresound.utils import create_folder, load_hparam


PYTORCH_CHECKPOINT_EXTENSIONS = (".ckpt", ".pt", ".pth")
ONNX_CHECKPOINT_EXTENSIONS = (".onnx",)
DEFAULT_CONFIG_PATH = "config/infer_dpcrn.yaml"
MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}
ORT_RUNTIME_CACHE: dict[tuple[str, str], Any] = {}
Metrics = None
LOGGER = logging.getLogger("voice_isolate_demo")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def report_progress(
    message: str,
    log_messages: list[str] | None = None,
    progress: Any | None = None,
    value: float | None = None,
) -> None:
    LOGGER.info(message)
    if log_messages is not None:
        log_messages.append(message)
    if progress is not None and value is not None:
        progress(value, desc=message)


def get_metrics_class():
    global Metrics
    if Metrics is None:
        from puresound.metrics import Metrics as _Metrics

        Metrics = _Metrics
    return Metrics


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()).joinpath(path).resolve()


def recipe_root_from_config(config_path: str | Path) -> Path:
    config_path = resolve_path(config_path)
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def load_demo_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = resolve_path(config_path)
    config = load_hparam(str(config_path))
    return config_path, recipe_root_from_config(config_path), config


def resolve_config_path(config_path: str | Path) -> str:
    return str(resolve_path(config_path))


def resolve_config_relative_path(path: str | Path | None, recipe_root: Path) -> Path | None:
    if not path:
        return None
    return resolve_path(path, base_dir=recipe_root)


def model_uses_distance(config: dict[str, Any]) -> bool:
    backbone_args = (
        config.get("model", {}).get("backbone", {}).get("backbone_args", {}) or {}
    )
    return int(backbone_args.get("distance_embedding_dim", 0) or 0) > 0


def backend_checkpoint_extensions(backend: str) -> tuple[str, ...]:
    if backend.lower().startswith("ort"):
        return ONNX_CHECKPOINT_EXTENSIONS
    return PYTORCH_CHECKPOINT_EXTENSIONS


def backend_artifact_name(backend: str) -> str:
    if backend.lower().startswith("ort"):
        return "ONNX model"
    return "checkpoint"


def validate_backend_artifact(backend: str, artifact_path: str | Path) -> None:
    artifact_path = Path(artifact_path)
    extensions = backend_checkpoint_extensions(backend)
    if artifact_path.suffix.lower() not in extensions:
        artifact = backend_artifact_name(backend)
        expected = ", ".join(extensions)
        raise ValueError(f"{backend} requires a {artifact} file with extension: {expected}.")


def scan_checkpoints(config_path: str | Path, backend: str = "PyTorch offline") -> list[tuple[str, str]]:
    config_path, recipe_root, config = load_demo_config(config_path)
    trainer_config = config.get("trainer", {})
    search_roots: list[Path] = []
    extensions = backend_checkpoint_extensions(backend)

    work_folder = resolve_config_relative_path(trainer_config.get("work_folder"), recipe_root)
    if work_folder is not None:
        search_roots.append(work_folder)
    LOGGER.info(
        "Scanning %s artifacts from config=%s roots=%s extensions=%s",
        backend,
        config_path,
        [str(root) for root in search_roots],
        extensions,
    )

    found: dict[Path, str] = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        for ext in extensions:
            for ckpt_path in root.rglob(f"*{ext}"):
                ckpt_path = ckpt_path.resolve()
                try:
                    label = str(ckpt_path.relative_to(recipe_root))
                except ValueError:
                    label = str(ckpt_path)
                found.setdefault(ckpt_path, label)

    choices = [(label, str(path)) for path, label in sorted(found.items(), key=lambda item: item[1])]
    LOGGER.info("Found %d checkpoint(s)", len(choices))
    return choices


def refresh_checkpoints(config_path: str | Path, backend: str = "PyTorch offline"):
    try:
        choices = scan_checkpoints(config_path, backend=backend)
    except Exception as exc:
        return gr.update(choices=[], value=None), f"Failed to scan checkpoints: {exc}"

    if not choices:
        artifact = backend_artifact_name(backend)
        extensions = ", ".join(backend_checkpoint_extensions(backend))
        return (
            gr.update(choices=[], value=None),
            f"No {artifact}s found under trainer.work_folder with extensions: {extensions}.",
        )

    artifact = backend_artifact_name(backend)
    return gr.update(choices=choices, value=choices[0][1]), f"Found {len(choices)} {artifact}(s)."


def load_model_from_checkpoint(
    config_path: str | Path, checkpoint_path: str | Path, device: torch.device
) -> torch.nn.Module:
    from puresound.recipes import init_siso_model, load_siso_recipe_config

    LOGGER.info("Loading model config from %s", config_path)
    (
        _corpus_dict,
        _trainer_dict,
        _optim_dict,
        _scheduler_dict,
        _loss_dict,
        model_dict,
        *_rest,
    ) = load_siso_recipe_config(str(config_path))

    model = init_siso_model(model_dict)
    LOGGER.info("Loading checkpoint from %s on %s", checkpoint_path, device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if hasattr(model, "reload_checkpoint"):
        model.reload_checkpoint(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def get_cached_model(config_path: str | Path, checkpoint_path: str | Path) -> tuple[torch.nn.Module, torch.device]:
    config_path = resolve_path(config_path)
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (str(config_path), str(checkpoint_path), str(device))
    if cache_key not in MODEL_CACHE:
        LOGGER.info("Model cache miss for checkpoint=%s device=%s", checkpoint_path, device)
        MODEL_CACHE[cache_key] = load_model_from_checkpoint(config_path, checkpoint_path, device)
    else:
        LOGGER.info("Model cache hit for checkpoint=%s device=%s", checkpoint_path, device)
    return MODEL_CACHE[cache_key], device


def get_cached_ort_runtime(onnx_path: str | Path, provider: str = "auto"):
    from puresound.streaming import StreamingDparnOrt

    onnx_path = resolve_path(onnx_path)
    cache_key = (str(onnx_path), provider)
    if cache_key not in ORT_RUNTIME_CACHE:
        LOGGER.info("ORT runtime cache miss for onnx=%s provider=%s", onnx_path, provider)
        ORT_RUNTIME_CACHE[cache_key] = StreamingDparnOrt(onnx_path=onnx_path, provider=provider)
    else:
        LOGGER.info("ORT runtime cache hit for onnx=%s provider=%s", onnx_path, provider)
    runtime = ORT_RUNTIME_CACHE[cache_key]
    runtime.reset()
    return runtime


def to_model_input(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.detach().float().cpu()
    if wav.dim() == 1:
        return wav.view(1, -1)
    if wav.dim() == 2:
        return wav[0].view(1, -1)
    raise ValueError(f"Unsupported audio shape: {tuple(wav.shape)}")


def rms_dbfs(wav: torch.Tensor, eps: float = 1e-12) -> float:
    wav = wav.detach().float().cpu()
    rms = torch.sqrt(torch.mean(wav.square())).clamp_min(eps)
    return float(20.0 * torch.log10(rms))


def peak_level(wav: torch.Tensor) -> float:
    return float(wav.detach().float().cpu().abs().max())


def clipping_ratio(wav: torch.Tensor, threshold: float = 0.999) -> float:
    wav = wav.detach().float().cpu()
    return float((wav.abs() >= threshold).float().mean())


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def compute_no_reference_metrics(
    input_wav: torch.Tensor, enhanced_wav: torch.Tensor, sample_rate: int
) -> tuple[list[list[str]], str | None]:
    input_wav = to_model_input(input_wav)
    enhanced_wav = to_model_input(enhanced_wav)
    duration = min(input_wav.shape[-1], enhanced_wav.shape[-1]) / float(sample_rate)

    input_rms = rms_dbfs(input_wav)
    enhanced_rms = rms_dbfs(enhanced_wav)
    rows = [
        ["sample_rate", _fmt(sample_rate), _fmt(sample_rate), "Hz"],
        ["duration", _fmt(duration), _fmt(duration), "seconds"],
        ["rms_dbfs", _fmt(input_rms), _fmt(enhanced_rms), _fmt(enhanced_rms - input_rms)],
        ["peak", _fmt(peak_level(input_wav)), _fmt(peak_level(enhanced_wav)), ""],
        [
            "clipping_ratio",
            _fmt(clipping_ratio(input_wav)),
            _fmt(clipping_ratio(enhanced_wav)),
            "",
        ],
    ]

    metrics_cls = get_metrics_class()
    try:
        energy_delta = float(metrics_cls.noise_reduction(input_wav, enhanced_wav).reshape(-1)[0])
        rows.append(["energy_delta_db", _fmt(energy_delta), "", "enhanced/input"])
    except Exception as exc:
        rows.append(["energy_delta_db", "n/a", "n/a", f"skipped: {exc}"])

    dnsmos_note = None
    try:
        input_dnsmos = metrics_cls.dnsmos_p835(input_wav, input_wav, sr=sample_rate)
        enhanced_dnsmos = metrics_cls.dnsmos_p835(input_wav, enhanced_wav, sr=sample_rate)
        for key in sorted(enhanced_dnsmos):
            before = float(input_dnsmos[key])
            after = float(enhanced_dnsmos[key])
            rows.append([key, _fmt(before), _fmt(after), _fmt(after - before)])
    except Exception as exc:
        dnsmos_note = f"DNSMOS skipped: {exc}"
        rows.append(["dnsmos_p835", "n/a", "n/a", dnsmos_note])

    return rows, dnsmos_note


def sanitize_filename(name: str) -> str:
    name = Path(name).stem or "audio"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "audio"


def output_folder_from_config(config: dict[str, Any], recipe_root: Path) -> Path:
    output_folder = config.get("dataset", {}).get("proc_output_folder") or "./proc"
    return resolve_path(output_folder, base_dir=recipe_root) / "demo"


def _spectrogram_db(wav: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    wav = to_model_input(wav).squeeze(0)
    if wav.numel() < n_fft:
        wav = torch.nn.functional.pad(wav, (0, n_fft - wav.numel()))
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-8)
    return 20.0 * torch.log10(mag)


def save_spectrogram_comparison(
    input_wav: torch.Tensor,
    enhanced_wav: torch.Tensor,
    sample_rate: int,
    output_path: str | Path,
) -> str:
    import matplotlib.pyplot as plt

    input_db = _spectrogram_db(input_wav)
    enhanced_db = _spectrogram_db(enhanced_wav)
    vmax = max(float(input_db.max()), float(enhanced_db.max()))
    vmin = vmax - 80.0
    extent = [
        0,
        input_wav.shape[-1] / float(sample_rate),
        0,
        sample_rate / 2,
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for ax, title, spec_db in zip(
        axes,
        ["Input spectrogram", "Enhanced spectrogram"],
        [input_db, enhanced_db],
    ):
        image = ax.imshow(
            spec_db.numpy(),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(image, ax=axes, label="Magnitude (dB)")
    output_path = str(output_path)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def enhance_audio(
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_audio_path: str | Path,
    backend: str = "PyTorch offline",
    ort_provider: str = "auto",
    query_distance: float | None = None,
    dry_blend: float = 1.0,
    spec_floor: float = 0.0,
    progress: Any | None = None,
) -> tuple[str, str, str, list[list[str]], str]:
    if not input_audio_path:
        raise ValueError("Please upload or record an input audio file.")
    if not checkpoint_path:
        raise ValueError("Please select a checkpoint.")

    log_messages: list[str] = []
    report_progress("Loading config", log_messages, progress, 0.03)
    config_path, recipe_root, config = load_demo_config(config_path)
    dataset_config = config.get("dataset", {})
    target_sample_rate = dataset_config.get("target_sample_rate")
    if target_sample_rate is None:
        raise ValueError("dataset.target_sample_rate must be set in the config.")
    target_sample_rate = int(target_sample_rate)
    report_progress(
        f"Using target sample rate {target_sample_rate} Hz without gain normalization",
        log_messages,
        progress,
        0.10,
    )

    use_ort = backend.lower().startswith("ort")
    validate_backend_artifact(backend, checkpoint_path)
    report_progress("Loading checkpoint/model", log_messages, progress, 0.18)
    pytorch_uses_distance = False
    if use_ort:
        runtime = get_cached_ort_runtime(checkpoint_path, provider=ort_provider)
        target_sample_rate = runtime.sample_rate
        device = f"onnxruntime:{','.join(runtime.providers)}"
        if runtime.uses_distance:
            d_value = float(query_distance) if query_distance is not None else 1.0
            runtime.set_query_distance(d_value)
            report_progress(
                f"ORT model uses distance conditioning; query_distance = {d_value:.2f} m",
                log_messages,
                progress,
                0.22,
            )
        elif query_distance is not None:
            report_progress(
                "ORT model has no distance conditioning; ignoring slider value",
                log_messages,
                progress,
                0.22,
            )
    else:
        model, device = get_cached_model(config_path, checkpoint_path)
        pytorch_uses_distance = model_uses_distance(config)
        if pytorch_uses_distance:
            report_progress(
                f"PyTorch model uses distance conditioning; query_distance = "
                f"{float(query_distance) if query_distance is not None else 1.0:.2f} m",
                log_messages,
                progress,
                0.22,
            )
        elif query_distance is not None:
            report_progress(
                "PyTorch model has no distance conditioning; ignoring slider value",
                log_messages,
                progress,
                0.22,
            )
    report_progress(f"Reading input audio: {input_audio_path}", log_messages, progress, 0.32)
    wav, sample_rate = AudioIO.open(
        f_path=str(input_audio_path),
        target_lvl=None,
        resample_to=target_sample_rate,
    )
    model_input = to_model_input(wav)
    if use_ort:
        report_progress(
            f"Running ORT streaming inference on {runtime.providers} for {model_input.shape[-1] / float(sample_rate):.2f}s audio",
            log_messages,
            progress,
            0.50,
        )
        samples = model_input.squeeze(0).detach().cpu().numpy()
        enhanced_np = runtime.process_samples(samples)
        enhanced_np = np.concatenate([enhanced_np, runtime.flush()])
        enhanced = torch.from_numpy(enhanced_np).view(1, -1).clamp(min=-1.0, max=1.0)
        if dry_blend < 1.0:  # over-suppression relief (waveform-level; works on ORT)
            mix_ref = model_input.detach().cpu().view(1, -1)
            n = min(enhanced.shape[-1], mix_ref.shape[-1])
            enhanced[..., :n] = (
                dry_blend * enhanced[..., :n] + (1.0 - dry_blend) * mix_ref[..., :n]
            ).clamp(min=-1.0, max=1.0)
        if spec_floor > 0.0:
            report_progress(
                "spec_floor applies to the PyTorch backend only; ignored for ORT streaming",
                log_messages, progress, 0.50,
            )
    else:
        report_progress(
            f"Running inference on {device} for {model_input.shape[-1] / float(sample_rate):.2f}s audio",
            log_messages,
            progress,
            0.50,
        )
        with torch.no_grad():
            if pytorch_uses_distance:
                d_value = float(query_distance) if query_distance is not None else 1.0
                qd_tensor = torch.tensor([d_value], dtype=torch.float32, device=device)
                enhanced = model(
                    model_input.to(device), query_distance=qd_tensor,
                    dry_blend=dry_blend, spec_floor=spec_floor,
                ).detach().cpu()
            else:
                enhanced = model(
                    model_input.to(device),
                    dry_blend=dry_blend, spec_floor=spec_floor,
                ).detach().cpu()
        enhanced = to_model_input(enhanced).clamp(min=-1.0, max=1.0)

    report_progress("Saving enhanced audio", log_messages, progress, 0.68)
    output_folder = output_folder_from_config(config, recipe_root)
    create_folder(str(output_folder))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_folder / f"{timestamp}_{sanitize_filename(str(input_audio_path))}_enhanced.wav"
    spectrogram_path = output_folder / f"{timestamp}_{sanitize_filename(str(input_audio_path))}_spectrogram.png"
    AudioIO.save(enhanced, str(output_path), int(sample_rate))
    report_progress("Rendering spectrogram comparison", log_messages, progress, 0.80)
    save_spectrogram_comparison(model_input, enhanced, int(sample_rate), spectrogram_path)

    report_progress("Computing no-reference metrics", log_messages, progress, 0.90)
    metrics_rows, dnsmos_note = compute_no_reference_metrics(model_input, enhanced, int(sample_rate))
    report_progress("Done", log_messages, progress, 1.0)
    status = f"Enhanced audio saved to {output_path}"
    if dnsmos_note:
        status = f"{status}\n{dnsmos_note}"
    status = "\n".join([status, "", "Progress:"] + [f"- {message}" for message in log_messages])
    return str(input_audio_path), str(output_path), str(spectrogram_path), metrics_rows, status


def run_demo_inference(
    config_path: str,
    checkpoint_path: str,
    input_audio_path: str,
    backend: str = "PyTorch offline",
    ort_provider: str = "auto",
    query_distance: float = 1.0,
    dry_blend: float = 1.0,
    spec_floor: float = 0.0,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        return enhance_audio(
            config_path,
            checkpoint_path,
            input_audio_path,
            backend=backend,
            ort_provider=ort_provider,
            query_distance=float(query_distance) if query_distance is not None else None,
            dry_blend=float(dry_blend),
            spec_floor=float(spec_floor),
            progress=progress,
        )
    except Exception as exc:
        LOGGER.exception("Demo inference failed")
        return input_audio_path, None, None, [], f"Error: {exc}"


def build_app(default_config_path: str = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    with gr.Blocks(title="PureSound Voice Isolate Demo") as app:
        gr.Markdown("# PureSound Voice Isolate Demo")
        config_path = gr.Textbox(label="Config Path", value=default_config_path)
        backend = gr.Radio(
            label="Backend",
            choices=["PyTorch offline", "ORT streaming"],
            value="PyTorch offline",
        )
        refresh_button = gr.Button("Refresh model list")
        checkpoint = gr.Dropdown(label="Checkpoint / ONNX Model", choices=[], value=None)
        ort_provider = gr.Dropdown(
            label="ORT Provider",
            choices=["auto", "cpu", "cuda"],
            value="auto",
        )
        query_distance = gr.Slider(
            label="Query distance (m)",
            minimum=0.3,
            maximum=5.0,
            step=0.1,
            value=1.0,
            info=(
                "Only honored when the loaded model has distance conditioning "
                "(backbone_args.distance_embedding_dim > 0). Matches the "
                "augmentation_query_distance range used at training time."
            ),
        )
        dry_blend = gr.Slider(
            label="Dry blend (over-suppression relief)",
            minimum=0.5,
            maximum=1.0,
            step=0.05,
            value=1.0,
            info=(
                "out = a*enhanced + (1-a)*input. 1.0 = off (default). Lower "
                "values blend the original mix back to recover deleted speech "
                "(trades a little interferer leakage for fewer deletions)."
            ),
        )
        spec_floor = gr.Slider(
            label="Spectral floor (over-suppression relief)",
            minimum=0.0,
            maximum=0.3,
            step=0.02,
            value=0.0,
            info=(
                "Clamps each enhanced magnitude bin to >= floor*|input| "
                "(keeps phase). 0.0 = off (default). PyTorch backend only."
            ),
        )
        input_audio = gr.Audio(
            label="Input Audio",
            sources=["upload", "microphone"],
            type="filepath",
            buttons=["download"],
        )
        run_button = gr.Button("Run")
        input_player = gr.Audio(label="Before", type="filepath")
        enhanced_player = gr.Audio(label="After", type="filepath")
        spectrogram_image = gr.Image(label="Spectrogram Comparison", type="filepath")
        metrics = gr.Dataframe(
            headers=["metric", "input", "enhanced", "delta_or_note"],
            label="Metrics",
            interactive=False,
        )
        status = gr.Textbox(label="Status", lines=4)

        refresh_button.click(refresh_checkpoints, inputs=[config_path, backend], outputs=[checkpoint, status])
        backend.change(refresh_checkpoints, inputs=[config_path, backend], outputs=[checkpoint, status])
        run_button.click(
            run_demo_inference,
            inputs=[
                config_path,
                checkpoint,
                input_audio,
                backend,
                ort_provider,
                query_distance,
                dry_blend,
                spec_floor,
            ],
            outputs=[input_player, enhanced_player, spectrogram_image, metrics, status],
        )

    return app


def main(args):
    configure_logging()
    app = build_app(default_config_path=args.config_path)
    app.launch(server_name=args.address, server_port=args.port, share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--address", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    main(parser.parse_args())
