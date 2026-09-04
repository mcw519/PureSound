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
from puresound.inference import (
    ModelZoo,
    PROVIDER_CHOICES,
    load_model as load_inference_model,
)
from puresound.utils import create_folder, load_hparam


PYTORCH_CHECKPOINT_EXTENSIONS = (".ckpt", ".pt", ".pth")
ONNX_CHECKPOINT_EXTENSIONS = (".onnx",)
DEFAULT_CONFIG_PATH = "config/infer_dpcrn.yaml"
MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}
ORT_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
STT_CACHE: dict[tuple[str, str, str], Any] = {}
Metrics = None
LOGGER = logging.getLogger("voice_isolate_demo")

STREAMING_SAMPLE_RATE = 16000


def default_torch_device() -> str:
    """Choose the best PyTorch device without assuming NVIDIA is present."""

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    try:
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        # Older PyTorch builds or non-macOS builds may not expose a usable MPS
        # backend. Keep the legacy CPU fallback in those environments.
        pass
    return "cpu"


DEVICE = default_torch_device()


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
    """The recipe root is the parent of the nearest ancestor named `config`,
    so configs nested under config/exp/ resolve the same root as config/."""
    config_path = resolve_path(config_path)
    for parent in config_path.parents:
        if parent.name == "config":
            return parent.parent
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
    from puresound.config import load_recipe
    from puresound.recipes import init_siso_model

    LOGGER.info("Loading model config from %s", config_path)
    model = init_siso_model(
        load_recipe(config_path, expected_task="voice_isolation").model
    )
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

    device = torch.device(default_torch_device())
    cache_key = (str(config_path), str(checkpoint_path), str(device))
    if cache_key not in MODEL_CACHE:
        LOGGER.info("Model cache miss for checkpoint=%s device=%s", checkpoint_path, device)
        MODEL_CACHE[cache_key] = load_model_from_checkpoint(config_path, checkpoint_path, device)
    else:
        LOGGER.info("Model cache hit for checkpoint=%s device=%s", checkpoint_path, device)
    return MODEL_CACHE[cache_key], device


def get_cached_ort_runtime(onnx_path: str | Path, provider: str = "auto",
                           collect_extras: bool = False,
                           postprocess_overrides: dict[str, Any] | None = None):
    from puresound.streaming import StreamingDparnOrt

    onnx_path = resolve_path(onnx_path)
    # collect_extras is part of the key: a gated run must not reuse a runtime
    # built without collection, which would silently hand back no logits.
    override_key = tuple(sorted((postprocess_overrides or {}).items()))
    cache_key = (str(onnx_path), provider, bool(collect_extras), override_key)
    if cache_key not in ORT_RUNTIME_CACHE:
        LOGGER.info("ORT runtime cache miss for onnx=%s provider=%s", onnx_path, provider)
        runtime_kwargs = {
            "onnx_path": onnx_path,
            "provider": provider,
            "collect_extras": collect_extras,
        }
        if postprocess_overrides:
            runtime_kwargs["postprocess_overrides"] = postprocess_overrides
        ORT_RUNTIME_CACHE[cache_key] = StreamingDparnOrt(**runtime_kwargs)
    else:
        LOGGER.info("ORT runtime cache hit for onnx=%s provider=%s", onnx_path, provider)
    runtime = ORT_RUNTIME_CACHE[cache_key]
    runtime.reset()
    return runtime


def _zoo_voice_isolation_choices(backend: str) -> list[tuple[str, str]]:
    """Return UI choices from the catalog, with the release default first."""
    zoo = ModelZoo.default()
    models = zoo.list(task="voice_isolation")
    models.sort(key=lambda model: ("default" not in model.roles, model.id))
    choices: list[tuple[str, str]] = []
    if backend.lower().startswith("ort"):
        for model in models:
            for artifact in model.artifacts:
                path = zoo.path_for(artifact.path)
                if path.is_file():
                    label = model.display_name
                    if artifact.variant != "default":
                        label = f"{label} [{artifact.variant}]"
                    choices.append((label, str(path.resolve())))
    else:
        for model in models:
            if not model.source_checkpoint:
                continue
            path = zoo.path_for(model.source_checkpoint)
            if path.is_file():
                choices.append((model.display_name, str(path.resolve())))
    return choices


def refresh_model_choices(config_path: str | Path, backend: str = "PyTorch offline"):
    """Refresh demo choices from Model Zoo, retaining old scan as a fallback."""
    try:
        choices = _zoo_voice_isolation_choices(backend)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Model Zoo unavailable; falling back to work-folder scan: %s", exc)
        try:
            choices = scan_checkpoints(config_path, backend=backend)
        except Exception:
            choices = []
    if not choices:
        kind = "ONNX models" if backend.lower().startswith("ort") else "checkpoints"
        return gr.update(choices=[], value=None), f"No {kind} registered in the Model Zoo."
    return gr.update(choices=choices, value=choices[0][1]), f"Model Zoo: {len(choices)} {backend_artifact_name(backend)} choice(s)."


def _catalog_model_for_path(path: str | Path):
    try:
        return ModelZoo.default().find_by_artifact(path)
    except Exception:
        return None


def _catalog_model_for_ref(ref: str | Path):
    """Resolve a logical id or an artifact path to a catalog entry."""
    try:
        zoo = ModelZoo.default()
        try:
            model = zoo.get(str(ref))
            return model, zoo.resolve_artifact(model.id)
        except KeyError:
            return zoo.find_by_artifact(ref)
    except Exception:
        return None


def model_default_dry_blend(ref: str | Path):
    """Gradio update for the catalog's recommended post-processing preset."""
    entry = _catalog_model_for_ref(ref)
    if entry is None:
        return gr.update()
    model, artifact = entry
    del artifact
    value = model.recommended_inference.get("dry_blend")
    return gr.update(value=float(value)) if value is not None else gr.update()


def to_model_input(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.detach().float().cpu()
    if wav.dim() == 1:
        return wav.view(1, -1)
    if wav.dim() == 2:
        return wav[0].view(1, -1)
    raise ValueError(f"Unsupported audio shape: {tuple(wav.shape)}")


def apply_vad_gate(
    enhanced: torch.Tensor,
    vad_logits: torch.Tensor | None,
    mode: str,
    threshold: float,
    hop_length: int,
    ema_alpha: float = 0.2,
    attack: float = 0.25,
    release: float = 0.05,
) -> torch.Tensor:
    """Apply the optional frame-level near-field gate to waveform output."""
    if mode == "Off":
        return enhanced
    if vad_logits is None:
        raise ValueError(
            "Gate is enabled, but this checkpoint has no VAD head. "
            "Use train_dpcrn_v2_sepgate.yaml with a gate checkpoint."
        )
    if hop_length <= 0:
        raise ValueError(f"Invalid gate hop length: {hop_length}")

    probability = torch.sigmoid(vad_logits[0].reshape(-1).float())
    frame_gain = gate_gain_from_probability(
        probability,
        mode,
        threshold,
        ema_alpha=ema_alpha,
        attack=attack,
        release=release,
    ).to(enhanced.dtype)

    gain = frame_gain.repeat_interleave(hop_length)
    n_samples = enhanced.shape[-1]
    if gain.numel() == 0:
        raise ValueError("Gate head returned no frame logits.")
    if gain.shape[-1] < n_samples:
        gain = torch.cat([gain, gain[-1:].expand(n_samples - gain.shape[-1])])
    return (enhanced * gain[:n_samples].view(1, -1)).clamp(min=-1.0, max=1.0)


def gate_gain_from_probability(
    probability: torch.Tensor,
    mode: str,
    threshold: float,
    ema_alpha: float = 0.2,
    attack: float = 0.25,
    release: float = 0.05,
) -> torch.Tensor:
    """Convert frame probabilities into raw, binary, or smoothed gate gains."""
    aliases = {"Soft": "Raw probability", "Hard": "Binary"}
    mode = aliases.get(mode, mode)
    if mode == "Raw probability":
        return probability
    if mode not in {"Binary", "Binary + EMA", "Binary + envelope"}:
        raise ValueError(f"Unknown gate mode: {mode}")

    binary = (probability >= threshold).float()
    if mode == "Binary":
        return binary
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in (0, 1], got {ema_alpha}")
    if not 0.0 < attack <= 1.0 or not 0.0 < release <= 1.0:
        raise ValueError("Envelope attack/release must be in (0, 1]")

    smoothed = torch.empty_like(binary)
    previous = binary[0]
    smoothed[0] = previous
    for index in range(1, binary.numel()):
        current = binary[index]
        coefficient = (
            ema_alpha
            if mode == "Binary + EMA"
            else (attack if current > previous else release)
        )
        previous = previous + coefficient * (current - previous)
        smoothed[index] = previous
    return smoothed


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
    gate_probability: torch.Tensor | None = None,
    gate_hop: int = 160,
    gate_threshold: float = 0.5,
    gate_mode: str = "Off",
    head_probabilities: dict[str, np.ndarray] | None = None,
) -> str:
    """Input / enhanced spectrograms, optionally what the heads say and what was applied.

    `head_probabilities` and `gate_probability` are deliberately separate panels.
    The first is the model's opinion; the second is the gain that reached the
    audio. Drawing them as one curve hides the case that matters -- a head that
    says "absent" while the gate is Off, which is exactly what a diagnostic run
    is looking for.
    """
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

    has_gate = gate_probability is not None
    has_heads = bool(head_probabilities)
    n_panels = 2 + int(has_heads) + int(has_gate)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(11, 3 + 2.5 * n_panels),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, title, spec_db in zip(
        axes[:2],
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
    panel = 2
    if has_heads:
        ax = axes[panel]
        styles = {
            "near": ("tab:blue", "near presence  p(a user is talking)"),
            "background": ("tab:orange", "background  p(non-target speech)"),
        }
        for name, values in head_probabilities.items():
            colour, label = styles.get(name, ("tab:grey", name))
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            t = np.arange(values.shape[0], dtype=np.float32) * gate_hop / sample_rate
            ax.plot(t, values, color=colour, linewidth=1.2, label=label)
        ax.axhline(gate_threshold, color="tab:red", linestyle="--", linewidth=1,
                   label=f"threshold={gate_threshold:.2f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("p")
        ax.set_title("VAD head outputs (model opinion -- not applied unless Gate mode is on)")
        ax.legend(loc="upper right", fontsize=8)
        panel += 1
    if has_gate:
        gate = gate_probability.detach().float().cpu().reshape(-1).numpy()
        gate_time = np.arange(gate.shape[0], dtype=np.float32) * gate_hop / sample_rate
        axes[panel].step(gate_time, gate, where="post", color="tab:green",
                         label="applied gate gain")
        axes[panel].axhline(
            gate_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label=f"threshold={gate_threshold:.2f}",
        )
        axes[panel].set_ylim(-0.05, 1.05)
        axes[panel].set_ylabel("Gate")
        axes[panel].set_title(f"Applied near-field gate ({gate_mode})")
        axes[panel].legend(loc="upper right")
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
    dry_blend: float = 1.0,
    spec_floor: float = 0.0,
    gate_mode: str = "Off",
    gate_threshold: float = 0.5,
    gate_ema_alpha: float = 0.2,
    gate_attack: float = 0.25,
    gate_release: float = 0.05,
    show_vad_heads: bool = False,
    progress: Any | None = None,
) -> tuple[str, str, str, list[list[str]], str]:
    if not input_audio_path:
        raise ValueError("Please upload or record an input audio file.")
    if not checkpoint_path:
        raise ValueError("Please select a checkpoint.")

    log_messages: list[str] = []
    use_ort = backend.lower().startswith("ort")
    catalog_ref = _catalog_model_for_ref(checkpoint_path) if use_ort else None
    report_progress("Loading config", log_messages, progress, 0.03)
    try:
        config_path, recipe_root, config = load_demo_config(config_path)
    except Exception:
        if not (use_ort and catalog_ref is not None):
            raise
        # Catalog-backed ONNX inference has its own audio contract.  Keep the
        # config optional for API callers; output files fall back beside the
        # input when no legacy recipe is available.
        config_path = Path(str(config_path))
        recipe_root = REPO_ROOT
        config = {}
        log_messages.append("Catalog-backed ORT run: legacy config unavailable; using Model Zoo metadata")
    dataset_config = config.get("dataset", {})
    target_sample_rate = dataset_config.get("target_sample_rate")
    if target_sample_rate is None and catalog_ref is not None:
        target_sample_rate = catalog_ref[0].sample_rate
    if target_sample_rate is None:
        raise ValueError("dataset.target_sample_rate must be set in the config.")
    target_sample_rate = int(target_sample_rate)
    report_progress(
        f"Using target sample rate {target_sample_rate} Hz without gain normalization",
        log_messages,
        progress,
        0.10,
    )

    if use_ort and catalog_ref is not None and Path(str(checkpoint_path)).suffix.lower() != ".onnx":
        checkpoint_path = ModelZoo.default().artifact_path(catalog_ref[0].id, catalog_ref[1].variant)
    validate_backend_artifact(backend, checkpoint_path)
    report_progress("Loading checkpoint/model", log_messages, progress, 0.18)
    gate_probability = None
    head_probabilities: dict[str, np.ndarray] = {}
    gate_hop = 160
    catalog_runtime = None
    ort_override_applied = False
    if use_ort:
        catalog_entry = _catalog_model_for_path(checkpoint_path) or catalog_ref
        if catalog_entry is not None:
            model_spec, artifact_spec = catalog_entry
            catalog_runtime = load_inference_model(
                model_spec.id,
                provider=ort_provider,
                variant=artifact_spec.variant,
            )
            runtime = catalog_runtime
            ort_override_applied = True
        else:
            runtime_kwargs: dict[str, Any] = {
                "provider": ort_provider,
                "collect_extras": gate_mode != "Off" or show_vad_heads,
            }
            if float(dry_blend) < 1.0:
                runtime_kwargs["postprocess_overrides"] = {"dry_blend": float(dry_blend)}
            try:
                runtime = get_cached_ort_runtime(checkpoint_path, **runtime_kwargs)
                ort_override_applied = "postprocess_overrides" in runtime_kwargs
            except TypeError:
                # Third-party callers/tests may still monkeypatch the old
                # three-argument helper.  Keep that compatibility bridge while
                # catalog-backed runs always use the facade above.
                runtime = get_cached_ort_runtime(
                    checkpoint_path,
                    provider=ort_provider,
                    collect_extras=gate_mode != "Off" or show_vad_heads,
                )
        if show_vad_heads and not runtime.extra_names:
            raise ValueError(
                f"Show VAD heads needs an ONNX graph that exports them; "
                f"{Path(checkpoint_path).name} exports nothing. Export with "
                "config/infer_dpcrn_heads.yaml (see "
                "pretrained_ckpt/streaming/dpcrn_v11_ep19_heads.onnx)."
            )
        if gate_mode != "Off" and "vad_logit" not in runtime.extra_names:
            raise ValueError(
                f"Gate mode needs an ONNX graph that exports vad_logit; "
                f"{Path(checkpoint_path).name} exports {runtime.extra_names or 'nothing'}. "
                "Export with config/infer_dpcrn_heads.yaml (see "
                "pretrained_ckpt/streaming/dpcrn_v11_ep19_heads.onnx)."
            )
        target_sample_rate = runtime.sample_rate
        device = f"onnxruntime:{','.join(runtime.providers)}"
    else:
        model, device = get_cached_model(config_path, checkpoint_path)
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
        if catalog_runtime is not None:
            parameters: dict[str, Any] = {
                "dry_blend": float(dry_blend),
                "collect_extras": gate_mode != "Off" or show_vad_heads,
            }
            if float(spec_floor) > 0.0:
                log_messages.append(
                    "spec_floor applies to the PyTorch backend only; ignored for ORT streaming"
                )
            result = catalog_runtime.infer(
                inputs={"audio": str(input_audio_path)}, parameters=parameters
            )
            enhanced_np = np.asarray(result.outputs["audio"], dtype=np.float32)
            history = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in result.outputs.items()
                if name != "audio"
            }
            lead = int(result.metadata.get("streaming_delay_frames", 0))
        else:
            enhanced_np = runtime.process_samples(samples)
            enhanced_np = np.concatenate([enhanced_np, runtime.flush()])
            # Drain ONCE: drain_extras clears, so a second call would hand back
            # nothing and silently blank whichever consumer ran second.
            history = (runtime.drain_extras()
                       if (gate_mode != "Off" or show_vad_heads) else {})
            lead = int(runtime.manifest.get("streaming_delay_frames", 0))
        enhanced = torch.from_numpy(enhanced_np).view(1, -1).clamp(min=-1.0, max=1.0)
        if show_vad_heads:
            gate_hop = runtime.hop_length
            for port, name in (("vad_logit", "near"),
                               ("background_vad_logit", "background")):
                if port in history:
                    v = history[port][lead:] if lead else history[port]
                    head_probabilities[name] = torch.sigmoid(
                        torch.from_numpy(v)).numpy()
        if gate_mode != "Off":
            gate_hop = runtime.hop_length
            logits = torch.from_numpy(history["vad_logit"]).view(1, -1)
            # The graph's logits LEAD the emitted audio by streaming_delay_frames:
            # they describe the bottleneck frame they came from, which the output
            # has not reached yet. Dropping that many from the front is what puts
            # a logit on the sample it actually judges -- without it the gate acts
            # 30 ms early and clips the front of every kept span.
            if lead:
                logits = logits[..., lead:]
            probability = torch.sigmoid(logits[0].reshape(-1))
            gate_probability = gate_gain_from_probability(
                probability,
                gate_mode,
                float(gate_threshold),
                ema_alpha=float(gate_ema_alpha),
                attack=float(gate_attack),
                release=float(gate_release),
            )
            enhanced = apply_vad_gate(
                enhanced,
                logits,
                gate_mode,
                float(gate_threshold),
                gate_hop,
                ema_alpha=float(gate_ema_alpha),
                attack=float(gate_attack),
                release=float(gate_release),
            )
            log_messages.append(
                f"Applied {gate_mode.lower()} VAD gate from the ONNX graph "
                f"(threshold={float(gate_threshold):.2f}, hop={gate_hop}, "
                f"logit lead={lead} frames compensated)"
            )
        if dry_blend < 1.0 and not ort_override_applied:  # legacy runtime bridge
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
            enhanced = model(
                model_input.to(device),
                dry_blend=dry_blend, spec_floor=spec_floor,
            ).detach().cpu()
        enhanced = to_model_input(enhanced).clamp(min=-1.0, max=1.0)
        if show_vad_heads:
            gate_hop = int(
                config.get("model", {})
                .get("encoder", {})
                .get("encoder_args", {})
                .get("hop_length", 160)
            )
            # Offline, a head's frame lines up with the output frame -- there is
            # no algorithmic delay to compensate, unlike the streaming graph.
            found = False
            for attr, name in (("last_vad_logits", "near"),
                               ("last_background_vad_logits", "background")):
                logits = getattr(model.backbone, attr, None)
                if logits is not None:
                    head_probabilities[name] = torch.sigmoid(
                        logits[0].detach().float().cpu().reshape(-1)).numpy()
                    found = True
            if not found:
                raise ValueError(
                    "Show VAD heads needs a config whose backbone enables them; "
                    f"{Path(config_path).name} builds none. Use "
                    "config/infer_dpcrn_heads.yaml."
                )
        if gate_mode != "Off":
            gate_hop = int(
                config.get("model", {})
                .get("encoder", {})
                .get("encoder_args", {})
                .get("hop_length", 160)
            )
            vad_logits = getattr(model.backbone, "last_vad_logits", None)
            if vad_logits is not None:
                probability = torch.sigmoid(vad_logits[0].detach().cpu().reshape(-1))
                gate_probability = gate_gain_from_probability(
                    probability,
                    gate_mode,
                    float(gate_threshold),
                    ema_alpha=float(gate_ema_alpha),
                    attack=float(gate_attack),
                    release=float(gate_release),
                )
            enhanced = apply_vad_gate(
                enhanced,
                vad_logits.detach().cpu() if vad_logits is not None else None,
                gate_mode,
                float(gate_threshold),
                gate_hop,
                ema_alpha=float(gate_ema_alpha),
                attack=float(gate_attack),
                release=float(gate_release),
            )
            report_progress(
                f"Applied {gate_mode.lower()} VAD gate "
                f"(threshold={float(gate_threshold):.2f}, hop={gate_hop}, "
                f"ema_alpha={float(gate_ema_alpha):.2f}, "
                f"attack={float(gate_attack):.2f}, "
                f"release={float(gate_release):.2f})",
                log_messages,
                progress,
                0.62,
            )

    report_progress("Saving enhanced audio", log_messages, progress, 0.68)
    output_folder = output_folder_from_config(config, recipe_root)
    create_folder(str(output_folder))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_folder / f"{timestamp}_{sanitize_filename(str(input_audio_path))}_enhanced.wav"
    spectrogram_path = output_folder / f"{timestamp}_{sanitize_filename(str(input_audio_path))}_spectrogram.png"
    AudioIO.save(enhanced, str(output_path), int(sample_rate))
    report_progress("Rendering spectrogram comparison", log_messages, progress, 0.80)
    save_spectrogram_comparison(
        model_input,
        enhanced,
        int(sample_rate),
        spectrogram_path,
        gate_probability=gate_probability,
        gate_hop=gate_hop,
        gate_threshold=float(gate_threshold),
        gate_mode=gate_mode,
        head_probabilities=head_probabilities or None,
    )

    if head_probabilities:
        summary = ", ".join(
            f"{name} p median {float(np.median(v)):.3f}"
            for name, v in head_probabilities.items()
        )
        log_messages.append(f"VAD heads plotted ({summary})")
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
    dry_blend: float = 1.0,
    spec_floor: float = 0.0,
    gate_mode: str = "Off",
    gate_threshold: float = 0.5,
    gate_ema_alpha: float = 0.2,
    gate_attack: float = 0.25,
    gate_release: float = 0.05,
    show_vad_heads: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        return enhance_audio(
            config_path,
            checkpoint_path,
            input_audio_path,
            backend=backend,
            ort_provider=ort_provider,
            dry_blend=float(dry_blend),
            spec_floor=float(spec_floor),
            gate_mode=gate_mode,
            gate_threshold=float(gate_threshold),
            gate_ema_alpha=float(gate_ema_alpha),
            gate_attack=float(gate_attack),
            gate_release=float(gate_release),
            show_vad_heads=bool(show_vad_heads),
            progress=progress,
        )
    except Exception as exc:
        LOGGER.exception("Demo inference failed")
        return input_audio_path, None, None, [], f"Error: {exc}"


# ---------------------------------------------------------------------------
# Realtime microphone streaming (ORT streaming backbone) + optional live STT
# ---------------------------------------------------------------------------


def _chunk_to_float_mono(sr: int, data: np.ndarray) -> np.ndarray:
    """Gradio numpy audio chunk -> mono float32 in [-1, 1]."""
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        return (data.astype(np.float32) / max_val) if max_val else data.astype(np.float32)
    return data.astype(np.float32)


def _resample_to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if int(sr) == STREAMING_SAMPLE_RATE or data.size == 0:
        return data.astype(np.float32)
    import librosa

    return librosa.resample(data, orig_sr=int(sr), target_sr=STREAMING_SAMPLE_RATE).astype(np.float32)


def get_cached_stt(backend: str, model_size: str, device: str):
    """Lazily build and cache a transcribe(np.float32 @16k) -> str function."""
    cache_key = (backend, model_size, device)
    if cache_key not in STT_CACHE:
        LOGGER.info("STT cache miss backend=%s model=%s device=%s", backend, model_size, device)
        if backend == "openai-whisper":
            import whisper

            model = whisper.load_model(model_size, device=device)

            def transcribe(wav: np.ndarray) -> str:
                return model.transcribe(wav.astype(np.float32))["text"].strip()
        else:  # faster-whisper (default)
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_size,
                device="cuda" if device.startswith("cuda") else "cpu",
                compute_type="float16" if device.startswith("cuda") else "int8",
            )

            def transcribe(wav: np.ndarray) -> str:
                segments, _ = model.transcribe(wav.astype(np.float32), beam_size=1)
                return " ".join(segment.text for segment in segments).strip()

        STT_CACHE[cache_key] = transcribe
    return STT_CACHE[cache_key]


def _new_realtime_session() -> dict[str, Any]:
    return {
        "runtime": None,
        "key": None,
        "stt_buf": np.zeros(0, dtype=np.float32),
        "transcript": "",
        "in_all": [],   # raw mic input (16k mono), saved as the "Before" WAV on stop
        "enh_all": [],  # every enhanced chunk, concatenated + saved as the "After" WAV on stop
    }


def realtime_start():
    """Fired on start_recording: reset display + state for a new take.

    The ORT runtime is built lazily on the first audio chunk (see realtime_stream), NOT
    here: building it loads the ONNX and takes time, and if the first mic chunk arrived
    before this handler finished it would race and clobber the runtime into the state.
    """
    return _new_realtime_session(), "", "Recording... speak into the mic.", None, None


def _ensure_runtime(session: dict[str, Any], onnx_path: str, provider: str):
    """Build the per-session ORT runtime on first use; returns (runtime, error_or_None)."""
    if session.get("runtime") is not None:
        return session["runtime"], None
    if not onnx_path:
        return None, "Select an ONNX streaming model first (Refresh model list)."
    try:
        catalog_entry = _catalog_model_for_path(onnx_path)
        if catalog_entry is not None:
            model_spec, artifact_spec = catalog_entry
            runtime = load_inference_model(
                model_spec.id,
                provider=provider,
                variant=artifact_spec.variant,
            )
            runtime.reset()
        else:
            from puresound.streaming import StreamingDparnOrt

            runtime = StreamingDparnOrt(onnx_path=resolve_path(onnx_path), provider=provider)
            runtime.reset()
        session["runtime"] = runtime
        session["key"] = (str(onnx_path), provider)
        return runtime, None
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Realtime runtime init failed")
        return None, f"Failed to init streaming runtime: {exc}"


def realtime_stream(
    new_chunk,
    session: dict[str, Any] | None,
    onnx_path: str,
    provider: str,
    stt_enabled: bool,
    stt_backend: str,
    stt_model_size: str,
    stt_interval: float,
):
    """Fired per mic chunk: lazily build the runtime, enhance the chunk, accumulate it,
    and optionally transcribe. Runtime is built here (not in start_recording) so it can
    never be clobbered by a chunk that arrives mid-initialization."""
    if session is None:
        session = _new_realtime_session()
    transcript = session.get("transcript", "")
    if new_chunk is None:
        return None, transcript, session

    runtime, err = _ensure_runtime(session, onnx_path, provider)
    if runtime is None:
        return None, err or transcript, session

    sr, data = new_chunk
    samples16k = _resample_to_16k(_chunk_to_float_mono(sr, data), sr)
    if samples16k.size:
        session.setdefault("in_all", []).append(samples16k.copy())
    enhanced = runtime.process_samples(samples16k)
    if enhanced.size:
        session.setdefault("enh_all", []).append(enhanced)

    if stt_enabled and enhanced.size:
        session["stt_buf"] = np.concatenate([session["stt_buf"], enhanced])
        needed = int(max(1.0, float(stt_interval)) * STREAMING_SAMPLE_RATE)
        if session["stt_buf"].shape[0] >= needed:
            segment = session["stt_buf"]
            session["stt_buf"] = np.zeros(0, dtype=np.float32)
            try:
                text = get_cached_stt(stt_backend, stt_model_size, DEVICE)(segment)
                if text:
                    transcript = (transcript + " " + text).strip()
                    session["transcript"] = transcript
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime STT failed")
                transcript = f"{transcript}\n[STT error: {exc}]".strip()
                session["transcript"] = transcript

    out_audio = (STREAMING_SAMPLE_RATE, enhanced) if enhanced.size else None
    return out_audio, transcript, session


def realtime_stop(
    session: dict[str, Any] | None,
    stt_enabled: bool,
    stt_backend: str,
    stt_model_size: str,
    config_path: str,
):
    """Fired on stop_recording: flush the runtime tail, transcribe any remainder, and
    write the full enhanced take to a seekable WAV (the live streaming player reports a
    0 s / non-seekable track once the stream ends, so the saved file is the real result).
    """
    if session is None or session.get("runtime") is None:
        return (session or {}).get("transcript", ""), "Stopped (no audio captured).", None, None, session
    runtime = session["runtime"]
    tail = runtime.flush()
    transcript = session.get("transcript", "")

    if tail.size:
        session.setdefault("enh_all", []).append(tail)

    if stt_enabled:
        leftover = session.get("stt_buf", np.zeros(0, dtype=np.float32))
        if tail.size:
            leftover = np.concatenate([leftover, tail])
        if leftover.size > STREAMING_SAMPLE_RATE // 10:  # >0.1s worth of audio
            try:
                text = get_cached_stt(stt_backend, stt_model_size, DEVICE)(leftover)
                if text:
                    transcript = (transcript + " " + text).strip()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Realtime STT (final) failed")
                transcript = f"{transcript}\n[STT error: {exc}]".strip()
        session["stt_buf"] = np.zeros(0, dtype=np.float32)
        session["transcript"] = transcript

    output_folder = None
    try:
        _, recipe_root, config = load_demo_config(config_path)
        output_folder = output_folder_from_config(config, recipe_root)
        create_folder(str(output_folder))
    except Exception:  # noqa: BLE001
        LOGGER.exception("Realtime output folder resolution failed")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    def _save(chunks: list, suffix: str) -> str | None:
        if not chunks or output_folder is None:
            return None
        full = np.concatenate(chunks)
        if not full.size:
            return None
        try:
            path = str(output_folder / f"{timestamp}_realtime_{suffix}.wav")
            wav = torch.from_numpy(full).clamp(min=-1.0, max=1.0).view(1, -1)
            AudioIO.save(wav, path, STREAMING_SAMPLE_RATE)
            return path
        except Exception:  # noqa: BLE001
            LOGGER.exception("Realtime save failed (%s)", suffix)
            return None

    enh_chunks = session.get("enh_all", [])
    input_path = _save(session.get("in_all", []), "input")
    enhanced_path = _save(enh_chunks, "enhanced")
    session["in_all"] = []
    session["enh_all"] = []

    duration = 0.0 if not enh_chunks else float(sum(c.shape[0] for c in enh_chunks)) / STREAMING_SAMPLE_RATE
    status = f"Stopped. {duration:.2f}s saved to {enhanced_path}" if enhanced_path else "Stopped. No audio captured."
    return transcript, status, input_path, enhanced_path, session


def build_realtime_tab(default_config_path: str) -> None:
    gr.Markdown(
        "Realtime microphone streaming through the **ORT streaming** backbone "
        "(per-frame ONNX). Pick an `.onnx` model, then press the mic record button. "
        "Enhanced audio plays back live; optional STT is **off by default**.\n\n"
        "Fixed latency = model look-ahead (`streaming_delay_frames`, ~30 ms for the "
        "voice-isolate recipe) plus your chunk size. Only ORT models work here."
    )
    session_state = gr.State(None)
    rt_config = gr.Textbox(label="Config Path (for model discovery)", value=default_config_path)
    with gr.Row():
        rt_refresh = gr.Button("Refresh model list")
        rt_provider = gr.Dropdown(label="ORT Provider", choices=list(PROVIDER_CHOICES), value="auto")
    try:
        rt_initial_choices = _zoo_voice_isolation_choices("ORT streaming")
    except Exception:
        rt_initial_choices = []
    rt_onnx = gr.Dropdown(
        label="ONNX Streaming Model",
        choices=rt_initial_choices,
        value=rt_initial_choices[0][1] if rt_initial_choices else None,
    )
    with gr.Group():
        stt_enabled = gr.Checkbox(label="Enable realtime STT", value=False)
        with gr.Row():
            stt_backend = gr.Dropdown(
                label="STT backend",
                choices=["faster-whisper", "openai-whisper"],
                value="faster-whisper",
            )
            stt_model_size = gr.Dropdown(
                label="STT model",
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="base",
            )
            stt_interval = gr.Slider(
                label="STT segment length (s)",
                minimum=1.0,
                maximum=10.0,
                step=0.5,
                value=4.0,
                info="Enhanced audio is transcribed in segments of this length; longer = fewer, more accurate calls.",
            )
    rt_audio_in = gr.Audio(
        label="Microphone",
        sources=["microphone"],
        streaming=True,
        type="numpy",
    )
    rt_enhanced_out = gr.Audio(
        label="Enhanced (live monitor)",
        streaming=True,
        autoplay=True,
    )
    with gr.Row():
        rt_input_file = gr.Audio(
            label="Before — recorded mic input (16k)",
            type="filepath",
        )
        rt_enhanced_file = gr.Audio(
            label="After — enhanced (seekable, downloadable)",
            type="filepath",
        )
    rt_transcript = gr.Textbox(label="Realtime transcript", lines=6)
    rt_status = gr.Textbox(label="Status", lines=2)

    rt_refresh.click(
        lambda cp: refresh_model_choices(cp, "ORT streaming"),
        inputs=[rt_config],
        outputs=[rt_onnx, rt_status],
    )
    rt_audio_in.start_recording(
        realtime_start,
        inputs=[],
        outputs=[session_state, rt_transcript, rt_status, rt_input_file, rt_enhanced_file],
    )
    rt_audio_in.stream(
        realtime_stream,
        inputs=[
            rt_audio_in, session_state, rt_onnx, rt_provider,
            stt_enabled, stt_backend, stt_model_size, stt_interval,
        ],
        outputs=[rt_enhanced_out, rt_transcript, session_state],
        stream_every=0.5,
        show_progress="hidden",
    )
    rt_audio_in.stop_recording(
        realtime_stop,
        inputs=[session_state, stt_enabled, stt_backend, stt_model_size, rt_config],
        outputs=[rt_transcript, rt_status, rt_input_file, rt_enhanced_file, session_state],
    )


def build_app(default_config_path: str = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    with gr.Blocks(title="PureSound Voice Isolate Demo") as app:
        gr.Markdown("# PureSound Voice Isolate Demo")
        with gr.Tab("Offline / File"):
            build_offline_tab(default_config_path)
        with gr.Tab("Realtime Mic"):
            build_realtime_tab(default_config_path)

    return app


def build_offline_tab(default_config_path: str) -> None:
    with gr.Column():
        config_path = gr.Textbox(label="Config Path", value=default_config_path)
        backend = gr.Radio(
            label="Backend",
            choices=["PyTorch offline", "ORT streaming"],
            value="PyTorch offline",
        )
        refresh_button = gr.Button("Refresh model list")
        try:
            initial_choices = _zoo_voice_isolation_choices("PyTorch offline")
        except Exception:
            initial_choices = []
        checkpoint = gr.Dropdown(
            label="Checkpoint / ONNX Model",
            choices=initial_choices,
            value=initial_choices[0][1] if initial_choices else None,
        )
        ort_provider = gr.Dropdown(
            label="ORT Provider",
            choices=list(PROVIDER_CHOICES),
            value="auto",
        )
        dry_blend = gr.Slider(
            label="Dry blend (over-suppression relief)",
            minimum=0.5,
            maximum=1.0,
            step=0.05,
            value=0.9,
            info=(
                "out = a*enhanced + (1-a)*input. The catalog release preset is "
                "0.9; 1.0 turns blending off. Lower "
                "values blend the original mix back to recover deleted speech "
                "(trades a little interferer leakage for fewer deletions)."
            ),
        )
        show_vad_heads = gr.Checkbox(
            label="Show VAD head outputs",
            value=False,
            info=(
                "Plots what the presence heads say -- near (a user is talking) "
                "and background (non-target speech) -- WITHOUT applying them. "
                "Needs a heads-enabled config or ONNX export "
                "(config/infer_dpcrn_heads.yaml). Independent of the gate below: "
                "leave the gate Off to read the heads without touching the audio."
            ),
        )
        gate_mode = gr.Radio(
            label="Near-field gate",
            choices=[
                "Off",
                "Raw probability",
                "Binary",
                "Binary + EMA",
                "Binary + envelope",
            ],
            value="Off",
            info=(
                "Applies the near head to the audio. Works on PyTorch and on ORT "
                "streaming when the graph exports vad_logit. Raw probability uses "
                "sigmoid(gate); the other modes threshold first, then optionally "
                "smooth the binary gain. MEASURED: soft costs a keep span up to "
                "-35.5 dB and binary up to -61.6 dB, and the head is inverted on "
                "some recording chains -- a diagnostic, not a deployment setting."
            ),
        )
        gate_threshold = gr.Slider(
            label="Gate threshold",
            minimum=0.1,
            maximum=0.9,
            step=0.05,
            value=0.5,
            info="Used by binary modes. Higher values suppress more aggressively.",
        )
        gate_ema_alpha = gr.Slider(
            label="Gate EMA alpha",
            minimum=0.01,
            maximum=1.0,
            step=0.01,
            value=0.2,
            info="Used by Binary + EMA. Lower values make switching smoother.",
        )
        with gr.Row():
            gate_attack = gr.Slider(
                label="Envelope attack",
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=0.25,
                info="Used by Binary + envelope. Higher values turn on faster.",
            )
            gate_release = gr.Slider(
                label="Envelope release",
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=0.05,
                info="Used by Binary + envelope. Higher values turn off faster.",
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

        refresh_button.click(refresh_model_choices, inputs=[config_path, backend], outputs=[checkpoint, status])
        backend.change(refresh_model_choices, inputs=[config_path, backend], outputs=[checkpoint, status])
        checkpoint.change(model_default_dry_blend, inputs=[checkpoint], outputs=[dry_blend])
        run_button.click(
            run_demo_inference,
            inputs=[
                config_path,
                checkpoint,
                input_audio,
                backend,
                ort_provider,
                dry_blend,
                spec_floor,
                gate_mode,
                gate_threshold,
                gate_ema_alpha,
                gate_attack,
                gate_release,
                show_vad_heads,
            ],
            outputs=[input_player, enhanced_player, spectrogram_image, metrics, status],
        )


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
