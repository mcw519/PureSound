"""Build the web app's on-device inference payload from pinned dependencies and
zoo artifacts: the compiled streaming runtime, ONNX Runtime Web, and each
released model's ONNX file with a strict JSON manifest, plus a SHA-256 catalog
that the page checks every download against."""

import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT / "sdk/web"
DEST = ROOT / "puresound/web/static/device"
# Everything under these is generated; the rest of DEST is version-controlled.
GENERATED = ["assets", "runtime", "vendor"]
ORT_DIST = SDK / "node_modules/onnxruntime-web/dist"
ORT_FILES = [
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
]
RUNTIME_FILES = ["runtime.js", "audio.js"]
MODELS = {
    "noise-suppression-dpcrn-mamba-v3": ROOT
    / "egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v3",
    "noise-suppression-dpcrn-mamba-v2": ROOT
    / "egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v2",
    "voice-isolate-dpcrn-curriculum-v2": ROOT
    / "egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_curriculum_v2",
    "voice-isolate-dpcrn-curriculum-v1": ROOT
    / "egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_curriculum_v1",
}


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def missing_inputs():
    """What the payload is built from that is not there yet."""

    inputs = [SDK / "dist" / name for name in RUNTIME_FILES]
    inputs += [ORT_DIST / name for name in ORT_FILES]
    inputs += [SDK / "licenses/LICENSE", SDK / "licenses/ThirdPartyNotices.txt"]
    for stem in MODELS.values():
        inputs += [stem.with_suffix(".json"), stem.with_suffix(".onnx")]
    return [str(path) for path in inputs if not path.is_file()]


def main():
    missing = missing_inputs()
    if missing:
        raise SystemExit(
            "Cannot build the device payload; run `npm ci` and `npm run build` in sdk/web first. Missing:\n  "
            + "\n  ".join(missing)
        )
    for folder in GENERATED:
        shutil.rmtree(DEST / folder, ignore_errors=True)
    for folder in ["assets", "runtime", "vendor/ort"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    for source in (SDK / "dist").glob("*.js"):
        shutil.copyfile(source, DEST / "runtime" / source.name)
    for name in ORT_FILES:
        shutil.copyfile(ORT_DIST / name, DEST / "vendor/ort" / name)
    ort_license = (SDK / "licenses/LICENSE").read_text()
    ort_notices = (SDK / "licenses/ThirdPartyNotices.txt").read_text()
    (DEST / "THIRD_PARTY_NOTICES.txt").write_text(
        "ONNX Runtime Web 1.24.3\n" + ort_license + "\n" + ort_notices
    )
    catalog = []
    keys = [
        "processor",
        "sample_rate",
        "fft_length",
        "win_length",
        "hop_length",
        "freq_bins",
        "output_names",
        "state_input_names",
        "state_output_names",
        "state_shapes",
        "streaming_delay_frames",
        "algorithmic_latency",
        "onset_guard",
    ]
    for model, stem in MODELS.items():
        original = json.loads(stem.with_suffix(".json").read_text())
        manifest = {k: original[k] for k in keys if k in original}
        manifest["recommended_inference"] = {
            k: original.get("recommended_inference", {}).get(k, default)
            for k, default in [
                ("dry_blend", 1),
                ("spec_floor", 0),
                ("mix_phase", False),
            ]
        }
        (DEST / f"assets/{model}.json").write_text(
            json.dumps(sanitize(manifest), allow_nan=False)
        )
        shutil.copyfile(stem.with_suffix(".onnx"), DEST / f"assets/{model}.onnx")
        catalog.append(
            {
                "id": model,
                "manifest": f"assets/{model}.json",
                "model": f"assets/{model}.onnx",
            }
        )
    files = {
        str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in GENERATED
        for p in sorted((DEST / folder).rglob("*"))
        if p.is_file()
    }
    version = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    (DEST / "catalog.json").write_text(
        json.dumps({"version": version, "models": catalog, "files": files}, indent=2)
    )
    print(f"Device inference assets built: {DEST} ({version})")


if __name__ == "__main__":
    main()
