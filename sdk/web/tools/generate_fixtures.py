"""Python SDK reference streams for the independent WebAssembly port."""

from pathlib import Path
import json
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sdk/python"))
from puresound_streaming import PureSoundStreamingRuntime

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/puresound-web-fixtures")
out.mkdir(parents=True, exist_ok=True)
models = {
    "noise-v3": "egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v3",
    "noise": "egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v2",
    "voice": "egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_curriculum_v1",
}
cases = []
for label, relative in models.items():
    stem = ROOT / relative
    for kind, n in [
        ("short", 7),
        ("silence", 1601),
        ("speech", 48317),
        ("guard", 128037),
    ]:
        x = np.zeros(n, dtype=np.float32)
        if kind == "short":
            x[:] = np.linspace(-0.1, 0.1, n, dtype=np.float32)
        if kind in {"speech", "guard"}:
            rng = np.random.default_rng(4)
            x[:] = rng.normal(0, 0.0001, n)
            x[6400:] = (
                0.1 * np.sin(np.arange(n - 6400) * 2 * np.pi * 700 / 16000)
                + rng.normal(0, 0.003, n - 6400)
            ).astype(np.float32)
        manifest = stem.with_suffix(".json")
        if kind == "guard":
            spec = json.loads(manifest.read_text())
            spec["onset_guard"] = {
                "t_arm_s": 0.4,
                "t_forget_s": 0.5,
                "tau_dn_s": 0.1,
                "init_s": 0.1,
                "floor_win_s": 0.3,
                "margin_db": 8,
            }
            x[48000:80000] = 0
            manifest = out / f"{label}-guard.json"
            manifest.write_text(json.dumps(spec))
        runtime = PureSoundStreamingRuntime(
            stem.with_suffix(".onnx"), manifest, provider="cpu"
        )
        y = np.concatenate([runtime.process_samples(x), runtime.flush()])
        name = f"{label}-{kind}"
        x.tofile(out / f"{name}-in.f32")
        y.tofile(out / f"{name}-out.f32")
        cases.append(
            {
                "name": name,
                "stem": str(stem),
                "manifest": str(manifest),
                "input": str(out / f"{name}-in.f32"),
                "output": str(out / f"{name}-out.f32"),
            }
        )
(out / "cases.json").write_text(json.dumps(cases))
wav_cases = []
for rate, channels, subtype in [
    (16000, 1, "PCM_16"),
    (44100, 2, "PCM_16"),
    (48000, 2, "FLOAT"),
]:
    wave = np.stack(
        [
            0.1 * np.sin(2 * np.pi * 500 * np.arange(1001) / rate) * (i + 1)
            for i in range(channels)
        ],
        axis=1,
    )
    name = f"{rate}-{channels}-{subtype}"
    wav_path = out / f"{name}.wav"
    sf.write(wav_path, wave, rate, subtype=subtype)
    decoded, _ = sf.read(wav_path, dtype="float32", always_2d=True)
    expected = decoded.mean(axis=1)
    expected.tofile(out / f"{name}.f32")
    wav_cases.append(
        {"file": str(wav_path), "reference": str(out / f"{name}.f32"), "rate": rate}
    )
(out / "wav-cases.json").write_text(json.dumps(wav_cases))
print(f"{len(cases)} reference streams: {out}")
