# PureSound Model Zoo

`catalog.yaml` is the version-controlled registry for the ONNX artifacts that
ship with this repository.  It contains eight logical models and nine artifacts;
weights stay in their existing `egs/` directories.  Every artifact has a
processor contract, sidecar (where required), and SHA256 digest.

All eight default ONNX artifacts have been re-exported from their source
checkpoints. The seven DPCRN streaming artifacts (six defaults and the NS v3
`flash` variant) use fixed batch=1 and frequency-major layout. Voice-isolation
DPCRN also uses a vectorized single-time LSTM update; NS DPCRN-Mamba additionally
ships four optional `.native.onnx` companions, one per NS artifact. Companion filenames and SHA256 values live in each sidecar's
`cpu_optimization` section and are checked by catalog validation. The two speaker
embedding exports retain dynamic waveform length and batch size.

NS v3 also ships a `flash` variant: the same weights with int8 LSTM and
matrix-product weights, for tighter CPU budgets. Select it with `--variant flash`
(or `variant="flash"` in `load_model`); the default variant is unchanged.

The default ONNX graphs use standard operators and run without a custom library.
For native CPU SSM execution, build locally and set `PURESOUND_ORT_SSM_LIBRARY`:

```bash
uv run python -m puresound.streaming.native.build --output /path/libpuresound_ssm.so
export PURESOUND_ORT_SSM_LIBRARY=/path/libpuresound_ssm.so
```

Reproduce all exports, state/audio parity checks, and streaming CPU measurements:

```bash
uv run python model_zoo/reexport.py --native-library /path/libpuresound_ssm.so
uv run python sdk/web/tools/build_assets.py
```

The [re-export report](benchmarks/onnx_reexport_20261004.json) records original and
new hashes, unchanged checkpoint hashes, per-model timing repeats, long streaming
checks, and dynamic-length embedding parity. The native library selects its
AVX2 or AVX-512 kernel at load time on x86-64 Linux; deployment hosts need a
compatible C library. Browser/WASM uses the standard primary graphs and the
generated web payload.

Validate the checkout on CPU with:

```bash
puresound models validate
```

The same catalog powers the public facade and the demos:

```python
from puresound.inference import ModelZoo, load_model

models = ModelZoo.default().list(task="voice_isolation")
runtime = load_model("voice-isolate-dpcrn-curriculum-v1", provider="auto")
result = runtime.infer({"audio": "input.wav"})
```

`target_speaker_extraction` remains a valid task value for future artifacts,
but is intentionally absent from the runnable catalog.

## License

The PureSound models published in this catalog, including their source checkpoints
and ONNX exports, are licensed under [Apache-2.0](../LICENSE). You may use, modify,
use commercially, and redistribute them under the license's terms. Include LICENSE
and [NOTICE](../NOTICE) when redistributing models, and mark modified files.
Third-party models and the datasets or recordings used for training retain their
own licensing terms.
