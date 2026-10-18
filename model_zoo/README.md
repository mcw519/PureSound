# PureSound Model Zoo

`catalog.yaml` is the version-controlled registry for the ONNX artifacts that
ship with this repository.  It contains eight logical models and fourteen artifacts;
weights stay in their existing `egs/` directories.  Every artifact has a
processor contract, sidecar (where required), and SHA256 digest.

All eight default ONNX artifacts have been re-exported from their source
checkpoints. The twelve DPCRN streaming artifacts (six defaults and their six
`flash` variants) use fixed batch=1 and frequency-major layout. Voice-isolation
DPCRN also uses a vectorized single-time LSTM update; NS DPCRN-Mamba additionally
ships six optional `.native.onnx` companions, one per NS artifact. Companion filenames and SHA256 values live in each sidecar's
`cpu_optimization` section and are checked by catalog validation. The two speaker
embedding exports retain dynamic waveform length and batch size.

Every DPCRN model (voice isolation v8, curriculum-v1, curriculum-v2; noise
suppression v1, v2, v3) also ships a `flash` variant: the same weights with int8
LSTM and matrix-product weights, 11-21% less single-thread CPU time, quality
changes of a few thousandths and no resolvable change in words. Select it with
`--variant flash` (or `variant="flash"` in `load_model`); the default variant is
unchanged. Each model card has its paired measurements. Browser/WASM assets keep
the float graphs.

NS v3 also takes a `pcs` request parameter (perceptual contrast stretching of the
enhanced spectrum, off by default): a higher PESQ and sharper formants, paid for
in DNSMOS signal quality and inserted words on hard noise. Both sides are on the
[model card](../egs/noise_suppression/pretrained_ckpt/README.md), from the
records `ns_dpcrn-mamba_capWL3_mg_ep3_ort` (pcs off) and `..._ort_pcs` (on).

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
