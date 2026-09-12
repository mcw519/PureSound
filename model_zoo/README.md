# PureSound Model Zoo

`catalog.yaml` is the version-controlled registry for the ONNX artifacts that
ship with this repository.  It contains four logical models and four artifacts;
weights stay in their existing `egs/` directories.  Every artifact has a
processor contract, sidecar (where required), and SHA256 digest.

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

`noise_suppression` and `target_speaker_extraction` remain valid task values
for future artifacts, but are intentionally absent from the runnable catalog.
