# PureSound

[License: Apache-2.0](LICENSE)

Traditional Chinese: [README.zh-TW.md](README.zh-TW.md)

PureSound is a PyTorch toolkit for single-channel speech enhancement: training
recipes with on-the-fly data synthesis, a benchmark gate, streaming ONNX export,
and a model zoo you can run from the command line, a local web UI or a small
standalone Python runtime.

## Released models

Listed in [`model_zoo/catalog.yaml`](model_zoo/catalog.yaml). Every model takes
16 kHz mono audio; input is converted when needed.

| task | model id | role | use it for |
| --- | --- | --- | --- |
| Voice isolation | `voice-isolate-dpcrn-curriculum-v1` | default | keeps the talker near the microphone and suppresses distant talkers and noise; strongest on the capture hardware it was tuned for |
| Voice isolation | `voice-isolate-dpcrn-curriculum-v2` | candidate | a wider curriculum model: lower WER and deeper distant-talker suppression than v1, at 1.4x the CPU |
| Voice isolation | `voice-isolate-dpcrn-v8` | candidate | the conservative choice when the capture hardware is unknown |
| Noise suppression | `noise-suppression-dpcrn-mamba-v3` | candidate | wider streaming model with higher PESQ and STOI; word-deletion gates remain unresolved |
| Noise suppression | `noise-suppression-dpcrn-mamba-v2` | default | streaming noise suppression, balanced quality and CPU cost |
| Noise suppression | `noise-suppression-dpcrn-mamba-v1` | candidate | the same model without its final quality fine-tune; deletes fewer words on clean read speech |
| Speaker verification | `speaker-verification-ps-spk-v1-1` | default | 192-dimensional speaker embeddings |
| Speaker verification | `speaker-verification-ps-spk-v1` | alternative | the earlier embedding release |

What each version was judged on is in the recipe's checkpoint notes:
[voice isolation](egs/voice_isolate/pretrained_ckpt/README.md),
[noise suppression](egs/noise_suppression/pretrained_ckpt/README.md),
[speaker embedding](egs/speaker_embedding/README.md).

## Install

Python 3.12 or newer; [uv](https://docs.astral.sh/uv/) is the recommended setup.
Choose one ONNX Runtime backend:

```bash
git clone <project-url>
cd PureSound

# CPU ONNX Runtime (CoreML on macOS)
uv sync --locked --group dev --extra cpu

# NVIDIA CUDA ONNX Runtime
uv sync --locked --group dev --extra cuda
```

The `cpu` and `cuda` extras cannot be installed together, and the `asr` extra
(local Whisper for the WER stage) cannot be combined with `cuda`.

With pip instead:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Quick start

List the released models and the ONNX Runtime providers available here:

```bash
uv run puresound models list
uv run puresound providers
```

Run a model on a file:

```bash
uv run puresound infer voice-isolate-dpcrn-curriculum-v1 \
  --input audio=input.wav --output audio=output.wav --provider auto

uv run puresound infer speaker-verification-ps-spk-v1-1 \
  --input enrollment=enrollment.wav --input test=test.wav
```

The speaker-verification call prints the cosine similarity and the verdict.

Start the local web UI at <http://127.0.0.1:7860>:

```bash
uv run puresound web
```

It has no authentication; do not expose it to an untrusted network.

To embed a released model in another project without installing PureSound, use
the standalone runtime in [`sdk/python`](sdk/python/README.md): it needs only
NumPy, ONNX Runtime, the ONNX file and its JSON manifest.

## Train a model

Each recipe under `egs/` is run from its own directory, because a recipe's data
and output paths are relative to it. Point the metafile and folder paths in the
recipe at your data first.

```bash
cd egs/voice_isolate
uv run python main.py config/train_dpcrn.yaml --training
```

| task | recipe |
| --- | --- |
| Noise suppression | [egs/noise_suppression](egs/noise_suppression/README.md) |
| Voice isolation | [egs/voice_isolate](egs/voice_isolate/README.md) |
| Speaker embedding | [egs/speaker_embedding](egs/speaker_embedding/README.md) |
| Target speaker extraction (frozen legacy) | [egs/target_speaker_extraction](egs/target_speaker_extraction/README.md) |
| RIR bank generation | [egs/rir_generation](egs/rir_generation/README.md) |

## Documentation

[docs/index.md](docs/index.md) is the entry point, in three parts:

- [Architecture](docs/architecture/index.md) — how the packages fit together and
  how data flows from corpus to deployment
- [Algorithms](docs/index.md#algorithms) — what each model, loss, synthesis stage
  and metric computes
- [Usage](docs/index.md#usage) — preparing data, training, evaluating, exporting
  and deploying

Where a new file belongs: [docs/repository_layout.md](docs/repository_layout.md).

## Development

```bash
uv run python test/run_repo_checks.py --suite standard   # everyday
uv run python test/run_repo_checks.py                    # full suite, before a release
./build_puresound.sh                                     # build the package
```

```text
puresound/   the library
egs/         recipe drivers, configs, released checkpoints
docs/        documentation
model_zoo/   released-model catalog
sdk/python/  standalone streaming runtime
test/        tests and public fixtures
```

## License

Original PureSound code, SDKs, configuration files, documentation, and models
trained by Milo Wu or PureSound contributors are licensed under [Apache-2.0](LICENSE),
including the model zoo's checkpoints and ONNX exports. Others may use, modify,
use commercially, and redistribute these models under the license's terms.
See [NOTICE](NOTICE) for the scope and exclusions. Third-party material, including
third-party models, retains its own license. Datasets and audio recordings are
excluded unless explicitly licensed otherwise; check their individual license and
provenance information before reuse.
