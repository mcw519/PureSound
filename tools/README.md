# Repository tools

These are standalone development and evaluation tools. They are kept in the
repository because they support reproducible maintenance, but they are not
imported by `puresound` and are not part of the runtime package.

| Tool | Use |
|---|---|
| [`rng_fingerprint.py`](rng_fingerprint.py) | Check that a synthesis refactor preserves seeded outputs |
| [`build_web_samples.py`](build_web_samples.py) | Rebuild the web playground's sample clips from `test/test_case/` |
| [`build_pipeline_samples.py`](build_pipeline_samples.py) | Rebuild the pipeline inspector's sample noise and rooms (`--bank` copies rooms from a synthetic RIR bank once) |

## Audio annotator

The span annotator is now the **Annotate** screen of `puresound web`
(`#/annotate`): mark keep / suppress ranges on a waveform or spectrogram and
export JSON, CSV, `windows.json` or Audacity labels. The formats are the same
byte for byte as the standalone `audio_annotator.html` that used to live here.
See [the web usage guide](../docs/usage/web.md#annotate).

## RNG fingerprint

`rng_fingerprint.py` compares dataset synthesis before and after a refactor.
It is intended for changes that must preserve generated samples exactly, such
as moving augmentation code, changing defaults, or splitting a processing
stage.

Unit tests verify that the current implementation is deterministic. A
before/after fingerprint adds a different guarantee: the refactor did not move
all deterministic outputs to a new result.

### Capture and compare

Capture a baseline from the original tree:

```bash
uv run python tools/rng_fingerprint.py path/to/config.yaml before.json
```

Apply the refactor and capture the same recipe again:

```bash
uv run python tools/rng_fingerprint.py path/to/config.yaml after.json
```

Compare the files:

```bash
uv run python tools/rng_fingerprint.py --compare before.json after.json
```

The command exits with status 0 when all recorded hashes match and status 1
when any output changed.

Use `--n-items` to change the default sample count of 32:

```bash
uv run python tools/rng_fingerprint.py \
  path/to/config.yaml fingerprint.json \
  --n-items 64
```

### Scope and limitations

- Use a real recipe that enables every block whose behavior must be preserved.
- The recipe's train metafile and referenced audio must be available.
- A fingerprint only covers the sampled items and enabled pipeline.
- Store temporary before/after JSON outside version control.
- An intentional behavior change should update tests or release notes instead
  of forcing the old fingerprint to pass.
