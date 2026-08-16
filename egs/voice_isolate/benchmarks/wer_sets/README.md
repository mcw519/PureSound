# WER benchmarks — four reverberation regimes

Built entirely from public corpora, so the sets themselves are not tracked: 391 MB of
derived mix/ref pairs that `../../scripts/build_wer_set.py` regenerates. What matters is
that all four exist, because reverberation is the axis that separates the checkpoints and
a single set hides the split.

| set (under `../../data_report/`) | reverberation | role |
|---|---|---|
| `indomain_wer_set/` | the training bank | sanity: does the model help where it was trained |
| `wer_set_moderate_test/` | RT60 0.20–0.65, p50 0.44 | **the deployment domain** |
| `but_wer_set_office/` | reverberant office | the gate that separates v8 from v9 |
| `but_wer_set/` | RT60 1.15–1.84, p50 1.48 | extreme-reverb monitor, well past training |

Foreground is LibriTTS test-clean, so the reference transcript is ground truth rather than
another model's guess. Interferers are other LibriTTS speakers on the same room's far
channels; noise is DNS-5.

## Rebuild

```bash
uv run python egs/voice_isolate/scripts/build_wer_set.py \
    --libritts-dir /data/audio/LibriTTS/test-clean \
    --noise-dir /data/audio/dns-5/datasets_fullband_16k/noise_fullband \
    --rir-dir <PreGeneratedRoomBank folder> \
    --out-dir egs/voice_isolate/data_report/<set name>
```

The RIR folder is what distinguishes the four: a measured BUT ReverbDB bank for the two
`but_*` sets, the synthetic training bank for `indomain_wer_set`, and a held-out
moderate-RT60 bank for `wer_set_moderate_test`.

## Score

```bash
uv run python egs/voice_isolate/scripts/eval_wer.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt --dry-blend 0.9 \
    --set-dir egs/voice_isolate/data_report/but_wer_set_office
```

Drop the resulting report in `records/`.
