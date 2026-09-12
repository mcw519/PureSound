# WER benchmarks — four reverberation regimes

Built entirely from public corpora, so the sets themselves are not tracked: 391 MB of
derived mix/ref pairs that `build_wer_set.py` regenerates. What matters is
that all four exist, because reverberation is the axis that separates the checkpoints and
a single set hides the split.

| set (under `../data_report/`) | RIRs | reverberation | role |
|---|---|---|---|
| `indomain_wer_set/` | synthetic, training bank | in-domain | sanity: does the model help where it was trained |
| `wer_set_moderate_test/` | synthetic, held-out rooms | RT60 0.20–0.65, p50 0.44 | **the primary WER gate** |
| `but_wer_set_office/` | **measured** (BUT ReverbDB) | RT30 0.56–0.69 | measured-RIR monitor |
| `but_wer_set/` | **measured** | RT60 1.15–1.84, p50 1.48 | extreme-reverb monitor, well past training |

## Read the interval, not the point estimate

`eval_wer.py` reports a 95% bootstrap interval on the enh-vs-mix difference and the share
of the set's headroom the checkpoint captured. Both matter, because these sets differ
enormously in how much they can actually resolve (2026-08-16, v8 and v10 at `dry_blend 0.9`,
n=200 each, paired on identical utterances):

| set | v8 enh−mix | v10 enh−mix | headroom captured | v10 − v8, paired |
|---|---|---|---|---|
| `wer_set_moderate_test` | **−0.171** [−0.212, −0.131] | **−0.147** [−0.187, −0.109] | 43% / 37% | **+0.024** [+0.008, +0.041] |
| `but_wer_set_office` | −0.022 [−0.049, **+0.002**] | −0.006 [−0.033, +0.018] | 6% / 2% | +0.016 [−0.001, +0.035] |

On BUT-OFFICE **neither checkpoint is distinguishable from doing nothing**, so the version
ordering this repository quoted from that set for months (v8 −0.024, v9 −0.050, v10 −0.004)
sat inside its noise band. It is a monitor now, not a gate. The moderate set resolves the
same comparison cleanly and gives the same sign, which is why the v10 verdict survived the
correction.

**Known confound.** Our sets vary reverberation level and measured-vs-synthetic RIRs
together, so we cannot yet say whether BUT-OFFICE collapses because RT is higher or because
its impulse responses are measured. The missing cell is a measured-RIR set at RT60 0.2–0.45
(or a synthetic one at 0.56–0.69); build it before concluding anything about measured
acoustics from these numbers.

Foreground is LibriTTS test-clean, so the reference transcript is ground truth rather than
another model's guess. Interferers are other LibriTTS speakers on the same room's far
channels; noise is DNS-5.

## Rebuild

```bash
uv run python egs/voice_isolate/scripts/build_wer_set.py \
    --libritts-dir /path/to/audio/LibriTTS/test-clean \
    --noise-dir /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband \
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
