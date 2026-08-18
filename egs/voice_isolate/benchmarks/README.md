# voice_isolate — benchmarks

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Everything here is version-controlled and **contains no audio**. Definitions (span labels,
window files, build scripts) and test records (per-checkpoint scorecards) live here; the
audio payloads they describe live under the ignored `../data_report/` tree.

That line is the point of the split. Some of this material is public — LibriTTS, DNS-5,
BUT ReverbDB, VOiCES, RealMAN, ai-coustics Dawn Chorus — and rebuilding it from the
scripts here is expected. Some of it is internal field material that must not be
redistributed. Keeping the two in separate directories means the boundary is enforced by
`.gitignore` rather than by remembering, and no negative rule is ever needed under a
private path.

| benchmark | corpus | audio | what is tracked |
|---|---|---|---|
| [`field_test_vector/`](field_test_vector/) | **internal field recordings** | never | span labels, `windows.json`, `build_cases.py`, `RESULTS.md`, per-version scorecards |
| [`qvf22/`](qvf22/) | **internal**, carries a commercial system's output | never | `windows.json` |
| [`wer_sets/`](wer_sets/) | LibriTTS test-clean + DNS-5 + BUT ReverbDB (public) | rebuildable | how to rebuild each of the four reverberation regimes |
| [`dawn_chorus/`](dawn_chorus/) | `ai-coustics/dawn_chorus_en` (public) | external | per-checkpoint reports |
| [`probes/`](probes/) | VOiCES, RealMAN (CC-BY) | external | probe outputs backing the cross-chain findings |
| [`full_gate/`](full_gate/) | all of the above at once | — | one file per checkpoint: the complete nine-stage summary from `../run_full_benchmark.sh`, which is what a release decision is actually made on |

**No single benchmark decides a release.** `dpcrn_v10` tops the field benchmark's cold-start
column and fails the deployment gate; reading either one alone gets the decision wrong.
`full_gate/` exists so the whole picture stays together.

## Which numbers survive a synthesis change

Half of these stages read fixed audio off disk; half synthesise it at eval time
through `puresound`'s device chain. Only the second half moves when that chain
changes, and mixing the two across a chain boundary is how a data change gets
read as a model change.

| stage | audio comes from | chain-dependent |
|---|---|---|
| 1 real scorecard (QVF + field) | real recordings | no |
| **2 in-domain SI-SDRi + buckets** | `eval_indomain.py`, synthesised per batch | **yes** |
| **3-5 far-only probes** (expand / high / boundary) | same | **yes** |
| 6 Dawn Chorus WER | real corpus | no |
| 7a / 7b / 8 WER sets | pre-built by `build_wer_set.py`, own convolution | no |
| 9 turn-taking scorecard | frozen set on disk | no — but `build_turntaking_set.py` **is** chain-dependent, so rebuilding the set moves it |

**The boundary is `9c56e02` (2026-08-18)**, which made the device chain's
analogue path actually linear. Six stages were silently saturating at full
scale, and the fix changed 9.0% of mixtures and 2.8% of targets by more than 1%
of peak. Output level barely moved (RMS median −20.5 → −20.8 dBFS) and the RNG
stream did not move at all, so the same seed still draws the same utterances --
but rows 2-5 are a different test set either side of it.

`run_full_benchmark.sh` stamps `chain=<sha>` into the summary header. **Every
record committed before this section existed predates the fix**; treat a record
with no `chain=` in its header as pre-`9c56e02`, and do not rank its synthetic
rows against a post-fix run. The real-recording stages -- which is where the
deployment gate and the QVF cross-chain wall live -- are unaffected and remain
comparable all the way back.


## The field benchmark

`field_test_vector/` is the one that decides deployment questions: two hand-labelled real
recordings, 27 clips, scored two ways that are never averaged — STREAM (the continuous
recording, near anchor present, mid-conversation behaviour) and COLD-START (each span cut
out and fed from t=0, bot-idle behaviour). Standings and method:
[`field_test_vector/RESULTS.md`](field_test_vector/RESULTS.md).

Rebuild the clips (needs the private recordings in place under `../test_vector/`):

```bash
uv run python egs/voice_isolate/benchmarks/field_test_vector/build_cases.py
```

Score a checkpoint:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    --cases-dir egs/voice_isolate/data_report/field_cases/test_vector_cases \
    --device cuda:0 --dry-blend 0.9
```

## Adding a record

A record is the scorer's own output, unedited, named after the checkpoint it describes.
Drop it in that benchmark's `records/`. Records are cheap, they carry no audio, and they
are the only durable evidence in the repository — `EXPERIMENT_LOG.md` is deliberately not
version-controlled, so a number that exists only there does not survive a fresh clone.

## Adding a benchmark

Decide the corpus first, because it decides the layout: public corpora get a build script
and a note on how to rebuild, private material gets labels and records only and its audio
goes under `../data_report/` with an ignore rule. If a definition file would carry a
capture filename, device identifier or timestamp, rewrite that field before committing —
`field_test_vector/spans/` shows the shape after de-identification.
