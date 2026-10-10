# Evaluation and the gate

Traditional Chinese: [evaluation.zh-TW.md](evaluation.zh-TW.md)

`puresound.evaluation` holds the benchmark protocol, its statistics and its
records. A recipe's gate script is a list of these tools; the protocol lives in
the library so two recipes cannot drift into two protocols. This page covers the
noise-suppression gate, `egs/noise_suppression/run_full_gate.sh`; what each of
its stages can and cannot decide is argued in
[`egs/noise_suppression/benchmarks/stages.md`](../../egs/noise_suppression/benchmarks/stages.md).
The voice-isolation recipe drives its own stage list,
`egs/voice_isolate/run_full_benchmark.sh`, through the same tools; its
field-recording stages use private recordings that are not distributed with the
repository.

| Module | Description |
|---|---|
| `evaluation.statistics` | bootstrap intervals, paired comparison, block summaries, verdicts |
| `evaluation.records` | `StageResult` / `GateRecord`, the record a benchmark directory keeps |
| `evaluation.systems` | what a stage runs audio through: a checkpoint, the do-nothing baseline (`Passthrough`), or a directory of audio another system already produced (`PrecomputedSystem`) |
| `evaluation.transcribers` | local Whisper, Azure Speech and ElevenLabs behind one call (the web playground's word check) |
| `evaluation.tools.preflight` | prove a checkpoint loads whole into every recipe a gate uses |
| `evaluation.tools.build_eval_set` | freeze a synthetic (mix, clean) set from the recipe's own synthesis |
| `evaluation.tools.mix_paired_set` | mix a transcript-bearing speech folder with a held-out noise pool into a WER set |
| `evaluation.tools.reference` | PESQ, STOI, ESTOI, SI-SDR and two spectral-detail measures on a frozen set, paired and banded |
| `evaluation.tools.noreference` | DNSMOS P.835 where no clean reference exists, broken down by tag |
| `evaluation.tools.wer` | word error rate against the corpus transcript, split into deletions, insertions and substitutions |
| `evaluation.tools.rtf` | CPU real-time factor against a budget |
| `evaluation.tools.collect` | merge stage files into one record and return the verdict |

Every tool runs as `python -m puresound.evaluation.tools.<name>`; `--help` lists
its flags.

## Build the sets once

The gate's sets are built once and then frozen. They live under `data_report/`
(not version-controlled); the gate reads them from the default paths below, and
each path has an environment override.

```bash
# stage 1: a synthetic set from the recipe's own synthesis           (NS_TESTSET)
python -m puresound.evaluation.tools.build_eval_set \
    egs/noise_suppression/config/eval/ns_testset.yaml \
    --out-dir egs/noise_suppression/data_report/ns_testset --n 500 --seed 1234

# stages 2 and 4: VCTK-DEMAND, imported with its transcripts          (NS_VCTK)
python -m puresound.dataset.corpus.vctk_demand testset /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data_report/vctk_demand_test

# stage 3: the DNS-5 dev-set inventory                                 (NS_DEVSET)
python -m puresound.dataset.corpus.dns_challenge devset /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5

# stage 4b: the hard WER set -- held-out DEMAND noise under LibriTTS   (NS_WERSET_HARD)
python -m puresound.dataset.corpus.vctk_demand noise /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data/demand_noise
python -m puresound.evaluation.tools.mix_paired_set \
    --speech-dir /path/to/audio/LibriTTS/test-clean --transcript-suffix .normalized.txt \
    --noise-dir egs/noise_suppression/data/demand_noise \
    --snr -10 5 --min-duration 8 --n 500 --seed 1234 \
    --out-dir egs/noise_suppression/data_report/libritts_demand_hard
```

`build_eval_set` measures each item's SNR from `noisy - clean` and records the
build commit, recipe SHA-256 and seed in the set's provenance; `--min-active`
drops an item whose clean target is essentially silent. `mix_paired_set` takes
its SNR window (`--snr LOW HIGH`), minimum and maximum duration, and the noise
pool as one track per file.

## Run the gate

Run from the repository root.

```bash
# the do-nothing reference every stage is read against -- no checkpoint
bash egs/noise_suppression/run_full_gate.sh baseline

# a checkpoint, with the recipe it was built from
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> [recipe]

# audio another system already produced, one file per input named by its stem
PRECOMPUTED_DIR=/path/to/enhanced bash egs/noise_suppression/run_full_gate.sh <tag>
```

`<tag>` names the record: it is written to
`egs/noise_suppression/benchmarks/records/<tag>.json`. `[recipe]` defaults to
`egs/noise_suppression/config/dpcrn.yaml`; pass the recipe the checkpoint was
trained from, or the model-only `config/infer_dpcrn.yaml` for a released
checkpoint. A checkpoint and `PRECOMPUTED_DIR` are two systems; the script
refuses both at once.

| Variable | Default | Effect |
|---|---|---|
| `DEVICE` | `cpu` | device the model runs on in stages 1, 2, 4 and 4b |
| `DRY_BLEND` | `1.0` | inference operating point, passed as `--dry-blend` and recorded as `inference.dry_blend` |
| `SKIP_WER` | `0` | `1` skips stages 4 and 4b |
| `SKIP_WER_HARD` | `0` | `1` skips stage 4b |
| `NS_DATA` | `egs/noise_suppression/data/dns5` | where the dev-set inventory is looked for |
| `NS_TESTSET` | `egs/noise_suppression/data_report/ns_testset` | stage 1 set |
| `NS_VCTK` | `egs/noise_suppression/data_report/vctk_demand_test` | stage 2 set |
| `NS_WERSET` | `$NS_VCTK` | stage 4 set |
| `NS_WERSET_HARD` | `egs/noise_suppression/data_report/libritts_demand_hard` | stage 4b set |
| `NS_DEVSET` | `$NS_DATA/dns5_devset.jsonl` | stage 3 inventory |
| `RTF_BUDGET` | `0.5` | stage 5 budget |
| `GATE_JOBS` | half the cores | scoring worker processes for stages 1-3 |
| `ASR_MODEL` / `ASR_DEVICE` | `large-v3` / `cuda` | recogniser for stages 4 and 4b |
| `GATE_OUT` | `$TMPDIR` or `/tmp` | parent of the per-run log directory |
| `PRECOMPUTED_DIR` / `PRECOMPUTED_NAME` | unset / the directory name | score precomputed audio instead of a checkpoint |

## The stages

| # | Stage | Set | Metric | Role | Stage name in the record |
|---|---|---|---|---|---|
| 0 | preflight | every recipe the gate uses | parameters loaded | abort | -- |
| 1 | frozen synthetic set | `NS_TESTSET` | PESQ-WB; STOI, ESTOI, SI-SDR, harmonic gap, transient correlation; by SNR band | **gate** on PESQ-WB, the rest monitor | `frozen_testset` |
| 2 | VCTK-DEMAND | `NS_VCTK` | as stage 1 | **gate** on PESQ-WB, the rest monitor | `vctk_demand` |
| 3 | DNS-5 dev set | `NS_DEVSET` | DNSMOS SIG, BAK, OVRL, P.808; by category and device | monitor | `dns_devset` |
| 4 | WER | `NS_WERSET` | deletions; WER, insertions, substitutions | **gate** on deletions; the rest monitor | `wer` |
| 4b | WER, hard set | `NS_WERSET_HARD` | as stage 4, banded by SNR and noise type, hypotheses kept | **gate** on deletions; the rest monitor | `wer_hard` |
| 5 | CPU real-time factor | synthetic audio | RTF, one thread | **gate** against `RTF_BUDGET` | `cpu_rtf` |

- **Stage 0** runs only with a checkpoint. If the checkpoint does not load whole
  into the recipe, the gate stops before stage 1: a partly untrained model would
  report a regression nobody can explain.
- **Stage 3** runs the model on CPU whatever `DEVICE` says -- the dev-set clips
  are long enough that a GPU forward with several workers runs out of memory, and
  the model is a small share of this stage next to DNSMOS's own sessions.
- **Stage 4** transcribes both the mixture and the output with the recogniser
  and scores them against the corpus transcript. `wer.del` is the gate; `wer.wer`,
  `wer.ins` and `wer.sub` are reported as monitors and never fail a release.
- **Stage 4b** is skipped when `SKIP_WER=1`, when `SKIP_WER_HARD=1`, or when
  `$NS_WERSET_HARD/manifest.jsonl` does not exist; the log says which. When it
  runs it adds `--band snr_band --band noise` and writes every hypothesis to
  `4b_wer_hard_hyp.jsonl` in the log directory.
- **Stage 5** is skipped for `PRECOMPUTED_DIR` -- there is no model to time.

Collection then merges the stage files into one record and requires the stages
the run was supposed to produce: `cpu_rtf` (unless precomputed), and with a
checkpoint or precomputed audio also `frozen_testset.pesq_wb`,
`vctk_demand.pesq_wb`, `dns_devset.dnsmos_ovr`, `wer.del` (unless `SKIP_WER=1`)
and `wer_hard.del` (when stage 4b ran). A missing required stage fails
collection. The script exits non-zero when any stage command failed or when the
record's verdict is not `pass`, and prints where the logs are.

## Every stage scores two systems

The candidate, and **doing nothing**. An absolute score says nothing on its own:
a set where the unprocessed mixture already scores well has no headroom to win,
and a large improvement on a terrible starting point is not a good end state. The
verdict is read from the paired difference against `Passthrough`.

It also means the whole benchmark runs before a model exists, which is why a gate
can be built first: `run_full_gate.sh baseline` prints every stage's unprocessed
score.

A third system can be put on the same axis: run any other model offline over a
set's input files, then pass `--precomputed DIR` to `reference`, `noreference`
and `wer` (or `PRECOMPUTED_DIR=DIR` to the gate). Files are matched by the input
file's stem; a tool's own suffix on the output name is accepted when it is
unambiguous, and two candidates are an error rather than a guess. This is how a
published model is compared against ours on *our* sets rather than only on its
own.

The gate scores a checkpoint at an **inference operating point** and records it
in the `inference` field, so a number is never read without the knobs it was
measured at. `reference`, `noreference`, `wer` and `rtf` take the same flag,
`--dry-blend` (see [`docs/architecture/system/siso.md`](../architecture/system/siso.md));
the gate script exposes it as `DRY_BLEND=`. `SKIP_WER=1` is for iterating on the
quality stages, not for deciding a release: the record then carries no WER stage,
and WER is the guardrail the quality stages cannot supply.

## Three ways of claiming more than you measured

Each has a guard here.

**A set that cannot resolve the difference.** `verdict()` returns
`"no-resolution"` when the interval covers zero -- or has no spread to read: one
item, or a non-finite end -- and `GateRecord.unresolved` lists the gate stages
that did. The aggregate record keeps that verdict and `collect` returns non-zero:
it is not a regression, but it cannot support a release either.

```python
from puresound.evaluation.statistics import paired_bootstrap_ci, verdict

delta = paired_bootstrap_ci(enhanced_scores, unprocessed_scores)
decision = verdict(delta, direction="higher_is_better")   # pass / fail / no-resolution
```

`verdict(..., tolerance=x)` lets a do-no-harm stage absorb a regression whose
whole interval stays within `x`; it never turns an unresolved interval into a
pass. The WER tool exposes it as `--tolerance`, applied to deletions.

**One checkpoint read as a measurement.** Neighbouring epochs of the same run can
move a metric by more than the version differences being compared.
`block_summary()` scores a run as a block of its last checkpoints and reports the
range, not a standard deviation -- five points do not support a standard
deviation.

**Unpaired comparison.** The same items go through both systems, so the paired
difference has a fraction of the spread of either side. Resampling the two sides
independently reports the metric's spread instead, which is how a real effect
gets called noise.

## A record has to be comparable to another record

`GateRecord` requires the fields whose absence has silently invalidated
comparisons: the `recipe` (a recipe and a checkpoint that disagree load partially
and still produce numbers), the scoring `chain_commit`, the `inference` operating
point, and each stage's `n` and interval. A frozen synthetic stage additionally
carries the set's build commit, recipe SHA-256 and seed in
`extra.set_provenance`; those values, not the later scoring commit, identify the
synthesis that produced its audio. The record contract, including the `lineage`
block filed with each record, is in
[`egs/noise_suppression/benchmarks/README.md`](../../egs/noise_suppression/benchmarks/README.md).

`load_checkpoint_system` enforces the first of those at load time: a missing
parameter -- which is also how a renamed one shows up -- is an error, not a
warning. A checkpoint that carries *more* than the recipe builds is reported and
recorded instead, because the audio path is still whole.

## Roles: gate versus monitor

A **gate** can fail a release. A **monitor** is reported and never does -- either
because the set cannot resolve the differences between our checkpoints, or
because the number exists to be comparable outward rather than to decide inward.
The role is a property of what the stage can measure, not of how much anyone
cares about it.

The WER stages show the split inside one stage. Deletions are the failure with a
direction -- a model removing speech pushes them up against a fixed reference --
and the one the quality metrics cannot see, so `del` is the gate. WER,
insertions and substitutions move with the recogniser as much as with the model,
and a difference between two checkpoints can be statistically resolvable while
amounting to a few words per thousand, so they are monitors: read them, never
rank or veto on them. Rank versions on the quality stages, which have headroom.

`collect` exits non-zero when a gate stage failed or could not resolve, so a
driver script fails with it.
