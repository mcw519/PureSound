# Gate stages: what each one is, and what it can decide

Traditional Chinese: [`stages.zh-TW.md`](stages.zh-TW.md)

Driver: [`../run_full_gate.sh`](../run_full_gate.sh). How to run it, its
variables and how collection works:
[`docs/usage/evaluation.md`](../../../docs/usage/evaluation.md). Record format:
[`README.md`](README.md).

**Gate** stages can fail a release. **Monitor** stages are reported and never fail
one -- either because the set cannot resolve the differences between our
checkpoints, or because the number is there to be comparable outward rather than
to decide inward. A stage's role is a property of what it can measure, not of how
much we care about it.

Every stage scores the **unprocessed mixture alongside the candidate** and reads
the paired difference. Running the driver with no checkpoint scores only that
baseline, which is how the gate is checked before there is a model to gate. The
candidate can also be a directory of audio another system produced
(`PRECOMPUTED_DIR=`), which is how a published model is placed on this axis
rather than only on its own; preflight and RTF are then skipped.

## The stages

| # | Stage | Set (n) | Metric | Role | Why |
|---|---|---|---|---|---|
| 0 | preflight | -- | parameters loaded | abort | A recipe and a checkpoint that disagree load partially and still produce numbers |
| 1 | frozen synthetic set | `data_report/ns_testset` (500), `NS_TESTSET=` | PESQ-WB; STOI, ESTOI, SI-SDR, harmonic gap, transient correlation | **gate** on PESQ-WB, the rest monitor | Our own synthesis, held-out speakers, reaching the SNRs deployment cares about |
| 2 | VCTK-DEMAND | `data_report/vctk_demand_test` (824), `NS_VCTK=` | as stage 1 | **gate** on PESQ-WB, the rest monitor | Real recorded noise, fixed on disk, and the set most of the literature reports on |
| 3 | DNS-5 dev set | `data/dns5/dns5_devset.jsonl` (921), `NS_DEVSET=` | DNSMOS SIG / BAK / OVRL / P.808, by category | monitor | No clean reference exists; a passthrough scores well by doing nothing |
| 4 | WER | VCTK-DEMAND cuts (824), `NS_WERSET=` | deletions; WER, insertions, substitutions | **gate** on deletions, the rest monitor | The failure the quality metrics cannot see: removing words |
| 4b | WER, hard set | `data_report/libritts_demand_hard` (500), `NS_WERSET_HARD=`; `SKIP_WER_HARD=1` skips it | as stage 4, banded by SNR and noise type, hypotheses kept | **gate** on deletions, the rest monitor | Enough words, long enough utterances and low enough SNR to resolve what the VCTK cuts cannot |
| 5 | CPU real-time factor | synthetic | RTF, one thread | **gate** | Real-time is a requirement, not a trade-off |

`SKIP_WER=1` skips stages 4 and 4b; stage 4b is also skipped when its set is not
built. The record then carries no stage for them and says so.

### 1. Frozen synthetic set -- gate

Built by `puresound.evaluation.tools.build_eval_set` from the recipe's own
synthesis with a fixed seed, so the audio is pinned and two checkpoints scored
weeks apart are comparable. The speakers are the metafile's valid split, which
is speaker-disjoint from training.

```bash
python -m puresound.evaluation.tools.build_eval_set \
    egs/noise_suppression/config/eval/ns_testset.yaml \
    --out-dir egs/noise_suppression/data_report/ns_testset --n 500 --seed 1234
```

Each item's SNR is **measured** from `noisy - clean`, not taken from the config
that asked for it, so the per-SNR-band breakdown means something. Read that
breakdown: a system can gain PESQ at high SNR while destroying speech near 0 dB,
and the mean over both reports neither.

This stage is synthesised, so its numbers only compare when the build commit and
recipe digest in `extra.set_provenance` match.

### 2. VCTK-DEMAND -- gate

Real recorded noise mixed with VCTK speech, already paired on disk, imported to
16 kHz by `puresound.dataset.corpus.vctk_demand testset`. It gives two things the
synthetic set cannot.

It is **externally comparable**: PESQ on VCTK-DEMAND is the number most
noise-suppression papers report, so a result here can be read against published
work rather than only against our own previous runs.

It **does not move when our synthesis does**. Stage 1's audio is produced by this
repository's device chain, so its numbers only compare within one build of the
set; this set is fixed audio on disk and compares across every change we make.

Its SNR distribution is milder than stage 1's, which is a feature: the two stages
sample different halves of the problem, and a model that only works in one of
them fails one of them.

### 3. DNS-5 dev set -- monitor

Real clips with no clean reference, scored with DNSMOS P.835. It is a monitor for
two reasons: a distortion-free passthrough scores well by doing nothing, and a
composite quality predictor can miss a real defect that a listener or a
recogniser catches.

SIG and BAK are reported separately and OVRL is never read alone -- OVRL cannot
tell "removed the noise" from "removed the speech", and those are this task's two
failure directions.

The breakdown by noise category is the point of this stage. Noise suppression
fails per class: it can hold up on a fan and fall over on a crying baby. The
category comes from the dev set's own filenames.

### 4. WER -- gate on deletions

The guardrail the quality metrics are worst at covering. A model that removes the
noise and some of the speech with it can hold its PESQ and lose the sentence.

Three rules keep the number honest:

**Read the delta, not the rate.** A WER means nothing until the unprocessed
mixture's WER on the same cuts is known. Both are transcribed and the paired
difference is what the verdict reads.

**Deletions are reported separately**, because they are the failure mode with a
direction. Substitutions and insertions can come from the recogniser; a rising
deletion rate against a fixed reference is the model removing speech.

**The reference is the corpus transcript**, never the recogniser run on clean
audio -- that would measure the recogniser's agreement with itself and flatter
whichever system sounds most like its training data.

The recogniser defaults to a large model on purpose: a weak recogniser can hide
exactly the over-suppression a strong one reveals, and the point of this stage is
to expose it, not to be quick.

**The recogniser has to be a function of the audio.** faster-whisper's default
decoding is a temperature fallback ladder: a segment failing its quality checks
is re-decoded with sampling, so identical audio transcribes differently run to
run, and a paired gate cannot read a small difference through that.
`evaluation.tools.wer` pins `--asr-temperature` (default 0.0), which makes two
runs on the same audio identical.

**What the VCTK cuts can and cannot resolve.** The utterances are short, the SNRs
mild and the reference small, so the unprocessed mixture already sits near the
recogniser's floor, and the whole range over which our versions differ is only a
small multiple of the recogniser's own run-to-run spread. On this set:

- `wer.wer` and `wer.sub` are **monitors** and must not rank two versions. A
  difference between two checkpoints here can be statistically resolvable and
  still amount to a couple of misheard words per thousand -- resolvable is not
  the same as important.
- `wer.del` stays a **gate**. Deletion is the failure the quality metrics cannot
  see, and a model removing speech pushes it off its floor unmistakably. Read it
  as a guardrail question -- "is this version eating words?" -- and nothing else.
- **Ranking belongs to the stages with headroom.** An oracle mask on the same
  STFT sits far above every model on PESQ, STOI and DNSMOS, while WER here is
  already near its floor. A version that needs an ASR number that can rank needs
  harder audio, which is stage 4b.

`--band` (default `snr_band`) breaks the per-utterance rates down the same way
stage 1 does; what usable range this set has is in its lowest-SNR band.

### 4b. WER on the hard set -- gate on deletions

The harder audio stage 4 asks for, with the same roles: deletions gate, the rest
monitor.

```bash
python -m puresound.dataset.corpus.vctk_demand noise /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data/demand_noise
python -m puresound.evaluation.tools.mix_paired_set \
    --speech-dir /path/to/audio/LibriTTS/test-clean --transcript-suffix .normalized.txt \
    --noise-dir egs/noise_suppression/data/demand_noise \
    --snr -10 5 --min-duration 8 --n 500 --seed 1234 \
    --out-dir egs/noise_suppression/data_report/libritts_demand_hard
```

Four properties make it resolve what the VCTK cuts cannot:

- **Several times as many reference words**, so a fixed number of errors is a
  smaller share of noise.
- **Long utterances** (at least 8 s), so one deletion is not a tenth of the
  sentence.
- **SNR from -10 to 5 dB**, most of it below 0 dB, where VCTK-DEMAND barely goes.
- **Noise no model here has heard**: the DEMAND noise VCTK-DEMAND mixed in,
  recovered as `noisy - clean` -- a pool held out from every DNS-trained model.

There is deliberately **no reverb**: the difficulty is the SNR axis alone, so a
measured band is that band and not that band plus a room.

Resolution has limits here too. The set separates a system that costs words from
one that does not, and separates gaps the size of ours against a published model;
a gap the size of two of our own neighbouring checkpoints can still sit inside
the interval. Check the paired interval, and the sign count (items worse versus
better), before ranking two of our own checkpoints on this stage.

**Trim the loops before reading a delta.** Whisper answers a segment it cannot
parse by repeating a phrase, and one such hypothesis carries an insertion rate far
above the corpus mean. The stage prints a loop count per system (a hypothesis over
1.5 times its reference) for that reason, and the gate keeps every hypothesis in
`4b_wer_hard_hyp.jsonl`. Loop counts differ between systems, so an untrimmed
spread partly ranks which output breaks the recogniser.

Keep `--band noise` on: a corpus mean hides whether the damage sits in the noise
types that overlap speech spectrally.

The two WER stages do not replace each other. The VCTK stage is externally
comparable and has little resolution; this one has resolution and no external
comparability.

### 5. CPU real-time factor -- gate

Measured on CPU, single-threaded, on synthetic audio. Not on a GPU, and not with
every core of a many-core machine: neither says anything about whether this
ships. Measure on an idle machine; a load average near the core count distorts
the ranking, not only the number.

The default budget (`RTF_BUDGET`, 0.5) leaves headroom for everything else on
the device. Raise it only with a stated reason, in the record.

## Reading a result

1. **A `no-resolution` gate stage is not a pass.** The set could not tell the
   candidate from doing nothing. The record keeps `no-resolution`, and collection
   exits non-zero because it cannot support a release.
2. **Read the absolute value next to the delta.** A large improvement on a
   terrible starting point is not a good end state.
3. **One checkpoint is not a measurement.** Score a block of the run's last
   checkpoints before believing a difference between versions.
4. **Check `extra.set_provenance` before comparing stage 1**, and the scoring
   `chain_commit` before comparing runtime measurements.
