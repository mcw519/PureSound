# Preparing data

Traditional Chinese: [data_preparation.zh-TW.md](data_preparation.zh-TW.md)

Corpus preparation lives in `puresound.dataset.corpus`: scan audio into records,
convert it to one sample rate, split it without leakage, and write the metafile a
recipe reads. The tools are corpus-agnostic -- what differs between corpora is a
parameter or a small corpus module, never a copy of this code under `egs/`.

Every module with a command line runs as
`python -m puresound.dataset.corpus.<module>` from the repository root, and
`--help` prints its complete flag set. The commands below show the flags that
matter; paths are examples.

## Modules

| Module | What it does | Command line |
|---|---|---|
| `records` | `AudioRecord`, the seven-column metafile, the JSONL inventory | library |
| `scan` | scan a folder into records; speaker-disjoint splitting | library |
| `resample` | mirror a corpus into a fixed-sample-rate tree, once | `--resample-*` flags on the corpus commands |
| `dns_challenge` | DNS Challenge clean speech, noise, and the dev set | `speech`, `noise`, `devset` |
| `vctk_demand` | VCTK-DEMAND test set, training speech, and its recovered noise | `testset`, `speech`, `noise` |
| `librilight` | LibriLight chapters: pick the cleanest, cut them into pieces | `select`, `segment` |
| `kaldi` | Kaldi `wav.scp` / `utt2spk` lists to a metafile | yes |
| `paired` | write an on-disk (noisy, clean) benchmark in the scored-set format | library (behind `vctk_demand testset`) |
| `concat_short` | join a speaker's short utterances into pieces long enough to train on | yes |
| `clean_speech` | run a speech corpus through a checkpoint into a separate tree | yes |
| `target_policy` | choose, file by file, the cleaned or the original target | yes |
| `noise_corpora` | FSD50K, CochlScene and MUSAN to 16 kHz noise folders, filtered for licence and voices | yes |
| `speech_screen` | find intelligible speech in a noise folder and move it out | yes |
| `ssn` | synthesise speech-shaped noise from training speech | yes |
| `pool` | merge several metafiles into one training pool and report the batch mix | `merge` |

## Two files, two jobs

The metafile is seven columns and its shape is fixed -- it is what
`DynamicBaseDataset` parses:

```
uttid, spkid, gender, path, length, sample rate, channels
```

Anything else a corpus knows (a noise category, a capture device, a licence) goes
in `AudioRecord.tags` and is written to a **JSONL inventory** beside it. Keeping
them apart is what lets a corpus carry arbitrary metadata without every parser
growing a column for it.

```python
from puresound.dataset.corpus import scan_folder, split_records, write_metafile

records = scan_folder("/corpus/read_speech", id_prefix="dns5",
                      speaker_pattern=r"reader_(\d+)")
train, valid = split_records(records, valid_ratio=0.05, split_by="speaker")
write_metafile("data/train.csv", train)
```

## Splitting: the leak this prevents

`split_by="speaker"` is the default because it is the only mode whose two sides
are disjoint in the way a dev set has to be. With `"utterance"`, every dev speaker
is also a train speaker, and the dev loss reports how well the model memorised
voices it was trained on. Call `assert_disjoint(train, valid)` -- the split is
checked, not assumed.

Two failure modes are errors rather than surprises:

- A directory strategy (`parent`, `grandparent`, `path-prefix`) on a tree that
  has no such directory **raises**. Falling back to the filename would turn every
  utterance into its own speaker, and a speaker-disjoint split over
  one-utterance speakers is disjoint in name only. Use `speaker_pattern=` for a
  flat tree that names its speaker in the filename, or
  `speaker_id_strategy="per-file"` if every file really is its own speaker.
- A `speaker_pattern` that misses a path **raises**, for the same reason.

## Utterance ids: unique, or parseable

`utt_id_style="digest"` (the default) appends a hash of the relative path, so two
corpora can never collide. `"stem"` keeps the corpus's own file naming instead,
and raises on a collision rather than letting two files share an id.

Use `"stem"` when something downstream parses structure out of the id:
`egs/voice_isolate`'s `build_chapter_corpus.py` reads the segment index out of
`..._seg_2`, and a digest lands after it, so that recipe's setup passes
`--utt-id-style stem`.

Every corpus command also takes `--id-prefix`. The prefix leads every `uttid`
and `spkid` the command writes, which is what keeps two corpora apart when they
are pooled, and what `trainer.speaker_source_weights` matches on.

## Resampling: once, not every epoch

The corpus commands take `--resample-to RATE` (with `--resample-root DIR` and
`--jobs N`): the audio is converted once into a mirrored tree, and the metafile
points at the converted files. From Python:

```python
from puresound.dataset.corpus import build_resampled_tree

converted, report = build_resampled_tree(
    records, source_root="/corpus", dest_root="/corpus_16k",
    target_sample_rate=16000, jobs=16,
)
print(report.summary())    # "N file(s): ... converted, ... reused, ... failed"
```

Three properties worth knowing:

- **Same resampler as training.** Conversion goes through `AudioIO`, which is
  what the synthesis pipeline uses. A corpus converted by another resampler is a
  slightly different corpus, and the difference lands in the model rather than in
  a log.
- **Lengths come from the written file**, never from scaling the source numbers.
- **Processes, not threads.** `torchaudio`'s sox effects sit on libsox's global
  state, and calling them from several threads aborts the interpreter
  intermittently. `jobs=1` runs inline, which is what you want under a debugger.

Conversion is resumable -- an existing non-empty destination is reused. A file
that fails is dropped from the returned records and listed in the report, so one
bad file neither ends the run nor becomes a metafile row pointing at nothing.
Two source files that would mirror to the same destination (`a.wav` beside
`a.flac`) are refused up front, because the second would otherwise be taken as
the first's converted copy.

## Speech corpora

### DNS Challenge

```bash
python -m puresound.dataset.corpus.dns_challenge speech <dns_root> --output-dir DIR \
    --subset read_speech --valid-ratio 0.05 \
    --resample-to 16000 --resample-root <dns_root>/datasets_fullband_16k/clean_fullband
python -m puresound.dataset.corpus.dns_challenge noise <dns_root> --output-dir DIR \
    --resample-to 16000 --resample-root <dns_root>/datasets_fullband_16k/noise_fullband
python -m puresound.dataset.corpus.dns_challenge devset <dns_root> --output-dir DIR
```

| Subcommand | Writes |
|---|---|
| `speech` | speaker-disjoint `<id-prefix>_train.csv` / `<id-prefix>_valid.csv` from `clean_fullband` (or `--train-metafile` / `--valid-metafile`) |
| `noise` | the folder `augmentation_noise` points at, at the training rate, plus `<id-prefix>_noise.jsonl` |
| `devset` | `<id-prefix>_devset.jsonl`, an inventory of the shipped dev set tagged with noise category and capture device |

The three parts of the corpus are used differently, which is why they are three
commands. Noise is never listed in a metafile -- the synthesis pipeline draws it
from a folder -- so `noise` exists to produce that folder at the training sample
rate.

`speech` takes `--subset` once per subset (omitted: the whole clean folder) and
knows where each shipped subset keeps its speaker:

- `read_speech` is one flat directory of `book_..._reader_<id>_..._seg_N.wav`,
  so the speaker comes out of the filename. The M-AILABS-based French, Italian,
  Russian, German and Spanish subsets, `emotional_speech` and
  `VocalSet_48kHz_mono` each have their own pattern
  (`SUBSET_SPEAKER_PATTERNS`); `vctk_wav48_silence_trimmed` is one directory
  per speaker. `--speaker-pattern` or `--speaker-id-strategy` covers a subset
  the table does not know.
- The German Spoken Wikipedia part names the article, not the reader. An article
  is one reader, so it is a sound unit to split on, but its "speaker" count is
  not a speaker count -- weight that source by share
  (`trainer.speaker_source_weights`), never by how many speakers it appears to
  have.
- `vctk_wav48_silence_trimmed` never contributes p232 or p257, the two
  VoiceBank-DEMAND test speakers, so the VCTK-DEMAND gate stage stays out of
  domain. `--exclude-speaker ID` (repeatable) leaves out more.
- Files whose name and size repeat within a subset are dropped, because DNS
  ships second copies of several subsets; `--keep-duplicates` keeps them.

#### The dev set carries the only per-category axis

The dev set has no clean reference -- it is scored no-reference. What it does
have is a noise category and a capture device in **every filename**, which
`parse_devset_name` recovers. Diagnosis needs that axis: noise suppression fails
per class, and a mean over classes hides which one.

The labels were written by hand, so one source appears as `fan`, `fan_noise` and
`fannoice`. `CATEGORY_ALIASES` collapses the clear cases and `category_raw`
keeps the original. A spelling not in the table keeps its own normalised form
rather than being forced into a neighbour: a wrong merge is worse than a long
tail. A name with nothing to anchor on returns `{}`, an honest gap rather than a
guess at where the device stops and the noise starts.

The **training** noise carries no labels -- `noise_fullband` files are named
after AudioSet clip ids. Nothing here invents one; the `tagger` argument to
`scan_folder` is the hook for a label source when there is one.

### VCTK-DEMAND

```bash
python -m puresound.dataset.corpus.vctk_demand testset <root> --out-dir DIR
python -m puresound.dataset.corpus.vctk_demand speech  <root> --out-dir DIR --resample-to 16000
python -m puresound.dataset.corpus.vctk_demand noise   <root> --out-dir DIR
```

Valentini's VCTK + DEMAND set is what most of the noise-suppression literature
reports PESQ on, so a result on it reads against published work rather than only
against previous runs of our own.

`testset` **imports** the test set rather than synthesising it -- the corpus
ships both sides already paired -- and keeps its transcripts, which is what lets
one set serve both the reference-metric stages and the WER stage. A WER set
assembled separately would answer a slightly different question on slightly
different cuts.

`speech` builds train/valid metafiles from the clean training half. The
utterances sit in one flat directory with the speaker in the filename
(`p232_001.wav`); ids default to `--id-prefix vctk --utt-id-style stem`, and the
28-speaker training split is preferred because it is the one the published
baselines train on. Both training splits are disjoint from the test speakers.
VCTK ships at 48 kHz: `--resample-to 16000` converts it once into a sibling
`<root>_16k/` tree (or `--resample-root`).

`noise` recovers the DEMAND side. The corpus has no noise folder, but both halves
are sample-aligned, so `noisy - clean` is the noise; the subcommand writes one
`<type>.wav` track per noise type, read from the `log_*.txt` files the corpus
ships (the only place the type is written down). `--splits` picks the test split,
the training split or both; the two carry different noise types. A model trained
on DNS noise has heard none of them, which makes this the held-out noise pool:
`evaluation.tools.mix_paired_set` builds the hard WER set from it. Do not train
on it.

### LibriLight

```bash
python -m puresound.dataset.corpus.librilight select /path/to/audio/LibriLight \
    --hours 4000 --max-hours-per-speaker 4 \
    --exclude-speakers-in /path/to/audio/LibriTTS/test-clean \
    --out data/librilight/selection.jsonl
python -m puresound.dataset.corpus.librilight segment data/librilight/selection.jsonl \
    --source-root /path/to/audio/LibriLight \
    --dest-root /path/to/training_set/ns_speech/librilight_segments \
    --out data/librilight/ll_train.csv
```

LibriLight is LibriVox audiobooks, one 16 kHz FLAC per chapter with a JSON
beside it carrying the reader, a voice-activity list and an estimated SNR.

- `select` ranks every chapter by that SNR and takes the cleanest until
  `--hours` of voiced audio are spent, with `--max-hours-per-speaker` so a few
  prolific readers do not become most of the corpus, and optionally a
  `--min-snr` floor. Chapters without a finite SNR are skipped. Readers of any
  held-out set named by `--exclude-speakers-in DIR` (a directory of
  `<speaker>/` folders) are refused first: LibriLight shares LibriVox's reader
  ids with LibriSpeech and LibriTTS, and the hard WER set is LibriTTS
  `test-clean`. `--exclude-speaker ID` refuses single readers.
- `segment` cuts each chosen chapter along its voice activity into
  `--min-seconds` .. `--max-seconds` pieces (a gap over `--max-gap` always ends
  a piece; `--context` seconds of surrounding audio are kept on each side), so a
  few-second training crop does not decode a whole chapter. Speakers are written
  as `ll_<reader>`.

### Kaldi-style lists

For a corpus that already ships `wav.scp` / `utt2spk` (and optionally
`utt2gender`):

```bash
python -m puresound.dataset.corpus.kaldi data/train.csv \
    data/train_wav2scp.txt data/train_utt2spk.txt \
    --utt2gender_path data/train_utt2gender.txt \
    --insert_root_path /corpus/root --separator " "
```

| Argument | Meaning |
|---|---|
| `output_path` (positional) | the CSV metafile to write |
| `wav2scp_path` (positional) | `<uttid> <wav_path>` per line |
| `utt2spk_path` (positional) | `<uttid> <spk_id>` per line |
| `--utt2gender_path` | `<uttid> <gender>` per line; omitted, every row gets `None`, which the sampler treats as unknown gender |
| `--separator` | column separator of the **input** lists; the output is always comma-separated |
| `--insert_root_path` | prefix prepended to every path read from `wav2scp` |
| `--on-error {skip,raise}` | `skip` (default) drops a row whose audio cannot be read, `raise` stops |

Rows missing from `utt2spk` (or from `utt2gender`, when given) are dropped, and
every drop is counted in the summary. The converter writes one file, so run it
once per split. `dataset.test_folder`
(read by a recipe's `--scoring` / `--inference`) is a different format: a
directory with `wav2scp.txt` (plus `wav2ref.txt` for `--scoring`), in the same
`<uttid> <path>` form this converter reads as input.

## Shaping a speech corpus

### Joining short utterances

```bash
python -m puresound.dataset.corpus.concat_short data/vctk/vctk_train.csv \
    --source-root /path/to/audio/vctk_16k --dest-root /path/to/training_set/ns_speech/vctk_concat \
    --out data/vctk/vctk_concat_train.csv
```

The recipes drop every utterance shorter than
`dataset.filter_min_utterance_length`. That is right for corpora cut into long
segments and silently deletes most of a corpus made of single sentences.
Lowering the filter would change every source's rows; joining changes only the
corpora that need it. A speaker's short utterances are taken in id order and
joined with `--gap` seconds of silence until a piece reaches `--min-seconds`
(never past `--max-seconds`). Utterances already long enough pass through
untouched, and a leftover shorter than the minimum is kept for the recipe's
filter to decide. A joined row's uttid ends in `_cat<N>` and its inventory tag
`joined` lists the originals.

### Cleaning targets through a checkpoint

```bash
python -m puresound.dataset.corpus.clean_speech \
    --metafile data/dns5/dns5_train.csv \
    --source-root /path/to/audio/dns-5/datasets_fullband_16k \
    --dest-root /path/to/training_set/ns_speech_cleaned/dpcrn_mamba_v2/dns5 \
    --recipe egs/noise_suppression/config/infer_dpcrn.yaml \
    --ckpt egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    --out-metafile data/dns5/dns5_train.v2clean.csv
```

"Clean" speech corpora are not clean: read audiobooks carry room tone, hum and
the recorder's noise floor, and a model trained to reproduce its target
reproduces that too. `clean_speech` runs every file through any checkpoint
`puresound.evaluation.systems` can load and writes the output to **its own
tree**, named after the cleaner and mirroring the source layout -- never beside
the source, so a target that has been through a model is not mistaken for a
recording and two cleaners' outputs cannot mix. It writes a metafile pointing at
the cleaned copies (default: the input metafile with a `.clean.csv` suffix) and a
stats file (default: the output metafile with a `.stats.jsonl` suffix) with, per
file:

| Field | Meaning |
|---|---|
| `removed_db` | energy of `input - output` relative to the input: how much the cleaner changed |
| `floor_in_db` / `floor_out_db` | level of the quietest 20 % of 32 ms frames before and after; the difference is the noise floor the cleaner took away |

Practical points:

- `--dry-blend` defaults to 1.0, the model's own output. Do not clean with the
  export default of 0.9, which keeps a tenth of the input.
- Files are batched by length (`--max-batch-seconds`, `--max-batch-items`) and
  padded at the end only; the models are causal, so an item's output does not
  depend on its batch. `--verify N` (default 4) checks that on real files before
  the run, and the run refuses to start if batched and single-file outputs
  differ. TF32 is switched off so the output is a function of the cleaner and the
  input only.
- The run is resumable: a file already in the stats file whose output exists is
  skipped. `--limit N` cleans the first N rows, for a trial.

### Choosing cleaned or original targets

```bash
python -m puresound.dataset.corpus.target_policy \
    --original data/dns5/dns5_train.csv \
    --cleaned data/dns5/dns5_train.v2clean.csv \
    --policy floor-drop --min-floor-drop 3 \
    --out data/pool/dns5_train.csv
```

Cleaning and using the cleaned copy are separate decisions, because the cleaner
is not transparent: even speech that is already clean comes back measurably
changed, and replacing a file that had no noise to remove writes the cleaner's
colouring into the target.

| `--policy` | Target |
|---|---|
| `all` | every file cleaned |
| `none` | every file original (the control) |
| `floor-drop` | cleaned only where `floor_in_db - floor_out_db >= --min-floor-drop` -- where there was something to clean |

A file with no cleaned copy, or no stats under `floor-drop`, keeps its original:
missing cleaning is never a reason to lose a file. `--cleaned` (and `--stats`)
repeat when the originals were cleaned in several runs; the stats default to the
`.stats.jsonl` beside each cleaned metafile. The command prints the share of
targets it replaced.

## Noise

Noise is never listed in a metafile. The synthesis pipeline reads every `.wav`
under a folder (recursively), so a noise source is a folder at the training
sample rate:

| Source | Built by |
|---|---|
| DNS Challenge noise | `dns_challenge noise` |
| FSD50K, CochlScene, MUSAN | `noise_corpora`, then `speech_screen` |
| speech-shaped noise | `ssn` |
| DEMAND (held out, for WER sets only) | `vctk_demand noise` |

### Public noise corpora

```bash
python -m puresound.dataset.corpus.noise_corpora fsd50k /path/to/audio/FSD50K \
    --dest-root /path/to/training_set/ns_noise/fsd50k
python -m puresound.dataset.corpus.noise_corpora cochlscene /path/to/audio/CochlScene \
    --dest-root /path/to/training_set/ns_noise/cochlscene
python -m puresound.dataset.corpus.noise_corpora musan /path/to/audio/musan \
    --dest-root /path/to/training_set/ns_noise/musan
```

Each corpus is listed with the tags that decide whether a clip may be used,
filtered, and converted to 16 kHz with the shared resampler. The command prints
how many clips it keeps and why each of the others went; `--dry-run` stops after
that tally. What is filtered:

- **Licence.** FSD50K licenses clip by clip; only CC0 and CC BY are kept, since
  a commercial model cannot train on the NC and Sampling+ clips.
- **Labelled voices.** Noise suppression keeps every voice in its input, so a
  noise clip that is somebody talking teaches deletion. FSD50K clips labelled
  anywhere in AudioSet's human-voice subtree are dropped. Crowd and chatter are
  kept on purpose -- unintelligible babble is exactly the restaurant and station
  noise a model has to handle -- and go through `speech_screen` to catch the
  talkers labels miss.
- **Vocals in music.** MUSAN's annotation marks which music tracks have vocals;
  those go. MUSAN's speech part is LibriVox, the source of the LibriTTS test
  speakers, and is never used.
- **Very short clips** (`--min-seconds`, default 2): the pipeline tiles a clip to
  the row length, and a short knock tiled over a row is a metronome, not a
  noise.

Beside the folder it writes `<folder>.inventory.jsonl` with the tags and, for
attribution licences, `<folder>.attribution.jsonl`.

### Screening out intelligible speech

```bash
python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \
    --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl --device cuda
python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \
    --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl \
    --apply --rejected-root /path/to/training_set/ns_noise_rejected/cochlscene
```

The line is drawn at **intelligible**, in two passes: Silero VAD marks where
anything voice-like is, then faster-whisper (`--whisper-model`, default
`large-v3`) transcribes only those regions. A file counts as speech when whisper
both believes there is speech and is confident in at least a few words; babble
fails the confidence test, a clear talker passes it. The first command writes one
JSONL row per file with the numbers (resumable by path), so the threshold can be
revisited afterwards. `--apply` then **moves** the files judged intelligible into
the `--rejected-root` tree -- moved, not deleted, so the call can be audited.
Needs the `silero-vad` and `faster-whisper` packages.

### Speech-shaped noise

```bash
python -m puresound.dataset.corpus.ssn data/dns5/dns5_train.csv \
    --dest /path/to/training_set/ns_noise/ssn --clips 300 --seconds 30
```

Speech-shaped noise has the long-term spectrum of speech, so it overlaps speech
everywhere a mask could separate it, and no recording corpus supplies it. Each
clip takes the long-term spectrum of `--utts-per-clip` random utterances (a
different draw per clip) and applies it to white noise; every second clip is
also multiplied by the envelope of one more utterance, adding the syllable-rate
fluctuation a stationary mask handles too easily. Build it from the **training**
metafile only, never a test set. A `<dest>.manifest.jsonl` records each clip's
sources and seed.

## Pooling speech corpora

```bash
python -m puresound.dataset.corpus.pool merge \
    --source data/dns5/dns5_train.csv \
    --source data/vctk/vctk_train.csv \
    --out data/pool/pool_train.csv
```

A recipe names one `train_metafile`, so training on several corpora is a merge.
The report is why this is a module: **the sampler draws speakers uniformly**, so
a corpus's share of the batches is its *speaker count* over the pool's, not its
hours, and the command prints that share per source:

```
  dns5_train     <spk> spk  <utt> utt  <hours> h  -> <share> of batches
  vctk_train     <spk> spk  <utt> utt  <hours> h  -> <share> of batches
```

- `--max-speakers STEM=N` caps a source (named by its metafile stem) by dropping
  whole speakers, the unit the sampler works in. Capping rows would change the
  hours and leave the batch share where it was.
- `--exclude-speakers FILE` (repeatable) names spkids, one per line, that no
  source may contribute -- for example readers matched by voice to a test set.
- The merge refuses a `uttid` or `spkid` present in two sources: the first is a
  file the dataset would never draw, the second fuses two corpora's speakers into
  one sampler class. Both are what each corpus command's `--id-prefix` prevents.
- Every row's audio is read once, and a file that is all zeros, empty or
  unreadable is dropped and counted per source. The coverage sampler names the
  utterance the dataset loads, and the dataset treats such a file as an error
  instead of drawing another, so one left in the pool stops a run hours in. The
  DNS-5 German Wikipedia set alone holds 88 all-zero files. `--jobs N` sets the
  processes for the read (default half the cores; about 9 minutes for 1.85M files
  on 22); `--skip-audio-check` skips it. If no rows survive filtering, the merge
  fails without replacing the output metafile.

To choose the shares instead of inheriting them from speaker counts, set
`trainer.speaker_source_weights` (below).

## Recipe knobs these feed

### Speech: `dataset` and `trainer.speaker_source_weights`

```yaml
dataset:
  train_metafile: data/pool/pool_train.csv
  valid_metafile: data/pool/pool_valid.csv

trainer:
  speaker_source_weights: {dns5_: 0.6, ll_: 0.3, vctk_: 0.1}
```

`speaker_source_weights` maps a **spkid prefix** to a weight. Each training batch
slot first draws a source by weight, then a speaker uniformly inside it; without
the block every speaker is equally likely, so a corpus's share is its share of
the speakers. Rules, all checked when the sampler is built:

- Every speaker must fall under exactly one prefix; the longest matching prefix
  wins. A speaker under no prefix, or a prefix matching no speaker, is an error.
- Weights must be positive and are normalised; the log prints each source's
  speaker count and share of batch slots.
- It cannot be combined with sample-rate-first selection (a recipe with no
  `dataset.target_sample_rate`).
- Validation stays uniform over its own metafile, so validation loss keeps
  measuring the same thing whatever the training mixture.

### Noise: `augmentation_noise.noise_folder` or `noise_sources`

```yaml
augmentation_noise:
  used: True
  prob: 0.9
  noise_sources:
    - {name: dns5,   folder: /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband, weight: 0.6}
    - {name: fsd50k, folder: /path/to/training_set/ns_noise/fsd50k, weight: 0.3}
    - {name: ssn,    folder: /path/to/training_set/ns_noise/ssn, weight: 0.1}
  snr_range: [-5, 40]
  prob_white_noise: 0.05
  white_noise_snr_range: [10, 30]
```

`noise_folder` names one folder, and every file in it is equally likely, so a
corpus's share is its file count. `noise_sources` gives several folders at chosen
shares: each draw picks a source by `weight` (default 1.0), then a file uniformly
inside it. Give one or the other, not both. Source names must be unique, and a
source folder with no `.wav` files is an error. Files are keyed by source name
plus basename, so two corpora that reuse a filename do not overwrite each other.

### SNR: `augmentation_noise.snr_bands`

```yaml
augmentation_noise:
  snr_range: [-5, 40]
  snr_bands:
    - {low: -5, high: 5,  prob: 0.4}
    - {low: 5,  high: 15, prob: 0.4}
    - {low: 15, high: 40, prob: 0.2}
```

Without `snr_bands` the SNR is uniform over `snr_range`. With it, a band is
picked by `prob` and the SNR drawn uniformly inside it -- a piecewise-uniform
distribution, so the range can reach high SNRs without thinning the hard
mixtures. `snr_range` stays as the envelope: every band must lie inside it, each
band needs `high > low`, and the probabilities must sum to 1.
