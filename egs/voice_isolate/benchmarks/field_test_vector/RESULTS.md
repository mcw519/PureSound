# voice_isolate field benchmark — test_vector

The field benchmark for this recipe. `build_cases.py` cuts it from two internal field
recordings using the hand labelling in `spans/` (`tools/audio_annotator.html`), writing the
clips into the ignored `../../data_report/field_cases/test_vector_cases/`.

**The recordings are internal and are never version-controlled.** What lives here — the
labels, `windows.json`, this file, and the scorecards under `records/` — carries no audio.
A clean clone can read the method and the standings but cannot rerun the benchmark without
the recordings, which travel out of band.

Supersedes the retired `laptop90d_cases`, whose old label set is kept beside the recordings
under `../../data_report/field_cases/_superseded/` for provenance.

| tag | file | length | near talker | far talkers | noise floor | near reference |
|---|---|---|---|---|---|---|
| `90d` | `real_record_test_90D.wav` | 134.8 s | 30 cm | 200 / 300 cm | −78.71 dBFS | −40.95 dBFS |
| `270d` | `real_record_test_270D.wav` | 98.3 s | 50 cm | 200 / 300 cm | −78.53 dBFS | −43.36 dBFS |

`real_record_test_90D.wav` is bit-identical to the retired `field_cases/raw/real_distance_test_vector_90D.wav`
(md5 `1712d4cb…`) — only the labels changed, which is what makes the old/new comparison below exact.

## Two modes, never averaged

* **STREAM** (`*_session`) — the whole recording, spans in place. The near anchor is present:
  this is mid-conversation behaviour.
* **COLD-START** (`*_near*`, `*_far*`, `*_dt*`) — each span cut out and fed from t=0 with no
  context. This is bot-idle behaviour and a different product metric.

`double-talk` spans are scored keep-only: the near voice must survive the overlap; the far
side is not separable without a reference.

## Why attenuation alone is not the score

`reduction_db` is relative, so it only compares spans that started at comparable levels. A far
voice already near the capture noise floor has little removable energy, and a span that *is*
the floor scores 0 dB no matter what the model does. Every suppress span is therefore reported
three ways, and a span with less than 6 dB of headroom over the floor is NOT-SCORABLE rather
than given a misleading number:

| column | meaning |
|---|---|
| `reduc` | out/in over the span — how hard the model pushed |
| `residual` | out − floor — **what is still above room tone: did the bystander go away** |
| `sir_out` | near_ref − out — how far the leftover sits below the user's own voice |
| `headroom` | in − floor — how much was removable in the first place |

On this material every far span has 24.5–31.2 dB of headroom, so nothing here is floor-limited
and no span is NOT-SCORABLE. That is now a measured fact in the scorecard rather than an
assumption.

## Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    --cases-dir egs/voice_isolate/data_report/field_cases/test_vector_cases \
    --device cuda:0 --dry-blend 0.9
```

`expandable_segments` is required on a 24 GB card: the 134.8 s session is one forward pass and
the DPCRN LSTM asks for a single 9.8 GiB block. CPU and GPU agree to 0.01 dB.

## Standings (`dry_blend 0.9`, 2026-08-16)

| metric | v7 | v8 | v9 | v10 |
|---|---|---|---|---|
| STREAM 90d — keep / reduc / residual | −0.46 / **−13.73** / 14.6 | −0.47 / −12.35 / 16.0 | −0.46 / −11.82 / 16.5 | −0.56 / −11.19 / 17.1 |
| STREAM 270d — keep / reduc / residual | −1.08 / −9.86 / 17.3 | −1.11 / −11.29 / 15.9 | −1.12 / **−13.18** / 14.0 | −1.17 / −11.68 / 15.5 |
| COLD-START far — median residual | 25.3 | 25.7 | 23.3 | **23.6** |
| COLD-START far — best residual | 19.9 | 20.2 | 19.0 | **12.6** |
| COLD-START far — median sir_out | 10.9 | 11.1 | **14.9** | 14.2 |
| COLD-START far verdicts | 9 FAIL / 1 PARTIAL / **0 ok** | 9 FAIL / 1 PARTIAL / **0 ok** | 8 FAIL / 2 PARTIAL / **0 ok** | 5 FAIL / 5 PARTIAL / **0 ok** |
| COLD-START near keep, worst | −0.79 | −0.81 | −0.68 | −0.34 |
| COLD-START double-talk keep, worst | −2.21 | −2.26 | −2.27 | −2.35 |
| KEEP-VIOLATION | 0/15 | 0/15 | 0/15 | 0/15 |

Per clip: `records/scorecard_<version>.tsv`.

## What the numbers say

1. **The keep side is not at risk on this material.** Near 30/50 cm preserves within 0.9 dB and
   every double-talk near voice survives, in all four versions, in both modes. v10 is the best
   of the four on worst-case near keep (−0.34 dB).

2. **No version makes a cold-start bystander go away.** Zero `ok` verdicts across all four.
   Everything that clears the −6 dB reduction bar is SUPPRESS-PARTIAL: the residual still sits
   12.6–23.6 dB above room tone and only 11–15 dB below the user's own voice. An ASR competing
   with a voice 14 dB down will still transcribe it. Judged on relative attenuation alone, v10
   looked like "5 of 10 pass"; judged in absolute levels, nothing passes.

3. **Cold start is still where the versions separate.** The same far speech the stream scores
   at −12 dB reads −0.2 dB when cut out and played alone. v7 and v8 are effectively pass-through
   there; v9 has the first real signal; v10 roughly halves the failure count and produces the
   single best residual in the set (270d_far1, 12.6 dB over the floor).

4. **v10's gain is confined to 200 cm, and the split is by distance, not level.** All 4 of the
   200 cm clips clear the reduction bar; 5 of the 6 at 300 cm still pass through. Rank
   correlation of cold-start depth against clip level is −0.08 — nothing — while against
   labelled distance it is **+0.96** (v9 +0.76, v8 +0.62). The decisive pair: `270d_far1` is the
   quietest clip in the set (−54.0 dBFS) and is suppressed deepest, while `90d_far4` is 3.9 dB
   louder and passes through. Duration does not explain it either (−0.25).

   So the remaining cold-start wall is at the **far** end — the counterintuitive direction, since
   more distance means more of the reverberation cue the model should be reading. Caveat: here
   distance is confounded with talker identity (on 90D the 2 m and 3 m voices are two different
   people), so "300 cm" means "that talker at that distance".

5. **STREAM suppression orders differently on the two recordings** (90d: v7 > v8 > v9 > v10;
   270d: v9 > v10 > v8 > v7). One session cannot rank versions. v10 costs about 1 dB of stream
   suppression against v8 and 0.1 dB of double-talk keep — neither near a threshold.

## Old labels vs new labels

Same audio, same v8 output, only the span set differs:

| 90D STREAM | keep | suppress | labelled duration |
|---|---|---|---|
| old (`laptop90d_cases` `session`) | −0.51 | **−16.81** | keep 74.2 s / suppress 26.6 s |
| new | −0.47 | **−12.35** | keep 80.7 s / suppress 42.2 s |

The old set labelled 26.6 s of far speech where there are 42.2 s, and the 15.6 s it missed are
the shallow-suppression ones — so it overstated far suppression by 4.5 dB. The keep side was
unaffected. Any figure quoted from `laptop90d_cases` on the suppress axis is void.

## What the rest of the gate said about v10

This scorecard is one axis, and on its own it made v10 look like the winner. The full
nine-stage gate ([`../full_gate/v10.txt`](../full_gate/v10.txt)) says otherwise:

| gate | v8 | v9 | v10 |
|---|---|---|---|
| **moderate-reverb WER, paired vs v8** — primary WER gate | — | not yet run | **+0.024 worse** [+0.008, +0.041] |
| BUT-OFFICE WER vs mix (monitor; n=200 resolves ~±0.03) | −0.022 [−0.049, +0.002] | −0.050 | −0.006 [−0.033, +0.018] |
| Dawn Chorus WER / deletion (raw 0.184) | 0.180 / 0.094 | **0.172 / 0.086** | 0.173 / 0.088 |
| extreme-reverb monitor vs mix | +0.020 | **−0.003** | +0.017 |
| turn-taking KEEP violations | 6 | 6 | **10** |
| turn-taking SUPPRESS | 87/13 @ −16.14 | 80/20 @ −13.43 | **89/11 @ −15.98** |
| in-domain SI-SDRi | +7.99 | **+8.10** | +7.56 |
| cold-start far clearing −6 dB (this set) | 1/10 | 2/10 | **5/10** |

So the cold-start curriculum bought exactly what it was built to buy, and paid for it on the
WER gate: on the deployment reverberation range v10 is 0.024 WER worse than v8, paired on
identical utterances, with the interval clear of zero. `dpcrn_v8.ckpt` remains the default;
v10 is kept as the cold-start reference.

Two lessons, and the second is the bigger one:

1. A recipe optimised against one axis will move that axis. This scorecard cannot be read
   alone — read it beside [`../full_gate/`](../full_gate/).
2. **The gate it was first judged against could not measure what it was asked to.** The
   original verdict said v10's BUT-OFFICE result was a collapse and that its
   single-interferer half came out worse than the unprocessed mixture. Bootstrapping that
   set showed neither claim survives: at n=200 nothing we have is distinguishable from
   doing nothing on BUT-OFFICE, including v8's own headline −0.024. The verdict held only
   because a set with real resolution agreed with it. See
   [`../wer_sets/README.md`](../wer_sets/README.md).
