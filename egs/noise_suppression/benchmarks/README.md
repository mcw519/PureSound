# noise_suppression — benchmark definitions and records

Traditional Chinese: [`README.zh-TW.md`](README.zh-TW.md)

This directory is version-controlled and carries **no audio**. It holds what each
benchmark stage is, and what every released checkpoint and compared external
system got. The audio those numbers were computed on lives under `data_report/`,
which is not tracked; the command that rebuilds it is part of each stage's
definition in [`stages.md`](stages.md).

Tracking the numbers is the point. A version comparison that exists only in a
chat log or a run directory gets re-argued from scratch later, usually with a
different protocol.

## Layout

```
benchmarks/
  README.md            this file -- the record contract
  stages.md            what each gate stage is, and what it can resolve
  records/<tag>.json   one scored checkpoint (or external system), one file
```

`run_full_gate.sh <tag> ...` writes `records/<tag>.json`; how to run it is in
[`docs/usage/evaluation.md`](../../../docs/usage/evaluation.md).

## What a record must carry

Every record is a JSON object with these keys. Anything missing makes the record
unusable for comparison rather than merely incomplete.

| Key | Why it is mandatory |
| --- | --- |
| `tag` | the record's name, the same as its filename |
| `checkpoint` | path and epoch of what was scored (`unprocessed` for the baseline, the directory for precomputed audio) |
| `recipe` | the config the model was **built** from -- a recipe/checkpoint mismatch silently drops weights |
| `chain_commit` | scoring-time `git rev-parse --short HEAD`, plus `+dirty`; a frozen set's build identity lives in its stage's `extra.set_provenance` |
| `inference` | the knobs in force (`dry_blend`, guards, `precomputed`) -- a number without its operating point is not reproducible |
| `stages` | one entry per stage, each with its role, `n`, the metric, its confidence interval and its own verdict |
| `verdict` / `unresolved_gates` | the aggregate `pass` / `fail` / `no-resolution`, and the gate stages that could not resolve |
| `lineage` | what the run was testing, its parent stage, and `released_as` if the checkpoint was promoted -- a tag alone does not survive a year |

`evaluation.tools.collect` writes everything except `lineage`, which is added by
hand when the record is filed: `variable` (what the run changed), `parent`,
`snr_range`, and `released_as` / `release_note` when the checkpoint was promoted.

`no-resolution` is a real verdict and must be used. A stage whose confidence
interval covers the difference being claimed has not measured anything, and
recording it as a win is how a non-deployable version gets promoted.

## Record names

A record's filename is the run directory's name plus the epoch scored,
`<task>_<arch>_<variable>_<stage>_ep<N>` -- for example
`ns_dpcrn-mamba_activebin_ft_ep1.json`. A record of the same checkpoint on
another set or at another operating point adds a suffix (`_werhard` for a record
scored on the hard WER set alone). A published system scored through
`PRECOMPUTED_DIR` is `ext_<system>`.

A version number appears in exactly one place: a catalog id, assigned when a
checkpoint is promoted. `noise-suppression-dpcrn-mamba-v1` is the record
`ns_dpcrn-mamba_activebin_ft_ep1`; the run never carried the `v1`, and the
record's `lineage.released_as` names the catalog id. A record that was renamed
keeps its earlier name in `lineage.previously_recorded_as`, and its `checkpoint`
/ `recipe` still point where the artifacts really are.

## Rules the records follow

1. **Report the absolute residual, not only the improvement.** A large delta on a
   bad starting point is not a good end state.
2. **One checkpoint is not a measurement.** Neighbouring epochs of the same run
   can move a metric by more than the version differences being compared. Score
   a block of the run's last checkpoints and report the block statistic.
3. **Print `n` and the interval on every stage.** A set that cannot separate the
   candidate from doing nothing is a monitor, not a gate, and `stages.md` says
   which it is.
4. **WER is read as a delta against the unprocessed mixture**, on the same cuts,
   with a strong recogniser. A weak recogniser hides over-suppression.
5. **dB differences are compared at matched SNR.** Across sets, absolute dB
   carries a set-dependent bias.
6. **Composite quality scores are monitors.** They are reported because they are
   externally comparable, not because they can resolve our failure modes -- a
   distortion-free passthrough can score well.
