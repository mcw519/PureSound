import json, pathlib
D = pathlib.Path("/tmp/claude-1001/-home-milowu-A4Audio-PureSound/c52f3b49-8e8f-4f9f-a5ca-862771bd494c/scratchpad/v19_plan/chain_readability")
o = []
w = o.append

w("# chain_readability -- full tables (2026-09-04)\n")
w("Scripts in this directory; cache at `<scratch>/agcache/{v8,v16,v11b}`.")
w("v11b ckpt = `egs/voice_isolate/exp/dpcrn_v11b_compinv/lightning_logs/version_0/checkpoints/epoch=31-step=16000.ckpt`")
w("(ep31 is the latest that exists; training was stopped mid-cosine by hand -- see `probes/v11b_VERDICT.md`).\n")

d = json.load(open(D / "readability_3tags.json"))
w("## 1. `anchor_gate_sim.py readability`, all three tags (pooled frames)\n")
w("| tag | chain | scope | W | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | nK | nS |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag in ("v8", "v16", "v11b"):
    for r in d["tags"][tag]["auc"]:
        w(f"| {tag} | {r['chain']} | {r['scope']} | {r['W']} | {r['auc_dist']:.3f} | {r['auc_drr']:.3f} | "
          f"{r['median_keep_m']:.3f} | {r['median_supp_m']:.3f} | {r['median_keep_drr']:.2f} | "
          f"{r['median_supp_drr']:.2f} | {r['n_keep_frames']} | {r['n_supp_frames']} |")
w("\n## 2. Prefix (anchor) reads, W = 1 s\n")
w("| tag | chain | prefix condition | n | median m | median DRR dB | p10 m | p90 m |")
w("|---|---|---|---|---|---|---|---|")
for tag in ("v8", "v16", "v11b"):
    for r in d["tags"][tag]["prefix"]:
        if r["W"] != 1.0:
            continue
        w(f"| {tag} | {r['chain']} | {r['condition']} | {r['n']} | {r['median_m']:.3f} | "
          f"{r['median_drr_db']:.2f} | {r['p10_m']:.3f} | {r['p90_m']:.3f} |")

g = json.load(open(D / "per_group_auc.json"))
w("\n## 3. Per-recording-group breakdown (W = 1 s, condition `none`)\n")
w("| tag | group | scope | chain | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | nK | nS |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in g:
    w(f"| {r['tag']} | {r['group']} | {r['scope']} | {r['chain']} | {r['auc_dist']:.3f} | {r['auc_drr']:.3f} | "
      f"{r['keep_m']:.3f} | {r['supp_m']:.3f} | {r['keep_drr']:.2f} | {r['supp_drr']:.2f} | {r['nK']} | {r['nS']} |")

c = json.load(open(D / "auc_ci.json"))
w("\n## 4. Clip-level bootstrap CI (resampling unit = recording/clip, B = 2000)\n")
w("| tag | chain | scope | readout | AUC | 95% CI | n keep units | n supp units |")
w("|---|---|---|---|---|---|---|---|")
for r in c:
    w(f"| {r['tag']} | {r['chain']} | {r['scope']} | {r['readout']} | {r['auc']:.3f} | "
      f"[{r['lo95']:.3f}, {r['hi95']:.3f}] | {r['n_keep_units']} | {r['n_supp_units']} |")

p = D / "auc_delta.json"
if p.exists():
    dl = json.load(open(p))
    w("\n## 5. PAIRED bootstrap on AUC differences (same resampled clips for every tag, B = 4000)\n")
    w("| chain | scope | readout | pair | AUC v8 | AUC v16 | AUC v11b | delta | 95% CI | p (two-sided) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in dl:
        w(f"| {r['chain']} | {r['scope']} | {r['readout']} | {r['pair']} | {r['auc_v8']:.3f} | {r['auc_v16']:.3f} | "
          f"{r['auc_v11b']:.3f} | {r['delta']:+.3f} | [{r['lo95']:+.3f}, {r['hi95']:+.3f}] | {r['p_two_sided']:.3f} |")

for label, files, note in (
    ("6. BROKEN offline chain probe (`comp_readability.py`) -- kept as the record of the defect",
     ["comp_v8.json", "comp_v16.json", "comp_v11b.json"],
     "`compressor_gain` RETURNS a gain curve; this run (and `benchmarks/probes/eq_probe.py`) never "
     "multiplied it into the signal, and on these clips the curve is all-ones, so the model was fed a "
     "CONSTANT DC signal. Identical numbers across three operating points are the signature."),
    ("7. FIXED offline chain probe (`comp_readability2.py`)",
     ["comp2_v8.json", "comp2_v16.json", "comp2_v11b.json"],
     "Gain multiplied in, clip normalised to the recipe's `gain_normalized_to: -28` dBFS RMS before the "
     "compressor and restored after, so the stage operates in its training range; `GRmean/GRmax` is the "
     "gain reduction it actually applied. `wshape_*` is the |x|^p waveshaper of `compression_probe.py` "
     "(a different operator -- the one whose causality was established in 2026-08-21)."),
):
    got = [f for f in files if (D / f).exists()]
    if not got:
        continue
    w(f"\n## {label}\n")
    w(note + "\n")
    w("| tag | chain | cond | AUC dist | AUC DRR | keep m | supp m | keep DRR | supp DRR | GR mean dB | GR max dB |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in got:
        for r in json.load(open(D / f))["rows"]:
            if r["W"] != 1.0:
                continue
            w(f"| {r['tag']} | {r['chain']} | {r['cond']} | {r['auc_dist']:.3f} | {r['auc_drr']:.3f} | "
              f"{r['median_keep_m']:.3f} | {r['median_supp_m']:.3f} | {r['median_keep_drr']:.2f} | "
              f"{r['median_supp_drr']:.2f} | {r.get('gr_mean_db', float('nan')):.2f} | {r.get('gr_max_db', float('nan')):.2f} |")

lp = D / "linear_probe.json"
if lp.exists():
    w("\n## 8. Fresh linear probe on the SAME cached bottleneck frames (`linear_probe.py`)\n")
    w("Logistic regression (C = 0.1, standardised) on the 128-d pooled bottleneck window the DistHead reads.")
    w("Holdout unit = recording group, so no group is in both fit and test. This separates *is the near/far")
    w("information in the representation* from *does the DistHead's learned mapping point the right way*.\n")
    w("| tag | arm | AUC | detail |")
    w("|---|---|---|---|")
    for r in json.load(open(lp)):
        a = "n/a" if r["auc"] != r["auc"] else f"{r['auc']:.3f}"
        w(f"| {r['tag']} | {r['arm']} | {a} | {r['detail']} |")

(D / "TABLES.md").write_text("\n".join(o) + "\n")
print("wrote", D / "TABLES.md", len(o), "lines")
