import json, collections, numpy as np, sys
from scipy.stats import wilcoxon, spearmanr
S = sys.argv[1]
man = {j["id"]: j for j in (json.loads(l) for l in
       open("data_report/wer_set_moderate_test/manifest.jsonl"))}
print("# Is the wrong-anchor deletion graded by how much NEARER the anchor was?\n")
print("Condition 'other3' (= diffspk3) on the utt arm. d = anchor utterance's near_dist minus")
print("the user's near_dist (negative = the anchor talker was CLOSER to the mic than the user")
print("who follows). Metric = Delta_on1: the first 1 s of the user's speech, paired vs 'none'.\n")
for tag in ("v8", "v16ep19"):
    rows = [json.loads(l) for l in open(f"{S}/out2/rows_utt_{tag}.jsonl")]
    by = collections.defaultdict(dict)
    for r in rows:
        if not r.get("skipped"):
            by[r["id"]][r["condition"]] = r
    d, y = [], []
    for i, c in by.items():
        if "other3" not in c or "none" not in c:
            continue
        d.append(man[c["other3"]["partner"]]["near_dist"] - man[i]["near_dist"])
        y.append(c["other3"]["db_on1"] - c["none"]["db_on1"])
    d, y = np.array(d), np.array(y)
    rho, p = spearmanr(d, y)
    print(f"\n## {tag}  (n={len(d)})   Spearman rho(d, Delta_on1) = {rho:+.3f}, p = {p:.2g}\n")
    print("| stratum | n | median d (m) | median Delta_on1 (dB) | p (vs 0) |")
    print("|---|---|---|---|---|")
    q = np.quantile(d, [0, 1/3, 2/3, 1.0])
    for a, b, lab in ((q[0], q[1], "anchor nearer than user"), (q[1], q[2], "similar distance"),
                      (q[2], q[3], "anchor farther than user")):
        m = (d >= a) & (d <= b)
        yy = y[m]; yn = yy[yy != 0]
        pv = wilcoxon(yn).pvalue if len(yn) > 5 else float("nan")
        print(f"| {lab} | {m.sum()} | {np.median(d[m]):+.2f} | {np.median(yy):+.2f} | {pv:.2g} |")
print("\nReading: the effect is graded in the predicted direction (a nearer anchor deletes the")
print("next talker more) but weakly -- and it is highly significant in EVERY stratum, including")
print("when the anchor was FARTHER than the user. So the rule is not 'keep whatever was nearest';")
print("a talker-change component sits on top of the relative-distance component.")
