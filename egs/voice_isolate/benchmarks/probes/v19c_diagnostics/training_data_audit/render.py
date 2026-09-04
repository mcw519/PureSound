import json, sys
d = json.load(open(sys.argv[1]))
def g(o, k):
    v = o.get(k)
    if v is None: return "--"
    if isinstance(v, dict): return f"{v['median']:.2f} [{v['p10']:.2f},{v['p90']:.2f}]"
    if isinstance(v, list): return "--" if v[0] is None else f"{v[0]:.3f}"
    if isinstance(v, tuple): return f"{v[0]:.3f}"
    return f"{v}"
cols = [
 ("n", "n_rows"),
 ("row_s med[p10,p90]", "row_seconds"),
 ("tgt_onset_s", "target_onset_s"),
 ("onset<=0.5s", "frac_onset_le_0.5s"),
 ("onset>=1s", "frac_onset_ge_1s"),
 ("onset>=2s", "frac_onset_ge_2s"),
 ("itf_pre_onset_s", "itf_before_onset_s"),
 ("itf>=1s pre", "frac_itf_ge_1s_before_onset"),
 ("itf any pre", "frac_itf_any_before_onset"),
 ("tgt_gap_s", "target_longest_gap_s"),
 ("tgt_int_gap_s", "target_longest_interior_gap_s"),
 ("gap>=2s", "frac_target_gap_ge_2.0s"),
 ("gap>=3s", "frac_target_gap_ge_3.0s"),
 ("gap>=5s", "frac_target_gap_ge_5.0s"),
 ("reentry>=5s", "frac_target_reentry_after_5s"),
 ("max_int_gap_s", "max_target_interior_gap_s"),
 ("anyspeech_gap_s", "any_speech_longest_gap_s"),
 ("anygap>=2s", "frac_anyspeech_gap_ge_2s"),
 ("open>=0.5s_nospeech", "frac_row_opens_with_0.5s_no_speech"),
 ("open>=1s_nospeech", "frac_row_opens_with_1s_no_speech"),
 ("tgt_absent_frac", "frac_target_absent"),
 ("turn_frac", "turn_taking_frac"),
 ("tgt_active_frac", "target_active_frac"),
 ("itf_active_frac", "itf_active_frac"),
]
for det in d:
    print(f"\n\n##### detector = {det}\n")
    print("| group | " + " | ".join(c[0] for c in cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for label, o in d[det].items():
        print(f"| {label} | " + " | ".join(g(o, c[1]) for c in cols) + " |")
