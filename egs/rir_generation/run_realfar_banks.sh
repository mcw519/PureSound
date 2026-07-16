#!/usr/bin/env bash
# real-far rung: manifests -> train/held-out real RIR banks (Stage B).
#
# Reads the per-corpus manifests emitted by scan_public_rir_corpora.py,
# splits held-out rooms, builds the cross-corpus real <1m near-pool (used by
# far-only rooms: ACE, DIFFRIR classroom/complex), then runs
# real_rir_to_bank.py from-manifest per corpus with a per-corpus
# items_per_room (scaled to each corpus' unique-RIR count).
#
# Outputs:
#   $EXP/real_rir_16k_train    (train bank, origin=real)
#   $EXP/real_rir_16k_heldout  (held-out real rooms, F-only probe / eval)
set -euo pipefail

ROOT=/home/milowu/A4Audio/PureSound
D=/work/any_exp_link/puresound_exp/real_rir_corpora
MF=$D/manifests
EXP=/work/any_exp_link/puresound_exp
SPLIT=$D/manifests_split

HELDOUT_ROOMS="dech_000001 dech_011110 reverb_largeroom2 ace_Building_Lobby diffrir_complex_complexBase"

mkdir -p "$SPLIT"

python3 - "$MF" "$SPLIT" $HELDOUT_ROOMS <<'EOF'
import json, sys
from pathlib import Path

mf_dir, split_dir, *heldout = sys.argv[1:]
heldout = set(heldout)
split_dir = Path(split_dir)

pools = {"train": [], "heldout": []}
for mf in sorted(Path(mf_dir).glob("*.jsonl")):
    rows = {"train": [], "heldout": []}
    for line in mf.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        part = "heldout" if e["room_id"] in heldout else "train"
        rows[part].append(e)
        if e["distance_m"] < 1.0:
            pools[part].append(e)
    for part, rr in rows.items():
        out = split_dir / f"{mf.stem}.{part}.jsonl"
        out.write_text("".join(json.dumps(e) + "\n" for e in rr))
        print(f"{out.name}: {len(rr)} entries")
for part, rr in pools.items():
    out = split_dir / f"near_pool.{part}.jsonl"
    out.write_text("".join(json.dumps(e) + "\n" for e in rr))
    print(f"{out.name}: {len(rr)} near (<1m) entries")
EOF

# items_per_room per corpus, scaled to unique-RIR counts (heavier reuse is
# pointless for tiny corpora like slr28/ace). Item count only steers each
# corpus' sampling share inside the bank -- RIR diversity is fixed anyway.
# Train total ~27k -> ~22% real rows after merging with wide (95,344).
declare -A IPR=( [dechorate]=1200 [brudex]=1200 [diffrir]=1000 [slr28]=256 [ace]=128 )

build () {  # $1=split (train|heldout) $2=output_dir $3=seed $4=ipr_divisor
    local part="$1" out="$2" seed="$3" div="$4"
    rm -rf "$out"
    for corpus in dechorate brudex diffrir slr28 ace; do
        local mf="$SPLIT/$corpus.$part.jsonl"
        [ -s "$mf" ] || { echo "skip $corpus.$part (empty)"; continue; }
        uv run python "$ROOT/egs/rir_generation/real_rir_to_bank.py" from-manifest \
            --manifest "$mf" --output "$out" \
            --d0 1.0 --target-sr 16000 \
            --items-per-room "$(( ${IPR[$corpus]} / div ))" --seed "$seed" \
            --near-pool "$SPLIT/near_pool.$part.jsonl"
    done
    # flat items/ view (merge_rir_views.py input format)
    local view="${out}_view"
    rm -rf "$view" && mkdir -p "$view/items"
    ( cd "$view/items" && for d in "$out"/*/; do
        n=$(basename "$d")
        ln -s "../../$(basename "$out")/$n/$n.wav" "$n.wav"
        ln -s "../../$(basename "$out")/$n/$n.json" "$n.json"
    done )
    echo "== $out: $(ls "$out" | wc -l) items (view: $view/items)"
}

cd "$ROOT"
build train   "$EXP/real_rir_16k_train"   3141 1
build heldout "$EXP/real_rir_16k_heldout" 1618 4
