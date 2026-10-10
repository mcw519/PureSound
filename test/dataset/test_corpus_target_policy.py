"""Which target a file trains on: cleaned only where there was noise to clean."""
from pathlib import Path

import pytest

from puresound.dataset.corpus.records import AudioRecord
from puresound.dataset.corpus.target_policy import choose


def _rec(uttid, root):
    return AudioRecord(uttid=uttid, spkid="s", gender="None", path=Path(root) / f"{uttid}.wav",
                       length=16000, sample_rate=16000, channels=1)


ORIGINAL = [_rec(u, "/orig") for u in ("a", "b", "c")]
CLEANED = [_rec(u, "/clean") for u in ("a", "b")]  # "c" never got cleaned
STATS = {"a": {"floor_in_db": -50.0, "floor_out_db": -58.0}, "b": {"floor_in_db": -60.0, "floor_out_db": -60.5}}


@pytest.mark.parametrize(
    "policy, roots",
    [
        ("floor-drop", ["/clean", "/orig", "/orig"]),  # only where the floor came down
        ("all", ["/clean", "/clean", "/orig"]),
        ("none", ["/orig", "/orig", "/orig"]),
    ],
)
def test_a_cleaned_target_replaces_the_original_per_policy_and_nothing_is_lost(policy, roots):
    records, replaced = choose(ORIGINAL, CLEANED, STATS, policy=policy, min_floor_drop=3.0)
    assert [r.uttid for r in records] == ["a", "b", "c"]
    assert [str(r.path.parent) for r in records] == roots
    assert replaced == roots.count("/clean")


def test_an_unknown_policy_is_an_error():
    with pytest.raises(ValueError):
        choose(ORIGINAL, CLEANED, STATS, policy="some")
