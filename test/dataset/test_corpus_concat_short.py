"""Short utterances are joined per speaker; long ones pass through untouched."""
from pathlib import Path

from puresound.dataset.corpus.concat_short import main, plan_pieces
from puresound.dataset.corpus.records import AudioRecord, read_metafile, write_metafile


def _rec(uttid, spk, seconds, path="/x.wav"):
    return AudioRecord(uttid=uttid, spkid=spk, gender="None", path=Path(path),
                       length=int(seconds * 16000), sample_rate=16000, channels=1)


def test_short_utterances_join_until_the_minimum_and_never_across_speakers():
    records = [_rec("a1", "a", 2.0), _rec("a2", "a", 2.0), _rec("a3", "a", 2.0), _rec("a4", "a", 6.0),
               _rec("b1", "b", 1.0)]
    groups = plan_pieces(records, min_seconds=4.0, max_seconds=12.0, gap=0.25)
    assert sorted([r.uttid for r in g] for g in groups) == [["a1", "a2"], ["a3"], ["a4"], ["b1"]]


def test_the_cli_writes_joined_audio_of_the_right_length(tmp_path, write_tone_wav):
    src = tmp_path / "src"
    records = []
    for i in range(3):
        path = src / "spk" / f"u{i}.wav"
        write_tone_wav(path, duration=2.0)
        records.append(_rec(f"u{i}", "spk", 2.0, str(path)))
    meta = tmp_path / "m.csv"
    write_metafile(meta, records)
    out = tmp_path / "o.csv"
    assert main([str(meta), "--source-root", str(src), "--dest-root", str(tmp_path / "dst"),
                 "--out", str(out), "--min-seconds", "4.0", "--jobs", "1"]) == 0
    rows = read_metafile(out)
    joined = [r for r in rows if "cat" in r.uttid]
    assert len(joined) == 1 and joined[0].length == int(16000 * 4.25) and joined[0].path.is_file()


def test_a_rerun_with_another_gap_describes_the_file_that_is_on_disk(tmp_path, write_tone_wav):
    src = tmp_path / "src"
    records = []
    for i in range(2):
        path = src / "spk" / f"u{i}.wav"
        write_tone_wav(path, duration=2.0)
        records.append(_rec(f"u{i}", "spk", 2.0, str(path)))
    meta = tmp_path / "m.csv"
    write_metafile(meta, records)

    def run(gap):
        out = tmp_path / f"o{gap}.csv"
        assert main([str(meta), "--source-root", str(src), "--dest-root", str(tmp_path / "dst"),
                     "--out", str(out), "--min-seconds", "4.0", "--gap", str(gap), "--jobs", "1"]) == 0
        return [r for r in read_metafile(out) if "cat" in r.uttid][0]

    first, second = run(0.25), run(1.0)

    assert first.path == second.path
    assert second.length == first.length == int(16000 * 4.25)
