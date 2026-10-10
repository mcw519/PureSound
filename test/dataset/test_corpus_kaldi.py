"""One Kaldi conversion for every recipe, with one behaviour on a bad file."""

import pytest

from puresound.dataset.corpus import read_metafile
from puresound.dataset.corpus.kaldi import convert_kaldi_metafile


@pytest.fixture
def kaldi_lists(tmp_path, write_tone_wav):
    audio = tmp_path / "audio"
    for index in range(3):
        write_tone_wav(audio / f"utt{index}.wav")

    (tmp_path / "wav2scp.txt").write_text(
        "".join(f"utt{index} {audio / f'utt{index}.wav'}\n" for index in range(3)),
        encoding="utf-8",
    )
    (tmp_path / "utt2spk.txt").write_text(
        "utt0 spkA\nutt1 spkA\nutt2 spkB\n", encoding="utf-8"
    )
    (tmp_path / "utt2gender.txt").write_text("utt0 m\nutt1 m\n", encoding="utf-8")
    return tmp_path


def _convert(lists, output, **kwargs):
    return convert_kaldi_metafile(
        output, lists / "wav2scp.txt", lists / "utt2spk.txt", progress=False, **kwargs
    )


@pytest.mark.parametrize(
    "utt2spk, with_gender, uttids, speakers, genders",
    [
        (None, False, ["utt0", "utt1", "utt2"], ["spkA", "spkA", "spkB"], {"None"}),
        # A gender list filters to the utterances it covers.
        (None, True, ["utt0", "utt1"], ["spkA", "spkA"], {"m"}),
        # An utterance with no speaker is dropped.
        ("utt0 spkA\n", False, ["utt0"], ["spkA"], {"None"}),
    ],
    ids=["plain", "gender-list", "missing-speaker"],
)
def test_conversion_writes_the_utterances_every_list_covers(
    kaldi_lists, tmp_path, utt2spk, with_gender, uttids, speakers, genders
):
    if utt2spk is not None:
        (kaldi_lists / "utt2spk.txt").write_text(utt2spk, encoding="utf-8")
    output = tmp_path / "train.csv"
    kwargs = {"utt2gender_path": kaldi_lists / "utt2gender.txt"} if with_gender else {}

    assert _convert(kaldi_lists, output, **kwargs) == len(uttids)
    records = read_metafile(output)
    assert [record.uttid for record in records] == uttids
    assert [record.spkid for record in records] == speakers
    assert {record.gender for record in records} == genders


def test_insert_root_path_prefixes_relative_entries(tmp_path, write_tone_wav):
    audio = tmp_path / "corpus"
    write_tone_wav(audio / "utt0.wav")
    (tmp_path / "wav2scp.txt").write_text("utt0 utt0.wav\n", encoding="utf-8")
    (tmp_path / "utt2spk.txt").write_text("utt0 spkA\n", encoding="utf-8")

    output = tmp_path / "train.csv"
    _convert(tmp_path, output, insert_root_path=str(audio))
    assert read_metafile(output)[0].path == audio / "utt0.wav"


def test_an_unreadable_file_is_skipped_by_default_raised_on_demand_and_the_mode_is_checked(
    kaldi_lists, tmp_path
):
    (kaldi_lists / "audio" / "utt1.wav").write_bytes(b"not audio")
    output = tmp_path / "train.csv"

    assert _convert(kaldi_lists, output) == 2
    with pytest.raises(Exception):
        _convert(kaldi_lists, output, on_error="raise")
    with pytest.raises(ValueError, match="on_error"):
        convert_kaldi_metafile(
            output, kaldi_lists / "wav2scp.txt", kaldi_lists / "utt2spk.txt", on_error="ignore"
        )
