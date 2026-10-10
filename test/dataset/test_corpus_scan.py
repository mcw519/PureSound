"""Scanning a corpus, and splitting it without the dev set leaking into training."""

import pytest

from puresound.dataset.corpus import (
    assert_disjoint,
    drop_duplicate_files,
    parse_suffixes,
    sanitize_id,
    scan_folder,
    split_records,
)

READER = r"reader_(\d+)"


@pytest.fixture
def flat_corpus(tmp_path, write_tone_wav):
    """One directory, speaker encoded in the filename -- the DNS read_speech shape."""
    root = tmp_path / "read_speech"
    for reader in ("0001", "0002", "0003", "0004"):
        for segment in range(3):
            write_tone_wav(root / f"book_00_chp_0_reader_{reader}_0_seg_{segment}.wav")
    return root


@pytest.fixture
def nested_corpus(tmp_path, write_tone_wav):
    root = tmp_path / "vctk"
    for speaker in ("p225", "p226", "p227"):
        for index in range(2):
            write_tone_wav(root / speaker / f"{speaker}_{index}.wav")
    return root


def _with_clashing_names(nested_corpus, write_tone_wav):
    # Same basename in two speaker directories -- unique by path, identical by stem.
    write_tone_wav(nested_corpus / "p225" / "clash.wav")
    write_tone_wav(nested_corpus / "p226" / "clash.wav")
    return nested_corpus


@pytest.mark.parametrize(
    "tree, kwargs, speakers",
    [
        ("flat", {"id_prefix": "dns5", "speaker_pattern": READER},
         {"dns5_0001", "dns5_0002", "dns5_0003", "dns5_0004"}),
        ("nested", {"speaker_id_strategy": "parent"}, {"p225", "p226", "p227"}),
        ("flat", {"speaker_id_strategy": "per-file"}, None),  # one speaker per file, on request
    ],
    ids=["pattern-on-a-flat-tree", "directory-per-speaker", "per-file"],
)
def test_each_speaker_strategy_recovers_the_speakers_it_describes(request, tree, kwargs, speakers):
    root = request.getfixturevalue(f"{tree}_corpus")
    records = scan_folder(root, tagger=lambda path: {"stem": path.stem[:4]}, **kwargs)

    if speakers is None:
        assert len({record.spkid for record in records}) == len(records)
    else:
        assert {record.spkid for record in records} == speakers
    assert all(record.tags == {"stem": record.path.stem[:4]} for record in records)
    # `DynamicBaseDataset` buckets anything that is not m/f as "other"; the repo's
    # metafiles spell that unknown as the literal "None".
    assert {record.gender for record in records} == {"None"}


def test_several_patterns_are_tried_in_order(tmp_path, write_tone_wav):
    root = tmp_path / "bundle"
    write_tone_wav(root / "M-AILABS" / "female" / "eva" / "book" / "a.wav")
    write_tone_wav(root / "German_Wikipedia_Dresden_audio2_seg_1.wav")

    records = scan_folder(
        root,
        speaker_pattern=(r"M-AILABS/(?:fe)?male/([^/]+)/", r"German_Wikipedia_(.+?)_audio"),
    )

    assert {record.spkid for record in records} == {"eva", "Dresden"}


@pytest.mark.parametrize(
    "call, match",
    [
        # Better than silently making every utterance its own speaker.
        (lambda flat, nested, w: scan_folder(flat, speaker_id_strategy="parent"), "directory level"),
        (lambda flat, nested, w: scan_folder(nested, speaker_pattern=READER), "did not match"),
        (lambda flat, nested, w: scan_folder(flat, speaker_pattern=READER, utt_id_style="uuid"),
         "utt_id_style"),
        (lambda flat, nested, w: scan_folder(_with_clashing_names(nested, w),
                                             speaker_id_strategy="parent", utt_id_style="stem"),
         "Duplicate utterance id"),
        (lambda flat, nested, w: split_records(scan_folder(nested, speaker_pattern=r"(p)\d+"),
                                               valid_ratio=0.25, split_by="speaker"),
         "single speaker"),
    ],
    ids=["flat-tree-directory-strategy", "pattern-misses-a-file", "unknown-id-style",
         "stem-ids-collide", "single-speaker-split"],
)
def test_an_ambiguous_scan_or_split_is_refused(flat_corpus, nested_corpus, write_tone_wav, call, match):
    with pytest.raises(ValueError, match=match):
        call(flat_corpus, nested_corpus, write_tone_wav)


def test_digest_ids_are_unique_where_names_collide_but_hide_the_corpus_naming(
    flat_corpus, nested_corpus, write_tone_wav
):
    records = scan_folder(flat_corpus, id_prefix="dns5", speaker_pattern=READER)
    assert all(record.uttid.startswith("dns5_book_") for record in records)
    # The hash lands after the segment index, so `..._seg_0` is no longer parseable.
    assert not any(record.uttid.endswith("_seg_0") for record in records)

    clashing = scan_folder(_with_clashing_names(nested_corpus, write_tone_wav),
                           speaker_id_strategy="parent")
    assert len({record.uttid for record in clashing}) == len(clashing)


def test_stem_style_keeps_the_corpus_naming_parseable(flat_corpus):
    """`build_chapter_corpus.py` reads the segment index out of the id."""
    records = scan_folder(flat_corpus, id_prefix="dns5", speaker_pattern=READER, utt_id_style="stem")

    assert all(record.uttid.startswith("book_") for record in records)
    segments = {int(record.uttid.rpartition("_seg_")[2]) for record in records}
    assert segments == {0, 1, 2}


def test_max_files_caps_the_scan_deterministically_and_min_duration_drops_short_files(
    flat_corpus, write_tone_wav
):
    first = scan_folder(flat_corpus, speaker_pattern=READER, max_files=5)
    second = scan_folder(flat_corpus, speaker_pattern=READER, max_files=5)
    assert [record.uttid for record in first] == [record.uttid for record in second]
    assert len(first) == 5

    write_tone_wav(flat_corpus / "book_00_chp_0_reader_0009_0_seg_0.wav", duration=0.05)
    records = scan_folder(flat_corpus, speaker_pattern=READER, min_duration=0.2)
    assert records and all("0009" not in record.spkid for record in records)


def test_a_speaker_split_is_disjoint_and_an_utterance_split_is_not(flat_corpus):
    records = scan_folder(flat_corpus, speaker_pattern=READER)
    train, valid = split_records(records, valid_ratio=0.25, split_by="speaker")
    assert train and valid
    assert len(train) + len(valid) == len(records)
    assert_disjoint(train, valid, by="spkid")

    train, valid = split_records(records, valid_ratio=0.25, split_by="utterance")
    with pytest.raises(AssertionError, match="share"):
        assert_disjoint(train, valid, by="spkid")


def test_the_split_is_fixed_by_its_seed_and_a_zero_ratio_keeps_everything(flat_corpus):
    records = scan_folder(flat_corpus, speaker_pattern=READER)
    first, _ = split_records(records, valid_ratio=0.25, seed=7)
    second, _ = split_records(records, valid_ratio=0.25, seed=7)
    assert [record.uttid for record in first] == [record.uttid for record in second]

    train, valid = split_records(records, valid_ratio=0.0)
    assert len(train) == len(records) and valid == []


def test_sanitize_and_suffix_helpers():
    assert sanitize_id("a b/c!!") == "a_b_c"
    assert sanitize_id("///") == "unknown"
    assert parse_suffixes("wav, .flac") == (".wav", ".flac")
    assert parse_suffixes("") == (".wav", ".flac")


def test_drop_duplicate_files_keeps_one_copy_per_name_and_size(tmp_path, write_tone_wav):
    root = tmp_path / "tree"
    write_tone_wav(root / "a.wav")
    write_tone_wav(root / "copy" / "a.wav")
    # Same name, different recording: not a duplicate.
    write_tone_wav(root / "other" / "b.wav", duration=0.5)
    write_tone_wav(root / "b.wav", duration=0.25)
    records = scan_folder(root, speaker_id_strategy="per-file", utt_id_style="digest")

    kept, dropped = drop_duplicate_files(records)

    assert len(kept) == 3 and len(dropped) == 1
    assert dropped[0].path.name == "a.wav"
