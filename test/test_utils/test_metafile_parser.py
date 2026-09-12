from puresound.dataset.parser import MetafileParser


def test_metafile_parser_keeps_first_data_row_without_header(tmp_path):
    metafile = tmp_path / "meta.csv"
    metafile.write_text(
        "\n".join(
            [
                "utt0, corpus_spk0, m, /tmp/utt0.wav, 16000, 16000, 1",
                "utt1, corpus_spk0, m, /tmp/utt1.wav, 32000, 16000, 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    meta = MetafileParser.read_from_metafile(str(metafile), use_speaker_as_key=True)

    assert sorted(meta["corpus_spk0"]["utts"]) == ["utt0", "utt1"]


def test_metafile_parser_skips_embedded_header_rows(tmp_path):
    metafile = tmp_path / "meta.csv"
    metafile.write_text(
        "\n".join(
            [
                "uttid, spkid, gender, path, length, sample rate, channels",
                "utt0, corpus_spk0, m, /tmp/utt0.wav, 16000, 16000, 1",
                "uttid, spkid, gender, path, length, sample rate, channels",
                "utt1, corpus_spk0, m, /tmp/utt1.wav, 32000, 16000, 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    meta = MetafileParser.read_from_metafile(str(metafile), use_speaker_as_key=True)

    assert sorted(meta["corpus_spk0"]["utts"]) == ["utt0", "utt1"]
