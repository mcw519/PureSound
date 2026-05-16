from argparse import Namespace

from puresound.dataset.parser import MetafileParser


def test_dns_adapter_scans_clean_speech_with_parent_speaker_ids(
    tmp_path, voice_isolate_dns_adapter, write_silent_wav
):
    clean_root = tmp_path / "datasets_fullband" / "clean_fullband"
    write_silent_wav(clean_root / "spk-a" / "utt-1.wav")
    write_silent_wav(clean_root / "spk-a" / "utt-2.wav")

    records = voice_isolate_dns_adapter.scan_clean_speech(
        clean_root, speaker_id_strategy="parent"
    )

    assert len(records) == 2
    assert {record.spkid for record in records} == {"dns4_spk-a"}
    assert all(record.sample_rate == 16000 for record in records)


def test_dns_adapter_splits_by_speaker(
    tmp_path, voice_isolate_dns_adapter, write_silent_wav
):
    clean_root = tmp_path / "clean"
    for speaker in ["spk-a", "spk-b", "spk-c"]:
        write_silent_wav(clean_root / speaker / "utt-1.wav")

    records = voice_isolate_dns_adapter.scan_clean_speech(clean_root)
    train_records, valid_records = voice_isolate_dns_adapter.split_records(
        records, valid_ratio=0.34, seed=0, split_by="speaker"
    )

    assert train_records
    assert valid_records
    assert {record.spkid for record in train_records}.isdisjoint(
        {record.spkid for record in valid_records}
    )


def test_dns_adapter_writes_puresound_metafiles_and_config_hint(
    tmp_path, voice_isolate_dns_adapter, write_silent_wav
):
    dns_root = tmp_path / "dns4"
    clean_root = dns_root / "datasets_fullband" / "clean_fullband"
    noise_root = dns_root / "datasets_fullband" / "noise_fullband"
    rir_root = dns_root / "datasets_fullband" / "impulse_responses"
    for speaker in ["spk-a", "spk-b"]:
        write_silent_wav(clean_root / speaker / "utt-1.wav")
        write_silent_wav(clean_root / speaker / "utt-2.wav")
    noise_root.mkdir(parents=True)
    rir_root.mkdir(parents=True)

    output_dir = tmp_path / "out"
    train_metafile, valid_metafile = voice_isolate_dns_adapter.prepare_dns_challenge_metafiles(
        Namespace(
            dns_root=str(dns_root),
            output_dir=str(output_dir),
            train_metafile=None,
            valid_metafile=None,
            clean_dir=None,
            noise_dir=None,
            rir_dir=None,
            audio_suffixes=".wav",
            speaker_id_strategy="parent",
            gender="other",
            utt_prefix="dns4",
            min_duration=0.0,
            max_files=None,
            valid_ratio=0.5,
            seed=0,
            split_by="speaker",
            skip_bad_files=True,
            write_config_hint=True,
        )
    )

    train_lines = train_metafile.read_text(encoding="utf-8").splitlines()
    valid_lines = valid_metafile.read_text(encoding="utf-8").splitlines()
    hint = (output_dir / "voice_isolate_dns4_config_hint.yaml").read_text(
        encoding="utf-8"
    )

    assert train_lines[0] == "uttid, spkid, gender, path, length, sample rate, channels"
    assert valid_lines[0] == "uttid, spkid, gender, path, length, sample rate, channels"
    assert len(train_lines) == 3
    assert len(valid_lines) == 3
    assert str(train_metafile) in hint
    assert str(noise_root) in hint
    assert str(rir_root) in hint

    parsed_train = MetafileParser.read_from_metafile(
        str(train_metafile), use_speaker_as_key=True
    )

    assert len(parsed_train) == 1
