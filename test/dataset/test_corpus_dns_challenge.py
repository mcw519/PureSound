"""DNS Challenge preparation: speakers out of paths, categories out of dev-set
filenames, and the files that must never reach training."""

import re

import pytest

from puresound.dataset.corpus.dns_challenge import (
    SUBSET_SPEAKER_PATTERNS,
    main,
    parse_devset_name,
    resolve_subdir,
)
from puresound.dataset.corpus.records import read_metafile


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ms_pns_HPdesktop_A1QUQ0TV9KVD4C_copy_machine_primarywithnoise_fileid_3.wav",
         {"device": "hpdesktop", "category": "copymachine"}),
        ("ms_emotional_Happy_pns_Lenevo_Laptop_A8B4AE3QMECV_traffic_noise_fileid_7.wav",
         {"device": "lenevo_laptop", "category": "traffic"}),
        ("ms_realrec_emotional_headset_A72LC42LU78IP_munching_far_sad_fileid_1.wav",
         {"device": "headset", "category": "munching"}),
        # No worker id at all -- the device model number is the anchor instead.
        ("ms_pns_ASUS_X205TA_clattering_primarywithnoise_fileid_1.wav",
         {"device": "asus", "category": "clatter"}),
        # Doubled extension, no `fileid` marker before the index.
        ("ms_emotional_pns_HPLaptop_A1PUWQYUQRGCO_typing_19.wav.wav",
         {"device": "hplaptop", "category": "typing"}),
        # `.` used as a separator instead of `_`.
        ("ms_pns_desktopwindows_headset_A7EP668Z0WF6G_Dog.barking_primarywithnoise_fileid_30.wav",
         {"device": "desktopwindows_headset", "category": "dogbarking"}),
        # Hand-labelled spellings of one source collapse to one category; the raw
        # spelling is kept beside it.
        ("ms_pns_dell_A1I63OL2TAXTGJ_fan_primarywithnoise_fileid_8.wav", {"category": "fan"}),
        ("ms_pns_dell_A1I63OL2TAXTGJ_fan_noise_primarywithnoise_fileid_8.wav", {"category": "fan"}),
        ("ms_pns_dell_A1I63OL2TAXTGJ_fannoice_primarywithnoise_fileid_8.wav", {"category": "fan"}),
        ("ms_pns_dell_A1I63OL2TAXTGJ_ceilingfan_primarywithnoise_fileid_8.wav",
         {"category": "fan", "category_raw": "ceilingfan"}),
        # An unrecognised category keeps its own spelling.
        ("ms_pns_dell_A1I63OL2TAXTGJ_didgeridoo_fileid_1.wav", {"category": "didgeridoo"}),
        # Nothing to anchor on: an honest gap rather than a guess at where the
        # device stops and the noise starts.
        ("some_recording.wav", {}),
    ],
)
def test_device_and_category_come_out_of_the_devset_filename(name, expected):
    parsed = parse_devset_name(name)
    if not expected:
        assert parsed == {}
    for key, value in expected.items():
        assert parsed[key] == value, key


@pytest.mark.parametrize(
    "subset, relative, speaker",
    [
        # The read_speech tree is flat, so the speaker is in the filename.
        ("read_speech", "book_00000_chp_0009_reader_06709_0_seg_0.wav", "06709"),
        ("french_speech", "M-AILABS_Speech_Dataset/fr_FR_190hrs_16k/female/ezwa/book/wavs/x_f000182.wav", "ezwa"),
        # A `mix` book names no reader, so the book stands in for one.
        ("italian_speech", "M-AILABS_Speech_Dataset/it_IT_128hrs_16k/mix/novelle_02/wavs/x_f000039.wav", "mix/novelle_02"),
        ("german_speech", "CC_BY_SA_4.0_249hrs_339spk_German_Wikipedia_16k/data/German_Wikipedia_Mikrofon_audio_48kHz_seg_33.wav", "Mikrofon"),
        # The corpus directory also contains "German_Wikipedia_"; only the filename counts.
        ("german_speech", "CC_BY_SA_4.0_249hrs_339spk_German_Wikipedia_16k/German_Wikipedia_Martin_Luther_audio2_48kHz_seg_157.wav", "Martin_Luther"),
        ("spanish_speech", "SLR39_48kHz/SLR39_100_11_48kHz.wav", "SLR39_100"),
        ("spanish_speech", "SLR39_5_337_48kHz_seg_0.wav", "SLR39_5"),
        ("spanish_speech", "SLR39_native-f-Mexico-190-45-62-NA-mgh8170_s120_48kHz.wav", "mgh8170"),
        ("spanish_speech", "SLR61_48kHz/SLR61_es_ar_female_arf_03397_01665251952_48kHz.wav", "arf_03397"),
        ("spanish_speech", "SLR61_es-es_esw_02485_00146903919_48kHz.wav", "esw_02485"),
        ("spanish_speech", "SLR71_es_cl_female_clf_00610_00103371024_48kHz_seg_0.wav", "clf_00610"),
        ("emotional_speech", "crema_d/1025_IWL_SAD_XX.wav", "1025"),
        ("VocalSet_48kHz_mono", "vocalset_female7_scales_fast_forte_f7_scales_f_fast_forte_i_48kHz.wav", "female7"),
    ],
)
def test_every_bundled_corpus_names_its_speaker(subset, relative, speaker):
    patterns = SUBSET_SPEAKER_PATTERNS[subset]
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    match = next((m for m in (re.search(p, relative) for p in patterns) if m), None)
    assert match is not None
    assert next(g for g in match.groups() if g is not None) == speaker


def test_resolve_subdir_prefers_an_explicit_override(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "clean_fullband").mkdir()

    assert resolve_subdir(tmp_path, "elsewhere", ("clean_fullband",)) == tmp_path / "elsewhere"
    assert resolve_subdir(tmp_path, None, ("clean_fullband",)) == tmp_path / "clean_fullband"
    assert resolve_subdir(tmp_path, None, ("missing",)) is None


def test_the_voicebank_demand_test_speakers_never_enter_training(tmp_path, write_tone_wav):
    clean = tmp_path / "clean_fullband" / "vctk_wav48_silence_trimmed"
    for speaker in ("p225", "p226", "p227", "p232", "p257"):
        for index in range(2):
            write_tone_wav(clean / speaker / f"{speaker}_{index:03d}_mic1.wav")

    assert main([
        "speech", str(tmp_path), "--clean-dir", "clean_fullband",
        "--subset", "vctk_wav48_silence_trimmed", "--id-prefix", "dnsvctk",
        "--output-dir", str(tmp_path / "out"), "--valid-ratio", "0",
    ]) == 0
    speakers = {r.spkid for r in read_metafile(tmp_path / "out" / "dnsvctk_train.csv")}
    assert speakers == {"dnsvctk_p225", "dnsvctk_p226", "dnsvctk_p227"}


def test_a_second_copy_of_the_tree_is_dropped_not_counted_twice(tmp_path, write_tone_wav):
    clean = tmp_path / "clean_fullband" / "read_speech"
    for reader in ("0001", "0002"):
        name = f"book_00_chp_0_reader_{reader}_0_seg_0.wav"
        write_tone_wav(clean / name)
        write_tone_wav(clean / "read_speech" / name)  # the shipped nested copy

    assert main([
        "speech", str(tmp_path), "--clean-dir", "clean_fullband",
        "--subset", "read_speech", "--output-dir", str(tmp_path / "out"),
        "--valid-ratio", "0",
    ]) == 0
    assert len(read_metafile(tmp_path / "out" / "dns5_train.csv")) == 2
