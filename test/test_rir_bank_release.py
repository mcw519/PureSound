import numpy as np
import soundfile as sf

from puresound.audio.rir_bank_manifest import (
    canonicalize_float_wav_header,
    sha256_file,
)


def _set_peak_timestamp(path, timestamp):
    payload = bytearray(path.read_bytes())
    offset = payload.index(b"PEAK")
    payload[offset + 12 : offset + 16] = int(timestamp).to_bytes(4, "little")
    path.write_bytes(payload)


def test_float_wav_peak_timestamp_is_canonicalized(tmp_path):
    samples = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, samples, 16000, subtype="FLOAT")
    sf.write(second, samples, 16000, subtype="FLOAT")
    _set_peak_timestamp(first, 1)
    _set_peak_timestamp(second, 4_000_000_000)

    assert sha256_file(first) != sha256_file(second)
    canonicalize_float_wav_header(first)
    canonicalize_float_wav_header(second)

    assert sha256_file(first) == sha256_file(second)
    first_audio, first_sr = sf.read(first, dtype="float32")
    second_audio, second_sr = sf.read(second, dtype="float32")
    assert first_sr == second_sr == 16000
    np.testing.assert_array_equal(first_audio, second_audio)
