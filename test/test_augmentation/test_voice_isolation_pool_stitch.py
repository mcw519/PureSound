"""augmentation_real{far,near}.stitch_to_length -- the knob that keeps a long row's
real-recording target from being half digital silence.

A pool take is ~16 s. Ask for a 30 s row without stitching and the aligner zero-pads
the rest; on a real-near KEEP row that pad IS the training target. Measured on the
recipe: 30 s rows carry 0.472 median digital-silence fraction unstitched and 0.000
stitched, against 0.000 for the 6 s recipe (turn-taking off in all three).
"""
import torch

from puresound.task.voice_isolation import VoiceIsolationDataset


class _Stub:
    """Only the pieces _open_pool_wav touches -- the dataset itself needs a corpus.

    The methods are borrowed unbound rather than inherited: the real class exposes
    audio_sr / sample_length as read-only properties fed by a recipe.
    """

    _channel_key = staticmethod(VoiceIsolationDataset._channel_key)
    _channel_index = VoiceIsolationDataset._channel_index
    _open_pool_wav = VoiceIsolationDataset._open_pool_wav

    def __init__(self, tmp_path, take_seconds=4.0, n_takes=6, row_seconds=10.0):
        import soundfile as sf
        self.audio_sr = 16000
        self.audio_gain_normalized_to = None
        self.sample_length = int(16000 * row_seconds)
        self.pool = []
        for i in range(n_takes):
            path = tmp_path / f"take{i}.wav"
            sf.write(str(path), (torch.rand(int(16000 * take_seconds)) - 0.5).numpy(), 16000)
            self.pool.append({"wav_path": str(path), "speaker": "sp1",
                              "room": "rm1", "mic": 1, "distance_m": 0.9})
        self.index = self._channel_index(self.pool)


def test_without_stitch_one_take_comes_back_short(tmp_path):
    s = _Stub(tmp_path)
    wav = s._open_pool_wav(s.pool[0], s.index, stitch=False)
    assert wav.shape[-1] == int(16000 * 4.0)          # the aligner would pad the rest


def test_stitch_covers_the_row_length(tmp_path):
    s = _Stub(tmp_path)
    wav = s._open_pool_wav(s.pool[0], s.index, stitch=True)
    assert wav.shape[-1] >= s.sample_length
    assert torch.isfinite(wav).all()


def test_stitch_only_uses_the_same_channel(tmp_path):
    """A different (speaker, room, mic) is a different chain: never appended."""
    s = _Stub(tmp_path)
    for entry in s.pool[1:]:
        entry["mic"] = 2                                # same speaker, other mic
    s.index = s._channel_index(s.pool)
    wav = s._open_pool_wav(s.pool[0], s.index, stitch=True)
    assert wav.shape[-1] == int(16000 * 4.0)            # nothing eligible to append


def test_stitch_is_a_noop_when_the_take_already_covers_the_row(tmp_path):
    s = _Stub(tmp_path, take_seconds=12.0, row_seconds=10.0)
    a = s._open_pool_wav(s.pool[0], s.index, stitch=False)
    b = s._open_pool_wav(s.pool[0], s.index, stitch=True)
    assert torch.equal(a, b)
