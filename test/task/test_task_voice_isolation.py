"""``stitch_to_length`` on the real-recording pools of the voice-isolation dataset.

A pool take is shorter than a long row. Without stitching the aligner zero-pads
the rest, and on a real-near KEEP row that pad IS the training target; stitching
appends further takes of the same recording chain until the row is covered.
"""
import pytest
import soundfile as sf
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

    def __init__(self, tmp_path, take_seconds, row_seconds=10.0, n_takes=6, other_mic=False):
        self.audio_sr = 16000
        self.audio_gain_normalized_to = None
        self.sample_length = int(16000 * row_seconds)
        self.pool = []
        for i in range(n_takes):
            path = tmp_path / f"take{i}.wav"
            sf.write(str(path), (torch.rand(int(16000 * take_seconds)) - 0.5).numpy(), 16000)
            # A different (speaker, room, mic) is a different chain: never appended.
            mic = 2 if (other_mic and i) else 1
            self.pool.append({"wav_path": str(path), "speaker": "sp1",
                              "room": "rm1", "mic": mic, "distance_m": 0.9})
        self.index = self._channel_index(self.pool)


@pytest.mark.parametrize(
    "take_seconds, other_mic, stitch, covers_row",
    [(4.0, False, False, False), (4.0, False, True, True), (4.0, True, True, False),
     (12.0, False, True, True)],
    ids=["unstitched-take-is-short", "stitched-covers-the-row", "other-chain-never-appended",
         "long-take-is-untouched"],
)
def test_stitching_covers_the_row_from_the_same_chain_only(tmp_path, take_seconds, other_mic,
                                                           stitch, covers_row):
    stub = _Stub(tmp_path, take_seconds, other_mic=other_mic)
    wav = stub._open_pool_wav(stub.pool[0], stub.index, stitch=stitch)
    assert torch.isfinite(wav).all()
    if covers_row:
        assert wav.shape[-1] >= stub.sample_length
    else:
        assert wav.shape[-1] == int(16000 * take_seconds)
    if take_seconds * 16000 >= stub.sample_length:
        assert torch.equal(wav, stub._open_pool_wav(stub.pool[0], stub.index, stitch=False))
