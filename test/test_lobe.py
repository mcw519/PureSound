import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.nnet.lobe.dsp import FrequencyEQLayer
from puresound.nnet.lobe.encoder import ConvEncDec, UnifiedConvEncDec
from puresound.nnet.lobe.rnn import FSMN, ConditionFSMN
from puresound.nnet.lobe.trivial import SplitMerge
from puresound.utils import create_folder

sys.path.insert(0, "./")

TEST_CASE_DIR = Path(__file__).resolve().parent / "test_case"
TEST_AUDIO_PATH = str(TEST_CASE_DIR / "1272-141231-0008.flac")
OUT_TEST_FOLDER = str(TEST_CASE_DIR / "outputs")
SAVE_TEST_AUDIO = True

create_folder(OUT_TEST_FOLDER)


def align_and_stack(wav1: torch.Tensor, wav2: torch.Tensor):
    if wav1.shape[-1] > wav2.shape[-1]:
        wav1 = wav1[..., : wav2.shape[-1]]
    else:
        wav2 = wav2[..., : wav1.shape[-1]]

    return torch.cat([wav1, wav2], dim=0)


@pytest.mark.nnet
@pytest.mark.parametrize("l_ctx, r_ctx", [(3, 3), (3, 0)])
def test_fsmn_block(l_ctx, r_ctx):
    input_x = torch.rand(3, 256, 100)
    memory = torch.rand(3, 192, 100)
    fsmn_block = FSMN(256, 256, 192, l_ctx, r_ctx)

    with torch.no_grad():
        output_x, memory = fsmn_block(input_x, memory)

    assert input_x.shape[-1] == output_x.shape[-1] == memory.shape[-1]

    input_x[..., 50:] = np.inf
    fsmn_block = fsmn_block.eval()
    with torch.no_grad():
        output_x, memory = fsmn_block(input_x, memory)

    if r_ctx == 0:
        assert np.where(np.isnan(output_x))[-1][0] == 50


@pytest.mark.nnet
def test_conditional_fsmn_block():
    input_x = torch.rand(3, 256, 100)
    memory = torch.rand(3, 128, 100)
    dvec = torch.rand(3, 192)
    fsmn_block1 = ConditionFSMN(256, 256, 128, 192, 3, 3)
    fsmn_block2 = ConditionFSMN(256, 256, 128, 192, 3, 3, use_film=True)

    with torch.no_grad():
        output_x1, memory1 = fsmn_block1(input_x, dvec, memory)
        output_x2, memory2 = fsmn_block2(input_x, dvec, memory)

    assert input_x.shape[-1] == output_x1.shape[-1] == memory1.shape[-1]
    assert input_x.shape[-1] == output_x2.shape[-1] == memory2.shape[-1]


@pytest.mark.nnet
def test_split_and_merge():
    input_x = torch.rand(3, 256, 1000)
    split_x, rest = SplitMerge.split(input_x, 40)
    merge_x = SplitMerge.merge(split_x, rest)
    assert torch.allclose(input_x, merge_x)


@pytest.mark.nnet
def test_freq_peq_layer():
    Fpeq = FrequencyEQLayer()
    input_x = torch.rand(1, 2, 257, 100)
    output_x = Fpeq(input_x)
    output_x.sum().backward()


@pytest.mark.nnet
@pytest.mark.parametrize(
    "n_fft, hop_length, win_type, trainable",
    [[512, 128, "hann", False], [1024, 160, "hamming", True]],
)
def test_trainable_stft_layer(n_fft, hop_length, win_type, trainable):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH,
        normalized=False,
        target_lvl=None,
        verbose=True,
        resample_to=16000,
    )
    encoder = ConvEncDec(
        fft_length=n_fft,
        win_type=win_type,
        win_length=n_fft,
        sr=sr,
        fmin=0,
        fmax=8000,
        freq_scale="no",
        trainable=trainable,
    )
    stft = encoder(wav)
    reconstructed_wav = encoder.inverse(stft)
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=reconstructed_wav),
            f_path=f"{OUT_TEST_FOLDER}/stft_encdec_fft={n_fft}_hop={hop_length}_win={win_type}.wav",
            sr=sr,
        )


@pytest.mark.nnet
@pytest.mark.parametrize(
    "sr",
    [8000, 16000, 22050, torch.Tensor([24000]), torch.Tensor([32000]), 44100, 48000],
)
def test_unified_stft_encoder(sr):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH,
        normalized=False,
        target_lvl=None,
        verbose=True,
        resample_to=sr if isinstance(sr, int) else int(sr.item()),
    )
    encoder = UnifiedConvEncDec(win_type="hann", trainable=False)
    stft = encoder(wav, sr)
    reconstructed_wav = encoder.inverse(stft, sr)
    if SAVE_TEST_AUDIO:
        if isinstance(sr, torch.Tensor):
            sr = int(sr.item())
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=reconstructed_wav),
            f_path=f"{OUT_TEST_FOLDER}/unified_stft_encdec_sr={sr}.wav",
            sr=sr,
        )
