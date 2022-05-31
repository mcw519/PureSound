import io
import os
from typing import Dict, List, Optional
import torch
import yaml


def str2bool(v: str):
    """convert string to boolean"""
    return v.lower() in ('true', 'yes')


def str2list(s: str) -> List:
    """convert string to list by space"""
    return s.strip().split()


def load_text_as_dict(file_path: str, separator: str = ' ', coding: str = 'utf8') -> Dict:
    """
    Load a text file to dict format. The first column would be the Dict's key for each line.
    Most usage this in read a *.scp file.
    Ex:
        input file:
            aaa bbb
            uttid /Path/data/file/abc.wav
        return:
            {'aaa': 'bbb', 'uttid': '/Path/data/file/abc.wav'}
    
    Args:
        file_path: input file path
        separator: symbol to split in lines
        coding: encoding type for open text file, default as utf8
    
    Returns:
        return a dict which used first column as keys.
    """
    dct = {}
    with io.open(file_path, 'r', encoding=coding) as f:
        for line in f.readlines():
            key = line.strip().split(separator)[0]
            content = line.strip().split(separator)[1:]
            assert isinstance(content, list)
            dct[key] = content
    
    return dct


def recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None:
    """
    Recursive reading a folder, and list related file path in the output list.
    Ex:
        _list = []
        recursive_read_folder('corpus', '.flac', _list)
    
    Args:
        folder: input folder path
        file_type: suffix for parsing file
        output: storage parser result
    """
    for file in os.listdir(folder):
        cur_path = os.path.join(folder, file)
        if os.path.isdir(cur_path):
            recursive_read_folder(cur_path, file_type, output)
        
        else:
            if file_type in file:
                output.append(f"{file} {cur_path}")


def load_hparam(filename: str) -> Dict:
    """
    Loading configuration file in a dict.

    Args:
        filename: configure file (*.yaml) path
    
    Returns:
        hparam in dict form
    """
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.FullLoader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v

    return hparam_dict


def create_folder(folder_name: str) -> None:
    """
    If folder not exist, create it

    Args:
        folder_name: folder path string

    Raises:
        FileExistsError: if folder not exist
    """
    try:
        if not os.path.isdir(folder_name): os.makedirs(folder_name, exist_ok=True)
    except FileExistsError:
        print(f"File exists passing it: {folder_name}")


def convolve(x: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    """Doing convolution on x with filter"""
    # padding = 1 * (filter.shape[-1] -1) / 2
    weight = filter.float().repeat(1, 1, 1)
    x = torch.nn.functional.pad(x, (filter.shape[-1]-1, 0))
    x = torch.nn.functional.conv1d(x[None, ...], weight)

    return x.view(1, -1)


_NEXT_FAST_LEN = {}
def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    Note: This function was originally copied from the https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def fftconvolve(x: torch.Tensor, kernel: torch.Tensor, mode: str = 'full'):
    """
    Asteroid implemented FFT convolution.

    Usage:
        wav_rverb = fftconvolve2(wav, rir, mode='full')
        # Convolving audio with a RIR normally introduces a bit of delay, especially when the peak absolute amplitude in the RIR is not in the very beginning.
        propagation_delays = rir.abs().argmax(dim=-1, keepdim=False)[0]
        wav_rverb = wav_rverb[..., propagation_delays:propagation_delays+wav.shape[-1]]
    """
    m = x.shape[-1]
    n = kernel.shape[-1]
    if mode == "full":
        truncate = m + n - 1
    elif mode == "valid":
        truncate = max(m, n) - min(m, n) + 1
    elif mode == "same":
        truncate = max(m, n)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    # Compute convolution using fft.
    padded_size = m + n - 1

    # Round up for cheaper fft.
    fast_fft_size = next_fast_len(padded_size)
    f_signal = torch.fft.rfft(x, n=fast_fft_size)
    f_kernel = torch.fft.rfft(kernel, n=fast_fft_size)
    f_result = f_signal * f_kernel
    result = torch.fft.irfft(f_result, n=fast_fft_size)
    
    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx : start_idx + truncate]
