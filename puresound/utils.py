import logging
import io
import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import yaml


logger = logging.getLogger(__name__)


def str2bool(v: str):
    """convert string to boolean"""
    return v.lower() in ("true", "yes")


def str2list(s: str) -> List:
    """convert string to list by space"""
    return s.strip().split()


def load_text_as_dict(
    file_path: str, separator: str = " ", coding: str = "utf8"
) -> Dict:
    """
    Load a text file to dict format. The first column would be the Dict's key for each line.
    Most usage this in read a *.scp file.
    Ex:
        input file:
            aaa bbb
            uttid /Path/data/file/abc.wav
        return:
            {'aaa': ['bbb'], 'uttid': ['/Path/data/file/abc.wav']}

    Args:
        file_path: input file path
        separator: symbol to split in lines
        coding: encoding type for open text file, default as utf8

    Returns:
        return a dict which used first column as keys.
    """
    dct = {}
    with io.open(file_path, "r", encoding=coding) as f:
        for line in f.readlines():
            key = line.strip().split(separator)[0]
            content = line.strip().split(separator)[1:]
            assert isinstance(content, list)
            dct[key] = content

    return dct


def pin_thread_pools() -> None:
    """Pin every thread pool this process could spawn to one thread.

    Meant for worker processes -- DataLoader workers, scoring pools -- of which
    the machine runs many at once. Workers are forked, so they inherit libraries
    the parent already imported with their thread counts already decided -- an
    environment variable set here is too late for those. Each pool gets its own
    API call instead:

    * torch: ``set_num_threads(1)``.
    * numba: ``set_num_threads(1)``. librosa's resampling inside DNSMOS is
      numba-jitted, and the inherited default is one thread per core.
    * onnxruntime: has no process-wide knob; the caller passes ``num_threads=1``
      per session (``Metrics.dnsmos_p835``).
    * BLAS and OpenMP, through ``threadpoolctl``. numpy/scipy work (PESQ, STOI,
      synthesis-side filtering) runs there, and ``torch.set_num_threads`` never
      reaches it. Left at one thread per core, N workers oversubscribe the
      machine and starve the training or scoring they feed.
    * torch's inter-op pool, which ``set_num_threads`` does not cover (it
      defaults to half the cores). It can only be set before the first parallel
      region, so a worker that has already run one keeps what it has.

    ``OMP_NUM_THREADS`` is still set for anything imported for the first time in
    the worker.
    """
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already parallelised in this process; the pool is fixed for its life.
        pass
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        # Not a context manager here: the limit has to outlive this call and
        # cover every task the worker goes on to run.
        threadpool_limits(limits=1)
    except ImportError:
        pass


def iter_files_recursive(folder: str, file_type: str) -> Iterator[Tuple[str, str]]:
    """
    Walk a folder depth-first in ``os.listdir`` order, yielding ``(file name, full path)``
    for every file whose name contains ``file_type``.

    Unlike the scp-style lines `recursive_read_folder` builds, the pair stays
    unambiguous when a name or a directory contains a space.
    """
    for file in os.listdir(folder):
        cur_path = os.path.join(folder, file)
        if os.path.isdir(cur_path):
            yield from iter_files_recursive(cur_path, file_type)

        else:
            if file_type in file:
                yield file, cur_path


def recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None:
    """
    Recursive reading a folder, and list related file path in the output list.

    Args:
        folder: input folder path
        file_type: suffix for parsing file
        output: storage parser result
    
    Ex:
        _list = []
        recursive_read_folder('corpus', '.flac', _list)
    """
    for file, cur_path in iter_files_recursive(folder, file_type):
        output.append(f"{file} {cur_path}")


def load_hparam(file_path: str):
    """
    Loading configuration file in a dict.

    Args:
        file_path: configure file (*.yaml) path

    Returns:
        hparam in dict form
    """
    hparam_dict = dict()
    with open(file_path, "r", encoding="utf-8") as stream:
        for doc in yaml.safe_load_all(stream):
            if doc:
                hparam_dict.update(doc)

    return hparam_dict


def create_folder(folder_name: str) -> None:
    """
    Create the folder (and its parents) if it does not exist.

    A ``FileExistsError`` from a concurrent creator is logged and ignored.

    Args:
        folder_name: folder path string
    """
    try:
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name, exist_ok=True)
    except FileExistsError:
        logger.debug("File exists passing it: %s", folder_name)


def convolve(x: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    """Doing convolution on x with filter"""
    weight = filter.float().repeat(1, 1, 1)
    x = torch.nn.functional.pad(x, (filter.shape[-1] - 1, 0))
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


def fftconvolve(x: torch.Tensor, kernel: torch.Tensor, mode: str = "full"):
    """
    Asteroid implemented FFT convolution.

    Usage:
        wav_rverb = fftconvolve(wav, rir, mode='full')
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
