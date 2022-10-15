from typing import Optional

import numpy as np
import torch
from torch.nn.functional import fold


def create_fourier_kernels(n_fft, win_length=None, freq_bins=None, fmin=50,fmax=6000, sr=44100, freq_scale='linear'):
    """
    This code comes from nnAudio
    n_fft : int
        The window size
    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins
    fmin : int
        The starting frequency for the lowest frequency bin.
        If freq_scale is ``no``, this argument does nothing.
    fmax : int
        The ending frequency for the highest frequency bin.
        If freq_scale is ``no``, this argument does nothing.
    sr : int
        The sampling rate for the input audio. It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.
    freq_scale: 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin.
        When 'linear' or 'log' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``.
        If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.
    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``
    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``
    bins2freq : list
        Mapping each frequency bin to frequency in Hz.
    binslist : list
        The normalized frequency ``k`` in digital domain.
        This ``k`` is in the Discrete Fourier Transform equation $$

    """
                           
    if freq_bins==None: freq_bins = n_fft//2+1
    if win_length==None: win_length = n_fft

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []

    if freq_scale == 'linear':
        start_bin = start_freq*n_fft/sr
        scaling_ind = (end_freq-start_freq)*(n_fft/sr)/freq_bins

        for k in range(freq_bins): # Only half of the bins contain useful info
            bins2freq.append((k*scaling_ind+start_bin)*sr/n_fft)
            binslist.append((k*scaling_ind+start_bin))
            wsin[k,0,:] = np.sin(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)

    elif freq_scale == 'log':
        start_bin = start_freq*n_fft/sr
        scaling_ind = np.log(end_freq/start_freq)/freq_bins

        for k in range(freq_bins): # Only half of the bins contain useful info
            bins2freq.append(np.exp(k*scaling_ind)*start_bin*sr/n_fft)
            binslist.append((np.exp(k*scaling_ind)*start_bin))
            wsin[k,0,:] = np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)

    elif freq_scale == 'no':
        for k in range(freq_bins): # Only half of the bins contain useful info
            bins2freq.append(k*sr/n_fft)
            binslist.append(k)
            wsin[k,0,:] = np.sin(2*np.pi*k*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*k*s/n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")

    return wsin.astype(np.float32), wcos.astype(np.float32), bins2freq, binslist


def overlap_add(X, stride):
    n_fft = X.shape[1]
    output_len = n_fft + stride*(X.shape[2]-1)
    return fold(X, (1,output_len), kernel_size=(1,n_fft), stride=stride).flatten(1)


def torch_window_sumsquare(w, n_frames, stride, n_fft, power=2):
    w_stacks = w.unsqueeze(-1).repeat((1,n_frames)).unsqueeze(0)
    # Window length + stride*(frames-1)
    output_len = w_stacks.shape[1] + stride*(w_stacks.shape[2]-1) 
    return fold(w_stacks**power, (1,output_len), kernel_size=(1,n_fft), stride=stride)


def extend_fbins(X):
    """Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
       reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
    X_upper = X[:,1:-1].flip(1)
    X_upper[:,:,:,1] = -X_upper[:,:,:,1] # For the imaganinry part, it is an odd function
    return torch.cat((X[:, :, :], X_upper), 1)


# mel frequency related
def hz2mel(frequencies):
    """
    Convert Hz to Mels.\n
    Ex:
        hz2mel(60)
        >> 0.8999999999999999
        hz2mel([110, 220, 440])
        >> array([1.65, 3.3 , 6.6 ])
    """
    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel2hz(mels):
    """
    Convert mel bin numbers to frequencies,\n
    Ex:
        mel2hz(1)
        >> 66.66666666666667
        mel2hz([0,1,2,3,4])
        >> array([0., 66.66666667, 133.33333333, 200., 266.66666667])
    """
    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def fft_frequencies(sr: int = 16000, n_fft: int = 512) -> np.ndarray:
    """
    Center freqs of each FFT bin.\n
    Ex:
        fft_frequencies()
        >> array([   0.  ,   31.25,   62.5 ,   93.75,  125.  ,  156.25,  187.5 ,
                    ..., 
                    7656.25, 7687.5 , 7718.75, 7750.  , 7781.25, 7812.5 , 7843.75,
                    7875.  , 7906.25, 7937.5 , 7968.75, 8000.  ])
    
    Args:
        sr: sampling rate
        n_fft: number of fft bin
    
    Returns:
        each bin's center frequency, length is (n_fft//2)+1
    """
    return np.linspace(0, float(sr) / 2, int(1 + n_fft//2), endpoint=True)


def mel_frequencies(n_mels: int = 128, fmin: float = 0.0, fmax: float = 8000):
    """
    Center freqs' of mel bands - uniformly spaced between limits
    example:
        mel_frequencies(n_mels=40)
        >>  array([   0.        ,   77.34297517,  154.68595033,  232.0289255 ,
                    309.37190066,  386.71487583,  464.05785099,  541.40082616,
                    618.74380133,  696.08677649,  773.42975166,  850.77272682,
                    928.11570199, 1005.64528123, 1089.14328655, 1179.57407126,
                    1277.51325907, 1383.5842673 , 1498.46227514, 1622.87852145,
                    1757.62495932, 1903.55929714, 2061.61045819, 2232.78449362,
                    2418.17098624, 2618.94998618, 2836.39952226, 3071.90373712,
                    3326.96169776, 3603.19693767, 3902.36779111, 4226.37858561,
                    4577.29176363, 4957.34101076, 5368.94547386, 5814.72515988,
                    6297.517613  , 6820.3959767 , 7386.68855534, 8000.        ])
    """
    min_mel = hz2mel(fmin)
    max_mel = hz2mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)
    
    return mel2hz(mels)


def mel_filterbank(sr: int, n_fft: int, n_banks: int = 128, fmin: float = 0., fmax: Optional[float] = None, norm: int = 1) -> torch.Tensor:
    """
    Create Mel-Filterbank weights.

    Args:
        sr: sampling rate
        n_fft: number of fft bin
        n_banks: number of mel-filter banks, feature dimension
        fmax: highest frequency range, if not, using Nyquist sampling theory
        
    Return:
        return mel-filter banks, shape is [n_fft, n_mels]
    """
    if fmax is None:
        fmax = float(sr/2)
    
    weights = np.zeros((n_banks, int(1 + n_fft//2)), dtype=np.float32)
    
    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_banks + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_banks):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    
    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_banks+2] - mel_f[:n_banks])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        raise ValueError('Empty filters detected in mel frequency basis. \
                      Some channels will produce empty responses. \
                      Try increasing your sampling rate (and fmax) or reducing n_banks.')
    
    return torch.from_numpy(weights)
