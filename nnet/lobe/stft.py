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
