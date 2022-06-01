from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft import (create_fourier_kernels, extend_fbins, overlap_add,
                   torch_window_sumsquare)


class FreeEncDec(nn.Module):
    """
    Free filters without any constraints
    Args:
        win_len: samples in time axis
        latten_len: feature dimension
        hop_len: stride step in time axis

    Flows:
        waveform -> laten-feats -> waveform
    """
    def __init__(self, win_length: int = 512, laten_length: int = 512, hop_length: int = 128):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.encoder = self.get_encoder(output_length=laten_length, win_length=win_length, hop_length=hop_length)
        self.decoder = self.get_decoder(input_dim=laten_length, win_length=win_length, hop_length=hop_length)
    
    def get_encoder(self, output_length: int, win_length: int, hop_length: int) -> nn.Module:
        encoder = nn.Conv1d(in_channels=1, out_channels=output_length, kernel_size=win_length, stride=hop_length, bias=False)
        return encoder
    
    def get_decoder(self, input_dim: int, win_length: int, hop_length: int) -> nn.Module:
        decoder = nn.ConvTranspose1d(in_channels=input_dim, out_channels=1, kernel_size=win_length, stride=hop_length, bias=False)
        return decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor x shape is [N, L]
        
        Returns:
            output tensor shape is [N, C, T]
        """
        x = x.unsqueeze(1) # [N, 1, L]
        return self.encoder(x)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor shape is [N, C, T]
        
        Returns:
            output tensor shape is [N, L]
        """
        x = self.decoder(x)
        return x.squeeze(1)


class ConvEncDec(nn.Module):
    """
    ConvEncDec is the fully trainable feature processing
    backbone class: `ConvSTFT` based on convolution layer with STFT kernels
    Flows:
        Forward:
        raw wave -> emphasis wave -> Complex-STFT / Magnitude+Phase
        Inverse:
        Complex-STFT / Magnitude+Phase -> generate wave -> de-emphasis
    """
    def __init__(self, fft_length: int = 512, win_type: str = 'hann', win_length: int = 512, \
        freq_bins: int = None, hop_length: int = 128, freq_scale: str = 'no', iSTFT: bool = True, \
        fmin: int = 0, fmax: int = 8000, sr: int = 16000, trainable: bool = True, output_format: str = "Complex"):
        super().__init__()

        self.n_fft = fft_length
        self.win_length = win_length
        self.freq_bins = freq_bins
        self.hop_length = hop_length
        self.freq_scale = freq_scale
        self.iSTFT = iSTFT
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
        self.trainable = trainable
        self.output_format = output_format
        
        self.window = self.get_windows(win_type)
        self.encoder = self.get_encoder(n_fft=self.n_fft, win_length=self.win_length, freq_scale=self.freq_scale, \
            iSTFT=self.iSTFT, sr=self.sr, fmin=self.fmin, fmax=self.fmax, output_format=self.output_format, \
            trainable=self.trainable, hop_length=self.hop_length)

    def get_windows(self, type: str) -> torch.Tensor:
        if type.lower() == 'hann':
            win = torch.hann_window(self.win_length)
        else:
            raise NotImplementedError(f"window type not support")
        return win
    
    def get_encoder(self, **kwargs) -> nn.Module:
        return ConvSTFT(self.window, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor shape is [N, L]
        
        Returns:
            output tensor shape is [N, C, T, 2]
        """
        x = x.unsqueeze(1) # [N, 1, L]
        return self.encoder(x)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor shape is [N, C, T, 2]
        
        Returns:
            output tensor shape is [N, L]
        """
        gen = self.encoder.inverse(x)
        if gen.dim() == 3: gen = gen.squeeze(1)
        return gen


class ConvSTFT(nn.Module):
    """
    This code majorly comes from nnAudio.

    Reference:
        https://github.com/KinWaiCheuk/nnAudio
    """
    def __init__(self, window_mask: torch.Tensor, n_fft: int = 2048, win_length: Optional[int] = None, \
                freq_bins: Optional[int] = None, hop_length: Optional[int] = None, freq_scale: str = 'no', \
                iSTFT: bool = False, fmin: int = 50, fmax: int = 6000, sr: int = 22050, trainable: bool = False, output_format: str = "Complex"):

        super().__init__()

        if win_length==None: win_length = n_fft
        if hop_length==None: hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.win_length = win_length
        self.iSTFT = iSTFT
        self.trainable = trainable

        # Create filter windows for stft
        kernel_sin, kernel_cos, self.bins2freq, self.bin_list = create_fourier_kernels(n_fft,
            win_length=win_length, freq_bins=freq_bins, freq_scale=freq_scale, fmin=fmin, fmax=fmax, sr=sr)

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)
        
        # In this way, the inverse kernel and the forward kernel do not share the same memory...
        kernel_sin_inv = torch.cat((kernel_sin, -kernel_sin[1:-1].flip(0)), 0)
        kernel_cos_inv = torch.cat((kernel_cos, kernel_cos[1:-1].flip(0)), 0)
                
        if iSTFT:
            self.register_buffer('kernel_sin_inv', kernel_sin_inv.unsqueeze(-1))
            self.register_buffer('kernel_cos_inv', kernel_cos_inv.unsqueeze(-1))

        # Applying window functions to the Fourier kernels

        if len(window_mask) != self.n_fft:
            raise TypeError(f"only support window length == n_fft")

        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask
                
        if self.trainable==True:
            # set kernel required_grad=True
            wsin = torch.nn.Parameter(wsin, requires_grad=self.trainable)
            wcos = torch.nn.Parameter(wcos, requires_grad=self.trainable)  
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
        
        # Prepare the shape of window mask so that it can be used later in inverse
        self.register_buffer('window_mask', window_mask.unsqueeze(0).unsqueeze(-1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of waveforms to spectrograms.
        ----------
        Input:
            input tensor x shape is [N, channel, L]
                
        Returns:
            output tensor shape is [N, C, T]
        """
        output_format = self.output_format
        
        spec_imag = F.conv1d(x, self.wsin, stride=self.stride)
        spec_real = F.conv1d(x, self.wcos, stride=self.stride)  # Doing STFT by using conv1d

        # remove redundant parts
        spec_real = spec_real[:, :self.freq_bins, :]
        spec_imag = spec_imag[:, :self.freq_bins, :]

        if output_format=='Complex':
            return torch.stack((spec_real,-spec_imag), -1)  # Remember the minus sign for imaginary part

        elif output_format=='MagPhase':
            mags = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable==True:
                mags = torch.sqrt(mags+1e-8)
            phase = torch.atan2(-spec_imag+0.0,spec_real)
            return torch.stack([mags, phase], dim=-1)
        else:
            raise NotImplementedError

    def inverse(self, X: torch.Tensor, refresh_win: bool = True) -> torch.Tensor:
        """
        which is to convert spectrograms back to waveforms. 
        It only works for the complex value spectrograms. If you have the magnitude spectrograms,
        please use :func:`~nnAudio.Spectrogram.Griffin_Lim`.
        
        Parameters
        ----------
        refresh_win : bool
            Recalculating the window sum square. If you have an input with fixed number of timesteps,
            you can increase the speed by setting ``refresh_win=False``. Else please keep ``refresh_win=True``
           
        """
        if (hasattr(self, 'kernel_sin_inv') != True) or (hasattr(self, 'kernel_cos_inv') != True):
            raise NameError("Please activate the iSTFT module by setting `iSTFT=True` if you want to use `inverse`")      
        
        assert X.dim()==4 , "Inverse iSTFT only works for complex number," \
                            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)."\
                            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."

        if self.output_format == 'Complex':
            # n_fft//2+1 -> n_fft
            X = extend_fbins(X) # extend freq
            X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]
        else:
            raise NotImplementedError('Inverse only support complex input')


        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = F.conv2d(X_real_bc, self.kernel_cos_inv, stride=(1,1))
        b2 = F.conv2d(X_imag_bc, self.kernel_sin_inv, stride=(1,1))
        
        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2)*self.window_mask

        # Normalize the amplitude with n_fft
        real /= (self.n_fft)

        # Overlap and Add algorithm to connect all the frames
        real = overlap_add(real, self.stride)
    
        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        if hasattr(self, 'w_sum')==False or refresh_win==True:
            self.w_sum = torch_window_sumsquare(self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft).flatten()
            self.nonzero_indices = (self.w_sum>1e-10)    
        else:
            pass
        real[:, self.nonzero_indices] = real[:,self.nonzero_indices].div(self.w_sum[self.nonzero_indices])
        
        return real
