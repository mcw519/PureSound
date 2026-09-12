from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stft import create_fourier_kernels, extend_fbins, overlap_add, torch_window_sumsquare


class FreeEncDec(nn.Module):
    """
    Free filters without any constraints
    Args:
        win_len: samples in time axis
        latten_len: feature dimension
        hop_len: stride step in time axis
        output_active: if given, add ReLU activation after encoder's output

    Flows:
        waveform -> laten-feats -> waveform
    """

    def __init__(
        self,
        win_length: int = 512,
        laten_length: int = 512,
        hop_length: int = 128,
        output_active: Optional[str] = None,
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.output_active = output_active
        self.encoder = self.get_encoder(
            output_length=laten_length, win_length=win_length, hop_length=hop_length
        )
        self.decoder = self.get_decoder(
            input_dim=laten_length, win_length=win_length, hop_length=hop_length
        )

    def get_encoder(
        self, output_length: int, win_length: int, hop_length: int
    ) -> nn.Module:
        encoder = nn.Conv1d(
            in_channels=1,
            out_channels=output_length,
            kernel_size=win_length,
            stride=hop_length,
            bias=False,
        )

        if self.output_active is not None:
            nonlinear = getattr(nn, self.output_active)()
            encoder = nn.Sequential(encoder, nonlinear)

        return encoder

    def get_decoder(
        self, input_dim: int, win_length: int, hop_length: int
    ) -> nn.Module:
        decoder = nn.ConvTranspose1d(
            in_channels=input_dim,
            out_channels=1,
            kernel_size=win_length,
            stride=hop_length,
            bias=False,
        )
        return decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor x shape is [N, L] or [N, 1, L]

        Returns:
            output tensor shape is [N, C, T]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N, 1, L]
        x = self.encoder(x)
        return x

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
        raw wave -> Complex-STFT
        Inverse:
        Complex-STFT -> generate wave
    """

    def __init__(
        self,
        fft_length: int = 512,
        win_type: str = "hann",
        win_length: int = 512,
        freq_bins: int = None,
        hop_length: int = 128,
        freq_scale: str = "no",
        iSTFT: bool = True,
        fmin: int = 0,
        fmax: int = 8000,
        sr: int = 16000,
        preemphasis: Optional[float] = None,
        trainable: bool = True,
    ):
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
        self.preemphasis = preemphasis
        self.trainable = trainable

        self.window = self.get_windows(win_type)
        self.encoder = self.get_encoder(
            n_fft=self.n_fft,
            win_length=self.win_length,
            freq_scale=self.freq_scale,
            iSTFT=self.iSTFT,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            trainable=self.trainable,
            hop_length=self.hop_length,
        )

    def get_windows(self, type: str) -> torch.Tensor:
        if type.lower() == "hann":
            win = torch.hann_window(self.win_length)
        elif type.lower() == "hamming":
            win = torch.hamming_window(self.win_length)
        elif type.lower() == "blackman":
            win = torch.blackman_window(self.win_length)
        else:
            raise NotImplementedError("window type not support")
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
        if self.preemphasis is not None:
            padded = torch.nn.functional.pad(x, (1, 0))
            x = x - self.preemphasis * padded[:, :-1]

        x = x.unsqueeze(1)  # [N, 1, L]
        return self.encoder(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input tensor shape is [N, C, T, 2]

        Returns:
            output tensor shape is [N, L]
        """
        gen = self.encoder.inverse(x)
        if gen.dim() == 3:
            gen = gen.squeeze(1)
        return gen


class ConvSTFT(nn.Module):
    """
    This code majorly comes from nnAudio.

    Reference:
        https://github.com/KinWaiCheuk/nnAudio
    """

    def __init__(
        self,
        window_mask: torch.Tensor,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        freq_bins: Optional[int] = None,
        hop_length: Optional[int] = None,
        freq_scale: str = "no",
        iSTFT: bool = False,
        fmin: int = 50,
        fmax: int = 6000,
        sr: int = 22050,
        trainable: bool = False,
    ):
        super().__init__()

        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.trainable = trainable
        self.stride = hop_length
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.win_length = win_length
        self.iSTFT = iSTFT
        self.trainable = trainable

        # Create filter windows for stft
        kernel_sin, kernel_cos, self.bins2freq, self.bin_list = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=freq_bins,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
        )

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        # In this way, the inverse kernel and the forward kernel do not share the same memory...
        kernel_sin_inv = torch.cat((kernel_sin, -kernel_sin[1:-1].flip(0)), 0)
        kernel_cos_inv = torch.cat((kernel_cos, kernel_cos[1:-1].flip(0)), 0)

        if iSTFT:
            self.register_buffer("kernel_sin_inv", kernel_sin_inv.unsqueeze(-1))
            self.register_buffer("kernel_cos_inv", kernel_cos_inv.unsqueeze(-1))

        # Applying window functions to the Fourier kernels

        if len(window_mask) != self.n_fft:
            raise TypeError("only support window length == n_fft")

        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask

        if self.trainable == True:
            # set kernel required_grad=True
            wsin = torch.nn.Parameter(wsin, requires_grad=self.trainable)
            wcos = torch.nn.Parameter(wcos, requires_grad=self.trainable)
            self.register_parameter("wsin", wsin)
            self.register_parameter("wcos", wcos)
        else:
            self.register_buffer("wsin", wsin)
            self.register_buffer("wcos", wcos)

        # Prepare the shape of window mask so that it can be used later in inverse
        self.register_buffer("window_mask", window_mask.unsqueeze(0).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of waveforms to spectrum.
        ----------
        Input:
            input tensor x shape is [N, channel, L]

        Returns:
            output tensor shape is [N, C, T]
        """
        # Doing STFT by using conv1d
        spec_imag = F.conv1d(x, self.wsin, stride=self.stride)
        spec_real = F.conv1d(x, self.wcos, stride=self.stride)

        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        # Remember the minus sign for imaginary part
        return torch.stack((spec_real, -spec_imag), -1)

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
        if (hasattr(self, "kernel_sin_inv") != True) or (
            hasattr(self, "kernel_cos_inv") != True
        ):
            raise NameError(
                "Please activate the iSTFT module by setting `iSTFT=True` if you want to use `inverse`"
            )

        assert X.dim() == 4, (
            "Inverse iSTFT only works for complex number,"
            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)."
            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."
        )

        # n_fft//2+1 -> n_fft
        X = extend_fbins(X)  # extend freq
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = F.conv2d(X_real_bc, self.kernel_cos_inv, stride=(1, 1))
        b2 = F.conv2d(X_imag_bc, self.kernel_sin_inv, stride=(1, 1))

        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2) * self.window_mask

        # Normalize the amplitude with n_fft
        real /= self.n_fft

        # Overlap and Add algorithm to connect all the frames
        real = overlap_add(real, self.stride)

        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        if hasattr(self, "w_sum") == False or refresh_win == True:
            self.w_sum = torch_window_sumsquare(
                self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft
            ).flatten()
            self.nonzero_indices = self.w_sum > 1e-10
        else:
            pass
        real[:, self.nonzero_indices] = real[:, self.nonzero_indices].div(
            self.w_sum[self.nonzero_indices]
        )

        return real


class UnifiedConvEncDec(nn.Module):
    """
    Unifed ConvEncDec can handle any input sampling rate audios.
    All the I/O follows the 25 ms window length and 10 ms hop length
    backbone class: `ConvSTFT` based on convolution layer with STFT kernels
    Flows:
        Forward:
        raw wave -> Complex-STFT
        Inverse:
        Complex-STFT -> generate wave
    """

    def __init__(
        self,
        win_type: str = "hann",
        trainable: bool = False,
    ):
        super().__init__()
        self.trainable = trainable
        win_func = self.get_window_type(type=win_type)
        # Initialized different SR encoder
        self.encoder_params = self.get_stft_parms()
        self.encoder = {}
        for sr in self.encoder_params.keys():
            params = self.encoder_params[sr]
            params.update({"window_mask": win_func(params["n_fft"])})
            self.encoder[sr] = ConvSTFT(iSTFT=True, **params)

    def get_stft_parms(self):
        params = {
            8000: {
                "sr": 8000,
                "n_fft": 200,
                "hop_length": 80,
                "fmin": 0,
                "fmax": 4000,
                "freq_scale": "no",
                "trainable": self.trainable,
            },
            16000: {
                "sr": 16000,
                "n_fft": 400,
                "hop_length": 160,
                "fmin": 0,
                "fmax": 8000,
                "freq_scale": "no",
                "trainable": self.trainable,
            },
            22050: {
                "sr": 22050,
                "n_fft": 550,
                "hop_length": 220,
                "fmin": 0,
                "fmax": 11025,
                "freq_scale": "no",
                "trainable": self.trainable,
            },  # ?
            24000: {
                "sr": 24000,
                "n_fft": 600,
                "hop_length": 240,
                "fmin": 0,
                "fmax": 12000,
                "freq_scale": "no",
                "trainable": self.trainable,
            },
            32000: {
                "sr": 32000,
                "n_fft": 800,
                "hop_length": 320,
                "fmin": 0,
                "fmax": 16000,
                "freq_scale": "no",
                "trainable": self.trainable,
            },
            44100: {
                "sr": 44100,
                "n_fft": 1100,
                "hop_length": 441,
                "fmin": 0,
                "fmax": 22050,
                "freq_scale": "no",
                "trainable": self.trainable,
            },  # ?
            48000: {
                "sr": 48000,
                "n_fft": 1200,
                "hop_length": 480,
                "fmin": 0,
                "fmax": 24000,
                "freq_scale": "no",
                "trainable": self.trainable,
            },
        }
        return params

    def get_window_type(self, type: str) -> torch.Tensor:
        if type.lower() == "hann":
            win = torch.hann_window
        elif type.lower() == "hamming":
            win = torch.hamming_window
        elif type.lower() == "blackman":
            win = torch.blackman_window
        else:
            raise NotImplementedError("window type not support")
        return win

    def forward(self, x: torch.Tensor, sr: Union[torch.Tensor, int]):
        """
        Args:
            input tensor shape is [N, L]
            sr: sample rate shape is tensor [N] or int

        Returns:
            output tensor shape is [N, C, T, 2]
        """
        x = x.unsqueeze(1)  # [N, 1, L]

        out = []
        if isinstance(sr, int):
            return self.encoder[sr](x)

        else:
            for i in range(x.shape[0]):
                out.append(self.encoder[int(sr[i])](x[i].unsqueeze(0)))

            return torch.cat(out, dim=0)

    def inverse(self, x: torch.Tensor, sr: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Args:
            input tensor shape is [N, C, T, 2]
            sr: sample rate shape is tensor [N] or int

        Returns:
            output tensor shape is [N, L]
        """
        out = []

        if isinstance(sr, int):
            return self.encoder[sr].inverse(x)
        else:
            for i in range(x.shape[0]):
                gen = self.encoder[int(sr[i])].inverse(x[i].unsqueeze(0))
                if gen.dim() == 3:
                    gen = gen.squeeze(1)
                out.append(gen)

            return torch.cat(out, dim=0)
