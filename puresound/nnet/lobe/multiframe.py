import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .group_op import GroupedLinear, SqueezedGRU


class MultiFrameModule(nn.Module):
    """
    Multi-frame speech enhancement modules.

    Signal model and notation:
        Noisy: `x = s + n`
        Enhanced: `y = f(x)`
        Objective: `min ||s - y||`

        PSD: Power spectral density, notated eg. as `Rxx` for noisy PSD.
        IFC: Inter-frame correlation vector: PSD*u, u: selection vector. Notated as `rxx`
        RTF: Relative transfere function, also called steering vector.

    Args:
        num_freqs (int): Number of frequency bins used for filtering.
        frame_size (int): Frame size in FD domain.
        lookahead (int): Lookahead, may be used to select the output time step.
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int,
        real: bool = False,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.real = real
        if real:
            self.pad = nn.ConstantPad3d(
                (0, 0, 0, 0, frame_size - 1 - lookahead, lookahead), 0.0
            )
        else:
            self.pad = nn.ConstantPad2d(
                (0, 0, frame_size - 1 - lookahead, lookahead), 0.0
            )
        self.need_unfold = frame_size > 1
        self.lookahead = lookahead

    def spec_unfold_real(self, spec: torch.Tensor):
        if self.need_unfold:
            spec = self.pad(spec).unfold(-3, self.frame_size, 1)
            return spec.permute(0, 1, 5, 2, 3, 4)

        return spec.unsqueeze(-1)

    def spec_unfold(self, spec: torch.Tensor):
        """
        Pads and unfolds the spectrogram according to frame_size.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C, T, F]

        Returns:
            spec (Tensor): Unfolded spectrogram of shape [B, C, T, F, N], where N: frame_size.
        """
        if self.need_unfold:
            return self.pad(spec).unfold(2, self.frame_size, 1)
        return spec.unsqueeze(-1)

    @staticmethod
    def solve(Rxx, rss, diag_eps: float = 1e-8, eps: float = 1e-7):
        return torch.einsum(
            "...nm,...m->...n", torch.inverse(_tik_reg(Rxx, diag_eps, eps)), rss
        )  # [T, F, N]

    @staticmethod
    def apply_coefs(spec: torch.Tensor, coefs: torch.Tensor):
        """
        Applose coefs on spectogram

        Args:
            spec (Tensor): Spectrogram of shape [B, C, T, F, N]
            coefs (Tensor): Coefficients of shape [B, C, T, F, N]
        """
        return torch.einsum("...n,...n->...", spec, coefs)


def psd(x: torch.Tensor, n: int):
    """
    Compute the PSD correlation matrix Rxx for a spectrogram.

    That is, `X*conj(X)`, where `*` is the outer product.

    Args:
        x (complex Tensor): Spectrogram of shape [B, C, T, F]. Will be unfolded with `n` steps over
            the time axis.

    Returns:
        Rxx (complex Tensor): Correlation matrix of shape [B, C, T, F, N, N]
    """
    x = F.pad(x, (0, 0, n - 1, 0)).unfold(-2, n, 1)
    return torch.einsum("...n,...m->...mn", x, x.conj())


def df(spec: torch.Tensor, coefs: torch.Tensor):
    """
    Deep filter implementation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F, N]
        coefs (complex Tensor): Coefficients of shape [B, C, N, T, F]

    Returns:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F]
    """
    return torch.einsum("...tfn,...ntf->...tf", spec, coefs)


class DeepFilter(MultiFrameModule):
    """Deep Filtering"""

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int,
        real: bool = False,
        conj: bool = False,
    ):
        super().__init__(num_freqs, frame_size, lookahead, real)
        self.conj = conj

    def forward(self, spec: torch.Tensor, coefs: torch.Tensor):
        """
        Args:
            spec (Tensor): Spectrogram of shape [B, C, T, F, 2]
            coefs (Tensor): Coefficients of shape [B, N, T, F, 2]

        Returns:
            spec (Tensor): Enhanced spectrogram of shape [B, C, T, F, 2]
        """
        spec_u = self.spec_unfold(torch.view_as_complex(spec))  # [B, C, T, F, N]
        coefs = torch.view_as_complex(coefs)
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)
        coefs = coefs.view(
            coefs.shape[0], -1, self.frame_size, coefs.shape[2], coefs.shape[3]
        )

        if self.conj:
            coefs = coefs.conj()

        spec_f = df(spec_f, coefs)

        if self.training:
            spec = spec.clone()

        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec


class MultiFrameWienerFilter(MultiFrameModule):
    """
    Multi-frame Wiener filter

    Args:
        num_freqs (int): Number of frequency bins to apply MVDR filtering to.
        frame_size (int): Frame size of the MF MVDR filter.
        lookahead (int): Lookahead of the frame.
        cholesky_decomp (bool): Whether the input is a cholesky decomposition of the correlation matrix. Defauls to `False`.
        inverse (bool): Whether the input is a normal or inverse correlation matrix. Defaults to `True`.
        enforce_constraints (bool): Enforce hermetian matrix for non-inverse input and a triangular matrix for cholesky decomposition inpiut.
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int,
        cholesky_decomp: bool = False,
        inverse: bool = True,
        enforce_constraints: bool = True,
        eps: float = 1e-8,
        dload: float = 1e-7,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.cholesky_decomp = cholesky_decomp
        self.inverse = inverse
        self.enforce_constraints = enforce_constraints
        self.triu_idcs = torch.triu_indices(self.frame_size, self.frame_size, 1)
        self.tril_idcs = torch.empty_like(self.triu_idcs)
        self.tril_idcs[0] = self.triu_idcs[1]
        self.tril_idcs[1] = self.triu_idcs[0]
        self.eps = eps
        self.dload = dload

    def get_r_factor(self):
        """Return an factor f such that Rxx/f in range [-1, 1]"""
        if self.inverse and self.cholesky_decomp:
            return 2e3
        elif self.inverse and not self.cholesky_decomp:
            return 3e7
        elif not self.inverse and self.cholesky_decomp:
            return 2e-4
        elif not self.inverse and not self.cholesky_decomp:
            return 5e-6

    def forward(self, spec: torch.Tensor, ifc: torch.Tensor, iRxx: torch.Tensor):
        """
        Multi-frame Wiener filter based on Rxx**-1 and speech IFC vector.

        Args:
            spec (Tensor): Spectrogram of shape [B, 1, T, F, 2]
            ifc (Tensor): Inter-frame speech correlation vector [B, T, F, N*2]
            iRxx (Tensor): (Inverse) noisy covariance matrix Rxx**-1 [B, T, F, (N**2)*2] OR
                cholesky_decomp Rxx=L*L^H of the same shape.

        Returns:
            spec (Tensor): Filtered spectrogram of shape [B, C, T, F, 2]
        """
        spec_u = self.spec_unfold(torch.view_as_complex(spec))
        iRxx = torch.view_as_complex(
            iRxx.unflatten(3, (self.frame_size, self.frame_size, 2))
        )

        if self.cholesky_decomp:
            if self.enforce_constraints:
                # Upper triangular (wo. diagonal) must be zero
                iRxx[:, :, :, self.triu_idcs[0], self.triu_idcs[1]] = 0.0
            # Revert cholesky decomposition
            iRxx = iRxx.matmul(iRxx.transpose(3, 4).conj())

        if self.enforce_constraints and not self.inverse and not self.cholesky_decomp:
            # If we have a cholesky_decomp input the constraints are already enforced.
            # We have a standard correlation matrix as input. Imaginary part on the diagonal should be 0.
            torch.diagonal(iRxx, dim1=-1, dim2=-2).imag = 0.0
            # Triu should be complex conj of tril
            tril_conj = iRxx[:, :, :, self.tril_idcs[0], self.tril_idcs[1]].conj()
            iRxx[:, :, :, self.triu_idcs[0], self.triu_idcs[1]] = tril_conj

        ifc = torch.view_as_complex(ifc.unflatten(3, (self.frame_size, 2)))
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)

        if (
            not self.inverse
        ):  # Is already an inverse input. No need to inverse it again.
            # Regularization on diag for stability
            iRxx = _tik_reg(iRxx, self.dload, self.eps)
            # Compute weights by solving the equation system
            w = torch.linalg.solve(iRxx, ifc).unsqueeze(1)
        else:  # We already have an inverse estimate. Just compute the linear combination.
            w = torch.einsum("...nm,...m->...n", iRxx, ifc).unsqueeze(1)  # [B, 1, F, N]

        spec_f = self.apply_coefs(spec_f, w)
        if self.training:
            spec = spec.clone()

        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec


class MultiFrameMvdrFilter(MultiFrameModule):
    """
    Multi-frame MVDR filter

    Args:
        num_freqs (int): Number of frequency bins to apply MVDR filtering to.
        frame_size (int): Frame size of the MF MVDR filter.
        lookahead (int): Lookahead of the frame.
        cholesky_decomp (bool): Whether the input is a cholesky decomposition of the correlation matrix. Defauls to `False`.
        inverse (bool): Whether the input is a normal or inverse correlation matrix. Defaults to `True`.
        enforce_constraints (bool): Enforce hermetian matrix for non-inverse input and a triangular matrix for cholesky decomposition inpiut.
    """

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int,
        cholesky_decomp: bool = False,
        inverse: bool = True,
        enforce_constraints: bool = True,
        eps: float = 1e-8,
        dload: float = 1e-7,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.cholesky_decomp = cholesky_decomp
        self.inverse = inverse
        self.enforce_constraints = enforce_constraints
        self.triu_idcs = torch.triu_indices(self.frame_size, self.frame_size, 1)
        self.tril_idcs = torch.empty_like(self.triu_idcs)
        self.tril_idcs[0] = self.triu_idcs[1]
        self.tril_idcs[1] = self.triu_idcs[0]
        self.eps = eps
        self.dload = dload

    def get_r_factor(self):
        """Return an factor f such that Rxx/f in range [-1, 1]"""
        if self.inverse and self.cholesky_decomp:
            return 2e4
        elif self.inverse and not self.cholesky_decomp:
            return 3e8
        elif not self.inverse and self.cholesky_decomp:
            return 5e-5
        elif not self.inverse and not self.cholesky_decomp:
            return 1e-6

    def forward(self, spec: torch.Tensor, ifc: torch.Tensor, iRnn: torch.Tensor):
        """
        Multi-frame MVDR filter based on Rnn**-1 and speech IFC vector.

        Args:
            spec (Tensor): Spectrogram of shape [B, 1, T, F, 2]
            ifc (Tensor): Inter-frame speech correlation vector [B, T, F, N*2]
            iRnn (Tensor): (Inverse) noisy covariance matrix Rnn**-1 [B, T, F, (N**2)*2] OR
                cholesky_decomp Rxx=L*L^H of the same shape.

        Returns:
            spec (Tensor): Filtered spectrogram of shape [B, C, T, F, 2]
        """
        spec_u = self.spec_unfold(torch.view_as_complex(spec))
        iRnn = torch.view_as_complex(
            iRnn.unflatten(3, (self.frame_size, self.frame_size, 2))
        )

        if self.cholesky_decomp:
            if self.enforce_constraints:
                # Upper triangular (wo. diagonal) must be zero
                iRnn[:, :, :, self.triu_idcs[0], self.triu_idcs[1]] = 0.0
            # Revert cholesky decomposition
            iRnn = iRnn.matmul(iRnn.transpose(3, 4).conj())

        if self.enforce_constraints and not self.inverse and not self.cholesky_decomp:
            # If we have a cholesky_decomp input the constraints are already enforced.
            # We have a standard correlation matrix as input. Imaginary part on the diagonal should be 0.
            torch.diagonal(iRnn, dim1=-1, dim2=-2).imag = 0.0
            # Triu should be complex conj of tril
            tril_conj = iRnn[:, :, :, self.tril_idcs[0], self.tril_idcs[1]].conj()
            iRnn[:, :, :, self.triu_idcs[0], self.triu_idcs[1]] = tril_conj

        ifc = torch.view_as_complex(ifc.unflatten(3, (self.frame_size, 2)))
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)

        if (
            not self.inverse
        ):  # Is already an inverse input. No need to inverse it again.
            # Regularization on diag for stability
            iRnn = _tik_reg(iRnn, self.dload, self.eps)
            # Compute weights by solving the equation system
            numerator = torch.linalg.solve(iRnn, ifc)
        else:  # We already have an inverse estimate. Just compute the linear combination.
            numerator = torch.einsum("...nm,...m->...n", iRnn, ifc)  # [B, 1, F, N]

        denomminator = torch.einsum("...n,...n->...", ifc.conj(), numerator)
        # Normalize numerator
        scale = ifc[..., -1, None].conj()
        w = (numerator * scale / (denomminator.real.unsqueeze(-1) + self.eps)).unsqueeze(1)
        spec_f = self.apply_coefs(spec_f, w)
        if self.training:
            spec = spec.clone()

        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec


def _compute_mat_trace(input: torch.Tensor, dim1: int = -2, dim2: int = -1):
    """
    Compute the trace of a Tensor along ``dim1`` and ``dim2`` dimensions.

    Args:
        input (torch.Tensor): Tensor of dimension `(..., channel, channel)`
        dim1 (int, optional): the first dimension of the diagonal matrix
            (Default: -1)
        dim2 (int, optional): the second dimension of the diagonal matrix
            (Default: -2)

    Returns:
        Tensor: trace of the input Tensor
    """
    assert input.ndim >= 2, "The dimension of the tensor must be at least 2."
    assert (
        input.shape[dim1] == input.shape[dim2]
    ), "The size of ``dim1`` and ``dim2`` must be the same."
    input = torch.diagonal(input, 0, dim1=dim1, dim2=dim2)
    return input.sum(dim=-1)


def _tik_reg(mat: torch.Tensor, reg: float = 1e-7, eps: float = 1e-8) -> torch.Tensor:
    """
    Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (torch.Tensor): input matrix (..., channel, channel)
        reg (float, optional): regularization factor (Default: 1e-8)
        eps (float, optional): a value to avoid the correlation matrix is all-zero (Default: ``1e-8``)

    Returns:
        Tensor: regularized matrix (..., channel, channel)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    epsilon = _compute_mat_trace(mat).real[..., None, None] * reg
    # in case that correlation_matrix is all-zero
    epsilon = epsilon + eps
    mat = mat + epsilon * eye[..., :, :]
    return mat


class DeepFilterDecoder(nn.Module):
    """
    This model predicts the inverse noise/noisy covariance matrix and the speech intra-frame
    correlation (IFC) vector for usage in an Multi-frame Wiener filter or Multi-frame Mvdr filter.
    Args:
        df_bins (int): how much frequency bins
        cpx_in_channels (int): how much chaneels from complex input
        emb_in_dim (int): dimension of non-complex input
        emb_hid_dim (int): GRU's hidden size
        df_order (int): n taps of filter length
        df_n_layer (int): numbers of GRU layers
        df_pathway_kernel_size_t (int): conv2d kernel size along time axis
        df_lin_groups (int): groups number
    """

    def __init__(
        self,
        df_bins: int,
        cpx_in_channels: int,
        emb_in_dim: int,
        emb_hid_dim: int = 256,
        df_order: int = 3,
        df_n_layer: int = 3,
        df_pathway_kernel_size_t: int = 1,
        df_lin_groups: int = 1,
    ):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        self.cov_convp = self._create_conv2d_bn_relu(
            in_channels=cpx_in_channels,
            out_channels=(df_order**2) * 2,
            kernel_size=(df_pathway_kernel_size_t, 1),
            f_stride=1,
            separable=True,
            bias=False,
        )
        self.ifc_convp = self._create_conv2d_bn_relu(
            in_channels=cpx_in_channels,
            out_channels=df_order * 2,
            kernel_size=(df_pathway_kernel_size_t, 1),
            f_stride=1,
            separable=True,
            bias=False,
        )
        self.df_gru = SqueezedGRU(
            input_size=emb_in_dim,
            hidden_size=emb_hid_dim,
            num_layers=df_n_layer,
            linear_groups=8,
        )
        self.cov_out = GroupedLinear(
            input_size=emb_hid_dim,
            hidden_size=df_bins * df_order**2 * 2,
            groups=df_lin_groups,
            shuffle=False,
        )
        self.ifc_out = GroupedLinear(
            input_size=emb_hid_dim,
            hidden_size=df_bins * df_order * 2,
            groups=df_lin_groups,
            shuffle=False,
        )

    def _create_conv2d_bn_relu(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        f_stride: int = 1,
        dilation: int = 1,
        separable: bool = False,
        bias: bool = True,
    ):
        fpad = kernel_size[1] // 2 + dilation - 1
        pad = (fpad, fpad, kernel_size[0] - 1, 0)
        groups = 1
        if separable:
            groups = math.gcd(in_channels, out_channels)

        layers = []
        layers.append(
            nn.ConstantPad2d(pad, 0.0),
        )
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, f_stride),
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            ),
        )

        if separable:
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            )

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, c0: torch.Tensor):
        """
        Args:
            emb (Tensor): embedding of shape [N, C, T]
            c0  (Tensor): complex spectrogram of shape [N, CH, C, T]

        Returns:
            ifc (Tensor): intra-frame correlation matrix of shape [N, C, T, df_order * 2]
            cov (Tensor): covariance matrix of shape [N, C, T, df_order^2 * 2]
        """
        batch, _, nframes = emb.shape
        c = self.df_gru(emb)

        c0_ifc = self.ifc_convp(c0).permute(0, 2, 3, 1)  # [N, C, T, CH], channel last
        c0_cov = self.cov_convp(c0).permute(0, 2, 3, 1)  # [N, C, T, CH], channel last

        c_ifc = self.ifc_out(c)  # [N, C*O*2, T], O: df_order
        c_cov = self.cov_out(c)  # [N, C*O**2*2, T], O: df_order

        ifc = (
            c_ifc.view(batch, self.df_bins, self.df_order * 2, nframes).permute(
                0, 1, 3, 2
            )
            + c0_ifc
        )
        cov = (
            c_cov.view(batch, self.df_bins, self.df_order**2 * 2, nframes).permute(
                0, 1, 3, 2
            )
            + c0_cov
        )
        return ifc, cov
