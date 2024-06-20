import torch

from .lobe.multiframe import DeepFilter as DF
from .lobe.multiframe import MultiFrameMvdrFilter as Mvdr
from .lobe.multiframe import MultiFrameWienerFilter as Wiener


class Masker:
    @staticmethod
    def envelope_postfiltering_on_cpx_mask(
        tf_rep: torch.Tensor, est_masks: torch.Tensor, tau: float = 0.02
    ):
        """
        Post-filtering on mask values

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_masks: tf-mask shape [N, 2, C, T]

        Returns:
            FloatTensor -- [N, 2, C, T] -- post filter mask
        """
        eps = 1e-12
        torch_pi = torch.acos(torch.zeros(1)).item() * 2
        tf_rep = tf_rep.permute(0, 2, 3, 1).contiguous()
        est_masks = est_masks.permute(0, 2, 3, 1).contiguous()
        mask = (
            torch.view_as_complex(est_masks).abs()
            / torch.view_as_complex(tf_rep).abs().add(eps)
        ).clamp(eps, 1)
        mask_sin = mask * torch.sin(torch_pi * mask / 2).clamp_min(eps)
        pf = (1 + tau) / (1 + tau * mask.div(mask_sin).pow(2))  # [N, C, T]
        est_masks_pf = est_masks.permute(0, 3, 1, 2) * pf
        return est_masks_pf.contiguous()

    @staticmethod
    def apply_real_on_real(
        tf_rep: torch.Tensor, est_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a real-valued mask to a real representation.

        Args:
            tf_rep: feature has shape [N, C, T]
            est_masks: tf-mask shape [N, C, T]

        Returns:
            FloatTensor -- [N, C, T]
        """
        return tf_rep * est_masks

    @staticmethod
    def apply_mag_mask_on_reim(
        tf_rep: torch.Tensor, est_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a real-valued mask to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_masks: tf-mask shape [N, C, T]

        Returns:
            FloatTensor -- [N, C, T, 2]
        """
        mask = torch.stack([est_masks, est_masks], dim=1)
        return tf_rep * mask

    @staticmethod
    def apply_complex_mask_on_reim(
        tf_rep: torch.Tensor,
        est_masks: torch.Tensor,
        postfilter: bool = False,
    ) -> torch.Tensor:
        """
        Applies a complex-valued mask to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_masks: tf-mask shape [N, 2, C, T]
            postfilter: enable envelope postfiltering

        Returns:
            FloatTensor -- [N, 2, C, T]
        """
        if postfilter:
            est_masks = Masker.envelope_postfiltering_on_cpx_mask(
                tf_rep, est_masks, tau=0.02
            )

        real1, imag1 = torch.chunk(tf_rep, chunks=2, dim=1)
        real2, imag2 = torch.chunk(est_masks, chunks=2, dim=1)
        y_real = real1 * real2 - imag1 * imag2
        y_imag = real1 * imag2 + imag1 * real2
        return torch.cat([y_real, y_imag], dim=1)

    @staticmethod
    def apply_df_on_reim(
        tf_rep: torch.Tensor, est_masks: torch.Tensor, num_feats: int, order: int
    ) -> torch.Tensor:
        """
        Applies a complex-valued 1x3 causal deep filtering mask to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_masks: tf-mask shape [N, 2*order, C, T]
            order: frames of multi frames

        Returns:
            FloatTensor -- [N, 2, C, T]
        """
        deepfilter = DF(num_freqs=num_feats, frame_size=order, lookahead=0)

        tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1)
        batch, _, freq, nframe = est_masks.shape

        est_masks = (
            est_masks.permute(0, 2, 3, 1)
            .reshape(batch, freq, nframe, -1, 2)
            .contiguous()
        )  # [N, C, T, 3, 2]
        est_masks = est_masks.permute(0, 3, 2, 1, 4)

        enh = deepfilter(tf_rep, est_masks)
        enh = enh.squeeze(1).permute(0, 3, 2, 1)

        return enh

    @staticmethod
    def apply_wiener(
        tf_rep: torch.Tensor, est_ifc: torch.Tensor, est_cov: torch.Tensor, order: int
    ) -> torch.Tensor:
        """
        Applies a complex-valued 1 x order (N-taps) causal wiener filtering to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_cov: covariance matrix has shape [N, C, T, 2*order^2]
            est_ifc: intra-frame correlation has shape [N, C, T, 2*order]
            order: frames of multi frames

        Returns:
            enh (FloatTensor): enhanced spectrogram of shape [N, 2, C, T]
            nbins (int): df working bin size
        """
        nbins = est_cov.shape[1]

        # reorder chaneel last
        tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1).contiguous()  # [N, 1, T, C, 2]

        est_ifc = est_ifc.permute(0, 2, 1, 3).contiguous()  # [N, T, C, -1]
        est_cov = est_cov.permute(0, 2, 1, 3).contiguous()  # [N, T, C, -1]

        wiener = Wiener(num_freqs=nbins, frame_size=order, lookahead=0)
        enh = wiener(tf_rep, est_ifc, est_cov)  # [N, 1, T, C, 2]
        enh = enh.squeeze(1)  # [N, T, C, 2]
        enh = enh.permute(0, 3, 2, 1)

        return enh, nbins

    @staticmethod
    def apply_mvdr(
        tf_rep: torch.Tensor,
        est_ifc: torch.Tensor,
        est_cov: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """
        Applies a complex-valued 1x5 causal mvdr filtering to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, 2, C, T]
            est_cov: covariance matrix has shape [N, C, T, 2*order^2]
            est_ifc: intra-frame correlation has shape [N, C, T, 2*order]

        Returns:
            enh (FloatTensor): enhanced spectrogram of shape [N, 2, C, T]
            nbins (int): df working bin size
        """
        nbins = est_cov.shape[1]

        # reorder chaneel last
        tf_rep = tf_rep.permute(0, 3, 2, 1).unsqueeze(1).contiguous()  # [N, 1, T, C, 2]

        est_ifc = est_ifc.permute(0, 2, 1, 3).contiguous()  # [N, T, C, -1]
        est_cov = est_cov.permute(0, 2, 1, 3).contiguous()  # [N, T, C, -1]

        wiener = Mvdr(
            num_freqs=nbins,
            frame_size=order,
            lookahead=0,
            enforce_constraints=True,
            cholesky_decomp=True,
        )
        enh = wiener(tf_rep, est_ifc, est_cov)  # [N, 1, T, C, 2]
        enh = enh.squeeze(1)  # [N, T, C, 2]
        enh = enh.permute(0, 3, 2, 1)

        return enh, nbins
