"""
Single Input Single Ouput (SISO) PL-Module

Use cases:
    - Mask based speech enhancement
"""
import torch
import torch.nn as nn

from puresound.audio.dsp import wav_resampling
from puresound.audio.io import AudioIO
from puresound.nnet.masker import Masker

from .base import BaseLightningModule


class EncDecMaskBase(BaseLightningModule):
    """
    Structure:
        Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Restore Features -> Decoder -> Wav

    Args:
        encoder: STFT/Conv1D based encode/decode structure
        backbone: model backbone to predict mask
        mask_type: mask type choose
    """

    def __init__(
        self,
        encoder: nn.Module,
        feats: nn.Module,
        backbone: nn.Module,
        mask_type: str = "complex",
        encoder_lr_factor: float = 1.0,
        feats_lr_factor: float = 1.0,
        backbone_lr_factor: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        # Model
        self.encoder = encoder
        self.feats = feats
        self.backbone = backbone

        # Feature
        self.mask_type = mask_type.lower()

        # Parameter
        self.encoder_lr_factor = encoder_lr_factor
        self.feats_lr_factor = feats_lr_factor
        self.backbone_lr_factor = backbone_lr_factor

        # Loss

    def forward(self, wav: torch.Tensor):
        features = self.encoder(wav)
        features, features_for_enhanced = self.feats(features)
        mask = self.backbone(features)

        if self.mask_type in ["wiener", "mvdr"]:
            mask, ifc, cov = mask

        if self.mask_type == "complex":
            enh = Masker.apply_complex_mask_on_reim(
                tf_rep=features_for_enhanced, est_masks=mask
            )
        elif self.mask_type == "deepfilter":
            n_filter = mask.shape[2]
            n_order = int(mask.shape[1] / 2)
            enh = Masker.apply_df_on_reim(
                tf_rep=features_for_enhanced,
                est_masks=mask,
                num_feats=n_filter,
                order=n_order,
            )
        elif self.mask_type == "wiener":
            n_order = int(ifc.shape[-1] / 2)
            enh = Masker.apply_complex_mask_on_reim(
                tf_rep=features_for_enhanced, est_masks=mask
            )
            enh_filter, n_bins = Masker.apply_wiener(
                tf_rep=features_for_enhanced, est_ifc=ifc, est_cov=cov, order=n_order
            )
            enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]
        elif self.mask_type == "mvdr":
            n_order = int(ifc.shape[-1] / 2)
            enh = Masker.apply_complex_mask_on_reim(
                tf_rep=features_for_enhanced, est_masks=mask
            )
            enh_filter, n_bins = Masker.apply_mvdr(
                tf_rep=features_for_enhanced, est_ifc=ifc, est_cov=cov, order=n_order
            )
            enh[:, :, :n_bins, :] = enh_filter[:, :, :n_bins, :]
        else:
            raise NameError

        enh = self.feats.back_forward(enh)  # [N, CH, C, T]

        if enh.dim() == 4:
            enh = enh.permute(0, 2, 3, 1)
        else:
            enh = enh.squeeze(1)

        enh = self.encoder.inverse(enh)
        enh = torch.clamp_(enh, min=-1, max=1)
        return enh

    def compute_loss(self, enhanced: torch.Tensor, target: torch.Tensor):
        # wav aligned length
        if enhanced.shape[-1] < target.shape[-1]:
            target = target[..., : enhanced.shape[-1]]
        else:
            enhanced = enhanced[..., : target.shape[-1]]

        overall_loss = []
        losses = []
        for idx, loss_func in enumerate(self.loss_func_list):
            loss_func, weighted = loss_func
            weighted_loss = weighted * loss_func(enhanced, target)
            losses.append(weighted_loss.item())
            if idx == 0:
                overall_loss = weighted_loss
            else:
                overall_loss += weighted_loss

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech, target=clean_speech
        )
        self.log("train_step_loss", total_loss, prog_bar=True, sync_dist=True)
        if self.verbose:
            if len(losses) != 1:
                for i in range(len(losses)):
                    self.log(
                        f"train_step_loss_{i}",
                        losses[i],
                        prog_bar=False,
                        sync_dist=True,
                        on_step=True,
                    )
        self.puresound_logging.update({"epoch_train_loss": total_loss.item()})
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech, target=clean_speech
        )
        if len(losses) != 1:
            for i in range(len(losses)):
                self.log(
                    f"valid_step_loss_{i}",
                    losses[i],
                    prog_bar=False,
                    sync_dist=True,
                    on_step=True,
                )
        self.log(
            "valid_step_loss", total_loss, prog_bar=True, sync_dist=True, on_step=True
        )
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        """Each metrics has its working sample rate."""
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        input_sr = batch["sr"]
        enhanced_speech = self.forward(noisy_speech)

        # Move tensor to cpu
        clean_speech = clean_speech.cpu()
        noisy_speech = noisy_speech.cpu()
        enhanced_speech = enhanced_speech.cpu()
        input_sr = input_sr.cpu()

        # Compute each score in registered metrics funcs
        for name in sorted(self._metrics_func.keys()):
            # Avoid double resampling error
            _clean_speech = clean_speech.clone()
            _enhanced_speech = enhanced_speech.clone()
            if (
                self._metrics_func[name]["sr"] is not None
                and self._metrics_func[name]["sr"] != input_sr
            ):
                _clean_speech, _ = wav_resampling(
                    wav=_clean_speech,
                    origin_sr=input_sr,
                    target_sr=self._metrics_func[name]["sr"],
                    backend="sox",
                )
                _enhanced_speech, _ = wav_resampling(
                    wav=_enhanced_speech,
                    origin_sr=input_sr,
                    target_sr=self._metrics_func[name]["sr"],
                    backend="sox",
                )

            score = self._metrics_func[name]["func"](_clean_speech, _enhanced_speech)
            self.puresound_logging.update({name: score})

    def predict_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        input_sr = batch["sr"]
        enhanced_speech = self.forward(noisy_speech)
        AudioIO.save(
            wav=enhanced_speech,
            f_path=f"{self.eval_output_folder_path}/{batch['name'][0]}.wav",
            sr=input_sr,
        )

    def get_total_param_groups(self):
        overall_params = {}
        overall_params["encoder"] = {
            "params": self.encoder.parameters(),
            "lr_factor": self.encoder_lr_factor,
        }
        overall_params["feats"] = {
            "params": self.feats.parameters(),
            "lr_factor": self.feats_lr_factor,
        }
        overall_params["backbone"] = {
            "params": self.backbone.parameters(),
            "lr_factor": self.backbone_lr_factor,
        }
        return overall_params
