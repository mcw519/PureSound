from typing import Any, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from puresound.nnet.masker import Masker

from .logger import Logging


class BaseLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self._optimizer = None
        self._scheduler = None
        self.epoch_log = Logging()

    def forward(self):
        """Forward to get results"""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def register_loss_func(self, loss_func: Any):
        self.loss_func = loss_func

    def register_optimizer(self, optimizer: Any):
        self._optimizer = optimizer

    def register_scheduler(self, scheduler: Any):
        self._scheduler = scheduler

    def configure_optimizers(self):
        if self._scheduler is not None:
            return [self._optimizer], [self._scheduler]
        else:
            return self._optimizer
    
    def on_validation_epoch_end(self):
        scores = self.epoch_log.average()
        for key in scores:
            self.log(key, scores[key], prog_bar=True, sync_dist=False)
        
        self.epoch_log.clear()


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
    ):
        super().__init__()
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

        return self.loss_func(enhanced, target)

    def training_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        loss = self.compute_loss(enhanced=enhanced_speech, target=clean_speech)
        self.log("train_step_loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        loss = self.compute_loss(enhanced=enhanced_speech, target=clean_speech)
        self.log("valid_step_loss", loss, prog_bar=True, sync_dist=True)
        self.epoch_log.update({"epoch_val": loss.item()})
        return {"loss": loss}

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
