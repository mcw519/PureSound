"""
Multiple Input Single Ouput (MISO) PL-Module

Use cases:
    EncDecCondMaskBase:
        - Personalized Speech Enhancement / Target Speaker Extraction
"""

from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from puresound.audio.dsp import wav_resampling
from puresound.audio.io import AudioIO
from puresound.nnet.masker import Masker

from .base import BaseLightningModule


class EncDecCondMaskBase(BaseLightningModule):
    """
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
        c_backbone: nn.Module,
        jointed_trained: bool = True,
        siamese_encoder: bool = True,
        siamese_feats: bool = True,
        c_encoder: Optional[nn.Module] = None,
        c_feats: Optional[nn.Module] = None,
        mask_type: str = "complex",
        encoder_lr_factor: float = 1.0,
        feats_lr_factor: float = 1.0,
        backbone_lr_factor: float = 1.0,
        c_encoder_lr_factor: float = 1.0,
        c_feats_lr_factor: float = 1.0,
        c_backbone_lr_factor: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        # Model
        self.encoder = encoder
        self.feats = feats
        self.backbone = backbone
        self.siamese_encoder = siamese_encoder
        self.siamese_feats = siamese_feats
        self.c_backbone = c_backbone
        if siamese_encoder:
            self.c_encoder = deepcopy(encoder)
        else:
            self.c_encoder = c_encoder

        if siamese_feats:
            self.c_feats = deepcopy(feats)
        else:
            self.c_feats = c_feats

        self.jointed_trained = jointed_trained

        if not jointed_trained:
            self.c_backbone.eval()
            self.c_encoder.eval()
            self.c_feats.eval()

        # Feature
        self.mask_type = mask_type.lower()

        # Parameter
        self.encoder_lr_factor = encoder_lr_factor
        self.feats_lr_factor = feats_lr_factor
        self.backbone_lr_factor = backbone_lr_factor
        self.c_encoder_lr_factor = c_encoder_lr_factor
        self.c_feats_lr_factor = c_feats_lr_factor
        self.c_backbone_lr_factor = c_backbone_lr_factor

    def register_loss_func(
        self,
        loss_func_list: nn.ModuleList,
        loss_func_list_weights: List,
        c_loss_func_list: Optional[nn.ModuleList] = None,
        c_loss_func_list_weights: Optional[List] = None,
    ):
        self.loss_func_list = loss_func_list
        self.loss_func_list_w = loss_func_list_weights
        if c_loss_func_list is not None:
            self.c_loss_func_list = c_loss_func_list
        else:
            self.c_loss_func_list = None
        if c_loss_func_list_weights is not None:
            self.c_loss_func_list_w = c_loss_func_list_weights
        else:
            self.c_loss_func_list_w = None

    def forward(self, wav: torch.Tensor, conditional_wav: torch.Tensor):
        if wav.dim() != 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)

        if conditional_wav.dim() != 2 and conditional_wav.shape[0] == 1:
            conditional_wav = conditional_wav.squeeze(0)

        features = self.encoder(wav)
        features, features_for_enhanced = self.feats(features)

        if not self.jointed_trained:
            with torch.no_grad():
                c_features = self.c_encoder(conditional_wav)
                c_features, _ = self.c_feats(c_features)
                c_features = c_features.squeeze(1)
                c_features = self.c_backbone(c_features)
        else:
            c_features = self.c_encoder(conditional_wav)
            c_features, _ = self.c_feats(c_features)
            c_features = c_features.squeeze(1)
            c_features = self.c_backbone(c_features)

        mask = self.backbone(features, c_features)

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
        elif self.mask_type == "mapping":
            enh = mask
        else:
            raise NameError

        enh = self.feats.back_forward(enh)  # [N, CH, C, T]

        if enh.dim() == 4:
            enh = enh.permute(0, 2, 3, 1)
        else:
            enh = enh.squeeze(1)

        enh = self.encoder.inverse(enh)
        enh = torch.clamp_(enh, min=-1, max=1)
        return enh, c_features

    def compute_loss(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        vad_target: torch.Tensor | None = None,
    ):
        # wav aligned length
        if enhanced.shape[-1] < target.shape[-1]:
            target = target[..., : enhanced.shape[-1]]
        else:
            enhanced = enhanced[..., : target.shape[-1]]

        losses = []
        for idx, loss_func in enumerate(self.loss_func_list):
            weighted = self.loss_func_list_w[idx]
            if getattr(loss_func, "uses_vad_target", False):
                weighted_loss = weighted * loss_func(
                    enhanced, target, vad_target=vad_target
                )
            else:
                weighted_loss = weighted * loss_func(enhanced, target)
            losses.append(weighted_loss.item())
            if idx == 0:
                overall_loss = weighted_loss
            else:
                overall_loss += weighted_loss

        return overall_loss, losses

    def compute_loss2(self, pred: torch.Tensor, target: torch.Tensor):
        assert self.c_loss_func_list is not None

        losses = []
        for idx, loss_func in enumerate(self.c_loss_func_list):
            weighted = self.c_loss_func_list_w[idx]
            weighted_loss = weighted * loss_func(pred, target)
            losses.append(weighted_loss.item())
            if idx == 0:
                overall_loss = weighted_loss
            else:
                overall_loss += weighted_loss

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        conditional_speech = batch["conditional_speech"]
        conditional_target = batch["target"]
        enhanced_speech, embedding = self.forward(noisy_speech, conditional_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech,
            target=clean_speech,
            vad_target=batch.get("vad_target"),
        )
        if self.jointed_trained and self.c_loss_func_list is not None:
            total_loss2, losses2 = self.compute_loss2(
                pred=embedding, target=conditional_target
            )
            total_loss += total_loss2
            losses += losses2

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
        conditional_speech = batch["conditional_speech"]
        enhanced_speech, embedding = self.forward(noisy_speech, conditional_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech,
            target=clean_speech,
            vad_target=batch.get("vad_target"),
        )
        # if self.jointed_trained and self.c_loss_func_list is not None:
        #     total_loss2, losses2 = self.compute_loss2(
        #         pred=embedding, target=conditional_target
        #     )
        #     total_loss += total_loss2
        #     losses += losses2

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
        conditional_speech = batch["conditional_speech"]
        input_sr = batch["sr"]
        enhanced_speech, embedding = self.forward(noisy_speech, conditional_speech)

        # Move tensor to cpu
        clean_speech = clean_speech.cpu()
        noisy_speech = noisy_speech.cpu()
        enhanced_speech = enhanced_speech.cpu()
        embedding = embedding.cpu()
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
            if isinstance(score, dict):
                self.puresound_logging.update(score)
            else:
                self.puresound_logging.update({name: score})

    def predict_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        conditional_speech = batch["conditional_speech"]
        input_sr = batch["sr"]
        enhanced_speech, embedding = self.forward(noisy_speech, conditional_speech)
        AudioIO.save(
            wav=enhanced_speech.detach().cpu(),
            f_path=f"{self.eval_output_folder_path}/{batch['name'][0]}.wav",
            sr=input_sr,
        )
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        if embedding.shape[0] != 1:
            embedding = embedding.mean(dim=0, keepdim=True)
        embedding = embedding.squeeze()
        embedding = (embedding.cpu().numpy().astype("float32"),)
        np.savetxt(
            fname=f"{self.eval_output_folder_path}/{batch['name'][0]}.txt",
            X=embedding,
            fmt="%5.11f",
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
        if self.jointed_trained:
            overall_params["c_encoder"] = {
                "params": self.c_encoder.parameters(),
                "lr_factor": self.c_encoder_lr_factor,
            }
            overall_params["c_feats"] = {
                "params": self.c_feats.parameters(),
                "lr_factor": self.c_feats_lr_factor,
            }
            overall_params["c_backbone"] = {
                "params": self.c_backbone.parameters(),
                "lr_factor": self.c_backbone_lr_factor,
            }
        return overall_params
