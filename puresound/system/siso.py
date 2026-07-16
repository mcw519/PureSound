"""
Single Input Single Output (SISO) PL-Module

"Single input" counts *acoustic* observations: one noisy waveform in, one
enhanced waveform out. MISO (see ``miso.EncDecCondMaskBase``) is reserved for
the case where a second input is itself an *audio stream* that needs its own
front-end (e.g. an enrollment utterance for target-speaker extraction).

Use cases:
    EncDecMaskBase:
        - Mask based speech enhancement
        - Mapping based speech enhancement
    EncPredClassBase:
        - Speaker embedding
"""

import numpy as np
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
        train_vad_head_only: bool = False,
        gate_head_lr_factor: float = 1.0,
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
        self.train_vad_head_only = bool(train_vad_head_only)
        self.gate_head_lr_factor = float(gate_head_lr_factor)

        if self.train_vad_head_only:
            vad_head = getattr(self.backbone, "vad_head", None)
            if vad_head is None:
                raise ValueError(
                    "train_vad_head_only=True requires an enabled backbone vad_head"
                )
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.feats.parameters():
                parameter.requires_grad_(False)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in vad_head.parameters():
                parameter.requires_grad_(True)

        # Loss

    def train(self, mode: bool = True):
        """Keep the frozen separator deterministic during gate-only training.

        ``requires_grad=False`` does not stop BatchNorm running statistics or
        dropout from changing. Gate-only adaptation must preserve the exact
        separator, so frozen modules stay in eval while the VAD head follows
        the requested mode.
        """
        super().train(mode)
        if self.train_vad_head_only:
            self.encoder.eval()
            self.feats.eval()
            self.backbone.eval()
            self.backbone.vad_head.train(mode)
        return self

    def forward(
        self,
        wav: torch.Tensor,
        dry_blend: float = 1.0,
        spec_floor: float = 0.0,
    ):
        """Run enhancement.

        Args:
            wav: noisy waveform, shape ``[N, T]`` (a leading singleton channel
                is squeezed away).
            dry_blend: inference-only over-suppression relief in ``(0, 1]``.
                Output becomes ``dry_blend * enh + (1 - dry_blend) * input``;
                ``1.0`` (default) is a no-op. Values < 1 mix the original mix
                back to recover deleted target speech, trading a little
                interferer leakage for fewer deletions. Training callers leave
                the default, so the training path is unchanged.
            spec_floor: inference-only spectral over-suppression floor in
                ``[0, 1)``. Clamps each enhanced magnitude bin to at least
                ``spec_floor * |mix bin|`` (keeping the enhanced phase) so the
                mask can never attenuate a bin below that fraction of the
                input. ``0.0`` (default) is a no-op. Complex-mask models only.

        Returns:
            Enhanced waveform clamped to ``[-1, 1]``.
        """
        if wav.dim() != 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)

        features = self.encoder(wav)
        features, features_for_enhanced = self.feats(features)
        mask = self.backbone(features)

        if self.mask_type in ["wiener", "mvdr"]:
            mask, ifc, cov = mask

        if self.mask_type == "complex":
            enh = Masker.apply_complex_mask_on_reim(
                tf_rep=features_for_enhanced, est_masks=mask
            )
            if spec_floor > 0.0:
                enh = self._apply_spec_floor(enh, features_for_enhanced, spec_floor)
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

        enh = self._spec_to_wav(enh)

        if dry_blend < 1.0:
            ref = wav
            if ref.dim() == enh.dim() + 1 and ref.shape[0] == 1:
                ref = ref.squeeze(0)
            n = min(enh.shape[-1], ref.shape[-1])
            blended = dry_blend * enh[..., :n] + (1.0 - dry_blend) * ref[..., :n]
            enh = enh.clone()
            enh[..., :n] = torch.clamp(blended, min=-1.0, max=1.0)

        return enh

    def _spec_to_wav(self, enh: torch.Tensor) -> torch.Tensor:
        """iSTFT an enhanced [N,2,F,T] spectrum back to a clamped waveform.
        Shared by the near output and the far-parent decoder output."""
        enh = self.feats.back_forward(enh)  # [N, CH, C, T]
        if enh.dim() == 4:
            enh = enh.permute(0, 2, 3, 1)
        else:
            enh = enh.squeeze(1)
        enh = self.encoder.inverse(enh)
        return torch.clamp_(enh, min=-1, max=1)

    @staticmethod
    def _apply_spec_floor(
        enh: torch.Tensor, mix_tf: torch.Tensor, floor: float
    ) -> torch.Tensor:
        """Floor the enhanced magnitude to ``>= floor * |mix|`` per bin while
        keeping the enhanced phase. ``enh`` and ``mix_tf`` are real/imag stacked
        on dim=1 with shape ``[N, 2, C, T]``. No-op where the mask did not
        over-suppress (enh already above the floor)."""
        eps = 1e-8
        er, ei = torch.chunk(enh, chunks=2, dim=1)
        mr, mi = torch.chunk(mix_tf, chunks=2, dim=1)
        enh_mag = torch.sqrt(er * er + ei * ei + eps)
        mix_mag = torch.sqrt(mr * mr + mi * mi + eps)
        target_mag = torch.maximum(enh_mag, floor * mix_mag)
        scale = target_mag / enh_mag
        return torch.cat([er * scale, ei * scale], dim=1)

    def compute_loss(
        self,
        enhanced: torch.Tensor,
        target: torch.Tensor,
        vad_target: torch.Tensor | None = None,
        batch: dict | None = None,
    ):
        # wav aligned length
        if enhanced.shape[-1] < target.shape[-1]:
            target = target[..., : enhanced.shape[-1]]
        else:
            enhanced = enhanced[..., : target.shape[-1]]

        # Per-row "target absent" mask. Rows whose reference is fully silent
        # come from the target-absent training path; routing them through
        # iSDR (in losses that opt in via uses_inactive_labels) avoids the
        # degenerate 10*log10(0/X) explosion of vanilla SDR on zero refs.
        inactive_labels = target.abs().amax(dim=-1) == 0

        # VAD-head logits are produced as a side output of the backbone during
        # the forward() that precedes this call; route them to any loss that
        # opts in via uses_vad_logits (e.g. VADHeadBCELoss).
        vad_logits = getattr(self.backbone, "last_vad_logits", None)
        background_vad_logits = getattr(
            self.backbone,
            "last_background_vad_logits",
            None,
        )

        overall_loss = []
        losses = []
        for idx, loss_func in enumerate(self.loss_func_list):
            weighted = self.loss_func_list_w[idx]
            if getattr(loss_func, "uses_vad_logits", False):
                weighted_loss = weighted * loss_func(vad_logits, vad_target)
            elif getattr(loss_func, "uses_background_vad_logits", False):
                bg_target = (
                    None if batch is None else batch.get("background_vad_target")
                )
                # When no sample in the batch carries background speech, the
                # dataset emits no `background_vad_reference` and the collate
                # produces no `background_vad_target` at all (it only zero-fills
                # missing rows when *some* row has background speech). An
                # all-silent batch is a valid signal -- the background-activity
                # target is simply all-zeros -- so synthesize it rather than
                # crashing BackgroundVADHeadBCELoss on a None target.
                if bg_target is None and background_vad_logits is not None:
                    bg_target = torch.zeros_like(background_vad_logits)
                weighted_loss = weighted * loss_func(
                    background_vad_logits,
                    bg_target,
                )
            elif getattr(loss_func, "uses_batch", False):
                weighted_loss = weighted * loss_func(enhanced, target, batch or {})
            elif getattr(loss_func, "uses_vad_target", False):
                weighted_loss = weighted * loss_func(
                    enhanced, target, vad_target=vad_target
                )
            elif getattr(loss_func, "uses_inactive_labels", False):
                weighted_loss = weighted * loss_func(
                    enhanced, target, inactive_labels=inactive_labels
                )
            else:
                weighted_loss = weighted * loss_func(enhanced, target)
            losses.append(weighted_loss.item())
            if idx == 0:
                overall_loss = weighted_loss
            else:
                overall_loss += weighted_loss

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        batch = self.ensure_vad_targets(batch)
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech,
            target=clean_speech,
            vad_target=batch.get("vad_target"),
            batch=batch,
        )
        # sync_dist stays False for the progress-bar metric on purpose: a
        # synced (all-reduced) metric read by the progress bar deadlocks DDP,
        # because the bar's refresh -- and thus the metric access that triggers
        # the all-reduce -- is not in lockstep across ranks, so one rank enters
        # the metric collective while the other is already at the next
        # iteration's collective. The displayed value is just this rank's loss.
        self.log("train_step_loss", total_loss, prog_bar=True, sync_dist=False)
        if self.verbose:
            if len(losses) != 1:
                for i in range(len(losses)):
                    self.log(
                        f"train_step_loss_{i}",
                        losses[i],
                        prog_bar=False,
                        sync_dist=False,
                        on_step=True,
                    )
        self.puresound_logging.update({"epoch_train_loss": total_loss.item()})
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        batch = self.ensure_vad_targets(batch)
        noisy_speech = batch["noisy_speech"]
        clean_speech = batch["clean_speech"]
        enhanced_speech = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(
            enhanced=enhanced_speech,
            target=clean_speech,
            vad_target=batch.get("vad_target"),
            batch=batch,
        )
        if len(losses) != 1:
            for i in range(len(losses)):
                self.log(
                    f"valid_step_loss_{i}",
                    losses[i],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.log(
            "valid_step_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        """Each metrics has its working sample rate."""
        batch = self.ensure_vad_targets(batch)
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
            if isinstance(score, dict):
                self.puresound_logging.update(score)
            else:
                self.puresound_logging.update({name: score})

    def predict_step(self, batch, batch_idx):
        batch = self.ensure_vad_targets(batch)
        noisy_speech = batch["noisy_speech"]
        input_sr = batch["sr"]
        enhanced_speech = self.forward(noisy_speech)
        AudioIO.save(
            wav=enhanced_speech.detach().cpu(),
            f_path=f"{self.eval_output_folder_path}/{batch['name'][0]}.wav",
            sr=input_sr,
        )

    def get_total_param_groups(self):
        if self.train_vad_head_only:
            return {
                "gate_head": {
                    "params": self.backbone.vad_head.parameters(),
                    "lr_factor": self.gate_head_lr_factor,
                }
            }

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


class EncPredClassBase(BaseLightningModule):
    """
    Structure:
        Wav -> Encoder -> Features -> Backbone -> Predict classes

    Args:
        encoder: STFT/Conv1D based encode/decode structure
        backbone: model backbone to predict classes
    """

    def __init__(
        self,
        encoder: nn.Module,
        feats: nn.Module,
        backbone: nn.Module,
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

        # Parameter
        self.encoder_lr_factor = encoder_lr_factor
        self.feats_lr_factor = feats_lr_factor
        self.backbone_lr_factor = backbone_lr_factor

        # Loss

    def forward(self, wav: torch.Tensor):
        if wav.dim() != 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)

        features = self.encoder(wav)
        features, _ = self.feats(features)
        features = features.squeeze(1)
        pred = self.backbone(features)
        return pred

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        overall_loss = []
        losses = []
        for idx, loss_func in enumerate(self.loss_func_list):
            weighted = self.loss_func_list_w[idx]
            weighted_loss = weighted * loss_func(pred, target)
            losses.append(weighted_loss.item())
            if idx == 0:
                overall_loss = weighted_loss
            else:
                overall_loss += weighted_loss

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        target = batch["target"]
        pred = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(pred=pred, target=target)
        # sync_dist stays False for the progress-bar metric on purpose: a
        # synced (all-reduced) metric read by the progress bar deadlocks DDP,
        # because the bar's refresh -- and thus the metric access that triggers
        # the all-reduce -- is not in lockstep across ranks, so one rank enters
        # the metric collective while the other is already at the next
        # iteration's collective. The displayed value is just this rank's loss.
        self.log("train_step_loss", total_loss, prog_bar=True, sync_dist=False)
        if self.verbose:
            if len(losses) != 1:
                for i in range(len(losses)):
                    self.log(
                        f"train_step_loss_{i}",
                        losses[i],
                        prog_bar=False,
                        sync_dist=False,
                        on_step=True,
                    )
        self.puresound_logging.update({"epoch_train_loss": total_loss.item()})
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        target = batch["target"]
        pred = self.forward(noisy_speech)
        total_loss, losses = self.compute_loss(pred=pred, target=target)
        if len(losses) != 1:
            for i in range(len(losses)):
                self.log(
                    f"valid_step_loss_{i}",
                    losses[i],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.log(
            "valid_step_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": total_loss}

    def test_step(self, batch, batch_idx):
        """Each metrics has its working sample rate."""
        noisy_speech = batch["noisy_speech"]
        target = batch["target"]
        pred = self.forward(noisy_speech)

        # Move tensor to cpu
        noisy_speech = noisy_speech.cpu()
        pred = pred.cpu()

        # Compute each score in registered metrics funcs
        for name in sorted(self._metrics_func.keys()):
            score = self._metrics_func[name]["func"](pred, target)
            self.puresound_logging.update({name: score})

    def predict_step(self, batch, batch_idx):
        noisy_speech = batch["noisy_speech"]
        pred = self.forward(noisy_speech)
        pred = nn.functional.normalize(pred, p=2, dim=1)
        if pred.shape[0] != 1:
            pred = pred.mean(dim=0, keepdim=True)
        pred = pred.squeeze()
        pred = (pred.cpu().numpy().astype("float32"),)
        np.savetxt(
            fname=f"{self.eval_output_folder_path}/{batch['name'][0]}.txt",
            X=pred,
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
        for i in range(len(self.loss_func_list)):
            overall_params[f"loss{i}"] = {
                "params": self.loss_func_list[i].parameters(),
                "lr_factor": 1.0,
            }
        return overall_params
