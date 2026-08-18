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

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from puresound.audio.dsp import wav_resampling
from puresound.audio.io import AudioIO
from puresound.nnet.masker import Masker

from .base import BaseLightningModule, invoke_loss


class EncDecMaskBase(BaseLightningModule):
    """Mask-based (or mapping-based) enhancement trainer.

    Structure:
        Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Restore Features -> Decoder -> Wav

    Core args:
        encoder: STFT/Conv1D based encode/decode structure
        feats: feature transform between encoder and backbone
        backbone: model backbone that predicts the mask
        mask_type: mask domain (``real`` / ``complex`` / ``polar`` / ``mapping``...)
        *_lr_factor: per-module learning-rate multipliers for the optimizer's
            parameter groups

    Optional training features (all off by default, each a config knob):
        train_vad_head_only + gate_head_lr_factor:
            freeze encoder/features/backbone (kept in eval so BatchNorm
            statistics and dropout stay fixed) and train only the backbone's
            frame-level VAD gate head.
        channel_consistency:
            with prob ``prob`` per training step, re-run the forward on a
            channel-perturbed copy of the mixture (smooth random EQ + gain =
            the physical form of a device recording chain, applied to the WHOLE
            signal so the near/far level contrast is preserved and the ideal
            complex ratio mask is invariant) and penalize the mask for changing
            -- the recording chain must not change the keep/suppress decision.

    Optional inference knobs (forward() args, no effect on training):
        dry_blend, spec_floor -- over-suppression relief; see forward().
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
        channel_consistency: Optional[dict] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        # --- core enhancement pipeline -------------------------------------
        self.encoder = encoder
        self.feats = feats
        self.backbone = backbone
        self.mask_type = mask_type.lower()
        self.encoder_lr_factor = encoder_lr_factor
        self.feats_lr_factor = feats_lr_factor
        self.backbone_lr_factor = backbone_lr_factor

        # --- optional: gate-head-only training -----------------------------
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

        # --- optional: channel-perturbation mask consistency ----------------
        self.channel_consistency = (
            dict(channel_consistency)
            if channel_consistency and channel_consistency.get("enabled", False)
            else None
        )

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

        # Side output for the channel-consistency regularizer (training only;
        # overwritten by every forward, so read it right after the call).
        self.last_mask = mask

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
            raise ValueError(
                f"unknown mask_type {self.mask_type!r}; choose one of "
                "complex, deepfilter, wiener, mvdr, mapping."
            )

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

    def _random_channel_perturb(self, wav: torch.Tensor) -> torch.Tensor:
        """A random plausible recording chain applied to the whole mixture.

        Zero-phase smooth EQ (low-order random cosine series over frequency,
        larger swing at lower orders => tilt-like coloration) plus a per-row
        gain. Real positive H(f) scales every source in the mixture equally, so
        the ideal complex ratio mask is unchanged -- any mask change under this
        perturbation is channel sensitivity, which the consistency loss
        penalizes. No gradient flows through the perturbation itself."""
        cc = self.channel_consistency or {}
        eq_db = float(cc.get("eq_db", 6.0))
        gain_db = float(cc.get("gain_db", 4.0))
        n_orders = int(cc.get("eq_orders", 4))
        with torch.no_grad():
            n, t = wav.shape[0], wav.shape[-1]
            spec = torch.fft.rfft(wav.view(n, -1), dim=-1)
            n_bins = spec.shape[-1]
            grid = torch.linspace(0.0, np.pi, n_bins, device=wav.device)
            curve = torch.zeros(n, n_bins, device=wav.device)
            for k in range(1, n_orders + 1):
                amp = (torch.rand(n, 1, device=wav.device) * 2.0 - 1.0) * (eq_db / k)
                curve = curve + amp * torch.cos(k * grid).unsqueeze(0)
            h = 10.0 ** (curve / 20.0)
            out = torch.fft.irfft(spec * h, n=t, dim=-1)
            g = (torch.rand(n, 1, device=wav.device) * 2.0 - 1.0) * gain_db
            out = out * 10.0 ** (g / 20.0)
            return out.clamp(-1.0, 1.0).view_as(wav)

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
        # iSDR (in losses that declare `inactive_labels`) avoids the
        # degenerate 10*log10(0/X) explosion of vanilla SDR on zero refs.
        inactive_labels = target.abs().amax(dim=-1) == 0

        providers = self._loss_providers(
            enhanced=enhanced,
            target=target,
            vad_target=vad_target,
            batch=batch,
            inactive_labels=inactive_labels,
        )
        return self.reduce_losses(lambda loss: invoke_loss(loss, providers))

    def _loss_providers(
        self, *, enhanced, target, vad_target, batch, inactive_labels
    ) -> dict:
        """Everything this module can hand a loss, by name.

        A loss declares which of these it wants (`required_inputs`) and
        `invoke_loss` calls it with exactly those, in that order. One table, so
        "what a loss may ask for" has a single definition -- the test that checks
        every shipped loss against it reads this method rather than a second
        list that could drift from it.

        Callables, not values: a side output nothing asked for is never read off
        the backbone, and the background target is only synthesized when the loss
        that needs it is registered.
        """

        def side(name):
            # Produced during the forward() that precedes this call. Nothing
            # populates `last_background_vad_logits` today -- the head that did
            # left with the conformer axis (380da2e) while the gate
            # infrastructure was kept for reuse -- so the provider stays wired
            # and BackgroundVADHeadBCELoss raises naming the head it wants.
            return getattr(self.backbone, name, None)

        def background_vad_target():
            explicit = None if batch is None else batch.get("background_vad_target")
            # When no sample in the batch carries background speech the dataset
            # emits no `background_vad_reference` and the collate produces no
            # `background_vad_target` at all (it only zero-fills missing rows
            # when *some* row has it). An all-silent batch is a valid signal --
            # the background-activity target is simply all-zeros -- so
            # synthesize it rather than crashing the loss on a None target.
            logits = side("last_background_vad_logits")
            if explicit is None and logits is not None:
                return torch.zeros_like(logits)
            return explicit

        return {
            "enhanced": lambda: enhanced,
            "target": lambda: target,
            "batch": lambda: batch or {},
            "inactive_labels": lambda: inactive_labels,
            "vad_target": lambda: vad_target,
            "vad_logits": lambda: side("last_vad_logits"),
            "background_vad_logits": lambda: side("last_background_vad_logits"),
            "background_vad_target": background_vad_target,
            "dist_preds": lambda: side("last_dist_preds"),
        }

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

        # Channel-perturbation mask consistency (see __init__): the clean-view
        # mask is the (detached) teacher; the perturbed view learns to produce
        # the same mask despite the different "recording chain".
        #
        # The firing schedule MUST be rank-synchronized: the extra forward runs
        # SyncBatchNorm all-gathers, so a per-rank random draw here desyncs the
        # ranks' collective sequences and deadlocks DDP until the NCCL watchdog
        # kills the job. batch_idx is in lockstep across ranks -> deterministic
        # modulo schedule with the same average rate as `prob`.
        cc = self.channel_consistency
        if cc is not None:
            period = max(1, int(cc.get("period", 10)))
            fire = (batch_idx % period) < int(round(float(cc.get("prob", 0.0)) * period))
        else:
            fire = False
        if fire:
            mask_ref = getattr(self, "last_mask", None)
            if mask_ref is not None and torch.is_tensor(mask_ref):
                # Sub-batch: the perturbed-view forward keeps its activations
                # alive until backward ON TOP of the main forward's, so a
                # full-batch second pass can double peak memory. max_rows caps
                # the extra activation cost; a regularizer does not need the
                # full batch. Same rows on every rank (rank-synced schedule +
                # leading slice), so collectives stay aligned.
                k = int(cc.get("max_rows", 0)) or noisy_speech.shape[0]
                k = min(k, noisy_speech.shape[0])
                mask_ref = mask_ref[:k].detach()
                self.forward(self._random_channel_perturb(noisy_speech[:k]))
                cons_loss = nn.functional.l1_loss(self.last_mask, mask_ref)
                total_loss = total_loss + float(cc.get("weight", 1.0)) * cons_loss
                self.log("train_step_cons_loss", cons_loss, prog_bar=False,
                         sync_dist=False, on_step=True)

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
        return self.reduce_losses(lambda loss_func: loss_func(pred, target))

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
