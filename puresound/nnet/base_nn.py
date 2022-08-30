from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lobe.encoder import ConvEncDec


class BaseModel(nn.Module):
    """Crescendo nnet basic model."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def overall_parameters(self) -> int:
        """count and return overall parameters in model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def overall_trainable_parameters(self) -> int:
        """count and return overall trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()
    

class EncDecMaskerBaseModel(BaseModel):
    """Abstract class define E2E model which using tf-mask approach."""
    def __init__(self) -> None:
        super().__init__()

    def apply_tf_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor, mask_type: str, f_type: str) -> torch.Tensor:
        """
        Applies tf-masks on time-frequency feature representation.

        Args:
            tf_rep: feature must in type (Complex, MagPhase) has [N, C, T] or [N, C*2, T]
            est_masks: tf-mask shape [N, C, T], or [N, C*2, T]
            mask_type: mask type in (`complex`, `real`, `polar`)
            f_type: feature type in (`complex`, `real`, `polar`)
        
        Returns:
            Masked time-frequency representations.
        """
        if mask_type.lower() == 'complex' and f_type.lower() == 'complex':
            re, im = torch.chunk(tf_rep, 2, dim=1)
            tf_rep = torch.stack([re, im], dim=-1) # [N, C, T, 2]
            mask_re, mask_im = torch.chunk(est_masks, 2, dim=1)
            est_masks = torch.stack([mask_re, mask_im], dim=-1) # [N, C, T, 2]
            return self._apply_complex_mask_on_reim(tf_rep, est_masks)
        
        elif mask_type.lower() == 'real' and f_type.lower() == 'complex':
            re, im = torch.chunk(tf_rep, 2, dim=1)
            tf_rep = torch.stack([re, im], dim=-1) # [N, C, T, 2]
            return self._apply_mag_mask_on_reim(tf_rep, est_masks)

        elif mask_type.lower() == 'real' and f_type.lower() == 'real':
            return self._apply_mag_mask_on_mag(tf_rep, est_masks)

        elif mask_type.lower() == 'polar' and f_type.lower() == 'polar':
            re, im = torch.chunk(tf_rep, 2, dim=1)
            tf_rep = torch.stack([re, im], dim=-1) # [N, C, T, 2]
            mask_re, mask_im = torch.chunk(est_masks, 2, dim=1)
            est_masks = torch.stack([mask_re, mask_im], dim=1) # [N, C, T, 2]
            return self._apply_complex_mask_on_polar (tf_rep, est_masks)
        
        else:
            raise NameError
    
    def get_mask(self, mask: torch.Tensor, mask_constraint: str = 'linear') -> torch.Tensor:
        """forward feature to get tf-mask."""
        if mask_constraint.lower() == 'linear':
            return mask
        
        elif mask_constraint.lower() == 'relu':
            return torch.relu(mask)
        
        elif mask_constraint.lower() == 'sigmoid':
            return torch.sigmoid(mask)
        
        else:
            raise NotImplementedError
    
    def _mul_c(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Entrywise product for complex valued tensors.
        
        Args:
            x1: complex tensor has shape [N, C, T, 2], the last dimension is real/imag
            x2: complex tensor has shape [N, C, T, 2], the last dimension is real/imag
                    
        Returns:
            return complex product output
        """
        real1, imag1 = x1[..., 0], x1[..., 1]
        real2, imag2 = x2[..., 0], x2[..., 1]
        y_real = real1 * real2 - imag1 * imag2
        y_imag = real1 * imag2 + imag1 * real2
        return torch.stack([y_real, y_imag], dim=-1)
    
    def _apply_mag_mask_on_reim(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """
        Applies a real-valued mask to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, C, T, 2]
            est_masks: tf-mask shape [N, C, T] or [N, C*2, T]

        Returns:
            FloatTensor -- [N, C, T, 2]
        """
        if mask.shape != tf_rep.shape:
            mask = torch.stack([est_masks, est_masks], dim=-1)
        return tf_rep * mask
    
    def _apply_complex_mask_on_reim(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """
        Applies a complex-valued mask to a complex-valued representation.

        Args:
            tf_rep: feature has shape [N, C, T, 2]
            est_masks: tf-mask shape [N, C, T, 2]
        
        Returns:
            FloatTensor -- [N, C, T, 2]
        """
        return self._mul_c(tf_rep, est_masks)
    
    def _apply_mag_mask_on_mag(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """
        Applies a real-valued mask to a magnitude representation.
        
        Args:
            tf_rep: feature has shape [N, C, T]
            est_masks: tf-mask shape [N, C, T]
        
        Returns:
            FloatTensor -- [N, C, T]
        """
        return tf_rep * est_masks
    
    def _apply_complex_mask_on_polar(self, tf_rep: torch.Tensor, est_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies a complex-valued mask to a complex-valued representation with polar form.

        Args:
            tf_rep: feature has shape [N, C, T, 2]
            est_masks: tf-mask shape [N, C, T, 2]
        
        Returns:
            FloatTensor -- [N, C, T, 2]
        """
        re, im = tf_rep[..., 0], tf_rep[..., 1]
        tf_mag = torch.sqrt(re**2 + im**2 + 1e-8)
        tf_phase = torch.atan2(im, re)

        mask_re, mask_im = est_mask[..., 0], est_mask[..., 1]
        mask_mag = torch.sqrt(mask_re**2 + mask_im**2 + 1e-8)
        mask_re_phase = mask_re / (mask_mag + 1e-8)
        mask_im_phase = mask_im / (mask_mag + 1e-8)
        mask_phase = torch.atan2(mask_im_phase, mask_re_phase)
        mask_mag = torch.tanh(mask_mag)

        est_mag = tf_mag * mask_mag
        est_phase = tf_phase + mask_phase

        est_re = est_mag * torch.cos(est_phase)
        est_im = est_mag * torch.sin(est_phase)
        return torch.stack([est_re, est_im], dim=-1)


class SoTaskWrapModule(EncDecMaskerBaseModel):
    """
    Model wrapper for a Single Input Singal Output(SISO) or Multi Input Singal Output (MISO) tasks.
    Like single channel speech enhancement or target speech extraction.

    Args:
        encoder: nn.Module, speech encoder convert the speech samples to latent feature
        masker: nn.Module, mask generation backbone model
        encoder_spk: nn.Module, speech encoder (for SpeakerNet) convert the speech samples to latent feature
        speaker_net: nn.Module, speaker embedding generation backbone model
        loss_func_wav: nn.Module, signal-domain loss function
        loss_func_spk: nn.Module, classification or others loss function
        f_type: latent feature type, for STFT encoder we have [real, complex, polar] format
        mask_type: mask type, for STFT encoder we have [real, complex, polar] format
        mask_constraint: activation function on mask generation, [linear, relu, softmax]
        drop_first_bin: In STFT-based encoder, we can drop the DC bin (i.e idx=0).
        verbose: print model infomation

    Main structure has:
        Encoder -> Masker -> Decoder(inside encoder's inverse). // This is `Speech Enhancement` or `Speech Separation` Task
    
    When add a speaker_net:
        Encoder ----------------------------> Masker ------> Decoder.    // Multi-task combined `Target Speech Extraction` and `Speaker classification`
            |                                   ^                        // Both tasks share the same Speech Encoder
            |----> SpeakerNet --> Embedding ----|----------> Classifier

    When add both speaker_net and encoder_spk:
        Encoder ------------------------------> Masker ----> Decoder.    // Multi-task combined `Target Speech Extraction` and `Speaker classification`
                                                  ^                      // Each tasks has its own Speech Encoder
        Encoder-spk-> SpeakerNet --> Embedding ---|--------> Classifier
    
    """
    def __init__(self,
                encoder: nn.Module,
                masker: nn.Module,
                encoder_spk: Optional[nn.Module] = None,
                speaker_net: Optional[nn.Module] = None,
                loss_func_wav: Optional[nn.Module] = None,
                loss_func_spk: Optional[nn.Module] = None,
                loss_func_others: Optional[nn.Module] = None,
                f_type: str = 'real',
                mask_type: str = 'real',
                mask_constraint: str = 'linear',
                output_constraint: str = 'linear',
                drop_first_bin: bool = False,
                verbose: bool = True,
                ) -> None:
        super().__init__()
        self.f_type = f_type
        self.mask_type = mask_type
        self.encoder = encoder
        self.masker = masker
        self.encoder_spk = encoder_spk
        self.speaker_net = speaker_net
        self.loss_func_wav = loss_func_wav
        self.loss_func_spk = loss_func_spk
        self.loss_func_others = loss_func_others
        self.mask_constraint = mask_constraint
        self.output_constraint = output_constraint
        self.drop_first_bin = drop_first_bin
        self.task = self.check_task()
        print(f"Current task label: {self.task}")
        if verbose: self._verbose()
    
    def check_task(self):
        """
        Checking initialized sucess and return task-label.

        Return:
            task_label: 0: SE/BSS ; 1: TSE(wav+speaker) ; 2: TSE(contrastive learning via speaker loss)
        """
        if self.speaker_net is None:
            task_label = 0
            if self.encoder_spk is not None:
                print(f"Initialized a SE or BSS model, ignored the Encoder-spk.")
            else:
                print(f"Initialized a SE or BSS model.")

        else:
            if self.encoder_spk is not None:
                task_label = 1
                assert self.speaker_net is not None, f"Multi-task trainer missed the SpeakerNet."
                print(f"Initialized a multi-task model, including two separate speech encoder.")
            
            else:
                task_label = 1
                print(f"Initialized a multi-task model, sharing a same speech encoder.")               
        
            if self.loss_func_spk is not None:
                if self.loss_func_wav is None:
                    task_label = 2
                    print(f"Contrastive learning via speaker loss function.")

                else:
                    if self.loss_func_others is None:
                        task_label = 1
                        print(f"Multi-task training has two loss function.")
                    else:
                        task_label = 3
                        print(f"Multi-task training has three loss function.")
            
            else:
                if self.loss_func_wav is None and self.loss_func_spk is None:
                    task_label = None
                    print(f"Inference mode.")
                
                else:
                    task_label = 1
                    print(f"Multi-task training has only one loss function.")

        return task_label
    
    def _get_feature(self, noisy: Optional[torch.Tensor] = None, enroll: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward encoder to get features.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T], if None, skip it
        
        Returns:
            noisy tensor with shape [N, C, T]
            enroll tensor with shape [N, C, T] or None
        """
        if noisy is not None:
            noisy = self.encoder(noisy) # [N, C, T] or [N, C, T, 2]
            if isinstance(self.encoder, ConvEncDec):
                # STFT: ConvEncDec returns shape [N, C, T, 2]
                _re = noisy[..., 0]
                _im = noisy[..., 1]
                if self.drop_first_bin:
                    _re = _re[:, 1:, :]
                    _im = _im[:, 1:, :]
                
                noisy = torch.cat([_re, _im], dim=1)

        if enroll is not None:
            # Enable shared encoder structure like SpEx+
            if self.encoder_spk is None:
                enroll = self.encoder(enroll) # [N, C, T]
                if isinstance(self.encoder, ConvEncDec):
                    # STFT: ConvEncDec returns shape [N, C, T, 2]
                    _re = enroll[..., 0]
                    _im = enroll[..., 1]
                    if self.drop_first_bin:
                        _re = _re[:, 1:, :]
                        _im = _im[:, 1:, :]
                    
                    enroll = torch.cat([_re, _im], dim=1)

            else:
                enroll = self.encoder_spk(enroll) # [N, C, T]
                if isinstance(self.encoder_spk, ConvEncDec):
                    # STFT: ConvEncDec returns shape [N, C, T, 2]
                    _re = enroll[..., 0]
                    _im = enroll[..., 1]
                    if self.drop_first_bin:
                        _re = _re[:, 1:, :]
                        _im = _im[:, 1:, :]
                    
                    enroll = torch.cat([_re, _im], dim=1)

        return noisy, enroll
    
    def _get_waveform(self, enh_feats: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, ConvEncDec):
            if enh_feats.dim() != 4:
                _re, _im = torch.chunk(enh_feats, 2, dim=1)
                enh_feats = torch.stack([_re, _im], dim=-1)

            if self.drop_first_bin:
                _re = enh_feats[..., 0]
                _im = enh_feats[..., 1]
                pad = torch.zeros(_re.shape[0], 1, _re.shape[2], device=enh_feats.device)
                _re = torch.cat([pad, _re], dim=1)
                _im = torch.cat([pad, _im], dim=1)
                enh_feats = torch.stack([_re, _im], dim=-1)
            
        enh_wav = self.encoder.inverse(enh_feats)
        return enh_wav

    def _align_waveform(self, enh_wav: torch.Tensor, ref_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assume last axis is the time."""
        enh_wav_l = enh_wav.shape[-1]
        ref_wav_l = ref_wav.shape[-1]
        if enh_wav_l != ref_wav_l:
            if ref_wav_l < enh_wav_l:
                # align from last
                pad_num = enh_wav_l - ref_wav_l
                ref_wav = F.pad(ref_wav, (pad_num, 0))
            else:
                # align from begin
                enh_wav = enh_wav[..., :ref_wav_l]
        return enh_wav, ref_wav
    
    def _wav_output_constrain(self, wav: torch.Tensor, mode: str):
        if mode.lower() == 'linear':
            wav = torch.clamp_(wav, min=-1, max=1)
        
        elif mode.lower() == 'sigmoid':
            wav = torch.sigmoid(wav)
        
        else:
            raise NameError('Non support type.')
        
        return wav

    def _forward(self,
                noisy: torch.Tensor,
                enroll: torch.Tensor,
                ref_clean: torch.Tensor,
                inactive_labels: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Getting input noisy(n-mixed speech) and enrollment speech(target enroll) to calculate the training loss. 
        This way can usage DP benifit for balance GPU'e memory to avoid single machine OOM problem.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T] or None
            ref_clean: Target speech, [N, T]
            inactive_labels: Inactive speaker labels, [N]

        Return:
            Loss: loss_sdr
        """
        noisy, enroll = self._get_feature(noisy, enroll) # [N, C, T]
        dvec = enroll

        if dvec is not None:
            if type(self.speaker_net) == nn.ModuleList:
                for layer in self.speaker_net:
                    dvec = layer(dvec)
            else:
                dvec = self.speaker_net(dvec)
        
            dvec = dvec.squeeze(-1)

        if dvec is not None:
            # TSE task
            mask = self.masker(noisy, dvec)
        else:
            # SE task
            mask = self.masker(noisy)

        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self._get_waveform(enh_feats)
        enh_wav = self._wav_output_constrain(enh_wav, mode=self.output_constraint)
        enh_wav, ref_clean = self._align_waveform(enh_wav, ref_clean)
        loss_wav = self.loss_func_wav(enh_wav, ref_clean, inactive_labels)
        
        return loss_wav
    
    def _forward_join(self,
                    noisy: torch.Tensor,
                    enroll: torch.Tensor,
                    ref_clean: torch.Tensor,
                    spk_class: torch.Tensor,
                    alpha: float = 10,
                    return_loss_detail: bool = True,
                    inactive_labels: Optional[torch.Tensor] = None,
                    ) -> torch.Tensor:
        """
        Getting input noisy(n-mixed speech) and enrollment speech(target enroll) to calculate the SDR based loss.
        And also calculate the speaker classification loss.
        This way can usage DP benifit for balance GPU'e memory to avoid single machine OOM problem.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T]
            ref_clean: Target speech in the noisy
            spk_class: Speaker label
            alpha: Weight for loss combining
            inactive_labels: Inactive speaker labels, [N]
    
        Return:
            Joint loss: loss_sdr + alpha * loss_class
        """
        noisy, enroll = self._get_feature(noisy, enroll) # [N, C, T]
        
        dvec = enroll
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                dvec = layer(dvec)
        else:
            dvec = self.speaker_net(dvec)
                
        dvec = dvec.squeeze(-1)
        mask = self.masker(noisy, dvec)
        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self._get_waveform(enh_feats)
        enh_wav = self._wav_output_constrain(enh_wav, mode=self.output_constraint)
        enh_wav, ref_clean = self._align_waveform(enh_wav, ref_clean)
        loss_wav = self.loss_func_wav(enh_wav, ref_clean, inactive_labels)

        if self.loss_func_spk is not None and spk_class is not None:
            loss_spk = self.loss_func_spk(dvec, spk_class)
            if return_loss_detail:
                return loss_wav + alpha * loss_spk, (loss_wav, loss_spk)
            
            else:
                return loss_wav + alpha * loss_spk
        
        else:
            return loss_wav
    
    def _forward_contrastive(self,
                            noisy: torch.Tensor,
                            enroll: torch.Tensor,
                            spk_class: torch.Tensor,
                            ) -> torch.Tensor:
        """
        Getting input noisy(n-mixed speech) and enrollment speech(target enroll) to calculate the SDR based loss.
        And also calculate the speaker classification loss.
        This way can usage DP benifit for balance GPU'e memory to avoid single machine OOM problem.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T]
            spk_class: Speaker label
    
        Return:
            Joint loss: loss_class
        """
        noisy, enroll = self._get_feature(noisy, enroll) # [N, C, T]
        
        dvec = enroll
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                dvec = layer(dvec)
        else:
            dvec = self.speaker_net(dvec)
                
        dvec = dvec.squeeze(-1)
        mask = self.masker(noisy, dvec)
        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self._get_waveform(enh_feats)
        enh_wav = self._wav_output_constrain(enh_wav, mode=self.output_constraint)
        # enh_wav = torch.clamp_(enh_wav, min=-1, max=1)

        enh_feats, _ = self._get_feature(enh_wav, None)
        enh_dvec = enh_feats
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                enh_dvec = layer(enh_dvec)
        else:
            enh_dvec = self.speaker_net(enh_dvec)
        
        enh_dvec = enh_dvec.squeeze(-1)
        N = dvec.shape[0]
        total_dvec = torch.cat([dvec, enh_dvec], dim=-1).reshape(N*2, -1)
        spk_class = torch.cat([spk_class, spk_class], dim=-1).reshape(N*2, -1)
        loss_spk = self.loss_func_spk(total_dvec, spk_class)
        return loss_spk

    def _forward_join_loop(self,
                        noisy: torch.Tensor,
                        enroll: torch.Tensor,
                        ref_clean: torch.Tensor,
                        spk_class: torch.Tensor,
                        alpha: float = 10,
                        return_loss_detail: bool = True,
                        inactive_labels: Optional[torch.Tensor] = None,
                        ) -> torch.Tensor:
        """
        Getting input noisy(n-mixed speech) and enrollment speech(target enroll) to calculate the SDR based loss.
        And also calculate the speaker classification loss.
        This way can usage DP benifit for balance GPU'e memory to avoid single machine OOM problem.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T]
            ref_clean: Target speech in the noisy
            spk_class: Speaker label
            alpha: Weight for loss combining
            inactive_labels: Inactive speaker labels, [N]
    
        Return:
            Joint loss: loss_sdr + alpha * loss_class
        """
        noisy_wav = noisy.clone()
        enroll_wav = enroll.clone()
        noisy, enroll = self._get_feature(noisy, enroll) # [N, C, T]
        
        dvec = enroll
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                dvec = layer(dvec)
        else:
            dvec = self.speaker_net(dvec)
                
        dvec = dvec.squeeze(-1)
        mask = self.masker(noisy, dvec)
        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self._get_waveform(enh_feats)
        enh_wav = self._wav_output_constrain(enh_wav, mode=self.output_constraint)

        # non-target speech and noise etc.
        pred_noise = noisy_wav - enh_wav
        _, enh_dvec = self._get_feature(None, enh_wav)
        _, noise_dvec = self._get_feature(None, pred_noise)

        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                enh_dvec = layer(enh_dvec)
        else:
            enh_dvec = self.speaker_net(enh_dvec)
        
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                noise_dvec = layer(noise_dvec)
        else:
            noise_dvec = self.speaker_net(noise_dvec)

        triplet_dvec = torch.stack([dvec.squeeze(-1), enh_dvec.squeeze(-1), noise_dvec.squeeze(-1)], dim=1)
        
        enh_wav, ref_clean = self._align_waveform(enh_wav, ref_clean)
        loss_wav = self.loss_func_wav(enh_wav, ref_clean, inactive_labels)

        if self.loss_func_spk is not None and spk_class is not None:
            loss_spk = self.loss_func_spk(dvec, spk_class)
            loss_spk_loop = self.loss_func_others(triplet_dvec)
            if return_loss_detail:
                return loss_wav + alpha * loss_spk +  (1/alpha) * loss_spk_loop, (loss_wav, loss_spk, (1/alpha) * loss_spk_loop)
            
            else:
                return loss_wav + alpha * loss_spk + alpha * loss_spk_loop
        
        else:
            return loss_wav
    
    def forward(self, **kwargs):
        if self.task == 0:
            return self._forward(**kwargs)

        elif self.task == 1:
            return self._forward_join(**kwargs)
        
        elif self.task == 2:
            return self._forward_contrastive(**kwargs)
        
        elif self.task == 3:
            return self._forward_join_loop(**kwargs)

        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference(self, noisy: torch.Tensor, enroll: Optional[torch.Tensor] = None) -> torch.Tensor:
        noisy, enroll = self._get_feature(noisy, enroll) # [N, C, T]
        dvec = enroll

        if dvec is not None:
            if type(self.speaker_net) == nn.ModuleList:
                for layer in self.speaker_net:
                    dvec = layer(dvec)
            else:
                dvec = self.speaker_net(dvec)
        
            dvec = dvec.squeeze(-1)

        if dvec is not None:
            # TSE task
            mask = self.masker(noisy, dvec)
        else:
            # SE task
            mask = self.masker(noisy)

        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self._get_waveform(enh_feats)
        enh_wav = self._wav_output_constrain(enh_wav, mode=self.output_constraint)
        return enh_wav

    @torch.no_grad()
    def inference_tse_embedding(self, enroll: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward SpeakerNet to get speaker embedding in TSE-task setting."""
        # Enable shared encoder structure like SpEx+
        _, enroll = self._get_feature(None, enroll)
        dvec = enroll
        if type(self.speaker_net) == nn.ModuleList:
            for layer in self.speaker_net:
                dvec = layer(dvec) # [N, emb_dim]
        else:
            dvec = self.speaker_net(dvec)
        
        return dvec

    def _verbose(self):
        print(f"---------------Verbose logging---------------")
        self.eval()
        print(f"Current training mode is: {self.training}")
        print(f"Total params: {self.overall_parameters}")

        # compute lookahead
        x = torch.rand(1, 10*16000)
        x[..., 5*16000:] = np.inf
        x_spk = torch.rand(1, 10*16000)
        try:
            y = self.inference(x, x_spk).detach()
        except:
            y = self.inference(x).detach()
        lookahead = np.where(np.isnan(y) == True)[-1][0]
        if lookahead == 0:
            print('Lookahead(samples): infinite')
        else:
            lookahead = 80000 - lookahead
            print(f'Lookahead(samples): {lookahead}')
        
        # compute receptive field
        x = torch.rand(1, 10*16000)
        x[..., :-5*16000] = np.inf
        try:
            y = self.inference(x, x_spk).detach()
        except:
            y = self.inference(x).detach()
        receptive = np.where(np.isnan(y) == True)[-1][-1]
        if receptive - (80000 - 1) == 80000:
            print(f'Receptive Fields(samples): infinite')
        else:
            receptive = receptive - (80000 - 1)
            print(f'Receptive Fields(samples): {receptive}')

        self.train()
        print(f"Current training mode is: {self.training}")
        print(f"---------------Verbose logging---------------")
