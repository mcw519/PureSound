from typing import Optional

import torch
import torch.nn as nn
from .lobe.encoder import FreeEncDec, ConvEncDec


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
            tf_rep: feature must in type (Complex, MagPhase) has shape [N, C, T, 2], or (real) has shape [N, C, T]
            est_masks: tf-mask shape [N, C, T]
            mask_type: mask type in (`complex`, `real`, `polar`)
            f_type: feature type in (`complex`, `real`, `polar`)
        
        Returns:
            ComplexFloatTensor or FolatTensor -- Masked time-frequency representations.
        
        Raises:
            mask_type not in (`complex`, `real`)
            f_type not in (`complex`, `real`)
        """
        if mask_type.lower() == 'complex' and f_type.lower() == 'complex':
            return self._apply_complex_mask_on_reim(tf_rep, est_masks)
        elif mask_type.lower() == 'real' and f_type.lower() == 'complex':
            return self._apply_mag_mask_on_reim(tf_rep, est_masks)
        elif mask_type.lower() == 'real' and f_type.lower() == 'real':
            return self._apply_mag_mask_on_mag(tf_rep, est_masks)
        if mask_type.lower() == 'polar' and f_type.lower() == 'polar':
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
            est_masks: tf-mask shape [N, C, T]

        Returns:
            FloatTensor -- [N, C, T, 2]
        """
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


class TaskWarpModule(EncDecMaskerBaseModel):
    """
    Model wrapper for each task.

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
                f_type: str = 'real',
                mask_type: str = 'real',
                mask_constraint: str = 'linear',
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
        self.mask_constraint = mask_constraint
        self.task = self.check_task()
    
    def check_task(self):
        """
        Checking initialized sucess and return task-label.

        Return:
            task_label: 0: SE/BSS ; 1: TSE
        """
        if self.speaker_net is None:
            class_label = 0
            if self.encoder_spk is not None:
                print(f"Initialized a SE or BSS model, ignored the Encoder-spk.")
            else:
                print(f"Initialized a SE or BSS model.")

        else:
            if self.encoder_spk is not None:
                class_label = 1
                assert self.speaker_net is not None, f"Multi-task trainer missed the SpeakerNet."
                print(f"Initialized a multi-task model, including two separate speech encoder.")
            
            else:
                class_label = 1
                print(f"Initialized a multi-task model, sharing a same speech encoder.")               
        
            if self.loss_func_spk is not None:
                class_label = 1
                print(f"Multi-task training has two loss function.")
            
            else:
                class_label = 1
                print(f"Multi-task training has only one loss function.")

        return class_label
    
    def _forward(self, noisy: torch.Tensor, enroll: torch.Tensor, ref_clean: torch.Tensor) -> torch.Tensor:
        """
        Getting input noisy(n-mixed speech) and enrollment speech(target enroll) to calculate the training loss. 
        This way can usage DP benifit for balance GPU'e memory to avoid single machine OOM problem.

        Args:
            noisy: Input noisy mixture tensor, [N, T]
            enroll: Enrolment waveform tensor, [N, T]
            ref_clean: Target speech, [N, T]
            
        Return:
            Loss: loss_sdr
        """   
        noisy = self.encoder(noisy) # [N, C, T]

        # Enable shared encoder structure like SpEx+
        if self.encoder_spk is None:
            enroll = self.encoder(enroll) # [N, C, T]
        else:
            enroll = self.encoder_spk(enroll) # [N, C, T]
        
        dvec = enroll
        for layer in self.speaker_net:
            dvec = layer(dvec)
        
        dvec = dvec.squeeze(-1)
        mask = self.masker(noisy, dvec)
        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self.encoder.inverse(enh_feats)
        enh_wav = torch.clamp_(enh_wav, min=-1, max=1)
        loss_wav = self.loss_func_wav(enh_wav, ref_clean)
        
        return loss_wav
    
    def _forward_join(self, noisy: torch.Tensor, enroll: torch.Tensor, ref_clean: torch.Tensor, spk_class: torch.Tensor, alpha: float = 10,
        return_loss_detail: bool = True) -> torch.Tensor:
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
    
        Return:
            Joint loss: loss_sdr + alpha * loss_class
        """   
        noisy = self.encoder(noisy) # [N, C, T]

        # Enable shared encoder structure like SpEx+
        if self.encoder_spk is None:
            enroll = self.encoder(enroll) # [N, C, T]
        else:
            enroll = self.encoder_spk(enroll) # [N, C, T]
        
        dvec = enroll
        for layer in self.speaker_net:
            dvec = layer(dvec)
        
        dvec = dvec.squeeze(-1)
        mask = self.masker(noisy, dvec)
        mask = self.get_mask(mask, self.mask_constraint)
        enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
        enh_wav = self.encoder.inverse(enh_feats)
        enh_wav = torch.clamp_(enh_wav, min=-1, max=1)
        loss_wav = self.loss_func_wav(enh_wav, ref_clean)

        if self.loss_func_spk is not None and spk_class is not None:
            loss_spk = self.loss_func_spk(dvec, spk_class)
            if return_loss_detail:
                return loss_wav + alpha * loss_spk, (loss_wav, loss_spk)
            
            else:
                return loss_wav + alpha * loss_spk
        
        else:
            return loss_wav
    
    def forward(self, **kwargs):
        if self.task == 0:
            return self._forward(**kwargs)

        elif self.task == 1:
            return self._forward_join(**kwargs)
        
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference(self, noisy: torch.Tensor, enroll: Optional[torch.Tensor] = None) -> torch.Tensor:
        noisy = self.encoder(noisy) # [N, C, T]

        if enroll is not None:
            # Enable shared encoder structure like SpEx+
            if self.encoder_spk is None:
                enroll = self.encoder(enroll) # [N, C, T]
            else:
                enroll = self.encoder_spk(enroll) # [N, C, T]

            dvec = enroll
            for layer in self.speaker_net:
                dvec = layer(dvec) # [N, emb_dim]

            dvec = dvec.squeeze(-1)
            mask = self.masker(noisy, dvec)
            mask = self.get_mask(mask, self.mask_constraint)
            enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
            enh_wav = self.encoder.inverse(enh_feats)
            enh_wav = torch.clamp_(enh_wav, min=-1, max=1)
        
        else:
            mask = self.masker(noisy)
            mask = self.get_mask(mask, self.mask_constraint)
            enh_feats = self.apply_tf_masks(noisy, mask, f_type=self.f_type, mask_type=self.mask_type) # [N, C, T]
            enh_wav = self.encoder.inverse(enh_feats)
            enh_wav = torch.clamp_(enh_wav, min=-1, max=1)

        return enh_wav
