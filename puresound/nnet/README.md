# Neural Network (NNET)

In this PureSound project we tried to provide flexible adjustment for each tasks.


## EncDecMaskerBaseModel
Currently we only have Mask based model.  
Support mask type including Complex, Real and Polar forms.  
Based on our EncDecMaskerBaseModel, you can get the different combination between each mask types and feature types.

    1. self._apply_complex_mask_on_reim(tf_rep, est_masks)
    2. self._apply_mag_mask_on_reim(tf_rep, est_masks)
    3. self._apply_mag_mask_on_mag(tf_rep, est_masks)
    4. self._apply_complex_mask_on_polar (tf_rep, est_masks)


## Task Wrapper
Task wrapper can help you to minimum design your training pipeline.  
Based on the TaskWrapper, you just need to design your backbone model and desired loss function.  
Currently support:
- SoTaskWrapModule

### SoTaskWrapModule
Single-Output Mask-based wrapper.  
Below is an example of the TSE task:

    model = SoTaskWarpModule(
        encoder=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        encoder_spk=FreeEncDec(win_length=32, hop_length=16, laten_length=512),
        masker=ConvTasNet(512, 192, True, tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2, per_tcn_stack=8,
            tcn_with_embed=[1, 0, 0, 0, 0, 0, 0, 0], norm_type='gLN', causal=False, tcn_layer='normal'),
        speaker_net=nn.ModuleList(
                [TCN(512, 256, 3, dilation=2**i, causal=False, norm_type='gLN') for i in range(5)] + \
                [AttentiveStatisticsPooling(512, 128), nn.Conv1d(512*2, 192, 1, bias=False)]),
        loss_func_wav=sig_loss,
        loss_func_spk=cls_loss,
        mask_constraint='ReLU',
        **kwargs)

Describes
- Two speech encoders one for general input the other for a speaker verification model.
- One masker backbone is ConvTasnet.
- One SpeakerNet based one TCN structure.
- Multi-task loss is used here, one for signal-based loss the other is classification loss.
- Mask values is constrained by a ReLU activation.


### Implemented a sig_loss function for SoTaskWrapModule
Signal or waveform loss function must have three inputs named enh_wav, ref_wav and inactive_label.  
In general, this is a SDR-based loss function. But you can using lambda function to warp others like L1 or L2 etc.
        
    import torch.nn.functional as F
    sig_loss = lambda enh, ref, dummy: F.l1_loss(enh, ref)