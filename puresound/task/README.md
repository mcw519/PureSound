# Task

Task is a PyTorch based trainer.  
Including functions:
- Model training step
- Model validation step
- Schedule learning rate
- Save checkpoint and logging
- Tensorboard extension

## Dataset class
This class manage the data loading and on-the-fly data pipeline, just needing related metadata by user handcrafting.

## TseDataset & TseTask
Target Speech Extraction

    # Needed metadata
    wav2scp.txt: noisy wav path
    wav2ref.txt: clean wav path
    ref2list.txt: target enrollment speech list
    ref2spk.txt: target speaker id
    wav2spk.txt': speakers in mixture

TSE task is shared with PVAD task, just replacing the training objects from target speech to target speech activity.

    # Switch to PVAD dataset by adding one metadata
    ref2vad.txt: pvad labels

## NsDataset & NsTask
Noise Suppression speech enhancement

    # Needed metadata
    wav2scp.txt: noisy wav path
    wav2ref.txt: clean wav path