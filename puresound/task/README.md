# Task
Task is an PyTorch based trainer.  
Including functions:
- Model training step
- Model validation step
- Schedule learning rate
- Save checkpoint and logging
- Tensorboard extension

## TSE task
Target Speech Extraction

### TseDataset
This class can manage you to load and do online data pipeline, just providing below metadatas:

    wav2scp.txt: noisy wav path
    wav2ref.txt: clean wav path
    ref2list.txt: target enrollment speech list
    ref2spk.txt: target speaker id
    wav2spk.txt': speakers in mixture

## NS task
Noise Suppression speech enhancement
