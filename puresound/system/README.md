# System

Trainer realated codes are all based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)


## `SISO`: Single-Input and Single-Output Trainer

### EncDecMaskBase
Ex: mask-based speech enhancement, mapping-based speech enhancement tasks

Structure:

         Encoder             Backbone-NN                 Apply                     Decoder
    Wav ---------> Features -------------> Mapping/Mask -------> Restore Features ---------> Wav


### EncPredClassBase
Ex: speaker identification, sound event detection, deepfake detection tasks

Structure:
    
         Encoder             Backbone-NN
    Wav ---------> Features -------------> Predict classes
