# puresound.audio

Audio processing sub-package providing primitives for I/O, digital signal processing, spectrum analysis, augmentation, and room simulation.

## Sub-modules

| Module | Description |
|--------|-------------|
| [audio.io](io.md) | Audio file reading and writing |
| [audio.dsp](dsp.md) | Resampling and biquad/parametric EQ filtering |
| [audio.spectrum](spectrum.md) | STFT analysis and synthesis utilities |
| [audio.volume](volume.md) | Amplitude normalization and volume manipulation |
| [audio.noise](noise.md) | Background noise mixing |
| [audio.augmentation](augmentation.md) | Composable audio augmentation pipeline |
| [audio.impulse_response](impulse_response.md) | RIR convolution and IIR filtering |
| [audio.rir_bank](rir_bank.md) | Pre-generated RIR bank serving (rooms, near/far channels, origin tags) |
| [audio.room_simulator](room_simulator.md) | Physics-based shoebox room simulator |
| [audio.hybrid_rir](hybrid_rir.md) | Hybrid wave/geometric 5-channel RIR dataset generation |
