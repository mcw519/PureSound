# puresound.nnet.lobe.multiframe

Multi-frame speech enhancement modules that process temporal context windows for filtering.

## Class: `MultiFrameModule`

Base class for multi-frame filtering. Unfolds TF frames into context windows to enable joint processing of neighboring frames.

### Constructor

```python
MultiFrameModule(frame_size: int, lookahead: int = 0)
```

**Parameters:**
- `frame_size` – Number of frames in the context window
- `lookahead` – Number of future frames to include (0 = causal)

### `unfold(spec: Tensor) -> Tensor`

Unfolds a spectrum tensor into overlapping context windows.

**Parameters:**
- `spec` – Stacked spectrum `[batch, 2*F, T]`

**Returns:** Context-expanded tensor `[batch, 2*F*frame_size, T]`.

---

## Class: `DeepFilter`

Multi-frame deep filtering: applies learned filter coefficients over a temporal context window, enabling sub-band filtering with temporal smoothing.

**Reference:** Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering," ICASSP 2022.

### Constructor

```python
DeepFilter(num_freqs: int, frame_size: int, lookahead: int = 0)
```

**Parameters:**
- `num_freqs` – Number of frequency bins (F)
- `frame_size` – Context window size in frames
- `lookahead` – Number of look-ahead frames (0 = causal)

### `forward(spec: Tensor, coeff: Tensor) -> Tensor`

**Parameters:**
- `spec` – Stacked real/imag spectrum `[batch, 2*F, T]`
- `coeff` – Filter coefficients `[batch, 2*F*frame_size, T]`

**Returns:** Filtered spectrum `[batch, 2*F, T]`.

---

## Class: `MultiFrameWienerFilter`

Multi-frame Wiener filter that uses a temporal context to compute optimal Wiener filter coefficients and apply them.

### Constructor

```python
MultiFrameWienerFilter(num_freqs: int, frame_size: int, lookahead: int = 0)
```

### `forward(spec: Tensor, coeff: Tensor) -> Tensor`

**Parameters:**
- `spec` – Input spectrum `[batch, 2*F, T]`
- `coeff` – Wiener filter coefficients

**Returns:** Wiener-filtered spectrum `[batch, 2*F, T]`.

---

## Class: `MultiFrameMvdrFilter`

Multi-frame MVDR (Minimum Variance Distortionless Response) beamforming filter.

### Constructor

```python
MultiFrameMvdrFilter(num_freqs: int, frame_size: int, lookahead: int = 0)
```

### `forward(spec: Tensor, coeff: Tensor) -> Tensor`

**Parameters:**
- `spec` – Input spectrum `[batch, 2*F, T]`
- `coeff` – MVDR filter coefficients (steering vector or cross-correlation based)

**Returns:** MVDR-filtered spectrum `[batch, 2*F, T]`.

## Comparison of Multi-Frame Methods

| Method | Coefficients | Constraint | Best For |
|--------|-------------|------------|----------|
| `DeepFilter` | Learned directly | None (unconstrained) | End-to-end with direct output |
| `MultiFrameWienerFilter` | Wiener formulation | Minimum MSE | Noise reduction |
| `MultiFrameMvdrFilter` | MVDR formulation | Distortionless response | Spatial filtering / beamforming |
