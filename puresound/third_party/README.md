# Third-Party Code

This directory contains vendored research code used by optional PureSound
features.

## pytARD

`pytARD/` is vendored from `https://github.com/gpuard/pytARD`.

- Purpose: low-frequency wave-based room impulse response simulation.
- License: AGPL-3.0, see `pytARD/LICENSE`.
- Integration point: `puresound.audio.rir.render.low_frequency.pytard.GpuARDPytARDBackend`
  (also re-exported from `puresound.audio.rir.api`).

The upstream project is script-oriented and not packaged as a normal pip
dependency. PureSound imports it through a wrapper rather than modifying the
upstream files in place.
