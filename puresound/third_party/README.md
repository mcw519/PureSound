# Third-Party Code

Optional third-party dependencies that PureSound can use but does **not**
distribute. Nothing in this directory is shipped with the package; each entry
below describes what to install and how PureSound finds it.

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## pytARD — optional, user-installed

Wave-based low-frequency room impulse response simulation, used by
`puresound.audio.rir.render.low_frequency.pytard.GpuARDPytARDBackend` (also
re-exported from `puresound.audio.rir.api`).

- Upstream: <https://github.com/gpuard/pytARD>
- License: **AGPL-3.0**

It is not redistributed here, and it is not on PyPI, so it cannot be a pip
extra. Install it yourself:

```bash
git clone https://github.com/gpuard/pytARD.git /path/to/pytARD
export PURESOUND_PYTARD_ROOT=/path/to/pytARD
```

PureSound locates the checkout in this order:

1. the `third_party_root` argument passed to the backend,
2. `$PURESOUND_PYTARD_ROOT`,
3. `puresound/third_party/pytARD`, so an installation that vendors it keeps
   working.

The backend imports `pytARD_3D` and `common` from the checkout root at
simulation time and never modifies the upstream files. Without a checkout the
module still imports; only this backend raises, and the tests that drive the
solver skip.

**Every other low-frequency backend is unaffected** —
`AnalyticModalLowFrequencyBackend` and `ImpedanceModalLowFrequencyBackend` are
PureSound's own code and need nothing installed.
